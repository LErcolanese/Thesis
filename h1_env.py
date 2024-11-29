# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import torch

import omni.isaac.core.utils.torch as torch_utils
from prisma_h1.assets.unitree import H1_CFG_PD

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import Articulation, ArticulationCfg
from omni.isaac.lab.envs import DirectRLEnvCfg, DirectRLEnv
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sim import SimulationCfg
from omni.isaac.lab.sensors import ContactSensorCfg
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.math import sample_uniform
from omni.isaac.core.utils.torch.rotations import compute_heading_and_up, compute_rot, quat_conjugate



from omni.isaac.lab_tasks.direct.locomotion.locomotion_env import LocomotionEnv

def normalize_angle(x):
    return torch.atan2(torch.sin(x), torch.cos(x))


@configclass
class H1EnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 15.0
    decimation = 4
    action_scale = 1.0
    num_actions = 19
    num_observations = 50
    num_states = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 100)
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="average",
            restitution_combine_mode="average",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )
    contact=ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*",
        update_period=0.0, 
        history_length=6, 
        debug_vis=True
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=4.0, replicate_physics=True)

    # robot
    robot_cfg: ArticulationCfg = H1_CFG_PD.replace(prim_path="/World/envs/env_.*/Robot")


    heading_weight: float = 0.1
    up_weight: float = 0.4

    actions_cost_scale: float = 0.01
    alive_reward_scale: float = 2.0

    death_cost: float = -1.0
    termination_height: float = 0.8

    angvel_penalty_scale:float = 0.2
    zm_point_dist_scale: float = 0.25
    capture_point_dist_scale: float = 0.1
    contact_force_scale: float = 0.01


class H1Env(DirectRLEnv):
    cfg: H1EnvCfg

    def __init__(self, cfg: H1EnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        #RIEMPIRE SUCCESSIVAMENTE 
        # con cose che devo inizializzare per calcolare
        # azioni, osservazioni, reward etc
        self.dt = self.cfg.sim.dt * self.cfg.decimation

        self.robot_dof_lower_limits = self._robot.data.soft_joint_pos_limits[0, :, 0].to(device=self.device)
        self.robot_dof_upper_limits = self._robot.data.soft_joint_pos_limits[0, :, 1].to(device=self.device)

        self.robot_dof_speed_scales = torch.ones_like(self.robot_dof_lower_limits)
        self.torso_index=self._robot.find_bodies("torso_link")[0][0]
        self.left_foot_index=self._robot.find_bodies("left_ankle_link")[0][0]
        self.right_foot_index=self._robot.find_bodies("right_ankle_link")[0][0]
        # self.mass=(self._robot.data.default_mass).sum(dim=-1)
        # self.inertia=self._robot.data.default_inertia[:,self.torso_index]


        self.targets = torch.tensor([1000, 0, 0], dtype=torch.float32, device=self.sim.device).repeat(
            (self.num_envs, 1)
        )
        self.targets += self.scene.env_origins
        self.ang_momentum_old=torch.zeros_like(self.targets)

        self.joint_pos = self._robot.data.joint_pos
        self.joint_vel = self._robot.data.joint_vel

        self.joint_pos_targets = torch.zeros_like(self.joint_pos)

        self.S=torch.tensor([[0,-1],[1,0]]).repeat(self.num_envs,1,1)

        self.action_scale = self.cfg.action_scale

        self._joint_dof_idx, _ = self._robot.find_joints(".*")

        self.potentials = torch.zeros(self.num_envs, dtype=torch.float32, device=self.sim.device)
        self.prev_potentials = torch.zeros_like(self.potentials)

        self.start_rotation = torch.tensor([1, 0, 0, 0], device=self.sim.device, dtype=torch.float32)
        self.up_vec = torch.tensor([0, 0, 1], dtype=torch.float32, device=self.sim.device).repeat((self.num_envs, 1))
        self.heading_vec = torch.tensor([1, 0, 0], dtype=torch.float32, device=self.sim.device).repeat(
            (self.num_envs, 1)
        )
        self.inv_start_rot = quat_conjugate(self.start_rotation).repeat((self.num_envs, 1))
        self.basis_vec0 = self.heading_vec.clone()
        self.basis_vec1 = self.up_vec.clone()

    def _setup_scene(self):
        self._robot= Articulation(self.cfg.robot_cfg)

        self.scene.articulations["robot"]=self._robot


        self.cfg.terrain.num_envs=self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        

        # clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

#pre-physics step calls


    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone()
        self.joint_pos_targets=self.robot_dof_speed_scales * self.dt * self.actions * self.cfg.action_scale
        self.joint_pos_targets[:]=torch.clamp(self.joint_pos_targets, self.robot_dof_lower_limits, self.robot_dof_upper_limits)
        # targets = self.targets + self.robot_dof_speed_scales * self.dt * self.actions * self.cfg.action_scale
        # self.targets[:] = torch.clamp(targets, self.robot_dof_lower_limits, self.robot_dof_upper_limits)

    
    def _apply_action(self):
        # current_joint_pos = self.joint_pos
        # current_joint_vel = self.joint_vel

        # self.joint_target, _ = self._actuator.compute(
        #  current_joint_pos, current_joint_vel, self.cfg.action_scale * self.actions
        # )

        #l'inner loop lo fa direttamente l'attuatore PD che ho settato
        self._robot.set_joint_position_target(self.joint_pos_targets)


#post-physics step calls
    #termination conditions
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        #torso troppo in basso = caduto
        died = self.torso_position[:, 2] < self.cfg.termination_height

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return time_out, died
    

    def _get_rewards(self) -> torch.Tensor:
        # Refresh the intermediate values after the physics steps
        self._compute_intermediate_values()
        # robot_left_finger_pos = self._robot.data.body_pos_w[:, self.left_finger_link_idx]
        # robot_right_finger_pos = self._robot.data.body_pos_w[:, self.right_finger_link_idx]

        return compute_rewards(
            self.actions,
            self.reset_terminated,
            self.cfg.up_weight,
            self.cfg.heading_weight,
            self.heading_proj,
            self.up_proj,
            self.dof_pos_scaled,
            self.potentials,
            self.prev_potentials,
            self.cfg.actions_cost_scale,
            self.cfg.death_cost,
            self.cfg.alive_reward_scale,
            self.torso_ang_vel,
            self.cfg.angvel_penalty_scale,
            self.torso_position,
            self.torso_lin_vel,
            self.right_foot_pos,
            self.left_foot_pos,
            # self.cfg.zm_point_dist_scale,
            # self.zm_point,
            self.cfg.capture_point_dist_scale,
            self._robot.data.default_root_state[:,:3]
   
        )
        

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES
        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)

        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        #to target = errore rispetto a target position
        to_target = self.targets[env_ids] - default_root_state[:, :3]
        self.potentials[env_ids] = -torch.norm(to_target, p=2, dim=-1) / self.cfg.sim.dt

        self._compute_intermediate_values()


    def _get_observations(self) -> dict:

        obs = torch.cat(
            (
            self.joint_pos,
            self.joint_vel,
            self.torso_position,
            self.torso_rotation,
            self.torso_lin_vel,
            self.torso_ang_vel,
            # self.scene["self.contact_forces"].data.net_forces_w
            ), dim=-1
        )
        observations = {"policy": obs}
        return observations
        

    def _compute_intermediate_values(self):

        self.torso_position, self.torso_rotation = self._robot.data.root_pos_w, self._robot.data.root_quat_w
        self.torso_lin_vel, self.torso_ang_vel = self._robot.data.root_lin_vel_w, self._robot.data.root_ang_vel_w
        self.joint_pos, self.joint_vel = self._robot.data.joint_pos, self._robot.data.joint_vel

        self.torso_lin_acc=self._robot.data.body_lin_acc_w[:,self.torso_index]
        self.right_foot_pos=self._robot.data.body_pos_w[:,self.right_foot_index]
        self.left_foot_pos=self._robot.data.body_pos_w[:,self.left_foot_index]

        
        self.projected_gravity=self._robot.data.projected_gravity_b

        # self.ang_momentum=self.inertia.view(self.num_envs,3,3)*self.torso_ang_vel
        # self.ang_momentum_variation=(self.ang_momentum-self.ang_momentum_old)/self.dt

        # self.zm_point=(self.torso_position+self.torso_position[:,2]/(self.torso_lin_acc[:,2]
        #     +self.projected_gravity[:,2]) *(self.torso_lin_acc[:,:2]+self.projected_gravity[:,:2])+1
        #     /(self.mass*(self.torso_lin_acc[:,2]+self.projected_gravity[:,2]))*self.S*self.ang_momentum_variation[:,:2]
        # )
        
        (
            self.up_proj,
            self.heading_proj,
            self.up_vec,
            self.heading_vec,
            self.vel_loc,
            self.angvel_loc,
            self.roll,
            self.pitch,
            self.yaw,
            self.angle_to_target,
            self.dof_pos_scaled,
            self.prev_potentials,
            self.potentials,
        ) = compute_intermediate_values(
            self.targets,
            self.torso_position,
            self.torso_rotation,
            self.torso_lin_vel,
            self.torso_ang_vel,
            self.joint_pos,
            self._robot.data.soft_joint_pos_limits[0, :, 0],
            self._robot.data.soft_joint_pos_limits[0, :, 1],
            self.inv_start_rot,
            self.basis_vec0,
            self.basis_vec1,
            self.potentials,
            self.prev_potentials,
            self.cfg.sim.dt,
        )


#scripts
@torch.jit.script
def compute_intermediate_values(
    targets: torch.Tensor,
    torso_position: torch.Tensor,
    torso_rotation: torch.Tensor,
    velocity: torch.Tensor,
    ang_velocity: torch.Tensor,
    dof_pos: torch.Tensor,
    dof_lower_limits: torch.Tensor,
    dof_upper_limits: torch.Tensor,
    inv_start_rot: torch.Tensor,
    basis_vec0: torch.Tensor,
    basis_vec1: torch.Tensor,
    potentials: torch.Tensor,
    prev_potentials: torch.Tensor,
    dt: float,
):
    #errore tra posizione attuale e desiderata del com
    to_target = targets - torso_position

    torso_quat, up_proj, heading_proj, up_vec, heading_vec = compute_heading_and_up(
        torso_rotation, inv_start_rot, to_target, basis_vec0, basis_vec1, 2
    )

    vel_loc, angvel_loc, roll, pitch, yaw, angle_to_target = compute_rot(
        torso_quat, velocity, ang_velocity, targets, torso_position
    )

    dof_pos_scaled = torch_utils.maths.unscale(dof_pos, dof_lower_limits, dof_upper_limits)

    prev_potentials[:] = potentials
    potentials = -torch.norm(to_target, p=2, dim=-1) / dt

    return (
        up_proj,
        heading_proj,
        up_vec,
        heading_vec,
        vel_loc,
        angvel_loc,
        roll,
        pitch,
        yaw,
        angle_to_target,
        dof_pos_scaled,
        prev_potentials,
        potentials,
    )

@torch.jit.script
def compute_rewards(
    actions: torch.Tensor,
    reset_terminated: torch.Tensor,
    up_weight: float,
    heading_weight: float,
    heading_proj: torch.Tensor,
    up_proj: torch.Tensor,
    dof_pos_scaled: torch.Tensor,
    potentials: torch.Tensor,
    prev_potentials: torch.Tensor,
    actions_cost_scale: float,
    death_cost: float,
    alive_reward_scale: float,
    angvel: torch.Tensor,
    angvel_penalty_scale: float,
    torso_position: torch.Tensor,
    torso_lin_vel:torch.Tensor,
    right_foot_pos:torch.Tensor,
    left_foot_pos:torch.Tensor,
    # zm_point_dist_scale: float,
    # zm_point: torch.Tensor,
    capture_point_dist_scale: float,
    default_root_pos:torch.Tensor
):
    heading_weight_tensor = torch.ones_like(heading_proj) * heading_weight
    heading_reward = torch.where(heading_proj > 0.8, heading_weight_tensor, heading_weight * heading_proj / 0.8)

    # aligning up axis of robot and environment
    up_reward = torch.zeros_like(heading_reward)
    up_reward = torch.where(up_proj > 0.93, up_reward + up_weight, up_reward)


    # action cost
    actions_cost = torch.sum(actions**2, dim=-1)

    #odometry : current velocity from previous velocity - presuppone odometria
    #set com_target_velocity da cfg
    # odometry_dist = odometry_dist_scale * torch.norm(com_velocity_target-robot_com_velocity, p=2, dim=-1);

    # #joint position reference 
    # joint_pos_dist = joint_pos_dist_scale * torch.norm(joint_pos_target-robot_joint_pos, p=2, dim =-1)


    #compute reward for distance from capture point
    capt_point=torso_position[:,:2]+torso_lin_vel[:,:2]/angvel[:,:2]
    capt_point_target=0.5*(right_foot_pos[:,:2]+left_foot_pos[:,:2])
    capture_point_dist = capture_point_dist_scale * torch.norm(capt_point-capt_point_target, p=2, dim=-1)
    #richiamare funzioni per calcolare capture point actual e reference

   
    # zm_point_target=default_root_pos[:,:2]
    # zm_point_dist = zm_point_dist_scale * torch.norm(zm_point-zm_point_target, p=2, dim=-1)
    
    #zero moment point target è al centro del segmento che unisce la proiezione verticale dei due piedi
    #richiamare funzione che calcola posizione del zmp

    #reward for feet position and orientation che non capisco

    # #penalty per troppa ground friction
    # ground_frict_penalty = ground_frict_penalty_scale * torch.norm(ground_frict_vector, p=2, dim=-1)
    # #ground friction vector deve avere componenti x e y di ground friction su entrambi i piedi

    #reward per distribuzione del peso che non capisco

    #penalty per velocità angolare eccessiva
    ang_vel_penalty = angvel_penalty_scale * torch.norm(angvel, p=2, dim=-1)

    # dof at limit cost
    dof_at_limit_cost = torch.sum(dof_pos_scaled > 0.98, dim=-1)

    # reward for duration of staying alive
    alive_reward = torch.ones_like(potentials) * alive_reward_scale
    progress_reward = potentials - prev_potentials

    total_reward = (
        progress_reward
        + alive_reward
        + up_reward
        - actions_cost_scale * actions_cost
        - dof_at_limit_cost
        - capture_point_dist
        # - zm_point_dist
        - ang_vel_penalty
    )
    # adjust reward for fallen agents
    total_reward = torch.where(reset_terminated, torch.ones_like(total_reward) * death_cost, total_reward)
    return total_reward
