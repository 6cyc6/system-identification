from __future__ import annotations

import math
import torch
import logging
from collections import deque

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from twintrack.sim.isaac.envs.pushT.pushT_env_cfg_6 import PushTEnv6Cfg
from twintrack.sim.isaac.envs.push_box.push_box_env_2 import sample_obj_state

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s: %(message)s",
                          datefmt="%Y-%m-%d %H:%M:%S")
    )
    logger.addHandler(handler)


class PushTEnv6(DirectRLEnv):
    """PushT RL environment (v6) — 7 DOF joint space action.

    Action is delta joint position (7 DOF), scaled by action_scale. No IK.

    Observations (37): joint pos norm (7), joint vel (7), EE xyz (3), EE quat (4), fingertip xyz (3),
                       T-shape pose xyz+quat (7), goal xy (2), goal quat (4).
    Actions (7): delta joint positions for fr3_joint1..7.
    """

    cfg: PushTEnv6Cfg

    def __init__(self, cfg: PushTEnv6Cfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.goal_xy   = torch.tensor(self.cfg.goal_pos, device=self.device)  # (2,)
        half_yaw = cfg.goal_yaw * 0.5
        self.goal_quat = torch.tensor(
            [math.cos(half_yaw), 0.0, 0.0, math.sin(half_yaw)],
            device=self.device,
        )  # (4,)
        self._goal_yaw = float(cfg.goal_yaw)

        robot_entity_cfg = SceneEntityCfg(
            "robot",
            joint_names=["fr3_joint.*"],
            body_names=[cfg.ee_body_name],
        )
        robot_entity_cfg.resolve(self.scene)
        self._joint_ids  = robot_entity_cfg.joint_ids
        self._ee_body_id = robot_entity_cfg.body_ids[0]

        limits = self._robot.data.joint_limits[:, self._joint_ids, :]
        self._joint_low  = limits[0, :, 0]
        self._joint_high = limits[0, :, 1]

        # Joint position targets (in joint space)
        default_joint_pos = self._robot.data.default_joint_pos[:1, self._joint_ids].squeeze(0)
        self._joint_target      = default_joint_pos.unsqueeze(0).repeat(self.num_envs, 1)
        self._joint_target_prev = self._joint_target.clone()
        self._phys_sub_step     = 0

        self._success         = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._failure         = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._episode_success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._success_window  = deque(maxlen=100)

        self._prev_action    = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)
        self._cur_action     = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)
        self._prev_ee_to_obj = torch.zeros(self.num_envs, device=self.device)
        self._prev_obj_pos   = torch.zeros(self.num_envs, 3, device=self.device)
        self._prev_obj_yaw   = torch.zeros(self.num_envs, device=self.device)

        # Per-step cache
        self.obj_pos_local      = torch.zeros(self.num_envs, 3, device=self.device)
        self.obj_yaw            = torch.zeros(self.num_envs, device=self.device)
        self.obj_yaw_err        = torch.zeros(self.num_envs, device=self.device)
        self.ee_pos_local       = torch.zeros(self.num_envs, 3, device=self.device)
        self.ee_quat_w          = torch.zeros(self.num_envs, 4, device=self.device)
        self.fingertip_pos_local = torch.zeros(self.num_envs, 3, device=self.device)
        self.ee_tilt_dot        = torch.zeros(self.num_envs, device=self.device)
        self.ee_to_obj          = torch.zeros(self.num_envs, device=self.device)
        self.ee_to_obj_3d       = torch.zeros(self.num_envs, device=self.device)
        self.pos_err            = torch.zeros(self.num_envs, device=self.device)
        self.fingertip_z_margin = torch.zeros(self.num_envs, device=self.device)

        robot_x = self.cfg.robot.init_state.pos[0]
        robot_y = self.cfg.robot.init_state.pos[1]
        self._ws_x_lo = float(self.cfg.workspace_x[0] + robot_x)
        self._ws_x_hi = float(self.cfg.workspace_x[1] + robot_x)
        self._ws_y_lo = float(self.cfg.workspace_y[0] + robot_y)
        self._ws_y_hi = float(self.cfg.workspace_y[1] + robot_y)

        marker_cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/goal_marker",
            markers={
                "goal": sim_utils.UsdFileCfg(
                    usd_path=self.cfg.obj.spawn.usd_path,
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.0, 1.0, 0.0),
                        opacity=0.5,
                    ),
                ),
            },
        )
        self._goal_marker = VisualizationMarkers(marker_cfg)

        obj_z = self.cfg.obj.init_state.pos[2]
        goal_w = self.scene.env_origins.clone()
        goal_w[:, 0] += self.cfg.goal_pos[0]
        goal_w[:, 1] += self.cfg.goal_pos[1]
        goal_w[:, 2] += obj_z
        goal_quat_vis = self.goal_quat.unsqueeze(0).expand(self.num_envs, -1)
        self._goal_marker.visualize(goal_w, goal_quat_vis)

        logger.info(
            f"PushTEnv6: ee_body={cfg.ee_body_name}({self._ee_body_id}), "
            f"num_joints={len(self._joint_ids)}, "
            f"goal_yaw={cfg.goal_yaw:.3f} rad, action_scale={cfg.action_scale}"
        )

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self._obj   = RigidObject(self.cfg.obj)
        self.cfg.table.spawn.func(
            self.cfg.table.prim_path,
            self.cfg.table.spawn,
            translation=self.cfg.table.init_state.pos,
            orientation=self.cfg.table.init_state.rot,
        )
        self.scene.articulations["robot"] = self._robot
        self.scene.rigid_objects["obj"]   = self._obj

        self.cfg.terrain.num_envs    = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        self.scene.clone_environments(copy_from_source=False)

        light_cfg = sim_utils.DomeLightCfg(intensity=500.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # ------------------------------------------------------------------
    # DirectRLEnv interface
    # ------------------------------------------------------------------

    def _pre_physics_step(self, actions: torch.Tensor):
        self._cur_action = actions.clamp(-1.0, 1.0)
        delta_joint = self._cur_action * self.cfg.action_scale

        self._joint_target_prev = self._joint_target.clone()
        self._joint_target = (self._joint_target + delta_joint).clamp(
            self._joint_low, self._joint_high
        )
        self._phys_sub_step = 0

    def _apply_action(self):
        alpha = (self._phys_sub_step + 1) / self.cfg.decimation
        joint_pos_des = torch.lerp(self._joint_target_prev, self._joint_target, alpha)
        self._phys_sub_step += 1
        self._robot.set_joint_position_target(joint_pos_des, joint_ids=self._joint_ids)

    def _compute_intermediate_values(self):
        self.obj_pos_local = self._obj.data.root_pos_w - self.scene.env_origins
        obj_quat           = self._obj.data.root_quat_w
        self.obj_yaw       = _quat_to_yaw(obj_quat)
        self.obj_yaw_err   = _yaw_error(obj_quat, self._goal_yaw)
        self.ee_pos_local  = self._robot.data.body_pos_w[:, self._ee_body_id] - self.scene.env_origins
        self.ee_quat_w     = self._robot.data.body_quat_w[:, self._ee_body_id]
        # Fingertip: EE body position + offset d along EE z-axis (third column of R(q))
        qw = self.ee_quat_w[:, 0];  qx = self.ee_quat_w[:, 1]
        qy = self.ee_quat_w[:, 2];  qz = self.ee_quat_w[:, 3]
        rz_x = 2.0 * (qx * qz + qw * qy)
        rz_y = 2.0 * (qy * qz - qw * qx)
        rz_z = 1.0 - 2.0 * (qx * qx + qy * qy)
        d = 0.175  # fingertip offset along EE z-axis
        d2 = 0.1755
        self.fingertip_pos_local = torch.stack([
            self.ee_pos_local[:, 0] + d * rz_x,
            self.ee_pos_local[:, 1] + d * rz_y,
            self.ee_pos_local[:, 2] + d * rz_z,
        ], dim=1)
        # Margin above table (table top at z=0.1); negative means penetrating
        self.fingertip_z_margin = self.ee_pos_local[:, 2] + d2 * rz_z - 0.1
        # Dot product of EE z-axis with world (0,0,-1): 1 = pointing straight down
        self.ee_tilt_dot  = -rz_z
        self.ee_to_obj    = torch.linalg.vector_norm(
            self.ee_pos_local[:, :2] - self.obj_pos_local[:, :2], dim=1
        )
        self.ee_to_obj_3d = torch.linalg.vector_norm(
            self.fingertip_pos_local - self.obj_pos_local, dim=1
        )
        self.pos_err       = torch.linalg.vector_norm(
            self.obj_pos_local[:, :2] - self.goal_xy, dim=1
        )

    def _get_observations(self) -> dict:
        self._compute_intermediate_values()
        obs = compute_obs(
            self._robot.data.joint_pos[:, self._joint_ids],
            self._robot.data.joint_vel[:, self._joint_ids],
            self._joint_low,
            self._joint_high,
            self.obj_pos_local,
            self._obj.data.root_quat_w,
            self.ee_pos_local,
            self.ee_quat_w,
            self.fingertip_pos_local,
            self.goal_xy,
            self.goal_quat,
        )
        return {"policy": obs, "critic": obs}

    def _get_rewards(self) -> torch.Tensor:
        reward, r_robot, r_tcp, r_pos, r_orient, r_smooth, r_tilt, r_static, r_success, r_failure, r_table, r_flip = compute_rewards(
            self.pos_err,
            self.obj_yaw_err,
            self.obj_pos_local,
            self._prev_obj_pos,
            self.obj_yaw,
            self._prev_obj_yaw,
            self._cur_action,
            self._prev_action,
            self._prev_ee_to_obj,
            self.ee_to_obj_3d,
            self.ee_to_obj,
            self.ee_tilt_dot,
            self.fingertip_z_margin,
            self._success,
            self._failure,
            float(self.cfg.reward_pos_weight),
            float(self.cfg.reward_orient_weight),
            float(self.cfg.reward_smooth_weight),
            float(self.cfg.reward_robot_dist_weight),
            float(self.cfg.reward_tilt_weight),
            float(self.cfg.box_null_dist),
            float(self.cfg.reward_success),
            float(self.cfg.static_pos_threshold),
            float(self.cfg.static_rot_threshold),
            float(self.cfg.static_penalty),
            float(self.cfg.tilt_threshold),
            float(self.cfg.flip_z_threshold),
            float(self.cfg.reward_flip_penalty),
            float(self.cfg.reward_table_penalty),
        )
        self._prev_action    = self._cur_action.clone()
        self._prev_ee_to_obj = self.ee_to_obj_3d.detach()
        self._prev_obj_pos   = self.obj_pos_local.detach()
        self._prev_obj_yaw   = self.obj_yaw.detach()

        self.extras["log"] = {
            "rew/robot":             r_robot.mean().item(),
            "rew/tcp":               r_tcp.mean().item(),
            "rew/pos":               r_pos.mean().item(),
            "rew/orient":            r_orient.mean().item(),
            "rew/smooth":            r_smooth.mean().item(),
            "rew/tilt":              r_tilt.mean().item(),
            "rew/static":            r_static.mean().item(),
            "rew/table":             r_table.mean().item(),
            "rew/flip":              r_flip.mean().item(),
            "rew/success":           r_success.mean().item(),
            "rew/failure":           r_failure.mean().item(),
            "metrics/ee_to_obj_m":   self.ee_to_obj.mean().item(),
            "metrics/obj_to_goal_m": self.pos_err.mean().item(),
            "metrics/yaw_err_rad":   self.obj_yaw_err.abs().mean().item(),
            "metrics/failure_rate":  self._failure.float().mean().item(),
        }
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._compute_intermediate_values()
        self._success, self._failure = compute_dones(
            self.pos_err,
            self.obj_yaw_err,
            self.obj_pos_local,
            float(self.cfg.success_pos_threshold),
            float(self.cfg.success_orient_threshold),
            float(self.cfg.flip_z_threshold),
        )
        self._episode_success |= self._success
        terminated = self._failure | self._success
        time_out   = self.episode_length_buf >= self.max_episode_length

        new_failures = self._failure.nonzero(as_tuple=False).squeeze(-1)
        if len(new_failures) > 0:
            for idx in new_failures[:4]:
                i = idx.item()
                logger.info(f"FAIL env={i:4d}  cause=fallen")

        return terminated, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == 0:
            return

        super()._reset_idx(env_ids)

        for s in self._episode_success[env_ids].tolist():
            self._success_window.append(float(s))
        if len(self._success_window) >= 10:
            self.extras.setdefault("log", {})["metrics/success_rate"] = (
                sum(self._success_window) / len(self._success_window)
            )
        self._success[env_ids]         = False
        self._failure[env_ids]         = False
        self._episode_success[env_ids] = False
        self._prev_action[env_ids]    = 0.0
        self._cur_action[env_ids]     = 0.0
        self._prev_ee_to_obj[env_ids] = 0.0
        self._prev_obj_pos[env_ids]   = 0.0
        self._prev_obj_yaw[env_ids]   = 0.0

        joint_pos = self._robot.data.default_joint_pos[env_ids].clone()
        joint_vel = self._robot.data.default_joint_vel[env_ids].clone()
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        self._robot.set_joint_position_target(
            joint_pos[:, self._joint_ids], joint_ids=self._joint_ids, env_ids=env_ids
        )

        self._joint_target[env_ids]      = joint_pos[:, self._joint_ids]
        self._joint_target_prev[env_ids] = joint_pos[:, self._joint_ids]

        n = len(env_ids)
        default_state = self._obj.data.default_root_state[env_ids].clone()
        default_state[:, 7:] = 0.0

        r_lo, r_hi     = self.cfg.obj_init_r_range
        yaw_lo, yaw_hi = self.cfg.obj_init_yaw_range
        default_state = sample_obj_state(
            n,
            self.scene.env_origins[env_ids],
            self.goal_xy,
            float(r_lo), float(r_hi),
            self._ws_x_lo, self._ws_x_hi,
            self._ws_y_lo, self._ws_y_hi,
            float(yaw_lo), float(yaw_hi),
            default_state,
        )
        self._obj.write_root_state_to_sim(default_state, env_ids)


# ---------------------------------------------------------------------------
# JIT-scripted helpers
# ---------------------------------------------------------------------------

@torch.jit.script
def _quat_to_yaw(quat: torch.Tensor) -> torch.Tensor:
    """Extract yaw (rotation about z) from quaternion (w, x, y, z)."""
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


@torch.jit.script
def _yaw_error(quat: torch.Tensor, goal_yaw: float) -> torch.Tensor:
    """Signed yaw error (object yaw − goal_yaw), wrapped to [−π, π]."""
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    obj_yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    err = obj_yaw - goal_yaw
    return torch.atan2(torch.sin(err), torch.cos(err))


@torch.jit.script
def compute_obs(
    joint_pos: torch.Tensor,           # (N, 7)
    joint_vel: torch.Tensor,           # (N, 7)
    joint_low: torch.Tensor,           # (7,)
    joint_high: torch.Tensor,          # (7,)
    obj_pos_local: torch.Tensor,       # (N, 3)
    obj_quat: torch.Tensor,            # (N, 4)
    ee_pos_local: torch.Tensor,        # (N, 3)
    ee_quat: torch.Tensor,             # (N, 4)
    fingertip_pos_local: torch.Tensor, # (N, 3)
    goal_xy: torch.Tensor,             # (2,)
    goal_quat: torch.Tensor,           # (4,)
) -> torch.Tensor:                     # (N, 37)
    N = joint_pos.shape[0]
    joint_pos_norm = 2.0 * (joint_pos - joint_low) / (joint_high - joint_low) - 1.0  # (N, 7)
    obj_pose_obs   = torch.cat([obj_pos_local, obj_quat], dim=1)                      # (N, 7)
    goal_xy_exp    = goal_xy.unsqueeze(0).expand(N, -1)                               # (N, 2)
    goal_quat_exp  = goal_quat.unsqueeze(0).expand(N, -1)                             # (N, 4)
    return torch.cat([
        joint_pos_norm,        # (N, 7)
        joint_vel,             # (N, 7)  joint velocities
        ee_pos_local,          # (N, 3)  EE body xyz
        ee_quat,               # (N, 4)  EE orientation
        fingertip_pos_local,   # (N, 3)  fingertip xyz
        obj_pose_obs,          # (N, 7)
        goal_xy_exp,           # (N, 2)
        goal_quat_exp,         # (N, 4)
    ], dim=1)  # (N, 37)


@torch.jit.script
def compute_rewards(
    pos_err: torch.Tensor,
    obj_yaw_err: torch.Tensor,
    obj_pos_local: torch.Tensor,
    prev_obj_pos: torch.Tensor,
    obj_yaw: torch.Tensor,
    prev_obj_yaw: torch.Tensor,
    cur_action: torch.Tensor,
    prev_action: torch.Tensor,
    prev_ee_to_obj: torch.Tensor,
    ee_to_obj_3d: torch.Tensor,
    ee_to_obj: torch.Tensor,
    ee_tilt_dot: torch.Tensor,
    fingertip_z_margin: torch.Tensor,
    success: torch.Tensor,
    failure: torch.Tensor,
    reward_pos_weight: float,
    reward_orient_weight: float,
    reward_smooth_weight: float,
    reward_robot_dist_weight: float,
    reward_tilt_weight: float,
    obj_null_dist: float,
    reward_success: float,
    static_pos_threshold: float,
    static_rot_threshold: float,
    static_penalty: float,
    tilt_threshold: float,
    flip_z_threshold: float,
    reward_flip_penalty: float,
    reward_table_penalty: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    r_pos     = reward_pos_weight    * (torch.exp(-2.0 * pos_err)           - 1.0)
    r_orient  = reward_orient_weight * (torch.exp(-2.0 * obj_yaw_err.abs()) - 1.0)
    r_smooth  = -reward_smooth_weight * ((cur_action - prev_action) ** 2).sum(dim=1)
    active    = (prev_ee_to_obj > obj_null_dist).float()
    r_robot   = reward_robot_dist_weight * active * (prev_ee_to_obj - ee_to_obj_3d)
    # TCP approach reward: dense guidance toward object CoM in XY, scaled to [0, 0.05]
    r_tcp     = torch.sqrt(1.0 - torch.tanh(5.0 * ee_to_obj)) / 20.0
    # Tilt penalty: zero when EE z-axis points down (tilt_dot >= 0.866), negative otherwise
    r_tilt    = reward_tilt_weight * torch.clamp(ee_tilt_dot - tilt_threshold, max=0.0)
    # Static penalty: only when BOTH position and yaw are unchanged
    obj_moved    = torch.linalg.vector_norm(obj_pos_local - prev_obj_pos, dim=1)
    yaw_diff     = torch.atan2(torch.sin(obj_yaw - prev_obj_yaw), torch.cos(obj_yaw - prev_obj_yaw))
    is_static    = (obj_moved < static_pos_threshold) & (yaw_diff.abs() < static_rot_threshold)
    r_static     = torch.where(is_static,
                               torch.full_like(obj_moved, static_penalty),
                               torch.zeros_like(obj_moved))
    # Table proximity penalty: -0.1 when fingertip is within 0.005 m of the table
    r_table   = torch.where(fingertip_z_margin < 0.005,
                            torch.full_like(fingertip_z_margin, reward_table_penalty),
                            torch.zeros_like(fingertip_z_margin))
    # Flip penalty: applied when object CoM z exceeds resting height (object lifted/flipped)
    r_flip    = torch.where(obj_pos_local[:, 2] > flip_z_threshold,
                            torch.full_like(obj_pos_local[:, 2], reward_flip_penalty),
                            torch.zeros_like(obj_pos_local[:, 2]))
    r_success = reward_success * success.float()
    r_failure = -5000.0 * failure.float()
    reward    = r_robot + r_tcp + r_pos + r_orient + r_smooth + r_tilt + r_static + r_table + r_flip + r_success + r_failure
    return reward, r_robot, r_tcp, r_pos, r_orient, r_smooth, r_tilt, r_static, r_success, r_failure, r_table, r_flip


@torch.jit.script
def compute_dones(
    pos_err: torch.Tensor,
    obj_yaw_err: torch.Tensor,
    obj_pos_local: torch.Tensor,
    success_pos_threshold: float,
    success_orient_threshold: float,
    flip_z_threshold: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    success = (
        (pos_err < success_pos_threshold)
        & (obj_yaw_err.abs() < success_orient_threshold)
        & (obj_pos_local[:, 2] <= flip_z_threshold)
    )
    fallen  = obj_pos_local[:, 2] < 0.05
    failure = fallen & ~success
    return success, failure
