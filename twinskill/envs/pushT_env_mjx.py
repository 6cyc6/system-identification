"""PushT MJX environment — Franka Panda (7-DOF) pushes a T-shape to a goal pose.

Mirrors PushTEnv6 (IsaacLab) but runs on JAX/MJX for massively parallel training
with Brax PPO.

Observations (37): joint_pos_norm (7), joint_vel (7), ee_pos (3), ee_quat (4),
                   fingertip_pos (3), obj_pos+quat (7), goal_xy (2), goal_quat (4).
Actions     ( 7): delta joint positions (radians), scaled by action_scale.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
from ml_collections import config_dict
import mujoco
from mujoco import mjx
from mujoco.mjx._src import math as mjx_math

from mujoco_playground._src import mjx_env

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_ENV_DIR   = Path(__file__).parent
_SCENE_XML = _ENV_DIR / "assets" / "scene_pushT_mjx.xml"
_PANDA_ASSETS_DIR = Path("/home/galois/git_downloads/6cyc6/mj_ctrl/franka_emika_panda/assets")
_TSHAPE_STL       = Path("/home/galois/assets/assets/obj/Tshape/t_shape.stl")

# Joint names / order (must match actuator order in XML)
_JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"]


def _load_assets() -> Dict[str, bytes]:
    """Load mesh assets (panda links + T-shape) into a filename→bytes dict."""
    assets: Dict[str, bytes] = {}
    for f in _PANDA_ASSETS_DIR.iterdir():
        if f.is_file():
            assets[f.name] = f.read_bytes()
    assets["t_shape.stl"] = _TSHAPE_STL.read_bytes()
    return assets


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def default_config() -> config_dict.ConfigDict:
    """Returns the default environment config."""
    return config_dict.create(
        # Timing: physics dt = 1/120 Hz, control at 1/30 Hz → decimation = 4
        sim_dt=0.00833333,   # ~120 Hz
        ctrl_dt=0.03333333,  # ~30 Hz  (n_substeps = 4)
        episode_length=450,  # 15 s at 30 Hz

        # Action
        action_scale=0.02,  # max joint displacement per policy step (radians)

        # Goal (in env-local frame, i.e. robot frame with robot at -0.5, 0, 0.1)
        goal_xy=(0.03, 0.0),       # desired T-shape XY position (world frame)
        goal_yaw=0.0,              # desired T-shape yaw (radians)

        # Object sampling
        obj_init_xy_range=(-0.15, 0.15),   # relative to goal (both axes)
        obj_init_yaw_range=(-math.pi, math.pi),

        # Workspace limits for failure detection (world frame X, Y)
        workspace_x=(-0.2, 0.7),
        workspace_y=(-0.35, 0.35),

        # Success thresholds
        success_pos_threshold=0.003,    # metres
        success_orient_threshold=0.08,  # radians

        # Reward weights (same as IsaacLab PushTEnv6)
        reward_pos_weight=1.5,
        reward_orient_weight=0.5,
        reward_smooth_weight=0.0001,
        reward_robot_dist_weight=100.0,
        reward_tilt_weight=1.0,
        tilt_threshold=0.866,           # cos(30°)
        box_null_dist=0.12,
        reward_success=100.0,
        static_penalty=-0.6,
        static_pos_threshold=0.001,
        static_rot_threshold=0.01,
        flip_z_threshold=0.121,         # metres world-z; resting ≈ 0.121
        reward_flip_penalty=-0.5,
        reward_table_penalty=-0.5,

        # MJX implementation
        impl="jax",
    )


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class PushTEnvMjx(mjx_env.MjxEnv):
    """PushT MJX environment with Brax-compatible reset/step interface."""

    def __init__(
        self,
        config: config_dict.ConfigDict = default_config(),
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ):
        super().__init__(config, config_overrides)

        # Load XML + assets
        xml_str = _SCENE_XML.read_text()
        assets  = _load_assets()
        mj_model = mujoco.MjModel.from_xml_string(xml_str, assets=assets)
        mj_model.opt.timestep = self.sim_dt

        self._mj_model  = mj_model
        self._mjx_model = mjx.put_model(mj_model, impl=self._config.impl)

        # Body / site / joint indices
        self._tshape_body  = mj_model.body("tshape").id
        self._ee_body      = mj_model.body("ee_flange").id
        # sites no longer needed for observations (kept for rendering reference)
        self._joint_ids    = [mj_model.joint(n).id for n in _JOINT_NAMES]
        self._qpos_adr     = [mj_model.jnt_qposadr[j] for j in self._joint_ids]
        self._qvel_adr     = [mj_model.jnt_dofadr[j] for j in self._joint_ids]
        self._tshape_qadr  = mj_model.jnt_qposadr[mj_model.body("tshape").jntadr[0]]
        self._goal_mocap   = mj_model.body("goal_marker").mocapid[0]

        # Joint limits: shape (7, 2)
        jnt_range = jp.array(mj_model.jnt_range[self._joint_ids])
        self._jnt_low  = jnt_range[:, 0]
        self._jnt_high = jnt_range[:, 1]

        # Actuator ctrl range (7, 2)
        self._ctrl_low  = jp.array(mj_model.actuator_ctrlrange[:, 0])
        self._ctrl_high = jp.array(mj_model.actuator_ctrlrange[:, 1])

        # Goal constants
        goal_xy = jp.array(config.goal_xy)
        self._goal_xy = goal_xy
        half_yaw = float(config.goal_yaw) * 0.5
        self._goal_quat = jp.array(
            [math.cos(half_yaw), 0.0, 0.0, math.sin(half_yaw)]
        )
        self._goal_yaw = float(config.goal_yaw)

        # Keyframe initial state
        kf_id = mj_model.keyframe("home").id
        self._init_qpos = jp.array(mj_model.keyframe("home").qpos)
        self._init_ctrl = jp.array(mj_model.keyframe("home").ctrl)

    # ------------------------------------------------------------------
    # MjxEnv abstract properties
    # ------------------------------------------------------------------

    @property
    def xml_path(self) -> str:
        return str(_SCENE_XML)

    @property
    def action_size(self) -> int:
        return 7

    @property
    def mj_model(self) -> mujoco.MjModel:
        return self._mj_model

    @property
    def mjx_model(self) -> mjx.Model:
        return self._mjx_model

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self, rng: jax.Array) -> mjx_env.State:
        rng, rng_obj_xy, rng_obj_yaw = jax.random.split(rng, 3)

        # --- Sample T-shape initial pose ---
        xy_range = float(self._config.obj_init_xy_range[1])
        obj_xy = self._goal_xy + jax.random.uniform(
            rng_obj_xy, (2,), minval=-xy_range, maxval=xy_range
        )
        # Clip to workspace
        obj_xy = jp.clip(
            obj_xy,
            jp.array([self._config.workspace_x[0], self._config.workspace_y[0]]),
            jp.array([self._config.workspace_x[1], self._config.workspace_y[1]]),
        )
        obj_z   = 0.121  # resting height
        obj_yaw = jax.random.uniform(
            rng_obj_yaw,
            minval=self._config.obj_init_yaw_range[0],
            maxval=self._config.obj_init_yaw_range[1],
        )
        half_yaw = obj_yaw * 0.5
        obj_quat = jp.array([jp.cos(half_yaw), 0.0, 0.0, jp.sin(half_yaw)])

        # --- Build qpos ---
        qpos = self._init_qpos
        # T-shape freejoint: indices [7:14] = x, y, z, qw, qx, qy, qz
        tshape_start = self._tshape_qadr
        qpos = qpos.at[tshape_start:tshape_start + 3].set(
            jp.array([obj_xy[0], obj_xy[1], obj_z])
        )
        qpos = qpos.at[tshape_start + 3:tshape_start + 7].set(obj_quat)

        data = mjx_env.make_data(
            self._mj_model,
            qpos=qpos,
            qvel=jp.zeros(self._mjx_model.nv),
            ctrl=self._init_ctrl,
            mocap_pos=jp.array([[self._goal_xy[0], self._goal_xy[1], 0.121]]),
            mocap_quat=jp.array([self._goal_quat]),
            impl=self._config.impl,
        )

        # Info carries per-step state not naturally in mjx.Data
        info: Dict[str, Any] = {
            "rng":           rng,
            "joint_target":  self._init_ctrl,  # accumulated position target (7,)
            "prev_action":   jp.zeros(7),
            "prev_ee_to_obj": jp.zeros(()),
            "prev_obj_pos":  jp.zeros(3),
            "prev_obj_yaw":  jp.zeros(()),
        }

        metrics = {
            "rew/robot":    jp.zeros(()),
            "rew/tcp":      jp.zeros(()),
            "rew/pos":      jp.zeros(()),
            "rew/orient":   jp.zeros(()),
            "rew/smooth":   jp.zeros(()),
            "rew/tilt":     jp.zeros(()),
            "rew/static":   jp.zeros(()),
            "rew/table":    jp.zeros(()),
            "rew/flip":     jp.zeros(()),
            "rew/success":  jp.zeros(()),
            "rew/failure":  jp.zeros(()),
            "success":      jp.zeros(()),
        }

        obs    = self._compute_obs(data, info)
        reward = jp.zeros(())
        done   = jp.zeros(())
        return mjx_env.State(data, obs, reward, done, metrics, info)

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        action = jp.clip(action, -1.0, 1.0)

        # Integrate delta joint position target
        joint_target = state.info["joint_target"] + action * self._config.action_scale
        joint_target = jp.clip(joint_target, self._ctrl_low, self._ctrl_high)

        # Step physics (n_substeps times)
        data = mjx_env.step(self._mjx_model, state.data, joint_target, self.n_substeps)

        # Intermediate state for reward/done computation
        inter = self._compute_intermediate(data)

        # Reward
        (reward, r_robot, r_tcp, r_pos, r_orient, r_smooth,
         r_tilt, r_static, r_success, r_failure, r_table, r_flip,
         success) = self._compute_rewards(inter, state.info, action, joint_target)

        # Done
        done, _ = self._compute_done(inter)

        # Observations
        new_info: Dict[str, Any] = {
            "rng":            state.info["rng"],
            "joint_target":   joint_target,
            "prev_action":    action,
            "prev_ee_to_obj": inter["ee_to_obj_3d"],
            "prev_obj_pos":   inter["obj_pos"],
            "prev_obj_yaw":   inter["obj_yaw"],
        }
        obs = self._compute_obs(data, new_info)

        metrics = {
            "rew/robot":   r_robot,
            "rew/tcp":     r_tcp,
            "rew/pos":     r_pos,
            "rew/orient":  r_orient,
            "rew/smooth":  r_smooth,
            "rew/tilt":    r_tilt,
            "rew/static":  r_static,
            "rew/table":   r_table,
            "rew/flip":    r_flip,
            "rew/success": r_success,
            "rew/failure": r_failure,
            "success":     success.astype(float),
        }

        return mjx_env.State(data, obs, reward, done.astype(float), metrics, new_info)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_intermediate(self, data: mjx.Data) -> Dict[str, jax.Array]:
        """Extract all quantities needed for reward/done from mjx.Data."""
        # Joint state — index with static addresses (traced as constants)
        qpos_adr = jp.array(self._qpos_adr)
        qvel_adr = jp.array(self._qvel_adr)
        joint_qpos = data.qpos[qpos_adr]
        joint_qvel = data.qvel[qvel_adr]

        # T-shape pose
        obj_pos  = data.xpos[self._tshape_body]   # (3,) world
        obj_quat = data.xquat[self._tshape_body]  # (4,) w,x,y,z

        obj_yaw = _quat_to_yaw(obj_quat)
        obj_yaw_err = _yaw_error(obj_quat, self._goal_yaw)

        pos_err = jp.linalg.norm(obj_pos[:2] - self._goal_xy)

        # EE body position and orientation (use body xpos/xquat — already quaternion)
        ee_pos  = data.xpos[self._ee_body]         # (3,)
        ee_quat = data.xquat[self._ee_body]        # (4,) w,x,y,z  (MuJoCo convention)

        # EE z-axis in world frame: third column of R(ee_quat)
        # For quat (w,x,y,z): R_col2 = 2*(xz+wy, yz-wx, 0.5-x²-y²)
        w, x, y, z = ee_quat[0], ee_quat[1], ee_quat[2], ee_quat[3]
        rz = jp.array([
            2.0 * (x * z + w * y),
            2.0 * (y * z - w * x),
            1.0 - 2.0 * (x * x + y * y),
        ])
        ee_tilt_dot = -rz[2]  # dot with (0,0,-1); 1 = pointing down

        # Fingertip: EE origin + 0.175m along EE z-axis
        tip_pos = ee_pos + 0.175 * rz
        # Fingertip z-margin above table top (z=0.1)
        fingertip_z_margin = tip_pos[2] - 0.1

        ee_to_obj    = jp.linalg.norm(ee_pos[:2] - obj_pos[:2])
        ee_to_obj_3d = jp.linalg.norm(tip_pos - obj_pos)

        return {
            "joint_qpos":        joint_qpos,
            "joint_qvel":        joint_qvel,
            "obj_pos":           obj_pos,
            "obj_quat":          obj_quat,
            "obj_yaw":           obj_yaw,
            "obj_yaw_err":       obj_yaw_err,
            "pos_err":           pos_err,
            "ee_pos":            ee_pos,
            "ee_quat":           ee_quat,
            "ee_tilt_dot":       ee_tilt_dot,
            "tip_pos":           tip_pos,
            "fingertip_z_margin": fingertip_z_margin,
            "ee_to_obj":         ee_to_obj,
            "ee_to_obj_3d":      ee_to_obj_3d,
        }

    def _compute_obs(
        self, data: mjx.Data, info: Dict[str, Any]
    ) -> jax.Array:
        """Build 37-D observation vector (matches IsaacLab compute_obs)."""
        inter = self._compute_intermediate(data)

        joint_pos_norm = (
            2.0 * (inter["joint_qpos"] - self._jnt_low) /
            (self._jnt_high - self._jnt_low) - 1.0
        )

        return jp.concatenate([
            joint_pos_norm,           # (7)  normalised joint positions
            inter["joint_qvel"],      # (7)  joint velocities
            inter["ee_pos"],          # (3)  EE xyz
            inter["ee_quat"],         # (4)  EE quaternion (w,x,y,z)
            inter["tip_pos"],         # (3)  fingertip xyz
            inter["obj_pos"],         # (3)  T-shape xyz
            inter["obj_quat"],        # (4)  T-shape quaternion
            self._goal_xy,            # (2)  goal xy
            self._goal_quat,          # (4)  goal quaternion
        ])  # total: 7+7+3+4+3 + 3+4 + 2+4 = 37

    def _compute_rewards(
        self,
        inter: Dict[str, jax.Array],
        info:  Dict[str, Any],
        action: jax.Array,
        joint_target: jax.Array,
    ):
        cfg = self._config

        r_pos = float(cfg.reward_pos_weight) * (
            jp.exp(-2.0 * inter["pos_err"]) - 1.0
        )
        r_orient = float(cfg.reward_orient_weight) * (
            jp.exp(-2.0 * jp.abs(inter["obj_yaw_err"])) - 1.0
        )
        r_smooth = -float(cfg.reward_smooth_weight) * jp.sum(
            (action - info["prev_action"]) ** 2
        )
        prev_d = info["prev_ee_to_obj"]
        active = (prev_d > float(cfg.box_null_dist)).astype(float)
        r_robot = float(cfg.reward_robot_dist_weight) * active * (
            prev_d - inter["ee_to_obj_3d"]
        )
        r_tcp = jp.sqrt(1.0 - jp.tanh(5.0 * inter["ee_to_obj"])) / 20.0
        r_tilt = float(cfg.reward_tilt_weight) * jp.minimum(
            inter["ee_tilt_dot"] - float(cfg.tilt_threshold), 0.0
        )

        obj_moved = jp.linalg.norm(inter["obj_pos"] - info["prev_obj_pos"])
        yaw_diff  = _angle_wrap(inter["obj_yaw"] - info["prev_obj_yaw"])
        is_static = (obj_moved < float(cfg.static_pos_threshold)) & (
            jp.abs(yaw_diff) < float(cfg.static_rot_threshold)
        )
        r_static = jp.where(is_static, float(cfg.static_penalty), 0.0)

        r_table = jp.where(
            inter["fingertip_z_margin"] < 0.005,
            float(cfg.reward_table_penalty), 0.0
        )
        r_flip = jp.where(
            inter["obj_pos"][2] > float(cfg.flip_z_threshold),
            float(cfg.reward_flip_penalty), 0.0
        )

        success, failure = self._check_terminal(inter)

        r_success = float(cfg.reward_success) * success.astype(float)
        r_failure = -5000.0 * failure.astype(float)

        reward = (
            r_robot + r_tcp + r_pos + r_orient + r_smooth +
            r_tilt + r_static + r_table + r_flip + r_success + r_failure
        )
        return (
            reward, r_robot, r_tcp, r_pos, r_orient, r_smooth,
            r_tilt, r_static, r_success, r_failure, r_table, r_flip,
            success,
        )

    def _check_terminal(
        self, inter: Dict[str, jax.Array]
    ):
        cfg = self._config
        success = (
            (inter["pos_err"]           < float(cfg.success_pos_threshold)) &
            (jp.abs(inter["obj_yaw_err"]) < float(cfg.success_orient_threshold)) &
            (inter["obj_pos"][2]        <= float(cfg.flip_z_threshold))
        )
        fallen  = inter["obj_pos"][2] < 0.05
        failure = fallen & ~success
        return success, failure

    def _compute_done(self, inter: Dict[str, jax.Array]):
        success, failure = self._check_terminal(inter)
        done = success | failure
        return done, success


# ---------------------------------------------------------------------------
# Pure JAX helpers (work on single env, no batching)
# ---------------------------------------------------------------------------

def _quat_to_yaw(quat: jax.Array) -> jax.Array:
    """Yaw from quaternion (w, x, y, z)."""
    w, x, y, z = quat[0], quat[1], quat[2], quat[3]
    return jp.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _yaw_error(quat: jax.Array, goal_yaw: float) -> jax.Array:
    """Signed yaw error wrapped to (−π, π]."""
    obj_yaw = _quat_to_yaw(quat)
    return _angle_wrap(obj_yaw - goal_yaw)


def _angle_wrap(angle: jax.Array) -> jax.Array:
    """Wrap angle to (−π, π]."""
    return jp.arctan2(jp.sin(angle), jp.cos(angle))


