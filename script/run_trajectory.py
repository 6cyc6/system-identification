"""Run a reference trajectory in MuJoCo with feedforward + PD control.

Loads a saved trajectory (.npy with keys t, q, dq, ddq), tracks it using
    tau = tau_ff(q_ref, qd_ref, qdd_ref) + kp*(q_ref - q) + kd*(qd_ref - qd)
where tau_ff is computed by MuJoCo inverse dynamics (mj_inverse).

Saves: t, q, qd, qdd (numerical differentiation of qd), tau → .npy
"""

import argparse
import os
import re
import time
from datetime import datetime
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
from loguru import logger

# Paths relative to the project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_ROBOT_URDFS = {
    "franka": _PROJECT_ROOT / "robot_description/fr3_description/fr3.urdf",
    "iiwa": _PROJECT_ROOT / "robot_description/iiwas_description/urdf/iiwas14_edit.urdf",
}
_TRAJECTORY_SEARCH_GROUPS = (
    (_PROJECT_ROOT / "saves", _PROJECT_ROOT / "script" / "saves"),
    (_PROJECT_ROOT / "experiments" / "traj_data",),
)
_CAMERA_BOXES = (
    {
        "name": "camera_box_1_pos_y",
        "center": np.array([0.165, 0.340, 0.170]),
        "half_size": np.array([0.170, 0.160, 0.340]) * 1.25 / 2.0,
        "rgba": np.array([0.85, 0.20, 0.20, 0.30]),
    },
    {
        "name": "camera_box_2_pos_y",
        "center": np.array([0.870, 0.365, 0.180]),
        "half_size": np.array([0.180, 0.110, 0.360]) * 1.25 / 2.0,
        "rgba": np.array([0.20, 0.45, 0.90, 0.30]),
    },
    {
        "name": "camera_box_1_neg_y",
        "center": np.array([0.165, -0.340, 0.170]),
        "half_size": np.array([0.170, 0.160, 0.340]) * 1.25 / 2.0,
        "rgba": np.array([0.85, 0.20, 0.20, 0.30]),
    },
    {
        "name": "camera_box_2_neg_y",
        "center": np.array([0.870, -0.365, 0.180]),
        "half_size": np.array([0.180, 0.110, 0.360]) * 1.25 / 2.0,
        "rgba": np.array([0.20, 0.45, 0.90, 0.30]),
    },
)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _load_mj_model(urdf_path: Path, sim_dt: float) -> mujoco.MjModel:
    """Load URDF into MuJoCo, resolving relative mesh paths as inline assets."""
    urdf_dir = urdf_path.parent
    xml = urdf_path.read_text()

    # Collect mesh files and inline them (MuJoCo resolves paths from cwd,
    # which may differ from the urdf location).
    rel_paths = re.findall(r'filename="([^"]+\.(?:stl|dae|obj))"', xml)
    assets = {}
    for rel in set(rel_paths):
        abs_p = (urdf_dir / rel).resolve()
        key   = abs_p.name
        if abs_p.exists() and key not in assets:
            assets[key] = abs_p.read_bytes()
    # Rewrite paths to bare filenames so MuJoCo finds them in the assets dict.
    xml = re.sub(r'filename="([^"]*/)?([^/"]+\.(?:stl|dae|obj))"', r'filename="\2"', xml)

    model = mujoco.MjModel.from_xml_string(xml, assets=assets)
    model.opt.timestep = sim_dt
    return model


def _latest_saved_trajectory() -> Path:
    for folders in _TRAJECTORY_SEARCH_GROUPS:
        candidates = []
        for folder in folders:
            if not folder.exists():
                continue
            for pattern in ("*.csv", "*.npy"):
                candidates.extend(folder.glob(pattern))
        if candidates:
            return max(candidates, key=lambda path: path.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError("No saved trajectory found under saves/, script/saves/, or experiments/traj_data/")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _load_reference_trajectory(traj_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if traj_path.suffix == ".npy":
        traj = np.load(traj_path, allow_pickle=True).item()
        return traj["t"], traj["q"], traj["dq"], traj["ddq"]

    if traj_path.suffix == ".csv":
        raw = np.genfromtxt(traj_path, delimiter=",", names=True)
        columns = list(raw.dtype.names)
        t = np.asarray(raw["t"], dtype=float)
        q_cols = sorted([name for name in columns if name.startswith("q_")], key=lambda name: int(name.split("_")[1]))
        dq_cols = sorted([name for name in columns if name.startswith("dq_")], key=lambda name: int(name.split("_")[1]))
        ddq_cols = sorted([name for name in columns if name.startswith("ddq_")], key=lambda name: int(name.split("_")[1]))
        if not q_cols or not dq_cols or not ddq_cols:
            raise ValueError(f"CSV trajectory {traj_path} is missing q/dq/ddq columns")
        q = np.column_stack([raw[name] for name in q_cols]).astype(float)
        dq = np.column_stack([raw[name] for name in dq_cols]).astype(float)
        ddq = np.column_stack([raw[name] for name in ddq_cols]).astype(float)
        return t, q, dq, ddq

    raise ValueError(f"Unsupported trajectory file format: {traj_path.suffix}")


def _update_box_visualization(viewer_ctx):
    scene = viewer_ctx.user_scn
    scene.ngeom = 0
    identity_mat = np.eye(3).reshape(-1)
    for box in _CAMERA_BOXES:
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(
            geom,
            mujoco.mjtGeom.mjGEOM_BOX,
            box["half_size"],
            box["center"],
            identity_mat,
            box["rgba"],
        )
        scene.ngeom += 1


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------


def run_trajectory(
    ref_t:   np.ndarray,   # (T,)
    ref_q:   np.ndarray,   # (T, njoints)
    ref_qd:  np.ndarray,   # (T, njoints)
    ref_qdd: np.ndarray,   # (T, njoints)
    *,
    urdf_path: Path,
    omega_n:   float = 20.0,  # closed-loop natural frequency (rad/s) for PD gain design
    sim_dt:    float = 0.001,
    visualize: bool  = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Tracks ref trajectory with  tau = tau_ff + kp(t)*(q_ref-q) + kd(t)*(qd_ref-qd).

    Gains are recomputed at every control step from the current mass-matrix diagonal:
        kp_j(t) = omega_n^2 * M_j(t),   kd_j(t) = 2*omega_n * M_j(t)
    This gives a constant closed-loop natural frequency omega_n regardless of
    configuration, avoiding the instability that fixed gains cause when the
    effective inertia changes by up to 20x along the trajectory.

    Returns
    -------
    t, q, qd, qdd, tau  — all shape (T, ...), recorded at each reference step.
    """
    model = _load_mj_model(urdf_path, sim_dt)
    data  = mujoco.MjData(model)

    njoints = ref_q.shape[1]
    ctrl_dt = float(ref_t[1] - ref_t[0])
    steps_per_ctrl = max(1, round(ctrl_dt / sim_dt))

    viewer_ctx = (
        mujoco.viewer.launch_passive(model, data) if visualize else None
    )
    if viewer_ctx is not None:
        _update_box_visualization(viewer_ctx)

    # Inverse-dynamics helper — separate MjData so simulation state stays clean
    data_inv = mujoco.MjData(model)

    # Initialise at reference start
    data.qpos[:njoints] = ref_q[0]
    data.qvel[:njoints] = ref_qd[0]
    mujoco.mj_forward(model, data)

    T = len(ref_t)
    log_t   = np.empty(T)
    log_q   = np.empty((T, njoints))
    log_qd  = np.empty((T, njoints))
    log_tau = np.empty((T, njoints))

    for i in range(T):
        # --- computed torque control ---
        # Evaluate mj_inverse at the ACTUAL state with desired acceleration
        # (reference acceleration + PD correction). This exactly cancels the
        # actual nonlinear dynamics and decouples the joints, giving a simple
        # closed-loop: qdd_err + 2*omega_n*qd_err + omega_n^2*q_err = 0.
        q_err  = ref_q[i]  - data.qpos[:njoints]
        qd_err = ref_qd[i] - data.qvel[:njoints]
        qdd_des = ref_qdd[i] + omega_n**2 * q_err + 2.0 * omega_n * qd_err

        data_inv.qpos[:njoints] = data.qpos[:njoints]   # actual state
        data_inv.qvel[:njoints] = data.qvel[:njoints]
        data_inv.qacc[:njoints] = qdd_des
        mujoco.mj_inverse(model, data_inv)
        tau = data_inv.qfrc_inverse[:njoints].copy()

        # --- record before stepping ---
        log_t[i]   = data.time
        log_q[i]   = data.qpos[:njoints].copy()
        log_qd[i]  = data.qvel[:njoints].copy()
        log_tau[i] = tau

        # --- apply torque and integrate ---
        data.qfrc_applied[:njoints] = tau
        wall_start = time.perf_counter()
        for _ in range(steps_per_ctrl):
            mujoco.mj_step(model, data)

        if viewer_ctx is not None and viewer_ctx.is_running():
            _update_box_visualization(viewer_ctx)
            viewer_ctx.sync()
            # pace to real-time
            elapsed = time.perf_counter() - wall_start
            remaining = ctrl_dt - elapsed
            if remaining > 0:
                time.sleep(remaining)

    if viewer_ctx is not None:
        viewer_ctx.close()

    log_qdd = np.gradient(log_qd, log_t, axis=0)
    return log_t, log_q, log_qd, log_qdd, log_tau


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run trajectory in MuJoCo")
    parser.add_argument("--traj_file", type=str, default=None,
                        help="Path to trajectory file (.csv or .npy). Defaults to the most recently saved file.")
    parser.add_argument("--robot", type=str, default="franka", choices=sorted(_ROBOT_URDFS.keys()),
                        help="Robot model to run. Defaults to Franka FR3.")
    parser.add_argument("--omega_n", type=float, default=20.0,
                        help="Closed-loop natural frequency for auto PD gain design (rad/s)")
    parser.add_argument("--sim_dt", type=float, default=0.001, help="MuJoCo timestep (s)")
    parser.add_argument("--output_dir", type=str,
                        default=str(_PROJECT_ROOT / "experiments/traj_data"))
    parser.add_argument("--visualize", action=argparse.BooleanOptionalAction, default=True,
                        help="Open MuJoCo viewer and play back the trajectory in real time")
    args = parser.parse_args()

    traj_path = Path(args.traj_file) if args.traj_file is not None else _latest_saved_trajectory()
    urdf_path = _ROBOT_URDFS[args.robot]

    # Load reference trajectory
    ref_t, ref_q, ref_qd, ref_qdd = _load_reference_trajectory(traj_path)
    logger.info(f"Loaded trajectory from {traj_path}: {ref_q.shape[0]} steps, {ref_q.shape[1]} joints")
    logger.info(f"Using robot model {args.robot} from {urdf_path}")

    model_nq = _load_mj_model(urdf_path, args.sim_dt).nq
    if ref_q.shape[1] != model_nq:
        raise ValueError(
            f"Trajectory joint count ({ref_q.shape[1]}) does not match {args.robot} model nq ({model_nq})"
        )

    # Run simulation
    logger.info(f"Running with omega_n={args.omega_n}, sim_dt={args.sim_dt}, visualize={args.visualize}")
    t, q, qd, qdd, tau = run_trajectory(
        ref_t, ref_q, ref_qd, ref_qdd,
        urdf_path=urdf_path, omega_n=args.omega_n, sim_dt=args.sim_dt, visualize=args.visualize,
    )

    # Tracking error summary
    pos_err = np.abs(q - ref_q[:len(q)]).max(axis=0)
    logger.info(f"Max position tracking error per joint (rad): {np.round(pos_err, 4)}")

    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    stamp = datetime.now().strftime("%d%m%Y%H%M%S")
    out_path = os.path.join(args.output_dir, f"mujoco_traj_{stamp}.npy")
    np.save(out_path, {"t": t, "q": q, "dq": qd, "ddq": qdd, "tau": tau}, allow_pickle=True)
    logger.info(f"Saved to {out_path}")
