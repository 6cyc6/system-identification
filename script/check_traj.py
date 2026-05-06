import argparse
import os
from pathlib import Path

import numpy as np
import scipy.linalg

from loguru import logger

from system_identification.excitation_generator_new import camera_box_clearance
from system_identification.excitation_optimization import (
    generateAsymFrictionReg,
    generateSymFrictionReg,
)
from system_identification.inertia_model import InertiaModel
from system_identification.utils import retrieve_robot_config


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
CAMERA_BOX_MIN_CLEARANCE = 0.0


def resolve_traj_path(path):
    if path is not None:
        return Path(path).expanduser().resolve()

    candidates = []
    folder = SCRIPT_DIR / "saves"
    if folder.exists():
        for pattern in ("*.csv", "*.npy"):
            candidates.extend(folder.glob(pattern))

    if not candidates:
        raise FileNotFoundError("No saved trajectory found under script/saves/")
    return max(candidates, key=lambda candidate: candidate.stat().st_mtime)


def load_trajectory(path):
    if path.suffix == ".npy":
        traj = np.load(path, allow_pickle=True).item()
        return traj["t"], traj["q"], traj["dq"], traj["ddq"]

    if path.suffix == ".csv":
        raw = np.genfromtxt(path, delimiter=",", names=True)
        columns = list(raw.dtype.names)
        t = np.asarray(raw["t"], dtype=float)
        q_cols = sorted(
            [name for name in columns if name.startswith("q_")],
            key=lambda name: int(name.split("_")[1]),
        )
        dq_cols = sorted(
            [name for name in columns if name.startswith("dq_")],
            key=lambda name: int(name.split("_")[1]),
        )
        ddq_cols = sorted(
            [name for name in columns if name.startswith("ddq_")],
            key=lambda name: int(name.split("_")[1]),
        )
        if not q_cols or not dq_cols or not ddq_cols:
            raise ValueError(f"CSV trajectory {path} is missing q/dq/ddq columns")

        q = np.column_stack([raw[name] for name in q_cols]).astype(float)
        dq = np.column_stack([raw[name] for name in dq_cols]).astype(float)
        ddq = np.column_stack([raw[name] for name in ddq_cols]).astype(float)
        return t, q, dq, ddq

    raise ValueError(f"Unsupported trajectory file format: {path.suffix}")


def scaled_position_limits(robot_config, position_bound_scale):
    upper = np.array(robot_config["upper_joint_pos_limits"], dtype=float)
    lower = np.array(robot_config["lower_joint_pos_limits"], dtype=float)

    if len(upper) > 1:
        lower[1] = -1.0
        upper[1] = 1.0

    return lower * position_bound_scale, upper * position_bound_scale


def max_abs_violation(values, lower, upper):
    lower_violation = np.maximum(lower - values, 0.0)
    upper_violation = np.maximum(values - upper, 0.0)
    return float(np.max(np.maximum(lower_violation, upper_violation)))


def max_abs_eq_violation(values, target):
    return float(np.max(np.abs(values - target)))


def report_constraint(name, value, passed):
    status = "PASS" if passed else "FAIL"
    logger.info(f"{status} | {name}: {value}")


def build_regressor(q, dq, ddq, inertia_model, friction_model=None):
    regressor = inertia_model.regressor(q, dq, ddq)
    if friction_model is None:
        return regressor
    if friction_model == "symmetric":
        return np.hstack((regressor, generateSymFrictionReg(dq)))
    if friction_model == "asymmetric":
        return np.hstack((regressor, generateAsymFrictionReg(dq)))
    raise ValueError(f"Invalid friction model: {friction_model}")


def condition_report(matrix, singular_tol=None):
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    if singular_values.size == 0:
        return {
            "condition_number": np.inf,
            "rank": 0,
            "min_singular_value": 0.0,
            "max_singular_value": 0.0,
        }

    if singular_tol is None:
        singular_tol = (
            singular_values[0]
            * max(matrix.shape)
            * np.finfo(singular_values.dtype).eps
        )
    rank = int(np.sum(singular_values > singular_tol))
    min_singular_value = float(singular_values[-1])
    max_singular_value = float(singular_values[0])
    condition_number = (
        np.inf
        if min_singular_value <= singular_tol
        else max_singular_value / min_singular_value
    )
    return {
        "condition_number": condition_number,
        "rank": rank,
        "min_singular_value": min_singular_value,
        "max_singular_value": max_singular_value,
    }


def gramian_condition_report(regressor, eig_tol=None):
    gramian = regressor.T @ regressor
    gramian = 0.5 * (gramian + gramian.T)
    eigvals = np.linalg.eigvalsh(gramian)
    min_eig = max(float(eigvals[0]), 0.0)
    max_eig = max(float(eigvals[-1]), 0.0)
    if eig_tol is None:
        eig_tol = max_eig * max(gramian.shape) * np.finfo(eigvals.dtype).eps
    rank = int(np.sum(eigvals > eig_tol))
    condition_number = np.inf if min_eig <= eig_tol else max_eig / min_eig
    return {
        "condition_number": condition_number,
        "rank": rank,
        "min_eigenvalue": min_eig,
        "max_eigenvalue": max_eig,
    }


def select_identifiable_columns(regressor, tol=None):
    _, r, pivots = scipy.linalg.qr(regressor, mode="economic", pivoting=True)
    diag = np.abs(np.diag(r))
    if diag.size == 0:
        return np.array([], dtype=int)

    if tol is None:
        tol = diag.max() * max(regressor.shape) * np.finfo(regressor.dtype).eps
    rank = int(np.sum(diag > tol))
    return np.sort(pivots[:rank])


def check_trajectory(
    t,
    q,
    dq,
    ddq,
    robot_config,
    *,
    eq_tol,
    inequality_tol,
    position_bound_scale,
    velocity_bound_scale,
    camera_clearance_min,
):
    init_pos = np.array(robot_config["init_pos"], dtype=float)
    init_vel = np.array(robot_config["init_vel"], dtype=float)
    raw_lower = np.array(robot_config["lower_joint_pos_limits"], dtype=float)
    raw_upper = np.array(robot_config["upper_joint_pos_limits"], dtype=float)
    scaled_lower, scaled_upper = scaled_position_limits(
        robot_config,
        position_bound_scale,
    )
    vel_limit = np.array(robot_config["joint_vel_limits"], dtype=float)
    scaled_vel_limit = velocity_bound_scale * np.minimum(vel_limit, 10000.0)

    reports = []
    reports.append(
        (
            "initial position q[0] == init_pos",
            max_abs_eq_violation(q[0], init_pos),
            max_abs_eq_violation(q[0], init_pos) <= eq_tol,
        )
    )
    reports.append(
        (
            "initial velocity dq[0] == init_vel",
            max_abs_eq_violation(dq[0], init_vel),
            max_abs_eq_violation(dq[0], init_vel) <= eq_tol,
        )
    )
    reports.append(
        (
            "initial acceleration ddq[0] == 0",
            max_abs_eq_violation(ddq[0], np.zeros_like(ddq[0])),
            max_abs_eq_violation(ddq[0], np.zeros_like(ddq[0])) <= eq_tol,
        )
    )

    raw_pos_violation = max_abs_violation(q, raw_lower, raw_upper)
    reports.append(
        (
            "raw joint position limits",
            raw_pos_violation,
            raw_pos_violation <= inequality_tol,
        )
    )

    scaled_pos_violation = max_abs_violation(q, scaled_lower, scaled_upper)
    reports.append(
        (
            f"scaled joint position limits (scale={position_bound_scale})",
            scaled_pos_violation,
            scaled_pos_violation <= inequality_tol,
        )
    )

    velocity_violation = max_abs_violation(dq, -vel_limit, vel_limit)
    reports.append(
        (
            "raw joint velocity limits",
            velocity_violation,
            velocity_violation <= inequality_tol,
        )
    )

    scaled_velocity_violation = max_abs_violation(
        dq,
        -scaled_vel_limit,
        scaled_vel_limit,
    )
    reports.append(
        (
            f"scaled joint velocity limits (scale={velocity_bound_scale})",
            scaled_velocity_violation,
            scaled_velocity_violation <= inequality_tol,
        )
    )

    camera_clearance = camera_box_clearance(q)
    camera_violation = max(camera_clearance_min - camera_clearance, 0.0)
    reports.append(
        (
            f"camera box clearance >= {camera_clearance_min}",
            camera_clearance,
            camera_violation <= inequality_tol,
        )
    )

    finite = bool(
        np.all(np.isfinite(t))
        and np.all(np.isfinite(q))
        and np.all(np.isfinite(dq))
        and np.all(np.isfinite(ddq))
    )
    reports.append(("finite t/q/dq/ddq", finite, finite))
    return reports


def main():
    original_cwd = Path.cwd()

    parser = argparse.ArgumentParser(description="Check trajectory constraints and regressor conditioning")
    parser.add_argument("trajectory", nargs="?", default=None)
    parser.add_argument("--robot", type=str, default="fr3")
    parser.add_argument(
        "--excite_type",
        type=str,
        choices=("cond", "condFriction"),
        default="cond",
    )
    parser.add_argument(
        "--friction_model",
        type=str,
        choices=("symmetric", "asymmetric"),
        default="symmetric",
    )
    parser.add_argument("--eq_tol", type=float, default=1e-6)
    parser.add_argument("--inequality_tol", type=float, default=1e-8)
    parser.add_argument("--position_bound_scale", type=float, default=0.8)
    parser.add_argument("--velocity_bound_scale", type=float, default=0.95)
    parser.add_argument("--camera_clearance_min", type=float, default=CAMERA_BOX_MIN_CLEARANCE)
    args = parser.parse_args()

    traj_path = resolve_traj_path(
        None if args.trajectory is None else original_cwd / args.trajectory
    )
    os.chdir(SCRIPT_DIR)

    logger.info(f"Checking trajectory: {traj_path}")
    t, q, dq, ddq = load_trajectory(traj_path)
    logger.info(f"Samples: {len(t)}, joints: {q.shape[1]}")

    robot_config = retrieve_robot_config(args.robot)
    if q.shape[1] != robot_config["njoints"]:
        raise ValueError(
            f"Trajectory has {q.shape[1]} joints, but {args.robot} config has "
            f"{robot_config['njoints']} joints"
        )

    reports = check_trajectory(
        t,
        q,
        dq,
        ddq,
        robot_config,
        eq_tol=args.eq_tol,
        inequality_tol=args.inequality_tol,
        position_bound_scale=args.position_bound_scale,
        velocity_bound_scale=args.velocity_bound_scale,
        camera_clearance_min=args.camera_clearance_min,
    )

    all_passed = True
    logger.info("Constraint report:")
    for name, value, passed in reports:
        all_passed = all_passed and passed
        report_constraint(name, value, passed)

    inertia_model = InertiaModel(args.robot)
    friction_model = args.friction_model if args.excite_type == "condFriction" else None
    regressor = build_regressor(q, dq, ddq, inertia_model, friction_model)
    w_report = condition_report(regressor)
    wtw_report = gramian_condition_report(regressor)
    identifiable_columns = select_identifiable_columns(regressor)
    identifiable_regressor = regressor[:, identifiable_columns]
    identifiable_w_report = condition_report(identifiable_regressor)
    identifiable_wtw_report = gramian_condition_report(identifiable_regressor)

    logger.info(f"Regressor W shape: {regressor.shape}")
    logger.info(
        "Condition number of W: "
        f"{w_report['condition_number']} "
        f"(rank={w_report['rank']}, "
        f"sigma_min={w_report['min_singular_value']}, "
        f"sigma_max={w_report['max_singular_value']})"
    )
    logger.info(
        "Condition number of W.T @ W: "
        f"{wtw_report['condition_number']} "
        f"(rank={wtw_report['rank']}, "
        f"lambda_min={wtw_report['min_eigenvalue']}, "
        f"lambda_max={wtw_report['max_eigenvalue']})"
    )
    logger.info(
        f"Identifiable-column W shape: {identifiable_regressor.shape} "
        f"({len(identifiable_columns)} of {regressor.shape[1]} columns)"
    )
    logger.info(
        "Condition number of identifiable-column W: "
        f"{identifiable_w_report['condition_number']} "
        f"(rank={identifiable_w_report['rank']}, "
        f"sigma_min={identifiable_w_report['min_singular_value']}, "
        f"sigma_max={identifiable_w_report['max_singular_value']})"
    )
    logger.info(
        "Condition number of identifiable-column W.T @ W: "
        f"{identifiable_wtw_report['condition_number']} "
        f"(rank={identifiable_wtw_report['rank']}, "
        f"lambda_min={identifiable_wtw_report['min_eigenvalue']}, "
        f"lambda_max={identifiable_wtw_report['max_eigenvalue']})"
    )
    logger.info(f"All constraints satisfied: {all_passed}")

    if not all_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
