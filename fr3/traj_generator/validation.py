import numpy as np

from loguru import logger

from .excitation_generator import is_traj_valid
from ..utils.fourier_utils import flat_params_to_traj, unflatten_fourier_params


def fourier_constraint_report(flat_params, fourier_config, robot_config):
    order = fourier_config["order"]
    duration = fourier_config["duration"]
    omega = 2.0 * np.pi / duration
    harmonic_ids = np.arange(1, order + 1, dtype=float)
    params = unflatten_fourier_params(flat_params, fourier_config, robot_config)
    a_params, b_params = params[0], params[1]

    equality_residual = max(
        float(np.max(np.abs(np.sum(a_params, axis=0)))),
        float(np.max(np.abs(np.sum(b_params / harmonic_ids[:, None], axis=0)))),
        float(np.max(np.abs(np.sum(b_params * harmonic_ids[:, None], axis=0)))),
    )

    velocity_limit = np.minimum(
        np.array(robot_config["joint_vel_limits"], dtype=float),
        10000.0,
    )
    upper_pos = np.array(robot_config["upper_joint_pos_limits"], dtype=float)
    lower_pos = np.array(robot_config["lower_joint_pos_limits"], dtype=float)
    if len(upper_pos) > 1:
        lower_pos[1] = -1.0
        upper_pos[1] = 1.0
    lower_pos *= 0.95
    upper_pos *= 0.95
    init_pos = np.array(robot_config["init_pos"], dtype=float)
    offset_limit = np.minimum(upper_pos - init_pos, init_pos - lower_pos)

    root = np.sqrt(a_params**2 + b_params**2)
    velocity_margin = float(np.min(0.95 * velocity_limit - np.sum(root, axis=0)))
    position_margin = float(
        np.min(offset_limit * omega - np.sum(root / harmonic_ids[:, None], axis=0))
    )
    return {
        "equality_residual": equality_residual,
        "velocity_margin": velocity_margin,
        "position_margin": position_margin,
    }


def validate_trajectory(
    label,
    flat_params,
    fourier_config,
    robot_config,
    collision_checker,
    args,
):
    t, q, dq, ddq = flat_params_to_traj(
        flat_params,
        fourier_config,
        robot_config,
    )
    fourier_report = fourier_constraint_report(
        flat_params, fourier_config, robot_config
    )
    equality_tol = max(0.0, float(args.early_stop_constraint_tol))
    valid_joint_limits = bool(is_traj_valid(q, dq, ddq, robot_config))
    validation_stride = max(1, int(args.validation_stride))
    q_sampled = q[::validation_stride]
    collision_values = collision_checker.constraint_values(q_sampled)
    drake_collision_margin = float(np.min(collision_values))
    drake_clearance = float(collision_checker.min_clearance(q_sampled))
    link_wall_margin = np.inf
    if not args.disable_link_y_bounds:
        link_wall_margin = float(
            np.min(
                collision_checker.robot_link_wall_margins(
                    q_sampled,
                    y_lower=args.link_y_lower,
                    y_upper=args.link_y_upper,
                    z_lower=args.link_z_lower,
                )
            )
        )
    valid_fourier_constraints = (
        fourier_report["equality_residual"] <= equality_tol
        and fourier_report["velocity_margin"] >= 0.0
        and fourier_report["position_margin"] >= 0.0
    )
    valid = (
        valid_fourier_constraints
        and valid_joint_limits
        and drake_collision_margin >= 0.0
        and link_wall_margin >= 0.0
    )
    logger.info(
        f"{label} trajectory validity with Drake camera boxes: {valid}; "
        f"Fourier constraints valid: {valid_fourier_constraints}; "
        f"Fourier equality residual: {fourier_report['equality_residual']}; "
        f"Fourier velocity margin: {fourier_report['velocity_margin']}; "
        f"Fourier position margin: {fourier_report['position_margin']}; "
        f"joint limits valid: {valid_joint_limits}; "
        f"minimum Drake collision margin: {drake_collision_margin}; "
        f"minimum signed clearance: {drake_clearance}; "
        f"minimum link wall margin: {link_wall_margin}; "
        f"validation stride: {validation_stride}"
    )
    return {
        "t": t,
        "q": q,
        "dq": dq,
        "ddq": ddq,
        "valid": valid,
        "valid_fourier_constraints": valid_fourier_constraints,
        "valid_joint_limits": valid_joint_limits,
        **fourier_report,
        "drake_collision_margin": drake_collision_margin,
        "drake_clearance": drake_clearance,
        "link_wall_margin": link_wall_margin,
    }
