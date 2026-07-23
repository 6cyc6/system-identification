"""Feasible initial-trajectory search for FR3 excitation optimization."""

import numpy as np

from loguru import logger

from ..model.workspace_constraints import robot_link_workspace_margins
from ..solver.mathematical_program_solver import (
    fourier_constraint_bounds,
    fourier_constraint_values,
)
from ..utils.fourier_utils import flatten_fourier_params
from .excitation_generator import (
    generate_random_param,
    is_traj_valid,
    obtain_fourier_traj,
)
from .validation import validate_fr3_trajectory


def _max_constraint_violation(values, lower, upper):
    values = np.asarray(values, dtype=float).reshape(-1)
    lower = np.asarray(lower, dtype=float).reshape(-1)
    upper = np.asarray(upper, dtype=float).reshape(-1)
    lower_violation = np.maximum(lower - values, 0.0)
    upper_violation = np.maximum(values - upper, 0.0)
    return float(max(np.max(lower_violation), np.max(upper_violation)))


def _min_or_inf(values):
    values = np.asarray(values, dtype=float).reshape(-1)
    if values.size == 0:
        return np.inf
    return float(np.min(values))


def _constraint_report(
    flat_params,
    q,
    fourier_config,
    robot_config,
    collision_checker,
    config,
):
    """Evaluate a random seed against the constraints used by IPOPT."""
    fourier_lower, fourier_upper = fourier_constraint_bounds(
        fourier_config,
        robot_config,
        velocity_limit_scale=config.fourier_velocity_limit_scale,
        position_limit_scale=config.fourier_position_limit_scale,
    )
    fourier_values = fourier_constraint_values(
        flat_params,
        fourier_config,
        robot_config,
    )
    fourier_violation = _max_constraint_violation(
        fourier_values,
        fourier_lower,
        fourier_upper,
    )

    q_sampled = q[:: max(1, int(config.camera_collision_stride))]
    path_values = [collision_checker.minimum_distance_constraint_values(q_sampled)]
    path_lower = [np.zeros_like(path_values[0], dtype=float)]
    path_upper = [np.ones_like(path_values[0], dtype=float)]

    self_collision_values = np.array([], dtype=float)
    if not config.disable_self_collision_constraints:
        self_collision_values = (
            collision_checker.robot_self_collision_pair_margins(q_sampled).reshape(-1)
        )
        path_values.append(self_collision_values)
        path_lower.append(
            np.full_like(
                self_collision_values,
                float(config.self_collision_clearance),
                dtype=float,
            )
        )
        path_upper.append(
            np.full_like(self_collision_values, np.inf, dtype=float)
        )

    if not config.disable_link_y_bounds:
        workspace_values = robot_link_workspace_margins(
            collision_checker,
            q_sampled,
            x_lower=config.link_x_lower,
            y_lower=config.link_y_lower,
            y_upper=config.link_y_upper,
            z_lower=config.link_z_lower,
        ).reshape(-1)
        path_values.append(workspace_values)
        path_lower.append(np.zeros_like(workspace_values, dtype=float))
        path_upper.append(np.full_like(workspace_values, np.inf, dtype=float))

    path_values = np.concatenate(path_values)
    path_lower = np.concatenate(path_lower)
    path_upper = np.concatenate(path_upper)
    path_violation = _max_constraint_violation(
        path_values,
        path_lower,
        path_upper,
    )

    njoints = int(robot_config["njoints"])
    fourier_values_by_joint = fourier_values.reshape(njoints, 5)
    fourier_upper_by_joint = fourier_upper.reshape(njoints, 5)
    n_collision = len(q_sampled)
    collision_values = path_values[:n_collision]
    collision_upper = path_upper[:n_collision]
    n_self_collision = self_collision_values.size
    workspace_values = path_values[n_collision + n_self_collision :]

    max_violation = max(fourier_violation, path_violation)
    return {
        "valid": max_violation
        <= max(0.0, float(config.early_stop_constraint_tol)),
        "max_violation": max_violation,
        "fourier_equality_residual": float(
            np.max(np.abs(fourier_values_by_joint[:, :3]))
        ),
        "fourier_velocity_margin": _min_or_inf(
            fourier_upper_by_joint[:, 3] - fourier_values_by_joint[:, 3]
        ),
        "fourier_position_margin": _min_or_inf(
            fourier_upper_by_joint[:, 4] - fourier_values_by_joint[:, 4]
        ),
        "drake_collision_margin": _min_or_inf(
            np.minimum(collision_values, collision_upper - collision_values)
        ),
        "self_collision_margin": _min_or_inf(
            self_collision_values - float(config.self_collision_clearance)
        ),
        "wall_margin": _min_or_inf(workspace_values),
    }


def generate_valid_initial_trajectory(
    config,
    fourier_config,
    robot_config,
    collision_checker,
):
    """Random-search for a feasible Fourier seed for the nonlinear optimizer."""
    for attempt in range(1, int(config.max_initial_attempts) + 1):
        params = generate_random_param(
            int(fourier_config["order"]),
            int(robot_config["njoints"]),
        )
        _, q, dq, ddq = obtain_fourier_traj(
            params,
            fourier_config,
            robot_config,
        )
        if not is_traj_valid(q, dq, ddq, robot_config):
            continue

        flat_params = flatten_fourier_params(params)
        report = _constraint_report(
            flat_params,
            q,
            fourier_config,
            robot_config,
            collision_checker,
            config,
        )
        if not report["valid"]:
            if attempt % 50 == 0:
                logger.info(
                    "Discarding initial candidate "
                    f"{attempt}: max_violation={report['max_violation']}, "
                    f"drake_collision_margin={report['drake_collision_margin']}, "
                    f"self_collision_margin={report['self_collision_margin']}, "
                    f"wall_margin={report['wall_margin']}"
                )
            continue

        eval_result = validate_fr3_trajectory(
            "Initial",
            flat_params,
            fourier_config,
            robot_config,
            collision_checker,
            config,
        )
        if eval_result["valid"]:
            logger.info(f"Accepted initial trajectory after {attempt} attempt(s).")
            return flat_params, eval_result

    raise RuntimeError(
        "Could not find a valid initial FR3 trajectory after "
        f"{config.max_initial_attempts} attempts."
    )
