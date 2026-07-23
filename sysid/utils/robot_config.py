"""Robot-model loading and joint configuration helpers."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Mapping, TypedDict

import numpy as np
from numpy.typing import NDArray

from .path_utils import BASE_DIR, resolve_repo_path

if TYPE_CHECKING:
    import pinocchio as pin


FloatArray = NDArray[np.float64]
ROBOT_DESCRIPTION_DIR = BASE_DIR / "robot_description"


class RobotConfig(TypedDict):
    """Joint limits and initial state consumed by trajectory generation."""

    njoints: int
    upper_joint_pos_limits: FloatArray
    lower_joint_pos_limits: FloatArray
    joint_vel_limits: FloatArray
    init_pos: FloatArray
    init_vel: FloatArray


def _find_robot_urdf(
    robot_name: str,
    urdf_path: str | Path | None = None,
) -> Path:
    """Resolve one robot URDF deterministically from ``BASE_DIR``.

    An explicit path may be absolute or repository-relative. Without one, the
    repository's two established layouts are checked before a sorted recursive
    lookup. The fallback rejects duplicate names instead of silently choosing
    whichever file ``os.walk`` happens to encounter first.
    """
    if not isinstance(robot_name, str) or not robot_name.strip():
        raise ValueError("robot_name must be a non-empty string")

    if urdf_path is not None:
        resolved_path = resolve_repo_path(urdf_path)
        if not resolved_path.is_file():
            raise FileNotFoundError(f"Robot URDF does not exist: {resolved_path}")
        if resolved_path.suffix.lower() != ".urdf":
            raise ValueError(f"Robot model must be a URDF file: {resolved_path}")
        return resolved_path

    filename = f"{robot_name}.urdf"
    description_dir = ROBOT_DESCRIPTION_DIR / f"{robot_name}_description"
    candidates = (
        description_dir / filename,
        description_dir / "urdf" / filename,
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    matches = sorted(ROBOT_DESCRIPTION_DIR.rglob(filename))
    if not matches:
        raise FileNotFoundError(
            f"Cannot find {filename} below {ROBOT_DESCRIPTION_DIR}"
        )
    if len(matches) > 1:
        formatted_matches = ", ".join(str(path) for path in matches)
        raise RuntimeError(
            f"Multiple URDF files match {filename}; pass urdf_path explicitly: "
            f"{formatted_matches}"
        )
    return matches[0].resolve()


def _load_pin_model_for_robot(
    robot_name: str,
    urdf_path: str | Path | None = None,
) -> pin.Model:
    import pinocchio as pin

    return pin.buildModelFromUrdf(str(_find_robot_urdf(robot_name, urdf_path)))


def pin_joint_config(
    robot_name: str,
    urdf_path: str | Path | None = None,
) -> tuple[int, FloatArray, FloatArray, FloatArray]:
    """Return joint counts and position/velocity bounds from the robot URDF."""
    model = _load_pin_model_for_robot(robot_name, urdf_path)
    njoints = model.njoints - 1
    continuous_joints = {
        joint_index
        for joint_index, configuration_size in enumerate(model.nqs[1:])
        if configuration_size > 1
    }

    upper_position_limits = np.empty(njoints, dtype=np.float64)
    lower_position_limits = np.empty(njoints, dtype=np.float64)
    for joint_index in range(njoints):
        if joint_index in continuous_joints:
            upper_limit, lower_limit = np.pi * 4, -np.pi * 4
        else:
            upper_limit = model.upperPositionLimit[joint_index]
            lower_limit = model.lowerPositionLimit[joint_index]
        upper_position_limits[joint_index] = upper_limit
        lower_position_limits[joint_index] = lower_limit

    return (
        njoints,
        upper_position_limits.copy(),
        lower_position_limits.copy(),
        np.asarray(model.velocityLimit, dtype=np.float64).copy(),
    )


def validate_robot_config(robot_config: Mapping[str, object]) -> None:
    """Validate dimensions, finite values, limits, and the initial state."""
    required_keys = {
        "njoints",
        "upper_joint_pos_limits",
        "lower_joint_pos_limits",
        "joint_vel_limits",
        "init_pos",
        "init_vel",
    }
    missing_keys = sorted(required_keys - robot_config.keys())
    if missing_keys:
        raise ValueError(
            f"Robot configuration is missing: {', '.join(missing_keys)}"
        )

    njoints = robot_config["njoints"]
    if isinstance(njoints, bool) or not isinstance(njoints, (int, np.integer)):
        raise TypeError("Robot configuration 'njoints' must be an integer")
    njoints = int(njoints)
    if njoints < 1:
        raise ValueError("Robot configuration 'njoints' must be positive")

    vectors: dict[str, FloatArray] = {}
    vector_names = (
        "upper_joint_pos_limits",
        "lower_joint_pos_limits",
        "joint_vel_limits",
        "init_pos",
        "init_vel",
    )
    for name in vector_names:
        try:
            vector = np.asarray(robot_config[name], dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise TypeError(f"Robot configuration '{name}' must be numeric") from error
        if vector.shape != (njoints,):
            raise ValueError(
                f"Robot configuration '{name}' must have shape ({njoints},), "
                f"got {vector.shape}"
            )
        if not np.all(np.isfinite(vector)):
            raise ValueError(
                f"Robot configuration '{name}' must contain finite values"
            )
        vectors[name] = vector

    lower = vectors["lower_joint_pos_limits"]
    upper = vectors["upper_joint_pos_limits"]
    velocity = vectors["joint_vel_limits"]
    initial_position = vectors["init_pos"]
    initial_velocity = vectors["init_vel"]

    if np.any(lower >= upper):
        raise ValueError(
            "Every lower joint-position limit must be below its upper limit"
        )
    if np.any(velocity <= 0.0):
        raise ValueError("Every joint-velocity limit must be positive")
    if np.any(initial_position < lower) or np.any(initial_position > upper):
        raise ValueError("Initial joint positions must lie within the position limits")
    if np.any(np.abs(initial_velocity) > velocity):
        raise ValueError("Initial joint velocities must lie within the velocity limits")


def retrieve_robot_config(
    robot_name: str,
    urdf_path: str | Path | None = None,
) -> RobotConfig:
    """Build the trajectory-generator configuration for a robot."""
    njoints, upper_limits, lower_limits, velocity_limits = pin_joint_config(
        robot_name,
        urdf_path,
    )
    robot_config: RobotConfig = {
        "njoints": njoints,
        "upper_joint_pos_limits": upper_limits.copy(),
        "lower_joint_pos_limits": lower_limits.copy(),
        "joint_vel_limits": velocity_limits.copy(),
        "init_pos": ((upper_limits + lower_limits) / 2.0).astype(
            np.float64,
            copy=False,
        ),
        "init_vel": np.zeros(njoints, dtype=np.float64),
    }
    validate_robot_config(robot_config)
    return robot_config
