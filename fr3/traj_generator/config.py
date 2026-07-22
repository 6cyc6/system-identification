"""Configuration loading for the FR3 excitation trajectory generator."""

import json
from dataclasses import dataclass, field, fields
from pathlib import Path
from types import NoneType
from typing import get_args, get_type_hints

import yaml


def _coerce_value(name, value, expected_type):
    """Validate one decoded YAML/JSON value and normalize numeric/path types."""
    union_types = get_args(expected_type)
    if union_types:
        if value is None and NoneType in union_types:
            return None
        for option in union_types:
            if option is NoneType:
                continue
            try:
                return _coerce_value(name, value, option)
            except TypeError:
                pass
        expected_names = " or ".join(option.__name__ for option in union_types)
        raise TypeError(f"Config key '{name}' must be {expected_names}.")

    if expected_type is Path:
        if not isinstance(value, (str, Path)):
            raise TypeError(f"Config key '{name}' must be a path string.")
        return Path(value).expanduser()
    if expected_type is bool:
        if not isinstance(value, bool):
            raise TypeError(f"Config key '{name}' must be true or false.")
        return value
    if expected_type is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"Config key '{name}' must be an integer.")
        return value
    if expected_type is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"Config key '{name}' must be a number.")
        return float(value)
    if expected_type is str:
        if not isinstance(value, str):
            raise TypeError(f"Config key '{name}' must be a string.")
        return value

    raise TypeError(f"Unsupported type for config key '{name}': {expected_type}")


@dataclass
class Fr3ExcitationConfig:
    """Typed settings consumed by ``generate_excitation_trajectory.py``."""

    robot: str
    robot_urdf_path: Path
    save_dir: Path
    run_name: str | None
    seed: int
    max_initial_attempts: int
    fourier_order: int
    fourier_duration: float
    excite_type: str
    friction_model: str
    objective_lambda: float
    eig_eps: float
    fourier_velocity_limit_scale: float
    fourier_position_limit_scale: float
    ipopt_max_iter: int
    ipopt_print_level: int
    ipopt_hessian_approximation: str
    log_condition_every: int
    best_condition_check_every: int
    initial_best_condition_number: float
    best_condition_fourier_velocity_margin_tolerance: float
    best_condition_fourier_position_margin_tolerance: float
    best_condition_fourier_velocity_margin_min: float
    best_condition_fourier_position_margin_min: float
    early_stop_constraint_tol: float
    camera_collision_stride: int
    validation_stride: int
    drake_min_distance: float
    drake_robot_sphere_radius: float
    drake_robot_link_samples: int
    drake_camera_chamfer_radius: float
    drake_xy_prism_height: float | None
    drake_physical_camera_height: bool
    camera_box_xy_scale: float
    camera_box_z_scale: float
    link_y_lower: float
    link_y_upper: float
    link_x_lower: float
    link_z_lower: float
    disable_self_collision_constraints: bool
    self_collision_clearance: float
    disable_link_y_bounds: bool
    no_identifiable_column_reduction: bool
    initial_only: bool
    no_save: bool
    plot: bool
    config_path: Path | None = field(init=False, default=None)

    def __post_init__(self):
        if self.excite_type not in {"cond", "condFriction"}:
            raise ValueError("Config key 'excite_type' must be cond or condFriction.")
        if self.friction_model not in {"symmetric", "asymmetric"}:
            raise ValueError(
                "Config key 'friction_model' must be symmetric or asymmetric."
            )
        if self.ipopt_hessian_approximation not in {"limited-memory", "exact"}:
            raise ValueError(
                "Config key 'ipopt_hessian_approximation' must be "
                "limited-memory or exact."
            )

        positive_integers = (
            "max_initial_attempts",
            "fourier_order",
            "ipopt_max_iter",
            "best_condition_check_every",
            "camera_collision_stride",
            "validation_stride",
            "drake_robot_link_samples",
        )
        for name in positive_integers:
            if getattr(self, name) < 1:
                raise ValueError(f"Config key '{name}' must be at least 1.")
        if self.fourier_duration <= 0.0:
            raise ValueError("Config key 'fourier_duration' must be positive.")
        if self.link_y_lower >= self.link_y_upper:
            raise ValueError("link_y_lower must be smaller than link_y_upper.")

    def to_dict(self):
        """Return only file-backed settings, excluding loader metadata."""
        return {
            item.name: getattr(self, item.name) for item in fields(self) if item.init
        }

    @classmethod
    def from_mapping(cls, values):
        """Build a config while rejecting missing, unknown, or mistyped keys."""
        if not isinstance(values, dict):
            raise TypeError("The configuration root must be a mapping/object.")

        init_fields = {item.name: item for item in fields(cls) if item.init}
        unknown = sorted(set(values) - set(init_fields))
        missing = sorted(set(init_fields) - set(values))
        if unknown:
            raise ValueError(f"Unknown configuration key(s): {', '.join(unknown)}")
        if missing:
            raise ValueError(f"Missing configuration key(s): {', '.join(missing)}")

        type_hints = get_type_hints(cls)
        normalized = {
            name: _coerce_value(name, values[name], type_hints[name])
            for name in init_fields
        }
        return cls(**normalized)

    @classmethod
    def from_yaml(cls, file_path):
        """Load settings from a YAML file, following the ManiSkill pattern."""
        path = Path(file_path).expanduser().resolve()
        with path.open("r", encoding="utf-8") as stream:
            config = cls.from_mapping(yaml.safe_load(stream))
        config.config_path = path
        return config

    @classmethod
    def from_json(cls, file_path):
        """Load settings from a JSON file, following the ManiSkill pattern."""
        path = Path(file_path).expanduser().resolve()
        with path.open("r", encoding="utf-8") as stream:
            config = cls.from_mapping(json.load(stream))
        config.config_path = path
        return config

    @classmethod
    def from_file(cls, file_path):
        """Dispatch to the YAML or JSON loader based on the file suffix."""
        path = Path(file_path).expanduser()
        suffix = path.suffix.lower()
        if suffix in {".yaml", ".yml"}:
            return cls.from_yaml(path)
        if suffix == ".json":
            return cls.from_json(path)
        raise ValueError(
            f"Unsupported config format '{suffix}'. Use .yaml, .yml, or .json."
        )
