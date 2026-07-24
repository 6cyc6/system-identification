"""Typed configuration, primitive, and result models for collision proxies.

Primitive coordinates are always expressed in their ``LinkProxy.frame``. The
validation in these dataclasses keeps malformed values from reaching fitting,
serialization, or visualization code.
"""

from __future__ import annotations

import math

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import yaml

from .constants import PROXY_KINDS, PROXY_MODES


Vector3 = tuple[float, float, float]
Matrix3 = tuple[Vector3, Vector3, Vector3]


def _vector3(value: Iterable[float]) -> Vector3:
    """Normalize an iterable to a finite, immutable three-vector."""
    array = np.asarray(tuple(value), dtype=float).reshape(-1)
    if array.size != 3 or not np.all(np.isfinite(array)):
        raise ValueError(f"Expected a finite 3-vector, got {value!r}")
    return (float(array[0]), float(array[1]), float(array[2]))


def _matrix3(value: Iterable[Iterable[float]]) -> Matrix3:
    """Normalize nested iterables to a finite, immutable 3-by-3 matrix."""
    array = np.asarray(tuple(tuple(row) for row in value), dtype=float)
    if array.shape != (3, 3) or not np.all(np.isfinite(array)):
        raise ValueError(f"Expected a finite 3x3 matrix, got {value!r}")
    return tuple(_vector3(row) for row in array)  # type: ignore[return-value]


@dataclass(frozen=True)
class SphereProxy:
    """A sphere expressed in its collision-group frame."""

    center: Vector3
    radius: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "center", _vector3(self.center))
        if not math.isfinite(self.radius) or self.radius <= 0.0:
            raise ValueError(
                f"Sphere radius must be finite and positive: {self.radius}"
            )


@dataclass(frozen=True)
class CapsuleProxy:
    """A capsule expressed as segment endpoints and a radius."""

    point_a: Vector3
    point_b: Vector3
    radius: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "point_a", _vector3(self.point_a))
        object.__setattr__(self, "point_b", _vector3(self.point_b))
        if not math.isfinite(self.radius) or self.radius <= 0.0:
            raise ValueError(
                f"Capsule radius must be finite and positive: {self.radius}"
            )


@dataclass(frozen=True)
class SphereFitMetrics:
    """Measured outward error of an exposed sphere-union surface."""

    max_overhang: float
    p95_overhang: float
    mean_overhang: float
    tolerance: float
    surface_samples: int

    def __post_init__(self) -> None:
        values = (
            self.max_overhang,
            self.p95_overhang,
            self.mean_overhang,
            self.tolerance,
        )
        if not all(
            math.isfinite(value) and value >= 0.0 for value in values
        ):
            raise ValueError(
                "Sphere-fit distances must be finite and non-negative"
            )
        if self.surface_samples <= 0:
            raise ValueError("Sphere-fit surface sample count must be positive")

    @property
    def tolerance_met(self) -> bool:
        """Whether the measured union overhang satisfies the configured limit."""
        return self.max_overhang <= self.tolerance + 1e-12


@dataclass(frozen=True)
class EllipsoidProxy:
    """A mesh-aligned ellipsoid enlarged to cover collision geometry."""

    center: Vector3
    radii: Vector3
    rotation: Matrix3
    axis_ratio: float
    elongated: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "center", _vector3(self.center))
        object.__setattr__(self, "radii", _vector3(self.radii))
        object.__setattr__(self, "rotation", _matrix3(self.rotation))
        if any(radius <= 0.0 for radius in self.radii):
            raise ValueError("Ellipsoid radii must be positive")
        rotation = np.asarray(self.rotation)
        if not np.allclose(
            rotation.T @ rotation,
            np.eye(3),
            atol=1e-7,
        ):
            raise ValueError("Ellipsoid rotation must be orthonormal")
        if not math.isfinite(self.axis_ratio) or self.axis_ratio < 1.0:
            raise ValueError("Ellipsoid axis ratio must be at least one")


@dataclass(frozen=True)
class LinkProxy:
    """Collision primitives associated with one anchor-link frame."""

    name: str
    frame: str
    source_links: tuple[str, ...]
    spheres: tuple[SphereProxy, ...] = ()
    sphere_fit: SphereFitMetrics | None = None
    capsules: tuple[CapsuleProxy, ...] = ()
    ellipsoids: tuple[EllipsoidProxy, ...] = ()

    def __post_init__(self) -> None:
        if not self.name or not self.frame:
            raise ValueError("Proxy name and frame must be non-empty")
        if not self.source_links:
            raise ValueError(f"Group {self.name!r} has no source links")
        if not self.spheres and self.sphere_fit is not None:
            raise ValueError(
                "Sphere-fit metrics require at least one sphere primitive"
            )


@dataclass(frozen=True)
class GeneratorConfig:
    """Numerical settings for collision proxy generation."""

    max_spheres: int = 8
    max_capsules: int = 3
    margin: float = 0.002
    max_sphere_overhang: float = 0.05
    sphere_surface_subdivisions: int = 2
    voxel_divisions: int = 100
    precision: float = 0.95
    elongation_threshold: float = 2.0
    voxel_padding: int = 2
    min_radius_vox: int = 2
    min_center_distance_vox: int = 2

    def __post_init__(self) -> None:
        if self.max_spheres <= 0 or self.max_capsules <= 0:
            raise ValueError("Primitive limits must be positive")
        if not math.isfinite(self.margin) or self.margin < 0.0:
            raise ValueError("margin must be finite and non-negative")
        if (
            not math.isfinite(self.max_sphere_overhang)
            or self.max_sphere_overhang < 0.0
        ):
            raise ValueError(
                "max_sphere_overhang must be finite and non-negative"
            )
        if self.sphere_surface_subdivisions < 0:
            raise ValueError(
                "sphere_surface_subdivisions must be non-negative"
            )
        if self.voxel_divisions <= 0:
            raise ValueError("voxel_divisions must be positive")
        if not 0.0 < self.precision <= 1.0:
            raise ValueError("precision must be in (0, 1]")
        if (
            not math.isfinite(self.elongation_threshold)
            or self.elongation_threshold <= 1.0
        ):
            raise ValueError("elongation_threshold must be greater than 1")
        if self.voxel_padding < 0:
            raise ValueError("voxel_padding must be non-negative")
        if self.min_radius_vox < 0:
            raise ValueError("min_radius_vox must be non-negative")
        if self.min_center_distance_vox <= 0:
            raise ValueError("min_center_distance_vox must be positive")


@dataclass(frozen=True)
class CollisionProxyConfig:
    """Settings loaded from the collision-proxy YAML file."""

    robot: str
    urdf: Path | None
    mode: str
    max_spheres: int
    max_capsules: int
    margin: float
    max_sphere_overhang: float
    sphere_surface_subdivisions: int
    voxel_divisions: int
    precision: float
    elongation_threshold: float
    group_overrides: Mapping[str, tuple[str, ...]]
    joint_position_overrides: Mapping[str, float]
    output_dir: Path | None
    initial_view: str
    proxy_opacity: float
    hold: bool

    def __post_init__(self) -> None:
        if not self.robot.strip():
            raise ValueError("Config key 'robot' must be non-empty.")
        if self.mode not in PROXY_MODES:
            raise ValueError(
                "Config key 'mode' must be both, all, spheres, capsules, "
                "or ellipsoids."
            )
        if self.initial_view not in PROXY_MODES:
            raise ValueError(
                "Config key 'initial_view' must be both, all, spheres, "
                "capsules, or ellipsoids."
            )
        if not 0.0 <= self.proxy_opacity <= 1.0:
            raise ValueError("Config key 'proxy_opacity' must be in [0, 1].")
        GeneratorConfig(
            max_spheres=self.max_spheres,
            max_capsules=self.max_capsules,
            margin=self.margin,
            max_sphere_overhang=self.max_sphere_overhang,
            sphere_surface_subdivisions=self.sphere_surface_subdivisions,
            voxel_divisions=self.voxel_divisions,
            precision=self.precision,
            elongation_threshold=self.elongation_threshold,
        )

    @property
    def generator_config(self) -> GeneratorConfig:
        """Return the numerical settings consumed by the generator."""
        return GeneratorConfig(
            max_spheres=self.max_spheres,
            max_capsules=self.max_capsules,
            margin=self.margin,
            max_sphere_overhang=self.max_sphere_overhang,
            sphere_surface_subdivisions=self.sphere_surface_subdivisions,
            voxel_divisions=self.voxel_divisions,
            precision=self.precision,
            elongation_threshold=self.elongation_threshold,
        )

    @classmethod
    def from_yaml(cls, file_path: str | Path) -> CollisionProxyConfig:
        """Load and validate a complete configuration from YAML."""
        path = Path(file_path).expanduser().resolve()
        with path.open("r", encoding="utf-8") as stream:
            values = yaml.safe_load(stream)
        if not isinstance(values, dict):
            raise TypeError("The configuration root must be a mapping/object.")

        expected_keys = {
            "robot",
            "urdf",
            "mode",
            "max_spheres",
            "max_capsules",
            "margin",
            "max_sphere_overhang",
            "sphere_surface_subdivisions",
            "voxel_divisions",
            "precision",
            "elongation_threshold",
            "group_overrides",
            "joint_position_overrides",
            "output_dir",
            "initial_view",
            "proxy_opacity",
            "hold",
        }
        unknown = sorted(set(values) - expected_keys)
        missing = sorted(expected_keys - set(values))
        if unknown:
            raise ValueError(
                f"Unknown configuration key(s): {', '.join(unknown)}"
            )
        if missing:
            raise ValueError(
                f"Missing configuration key(s): {', '.join(missing)}"
            )

        return cls(
            robot=_config_string(values["robot"], name="robot"),
            urdf=_config_optional_path(values["urdf"], name="urdf"),
            mode=_config_string(values["mode"], name="mode"),
            max_spheres=_config_integer(
                values["max_spheres"],
                name="max_spheres",
            ),
            max_capsules=_config_integer(
                values["max_capsules"],
                name="max_capsules",
            ),
            margin=_config_number(values["margin"], name="margin"),
            max_sphere_overhang=_config_number(
                values["max_sphere_overhang"],
                name="max_sphere_overhang",
            ),
            sphere_surface_subdivisions=_config_integer(
                values["sphere_surface_subdivisions"],
                name="sphere_surface_subdivisions",
            ),
            voxel_divisions=_config_integer(
                values["voxel_divisions"],
                name="voxel_divisions",
            ),
            precision=_config_number(
                values["precision"],
                name="precision",
            ),
            elongation_threshold=_config_number(
                values["elongation_threshold"],
                name="elongation_threshold",
            ),
            group_overrides=_config_groups(values["group_overrides"]),
            joint_position_overrides=_config_joint_positions(
                values["joint_position_overrides"]
            ),
            output_dir=_config_optional_path(
                values["output_dir"],
                name="output_dir",
            ),
            initial_view=_config_string(
                values["initial_view"],
                name="initial_view",
            ),
            proxy_opacity=_config_number(
                values["proxy_opacity"],
                name="proxy_opacity",
            ),
            hold=_config_boolean(values["hold"], name="hold"),
        )


@dataclass(frozen=True)
class CollisionProxyResult:
    """Complete proxy metadata and per-link primitives."""

    robot: str
    source_urdf: Path
    config: GeneratorConfig
    joint_positions: Mapping[str, float]
    groups: tuple[LinkProxy, ...]

    def __post_init__(self) -> None:
        if not self.robot:
            raise ValueError("robot must be non-empty")
        if not self.groups:
            raise ValueError("At least one collision group is required")


def _config_string(value: Any, *, name: str) -> str:
    """Read a required string from the user configuration."""
    if not isinstance(value, str):
        raise TypeError(f"Config key '{name}' must be a string.")
    return value


def _config_integer(value: Any, *, name: str) -> int:
    """Read an integer without accepting booleans as integers."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"Config key '{name}' must be an integer.")
    return value


def _config_number(value: Any, *, name: str) -> float:
    """Read a finite floating-point configuration value."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"Config key '{name}' must be a number.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Config key '{name}' must be finite.")
    return result


def _config_boolean(value: Any, *, name: str) -> bool:
    """Read a strict boolean configuration value."""
    if not isinstance(value, bool):
        raise TypeError(f"Config key '{name}' must be true or false.")
    return value


def _config_optional_path(value: Any, *, name: str) -> Path | None:
    """Convert a nullable path string without resolving its base directory."""
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(
            f"Config key '{name}' must be a path string or null."
        )
    return Path(value).expanduser()


def _config_groups(value: Any) -> dict[str, tuple[str, ...]]:
    """Validate explicit anchor-to-source-link grouping overrides."""
    if not isinstance(value, dict):
        raise TypeError("Config key 'group_overrides' must be a mapping.")
    groups: dict[str, tuple[str, ...]] = {}
    for anchor, members_value in value.items():
        if not isinstance(anchor, str) or not anchor.strip():
            raise TypeError(
                "Group anchor names must be non-empty strings."
            )
        if (
            not isinstance(members_value, list)
            or not members_value
            or any(
                not isinstance(member, str) or not member.strip()
                for member in members_value
            )
        ):
            raise TypeError(
                f"Group '{anchor}' must contain a non-empty list of link names."
            )
        groups[anchor.strip()] = tuple(
            member.strip() for member in members_value
        )
    return groups


def _config_joint_positions(value: Any) -> dict[str, float]:
    """Validate explicit reference joint positions."""
    if not isinstance(value, dict):
        raise TypeError(
            "Config key 'joint_position_overrides' must be a mapping."
        )
    positions: dict[str, float] = {}
    for name, position in value.items():
        if not isinstance(name, str) or not name.strip():
            raise TypeError("Joint names must be non-empty strings.")
        positions[name.strip()] = _config_number(
            position,
            name=f"joint_position_overrides.{name}",
        )
    return positions


def _mode_kinds(mode: str) -> frozenset[str]:
    """Return the concrete proxy kinds selected by one CLI mode."""
    if mode not in PROXY_MODES:
        raise ValueError(
            "mode must be 'both', 'all', 'spheres', 'capsules', "
            "or 'ellipsoids'"
        )
    if mode == "both":
        return frozenset(("spheres", "capsules"))
    if mode == "all":
        return frozenset(PROXY_KINDS)
    return frozenset((mode,))


__all__ = [
    "CapsuleProxy",
    "CollisionProxyConfig",
    "CollisionProxyResult",
    "EllipsoidProxy",
    "GeneratorConfig",
    "LinkProxy",
    "Matrix3",
    "SphereFitMetrics",
    "SphereProxy",
    "Vector3",
]
