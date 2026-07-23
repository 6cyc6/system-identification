"""Generate conservative sphere, capsule, and ellipsoid proxies from URDFs."""

from __future__ import annotations

import contextlib
import importlib
import importlib.util
import io
import math
import select
import sys
import time
import types
import xml.etree.ElementTree as ET

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import unquote, urlparse

import numpy as np
import trimesh
import yaml

from scipy.optimize import minimize
from scipy.spatial.transform import Rotation
from sysid.utils.path_utils import BASE_DIR, get_config_path


ROBOT_DESCRIPTION_DIR = BASE_DIR / "robot_description"
DEFAULT_CONFIG_PATH = get_config_path("fr3", "collision_proxy.yaml")
SCHEMA_VERSION = 1
SPHERE_FILENAME = "sphere_collisions.yaml"
CAPSULE_FILENAME = "capsule_collisions.yaml"
ELLIPSOID_FILENAME = "ellipsoid_collisions.yaml"
GRIPPER_LINK_TOKENS = ("hand", "gripper", "finger")
MESHCAT_ROOT_PATH = "/collision_proxy"
PROXY_OPACITY_SLIDER = "Proxy opacity"
JOINT_SLIDER_PREFIX = "Joint position: "
PROXY_KINDS = ("spheres", "capsules", "ellipsoids")
PROXY_MODES = ("both", "all", *PROXY_KINDS)

Vector3 = tuple[float, float, float]
Matrix3 = tuple[Vector3, Vector3, Vector3]


def _vector3(value: Iterable[float]) -> Vector3:
    array = np.asarray(tuple(value), dtype=float).reshape(-1)
    if array.size != 3 or not np.all(np.isfinite(array)):
        raise ValueError(f"Expected a finite 3-vector, got {value!r}")
    return (float(array[0]), float(array[1]), float(array[2]))


def _matrix3(value: Iterable[Iterable[float]]) -> Matrix3:
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
            raise ValueError(f"Sphere radius must be finite and positive: {self.radius}")


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
            raise ValueError(f"Capsule radius must be finite and positive: {self.radius}")


@dataclass(frozen=True)
class EllipsoidProxy:
    """An inertia-aligned ellipsoid enlarged to cover collision geometry."""

    center: Vector3
    radii: Vector3
    rotation: Matrix3
    inertial_radii: Vector3
    inertia_matrix: Matrix3
    mass: float
    axis_ratio: float
    elongated: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "center", _vector3(self.center))
        object.__setattr__(self, "radii", _vector3(self.radii))
        object.__setattr__(
            self,
            "inertial_radii",
            _vector3(self.inertial_radii),
        )
        object.__setattr__(self, "rotation", _matrix3(self.rotation))
        object.__setattr__(
            self,
            "inertia_matrix",
            _matrix3(self.inertia_matrix),
        )
        if any(radius <= 0.0 for radius in self.radii):
            raise ValueError("Ellipsoid radii must be positive")
        if any(radius <= 0.0 for radius in self.inertial_radii):
            raise ValueError("Inertial ellipsoid radii must be positive")
        rotation = np.asarray(self.rotation)
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-7):
            raise ValueError("Ellipsoid rotation must be orthonormal")
        if not math.isfinite(self.mass) or self.mass <= 0.0:
            raise ValueError("Ellipsoid mass must be finite and positive")
        if not math.isfinite(self.axis_ratio) or self.axis_ratio < 1.0:
            raise ValueError("Ellipsoid axis ratio must be at least one")


@dataclass(frozen=True)
class LinkProxy:
    """Collision primitives associated with one anchor-link frame."""

    name: str
    frame: str
    source_links: tuple[str, ...]
    spheres: tuple[SphereProxy, ...] = ()
    capsules: tuple[CapsuleProxy, ...] = ()
    ellipsoids: tuple[EllipsoidProxy, ...] = ()

    def __post_init__(self) -> None:
        if not self.name or not self.frame:
            raise ValueError("Proxy name and frame must be non-empty")
        if not self.source_links:
            raise ValueError(f"Group {self.name!r} has no source links")


@dataclass(frozen=True)
class GeneratorConfig:
    """Numerical settings for collision proxy generation."""

    max_spheres: int = 8
    max_capsules: int = 3
    margin: float = 0.002
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


def _config_string(value: Any, *, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"Config key '{name}' must be a string.")
    return value


def _config_integer(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"Config key '{name}' must be an integer.")
    return value


def _config_number(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"Config key '{name}' must be a number.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Config key '{name}' must be finite.")
    return result


def _config_boolean(value: Any, *, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"Config key '{name}' must be true or false.")
    return value


def _config_optional_path(value: Any, *, name: str) -> Path | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"Config key '{name}' must be a path string or null.")
    return Path(value).expanduser()


def _config_groups(value: Any) -> dict[str, tuple[str, ...]]:
    if not isinstance(value, dict):
        raise TypeError("Config key 'group_overrides' must be a mapping.")
    groups: dict[str, tuple[str, ...]] = {}
    for anchor, members_value in value.items():
        if not isinstance(anchor, str) or not anchor.strip():
            raise TypeError("Group anchor names must be non-empty strings.")
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
        groups[anchor.strip()] = tuple(member.strip() for member in members_value)
    return groups


def _config_joint_positions(value: Any) -> dict[str, float]:
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


@dataclass(frozen=True)
class _LinkInertia:
    mass: float
    center: np.ndarray
    matrix: np.ndarray


@dataclass(frozen=True)
class _Joint:
    name: str
    joint_type: str
    parent: str
    child: str
    origin: np.ndarray
    axis: np.ndarray
    lower: float | None
    upper: float | None
    mimic_joint: str | None
    mimic_multiplier: float
    mimic_offset: float


@dataclass(frozen=True)
class _GroupDefinition:
    name: str
    frame: str
    source_links: tuple[str, ...]
    maximize_internal_joints: bool = False


@dataclass
class _RobotGeometry:
    urdf_path: Path
    root: ET.Element
    link_order: tuple[str, ...]
    joints: tuple[_Joint, ...]
    link_meshes: dict[str, tuple[trimesh.Trimesh, ...]]
    link_inertias: dict[str, _LinkInertia]
    group_definitions: tuple[_GroupDefinition, ...] = ()
    joint_positions: dict[str, float] = field(default_factory=dict)
    world_poses: dict[str, np.ndarray] = field(default_factory=dict)
    group_meshes: dict[str, trimesh.Trimesh] = field(default_factory=dict)


@dataclass(frozen=True)
class _JointSlider:
    control_name: str
    joint_name: str
    minimum: float
    maximum: float
    step: float
    initial_value: float


def resolve_robot_urdf(
    robot: str,
    urdf_path: str | Path | None = None,
) -> Path:
    """Resolve a robot URDF, preferring a gripper model when one exists."""
    if not robot or not robot.strip():
        raise ValueError("robot must be a non-empty name")

    if urdf_path is not None:
        candidate = Path(urdf_path).expanduser()
        if not candidate.is_absolute():
            candidate = BASE_DIR / candidate
        candidate = candidate.resolve()
        if not candidate.is_file():
            raise FileNotFoundError(f"URDF does not exist: {candidate}")
        if candidate.suffix.lower() != ".urdf":
            raise ValueError(f"Robot description is not a URDF: {candidate}")
        return candidate

    description_dir = ROBOT_DESCRIPTION_DIR / f"{robot}_description"
    candidates = (
        description_dir / "urdf" / f"{robot}_gripper.urdf",
        description_dir / f"{robot}_gripper.urdf",
        description_dir / f"{robot}.urdf",
        description_dir / "urdf" / f"{robot}.urdf",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    matches = sorted(description_dir.rglob("*.urdf")) if description_dir.is_dir() else []
    if not matches:
        raise FileNotFoundError(
            f"No URDF found for {robot!r} below {description_dir}"
        )
    if len(matches) > 1:
        formatted = "\n  ".join(str(path) for path in matches)
        raise RuntimeError(
            f"Multiple URDFs are available for {robot!r}; pass --urdf explicitly:\n"
            f"  {formatted}"
        )
    return matches[0].resolve()


def default_output_dir(robot: str) -> Path:
    return ROBOT_DESCRIPTION_DIR / f"{robot}_description" / "collisions"


def _parse_floats(
    value: str | None,
    *,
    default: Sequence[float],
    count: int,
    label: str,
) -> np.ndarray:
    if value is None:
        result = np.asarray(default, dtype=float)
    else:
        result = np.fromstring(value, sep=" ", dtype=float)
    if result.size != count or not np.all(np.isfinite(result)):
        raise ValueError(f"{label} must contain {count} finite values, got {value!r}")
    return result


def _origin_transform(element: ET.Element | None) -> np.ndarray:
    transform = np.eye(4)
    if element is None:
        return transform
    xyz = _parse_floats(
        element.attrib.get("xyz"),
        default=(0.0, 0.0, 0.0),
        count=3,
        label="origin xyz",
    )
    rpy = _parse_floats(
        element.attrib.get("rpy"),
        default=(0.0, 0.0, 0.0),
        count=3,
        label="origin rpy",
    )
    transform[:3, :3] = Rotation.from_euler("xyz", rpy).as_matrix()
    transform[:3, 3] = xyz
    return transform


def _parse_link_inertia(link: ET.Element) -> _LinkInertia | None:
    inertial = link.find("inertial")
    if inertial is None:
        return None
    mass_element = inertial.find("mass")
    matrix_element = inertial.find("inertia")
    if mass_element is None or matrix_element is None:
        return None
    mass = float(mass_element.attrib["value"])
    if not math.isfinite(mass) or mass <= 0.0:
        raise ValueError(
            f"Link {link.attrib.get('name', '')!r} has invalid inertial mass"
        )
    values = {
        name: float(matrix_element.attrib[name])
        for name in ("ixx", "ixy", "ixz", "iyy", "iyz", "izz")
    }
    inertia_in_inertial_frame = np.array(
        (
            (values["ixx"], values["ixy"], values["ixz"]),
            (values["ixy"], values["iyy"], values["iyz"]),
            (values["ixz"], values["iyz"], values["izz"]),
        ),
        dtype=float,
    )
    origin = _origin_transform(inertial.find("origin"))
    rotation = origin[:3, :3]
    inertia_in_link_frame = (
        rotation @ inertia_in_inertial_frame @ rotation.T
    )
    if (
        not np.all(np.isfinite(inertia_in_link_frame))
        or np.linalg.eigvalsh(inertia_in_link_frame).min() <= 0.0
    ):
        raise ValueError(
            f"Link {link.attrib.get('name', '')!r} has invalid inertia"
        )
    return _LinkInertia(
        mass=mass,
        center=origin[:3, 3].copy(),
        matrix=inertia_in_link_frame,
    )


def _resolve_mesh_path(
    filename: str,
    *,
    urdf_path: Path,
    description_dir: Path,
) -> Path:
    if not filename:
        raise ValueError("URDF mesh filename is empty")

    parsed = urlparse(filename)
    candidates: list[Path] = []
    if parsed.scheme == "file":
        candidates.append(Path(unquote(parsed.path)))
    elif filename.startswith("package://"):
        package_relative = filename[len("package://") :]
        package_name, separator, relative = package_relative.partition("/")
        if not separator:
            raise ValueError(f"Invalid package URI: {filename}")
        candidates.extend(
            (
                ROBOT_DESCRIPTION_DIR / package_name / relative,
                description_dir / relative,
            )
        )
    elif parsed.scheme:
        raise ValueError(f"Unsupported mesh URI scheme in {filename!r}")
    else:
        raw_path = Path(filename)
        if raw_path.is_absolute():
            candidates.append(raw_path)
        else:
            # The description-root fallback supports the repository's FR3 URDF,
            # which was moved into an urdf/ subfolder without rewriting meshes/.
            candidates.extend(
                (
                    urdf_path.parent / raw_path,
                    description_dir / raw_path,
                    BASE_DIR / raw_path,
                )
            )

    for candidate in candidates:
        candidate = candidate.expanduser().resolve()
        if candidate.is_file():
            return candidate
    formatted = "\n  ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        f"Could not resolve collision mesh {filename!r}. Tried:\n  {formatted}"
    )


def _load_trimesh(path: Path) -> trimesh.Trimesh:
    try:
        loaded = trimesh.load(path, process=False)
    except Exception as error:
        raise ValueError(f"Failed to load collision mesh {path}: {error}") from error

    if isinstance(loaded, trimesh.Scene):
        try:
            loaded = loaded.dump(concatenate=True)
        except Exception as error:
            raise ValueError(f"Could not flatten mesh scene {path}: {error}") from error
    if not isinstance(loaded, trimesh.Trimesh):
        raise TypeError(f"{path} did not load as a triangle mesh")
    if len(loaded.vertices) == 0 or len(loaded.faces) == 0:
        raise ValueError(f"Collision mesh has no triangles: {path}")
    return loaded.copy()


def _geometry_mesh(
    geometry: ET.Element,
    *,
    urdf_path: Path,
    description_dir: Path,
) -> trimesh.Trimesh:
    mesh_element = geometry.find("mesh")
    if mesh_element is not None:
        path = _resolve_mesh_path(
            mesh_element.attrib.get("filename", ""),
            urdf_path=urdf_path,
            description_dir=description_dir,
        )
        mesh = _load_trimesh(path)
        scale = _parse_floats(
            mesh_element.attrib.get("scale"),
            default=(1.0, 1.0, 1.0),
            count=3,
            label="mesh scale",
        )
        mesh.vertices = np.asarray(mesh.vertices) * scale[None, :]
        return mesh

    box = geometry.find("box")
    if box is not None:
        size = _parse_floats(
            box.attrib.get("size"),
            default=(),
            count=3,
            label="box size",
        )
        if np.any(size <= 0.0):
            raise ValueError(f"Box dimensions must be positive: {size}")
        return trimesh.creation.box(extents=size)

    cylinder = geometry.find("cylinder")
    if cylinder is not None:
        radius = float(cylinder.attrib["radius"])
        length = float(cylinder.attrib["length"])
        if radius <= 0.0 or length <= 0.0:
            raise ValueError("Cylinder radius and length must be positive")
        return trimesh.creation.cylinder(radius=radius, height=length, sections=32)

    sphere = geometry.find("sphere")
    if sphere is not None:
        radius = float(sphere.attrib["radius"])
        if radius <= 0.0:
            raise ValueError("Sphere radius must be positive")
        return trimesh.creation.icosphere(subdivisions=2, radius=radius)

    capsule = geometry.find("capsule")
    if capsule is not None:
        radius = float(capsule.attrib["radius"])
        length = float(capsule.attrib["length"])
        if radius <= 0.0 or length < 0.0:
            raise ValueError("Capsule radius must be positive and length non-negative")
        mesh = trimesh.creation.capsule(radius=radius, height=length)
        mesh.apply_translation((0.0, 0.0, -0.5 * length))
        return mesh

    tags = [child.tag for child in geometry]
    raise ValueError(f"Unsupported URDF collision geometry: {tags}")


def _parse_joint(joint: ET.Element) -> _Joint:
    name = joint.attrib.get("name", "")
    joint_type = joint.attrib.get("type", "fixed")
    parent = joint.find("parent")
    child = joint.find("child")
    if not name or parent is None or child is None:
        raise ValueError("Every URDF joint requires name, parent, and child")
    axis = _parse_floats(
        None if joint.find("axis") is None else joint.find("axis").attrib.get("xyz"),
        default=(1.0, 0.0, 0.0),
        count=3,
        label=f"joint {name} axis",
    )
    axis_norm = np.linalg.norm(axis)
    if joint_type != "fixed" and axis_norm <= 1e-12:
        raise ValueError(f"Joint {name} has a zero axis")
    if axis_norm > 0.0:
        axis = axis / axis_norm

    limit = joint.find("limit")
    lower = None if limit is None or "lower" not in limit.attrib else float(limit.attrib["lower"])
    upper = None if limit is None or "upper" not in limit.attrib else float(limit.attrib["upper"])
    mimic = joint.find("mimic")
    mimic_joint = None if mimic is None else mimic.attrib.get("joint")
    mimic_multiplier = 1.0 if mimic is None else float(mimic.attrib.get("multiplier", "1"))
    mimic_offset = 0.0 if mimic is None else float(mimic.attrib.get("offset", "0"))
    return _Joint(
        name=name,
        joint_type=joint_type,
        parent=parent.attrib["link"],
        child=child.attrib["link"],
        origin=_origin_transform(joint.find("origin")),
        axis=axis,
        lower=lower,
        upper=upper,
        mimic_joint=mimic_joint,
        mimic_multiplier=mimic_multiplier,
        mimic_offset=mimic_offset,
    )


def _read_robot_geometry(urdf_path: Path) -> _RobotGeometry:
    try:
        root = ET.parse(urdf_path).getroot()
    except ET.ParseError as error:
        raise ValueError(f"Invalid URDF XML in {urdf_path}: {error}") from error

    description_dir = urdf_path.parent
    while description_dir != ROBOT_DESCRIPTION_DIR and not description_dir.name.endswith(
        "_description"
    ):
        if description_dir.parent == description_dir:
            description_dir = urdf_path.parent
            break
        description_dir = description_dir.parent
    if description_dir == ROBOT_DESCRIPTION_DIR:
        description_dir = urdf_path.parent

    link_order: list[str] = []
    link_meshes: dict[str, tuple[trimesh.Trimesh, ...]] = {}
    link_inertias: dict[str, _LinkInertia] = {}
    for link in root.findall("link"):
        name = link.attrib.get("name", "")
        if not name:
            raise ValueError("URDF link is missing a name")
        link_order.append(name)
        link_inertia = _parse_link_inertia(link)
        if link_inertia is not None:
            link_inertias[name] = link_inertia
        meshes: list[trimesh.Trimesh] = []
        for collision in link.findall("collision"):
            geometry = collision.find("geometry")
            if geometry is None:
                continue
            mesh = _geometry_mesh(
                geometry,
                urdf_path=urdf_path,
                description_dir=description_dir,
            )
            mesh.apply_transform(_origin_transform(collision.find("origin")))
            meshes.append(mesh)
        if meshes:
            link_meshes[name] = tuple(meshes)

    joints = tuple(_parse_joint(joint) for joint in root.findall("joint"))
    return _RobotGeometry(
        urdf_path=urdf_path,
        root=root,
        link_order=tuple(link_order),
        joints=joints,
        link_meshes=link_meshes,
        link_inertias=link_inertias,
    )


def _is_gripper_link(name: str) -> bool:
    lowered = name.lower()
    return any(token in lowered for token in GRIPPER_LINK_TOKENS)


def _make_group_definitions(
    robot_geometry: _RobotGeometry,
    group_overrides: Mapping[str, Sequence[str]] | None,
) -> tuple[_GroupDefinition, ...]:
    collision_links = set(robot_geometry.link_meshes)
    link_index = {name: index for index, name in enumerate(robot_geometry.link_order)}
    link_names = set(robot_geometry.link_order)
    assigned: set[str] = set()
    definitions: list[_GroupDefinition] = []

    for anchor, members_value in (group_overrides or {}).items():
        if anchor not in link_names:
            raise ValueError(f"Unknown group anchor link: {anchor}")
        members = tuple(dict.fromkeys(members_value))
        if not members:
            raise ValueError(f"Explicit group {anchor!r} has no member links")
        unknown = set(members) - collision_links
        if unknown:
            raise ValueError(
                f"Explicit group {anchor!r} contains links without collision geometry: "
                f"{sorted(unknown)}"
            )
        duplicate = assigned.intersection(members)
        if duplicate:
            raise ValueError(f"Links occur in multiple explicit groups: {sorted(duplicate)}")
        assigned.update(members)
        definitions.append(
            _GroupDefinition(
                name=anchor,
                frame=anchor,
                source_links=members,
                maximize_internal_joints=any(_is_gripper_link(name) for name in members),
            )
        )

    candidates = {
        name for name in collision_links - assigned if _is_gripper_link(name)
    }
    adjacency: dict[str, set[str]] = {name: set() for name in candidates}
    for joint in robot_geometry.joints:
        if joint.parent in candidates and joint.child in candidates:
            adjacency[joint.parent].add(joint.child)
            adjacency[joint.child].add(joint.parent)

    while candidates:
        start = min(candidates, key=link_index.__getitem__)
        stack = [start]
        component: set[str] = set()
        while stack:
            current = stack.pop()
            if current in component:
                continue
            component.add(current)
            stack.extend(adjacency[current] - component)
        candidates -= component
        members = tuple(sorted(component, key=link_index.__getitem__))
        if len(members) == 1:
            continue
        component_roots = [
            name
            for name in members
            if not any(
                joint.child == name and joint.parent in component
                for joint in robot_geometry.joints
            )
        ]
        if len(component_roots) != 1:
            raise ValueError(
                "Could not infer one gripper anchor for "
                f"{members}; pass --group ANCHOR=LINK,..."
            )
        anchor = component_roots[0]
        definitions.append(
            _GroupDefinition(
                name=anchor,
                frame=anchor,
                source_links=members,
                maximize_internal_joints=True,
            )
        )
        assigned.update(members)

    for link in robot_geometry.link_order:
        if link in collision_links and link not in assigned:
            definitions.append(
                _GroupDefinition(name=link, frame=link, source_links=(link,))
            )

    definitions.sort(
        key=lambda group: min(link_index[name] for name in group.source_links)
    )
    return tuple(definitions)


def _clamp_reference_position(joint: _Joint, value: float = 0.0) -> float:
    if joint.joint_type == "fixed":
        return 0.0
    if joint.lower is not None:
        value = max(value, joint.lower)
    if joint.upper is not None:
        value = min(value, joint.upper)
    return float(value)


def _joint_paths_from_anchor(
    anchor: str,
    sources: Sequence[str],
    joints: Sequence[_Joint],
) -> set[str]:
    parent_joint = {joint.child: joint for joint in joints}
    path_names: set[str] = set()
    for source in sources:
        current = source
        visited: set[str] = set()
        while current != anchor:
            if current in visited or current not in parent_joint:
                raise ValueError(
                    f"Group anchor {anchor!r} is not an ancestor of source link {source!r}"
                )
            visited.add(current)
            joint = parent_joint[current]
            path_names.add(joint.name)
            current = joint.parent
    return path_names


def _reference_joint_positions(
    robot_geometry: _RobotGeometry,
    group_definitions: Sequence[_GroupDefinition],
    overrides: Mapping[str, float] | None,
) -> dict[str, float]:
    joints_by_name = {joint.name: joint for joint in robot_geometry.joints}
    unknown = set(overrides or {}) - set(joints_by_name)
    if unknown:
        raise ValueError(f"Unknown joint-position overrides: {sorted(unknown)}")

    positions = {
        joint.name: _clamp_reference_position(joint)
        for joint in robot_geometry.joints
        if joint.joint_type != "fixed"
    }
    for group in group_definitions:
        if not group.maximize_internal_joints:
            continue
        internal = _joint_paths_from_anchor(
            group.frame,
            group.source_links,
            robot_geometry.joints,
        )
        for name in internal:
            joint = joints_by_name[name]
            if joint.mimic_joint is None and joint.upper is not None:
                positions[name] = float(joint.upper)

    positions.update({name: float(value) for name, value in (overrides or {}).items()})

    unresolved = {
        joint.name
        for joint in robot_geometry.joints
        if joint.joint_type != "fixed" and joint.mimic_joint is not None
    }
    for _ in range(len(unresolved) + 1):
        progressed = False
        for name in tuple(unresolved):
            if name in (overrides or {}):
                unresolved.remove(name)
                progressed = True
                continue
            joint = joints_by_name[name]
            if joint.mimic_joint not in positions:
                continue
            value = (
                positions[joint.mimic_joint] * joint.mimic_multiplier
                + joint.mimic_offset
            )
            positions[name] = _clamp_reference_position(joint, value)
            unresolved.remove(name)
            progressed = True
        if not unresolved or not progressed:
            break
    if unresolved:
        raise ValueError(f"Unresolved mimic joints: {sorted(unresolved)}")

    for name, value in positions.items():
        joint = joints_by_name[name]
        if joint.lower is not None and value < joint.lower - 1e-12:
            raise ValueError(f"Joint {name}={value} is below lower limit {joint.lower}")
        if joint.upper is not None and value > joint.upper + 1e-12:
            raise ValueError(f"Joint {name}={value} is above upper limit {joint.upper}")
    return positions


def _joint_motion_transform(joint: _Joint, position: float) -> np.ndarray:
    transform = np.eye(4)
    if joint.joint_type in ("revolute", "continuous"):
        transform[:3, :3] = Rotation.from_rotvec(joint.axis * position).as_matrix()
    elif joint.joint_type == "prismatic":
        transform[:3, 3] = joint.axis * position
    elif joint.joint_type != "fixed":
        raise ValueError(
            f"Unsupported joint type {joint.joint_type!r} for joint {joint.name}"
        )
    return transform


def _world_link_poses(
    link_order: Sequence[str],
    joints: Sequence[_Joint],
    joint_positions: Mapping[str, float],
) -> dict[str, np.ndarray]:
    child_links = {joint.child for joint in joints}
    roots = [link for link in link_order if link not in child_links]
    if not roots:
        raise ValueError("URDF kinematic graph has no root link")

    children: dict[str, list[_Joint]] = {}
    for joint in joints:
        children.setdefault(joint.parent, []).append(joint)

    poses: dict[str, np.ndarray] = {}
    stack = [(root, np.eye(4)) for root in reversed(roots)]
    while stack:
        link, world_pose = stack.pop()
        if link in poses:
            raise ValueError(f"URDF kinematic graph revisits link {link!r}")
        poses[link] = world_pose
        for joint in reversed(children.get(link, [])):
            position = joint_positions.get(joint.name, 0.0)
            child_pose = world_pose @ joint.origin @ _joint_motion_transform(
                joint, position
            )
            stack.append((joint.child, child_pose))
    missing = set(link_order) - set(poses)
    if missing:
        raise ValueError(f"URDF contains unreachable links: {sorted(missing)}")
    return poses


def _joint_slider_specs(
    robot_geometry: _RobotGeometry,
    joint_positions: Mapping[str, float],
) -> tuple[_JointSlider, ...]:
    specs: list[_JointSlider] = []
    for joint in robot_geometry.joints:
        if joint.joint_type == "fixed" or joint.mimic_joint is not None:
            continue
        initial_value = float(joint_positions.get(joint.name, 0.0))
        if joint.joint_type in {"revolute", "continuous"}:
            default_extent = math.pi
            step = 0.01
        elif joint.joint_type == "prismatic":
            default_extent = 0.1
            step = 0.001
        else:
            continue
        minimum = (
            float(joint.lower)
            if joint.lower is not None
            else initial_value - default_extent
        )
        maximum = (
            float(joint.upper)
            if joint.upper is not None
            else initial_value + default_extent
        )
        if maximum <= minimum:
            continue
        step = min(step, (maximum - minimum) / 100.0)
        specs.append(
            _JointSlider(
                control_name=f"{JOINT_SLIDER_PREFIX}{joint.name}",
                joint_name=joint.name,
                minimum=minimum,
                maximum=maximum,
                step=step,
                initial_value=float(
                    np.clip(initial_value, minimum, maximum)
                ),
            )
        )
    return tuple(specs)


def _add_joint_sliders(
    meshcat: Any,
    specs: Sequence[_JointSlider],
    joint_positions: Mapping[str, float],
) -> dict[str, float]:
    positions = dict(joint_positions)
    for spec in specs:
        positions[spec.joint_name] = meshcat.AddSlider(
            spec.control_name,
            min=spec.minimum,
            max=spec.maximum,
            step=spec.step,
            value=spec.initial_value,
        )
    return positions


def _update_mimic_joint_positions(
    joints: Sequence[_Joint],
    joint_positions: dict[str, float],
) -> None:
    unresolved = {
        joint.name
        for joint in joints
        if joint.joint_type != "fixed" and joint.mimic_joint is not None
    }
    joints_by_name = {joint.name: joint for joint in joints}
    for _ in range(len(unresolved) + 1):
        progressed = False
        for name in tuple(unresolved):
            joint = joints_by_name[name]
            if joint.mimic_joint not in joint_positions:
                continue
            joint_positions[name] = _clamp_reference_position(
                joint,
                joint_positions[joint.mimic_joint]
                * joint.mimic_multiplier
                + joint.mimic_offset,
            )
            unresolved.remove(name)
            progressed = True
        if not unresolved or not progressed:
            break
    if unresolved:
        raise ValueError(
            f"Unresolved interactive mimic joints: {sorted(unresolved)}"
        )


def _read_joint_sliders(
    meshcat: Any,
    specs: Sequence[_JointSlider],
    robot_geometry: _RobotGeometry,
    joint_positions: Mapping[str, float],
) -> tuple[dict[str, float], bool]:
    positions = dict(joint_positions)
    changed = False
    for spec in specs:
        value = float(meshcat.GetSliderValue(spec.control_name))
        if not math.isclose(
            value,
            positions.get(spec.joint_name, value),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            positions[spec.joint_name] = value
            changed = True
    if changed:
        _update_mimic_joint_positions(robot_geometry.joints, positions)
    return positions, changed


def _combine_group_meshes(
    robot_geometry: _RobotGeometry,
    group_definitions: Sequence[_GroupDefinition],
    world_poses: Mapping[str, np.ndarray],
) -> dict[str, trimesh.Trimesh]:
    combined: dict[str, trimesh.Trimesh] = {}
    for group in group_definitions:
        inverse_anchor = np.linalg.inv(world_poses[group.frame])
        meshes: list[trimesh.Trimesh] = []
        for link in group.source_links:
            link_to_anchor = inverse_anchor @ world_poses[link]
            for source_mesh in robot_geometry.link_meshes[link]:
                mesh = source_mesh.copy()
                mesh.apply_transform(link_to_anchor)
                meshes.append(mesh)
        if not meshes:
            raise ValueError(f"Collision group {group.name!r} contains no geometry")
        group_mesh = trimesh.util.concatenate(meshes)
        if not isinstance(group_mesh, trimesh.Trimesh) or len(group_mesh.faces) == 0:
            raise ValueError(f"Failed to combine collision group {group.name!r}")
        combined[group.name] = group_mesh
    return combined


def _mesh_inertia(mesh: trimesh.Trimesh) -> _LinkInertia:
    candidates = (mesh, mesh.convex_hull)
    for candidate in candidates:
        try:
            properties = candidate.mass_properties
            mass = abs(float(properties.mass))
            center = np.asarray(properties.center_mass, dtype=float)
            matrix = np.asarray(properties.inertia, dtype=float)
            if float(properties.mass) < 0.0:
                matrix = -matrix
            matrix = 0.5 * (matrix + matrix.T)
            if (
                mass > 1e-12
                and np.all(np.isfinite(center))
                and np.all(np.isfinite(matrix))
                and np.linalg.eigvalsh(matrix).min() > 1e-14
            ):
                return _LinkInertia(
                    mass=mass,
                    center=center,
                    matrix=matrix,
                )
        except Exception:
            continue

    points = np.asarray(mesh.vertices, dtype=float)
    center = points.mean(axis=0)
    centered = points - center[None, :]
    covariance = centered.T @ centered / max(1, len(centered))
    matrix = np.trace(covariance) * np.eye(3) - covariance
    matrix += np.eye(3) * max(1e-12, np.trace(matrix) * 1e-9)
    return _LinkInertia(mass=1.0, center=center, matrix=matrix)


def _group_inertia(
    robot_geometry: _RobotGeometry,
    group: _GroupDefinition,
    world_poses: Mapping[str, np.ndarray],
    mesh: trimesh.Trimesh,
) -> _LinkInertia:
    if not all(
        link in robot_geometry.link_inertias for link in group.source_links
    ):
        return _mesh_inertia(mesh)

    inverse_anchor = np.linalg.inv(world_poses[group.frame])
    components: list[tuple[float, np.ndarray, np.ndarray]] = []
    for link in group.source_links:
        link_inertia = robot_geometry.link_inertias[link]
        link_to_anchor = inverse_anchor @ world_poses[link]
        rotation = link_to_anchor[:3, :3]
        center = (
            rotation @ link_inertia.center
            + link_to_anchor[:3, 3]
        )
        matrix = rotation @ link_inertia.matrix @ rotation.T
        components.append((link_inertia.mass, center, matrix))

    total_mass = sum(mass for mass, _center, _matrix in components)
    center = sum(
        mass * component_center
        for mass, component_center, _matrix in components
    ) / total_mass
    matrix = np.zeros((3, 3), dtype=float)
    for mass, component_center, component_matrix in components:
        offset = component_center - center
        matrix += component_matrix + mass * (
            float(offset @ offset) * np.eye(3)
            - np.outer(offset, offset)
        )
    return _LinkInertia(
        mass=float(total_mass),
        center=np.asarray(center),
        matrix=0.5 * (matrix + matrix.T),
    )


def _equivalent_ellipsoid_axes(
    inertia: _LinkInertia,
) -> tuple[np.ndarray, np.ndarray]:
    moments, rotation = np.linalg.eigh(inertia.matrix)
    squared_radii = np.empty(3, dtype=float)
    for axis_index in range(3):
        other_indices = [
            index for index in range(3) if index != axis_index
        ]
        squared_radii[axis_index] = (
            2.5
            / inertia.mass
            * (
                moments[other_indices[0]]
                + moments[other_indices[1]]
                - moments[axis_index]
            )
        )
    if (
        not np.all(np.isfinite(squared_radii))
        or squared_radii.min() <= 1e-14
    ):
        raise ValueError("Inertia does not define a physical solid ellipsoid")
    radii = np.sqrt(squared_radii)
    order = np.argsort(radii)[::-1]
    radii = radii[order]
    rotation = rotation[:, order]
    if np.linalg.det(rotation) < 0.0:
        rotation[:, -1] *= -1.0
    return radii, rotation


def _inertial_ellipsoid_proxy(
    robot_geometry: _RobotGeometry,
    group: _GroupDefinition,
    world_poses: Mapping[str, np.ndarray],
    mesh: trimesh.Trimesh,
    config: GeneratorConfig,
) -> EllipsoidProxy:
    inertia = _group_inertia(
        robot_geometry,
        group,
        world_poses,
        mesh,
    )
    try:
        inertial_radii, rotation = _equivalent_ellipsoid_axes(inertia)
    except ValueError:
        inertia = _mesh_inertia(mesh)
        inertial_radii, rotation = _equivalent_ellipsoid_axes(inertia)

    local_vertices = (
        np.asarray(mesh.vertices) - inertia.center[None, :]
    ) @ rotation
    radii = _minimum_enlarged_ellipsoid_radii(
        local_vertices,
        inertial_radii,
    )
    radii += config.margin
    axis_ratio = float(inertial_radii[0] / inertial_radii[1])
    ellipsoid = EllipsoidProxy(
        center=_vector3(inertia.center),
        radii=_vector3(radii),
        rotation=_matrix3(rotation),
        inertial_radii=_vector3(inertial_radii),
        inertia_matrix=_matrix3(inertia.matrix),
        mass=float(inertia.mass),
        axis_ratio=axis_ratio,
        elongated=axis_ratio >= config.elongation_threshold,
    )
    _validate_ellipsoid_coverage(mesh, ellipsoid)
    return ellipsoid


def _minimum_enlarged_ellipsoid_radii(
    local_vertices: np.ndarray,
    inertial_radii: np.ndarray,
) -> np.ndarray:
    """Minimize volume while only enlarging inertia-aligned semiaxes."""
    normalized_squared = (
        np.asarray(local_vertices, dtype=float)
        / np.asarray(inertial_radii, dtype=float)[None, :]
    ) ** 2
    uniform_factor = max(
        1.0,
        float(normalized_squared.sum(axis=1).max()),
    )
    initial = np.full(3, 1.0 / uniform_factor)

    # y_i = (inertial_radius_i / enlarged_radius_i)^2. Maximizing
    # product(y_i) minimizes ellipsoid volume. The coverage constraints
    # are linear in y, and 0 < y_i <= 1 only permits enlargement.
    log_lower_bound = math.log(1e-12)
    optimization = minimize(
        fun=lambda log_values: -float(log_values.sum()),
        x0=np.log(initial),
        jac=lambda _log_values: -np.ones(3),
        bounds=((log_lower_bound, 0.0),) * 3,
        constraints=(
            {
                "type": "ineq",
                "fun": lambda log_values: (
                    1.0 - normalized_squared @ np.exp(log_values)
                ),
                "jac": lambda log_values: (
                    -normalized_squared * np.exp(log_values)[None, :]
                ),
            },
        ),
        method="SLSQP",
        options={"ftol": 1e-12, "maxiter": 300},
    )
    values = (
        np.exp(np.asarray(optimization.x, dtype=float))
        if optimization.success
        else initial
    )
    if (
        values.shape != (3,)
        or not np.all(np.isfinite(values))
        or np.any(values <= 0.0)
    ):
        values = initial
    values = np.minimum(values, 1.0)
    radii = np.asarray(inertial_radii, dtype=float) / np.sqrt(values)

    # Absorb numerical optimizer tolerance conservatively.
    required_scale = float(
        np.linalg.norm(
            np.asarray(local_vertices) / radii[None, :],
            axis=1,
        ).max()
    )
    return radii * max(1.0, required_scale)


def _load_multisphere_module() -> Any:
    """Import multisphere core despite its 1.1 optional-PyVista import bug."""
    if importlib.util.find_spec("pyvista") is not None:
        return importlib.import_module("multisphere")

    existing_stub = sys.modules.get("pyvista")
    if existing_stub is None:
        sys.modules["pyvista"] = types.ModuleType("pyvista")
    try:
        return importlib.import_module("multisphere")
    finally:
        if existing_stub is None:
            sys.modules.pop("pyvista", None)


def _face_clusters(mesh: trimesh.Trimesh, count: int) -> tuple[np.ndarray, ...]:
    face_count = len(mesh.faces)
    target = min(max(1, count), face_count)
    centroids = np.asarray(mesh.triangles_center)
    clusters: list[np.ndarray] = [np.arange(face_count, dtype=int)]

    while len(clusters) < target:
        splittable = [
            (len(indices), index)
            for index, indices in enumerate(clusters)
            if len(indices) > 1
        ]
        if not splittable:
            break
        _, cluster_index = max(splittable)
        indices = clusters.pop(cluster_index)
        points = centroids[indices]
        centered = points - points.mean(axis=0)
        try:
            _u, _s, vh = np.linalg.svd(centered, full_matrices=False)
            direction = vh[0]
        except np.linalg.LinAlgError:
            direction = np.array((1.0, 0.0, 0.0))
        projection = centered @ direction
        order = np.argsort(projection, kind="mergesort")
        midpoint = len(order) // 2
        left = indices[order[:midpoint]]
        right = indices[order[midpoint:]]
        if len(left) == 0 or len(right) == 0:
            clusters.append(indices)
            break
        clusters.extend((left, right))

    clusters.sort(key=lambda indices: int(indices.min()))
    return tuple(clusters)


def _cluster_vertices(mesh: trimesh.Trimesh, face_indices: np.ndarray) -> np.ndarray:
    vertex_indices = np.unique(np.asarray(mesh.faces)[face_indices].reshape(-1))
    return np.asarray(mesh.vertices)[vertex_indices]


def _minimum_sphere(points: np.ndarray) -> tuple[np.ndarray, float]:
    try:
        center, radius = trimesh.nsphere.minimum_nsphere(points)
        center = np.asarray(center, dtype=float)
        radius = float(radius)
    except Exception:
        center = np.mean(points, axis=0)
        radius = float(np.linalg.norm(points - center[None, :], axis=1).max())
    if not np.all(np.isfinite(center)) or not math.isfinite(radius):
        raise ValueError("Minimum-sphere fitting returned non-finite values")
    return center, radius


def _fallback_spheres(
    mesh: trimesh.Trimesh,
    max_spheres: int,
) -> tuple[np.ndarray, np.ndarray]:
    centers: list[np.ndarray] = []
    radii: list[float] = []
    for face_indices in _face_clusters(mesh, max_spheres):
        center, radius = _minimum_sphere(_cluster_vertices(mesh, face_indices))
        centers.append(center)
        radii.append(radius)
    return np.asarray(centers), np.asarray(radii)


def _inflate_spheres_for_faces(
    mesh: trimesh.Trimesh,
    centers: np.ndarray,
    radii: np.ndarray,
    margin: float,
) -> tuple[np.ndarray, np.ndarray]:
    centers = np.asarray(centers, dtype=float).reshape(-1, 3)
    radii = np.asarray(radii, dtype=float).reshape(-1)
    if len(centers) == 0 or len(centers) != len(radii):
        raise ValueError("Sphere generator returned an empty or inconsistent pack")

    triangles = np.asarray(mesh.triangles)
    required = np.empty((len(triangles), len(centers)), dtype=float)
    for sphere_index, center in enumerate(centers):
        distances = np.linalg.norm(triangles - center[None, None, :], axis=2)
        required[:, sphere_index] = distances.max(axis=1)
    assignments = np.argmin(required, axis=1)
    used: list[int] = []
    inflated: list[float] = []
    for sphere_index in range(len(centers)):
        assigned = assignments == sphere_index
        if not np.any(assigned):
            continue
        required_radius = float(required[assigned, sphere_index].max()) + margin
        used.append(sphere_index)
        inflated.append(max(float(radii[sphere_index]) + margin, required_radius))
    if not used:
        raise ValueError("No collision triangles were assigned to generated spheres")
    return centers[np.asarray(used)], np.asarray(inflated)


def _sphere_candidate_cost(
    candidate: tuple[np.ndarray, np.ndarray],
) -> float:
    _centers, radii = candidate
    return float(np.sum(np.asarray(radii) ** 3))


def _select_compact_sphere_candidate(
    candidates: Sequence[tuple[np.ndarray, np.ndarray]],
    precision: float,
) -> tuple[np.ndarray, np.ndarray]:
    if not candidates:
        raise ValueError("No sphere candidates were generated")
    best_cost = min(_sphere_candidate_cost(candidate) for candidate in candidates)
    cost_limit = best_cost / precision
    acceptable = [
        candidate
        for candidate in candidates
        if _sphere_candidate_cost(candidate) <= cost_limit + 1e-15
    ]
    return min(
        acceptable,
        key=lambda candidate: (
            len(candidate[0]),
            _sphere_candidate_cost(candidate),
        ),
    )


def _skeleton_sphere_candidates(
    mesh: trimesh.Trimesh,
    ellipsoid: EllipsoidProxy,
    config: GeneratorConfig,
) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    center = np.asarray(ellipsoid.center)
    major_axis = np.asarray(ellipsoid.rotation)[:, 0]
    projections = (np.asarray(mesh.vertices) - center[None, :]) @ major_axis
    minimum = float(projections.min())
    maximum = float(projections.max())
    extent = maximum - minimum
    candidates: list[tuple[np.ndarray, np.ndarray]] = []
    for count in range(1, config.max_spheres + 1):
        if extent <= 1e-12:
            offsets = np.zeros(count)
        else:
            half_bin = 0.5 * extent / count
            offsets = np.linspace(
                minimum + half_bin,
                maximum - half_bin,
                count,
            )
        centers = (
            center[None, :]
            + offsets[:, None] * major_axis[None, :]
        )
        candidates.append(
            _inflate_skeleton_spheres_for_faces(
                mesh,
                centers,
                center,
                major_axis,
                offsets,
                config.margin,
            )
        )
    return tuple(candidates)


def _clip_polygon_at_axis_offset(
    polygon: np.ndarray,
    *,
    center: np.ndarray,
    axis: np.ndarray,
    boundary: float,
    keep_greater: bool,
) -> np.ndarray:
    if len(polygon) == 0:
        return polygon
    coordinates = (polygon - center[None, :]) @ axis
    inside = (
        coordinates >= boundary
        if keep_greater
        else coordinates <= boundary
    )
    clipped: list[np.ndarray] = []
    for index in range(len(polygon)):
        next_index = (index + 1) % len(polygon)
        point = polygon[index]
        next_point = polygon[next_index]
        point_inside = bool(inside[index])
        next_inside = bool(inside[next_index])
        if point_inside:
            clipped.append(point)
        if point_inside == next_inside:
            continue
        denominator = coordinates[next_index] - coordinates[index]
        if abs(float(denominator)) <= 1e-15:
            continue
        fraction = (boundary - coordinates[index]) / denominator
        clipped.append(point + fraction * (next_point - point))
    if not clipped:
        return np.empty((0, 3), dtype=float)
    return np.asarray(clipped)


def _inflate_skeleton_spheres_for_faces(
    mesh: trimesh.Trimesh,
    centers: np.ndarray,
    skeleton_center: np.ndarray,
    skeleton_axis: np.ndarray,
    offsets: np.ndarray,
    margin: float,
) -> tuple[np.ndarray, np.ndarray]:
    boundaries = 0.5 * (offsets[:-1] + offsets[1:])
    required_radii = np.zeros(len(centers), dtype=float)
    used = np.zeros(len(centers), dtype=bool)
    for triangle in np.asarray(mesh.triangles):
        for sphere_index, sphere_center in enumerate(centers):
            polygon = triangle
            if sphere_index > 0:
                polygon = _clip_polygon_at_axis_offset(
                    polygon,
                    center=skeleton_center,
                    axis=skeleton_axis,
                    boundary=float(boundaries[sphere_index - 1]),
                    keep_greater=True,
                )
            if sphere_index < len(centers) - 1:
                polygon = _clip_polygon_at_axis_offset(
                    polygon,
                    center=skeleton_center,
                    axis=skeleton_axis,
                    boundary=float(boundaries[sphere_index]),
                    keep_greater=False,
                )
            if len(polygon) == 0:
                continue
            used[sphere_index] = True
            required_radii[sphere_index] = max(
                required_radii[sphere_index],
                float(
                    np.linalg.norm(
                        polygon - sphere_center[None, :],
                        axis=1,
                    ).max()
                ),
            )
    if not np.any(used):
        raise ValueError("No faces intersect the inertial skeleton")
    return centers[used], required_radii[used] + margin


def _generate_spheres(
    mesh: trimesh.Trimesh,
    ellipsoid: EllipsoidProxy,
    config: GeneratorConfig,
) -> tuple[SphereProxy, ...]:
    if ellipsoid.elongated:
        candidates = _skeleton_sphere_candidates(
            mesh,
            ellipsoid,
            config,
        )
        if config.max_spheres > 1:
            candidates = candidates[1:]
        centers, radii = _select_compact_sphere_candidate(
            candidates,
            config.precision,
        )
        spheres = tuple(
            SphereProxy(center=_vector3(center), radius=float(radius))
            for center, radius in zip(centers, radii)
        )
        _validate_sphere_coverage(mesh, spheres)
        return spheres

    multisphere_candidate: tuple[np.ndarray, np.ndarray] | None = None
    try:
        multisphere = _load_multisphere_module()
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            sphere_pack = multisphere.multisphere_from_mesh(
                mesh=mesh,
                div=config.voxel_divisions,
                padding=config.voxel_padding,
                min_radius_vox=config.min_radius_vox,
                precision=config.precision,
                min_center_distance_vox=config.min_center_distance_vox,
                max_spheres=config.max_spheres,
                confine_mesh=False,
            )
        centers = np.asarray(sphere_pack.centers, dtype=float)
        radii = np.asarray(sphere_pack.radii, dtype=float)
        if len(centers) == 0:
            raise ValueError("multisphere returned no spheres")
        multisphere_candidate = _inflate_spheres_for_faces(
            mesh, centers, radii, config.margin
        )
    except Exception as error:
        print(
            f"warning: multisphere fitting failed ({error}); "
            "using enclosing-sphere fallback",
            file=sys.stderr,
        )

    candidates = []
    for count in range(1, config.max_spheres + 1):
        fallback_centers, fallback_radii = _fallback_spheres(mesh, count)
        candidates.append(
            _inflate_spheres_for_faces(
                mesh,
                fallback_centers,
                fallback_radii,
                config.margin,
            )
        )
    if multisphere_candidate is not None:
        candidates.append(multisphere_candidate)
    centers, radii = _select_compact_sphere_candidate(
        candidates,
        config.precision,
    )
    spheres = tuple(
        SphereProxy(center=_vector3(center), radius=float(radius))
        for center, radius in zip(centers, radii)
    )
    if len(spheres) > config.max_spheres:
        raise AssertionError("Sphere generator exceeded its configured budget")
    _validate_sphere_coverage(mesh, spheres)
    return spheres


def _point_segment_distances(
    points: np.ndarray,
    point_a: np.ndarray,
    point_b: np.ndarray,
) -> np.ndarray:
    segment = point_b - point_a
    length_squared = float(segment @ segment)
    if length_squared <= 1e-24:
        return np.linalg.norm(points - point_a[None, :], axis=1)
    parameter = ((points - point_a[None, :]) @ segment) / length_squared
    parameter = np.clip(parameter, 0.0, 1.0)
    witnesses = point_a[None, :] + parameter[:, None] * segment[None, :]
    return np.linalg.norm(points - witnesses, axis=1)


def _fit_capsule(points: np.ndarray, margin: float) -> CapsuleProxy:
    try:
        cylinder = trimesh.bounds.minimum_cylinder(points)
        transform = np.asarray(cylinder["transform"], dtype=float)
        axis = transform[:3, 2]
        axis /= np.linalg.norm(axis)
        center = transform[:3, 3]
        half_height = 0.5 * float(cylinder["height"])
        point_a = center - axis * half_height
        point_b = center + axis * half_height
    except Exception:
        center = points.mean(axis=0)
        centered = points - center[None, :]
        try:
            _u, _s, vh = np.linalg.svd(centered, full_matrices=False)
            axis = vh[0]
        except np.linalg.LinAlgError:
            axis = np.array((0.0, 0.0, 1.0))
        projection = centered @ axis
        point_a = center + axis * float(projection.min())
        point_b = center + axis * float(projection.max())
    radius = float(_point_segment_distances(points, point_a, point_b).max()) + margin
    return CapsuleProxy(
        point_a=_vector3(point_a),
        point_b=_vector3(point_b),
        radius=radius,
    )


def _generate_capsules(
    mesh: trimesh.Trimesh,
    ellipsoid: EllipsoidProxy,
    config: GeneratorConfig,
) -> tuple[CapsuleProxy, ...]:
    if ellipsoid.elongated:
        capsules = (
            _fit_axis_aligned_capsule(mesh, ellipsoid, config.margin),
        )
        _validate_capsule_coverage(mesh, capsules)
        return capsules

    candidates: list[tuple[CapsuleProxy, ...]] = []
    for count in range(1, config.max_capsules + 1):
        candidates.append(
            tuple(
                _fit_capsule(
                    _cluster_vertices(mesh, face_indices),
                    config.margin,
                )
                for face_indices in _face_clusters(mesh, count)
            )
        )
    best_cost = min(_capsule_candidate_cost(candidate) for candidate in candidates)
    cost_limit = best_cost / config.precision
    capsules = min(
        (
            candidate
            for candidate in candidates
            if _capsule_candidate_cost(candidate) <= cost_limit + 1e-15
        ),
        key=lambda candidate: (
            len(candidate),
            _capsule_candidate_cost(candidate),
        ),
    )
    if len(capsules) > config.max_capsules:
        raise AssertionError("Capsule generator exceeded its configured budget")
    _validate_capsule_coverage(mesh, capsules)
    return capsules


def _capsule_volume(capsule: CapsuleProxy) -> float:
    length = float(
        np.linalg.norm(
            np.asarray(capsule.point_b) - np.asarray(capsule.point_a)
        )
    )
    radius = capsule.radius
    return (
        math.pi * radius * radius * length
        + (4.0 / 3.0) * math.pi * radius**3
    )


def _capsule_candidate_cost(
    capsules: Sequence[CapsuleProxy],
) -> float:
    return float(sum(_capsule_volume(capsule) for capsule in capsules))


def _fit_axis_aligned_capsule(
    mesh: trimesh.Trimesh,
    ellipsoid: EllipsoidProxy,
    margin: float,
) -> CapsuleProxy:
    """Fit a compact capsule constrained to the inertial major axis."""
    points = np.asarray(mesh.vertices, dtype=float)
    center = np.asarray(ellipsoid.center)
    axis = np.asarray(ellipsoid.rotation)[:, 0]
    relative = points - center[None, :]
    projections = relative @ axis
    radial_squared = np.maximum(
        0.0,
        np.einsum("ij,ij->i", relative, relative) - projections**2,
    )

    # Optima occur near changes in the active witness point. Sampling both
    # uniform offsets and vertex quantiles is deterministic and stays bounded
    # for high-resolution meshes.
    endpoint_samples = np.unique(
        np.concatenate(
            (
                np.linspace(
                    float(projections.min()),
                    float(projections.max()),
                    25,
                ),
                np.quantile(projections, np.linspace(0.0, 1.0, 25)),
                np.array((0.0,)),
            )
        )
    )
    best: CapsuleProxy | None = None
    best_volume = math.inf
    for point_a_projection in endpoint_samples:
        for point_b_projection in endpoint_samples:
            if point_b_projection < point_a_projection:
                continue
            axial_offset = np.maximum(
                point_a_projection - projections,
                projections - point_b_projection,
            )
            axial_offset = np.maximum(axial_offset, 0.0)
            radius = float(
                np.sqrt(np.max(radial_squared + axial_offset**2))
            ) + margin
            candidate = CapsuleProxy(
                point_a=_vector3(
                    center + axis * point_a_projection
                ),
                point_b=_vector3(
                    center + axis * point_b_projection
                ),
                radius=radius,
            )
            volume = _capsule_volume(candidate)
            if volume < best_volume:
                best = candidate
                best_volume = volume
    if best is None:
        raise ValueError("Could not fit an inertia-aligned capsule")
    return best


def _validate_sphere_coverage(
    mesh: trimesh.Trimesh,
    spheres: Sequence[SphereProxy],
    tolerance: float = 1e-9,
) -> None:
    if not spheres:
        raise ValueError("No spheres were generated")
    uncovered = sum(
        not _triangle_covered_by_spheres(
            triangle,
            spheres,
            tolerance=tolerance,
            depth=0,
        )
        for triangle in np.asarray(mesh.triangles)
    )
    if uncovered:
        raise ValueError(
            f"Sphere proxy leaves {uncovered} triangles uncovered"
        )


def _triangle_covered_by_spheres(
    triangle: np.ndarray,
    spheres: Sequence[SphereProxy],
    *,
    tolerance: float,
    depth: int,
) -> bool:
    for sphere in spheres:
        distances = np.linalg.norm(
            triangle - np.asarray(sphere.center)[None, :],
            axis=1,
        )
        if float(distances.max()) <= sphere.radius + tolerance:
            return True
    if depth >= 24:
        return False

    edge_pairs = ((0, 1), (1, 2), (2, 0))
    first, second = max(
        edge_pairs,
        key=lambda pair: float(
            np.linalg.norm(triangle[pair[1]] - triangle[pair[0]])
        ),
    )
    third = 3 - first - second
    midpoint = 0.5 * (triangle[first] + triangle[second])
    first_half = np.asarray(
        (triangle[first], midpoint, triangle[third])
    )
    second_half = np.asarray(
        (midpoint, triangle[second], triangle[third])
    )
    return _triangle_covered_by_spheres(
        first_half,
        spheres,
        tolerance=tolerance,
        depth=depth + 1,
    ) and _triangle_covered_by_spheres(
        second_half,
        spheres,
        tolerance=tolerance,
        depth=depth + 1,
    )


def _validate_capsule_coverage(
    mesh: trimesh.Trimesh,
    capsules: Sequence[CapsuleProxy],
    tolerance: float = 1e-9,
) -> None:
    if not capsules:
        raise ValueError("No capsules were generated")
    triangles = np.asarray(mesh.triangles)
    covered = np.zeros(len(triangles), dtype=bool)
    flat = triangles.reshape(-1, 3)
    for capsule in capsules:
        distances = _point_segment_distances(
            flat,
            np.asarray(capsule.point_a),
            np.asarray(capsule.point_b),
        ).reshape(-1, 3)
        covered |= distances.max(axis=1) <= capsule.radius + tolerance
    if not np.all(covered):
        raise ValueError(
            f"Capsule proxy leaves {int((~covered).sum())} triangles uncovered"
        )


def _validate_ellipsoid_coverage(
    mesh: trimesh.Trimesh,
    ellipsoid: EllipsoidProxy,
    tolerance: float = 1e-9,
) -> None:
    vertices = np.asarray(mesh.vertices, dtype=float)
    local_vertices = (
        vertices - np.asarray(ellipsoid.center)[None, :]
    ) @ np.asarray(ellipsoid.rotation)
    normalized_radius = np.linalg.norm(
        local_vertices / np.asarray(ellipsoid.radii)[None, :],
        axis=1,
    )
    uncovered = normalized_radius > 1.0 + tolerance
    if np.any(uncovered):
        raise ValueError(
            f"Ellipsoid proxy leaves {int(uncovered.sum())} vertices uncovered"
        )


def _mode_kinds(mode: str) -> frozenset[str]:
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


def generate_collision_proxies(
    robot: str,
    *,
    urdf_path: str | Path | None = None,
    config: GeneratorConfig | None = None,
    mode: str = "all",
    group_overrides: Mapping[str, Sequence[str]] | None = None,
    joint_position_overrides: Mapping[str, float] | None = None,
) -> CollisionProxyResult:
    """Generate conservative primitives for all collision-bearing robot links."""
    requested_kinds = _mode_kinds(mode)
    config = config or GeneratorConfig()
    resolved_urdf = resolve_robot_urdf(robot, urdf_path)
    geometry = _read_robot_geometry(resolved_urdf)
    if not geometry.link_meshes:
        raise ValueError(f"URDF contains no collision geometry: {resolved_urdf}")
    groups = _make_group_definitions(geometry, group_overrides)
    positions = _reference_joint_positions(
        geometry, groups, joint_position_overrides
    )
    poses = _world_link_poses(geometry.link_order, geometry.joints, positions)
    group_meshes = _combine_group_meshes(geometry, groups, poses)

    proxies: list[LinkProxy] = []
    for index, group in enumerate(groups, start=1):
        print(
            f"[{index}/{len(groups)}] fitting collision group {group.name} "
            f"from {', '.join(group.source_links)}"
        )
        mesh = group_meshes[group.name]
        ellipsoid = _inertial_ellipsoid_proxy(
            geometry,
            group,
            poses,
            mesh,
            config,
        )
        spheres = (
            _generate_spheres(mesh, ellipsoid, config)
            if "spheres" in requested_kinds
            else ()
        )
        capsules = (
            _generate_capsules(mesh, ellipsoid, config)
            if "capsules" in requested_kinds
            else ()
        )
        proxies.append(
            LinkProxy(
                name=group.name,
                frame=group.frame,
                source_links=group.source_links,
                spheres=spheres,
                capsules=capsules,
                ellipsoids=(ellipsoid,),
            )
        )

    return CollisionProxyResult(
        robot=robot,
        source_urdf=resolved_urdf,
        config=config,
        joint_positions=dict(positions),
        groups=tuple(proxies),
    )


def _repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(BASE_DIR).as_posix()
    except ValueError:
        return str(path.resolve())


def _config_dict(config: GeneratorConfig) -> dict[str, Any]:
    return {
        "max_spheres_per_group": config.max_spheres,
        "max_capsules_per_group": config.max_capsules,
        "margin": config.margin,
        "voxel_divisions": config.voxel_divisions,
        "precision": config.precision,
        "elongation_threshold": config.elongation_threshold,
        "voxel_padding": config.voxel_padding,
        "min_radius_vox": config.min_radius_vox,
        "min_center_distance_vox": config.min_center_distance_vox,
    }


def _yaml_document(
    result: CollisionProxyResult,
    kind: str,
) -> dict[str, Any]:
    if kind not in PROXY_KINDS:
        raise ValueError(f"Unknown proxy kind: {kind}")
    group_documents: list[dict[str, Any]] = []
    for group in result.groups:
        if kind == "spheres":
            primitives = [
                {"center": list(sphere.center), "radius": sphere.radius}
                for sphere in group.spheres
            ]
        elif kind == "capsules":
            primitives = [
                {
                    "point_a": list(capsule.point_a),
                    "point_b": list(capsule.point_b),
                    "radius": capsule.radius,
                }
                for capsule in group.capsules
            ]
        else:
            primitives = [
                {
                    "center": list(ellipsoid.center),
                    "radii": list(ellipsoid.radii),
                    "rotation": [
                        list(row) for row in ellipsoid.rotation
                    ],
                    "inertial_radii": list(ellipsoid.inertial_radii),
                    "inertia_matrix": [
                        list(row) for row in ellipsoid.inertia_matrix
                    ],
                    "mass": ellipsoid.mass,
                    "axis_ratio": ellipsoid.axis_ratio,
                    "elongated": ellipsoid.elongated,
                }
                for ellipsoid in group.ellipsoids
            ]
        group_documents.append(
            {
                "name": group.name,
                "frame": group.frame,
                "source_links": list(group.source_links),
                "primitives": primitives,
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": kind,
        "robot": result.robot,
        "source_urdf": _repo_relative(result.source_urdf),
        "units": "m",
        "parameters": _config_dict(result.config),
        "joint_positions": {
            name: float(value) for name, value in result.joint_positions.items()
        },
        "groups": group_documents,
    }


def _atomic_write_yaml(path: Path, document: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(
            dict(document),
            stream,
            sort_keys=False,
            default_flow_style=False,
            width=100,
        )
    temporary.replace(path)


def save_collision_proxies(
    result: CollisionProxyResult,
    output_dir: str | Path | None = None,
    *,
    mode: str = "all",
) -> dict[str, Path]:
    """Save the canonical YAML files selected by ``mode``."""
    requested_kinds = _mode_kinds(mode)
    destination = (
        default_output_dir(result.robot)
        if output_dir is None
        else Path(output_dir).expanduser()
    )
    if not destination.is_absolute():
        destination = BASE_DIR / destination
    destination = destination.resolve()

    paths: dict[str, Path] = {}
    filenames = {
        "spheres": SPHERE_FILENAME,
        "capsules": CAPSULE_FILENAME,
        "ellipsoids": ELLIPSOID_FILENAME,
    }
    for kind in PROXY_KINDS:
        if kind not in requested_kinds:
            continue
        path = destination / filenames[kind]
        _atomic_write_yaml(path, _yaml_document(result, kind))
        paths[kind] = path
    return paths


def generate_configured_collision_proxies(
    config: CollisionProxyConfig,
) -> dict[str, Path]:
    """Generate and save the proxies described by a high-level config."""
    result = generate_collision_proxies(
        config.robot,
        urdf_path=config.urdf,
        config=config.generator_config,
        mode=config.mode,
        group_overrides=config.group_overrides,
        joint_position_overrides=config.joint_position_overrides,
    )
    return save_collision_proxies(
        result,
        config.output_dir,
        mode=config.mode,
    )


def _load_yaml_document(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        document = yaml.safe_load(stream)
    if not isinstance(document, dict):
        raise ValueError(f"Proxy YAML must contain a mapping: {path}")
    if document.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported schema version in {path}: {document.get('schema_version')}"
        )
    if document.get("units") != "m":
        raise ValueError(f"Only meter-valued proxies are supported: {path}")
    return document


def _config_from_document(document: Mapping[str, Any]) -> GeneratorConfig:
    parameters = document.get("parameters", {})
    return GeneratorConfig(
        max_spheres=int(parameters.get("max_spheres_per_group", 8)),
        max_capsules=int(parameters.get("max_capsules_per_group", 3)),
        margin=float(parameters.get("margin", 0.002)),
        voxel_divisions=int(parameters.get("voxel_divisions", 100)),
        precision=float(parameters.get("precision", 0.95)),
        elongation_threshold=float(
            parameters.get("elongation_threshold", 2.0)
        ),
        voxel_padding=int(parameters.get("voxel_padding", 2)),
        min_radius_vox=int(parameters.get("min_radius_vox", 2)),
        min_center_distance_vox=int(
            parameters.get("min_center_distance_vox", 2)
        ),
    )


def load_collision_proxies(
    sphere_path: str | Path | None = None,
    capsule_path: str | Path | None = None,
    ellipsoid_path: str | Path | None = None,
) -> CollisionProxyResult:
    """Load and merge canonical collision-proxy YAML files."""
    if (
        sphere_path is None
        and capsule_path is None
        and ellipsoid_path is None
    ):
        raise ValueError("At least one proxy YAML path is required")
    documents: dict[str, dict[str, Any]] = {}
    for kind, value in (
        ("spheres", sphere_path),
        ("capsules", capsule_path),
        ("ellipsoids", ellipsoid_path),
    ):
        if value is None:
            continue
        path = Path(value).expanduser().resolve()
        document = _load_yaml_document(path)
        if document.get("kind") != kind:
            raise ValueError(f"{path} contains {document.get('kind')!r}, expected {kind!r}")
        documents[kind] = document

    first = next(iter(documents.values()))
    for document in documents.values():
        for key in ("robot", "source_urdf", "joint_positions"):
            if document.get(key) != first.get(key):
                raise ValueError(
                    f"Collision-proxy YAML files disagree on {key}"
                )

    group_data: dict[str, dict[str, Any]] = {}
    group_order: list[str] = []
    for kind, document in documents.items():
        for item in document.get("groups", []):
            name = str(item["name"])
            if name not in group_data:
                group_order.append(name)
                group_data[name] = {
                    "frame": str(item["frame"]),
                    "source_links": tuple(str(link) for link in item["source_links"]),
                    "spheres": (),
                    "capsules": (),
                    "ellipsoids": (),
                }
            data = group_data[name]
            if (
                data["frame"] != str(item["frame"])
                or data["source_links"]
                != tuple(str(link) for link in item["source_links"])
            ):
                raise ValueError(
                    f"Collision-proxy YAML files disagree on group {name}"
                )
            if kind == "spheres":
                data["spheres"] = tuple(
                    SphereProxy(
                        center=_vector3(primitive["center"]),
                        radius=float(primitive["radius"]),
                    )
                    for primitive in item.get("primitives", [])
                )
            elif kind == "capsules":
                data["capsules"] = tuple(
                    CapsuleProxy(
                        point_a=_vector3(primitive["point_a"]),
                        point_b=_vector3(primitive["point_b"]),
                        radius=float(primitive["radius"]),
                    )
                    for primitive in item.get("primitives", [])
                )
            else:
                data["ellipsoids"] = tuple(
                    EllipsoidProxy(
                        center=_vector3(primitive["center"]),
                        radii=_vector3(primitive["radii"]),
                        rotation=_matrix3(primitive["rotation"]),
                        inertial_radii=_vector3(
                            primitive["inertial_radii"]
                        ),
                        inertia_matrix=_matrix3(
                            primitive["inertia_matrix"]
                        ),
                        mass=float(primitive["mass"]),
                        axis_ratio=float(primitive["axis_ratio"]),
                        elongated=bool(primitive["elongated"]),
                    )
                    for primitive in item.get("primitives", [])
                )

    source_urdf = Path(str(first["source_urdf"]))
    if not source_urdf.is_absolute():
        source_urdf = BASE_DIR / source_urdf
    groups = tuple(
        LinkProxy(
            name=name,
            frame=group_data[name]["frame"],
            source_links=group_data[name]["source_links"],
            spheres=group_data[name]["spheres"],
            capsules=group_data[name]["capsules"],
            ellipsoids=group_data[name]["ellipsoids"],
        )
        for name in group_order
    )
    return CollisionProxyResult(
        robot=str(first["robot"]),
        source_urdf=source_urdf.resolve(),
        config=_config_from_document(first),
        joint_positions={
            str(name): float(value)
            for name, value in first.get("joint_positions", {}).items()
        },
        groups=groups,
    )


def _configured_proxy_directory(config: CollisionProxyConfig) -> Path:
    if config.output_dir is None:
        return default_output_dir(config.robot)
    output_dir = config.output_dir
    if not output_dir.is_absolute():
        output_dir = BASE_DIR / output_dir
    return output_dir.resolve()


def load_collision_model(
    config: CollisionProxyConfig,
) -> CollisionProxyResult:
    """Load the collision model selected by config."""
    proxy_directory = _configured_proxy_directory(config)
    requested_kinds = _mode_kinds(config.mode)
    sphere_path = (
        proxy_directory / SPHERE_FILENAME
        if "spheres" in requested_kinds
        else None
    )
    capsule_path = (
        proxy_directory / CAPSULE_FILENAME
        if "capsules" in requested_kinds
        else None
    )
    ellipsoid_path = (
        proxy_directory / ELLIPSOID_FILENAME
        if "ellipsoids" in requested_kinds
        else None
    )
    return load_collision_proxies(
        sphere_path=sphere_path,
        capsule_path=capsule_path,
        ellipsoid_path=ellipsoid_path,
    )


def _rotation_from_z(direction: np.ndarray) -> np.ndarray:
    direction = np.asarray(direction, dtype=float)
    norm = np.linalg.norm(direction)
    if norm <= 1e-12:
        return np.eye(3)
    direction /= norm
    z_axis = np.array((0.0, 0.0, 1.0))
    dot = float(np.clip(z_axis @ direction, -1.0, 1.0))
    if dot > 1.0 - 1e-12:
        return np.eye(3)
    if dot < -1.0 + 1e-12:
        return Rotation.from_rotvec(np.array((math.pi, 0.0, 0.0))).as_matrix()
    axis = np.cross(z_axis, direction)
    axis /= np.linalg.norm(axis)
    return Rotation.from_rotvec(axis * math.acos(dot)).as_matrix()


def _meshcat_transform(matrix: np.ndarray) -> Any:
    from pydrake.math import RigidTransform

    return RigidTransform(np.asarray(matrix, dtype=float))


def _body_meshcat_path(group_name: str) -> str:
    return f"{MESHCAT_ROOT_PATH}/bodies/{group_name}"


def _source_meshcat_path(
    group_name: str,
    source_link: str,
    mesh_index: int,
) -> str:
    return (
        f"{_body_meshcat_path(group_name)}/source/"
        f"{source_link}_{mesh_index:02d}"
    )


def _sphere_meshcat_path(group_name: str, sphere_index: int) -> str:
    return (
        f"{_body_meshcat_path(group_name)}/spheres/"
        f"sphere_{sphere_index:02d}"
    )


def _capsule_meshcat_path(group_name: str, capsule_index: int) -> str:
    return (
        f"{_body_meshcat_path(group_name)}/capsules/"
        f"capsule_{capsule_index:02d}"
    )


def _ellipsoid_meshcat_path(
    group_name: str,
    ellipsoid_index: int,
) -> str:
    return (
        f"{_body_meshcat_path(group_name)}/ellipsoids/"
        f"ellipsoid_{ellipsoid_index:02d}"
    )


def _capsule_local_pose(capsule: CapsuleProxy) -> np.ndarray:
    point_a = np.asarray(capsule.point_a)
    point_b = np.asarray(capsule.point_b)
    local_pose = np.eye(4)
    local_pose[:3, :3] = _rotation_from_z(point_b - point_a)
    local_pose[:3, 3] = 0.5 * (point_a + point_b)
    return local_pose


def _ellipsoid_local_pose(ellipsoid: EllipsoidProxy) -> np.ndarray:
    local_pose = np.eye(4)
    local_pose[:3, :3] = np.asarray(ellipsoid.rotation)
    local_pose[:3, 3] = np.asarray(ellipsoid.center)
    return local_pose


def _update_meshcat_transforms(
    meshcat: Any,
    result: CollisionProxyResult,
    robot_geometry: _RobotGeometry,
    joint_positions: Mapping[str, float],
) -> None:
    poses = _world_link_poses(
        robot_geometry.link_order,
        robot_geometry.joints,
        joint_positions,
    )
    for group in result.groups:
        for source_link in group.source_links:
            for mesh_index, _source_mesh in enumerate(
                robot_geometry.link_meshes[source_link]
            ):
                meshcat.SetTransform(
                    _source_meshcat_path(
                        group.name,
                        source_link,
                        mesh_index,
                    ),
                    _meshcat_transform(poses[source_link]),
                )

        anchor_pose = poses[group.frame]
        for sphere_index, sphere in enumerate(group.spheres):
            transform = anchor_pose.copy()
            transform[:3, 3] = (
                anchor_pose[:3, :3] @ np.asarray(sphere.center)
                + anchor_pose[:3, 3]
            )
            meshcat.SetTransform(
                _sphere_meshcat_path(group.name, sphere_index),
                _meshcat_transform(transform),
            )

        for capsule_index, capsule in enumerate(group.capsules):
            meshcat.SetTransform(
                _capsule_meshcat_path(group.name, capsule_index),
                _meshcat_transform(
                    anchor_pose @ _capsule_local_pose(capsule)
                ),
            )

        for ellipsoid_index, ellipsoid in enumerate(group.ellipsoids):
            meshcat.SetTransform(
                _ellipsoid_meshcat_path(
                    group.name,
                    ellipsoid_index,
                ),
                _meshcat_transform(
                    anchor_pose @ _ellipsoid_local_pose(ellipsoid)
                ),
            )


def _set_proxy_view(
    meshcat: Any,
    result: CollisionProxyResult,
    view: str,
) -> None:
    visible_kinds = _mode_kinds(view)
    for group in result.groups:
        body_path = _body_meshcat_path(group.name)
        for kind in PROXY_KINDS:
            meshcat.SetProperty(
                f"{body_path}/{kind}",
                "visible",
                kind in visible_kinds,
            )


def _publish_meshcat_scene(
    meshcat: Any,
    result: CollisionProxyResult,
    *,
    initial_view: str,
    proxy_opacity: float,
) -> _RobotGeometry:
    from pydrake.geometry import Capsule, Ellipsoid, Rgba, Sphere

    robot_geometry = _read_robot_geometry(result.source_urdf)
    gray = Rgba(0.62, 0.65, 0.70, 1.0)
    blue = Rgba(0.05, 0.45, 1.0, proxy_opacity)
    orange = Rgba(1.0, 0.38, 0.03, proxy_opacity)
    green = Rgba(0.15, 0.75, 0.30, proxy_opacity)
    root_path = MESHCAT_ROOT_PATH
    meshcat.Delete(root_path)

    for group in result.groups:
        body_path = _body_meshcat_path(group.name)
        for source_link in group.source_links:
            for mesh_index, source_mesh in enumerate(
                robot_geometry.link_meshes[source_link]
            ):
                path = _source_meshcat_path(
                    group.name,
                    source_link,
                    mesh_index,
                )
                vertices = np.asfortranarray(
                    np.asarray(source_mesh.vertices, dtype=float).T
                )
                faces = np.asfortranarray(
                    np.asarray(source_mesh.faces, dtype=np.int32).T
                )
                meshcat.SetTriangleMesh(path, vertices, faces, gray)

        for sphere_index, sphere in enumerate(group.spheres):
            path = _sphere_meshcat_path(group.name, sphere_index)
            meshcat.SetObject(path, Sphere(sphere.radius), blue)

        for capsule_index, capsule in enumerate(group.capsules):
            point_a = np.asarray(capsule.point_a)
            point_b = np.asarray(capsule.point_b)
            segment = point_b - point_a
            length = float(np.linalg.norm(segment))
            path = _capsule_meshcat_path(group.name, capsule_index)
            if length <= 1e-12:
                meshcat.SetObject(path, Sphere(capsule.radius), orange)
            else:
                meshcat.SetObject(path, Capsule(capsule.radius, length), orange)

        for ellipsoid_index, ellipsoid in enumerate(group.ellipsoids):
            path = _ellipsoid_meshcat_path(
                group.name,
                ellipsoid_index,
            )
            meshcat.SetObject(
                path,
                Ellipsoid(*ellipsoid.radii),
                green,
            )
        meshcat.SetProperty(body_path, "visible", True)
        meshcat.SetProperty(f"{body_path}/source", "visible", True)

    _update_meshcat_transforms(
        meshcat,
        result,
        robot_geometry,
        result.joint_positions,
    )
    _set_proxy_view(meshcat, result, initial_view)
    _set_proxy_opacity(meshcat, result, proxy_opacity)
    return robot_geometry


def _set_proxy_opacity(
    meshcat: Any,
    result: CollisionProxyResult,
    opacity: float,
) -> None:
    transparent = opacity < 1.0
    for group in result.groups:
        body_path = _body_meshcat_path(group.name)
        for kind in PROXY_KINDS:
            path = f"{body_path}/{kind}"
            meshcat.SetProperty(path, "transparent", transparent)
            meshcat.SetProperty(path, "opacity", opacity)


def _hold_meshcat(
    meshcat: Any,
    result: CollisionProxyResult,
    robot_geometry: _RobotGeometry,
    *,
    proxy_opacity: float,
) -> None:
    buttons = (
        "Show spheres",
        "Show capsules",
        "Show ellipsoids",
        "Show both",
        "Show all",
        "Exit viewer",
    )
    for button in buttons:
        meshcat.AddButton(button)
    clicks = {button: meshcat.GetButtonClicks(button) for button in buttons}
    current_opacity = meshcat.AddSlider(
        PROXY_OPACITY_SLIDER,
        min=0.0,
        max=1.0,
        step=0.05,
        value=proxy_opacity,
    )
    _set_proxy_opacity(meshcat, result, current_opacity)
    joint_specs = _joint_slider_specs(
        robot_geometry,
        result.joint_positions,
    )
    joint_positions = _add_joint_sliders(
        meshcat,
        joint_specs,
        result.joint_positions,
    )
    _update_mimic_joint_positions(
        robot_geometry.joints,
        joint_positions,
    )
    _update_meshcat_transforms(
        meshcat,
        result,
        robot_geometry,
        joint_positions,
    )
    print(
        "Use the joint and opacity sliders in Controls. Expand "
        "collision_proxy > bodies in the scene tree to check or uncheck "
        "individual bodies; press Enter to exit."
    )
    try:
        while True:
            if sys.stdin.isatty():
                readable, _writable, _errors = select.select([sys.stdin], [], [], 0.1)
                if readable:
                    sys.stdin.readline()
                    return
            else:
                time.sleep(0.1)

            current = {
                button: meshcat.GetButtonClicks(button) for button in buttons
            }
            if current["Exit viewer"] > clicks["Exit viewer"]:
                return
            if current["Show spheres"] > clicks["Show spheres"]:
                _set_proxy_view(meshcat, result, "spheres")
            if current["Show capsules"] > clicks["Show capsules"]:
                _set_proxy_view(meshcat, result, "capsules")
            if current["Show ellipsoids"] > clicks["Show ellipsoids"]:
                _set_proxy_view(meshcat, result, "ellipsoids")
            if current["Show both"] > clicks["Show both"]:
                _set_proxy_view(meshcat, result, "both")
            if current["Show all"] > clicks["Show all"]:
                _set_proxy_view(meshcat, result, "all")
            opacity = meshcat.GetSliderValue(PROXY_OPACITY_SLIDER)
            if opacity != current_opacity:
                _set_proxy_opacity(meshcat, result, opacity)
                current_opacity = opacity
            joint_positions, joints_changed = _read_joint_sliders(
                meshcat,
                joint_specs,
                robot_geometry,
                joint_positions,
            )
            if joints_changed:
                _update_meshcat_transforms(
                    meshcat,
                    result,
                    robot_geometry,
                    joint_positions,
                )
            clicks = current
    except KeyboardInterrupt:
        return
    finally:
        for button in buttons:
            meshcat.DeleteButton(button, strict=False)
        meshcat.DeleteSlider(PROXY_OPACITY_SLIDER, strict=False)
        for spec in joint_specs:
            meshcat.DeleteSlider(spec.control_name, strict=False)


def visualize_collision_proxies(
    result: CollisionProxyResult,
    *,
    initial_view: str = "spheres",
    proxy_opacity: float = 0.25,
    hold: bool = True,
    meshcat: Any | None = None,
) -> Any:
    """Publish source collision geometry and saved primitives to Meshcat."""
    if initial_view not in PROXY_MODES:
        raise ValueError(
            "initial_view must be both, all, spheres, capsules, "
            "or ellipsoids"
        )
    if not 0.0 <= proxy_opacity <= 1.0:
        raise ValueError("proxy_opacity must be in [0, 1]")
    available_kinds = {
        kind
        for kind in PROXY_KINDS
        if any(getattr(group, kind) for group in result.groups)
    }
    if not (_mode_kinds(initial_view) & available_kinds):
        if not available_kinds:
            raise ValueError("The collision model contains no proxy primitives")
        initial_view = next(
            kind for kind in PROXY_KINDS if kind in available_kinds
        )

    if meshcat is None:
        from pydrake.geometry import StartMeshcat

        meshcat = StartMeshcat()
    robot_geometry = _publish_meshcat_scene(
        meshcat,
        result,
        initial_view=initial_view,
        proxy_opacity=proxy_opacity,
    )
    if hasattr(meshcat, "web_url"):
        print(f"Meshcat URL: {meshcat.web_url()}")
    if hold:
        _hold_meshcat(
            meshcat,
            result,
            robot_geometry,
            proxy_opacity=proxy_opacity,
        )
    return meshcat


def visualize_collision_model(
    config: CollisionProxyConfig,
    *,
    meshcat: Any | None = None,
) -> Any:
    """Load and visualize the configured source geometry and proxies."""
    collision_model = load_collision_model(config)
    return visualize_collision_proxies(
        collision_model,
        initial_view=config.initial_view,
        proxy_opacity=config.proxy_opacity,
        hold=config.hold,
        meshcat=meshcat,
    )


__all__ = [
    "CAPSULE_FILENAME",
    "DEFAULT_CONFIG_PATH",
    "ELLIPSOID_FILENAME",
    "ROBOT_DESCRIPTION_DIR",
    "SPHERE_FILENAME",
    "CapsuleProxy",
    "CollisionProxyConfig",
    "CollisionProxyResult",
    "EllipsoidProxy",
    "GeneratorConfig",
    "LinkProxy",
    "SphereProxy",
    "default_output_dir",
    "generate_collision_proxies",
    "generate_configured_collision_proxies",
    "load_collision_model",
    "load_collision_proxies",
    "resolve_robot_urdf",
    "save_collision_proxies",
    "visualize_collision_model",
    "visualize_collision_proxies",
]
