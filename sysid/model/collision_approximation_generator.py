"""Generate conservative sphere and capsule collision approximations from URDFs."""

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

from scipy.spatial.transform import Rotation


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROBOT_DESCRIPTION_DIR = PROJECT_ROOT / "robot_description"
SCHEMA_VERSION = 1
SPHERE_FILENAME = "sphere_collisions.yaml"
CAPSULE_FILENAME = "capsule_collisions.yaml"
GRIPPER_LINK_TOKENS = ("hand", "gripper", "finger")

Vector3 = tuple[float, float, float]


def _vector3(value: Iterable[float]) -> Vector3:
    array = np.asarray(tuple(value), dtype=float).reshape(-1)
    if array.size != 3 or not np.all(np.isfinite(array)):
        raise ValueError(f"Expected a finite 3-vector, got {value!r}")
    return (float(array[0]), float(array[1]), float(array[2]))


@dataclass(frozen=True)
class SphereApproximation:
    """A sphere expressed in its collision-group frame."""

    center: Vector3
    radius: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "center", _vector3(self.center))
        if not math.isfinite(self.radius) or self.radius <= 0.0:
            raise ValueError(f"Sphere radius must be finite and positive: {self.radius}")


@dataclass(frozen=True)
class CapsuleApproximation:
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
class LinkApproximation:
    """Collision primitives associated with one anchor-link frame."""

    name: str
    frame: str
    source_links: tuple[str, ...]
    spheres: tuple[SphereApproximation, ...] = ()
    capsules: tuple[CapsuleApproximation, ...] = ()

    def __post_init__(self) -> None:
        if not self.name or not self.frame:
            raise ValueError("Approximation name and frame must be non-empty")
        if not self.source_links:
            raise ValueError(f"Group {self.name!r} has no source links")


@dataclass(frozen=True)
class GeneratorConfig:
    """Numerical settings for collision approximation generation."""

    max_spheres: int = 8
    max_capsules: int = 3
    margin: float = 0.002
    voxel_divisions: int = 100
    precision: float = 0.95
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
        if self.voxel_padding < 0:
            raise ValueError("voxel_padding must be non-negative")
        if self.min_radius_vox < 0:
            raise ValueError("min_radius_vox must be non-negative")
        if self.min_center_distance_vox <= 0:
            raise ValueError("min_center_distance_vox must be positive")


@dataclass(frozen=True)
class CollisionApproximationResult:
    """Complete approximation metadata and per-link primitives."""

    robot: str
    source_urdf: Path
    config: GeneratorConfig
    joint_positions: Mapping[str, float]
    groups: tuple[LinkApproximation, ...]

    def __post_init__(self) -> None:
        if not self.robot:
            raise ValueError("robot must be non-empty")
        if not self.groups:
            raise ValueError("At least one collision group is required")


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
    group_definitions: tuple[_GroupDefinition, ...] = ()
    joint_positions: dict[str, float] = field(default_factory=dict)
    world_poses: dict[str, np.ndarray] = field(default_factory=dict)
    group_meshes: dict[str, trimesh.Trimesh] = field(default_factory=dict)


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
            candidate = PROJECT_ROOT / candidate
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
                    PROJECT_ROOT / raw_path,
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
    for link in root.findall("link"):
        name = link.attrib.get("name", "")
        if not name:
            raise ValueError("URDF link is missing a name")
        link_order.append(name)
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


def _generate_spheres(
    mesh: trimesh.Trimesh,
    config: GeneratorConfig,
) -> tuple[SphereApproximation, ...]:
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

    fallback_centers, fallback_radii = _fallback_spheres(
        mesh, config.max_spheres
    )
    fallback_candidate = _inflate_spheres_for_faces(
        mesh,
        fallback_centers,
        fallback_radii,
        config.margin,
    )
    candidates = [fallback_candidate]
    if multisphere_candidate is not None:
        candidates.append(multisphere_candidate)
    centers, radii = min(
        candidates,
        key=lambda candidate: float(np.sum(candidate[1] ** 3)),
    )
    spheres = tuple(
        SphereApproximation(center=_vector3(center), radius=float(radius))
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


def _fit_capsule(points: np.ndarray, margin: float) -> CapsuleApproximation:
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
    return CapsuleApproximation(
        point_a=_vector3(point_a),
        point_b=_vector3(point_b),
        radius=radius,
    )


def _generate_capsules(
    mesh: trimesh.Trimesh,
    config: GeneratorConfig,
) -> tuple[CapsuleApproximation, ...]:
    capsules = tuple(
        _fit_capsule(_cluster_vertices(mesh, face_indices), config.margin)
        for face_indices in _face_clusters(mesh, config.max_capsules)
    )
    if len(capsules) > config.max_capsules:
        raise AssertionError("Capsule generator exceeded its configured budget")
    _validate_capsule_coverage(mesh, capsules)
    return capsules


def _validate_sphere_coverage(
    mesh: trimesh.Trimesh,
    spheres: Sequence[SphereApproximation],
    tolerance: float = 1e-9,
) -> None:
    if not spheres:
        raise ValueError("No spheres were generated")
    triangles = np.asarray(mesh.triangles)
    covered = np.zeros(len(triangles), dtype=bool)
    for sphere in spheres:
        center = np.asarray(sphere.center)
        distances = np.linalg.norm(triangles - center[None, None, :], axis=2)
        covered |= distances.max(axis=1) <= sphere.radius + tolerance
    if not np.all(covered):
        raise ValueError(
            f"Sphere approximation leaves {int((~covered).sum())} triangles uncovered"
        )


def _validate_capsule_coverage(
    mesh: trimesh.Trimesh,
    capsules: Sequence[CapsuleApproximation],
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
            f"Capsule approximation leaves {int((~covered).sum())} triangles uncovered"
        )


def generate_collision_approximations(
    robot: str,
    *,
    urdf_path: str | Path | None = None,
    config: GeneratorConfig | None = None,
    mode: str = "both",
    group_overrides: Mapping[str, Sequence[str]] | None = None,
    joint_position_overrides: Mapping[str, float] | None = None,
) -> CollisionApproximationResult:
    """Generate conservative primitives for all collision-bearing robot links."""
    if mode not in {"both", "spheres", "capsules"}:
        raise ValueError("mode must be 'both', 'spheres', or 'capsules'")
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

    approximations: list[LinkApproximation] = []
    for index, group in enumerate(groups, start=1):
        print(
            f"[{index}/{len(groups)}] fitting collision group {group.name} "
            f"from {', '.join(group.source_links)}"
        )
        mesh = group_meshes[group.name]
        spheres = _generate_spheres(mesh, config) if mode in {"both", "spheres"} else ()
        capsules = (
            _generate_capsules(mesh, config) if mode in {"both", "capsules"} else ()
        )
        approximations.append(
            LinkApproximation(
                name=group.name,
                frame=group.frame,
                source_links=group.source_links,
                spheres=spheres,
                capsules=capsules,
            )
        )

    return CollisionApproximationResult(
        robot=robot,
        source_urdf=resolved_urdf,
        config=config,
        joint_positions=dict(positions),
        groups=tuple(approximations),
    )


def _repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def _config_dict(config: GeneratorConfig) -> dict[str, Any]:
    return {
        "max_spheres_per_group": config.max_spheres,
        "max_capsules_per_group": config.max_capsules,
        "margin": config.margin,
        "voxel_divisions": config.voxel_divisions,
        "precision": config.precision,
        "voxel_padding": config.voxel_padding,
        "min_radius_vox": config.min_radius_vox,
        "min_center_distance_vox": config.min_center_distance_vox,
    }


def _yaml_document(
    result: CollisionApproximationResult,
    kind: str,
) -> dict[str, Any]:
    if kind not in {"spheres", "capsules"}:
        raise ValueError(f"Unknown approximation kind: {kind}")
    group_documents: list[dict[str, Any]] = []
    for group in result.groups:
        if kind == "spheres":
            primitives = [
                {"center": list(sphere.center), "radius": sphere.radius}
                for sphere in group.spheres
            ]
        else:
            primitives = [
                {
                    "point_a": list(capsule.point_a),
                    "point_b": list(capsule.point_b),
                    "radius": capsule.radius,
                }
                for capsule in group.capsules
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


def save_collision_approximations(
    result: CollisionApproximationResult,
    output_dir: str | Path | None = None,
    *,
    mode: str = "both",
) -> dict[str, Path]:
    """Save one or both canonical YAML approximation files."""
    if mode not in {"both", "spheres", "capsules"}:
        raise ValueError("mode must be 'both', 'spheres', or 'capsules'")
    destination = (
        default_output_dir(result.robot)
        if output_dir is None
        else Path(output_dir).expanduser()
    )
    if not destination.is_absolute():
        destination = PROJECT_ROOT / destination
    destination = destination.resolve()

    paths: dict[str, Path] = {}
    if mode in {"both", "spheres"}:
        path = destination / SPHERE_FILENAME
        _atomic_write_yaml(path, _yaml_document(result, "spheres"))
        paths["spheres"] = path
    if mode in {"both", "capsules"}:
        path = destination / CAPSULE_FILENAME
        _atomic_write_yaml(path, _yaml_document(result, "capsules"))
        paths["capsules"] = path
    return paths


def _load_yaml_document(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        document = yaml.safe_load(stream)
    if not isinstance(document, dict):
        raise ValueError(f"Approximation YAML must contain a mapping: {path}")
    if document.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported schema version in {path}: {document.get('schema_version')}"
        )
    if document.get("units") != "m":
        raise ValueError(f"Only meter-valued approximations are supported: {path}")
    return document


def _config_from_document(document: Mapping[str, Any]) -> GeneratorConfig:
    parameters = document.get("parameters", {})
    return GeneratorConfig(
        max_spheres=int(parameters.get("max_spheres_per_group", 8)),
        max_capsules=int(parameters.get("max_capsules_per_group", 3)),
        margin=float(parameters.get("margin", 0.002)),
        voxel_divisions=int(parameters.get("voxel_divisions", 100)),
        precision=float(parameters.get("precision", 0.95)),
        voxel_padding=int(parameters.get("voxel_padding", 2)),
        min_radius_vox=int(parameters.get("min_radius_vox", 2)),
        min_center_distance_vox=int(
            parameters.get("min_center_distance_vox", 2)
        ),
    )


def load_collision_approximations(
    sphere_path: str | Path | None = None,
    capsule_path: str | Path | None = None,
) -> CollisionApproximationResult:
    """Load and merge canonical sphere and capsule YAML files."""
    if sphere_path is None and capsule_path is None:
        raise ValueError("At least one approximation YAML path is required")
    documents: dict[str, dict[str, Any]] = {}
    for kind, value in (("spheres", sphere_path), ("capsules", capsule_path)):
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
                raise ValueError(f"Sphere and capsule YAML disagree on {key}")

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
                }
            data = group_data[name]
            if (
                data["frame"] != str(item["frame"])
                or data["source_links"]
                != tuple(str(link) for link in item["source_links"])
            ):
                raise ValueError(f"Sphere and capsule YAML disagree on group {name}")
            if kind == "spheres":
                data["spheres"] = tuple(
                    SphereApproximation(
                        center=_vector3(primitive["center"]),
                        radius=float(primitive["radius"]),
                    )
                    for primitive in item.get("primitives", [])
                )
            else:
                data["capsules"] = tuple(
                    CapsuleApproximation(
                        point_a=_vector3(primitive["point_a"]),
                        point_b=_vector3(primitive["point_b"]),
                        radius=float(primitive["radius"]),
                    )
                    for primitive in item.get("primitives", [])
                )

    source_urdf = Path(str(first["source_urdf"]))
    if not source_urdf.is_absolute():
        source_urdf = PROJECT_ROOT / source_urdf
    groups = tuple(
        LinkApproximation(
            name=name,
            frame=group_data[name]["frame"],
            source_links=group_data[name]["source_links"],
            spheres=group_data[name]["spheres"],
            capsules=group_data[name]["capsules"],
        )
        for name in group_order
    )
    return CollisionApproximationResult(
        robot=str(first["robot"]),
        source_urdf=source_urdf.resolve(),
        config=_config_from_document(first),
        joint_positions={
            str(name): float(value)
            for name, value in first.get("joint_positions", {}).items()
        },
        groups=groups,
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


def _publish_meshcat_scene(
    meshcat: Any,
    result: CollisionApproximationResult,
    *,
    initial_view: str,
) -> None:
    from pydrake.geometry import Capsule, Rgba, Sphere

    robot_geometry = _read_robot_geometry(result.source_urdf)
    poses = _world_link_poses(
        robot_geometry.link_order,
        robot_geometry.joints,
        result.joint_positions,
    )
    gray = Rgba(0.55, 0.58, 0.62, 0.28)
    blue = Rgba(0.05, 0.45, 1.0, 0.62)
    orange = Rgba(1.0, 0.38, 0.03, 0.72)
    root_path = "/collision_approximation"
    meshcat.Delete(root_path)

    for group in result.groups:
        for source_link in group.source_links:
            for mesh_index, source_mesh in enumerate(
                robot_geometry.link_meshes[source_link]
            ):
                path = (
                    f"{root_path}/source/{group.name}/"
                    f"{source_link}_{mesh_index:02d}"
                )
                vertices = np.asfortranarray(
                    np.asarray(source_mesh.vertices, dtype=float).T
                )
                faces = np.asfortranarray(
                    np.asarray(source_mesh.faces, dtype=np.int32).T
                )
                meshcat.SetTriangleMesh(path, vertices, faces, gray)
                meshcat.SetTransform(path, _meshcat_transform(poses[source_link]))

        anchor_pose = poses[group.frame]
        for sphere_index, sphere in enumerate(group.spheres):
            path = (
                f"{root_path}/spheres/{group.name}/"
                f"sphere_{sphere_index:02d}"
            )
            meshcat.SetObject(path, Sphere(sphere.radius), blue)
            transform = anchor_pose.copy()
            transform[:3, 3] = (
                anchor_pose[:3, :3] @ np.asarray(sphere.center)
                + anchor_pose[:3, 3]
            )
            meshcat.SetTransform(path, _meshcat_transform(transform))

        for capsule_index, capsule in enumerate(group.capsules):
            point_a = np.asarray(capsule.point_a)
            point_b = np.asarray(capsule.point_b)
            segment = point_b - point_a
            length = float(np.linalg.norm(segment))
            path = (
                f"{root_path}/capsules/{group.name}/"
                f"capsule_{capsule_index:02d}"
            )
            if length <= 1e-12:
                meshcat.SetObject(path, Sphere(capsule.radius), orange)
            else:
                meshcat.SetObject(path, Capsule(capsule.radius, length), orange)
            local_pose = np.eye(4)
            local_pose[:3, :3] = _rotation_from_z(segment)
            local_pose[:3, 3] = 0.5 * (point_a + point_b)
            meshcat.SetTransform(
                path, _meshcat_transform(anchor_pose @ local_pose)
            )

    meshcat.SetProperty(
        f"{root_path}/spheres",
        "visible",
        initial_view in {"spheres", "both"},
    )
    meshcat.SetProperty(
        f"{root_path}/capsules",
        "visible",
        initial_view in {"capsules", "both"},
    )


def _hold_meshcat(meshcat: Any) -> None:
    buttons = ("Show spheres", "Show capsules", "Show both", "Exit viewer")
    for button in buttons:
        meshcat.AddButton(button)
    clicks = {button: meshcat.GetButtonClicks(button) for button in buttons}
    root_path = "/collision_approximation"
    print("Use the Meshcat buttons to switch approximations; press Enter to exit.")
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
                meshcat.SetProperty(f"{root_path}/spheres", "visible", True)
                meshcat.SetProperty(f"{root_path}/capsules", "visible", False)
            if current["Show capsules"] > clicks["Show capsules"]:
                meshcat.SetProperty(f"{root_path}/spheres", "visible", False)
                meshcat.SetProperty(f"{root_path}/capsules", "visible", True)
            if current["Show both"] > clicks["Show both"]:
                meshcat.SetProperty(f"{root_path}/spheres", "visible", True)
                meshcat.SetProperty(f"{root_path}/capsules", "visible", True)
            clicks = current
    except KeyboardInterrupt:
        return
    finally:
        for button in buttons:
            meshcat.DeleteButton(button, strict=False)


def visualize_collision_approximations(
    result: CollisionApproximationResult,
    *,
    initial_view: str = "spheres",
    hold: bool = True,
    meshcat: Any | None = None,
) -> Any:
    """Publish source collision geometry and saved primitives to Meshcat."""
    if initial_view not in {"spheres", "capsules", "both"}:
        raise ValueError("initial_view must be spheres, capsules, or both")
    if initial_view in {"spheres", "both"} and not any(
        group.spheres for group in result.groups
    ):
        initial_view = "capsules"
    if initial_view in {"capsules", "both"} and not any(
        group.capsules for group in result.groups
    ):
        initial_view = "spheres"

    if meshcat is None:
        from pydrake.geometry import StartMeshcat

        meshcat = StartMeshcat()
    _publish_meshcat_scene(meshcat, result, initial_view=initial_view)
    if hasattr(meshcat, "web_url"):
        print(f"Meshcat URL: {meshcat.web_url()}")
    if hold:
        _hold_meshcat(meshcat)
    return meshcat


__all__ = [
    "CAPSULE_FILENAME",
    "PROJECT_ROOT",
    "ROBOT_DESCRIPTION_DIR",
    "SPHERE_FILENAME",
    "CapsuleApproximation",
    "CollisionApproximationResult",
    "GeneratorConfig",
    "LinkApproximation",
    "SphereApproximation",
    "default_output_dir",
    "generate_collision_approximations",
    "load_collision_approximations",
    "resolve_robot_urdf",
    "save_collision_approximations",
    "visualize_collision_approximations",
]
