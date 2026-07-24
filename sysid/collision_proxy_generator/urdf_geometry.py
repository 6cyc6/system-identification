"""URDF parsing, collision-mesh loading, grouping, and forward kinematics.

Raw URDF collision elements are converted to Trimesh objects in their link
frames. Grouped geometry is later transformed into one anchor-link frame at the
chosen reference configuration, including an open gripper and mimic joints.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence
from urllib.parse import unquote, urlparse

import numpy as np
import trimesh

from scipy.spatial.transform import Rotation
from sysid.utils.path_utils import BASE_DIR

from .constants import GRIPPER_LINK_TOKENS, ROBOT_DESCRIPTION_DIR


@dataclass(frozen=True)
class _Joint:
    """Minimal URDF joint data needed for reference-pose kinematics."""

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
    """A named set of source links represented in one anchor-link frame."""

    name: str
    frame: str
    source_links: tuple[str, ...]
    maximize_internal_joints: bool = False


@dataclass
class _RobotGeometry:
    """Parsed robot topology, link meshes, and optional derived geometry."""

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

    matches = (
        sorted(description_dir.rglob("*.urdf"))
        if description_dir.is_dir()
        else []
    )
    if not matches:
        raise FileNotFoundError(
            f"No URDF found for {robot!r} below {description_dir}"
        )
    if len(matches) > 1:
        formatted = "\n  ".join(str(path) for path in matches)
        raise RuntimeError(
            f"Multiple URDFs are available for {robot!r}; "
            f"pass --urdf explicitly:\n  {formatted}"
        )
    return matches[0].resolve()


def default_output_dir(robot: str) -> Path:
    """Return the canonical collision-proxy output directory."""
    return ROBOT_DESCRIPTION_DIR / f"{robot}_description" / "collisions"


def _parse_floats(
    value: str | None,
    *,
    default: Sequence[float],
    count: int,
    label: str,
) -> np.ndarray:
    """Parse a fixed-length finite vector from one URDF attribute."""
    if value is None:
        result = np.asarray(default, dtype=float)
    else:
        result = np.fromstring(value, sep=" ", dtype=float)
    if result.size != count or not np.all(np.isfinite(result)):
        raise ValueError(
            f"{label} must contain {count} finite values, got {value!r}"
        )
    return result


def _origin_transform(element: ET.Element | None) -> np.ndarray:
    """Convert a URDF ``origin`` element to a homogeneous transform."""
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
    """Resolve file and package mesh URIs using repository search roots."""
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
        f"Could not resolve collision mesh {filename!r}. "
        f"Tried:\n  {formatted}"
    )


def _load_trimesh(path: Path) -> trimesh.Trimesh:
    """Load one triangle mesh and flatten scene-valued formats."""
    try:
        loaded = trimesh.load(path, process=False)
    except Exception as error:
        raise ValueError(
            f"Failed to load collision mesh {path}: {error}"
        ) from error

    if isinstance(loaded, trimesh.Scene):
        try:
            loaded = loaded.dump(concatenate=True)
        except Exception as error:
            raise ValueError(
                f"Could not flatten mesh scene {path}: {error}"
            ) from error
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
    """Convert a URDF collision geometry element to a link-frame mesh."""
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
            raise ValueError(
                "Cylinder radius and length must be positive"
            )
        return trimesh.creation.cylinder(
            radius=radius,
            height=length,
            sections=32,
        )

    sphere = geometry.find("sphere")
    if sphere is not None:
        radius = float(sphere.attrib["radius"])
        if radius <= 0.0:
            raise ValueError("Sphere radius must be positive")
        return trimesh.creation.icosphere(
            subdivisions=2,
            radius=radius,
        )

    capsule = geometry.find("capsule")
    if capsule is not None:
        radius = float(capsule.attrib["radius"])
        length = float(capsule.attrib["length"])
        if radius <= 0.0 or length < 0.0:
            raise ValueError(
                "Capsule radius must be positive and length non-negative"
            )
        mesh = trimesh.creation.capsule(radius=radius, height=length)
        mesh.apply_translation((0.0, 0.0, -0.5 * length))
        return mesh

    tags = [child.tag for child in geometry]
    raise ValueError(f"Unsupported URDF collision geometry: {tags}")


def _parse_joint(joint: ET.Element) -> _Joint:
    """Read joint limits, mimic metadata, axis, and parent/child frames."""
    name = joint.attrib.get("name", "")
    joint_type = joint.attrib.get("type", "fixed")
    parent = joint.find("parent")
    child = joint.find("child")
    if not name or parent is None or child is None:
        raise ValueError(
            "Every URDF joint requires name, parent, and child"
        )
    axis_element = joint.find("axis")
    axis = _parse_floats(
        (
            None
            if axis_element is None
            else axis_element.attrib.get("xyz")
        ),
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
    lower = (
        None
        if limit is None or "lower" not in limit.attrib
        else float(limit.attrib["lower"])
    )
    upper = (
        None
        if limit is None or "upper" not in limit.attrib
        else float(limit.attrib["upper"])
    )
    mimic = joint.find("mimic")
    mimic_joint = None if mimic is None else mimic.attrib.get("joint")
    mimic_multiplier = (
        1.0
        if mimic is None
        else float(mimic.attrib.get("multiplier", "1"))
    )
    mimic_offset = (
        0.0
        if mimic is None
        else float(mimic.attrib.get("offset", "0"))
    )
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
    """Parse topology and all collision elements from a resolved URDF."""
    try:
        root = ET.parse(urdf_path).getroot()
    except ET.ParseError as error:
        raise ValueError(f"Invalid URDF XML in {urdf_path}: {error}") from error

    description_dir = urdf_path.parent
    while (
        description_dir != ROBOT_DESCRIPTION_DIR
        and not description_dir.name.endswith("_description")
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
            mesh.apply_transform(
                _origin_transform(collision.find("origin"))
            )
            meshes.append(mesh)
        if meshes:
            link_meshes[name] = tuple(meshes)

    joints = tuple(
        _parse_joint(joint) for joint in root.findall("joint")
    )
    return _RobotGeometry(
        urdf_path=urdf_path,
        root=root,
        link_order=tuple(link_order),
        joints=joints,
        link_meshes=link_meshes,
    )


def _is_gripper_link(name: str) -> bool:
    """Return whether a link participates in automatic gripper grouping."""
    lowered = name.lower()
    return any(token in lowered for token in GRIPPER_LINK_TOKENS)


def _make_group_definitions(
    robot_geometry: _RobotGeometry,
    group_overrides: Mapping[str, Sequence[str]] | None,
) -> tuple[_GroupDefinition, ...]:
    """Build disjoint groups, applying overrides before automatic merging."""
    collision_links = set(robot_geometry.link_meshes)
    link_index = {
        name: index
        for index, name in enumerate(robot_geometry.link_order)
    }
    link_names = set(robot_geometry.link_order)
    assigned: set[str] = set()
    definitions: list[_GroupDefinition] = []

    for anchor, members_value in (group_overrides or {}).items():
        if anchor not in link_names:
            raise ValueError(f"Unknown group anchor link: {anchor}")
        members = tuple(dict.fromkeys(members_value))
        if not members:
            raise ValueError(
                f"Explicit group {anchor!r} has no member links"
            )
        unknown = set(members) - collision_links
        if unknown:
            raise ValueError(
                f"Explicit group {anchor!r} contains links without "
                f"collision geometry: {sorted(unknown)}"
            )
        duplicate = assigned.intersection(members)
        if duplicate:
            raise ValueError(
                f"Links occur in multiple explicit groups: "
                f"{sorted(duplicate)}"
            )
        assigned.update(members)
        definitions.append(
            _GroupDefinition(
                name=anchor,
                frame=anchor,
                source_links=members,
                maximize_internal_joints=any(
                    _is_gripper_link(name) for name in members
                ),
            )
        )

    candidates = {
        name for name in collision_links - assigned
        if _is_gripper_link(name)
    }
    adjacency: dict[str, set[str]] = {
        name: set() for name in candidates
    }
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
        members = tuple(
            sorted(component, key=link_index.__getitem__)
        )
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
                _GroupDefinition(
                    name=link,
                    frame=link,
                    source_links=(link,),
                )
            )

    definitions.sort(
        key=lambda group: min(
            link_index[name] for name in group.source_links
        )
    )
    return tuple(definitions)


def _clamp_reference_position(
    joint: _Joint,
    value: float = 0.0,
) -> float:
    """Clamp a candidate joint value to its finite URDF limits."""
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
    """Return joints connecting an anchor to all grouped descendants."""
    parent_joint = {joint.child: joint for joint in joints}
    path_names: set[str] = set()
    for source in sources:
        current = source
        visited: set[str] = set()
        while current != anchor:
            if current in visited or current not in parent_joint:
                raise ValueError(
                    f"Group anchor {anchor!r} is not an ancestor of "
                    f"source link {source!r}"
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
    """Choose a valid robot pose with grouped gripper joints fully open."""
    joints_by_name = {
        joint.name: joint for joint in robot_geometry.joints
    }
    unknown = set(overrides or {}) - set(joints_by_name)
    if unknown:
        raise ValueError(
            f"Unknown joint-position overrides: {sorted(unknown)}"
        )

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

    positions.update(
        {
            name: float(value)
            for name, value in (overrides or {}).items()
        }
    )

    unresolved = {
        joint.name
        for joint in robot_geometry.joints
        if (
            joint.joint_type != "fixed"
            and joint.mimic_joint is not None
        )
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
        raise ValueError(
            f"Unresolved mimic joints: {sorted(unresolved)}"
        )

    for name, value in positions.items():
        joint = joints_by_name[name]
        if (
            joint.lower is not None
            and value < joint.lower - 1e-12
        ):
            raise ValueError(
                f"Joint {name}={value} is below lower limit "
                f"{joint.lower}"
            )
        if (
            joint.upper is not None
            and value > joint.upper + 1e-12
        ):
            raise ValueError(
                f"Joint {name}={value} is above upper limit "
                f"{joint.upper}"
            )
    return positions


def _joint_motion_transform(
    joint: _Joint,
    position: float,
) -> np.ndarray:
    """Return the variable transform contributed by one URDF joint."""
    transform = np.eye(4)
    if joint.joint_type in ("revolute", "continuous"):
        transform[:3, :3] = Rotation.from_rotvec(
            joint.axis * position
        ).as_matrix()
    elif joint.joint_type == "prismatic":
        transform[:3, 3] = joint.axis * position
    elif joint.joint_type != "fixed":
        raise ValueError(
            f"Unsupported joint type {joint.joint_type!r} "
            f"for joint {joint.name}"
        )
    return transform


def _world_link_poses(
    link_order: Sequence[str],
    joints: Sequence[_Joint],
    joint_positions: Mapping[str, float],
) -> dict[str, np.ndarray]:
    """Evaluate forward kinematics for every link in the URDF forest."""
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
            raise ValueError(
                f"URDF kinematic graph revisits link {link!r}"
            )
        poses[link] = world_pose
        for joint in reversed(children.get(link, [])):
            position = joint_positions.get(joint.name, 0.0)
            child_pose = (
                world_pose
                @ joint.origin
                @ _joint_motion_transform(joint, position)
            )
            stack.append((joint.child, child_pose))
    missing = set(link_order) - set(poses)
    if missing:
        raise ValueError(
            f"URDF contains unreachable links: {sorted(missing)}"
        )
    return poses


def _combine_group_meshes(
    robot_geometry: _RobotGeometry,
    group_definitions: Sequence[_GroupDefinition],
    world_poses: Mapping[str, np.ndarray],
) -> dict[str, trimesh.Trimesh]:
    """Transform source-link meshes into each group's anchor-link frame."""
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
            raise ValueError(
                f"Collision group {group.name!r} contains no geometry"
            )
        group_mesh = trimesh.util.concatenate(meshes)
        if (
            not isinstance(group_mesh, trimesh.Trimesh)
            or len(group_mesh.faces) == 0
        ):
            raise ValueError(
                f"Failed to combine collision group {group.name!r}"
            )
        combined[group.name] = group_mesh
    return combined


__all__ = [
    "default_output_dir",
    "resolve_robot_urdf",
]
