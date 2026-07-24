"""Meshcat scene construction and controls for collision proxy models.

Source collision meshes are placed in link frames, whereas proxies are placed
in group anchor frames. Joint sliders update both through forward kinematics so
the persisted proxy coordinates remain unchanged while the robot moves.
"""

from __future__ import annotations

import math
import select
import sys
import time

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from scipy.spatial.transform import Rotation

from .constants import (
    BODY_VIEW_BUTTON_PREFIX,
    JOINT_SLIDER_PREFIX,
    MESHCAT_ROOT_PATH,
    PROXY_KINDS,
    PROXY_MODES,
    PROXY_OPACITY_SLIDER,
    SHOW_ALL_BODIES_BUTTON,
)
from .proxy_io import load_collision_model
from .proxy_types import (
    CapsuleProxy,
    CollisionProxyConfig,
    CollisionProxyResult,
    EllipsoidProxy,
    _mode_kinds,
)
from .urdf_geometry import (
    _Joint,
    _RobotGeometry,
    _clamp_reference_position,
    _read_robot_geometry,
    _world_link_poses,
)


@dataclass(frozen=True)
class _JointSlider:
    """Meshcat control metadata for one independent non-fixed joint."""

    control_name: str
    joint_name: str
    minimum: float
    maximum: float
    step: float
    initial_value: float


def _joint_slider_specs(
    robot_geometry: _RobotGeometry,
    joint_positions: Mapping[str, float],
) -> tuple[_JointSlider, ...]:
    """Create bounded controls for independent revolute/prismatic joints."""
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
    """Publish joint controls and return their initial joint-position map."""
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
    """Propagate independent joint values through URDF mimic relationships."""
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
    """Read controls and report whether forward kinematics must be refreshed."""
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


def _rotation_from_z(direction: np.ndarray) -> np.ndarray:
    """Rotate Meshcat's capsule z-axis onto a requested segment direction."""
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
        return Rotation.from_rotvec(
            np.array((math.pi, 0.0, 0.0))
        ).as_matrix()
    axis = np.cross(z_axis, direction)
    axis /= np.linalg.norm(axis)
    return Rotation.from_rotvec(axis * math.acos(dot)).as_matrix()


def _meshcat_transform(matrix: np.ndarray) -> Any:
    """Convert a homogeneous NumPy matrix to Drake's transform type."""
    from pydrake.math import RigidTransform

    return RigidTransform(np.asarray(matrix, dtype=float))


def _body_meshcat_path(group_name: str) -> str:
    """Return the scene-tree root for one collision group."""
    return f"{MESHCAT_ROOT_PATH}/bodies/{group_name}"


def _source_meshcat_path(
    group_name: str,
    source_link: str,
    mesh_index: int,
) -> str:
    """Return a stable scene path for one source collision mesh."""
    return (
        f"{_body_meshcat_path(group_name)}/source/"
        f"{source_link}_{mesh_index:02d}"
    )


def _sphere_meshcat_path(group_name: str, sphere_index: int) -> str:
    """Return a stable scene path for one sphere proxy."""
    return (
        f"{_body_meshcat_path(group_name)}/spheres/"
        f"sphere_{sphere_index:02d}"
    )


def _capsule_meshcat_path(group_name: str, capsule_index: int) -> str:
    """Return a stable scene path for one capsule proxy."""
    return (
        f"{_body_meshcat_path(group_name)}/capsules/"
        f"capsule_{capsule_index:02d}"
    )


def _ellipsoid_meshcat_path(
    group_name: str,
    ellipsoid_index: int,
) -> str:
    """Return a stable scene path for one ellipsoid proxy."""
    return (
        f"{_body_meshcat_path(group_name)}/ellipsoids/"
        f"ellipsoid_{ellipsoid_index:02d}"
    )


def _capsule_local_pose(capsule: CapsuleProxy) -> np.ndarray:
    """Return the capsule pose relative to its group anchor frame."""
    point_a = np.asarray(capsule.point_a)
    point_b = np.asarray(capsule.point_b)
    local_pose = np.eye(4)
    local_pose[:3, :3] = _rotation_from_z(point_b - point_a)
    local_pose[:3, 3] = 0.5 * (point_a + point_b)
    return local_pose


def _ellipsoid_local_pose(ellipsoid: EllipsoidProxy) -> np.ndarray:
    """Return the ellipsoid pose relative to its group anchor frame."""
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
    """Apply current world poses to source meshes and anchor-frame proxies."""
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
    """Show the primitive kinds selected by a proxy view mode."""
    visible_kinds = _mode_kinds(view)
    for group in result.groups:
        body_path = _body_meshcat_path(group.name)
        for kind in PROXY_KINDS:
            meshcat.SetProperty(
                f"{body_path}/{kind}",
                "visible",
                kind in visible_kinds,
            )


def _set_proxy_opacity(
    meshcat: Any,
    result: CollisionProxyResult,
    opacity: float,
) -> None:
    """Apply one opacity to proxies while preserving source mesh opacity."""
    transparent = opacity < 1.0
    for group in result.groups:
        body_path = _body_meshcat_path(group.name)
        for kind in PROXY_KINDS:
            path = f"{body_path}/{kind}"
            meshcat.SetProperty(path, "transparent", transparent)
            meshcat.SetProperty(path, "opacity", opacity)


def _publish_meshcat_scene(
    meshcat: Any,
    result: CollisionProxyResult,
    *,
    initial_view: str,
    proxy_opacity: float,
) -> _RobotGeometry:
    """Create source and proxy objects, then place them at the saved pose."""
    from pydrake.geometry import Capsule, Ellipsoid, Rgba, Sphere

    robot_geometry = _read_robot_geometry(result.source_urdf)
    gray = Rgba(0.62, 0.65, 0.70, 1.0)
    blue = Rgba(0.05, 0.45, 1.0, proxy_opacity)
    orange = Rgba(1.0, 0.38, 0.03, proxy_opacity)
    green = Rgba(0.15, 0.75, 0.30, proxy_opacity)
    meshcat.Delete(MESHCAT_ROOT_PATH)

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
            length = float(np.linalg.norm(point_b - point_a))
            path = _capsule_meshcat_path(group.name, capsule_index)
            if length <= 1e-12:
                meshcat.SetObject(path, Sphere(capsule.radius), orange)
            else:
                meshcat.SetObject(
                    path,
                    Capsule(capsule.radius, length),
                    orange,
                )

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


def _set_body_view(
    meshcat: Any,
    result: CollisionProxyResult,
    body_name: str | None,
) -> None:
    """Show every body, or exclusively show one named collision group."""
    body_names = {group.name for group in result.groups}
    if body_name is not None and body_name not in body_names:
        raise ValueError(f"Unknown collision-proxy body: {body_name}")
    for group in result.groups:
        meshcat.SetProperty(
            _body_meshcat_path(group.name),
            "visible",
            body_name is None or group.name == body_name,
        )


def _hold_meshcat(
    meshcat: Any,
    result: CollisionProxyResult,
    robot_geometry: _RobotGeometry,
    *,
    proxy_opacity: float,
) -> None:
    """Run the interactive control loop until Enter or Exit is pressed."""
    proxy_buttons = (
        "Show spheres",
        "Show capsules",
        "Show ellipsoids",
        "Show both",
        "Show all",
    )
    body_buttons = tuple(
        f"{BODY_VIEW_BUTTON_PREFIX}{group.name}"
        for group in result.groups
    )
    buttons = (
        *proxy_buttons,
        *body_buttons,
        SHOW_ALL_BODIES_BUTTON,
        "Exit viewer",
    )
    for button in buttons:
        meshcat.AddButton(button)
    clicks = {
        button: meshcat.GetButtonClicks(button) for button in buttons
    }
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
        "Use the joint and opacity sliders in Controls. Press a named "
        "'Show body' button to isolate one body, or Show all bodies to "
        "restore the robot; press Enter to exit."
    )
    try:
        while True:
            if sys.stdin.isatty():
                readable, _writable, _errors = select.select(
                    [sys.stdin],
                    [],
                    [],
                    0.1,
                )
                if readable:
                    sys.stdin.readline()
                    return
            else:
                time.sleep(0.1)

            current = {
                button: meshcat.GetButtonClicks(button)
                for button in buttons
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
            if (
                current[SHOW_ALL_BODIES_BUTTON]
                > clicks[SHOW_ALL_BODIES_BUTTON]
            ):
                _set_body_view(meshcat, result, None)
            for group, button in zip(result.groups, body_buttons):
                if current[button] > clicks[button]:
                    _set_body_view(meshcat, result, group.name)
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
            raise ValueError(
                "The collision model contains no proxy primitives"
            )
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
    "visualize_collision_model",
    "visualize_collision_proxies",
]
