"""Coordinate collision-proxy generation for a robot URDF.

Algorithm details live in the fitting modules; this module resolves the robot,
constructs grouped anchor-frame meshes, invokes the requested fitters, and
assembles the public result.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

from .capsule_fitting import _generate_capsules
from .constants import (
    CAPSULE_FILENAME,
    DEFAULT_CONFIG_PATH,
    ELLIPSOID_FILENAME,
    ROBOT_DESCRIPTION_DIR,
    SPHERE_FILENAME,
)
from .ellipsoid_fitting import _mesh_ellipsoid_proxy

# Keep legacy imports from this module working while implementations live in
# their focused visualization and persistence modules.
from .meshcat_visualizer import (
    _add_joint_sliders,
    _body_meshcat_path,
    _joint_slider_specs,
    _meshcat_transform,
    _publish_meshcat_scene,
    _read_joint_sliders,
    _set_body_view,
    _sphere_meshcat_path,
    _update_meshcat_transforms,
    visualize_collision_model,
    visualize_collision_proxies,
)
from .proxy_io import (
    _atomic_write_yaml,
    _yaml_document,
    load_collision_model,
    load_collision_proxies,
    save_collision_proxies,
)
from .proxy_types import (
    CapsuleProxy,
    CollisionProxyConfig,
    CollisionProxyResult,
    EllipsoidProxy,
    GeneratorConfig,
    LinkProxy,
    SphereFitMetrics,
    SphereProxy,
    _mode_kinds,
)
from .sphere_fitting import _generate_spheres
from .urdf_geometry import (
    _combine_group_meshes,
    _make_group_definitions,
    _read_robot_geometry,
    _reference_joint_positions,
    _world_link_poses,
    default_output_dir,
    resolve_robot_urdf,
)


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

    # All fitters receive geometry in the group's anchor-link frame at the same
    # saved reference configuration.
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

        # The mesh ellipsoid is always computed: it is a saved density bound
        # and also supplies deterministic principal axes to the other fitters.
        ellipsoid = _mesh_ellipsoid_proxy(mesh, config)
        if "spheres" in requested_kinds:
            spheres, sphere_fit = _generate_spheres(
                mesh,
                ellipsoid,
                config,
            )
        else:
            spheres = ()
            sphere_fit = None
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
                sphere_fit=sphere_fit,
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
    "SphereFitMetrics",
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
