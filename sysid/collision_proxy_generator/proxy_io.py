"""YAML serialization and loading for collision proxy models.

Each primitive kind is stored in a separate canonical file, but the metadata
and group ordering are identical. Loading multiple files merges those records
back into one ``CollisionProxyResult`` and rejects mismatched robot models.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml

from sysid.utils.path_utils import BASE_DIR

from .constants import (
    CAPSULE_FILENAME,
    ELLIPSOID_FILENAME,
    PROXY_KINDS,
    SCHEMA_VERSION,
    SPHERE_FILENAME,
    SUPPORTED_SCHEMA_VERSIONS,
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
    _matrix3,
    _mode_kinds,
    _vector3,
)
from .urdf_geometry import default_output_dir


def _repo_relative(path: Path) -> str:
    """Prefer portable repository-relative paths in generated YAML."""
    try:
        return path.resolve().relative_to(BASE_DIR).as_posix()
    except ValueError:
        return str(path.resolve())


def _config_dict(config: GeneratorConfig) -> dict[str, Any]:
    """Convert numerical generator settings to their stable YAML names."""
    return {
        "max_spheres_per_group": config.max_spheres,
        "max_capsules_per_group": config.max_capsules,
        "margin": config.margin,
        "max_sphere_overhang": config.max_sphere_overhang,
        "sphere_surface_subdivisions": config.sphere_surface_subdivisions,
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
    """Build one complete YAML document for a single primitive kind."""
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
                    "rotation": [list(row) for row in ellipsoid.rotation],
                    "axis_ratio": ellipsoid.axis_ratio,
                    "elongated": ellipsoid.elongated,
                }
                for ellipsoid in group.ellipsoids
            ]

        group_document: dict[str, Any] = {
            "name": group.name,
            "frame": group.frame,
            "source_links": list(group.source_links),
            "primitives": primitives,
        }
        if kind == "spheres" and group.sphere_fit is not None:
            group_document["fit"] = {
                "max_overhang": group.sphere_fit.max_overhang,
                "p95_overhang": group.sphere_fit.p95_overhang,
                "mean_overhang": group.sphere_fit.mean_overhang,
                "tolerance": group.sphere_fit.tolerance,
                "tolerance_met": group.sphere_fit.tolerance_met,
                "surface_samples": group.sphere_fit.surface_samples,
            }
        group_documents.append(group_document)

    return {
        "schema_version": SCHEMA_VERSION,
        "kind": kind,
        "robot": result.robot,
        "source_urdf": _repo_relative(result.source_urdf),
        "units": "m",
        "parameters": _config_dict(result.config),
        "joint_positions": {
            name: float(value)
            for name, value in result.joint_positions.items()
        },
        "groups": group_documents,
    }


def _atomic_write_yaml(path: Path, document: Mapping[str, Any]) -> None:
    """Replace a YAML file only after its temporary file is fully written."""
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

    filenames = {
        "spheres": SPHERE_FILENAME,
        "capsules": CAPSULE_FILENAME,
        "ellipsoids": ELLIPSOID_FILENAME,
    }
    paths: dict[str, Path] = {}
    for kind in PROXY_KINDS:
        if kind not in requested_kinds:
            continue
        path = destination / filenames[kind]
        _atomic_write_yaml(path, _yaml_document(result, kind))
        paths[kind] = path
    return paths


def _load_yaml_document(path: Path) -> dict[str, Any]:
    """Load and validate the common YAML envelope."""
    with path.open("r", encoding="utf-8") as stream:
        document = yaml.safe_load(stream)
    if not isinstance(document, dict):
        raise ValueError(f"Proxy YAML must contain a mapping: {path}")
    if document.get("schema_version") not in SUPPORTED_SCHEMA_VERSIONS:
        raise ValueError(
            f"Unsupported schema version in {path}: "
            f"{document.get('schema_version')}"
        )
    if document.get("units") != "m":
        raise ValueError(f"Only meter-valued proxies are supported: {path}")
    return document


def _config_from_document(document: Mapping[str, Any]) -> GeneratorConfig:
    """Recover generator settings, including defaults for older schemas."""
    parameters = document.get("parameters", {})
    return GeneratorConfig(
        max_spheres=int(parameters.get("max_spheres_per_group", 8)),
        max_capsules=int(parameters.get("max_capsules_per_group", 3)),
        margin=float(parameters.get("margin", 0.002)),
        max_sphere_overhang=float(
            parameters.get("max_sphere_overhang", 0.05)
        ),
        sphere_surface_subdivisions=int(
            parameters.get("sphere_surface_subdivisions", 2)
        ),
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
            raise ValueError(
                f"{path} contains {document.get('kind')!r}, "
                f"expected {kind!r}"
            )
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
                    "source_links": tuple(
                        str(link) for link in item["source_links"]
                    ),
                    "spheres": (),
                    "sphere_fit": None,
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
                fit = item.get("fit")
                if fit is not None:
                    data["sphere_fit"] = SphereFitMetrics(
                        max_overhang=float(fit["max_overhang"]),
                        p95_overhang=float(fit["p95_overhang"]),
                        mean_overhang=float(fit["mean_overhang"]),
                        tolerance=float(fit["tolerance"]),
                        surface_samples=int(fit["surface_samples"]),
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
            sphere_fit=group_data[name]["sphere_fit"],
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
    """Resolve the configured output directory against the repository root."""
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
    return load_collision_proxies(
        sphere_path=(
            proxy_directory / SPHERE_FILENAME
            if "spheres" in requested_kinds
            else None
        ),
        capsule_path=(
            proxy_directory / CAPSULE_FILENAME
            if "capsules" in requested_kinds
            else None
        ),
        ellipsoid_path=(
            proxy_directory / ELLIPSOID_FILENAME
            if "ellipsoids" in requested_kinds
            else None
        ),
    )


__all__ = [
    "load_collision_model",
    "load_collision_proxies",
    "save_collision_proxies",
]
