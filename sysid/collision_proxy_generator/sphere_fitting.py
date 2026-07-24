"""Greedy conservative sphere-union fitting.

The process starts with one enclosing sphere and replaces one sphere by two at
each refinement. Source triangles are clipped by each split plane, so every
child sphere encloses its complete assigned surface patch. The first union
meeting the overhang tolerance is therefore both conservative and economical.
"""

from __future__ import annotations

import math
import sys

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import trimesh

from .proxy_types import (
    EllipsoidProxy,
    GeneratorConfig,
    SphereFitMetrics,
    SphereProxy,
    _vector3,
)


def _minimum_sphere(points: np.ndarray) -> tuple[np.ndarray, float]:
    """Return a finite enclosing sphere, with a deterministic safe fallback."""
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


def _clip_polygon_at_axis_offset(
    polygon: np.ndarray,
    *,
    center: np.ndarray,
    axis: np.ndarray,
    boundary: float,
    keep_greater: bool,
) -> np.ndarray:
    """Clip a 3-D polygon against one side of an axis-aligned plane."""
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


@dataclass
class _SpherePatch:
    """A convexly covered collection of exact collision-triangle subpatches."""

    polygons: tuple[np.ndarray, ...]
    center: np.ndarray
    radius: float


def _fit_sphere_patch(
    polygons: Sequence[np.ndarray],
    margin: float,
) -> _SpherePatch:
    """Fit an enclosing sphere to every vertex of a triangle patch."""
    if not polygons:
        raise ValueError("Cannot fit a sphere to an empty collision patch")
    points = np.concatenate(polygons)
    center, radius = _minimum_sphere(points)
    return _SpherePatch(
        polygons=tuple(polygons),
        center=center,
        radius=radius + margin,
    )


def _canonical_direction(direction: np.ndarray) -> np.ndarray | None:
    """Normalize a split direction and choose a repeatable sign."""
    direction = np.asarray(direction, dtype=float)
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-12:
        return None
    direction = direction / norm
    nonzero = np.flatnonzero(np.abs(direction) > 1e-12)
    if len(nonzero) and direction[int(nonzero[0])] < 0.0:
        direction = -direction
    return direction


def _sphere_split_directions(
    patch: _SpherePatch,
    ellipsoid: EllipsoidProxy,
) -> tuple[np.ndarray, ...]:
    """Build deterministic world-, ellipsoid-, and patch-aligned split axes."""
    polygon_centers = np.asarray(
        [polygon.mean(axis=0) for polygon in patch.polygons]
    )
    centered = polygon_centers - polygon_centers.mean(axis=0)
    candidates = [
        *np.eye(3),
        *np.asarray(ellipsoid.rotation).T,
    ]
    if len(polygon_centers) > 1:
        try:
            candidates.extend(
                np.linalg.svd(centered, full_matrices=False)[2]
            )
        except np.linalg.LinAlgError:
            pass

    directions: list[np.ndarray] = []
    for candidate in candidates:
        direction = _canonical_direction(candidate)
        if direction is None:
            continue
        if any(
            np.allclose(direction, existing, atol=1e-10)
            for existing in directions
        ):
            continue
        directions.append(direction)
    return tuple(directions)


def _clean_clipped_polygon(polygon: np.ndarray) -> np.ndarray:
    """Remove duplicate points and reject degenerate clipped polygons."""
    if len(polygon) < 3:
        return np.empty((0, 3), dtype=float)
    cleaned = [polygon[0]]
    for point in polygon[1:]:
        if np.linalg.norm(point - cleaned[-1]) > 1e-12:
            cleaned.append(point)
    if len(cleaned) > 1 and np.linalg.norm(cleaned[0] - cleaned[-1]) <= 1e-12:
        cleaned.pop()
    if len(cleaned) < 3:
        return np.empty((0, 3), dtype=float)
    array = np.asarray(cleaned)
    area_vector = np.zeros(3)
    for index in range(1, len(array) - 1):
        area_vector += np.cross(
            array[index] - array[0],
            array[index + 1] - array[0],
        )
    if np.linalg.norm(area_vector) <= 1e-18:
        return np.empty((0, 3), dtype=float)
    return array


def _sphere_patch_splits(
    patch: _SpherePatch,
    ellipsoid: EllipsoidProxy,
) -> Iterable[tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...]]]:
    """Yield valid two-way patch partitions for greedy refinement."""
    all_points = np.concatenate(patch.polygons)
    seen: set[tuple[float, ...]] = set()
    for direction in _sphere_split_directions(patch, ellipsoid):
        projections = all_points @ direction
        minimum = float(projections.min())
        maximum = float(projections.max())
        if maximum - minimum <= 1e-12:
            continue
        for fraction in (0.25, 0.5, 0.75):
            boundary = minimum + fraction * (maximum - minimum)
            if boundary <= minimum + 1e-12 or boundary >= maximum - 1e-12:
                continue
            key = tuple(np.round(np.append(direction, boundary), 12))
            if key in seen:
                continue
            seen.add(key)
            lower: list[np.ndarray] = []
            upper: list[np.ndarray] = []
            for polygon in patch.polygons:
                lower_polygon = _clean_clipped_polygon(
                    _clip_polygon_at_axis_offset(
                        polygon,
                        center=np.zeros(3),
                        axis=direction,
                        boundary=boundary,
                        keep_greater=False,
                    )
                )
                upper_polygon = _clean_clipped_polygon(
                    _clip_polygon_at_axis_offset(
                        polygon,
                        center=np.zeros(3),
                        axis=direction,
                        boundary=boundary,
                        keep_greater=True,
                    )
                )
                if len(lower_polygon):
                    lower.append(lower_polygon)
                if len(upper_polygon):
                    upper.append(upper_polygon)
            if lower and upper:
                yield tuple(lower), tuple(upper)


def _sphere_directions(subdivisions: int) -> np.ndarray:
    """Return repeatable unit directions for exposed-union sampling."""
    directions = list(
        trimesh.creation.icosphere(
            subdivisions=subdivisions,
            radius=1.0,
        ).vertices
    )
    for direction in (*np.eye(3), *-np.eye(3)):
        if not any(
            np.allclose(direction, existing, atol=1e-12)
            for existing in directions
        ):
            directions.append(direction)
    return np.asarray(directions)


def _sphere_union_fit_metrics(
    mesh: trimesh.Trimesh,
    patches: Sequence[_SpherePatch],
    *,
    tolerance: float,
    subdivisions: int,
    cache: dict[
        tuple[bytes, float, int],
        tuple[np.ndarray, np.ndarray],
    ] | None = None,
) -> SphereFitMetrics:
    """Measure source-mesh distance only on the exposed sphere-union surface."""
    directions = _sphere_directions(subdivisions)
    cache = {} if cache is None else cache
    point_sets: list[np.ndarray] = []
    distance_sets: list[np.ndarray] = []
    for patch in patches:
        key = (patch.center.tobytes(), patch.radius, subdivisions)
        cached = cache.get(key)
        if cached is None:
            points = (
                patch.center[None, :]
                + patch.radius * directions
            )
            _closest, distances, _triangle_ids = (
                trimesh.proximity.closest_point_naive(mesh, points)
            )
            cached = (points, np.asarray(distances, dtype=float))
            cache[key] = cached
        points, distances = cached
        point_sets.append(points)
        distance_sets.append(distances)

    points = np.concatenate(point_sets)
    distances = np.concatenate(distance_sets)
    owners = np.repeat(np.arange(len(patches)), len(directions))
    # Samples inside another sphere are internal to the union and must not
    # contribute to the outward-overhang measurement.
    exposed = np.ones(len(points), dtype=bool)
    for patch_index, patch in enumerate(patches):
        inside = (
            np.linalg.norm(
                points - patch.center[None, :],
                axis=1,
            )
            < patch.radius - 1e-10
        )
        exposed &= (owners == patch_index) | ~inside
    exposed_distances = distances[exposed]
    if len(exposed_distances) == 0:
        raise ValueError("Sphere union has no exposed surface samples")
    return SphereFitMetrics(
        max_overhang=float(exposed_distances.max()),
        p95_overhang=float(np.quantile(exposed_distances, 0.95)),
        mean_overhang=float(exposed_distances.mean()),
        tolerance=tolerance,
        surface_samples=len(exposed_distances),
    )


def _sphere_fit_key(
    metrics: SphereFitMetrics,
    patches: Sequence[_SpherePatch],
) -> tuple[float, int, float, float, float]:
    """Order candidates by overhang, count, and then occupied volume."""
    return (
        metrics.max_overhang,
        len(patches),
        metrics.p95_overhang,
        metrics.mean_overhang,
        float(sum(patch.radius**3 for patch in patches)),
    )


def _sphere_proxies(
    patches: Sequence[_SpherePatch],
) -> tuple[SphereProxy, ...]:
    """Convert mutable fitting patches to the public immutable representation."""
    return tuple(
        SphereProxy(
            center=_vector3(patch.center),
            radius=float(patch.radius),
        )
        for patch in patches
    )


def _generate_spheres(
    mesh: trimesh.Trimesh,
    ellipsoid: EllipsoidProxy,
    config: GeneratorConfig,
) -> tuple[tuple[SphereProxy, ...], SphereFitMetrics]:
    """Return the first conservative greedy cover meeting the overhang limit.

    Each refinement replaces one sphere by two spheres.  Its source triangle
    patches are clipped exactly by a plane, so the child spheres continue to
    contain every point of the original collision surface.
    """
    patches = [
        _fit_sphere_patch(
            tuple(np.asarray(mesh.triangles)),
            config.margin,
        )
    ]
    selection_subdivisions = max(
        0,
        config.sphere_surface_subdivisions - 1,
    )
    selection_cache: dict[
        tuple[bytes, float, int],
        tuple[np.ndarray, np.ndarray],
    ] = {}
    validation_cache: dict[
        tuple[bytes, float, int],
        tuple[np.ndarray, np.ndarray],
    ] = {}
    best_patches = list(patches)
    best_metrics: SphereFitMetrics | None = None

    while True:
        metrics = _sphere_union_fit_metrics(
            mesh,
            patches,
            tolerance=config.max_sphere_overhang,
            subdivisions=config.sphere_surface_subdivisions,
            cache=validation_cache,
        )
        print(
            f"  spheres={len(patches)} "
            f"max_overhang={metrics.max_overhang:.6g} m "
            f"(limit={metrics.tolerance:.6g} m)"
        )
        if (
            best_metrics is None
            or _sphere_fit_key(metrics, patches)
            < _sphere_fit_key(best_metrics, best_patches)
        ):
            best_patches = list(patches)
            best_metrics = metrics
        if metrics.tolerance_met:
            spheres = _sphere_proxies(patches)
            _validate_sphere_coverage(mesh, spheres)
            return spheres, metrics
        if len(patches) >= config.max_spheres:
            break

        best_refinement: tuple[
            tuple[float, int, float, float, float],
            list[_SpherePatch],
        ] | None = None
        for patch_index, patch in enumerate(patches):
            for lower_polygons, upper_polygons in _sphere_patch_splits(
                patch,
                ellipsoid,
            ):
                lower = _fit_sphere_patch(lower_polygons, config.margin)
                upper = _fit_sphere_patch(upper_polygons, config.margin)
                if (
                    max(lower.radius, upper.radius)
                    >= patch.radius - 1e-9
                ):
                    continue
                refined = [
                    *patches[:patch_index],
                    lower,
                    upper,
                    *patches[patch_index + 1 :],
                ]
                coarse_metrics = _sphere_union_fit_metrics(
                    mesh,
                    refined,
                    tolerance=config.max_sphere_overhang,
                    subdivisions=selection_subdivisions,
                    cache=selection_cache,
                )
                key = _sphere_fit_key(coarse_metrics, refined)
                if best_refinement is None or key < best_refinement[0]:
                    best_refinement = (key, refined)
        if best_refinement is None:
            break
        patches = best_refinement[1]

    if best_metrics is None:
        raise AssertionError("Sphere fitting produced no candidate")
    spheres = _sphere_proxies(best_patches)
    _validate_sphere_coverage(mesh, spheres)
    print(
        "warning: sphere overhang limit was not reached within the "
        f"{config.max_spheres}-sphere budget; using {len(spheres)} spheres "
        f"with max overhang {best_metrics.max_overhang:.6g} m",
        file=sys.stderr,
    )
    return spheres, best_metrics


def _validate_sphere_coverage(
    mesh: trimesh.Trimesh,
    spheres: Sequence[SphereProxy],
    tolerance: float = 1e-9,
) -> None:
    """Raise when any complete triangle is not covered by the sphere union."""
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
    """Conservatively test union coverage by recursive triangle subdivision."""
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
