"""Conservative capsule fitting for collision meshes.

Capsules and input mesh vertices use the collision group's anchor-link frame.
Candidates are ranked by total primitive volume, then explicitly checked for
triangle-vertex coverage before they are returned.
"""

from __future__ import annotations

import math

from typing import Sequence

import numpy as np
import trimesh

from .proxy_types import (
    CapsuleProxy,
    EllipsoidProxy,
    GeneratorConfig,
    _vector3,
)


def _face_clusters(mesh: trimesh.Trimesh, count: int) -> tuple[np.ndarray, ...]:
    """Deterministically bisect face centroids into at most ``count`` groups."""
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
        # Split along the direction with the largest centroid variation.
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
    """Return the unique mesh vertices referenced by a face cluster."""
    vertex_indices = np.unique(np.asarray(mesh.faces)[face_indices].reshape(-1))
    return np.asarray(mesh.vertices)[vertex_indices]


def _point_segment_distances(
    points: np.ndarray,
    point_a: np.ndarray,
    point_b: np.ndarray,
) -> np.ndarray:
    """Compute Euclidean distances from points to a finite line segment."""
    segment = point_b - point_a
    length_squared = float(segment @ segment)
    if length_squared <= 1e-24:
        return np.linalg.norm(points - point_a[None, :], axis=1)
    parameter = ((points - point_a[None, :]) @ segment) / length_squared
    parameter = np.clip(parameter, 0.0, 1.0)
    witnesses = point_a[None, :] + parameter[:, None] * segment[None, :]
    return np.linalg.norm(points - witnesses, axis=1)


def _fit_capsule(points: np.ndarray, margin: float) -> CapsuleProxy:
    """Fit a capsule around points using Trimesh's minimum cylinder axis."""
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
        # Degenerate or numerically difficult clusters fall back to a PCA axis.
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
    """Choose a small conservative capsule set within the configured budget."""
    if ellipsoid.elongated:
        capsules = (
            _fit_axis_aligned_capsule(mesh, ellipsoid, config.margin),
        )
        _validate_capsule_coverage(mesh, capsules)
        return capsules

    candidates: list[tuple[CapsuleProxy, ...]] = []
    # Evaluate every allowed primitive count so precision can favor a simpler
    # candidate whose volume is close to the best-volume candidate.
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
    """Return cylinder-plus-two-hemispheres volume for ranking candidates."""
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
    """Use total capsule volume as the approximation overhang surrogate."""
    return float(sum(_capsule_volume(capsule) for capsule in capsules))


def _fit_axis_aligned_capsule(
    mesh: trimesh.Trimesh,
    ellipsoid: EllipsoidProxy,
    margin: float,
) -> CapsuleProxy:
    """Fit a compact capsule constrained to the mesh ellipsoid's major axis."""
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
        raise ValueError("Could not fit a mesh-aligned capsule")
    return best


def _validate_capsule_coverage(
    mesh: trimesh.Trimesh,
    capsules: Sequence[CapsuleProxy],
    tolerance: float = 1e-9,
) -> None:
    """Raise when no capsule contains all three vertices of a triangle."""
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
