"""Mesh-derived conservative ellipsoid fitting.

The mesh moments determine the center and principal-axis orientation. The
semiaxes are then enlarged—without changing that frame—until every collision
vertex is inside. No URDF inertial values are used by this module.
"""

from __future__ import annotations

import math

from dataclasses import dataclass

import numpy as np
import trimesh

from scipy.optimize import minimize

from .proxy_types import (
    EllipsoidProxy,
    GeneratorConfig,
    _matrix3,
    _vector3,
)


@dataclass(frozen=True)
class _MeshEllipsoidEstimate:
    """Uninflated principal-axis estimate computed from mesh moments."""

    center: np.ndarray
    radii: np.ndarray
    rotation: np.ndarray


def _ellipsoid_from_covariance(
    center: np.ndarray,
    covariance: np.ndarray,
) -> _MeshEllipsoidEstimate:
    """Return the equivalent uniform-solid ellipsoid for mesh moments."""
    covariance = 0.5 * (
        np.asarray(covariance, dtype=float)
        + np.asarray(covariance, dtype=float).T
    )
    eigenvalues, rotation = np.linalg.eigh(covariance)
    scale = max(float(np.trace(covariance)), 1e-18)
    if (
        np.asarray(center).shape != (3,)
        or not np.all(np.isfinite(center))
        or not np.all(np.isfinite(eigenvalues))
        or eigenvalues.min() <= scale * 1e-12
    ):
        raise ValueError("Mesh moments do not define a solid ellipsoid")

    # A uniform solid ellipsoid has covariance diag(a², b², c²) / 5.
    radii = np.sqrt(5.0 * eigenvalues)
    order = np.argsort(radii)[::-1]
    radii = radii[order]
    rotation = rotation[:, order]
    if np.linalg.det(rotation) < 0.0:
        rotation[:, -1] *= -1.0
    return _MeshEllipsoidEstimate(
        center=np.asarray(center, dtype=float),
        radii=radii,
        rotation=rotation,
    )


def _estimate_mesh_ellipsoid(
    mesh: trimesh.Trimesh,
) -> _MeshEllipsoidEstimate:
    """Estimate an ellipsoid solely from the collision mesh geometry."""
    candidates = (mesh, mesh.convex_hull)
    for candidate in candidates:
        try:
            properties = candidate.mass_properties
            mass = abs(float(properties.mass))
            center = np.asarray(properties.center_mass, dtype=float)
            inertia = np.asarray(properties.inertia, dtype=float)
            if float(properties.mass) < 0.0:
                inertia = -inertia
            inertia = 0.5 * (inertia + inertia.T)
            if (
                mass > 1e-12
                and np.all(np.isfinite(center))
                and np.all(np.isfinite(inertia))
            ):
                covariance = (
                    0.5 * float(np.trace(inertia)) * np.eye(3)
                    - inertia
                ) / mass
                return _ellipsoid_from_covariance(
                    center,
                    covariance,
                )
        except Exception:
            continue

    points = np.asarray(mesh.vertices, dtype=float)
    center = points.mean(axis=0)
    centered = points - center[None, :]
    covariance = centered.T @ centered / max(1, len(centered))
    scale = max(float(np.max(np.ptp(points, axis=0))) ** 2, 1e-18)
    covariance += np.eye(3) * scale * 1e-12
    return _ellipsoid_from_covariance(center, covariance)


def _mesh_ellipsoid_proxy(
    mesh: trimesh.Trimesh,
    config: GeneratorConfig,
) -> EllipsoidProxy:
    """Fit, conservatively enlarge, and validate one mesh ellipsoid."""
    estimate = _estimate_mesh_ellipsoid(mesh)
    local_vertices = (
        np.asarray(mesh.vertices) - estimate.center[None, :]
    ) @ estimate.rotation
    radii = _minimum_enlarged_ellipsoid_radii(
        local_vertices,
        estimate.radii,
    )
    radii += config.margin

    # The independent conservative enlargement can change the axis order.
    order = np.argsort(radii)[::-1]
    radii = radii[order]
    rotation = estimate.rotation[:, order]
    if np.linalg.det(rotation) < 0.0:
        rotation[:, -1] *= -1.0

    axis_ratio = float(radii[0] / radii[1])
    ellipsoid = EllipsoidProxy(
        center=_vector3(estimate.center),
        radii=_vector3(radii),
        rotation=_matrix3(rotation),
        axis_ratio=axis_ratio,
        elongated=axis_ratio >= config.elongation_threshold,
    )
    _validate_ellipsoid_coverage(mesh, ellipsoid)
    return ellipsoid


def _minimum_enlarged_ellipsoid_radii(
    local_vertices: np.ndarray,
    estimated_radii: np.ndarray,
) -> np.ndarray:
    """Minimize volume while only enlarging mesh-estimated semiaxes."""
    normalized_squared = (
        np.asarray(local_vertices, dtype=float)
        / np.asarray(estimated_radii, dtype=float)[None, :]
    ) ** 2
    uniform_factor = max(
        1.0,
        float(normalized_squared.sum(axis=1).max()),
    )
    initial = np.full(3, 1.0 / uniform_factor)

    # y_i = (estimated_radius_i / enlarged_radius_i)^2. Maximizing
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
    radii = np.asarray(estimated_radii, dtype=float) / np.sqrt(values)

    # Absorb numerical optimizer tolerance conservatively.
    required_scale = float(
        np.linalg.norm(
            np.asarray(local_vertices) / radii[None, :],
            axis=1,
        ).max()
    )
    return radii * max(1.0, required_scale)


def _validate_ellipsoid_coverage(
    mesh: trimesh.Trimesh,
    ellipsoid: EllipsoidProxy,
    tolerance: float = 1e-9,
) -> None:
    """Raise when a mesh vertex lies outside the fitted ellipsoid."""
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
