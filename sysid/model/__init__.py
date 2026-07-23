"""Robot-model construction and collision approximation tools."""

from .collision_approximation_generator import (
    CapsuleApproximation,
    CollisionApproximationResult,
    GeneratorConfig,
    LinkApproximation,
    SphereApproximation,
    generate_collision_approximations,
    load_collision_approximations,
    save_collision_approximations,
    visualize_collision_approximations,
)

__all__ = [
    "CapsuleApproximation",
    "CollisionApproximationResult",
    "GeneratorConfig",
    "LinkApproximation",
    "SphereApproximation",
    "generate_collision_approximations",
    "load_collision_approximations",
    "save_collision_approximations",
    "visualize_collision_approximations",
]
