"""Collision-proxy generation, persistence, editing, and visualization."""

from .collision_proxy_generator import (
    generate_collision_proxies,
    generate_configured_collision_proxies,
)
from .constants import (
    CAPSULE_FILENAME,
    DEFAULT_CONFIG_PATH,
    ELLIPSOID_FILENAME,
    ROBOT_DESCRIPTION_DIR,
    SPHERE_FILENAME,
)
from .meshcat_visualizer import (
    visualize_collision_model,
    visualize_collision_proxies,
)
from .proxy_io import (
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
)
from .urdf_geometry import (
    default_output_dir,
    resolve_robot_urdf,
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
