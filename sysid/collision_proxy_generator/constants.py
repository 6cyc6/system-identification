"""Shared names used across collision-proxy generation and visualization.

This module intentionally contains no fitting or I/O logic. Keeping filenames,
schema versions, and Meshcat control names here prevents the generator,
visualizer, and editor from silently drifting apart.
"""

from sysid.utils.path_utils import BASE_DIR, get_config_path


ROBOT_DESCRIPTION_DIR = BASE_DIR / "robot_description"
DEFAULT_CONFIG_PATH = get_config_path("fr3", "collision_proxy.yaml")

# Version 2 stores the mesh-derived ellipsoid and sphere-fit quality metrics.
SCHEMA_VERSION = 2
SUPPORTED_SCHEMA_VERSIONS = frozenset((1, SCHEMA_VERSION))

# These filenames form the persisted collision-proxy interface.
SPHERE_FILENAME = "sphere_collisions.yaml"
CAPSULE_FILENAME = "capsule_collisions.yaml"
ELLIPSOID_FILENAME = "ellipsoid_collisions.yaml"

# Connected links containing these tokens are merged into one gripper body.
GRIPPER_LINK_TOKENS = ("hand", "gripper", "finger")
PROXY_KINDS = ("spheres", "capsules", "ellipsoids")
PROXY_MODES = ("both", "all", *PROXY_KINDS)

# All viewer objects live under one root so a refresh can delete them safely.
MESHCAT_ROOT_PATH = "/collision_proxy"
PROXY_OPACITY_SLIDER = "Proxy opacity"
JOINT_SLIDER_PREFIX = "Joint position: "
BODY_VIEW_BUTTON_PREFIX = "Show body: "
SHOW_ALL_BODIES_BUTTON = "Show all bodies"
