"""FR3 workspace and camera-obstacle geometry helpers."""

import numpy as np

from .collision_model import (
    CAMERA_BOX_SPECS_MM,
    CameraBox,
    DrakeCameraCollisionChecker,
)


def _body_at_or_after_link(body_name, min_link_index):
    if body_name in {"base", "world"}:
        return False
    if "link" in body_name:
        suffix = body_name.rsplit("link", 1)[-1]
        digits = "".join(char for char in suffix if char.isdigit())
        if digits:
            return int(digits) >= int(min_link_index)
    return int(min_link_index) <= 2


def _sample_points_and_radii(collision_checker, q_i, *, min_link_index):
    """Transform selected link collision spheres into world coordinates."""
    if not hasattr(collision_checker, "_robot_sample_specs"):
        points = collision_checker._robot_sample_points(q_i)
        radii = np.full(len(points), collision_checker.robot_sphere_radius)
        return points, radii

    collision_checker.plant.SetPositions(
        collision_checker.plant_context,
        collision_checker.model_instance,
        q_i,
    )
    points = []
    radii = []
    for body, offset, radius in collision_checker._robot_sample_specs:
        if not _body_at_or_after_link(body.name(), min_link_index):
            continue
        x_wb = collision_checker.plant.EvalBodyPoseInWorld(
            collision_checker.plant_context,
            body,
        )
        points.append(x_wb.multiply(offset))
        radii.append(float(radius))

    if not points:
        points = collision_checker._robot_sample_points(q_i)
        radii = np.full(len(points), collision_checker.robot_sphere_radius)
        return points, radii
    return np.asarray(points, dtype=float), np.asarray(radii, dtype=float)


def robot_link_workspace_margins(
    collision_checker,
    q,
    *,
    x_lower,
    y_lower,
    y_upper,
    z_lower,
):
    """Compute clearance from the x/y/z workspace walls for each robot pose.

    Each output row is ``[x - x_lower, y - y_lower, y_upper - y, z - z_lower]``.
    Sphere radii are included, so non-negative values keep the full modeled
    robot volume inside the workspace.
    """
    q = np.asarray(q, dtype=float)
    if q.ndim == 1:
        q = q.reshape(1, -1)

    margins = []
    for q_i in q:
        # Link 1 and later participate in x/y walls. The z wall starts at link 2
        # so that the fixed base/link-1 region may remain below it.
        xy_points, xy_radii = _sample_points_and_radii(
            collision_checker,
            q_i,
            min_link_index=1,
        )
        z_points, z_radii = _sample_points_and_radii(
            collision_checker,
            q_i,
            min_link_index=2,
        )
        margins.append(
            (
                float(np.min(xy_points[:, 0] - xy_radii)) - float(x_lower),
                float(np.min(xy_points[:, 1] - xy_radii)) - float(y_lower),
                float(y_upper) - float(np.max(xy_points[:, 1] + xy_radii)),
                float(np.min(z_points[:, 2] - z_radii)) - float(z_lower),
            )
        )
    return np.asarray(margins, dtype=float)


def _scaled_camera_boxes(*, xy_prism_height, xy_scale, z_scale):
    """Convert measured camera boxes from millimeters to padded meter boxes."""
    boxes = []
    for name, center_mm, size_mm in CAMERA_BOX_SPECS_MM:
        center = np.asarray(center_mm, dtype=float) / 1000.0
        size = np.asarray(size_mm, dtype=float) / 1000.0
        size[:2] *= float(xy_scale)
        if xy_prism_height is None:
            size[2] *= float(z_scale)
        else:
            size[2] = float(xy_prism_height) * float(z_scale)
        boxes.append(CameraBox(name=name, center=center, size=size))
    return boxes


def build_collision_checker(config):
    """Construct the Drake collision checker described by the run config."""
    xy_prism_height = (
        None
        if config.drake_physical_camera_height
        else config.drake_xy_prism_height
    )
    camera_boxes = _scaled_camera_boxes(
        xy_prism_height=xy_prism_height,
        xy_scale=config.camera_box_xy_scale,
        z_scale=config.camera_box_z_scale,
    )
    return DrakeCameraCollisionChecker(
        robot_name=config.robot,
        robot_urdf_path=config.robot_urdf_path,
        min_distance=config.drake_min_distance,
        robot_sphere_radius=config.drake_robot_sphere_radius,
        robot_link_samples=config.drake_robot_link_samples,
        camera_boxes=camera_boxes,
        camera_chamfer_radius=config.drake_camera_chamfer_radius,
    )
