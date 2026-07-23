from __future__ import annotations

import tempfile
import unittest

from dataclasses import replace
from pathlib import Path

import numpy as np
import yaml

from sysid.collision_proxy_generator import (
    DEFAULT_CONFIG_PATH,
    CollisionProxyConfig,
    GeneratorConfig,
    generate_collision_proxies,
    load_collision_model,
    load_collision_proxies,
    resolve_robot_urdf,
    save_collision_proxies,
    visualize_collision_proxies,
)
from sysid.collision_proxy_generator.collision_proxy_generator import (
    _add_joint_sliders,
    _joint_slider_specs,
    _read_joint_sliders,
    _read_robot_geometry,
    _update_meshcat_transforms,
)


SYNTHETIC_URDF = """\
<?xml version="1.0"?>
<robot name="test_robot">
  <link name="base">
    <collision>
      <origin xyz="0.01 0 0"/>
      <geometry><box size="0.10 0.08 0.06"/></geometry>
    </collision>
  </link>
  <link name="tool_hand">
    <collision>
      <geometry><box size="0.08 0.06 0.04"/></geometry>
    </collision>
  </link>
  <link name="left_finger">
    <collision>
      <geometry><box size="0.02 0.02 0.08"/></geometry>
    </collision>
  </link>
  <link name="right_finger">
    <collision>
      <geometry><box size="0.02 0.02 0.08"/></geometry>
    </collision>
  </link>
  <joint name="hand_mount" type="fixed">
    <parent link="base"/>
    <child link="tool_hand"/>
    <origin xyz="0 0 0.10"/>
  </joint>
  <joint name="left_finger_joint" type="prismatic">
    <parent link="tool_hand"/>
    <child link="left_finger"/>
    <origin xyz="0 0 0.04"/>
    <axis xyz="0 1 0"/>
    <limit lower="0" upper="0.04" effort="1" velocity="1"/>
  </joint>
  <joint name="right_finger_joint" type="prismatic">
    <parent link="tool_hand"/>
    <child link="right_finger"/>
    <origin xyz="0 0 0.04"/>
    <axis xyz="0 -1 0"/>
    <limit lower="0" upper="0.04" effort="1" velocity="1"/>
    <mimic joint="left_finger_joint"/>
  </joint>
</robot>
"""

ELONGATED_URDF = """\
<?xml version="1.0"?>
<robot name="elongated_robot">
  <link name="long_link">
    <inertial>
      <origin xyz="0 0 0"/>
      <mass value="1"/>
      <inertia
        ixx="0.0002666666666666667"
        ixy="0"
        ixz="0"
        iyy="0.013466666666666667"
        iyz="0"
        izz="0.013466666666666667"/>
    </inertial>
    <collision>
      <geometry><box size="0.40 0.04 0.04"/></geometry>
    </collision>
  </link>
</robot>
"""


class _FakeMeshcat:
    def __init__(self) -> None:
        self.paths: set[str] = set()
        self.properties: dict[tuple[str, str], object] = {}
        self.object_rgba: dict[str, object] = {}
        self.sliders: dict[str, float] = {}
        self.transforms: dict[str, list[np.ndarray]] = {}

    def Delete(self, path: str) -> None:
        self.paths = {existing for existing in self.paths if not existing.startswith(path)}

    def SetTriangleMesh(self, path, vertices, faces, rgba) -> None:
        self.paths.add(path)
        self.last_vertices = vertices
        self.last_faces = faces
        self.last_triangle_mesh_rgba = rgba

    def SetTransform(self, path, transform) -> None:
        self.paths.add(path)
        self.transforms.setdefault(path, []).append(
            np.asarray(transform.GetAsMatrix4())
        )

    def SetObject(self, path, shape, rgba) -> None:
        self.paths.add(path)
        self.object_rgba[path] = rgba

    def SetProperty(self, path, property_name, value) -> None:
        self.properties[(path, property_name)] = value

    def AddSlider(self, name, min, max, step, value) -> float:
        self.sliders[name] = float(np.clip(value, min, max))
        return self.sliders[name]

    def GetSliderValue(self, name) -> float:
        return self.sliders[name]

    def web_url(self) -> str:
        return "http://example.invalid"


class CollisionProxyGeneratorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.root = Path(self.temporary_directory.name)
        self.urdf_path = self.root / "test_robot.urdf"
        self.urdf_path.write_text(SYNTHETIC_URDF, encoding="utf-8")

    def _generate(self):
        return generate_collision_proxies(
            "test_robot",
            urdf_path=self.urdf_path,
            config=GeneratorConfig(
                max_spheres=2,
                max_capsules=2,
                margin=0.001,
                voxel_divisions=8,
                precision=0.6,
                min_radius_vox=1,
                min_center_distance_vox=1,
            ),
        )

    def test_default_cli_config_loads_from_yaml(self) -> None:
        document = yaml.safe_load(
            DEFAULT_CONFIG_PATH.read_text(encoding="utf-8")
        )
        config = CollisionProxyConfig.from_yaml(DEFAULT_CONFIG_PATH)
        self.assertEqual(config.robot, "fr3")
        self.assertEqual(config.mode, document["mode"])
        self.assertEqual(config.max_spheres, document["max_spheres"])
        self.assertEqual(config.max_capsules, document["max_capsules"])
        self.assertEqual(config.group_overrides, {})
        self.assertEqual(config.joint_position_overrides, {})
        self.assertEqual(config.proxy_opacity, document["proxy_opacity"])

    def test_cli_config_rejects_unknown_yaml_keys(self) -> None:
        document = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
        document["unexpected_setting"] = True
        config_path = self.root / "invalid_config.yaml"
        config_path.write_text(yaml.safe_dump(document), encoding="utf-8")

        with self.assertRaisesRegex(
            ValueError,
            "Unknown configuration key.*unexpected_setting",
        ):
            CollisionProxyConfig.from_yaml(config_path)

    def test_fr3_resolver_prefers_gripper_urdf(self) -> None:
        path = resolve_robot_urdf("fr3")
        self.assertEqual(path.name, "fr3_gripper.urdf")
        self.assertEqual(path.parent.name, "urdf")

    def test_generation_groups_gripper_at_maximum_position(self) -> None:
        result = self._generate()
        self.assertEqual([group.name for group in result.groups], ["base", "tool_hand"])
        hand = result.groups[1]
        self.assertEqual(
            hand.source_links,
            ("tool_hand", "left_finger", "right_finger"),
        )
        self.assertEqual(result.joint_positions["left_finger_joint"], 0.04)
        self.assertEqual(result.joint_positions["right_finger_joint"], 0.04)
        for group in result.groups:
            self.assertLessEqual(len(group.spheres), 2)
            self.assertLessEqual(len(group.capsules), 2)
            self.assertEqual(len(group.ellipsoids), 1)
            self.assertTrue(all(sphere.radius > 0.0 for sphere in group.spheres))
            self.assertTrue(all(capsule.radius > 0.0 for capsule in group.capsules))
            self.assertTrue(
                all(
                    radius > 0.0
                    for ellipsoid in group.ellipsoids
                    for radius in ellipsoid.radii
                )
            )

    def test_yaml_round_trip_is_deterministic(self) -> None:
        result = self._generate()
        output = self.root / "collisions"
        paths = save_collision_proxies(result, output, mode="all")
        first_spheres = paths["spheres"].read_text(encoding="utf-8")
        first_capsules = paths["capsules"].read_text(encoding="utf-8")
        first_ellipsoids = paths["ellipsoids"].read_text(encoding="utf-8")
        save_collision_proxies(result, output, mode="all")
        self.assertEqual(
            paths["spheres"].read_text(encoding="utf-8"),
            first_spheres,
        )
        self.assertEqual(
            paths["capsules"].read_text(encoding="utf-8"),
            first_capsules,
        )
        self.assertEqual(
            paths["ellipsoids"].read_text(encoding="utf-8"),
            first_ellipsoids,
        )

        loaded = load_collision_proxies(
            paths["spheres"],
            paths["capsules"],
            paths["ellipsoids"],
        )
        self.assertEqual(loaded.robot, result.robot)
        self.assertEqual(loaded.groups, result.groups)
        document = yaml.safe_load(first_spheres)
        self.assertEqual(document["schema_version"], 1)
        self.assertEqual(document["units"], "m")

    def test_visualizer_loads_saved_collision_model(self) -> None:
        result = self._generate()
        output = self.root / "collisions"
        save_collision_proxies(result, output, mode="all")
        config = replace(
            CollisionProxyConfig.from_yaml(DEFAULT_CONFIG_PATH),
            robot="test_robot",
            output_dir=output,
        )

        loaded = load_collision_model(config)

        self.assertEqual(loaded.robot, result.robot)
        self.assertEqual(loaded.groups, result.groups)

    def test_joint_sliders_update_source_body_poses_and_mimic_joints(self) -> None:
        result = self._generate()
        robot_geometry = _read_robot_geometry(result.source_urdf)
        specs = _joint_slider_specs(
            robot_geometry,
            result.joint_positions,
        )
        self.assertEqual(
            [spec.joint_name for spec in specs],
            ["left_finger_joint"],
        )

        meshcat = _FakeMeshcat()
        visualize_collision_proxies(result, hold=False, meshcat=meshcat)
        left_finger_path = (
            "/collision_proxy/bodies/tool_hand/source/left_finger_00"
        )
        initial_transform = meshcat.transforms[left_finger_path][-1]

        positions = _add_joint_sliders(
            meshcat,
            specs,
            result.joint_positions,
        )
        meshcat.sliders[specs[0].control_name] = 0.0
        positions, changed = _read_joint_sliders(
            meshcat,
            specs,
            robot_geometry,
            positions,
        )
        self.assertTrue(changed)
        self.assertEqual(positions["left_finger_joint"], 0.0)
        self.assertEqual(positions["right_finger_joint"], 0.0)
        _update_meshcat_transforms(
            meshcat,
            result,
            robot_geometry,
            positions,
        )

        moved_transform = meshcat.transforms[left_finger_path][-1]
        self.assertFalse(np.allclose(initial_transform, moved_transform))

    def test_generated_primitives_contain_base_box_triangles(self) -> None:
        result = self._generate()
        base = result.groups[0]
        vertices = np.array(
            [
                [x + 0.01, y, z]
                for x in (-0.05, 0.05)
                for y in (-0.04, 0.04)
                for z in (-0.03, 0.03)
            ]
        )
        for vertex in vertices:
            self.assertTrue(
                any(
                    np.linalg.norm(vertex - np.asarray(sphere.center))
                    <= sphere.radius + 1e-9
                    for sphere in base.spheres
                )
            )
            self.assertTrue(
                any(
                    _point_segment_distance(
                        vertex,
                        np.asarray(capsule.point_a),
                        np.asarray(capsule.point_b),
                    )
                    <= capsule.radius + 1e-9
                    for capsule in base.capsules
                )
            )
            ellipsoid = base.ellipsoids[0]
            local_vertex = (
                vertex - np.asarray(ellipsoid.center)
            ) @ np.asarray(ellipsoid.rotation)
            self.assertLessEqual(
                np.linalg.norm(
                    local_vertex / np.asarray(ellipsoid.radii)
                ),
                1.0 + 1e-9,
            )

    def test_meshcat_scene_contains_source_and_all_proxy_types(self) -> None:
        result = self._generate()
        meshcat = _FakeMeshcat()
        returned = visualize_collision_proxies(
            result,
            initial_view="spheres",
            proxy_opacity=0.1,
            hold=False,
            meshcat=meshcat,
        )
        self.assertIs(returned, meshcat)
        self.assertTrue(
            any(
                path.startswith("/collision_proxy/bodies/base/source")
                for path in meshcat.paths
            )
        )
        self.assertTrue(
            any(
                path.startswith(
                    "/collision_proxy/bodies/tool_hand/ellipsoids"
                )
                for path in meshcat.paths
            )
        )
        self.assertTrue(
            any(
                path.startswith(
                    "/collision_proxy/bodies/tool_hand/spheres"
                )
                for path in meshcat.paths
            )
        )
        self.assertTrue(
            any(
                path.startswith(
                    "/collision_proxy/bodies/tool_hand/capsules"
                )
                for path in meshcat.paths
            )
        )
        self.assertEqual(meshcat.last_triangle_mesh_rgba.a(), 1.0)
        self.assertTrue(meshcat.object_rgba)
        self.assertTrue(
            all(rgba.a() == 0.1 for rgba in meshcat.object_rgba.values())
        )
        for body in ("base", "tool_hand"):
            body_path = f"/collision_proxy/bodies/{body}"
            self.assertTrue(meshcat.properties[(body_path, "visible")])
            self.assertTrue(
                meshcat.properties[(f"{body_path}/source", "visible")]
            )
            self.assertTrue(
                meshcat.properties[(f"{body_path}/spheres", "visible")]
            )
            self.assertFalse(
                meshcat.properties[(f"{body_path}/capsules", "visible")]
            )
            self.assertFalse(
                meshcat.properties[(f"{body_path}/ellipsoids", "visible")]
            )
            for kind in ("spheres", "capsules", "ellipsoids"):
                path = f"{body_path}/{kind}"
                self.assertEqual(
                    meshcat.properties[(path, "opacity")],
                    0.1,
                )
                self.assertTrue(
                    meshcat.properties[(path, "transparent")]
                )

    def test_elongated_inertia_uses_skeleton_and_one_capsule(self) -> None:
        elongated_path = self.root / "elongated_robot.urdf"
        elongated_path.write_text(ELONGATED_URDF, encoding="utf-8")
        result = generate_collision_proxies(
            "elongated_robot",
            urdf_path=elongated_path,
            config=GeneratorConfig(
                max_spheres=5,
                max_capsules=3,
                margin=0.001,
                voxel_divisions=8,
                precision=0.95,
                elongation_threshold=2.0,
                min_radius_vox=1,
                min_center_distance_vox=1,
            ),
            mode="all",
        )
        group = result.groups[0]
        ellipsoid = group.ellipsoids[0]
        self.assertTrue(ellipsoid.elongated)
        self.assertGreaterEqual(ellipsoid.axis_ratio, 2.0)
        self.assertEqual(len(group.capsules), 1)
        self.assertGreater(len(group.spheres), 1)
        self.assertLessEqual(len(group.spheres), 5)

        center = np.asarray(ellipsoid.center)
        major_axis = np.asarray(ellipsoid.rotation)[:, 0]
        for sphere in group.spheres:
            offset = np.asarray(sphere.center) - center
            perpendicular = offset - major_axis * (offset @ major_axis)
            self.assertLess(np.linalg.norm(perpendicular), 1e-10)

        capsule_axis = (
            np.asarray(group.capsules[0].point_b)
            - np.asarray(group.capsules[0].point_a)
        )
        capsule_axis /= np.linalg.norm(capsule_axis)
        self.assertGreater(abs(float(capsule_axis @ major_axis)), 1.0 - 1e-10)


def _point_segment_distance(
    point: np.ndarray,
    point_a: np.ndarray,
    point_b: np.ndarray,
) -> float:
    segment = point_b - point_a
    length_squared = float(segment @ segment)
    if length_squared == 0.0:
        return float(np.linalg.norm(point - point_a))
    parameter = float(np.clip(((point - point_a) @ segment) / length_squared, 0.0, 1.0))
    return float(np.linalg.norm(point - (point_a + parameter * segment)))


if __name__ == "__main__":
    unittest.main()
