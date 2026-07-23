from __future__ import annotations

import tempfile
import unittest

from pathlib import Path

import numpy as np
import yaml

from sysid.model.collision_approximation_generator import (
    GeneratorConfig,
    generate_collision_approximations,
    load_collision_approximations,
    resolve_robot_urdf,
    save_collision_approximations,
    visualize_collision_approximations,
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


class _FakeMeshcat:
    def __init__(self) -> None:
        self.paths: set[str] = set()
        self.properties: dict[tuple[str, str], object] = {}

    def Delete(self, path: str) -> None:
        self.paths = {existing for existing in self.paths if not existing.startswith(path)}

    def SetTriangleMesh(self, path, vertices, faces, rgba) -> None:
        self.paths.add(path)
        self.last_vertices = vertices
        self.last_faces = faces

    def SetTransform(self, path, transform) -> None:
        self.paths.add(path)

    def SetObject(self, path, shape, rgba) -> None:
        self.paths.add(path)

    def SetProperty(self, path, property_name, value) -> None:
        self.properties[(path, property_name)] = value

    def web_url(self) -> str:
        return "http://example.invalid"


class CollisionApproximationGeneratorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.root = Path(self.temporary_directory.name)
        self.urdf_path = self.root / "test_robot.urdf"
        self.urdf_path.write_text(SYNTHETIC_URDF, encoding="utf-8")

    def _generate(self):
        return generate_collision_approximations(
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
            self.assertTrue(all(sphere.radius > 0.0 for sphere in group.spheres))
            self.assertTrue(all(capsule.radius > 0.0 for capsule in group.capsules))

    def test_yaml_round_trip_is_deterministic(self) -> None:
        result = self._generate()
        output = self.root / "collisions"
        paths = save_collision_approximations(result, output)
        first_spheres = paths["spheres"].read_text(encoding="utf-8")
        first_capsules = paths["capsules"].read_text(encoding="utf-8")
        save_collision_approximations(result, output)
        self.assertEqual(
            paths["spheres"].read_text(encoding="utf-8"),
            first_spheres,
        )
        self.assertEqual(
            paths["capsules"].read_text(encoding="utf-8"),
            first_capsules,
        )

        loaded = load_collision_approximations(
            paths["spheres"],
            paths["capsules"],
        )
        self.assertEqual(loaded.robot, result.robot)
        self.assertEqual(loaded.groups, result.groups)
        document = yaml.safe_load(first_spheres)
        self.assertEqual(document["schema_version"], 1)
        self.assertEqual(document["units"], "m")

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

    def test_meshcat_scene_contains_source_spheres_and_capsules(self) -> None:
        result = self._generate()
        meshcat = _FakeMeshcat()
        returned = visualize_collision_approximations(
            result,
            initial_view="spheres",
            hold=False,
            meshcat=meshcat,
        )
        self.assertIs(returned, meshcat)
        self.assertTrue(
            any(path.startswith("/collision_approximation/source/base") for path in meshcat.paths)
        )
        self.assertTrue(
            any(
                path.startswith("/collision_approximation/spheres/tool_hand")
                for path in meshcat.paths
            )
        )
        self.assertTrue(
            any(
                path.startswith("/collision_approximation/capsules/tool_hand")
                for path in meshcat.paths
            )
        )
        self.assertTrue(
            meshcat.properties[("/collision_approximation/spheres", "visible")]
        )
        self.assertFalse(
            meshcat.properties[("/collision_approximation/capsules", "visible")]
        )


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
