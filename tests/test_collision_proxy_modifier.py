from __future__ import annotations

import tempfile
import unittest

from pathlib import Path

import numpy as np

from sysid.collision_proxy_generator import (
    GeneratorConfig,
    generate_collision_proxies,
    load_collision_proxies,
    save_collision_proxies,
)
from sysid.collision_proxy_generator.collision_proxy_modifier import (
    ADD_SPHERE_BUTTON,
    BODY_BUTTON_PREFIX,
    CENTER_X_SLIDER,
    DELETE_SPHERE_BUTTON,
    EXIT_EDITOR_BUTTON,
    MODIFICATION_MODE_SLIDER,
    RADIUS_SLIDER,
    SAVE_CHANGES_BUTTON,
    SPHERE_INDEX_SLIDER,
    SphereProxyEditor,
)


SYNTHETIC_URDF = """\
<?xml version="1.0"?>
<robot name="editor_test">
  <link name="base">
    <collision>
      <geometry><box size="0.10 0.08 0.06"/></geometry>
    </collision>
  </link>
  <link name="tool">
    <collision>
      <geometry><sphere radius="0.03"/></geometry>
    </collision>
  </link>
  <joint name="tool_mount" type="fixed">
    <parent link="base"/>
    <child link="tool"/>
    <origin xyz="0 0 0.1"/>
  </joint>
</robot>
"""


class _FakeMeshcat:
    def __init__(self) -> None:
        self.paths: set[str] = set()
        self.objects: dict[str, object] = {}
        self.colors: dict[str, object] = {}
        self.properties: dict[tuple[str, str], object] = {}
        self.transforms: dict[str, np.ndarray] = {}
        self.sliders: dict[str, float] = {}
        self.buttons: dict[str, int] = {}

    def Delete(self, path: str) -> None:
        self.paths = {
            existing
            for existing in self.paths
            if not existing.startswith(path)
        }
        self.objects = {
            existing: value
            for existing, value in self.objects.items()
            if not existing.startswith(path)
        }
        self.colors = {
            existing: value
            for existing, value in self.colors.items()
            if not existing.startswith(path)
        }

    def SetTriangleMesh(self, path, vertices, faces, rgba) -> None:
        self.paths.add(path)

    def SetTransform(self, path, transform) -> None:
        self.paths.add(path)
        self.transforms[path] = np.asarray(transform.GetAsMatrix4())

    def SetObject(self, path, shape, rgba) -> None:
        self.paths.add(path)
        self.objects[path] = shape
        self.colors[path] = rgba

    def SetProperty(self, path, property_name, value) -> None:
        self.properties[(path, property_name)] = value

    def AddSlider(self, name, min, max, step, value) -> float:
        self.sliders[name] = float(np.clip(value, min, max))
        return self.sliders[name]

    def DeleteSlider(self, name, strict=False) -> bool:
        return self.sliders.pop(name, None) is not None

    def GetSliderValue(self, name) -> float:
        return self.sliders[name]

    def SetSliderValue(self, name, value) -> None:
        self.sliders[name] = float(value)

    def AddButton(self, name, keycode="") -> None:
        self.buttons[name] = 0

    def DeleteButton(self, name, strict=False) -> bool:
        return self.buttons.pop(name, None) is not None

    def GetButtonClicks(self, name) -> int:
        return self.buttons[name]

    def click(self, name: str) -> None:
        self.buttons[name] += 1

    def web_url(self) -> str:
        return "http://example.invalid"


class CollisionProxyModifierTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.root = Path(self.temporary_directory.name)
        urdf_path = self.root / "editor_test.urdf"
        urdf_path.write_text(SYNTHETIC_URDF, encoding="utf-8")
        result = generate_collision_proxies(
            "editor_test",
            urdf_path=urdf_path,
            config=GeneratorConfig(
                max_spheres=2,
                max_capsules=1,
                max_sphere_overhang=0.2,
            ),
            mode="spheres",
        )
        self.sphere_path = save_collision_proxies(
            result,
            self.root / "collisions",
            mode="spheres",
        )["spheres"]

    def test_editor_modifies_adds_deletes_and_saves_on_button(self) -> None:
        original_text = self.sphere_path.read_text(encoding="utf-8")
        result = load_collision_proxies(sphere_path=self.sphere_path)
        meshcat = _FakeMeshcat()
        editor = SphereProxyEditor(
            result,
            self.sphere_path,
            meshcat=meshcat,
        )
        self.assertNotIn(SPHERE_INDEX_SLIDER, meshcat.sliders)
        self.assertNotIn(CENTER_X_SLIDER, meshcat.sliders)
        self.assertNotIn(RADIUS_SLIDER, meshcat.sliders)

        meshcat.sliders[MODIFICATION_MODE_SLIDER] = 1.0
        self.assertTrue(editor.poll_once())
        self.assertTrue(editor.modification_mode)
        self.assertNotIn(SPHERE_INDEX_SLIDER, meshcat.sliders)

        meshcat.click(f"{BODY_BUTTON_PREFIX}tool")
        self.assertTrue(editor.poll_once())
        self.assertEqual(editor.selected_group.name, "tool")
        self.assertIn(SPHERE_INDEX_SLIDER, meshcat.sliders)
        self.assertIn(CENTER_X_SLIDER, meshcat.sliders)
        self.assertIn(RADIUS_SLIDER, meshcat.sliders)
        self.assertFalse(
            meshcat.properties[
                ("/collision_proxy/bodies/base", "visible")
            ]
        )
        self.assertTrue(
            meshcat.properties[
                ("/collision_proxy/bodies/tool", "visible")
            ]
        )
        selected_path = (
            "/collision_proxy/bodies/tool/spheres/sphere_00"
        )
        selected_alpha = meshcat.colors[selected_path].a()
        self.assertGreater(selected_alpha, editor.proxy_opacity)
        self.assertLess(selected_alpha, 0.8)

        original_center = editor.selected_sphere.center
        original_radius = editor.selected_sphere.radius
        meshcat.sliders[CENTER_X_SLIDER] = original_center[0] + 0.01
        meshcat.sliders[RADIUS_SLIDER] = original_radius + 0.005
        self.assertTrue(editor.poll_once())
        self.assertAlmostEqual(
            editor.selected_sphere.center[0],
            original_center[0] + 0.01,
        )
        self.assertAlmostEqual(
            editor.selected_sphere.radius,
            original_radius + 0.005,
        )
        self.assertIsNone(editor.selected_group.sphere_fit)
        self.assertTrue(editor.dirty)
        self.assertEqual(
            self.sphere_path.read_text(encoding="utf-8"),
            original_text,
        )

        sphere_count = len(editor.selected_group.spheres)
        meshcat.click(ADD_SPHERE_BUTTON)
        self.assertTrue(editor.poll_once())
        self.assertEqual(
            len(editor.selected_group.spheres),
            sphere_count + 1,
        )
        meshcat.click(DELETE_SPHERE_BUTTON)
        self.assertTrue(editor.poll_once())
        self.assertEqual(
            len(editor.selected_group.spheres),
            sphere_count,
        )

        meshcat.click(SAVE_CHANGES_BUTTON)
        self.assertTrue(editor.poll_once())
        self.assertFalse(editor.dirty)
        self.assertNotEqual(
            self.sphere_path.read_text(encoding="utf-8"),
            original_text,
        )
        loaded = load_collision_proxies(sphere_path=self.sphere_path)
        loaded_tool = next(
            group for group in loaded.groups if group.name == "tool"
        )
        self.assertEqual(
            loaded_tool.spheres,
            editor.selected_group.spheres,
        )
        self.assertIsNone(loaded_tool.sphere_fit)
        loaded_base = next(
            group for group in loaded.groups if group.name == "base"
        )
        self.assertIsNotNone(loaded_base.sphere_fit)

        meshcat.sliders[MODIFICATION_MODE_SLIDER] = 0.0
        self.assertTrue(editor.poll_once())
        self.assertNotIn(SPHERE_INDEX_SLIDER, meshcat.sliders)
        self.assertNotIn(CENTER_X_SLIDER, meshcat.sliders)
        for body in ("base", "tool"):
            self.assertTrue(
                meshcat.properties[
                    (f"/collision_proxy/bodies/{body}", "visible")
                ]
            )

        meshcat.click(EXIT_EDITOR_BUTTON)
        self.assertFalse(editor.poll_once())
        editor.close()
        self.assertFalse(meshcat.buttons)
        self.assertFalse(meshcat.sliders)

    def test_editor_refuses_to_delete_the_only_sphere(self) -> None:
        result = load_collision_proxies(sphere_path=self.sphere_path)
        editor = SphereProxyEditor(
            result,
            self.sphere_path,
            meshcat=_FakeMeshcat(),
        )
        self.assertEqual(len(editor.selected_group.spheres), 1)
        self.assertFalse(editor.delete_sphere())
        self.assertFalse(editor.dirty)
        editor.close()


if __name__ == "__main__":
    unittest.main()
