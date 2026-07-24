"""Interactively edit saved sphere collision proxies in Drake Meshcat.

Run with::

    pixi run python script/fr3/generate_collision/modify_collision_proxy.py

The editor only writes to disk when the user presses ``Save changes``.
Manually edited groups lose their generated fit metrics because containment and
overhang are no longer guaranteed after an arbitrary edit.
"""

from __future__ import annotations

import argparse
import math
import select
import sys
import time

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from sysid.collision_proxy_generator.constants import (
    DEFAULT_CONFIG_PATH,
    SPHERE_FILENAME,
)
from sysid.collision_proxy_generator.meshcat_visualizer import (
    _body_meshcat_path,
    _meshcat_transform,
    _publish_meshcat_scene,
    _sphere_meshcat_path,
)
from sysid.collision_proxy_generator.proxy_io import (
    _atomic_write_yaml,
    _yaml_document,
    load_collision_proxies,
)
from sysid.collision_proxy_generator.proxy_types import (
    CollisionProxyConfig,
    CollisionProxyResult,
    SphereProxy,
)
from sysid.collision_proxy_generator.urdf_geometry import (
    _world_link_poses,
    default_output_dir,
)
from sysid.utils.path_utils import BASE_DIR


MODIFICATION_MODE_SLIDER = "Modification mode (0=off, 1=on)"
SPHERE_INDEX_SLIDER = "Sphere index"
CENTER_X_SLIDER = "Sphere center X"
CENTER_Y_SLIDER = "Sphere center Y"
CENTER_Z_SLIDER = "Sphere center Z"
RADIUS_SLIDER = "Sphere radius"
ADD_SPHERE_BUTTON = "Add sphere"
DELETE_SPHERE_BUTTON = "Delete sphere"
SAVE_CHANGES_BUTTON = "Save changes"
EXIT_EDITOR_BUTTON = "Exit editor"
BODY_BUTTON_PREFIX = "Edit body: "

EDIT_SLIDERS = (
    CENTER_X_SLIDER,
    CENTER_Y_SLIDER,
    CENTER_Z_SLIDER,
    RADIUS_SLIDER,
)
EDITOR_BUTTONS = (
    ADD_SPHERE_BUTTON,
    DELETE_SPHERE_BUTTON,
    SAVE_CHANGES_BUTTON,
    EXIT_EDITOR_BUTTON,
)


def _configured_sphere_path(
    config: CollisionProxyConfig,
    sphere_path: str | Path | None,
) -> Path:
    """Resolve an explicit or config-derived sphere YAML path."""
    if sphere_path is not None:
        return Path(sphere_path).expanduser().resolve()
    if config.output_dir is None:
        output_dir = default_output_dir(config.robot)
    else:
        output_dir = config.output_dir.expanduser()
        if not output_dir.is_absolute():
            output_dir = BASE_DIR / output_dir
    return (output_dir.resolve() / SPHERE_FILENAME)


def save_modified_sphere_proxies(
    result: CollisionProxyResult,
    sphere_path: str | Path,
) -> Path:
    """Atomically overwrite one sphere-proxy YAML with the edited result."""
    path = Path(sphere_path).expanduser().resolve()
    _atomic_write_yaml(path, _yaml_document(result, "spheres"))
    return path


class SphereProxyEditor:
    """Meshcat controls and mutable state for editing sphere proxies."""

    def __init__(
        self,
        result: CollisionProxyResult,
        sphere_path: str | Path,
        *,
        meshcat: Any | None = None,
        proxy_opacity: float = 0.35,
        position_step: float = 0.001,
        radius_step: float = 0.001,
    ) -> None:
        if not result.groups:
            raise ValueError("The collision model has no groups to edit")
        if any(not group.spheres for group in result.groups):
            empty = [
                group.name for group in result.groups if not group.spheres
            ]
            raise ValueError(
                "Every editable group must contain a sphere: "
                + ", ".join(empty)
            )
        if not 0.0 <= proxy_opacity <= 1.0:
            raise ValueError("proxy_opacity must be in [0, 1]")
        if (
            not math.isfinite(position_step)
            or position_step <= 0.0
            or not math.isfinite(radius_step)
            or radius_step <= 0.0
        ):
            raise ValueError("Editor slider steps must be finite and positive")

        self.result = result
        self.sphere_path = Path(sphere_path).expanduser().resolve()
        self.proxy_opacity = proxy_opacity
        self.position_step = position_step
        self.radius_step = radius_step
        self.selected_group_index = 0
        self.selected_sphere_index = 0
        self.modification_mode = False
        self._body_selected = False
        self._edit_controls_visible = False
        self.dirty = False
        self._closed = False

        if meshcat is None:
            from pydrake.geometry import StartMeshcat

            meshcat = StartMeshcat()
        self.meshcat = meshcat
        self._robot_geometry = _publish_meshcat_scene(
            self.meshcat,
            self.result,
            initial_view="spheres",
            proxy_opacity=self.proxy_opacity,
        )
        poses = _world_link_poses(
            self._robot_geometry.link_order,
            self._robot_geometry.joints,
            self.result.joint_positions,
        )
        self._anchor_poses = {
            group.name: poses[group.frame] for group in self.result.groups
        }
        self._position_limit, self._radius_limit = self._editor_limits()
        self._body_buttons = tuple(
            f"{BODY_BUTTON_PREFIX}{group.name}"
            for group in self.result.groups
        )
        self._button_clicks: dict[str, int] = {}
        self._edit_values = self._selected_values()
        self._add_controls()
        self._render_spheres()
        self._print_instructions()

    @property
    def selected_group(self):
        """Return the currently selected proxy group."""
        return self.result.groups[self.selected_group_index]

    @property
    def selected_sphere(self) -> SphereProxy:
        """Return the currently selected sphere."""
        return self.selected_group.spheres[self.selected_sphere_index]

    def _editor_limits(self) -> tuple[float, float]:
        spheres = [
            sphere
            for group in self.result.groups
            for sphere in group.spheres
        ]
        position_extent = max(
            max(
                abs(coordinate) + sphere.radius
                for coordinate in sphere.center
            )
            for sphere in spheres
        )
        largest_radius = max(sphere.radius for sphere in spheres)
        return (
            max(0.25, 2.0 * position_extent),
            max(0.1, 4.0 * largest_radius),
        )

    def _selected_values(self) -> tuple[float, float, float, float]:
        sphere = self.selected_sphere
        return (*sphere.center, sphere.radius)

    def _add_controls(self) -> None:
        self.meshcat.AddSlider(
            MODIFICATION_MODE_SLIDER,
            min=0.0,
            max=1.0,
            step=1.0,
            value=0.0,
        )
        for button in (*self._body_buttons, *EDITOR_BUTTONS):
            self.meshcat.AddButton(button)
            self._button_clicks[button] = self.meshcat.GetButtonClicks(
                button
            )

    def _show_edit_controls(self) -> None:
        if self._edit_controls_visible:
            self._rebuild_sphere_index_slider()
            self._sync_edit_sliders()
            return
        self._edit_controls_visible = True
        self._rebuild_sphere_index_slider()
        x, y, z, radius = self._edit_values
        for name, value in zip(
            (CENTER_X_SLIDER, CENTER_Y_SLIDER, CENTER_Z_SLIDER),
            (x, y, z),
        ):
            self.meshcat.AddSlider(
                name,
                min=-self._position_limit,
                max=self._position_limit,
                step=self.position_step,
                value=value,
            )
        self.meshcat.AddSlider(
            RADIUS_SLIDER,
            min=self.radius_step,
            max=self._radius_limit,
            step=self.radius_step,
            value=radius,
        )

    def _hide_edit_controls(self) -> None:
        self.meshcat.DeleteSlider(SPHERE_INDEX_SLIDER, strict=False)
        for slider in EDIT_SLIDERS:
            self.meshcat.DeleteSlider(slider, strict=False)
        self._edit_controls_visible = False

    def _rebuild_sphere_index_slider(self) -> None:
        self.meshcat.DeleteSlider(SPHERE_INDEX_SLIDER, strict=False)
        if not self._edit_controls_visible:
            return
        sphere_count = len(self.selected_group.spheres)
        self.selected_sphere_index = min(
            self.selected_sphere_index,
            sphere_count - 1,
        )
        self.meshcat.AddSlider(
            SPHERE_INDEX_SLIDER,
            min=0.0,
            max=float(max(0, sphere_count - 1)),
            step=1.0,
            value=float(self.selected_sphere_index),
        )

    def _sync_edit_sliders(self) -> None:
        self._edit_values = self._selected_values()
        if not self._edit_controls_visible:
            return
        for name, value in zip(EDIT_SLIDERS, self._edit_values):
            self.meshcat.SetSliderValue(name, value)

    def _replace_selected_group(
        self,
        spheres: tuple[SphereProxy, ...],
    ) -> None:
        groups = list(self.result.groups)
        groups[self.selected_group_index] = replace(
            self.selected_group,
            spheres=spheres,
            sphere_fit=None,
        )
        self.result = replace(self.result, groups=tuple(groups))
        self.dirty = True

    def set_selected_sphere(
        self,
        *,
        center: tuple[float, float, float],
        radius: float,
    ) -> None:
        """Replace the selected sphere and invalidate generated fit metrics."""
        sphere = SphereProxy(center=center, radius=radius)
        spheres = list(self.selected_group.spheres)
        if sphere == spheres[self.selected_sphere_index]:
            return
        spheres[self.selected_sphere_index] = sphere
        self._replace_selected_group(tuple(spheres))
        self._edit_values = self._selected_values()
        self._render_sphere(
            self.selected_group_index,
            self.selected_sphere_index,
        )

    def add_sphere(self) -> None:
        """Duplicate the selected sphere and select the new copy."""
        spheres = [
            *self.selected_group.spheres,
            self.selected_sphere,
        ]
        self._replace_selected_group(tuple(spheres))
        self.selected_sphere_index = len(spheres) - 1
        self._rebuild_sphere_index_slider()
        self._sync_edit_sliders()
        self._render_spheres()
        print(
            f"Added sphere {self.selected_sphere_index} to "
            f"{self.selected_group.name}"
        )

    def delete_sphere(self) -> bool:
        """Delete the selected sphere, retaining at least one per group."""
        spheres = list(self.selected_group.spheres)
        if len(spheres) == 1:
            print(
                f"Cannot delete the last sphere from "
                f"{self.selected_group.name}",
                file=sys.stderr,
            )
            return False
        deleted_index = self.selected_sphere_index
        del spheres[deleted_index]
        self._replace_selected_group(tuple(spheres))
        self.selected_sphere_index = min(
            deleted_index,
            len(spheres) - 1,
        )
        self._rebuild_sphere_index_slider()
        self._sync_edit_sliders()
        self._render_spheres()
        print(
            f"Deleted sphere {deleted_index} from "
            f"{self.selected_group.name}"
        )
        return True

    def save(self) -> Path:
        """Atomically save the current result over the loaded YAML."""
        path = save_modified_sphere_proxies(
            self.result,
            self.sphere_path,
        )
        self.dirty = False
        print(f"Saved sphere proxy changes: {path}")
        print(
            "Fit metrics were removed from modified groups; regenerate "
            "the proxies to recompute containment and overhang quality."
        )
        return path

    def _sphere_color(self, group_index: int, sphere_index: int) -> Any:
        from pydrake.geometry import Rgba

        selected = (
            self.modification_mode
            and self._body_selected
            and group_index == self.selected_group_index
            and sphere_index == self.selected_sphere_index
        )
        if selected:
            selected_alpha = min(
                0.75,
                max(0.45, self.proxy_opacity + 0.2),
            )
            return Rgba(1.0, 0.85, 0.05, selected_alpha)
        return Rgba(0.05, 0.45, 1.0, self.proxy_opacity)

    def _render_sphere(
        self,
        group_index: int,
        sphere_index: int,
    ) -> None:
        from pydrake.geometry import Sphere

        group = self.result.groups[group_index]
        sphere = group.spheres[sphere_index]
        path = _sphere_meshcat_path(group.name, sphere_index)
        self.meshcat.SetObject(
            path,
            Sphere(sphere.radius),
            self._sphere_color(group_index, sphere_index),
        )
        anchor_pose = self._anchor_poses[group.name]
        transform = anchor_pose.copy()
        transform[:3, 3] = (
            anchor_pose[:3, :3] @ np.asarray(sphere.center)
            + anchor_pose[:3, 3]
        )
        self.meshcat.SetTransform(
            path,
            _meshcat_transform(transform),
        )

    def _render_spheres(self) -> None:
        for group_index, group in enumerate(self.result.groups):
            sphere_root = f"{_body_meshcat_path(group.name)}/spheres"
            self.meshcat.Delete(sphere_root)
            for sphere_index, _sphere in enumerate(group.spheres):
                self._render_sphere(group_index, sphere_index)
            self.meshcat.SetProperty(sphere_root, "visible", True)

    def _set_body_visibility(self) -> None:
        isolate = self.modification_mode and self._body_selected
        for group_index, group in enumerate(self.result.groups):
            self.meshcat.SetProperty(
                _body_meshcat_path(group.name),
                "visible",
                not isolate or group_index == self.selected_group_index,
            )

    def _select_group(self, group_index: int) -> None:
        self.selected_group_index = group_index
        self.selected_sphere_index = 0
        self._body_selected = True
        self._edit_values = self._selected_values()
        self._show_edit_controls()
        self._set_body_visibility()
        self._render_spheres()
        print(f"Selected body: {self.selected_group.name}")

    def _select_sphere(self, sphere_index: int) -> None:
        sphere_index = int(
            np.clip(
                sphere_index,
                0,
                len(self.selected_group.spheres) - 1,
            )
        )
        if sphere_index == self.selected_sphere_index:
            return
        self.selected_sphere_index = sphere_index
        self._sync_edit_sliders()
        self._render_spheres()
        print(
            f"Selected sphere {sphere_index} on "
            f"{self.selected_group.name}"
        )

    def _button_was_clicked(
        self,
        button: str,
        current_clicks: dict[str, int],
    ) -> bool:
        return current_clicks[button] > self._button_clicks[button]

    def poll_once(self) -> bool:
        """Process one control update; return false after Exit editor."""
        if self._closed:
            return False
        mode = self.meshcat.GetSliderValue(
            MODIFICATION_MODE_SLIDER
        ) >= 0.5
        if mode != self.modification_mode:
            self.modification_mode = mode
            self._body_selected = False
            self._hide_edit_controls()
            self._set_body_visibility()
            self._render_spheres()
            print(
                "Modification mode enabled"
                if mode
                else "Modification mode disabled"
            )

        buttons = (*self._body_buttons, *EDITOR_BUTTONS)
        current_clicks = {
            button: self.meshcat.GetButtonClicks(button)
            for button in buttons
        }
        if self.modification_mode:
            for group_index, button in enumerate(self._body_buttons):
                if self._button_was_clicked(button, current_clicks):
                    self._select_group(group_index)

            if self._body_selected:
                sphere_index = int(
                    round(
                        self.meshcat.GetSliderValue(
                            SPHERE_INDEX_SLIDER
                        )
                    )
                )
                self._select_sphere(sphere_index)

                if self._button_was_clicked(
                    ADD_SPHERE_BUTTON,
                    current_clicks,
                ):
                    self.add_sphere()
                if self._button_was_clicked(
                    DELETE_SPHERE_BUTTON,
                    current_clicks,
                ):
                    self.delete_sphere()

                values = tuple(
                    self.meshcat.GetSliderValue(name)
                    for name in EDIT_SLIDERS
                )
                if values != self._edit_values:
                    self.set_selected_sphere(
                        center=(values[0], values[1], values[2]),
                        radius=values[3],
                    )

        if self._button_was_clicked(
            SAVE_CHANGES_BUTTON,
            current_clicks,
        ):
            self.save()
        should_continue = not self._button_was_clicked(
            EXIT_EDITOR_BUTTON,
            current_clicks,
        )
        self._button_clicks = current_clicks
        return should_continue

    def run(self, *, poll_period: float = 0.05) -> None:
        """Keep the Meshcat editor alive until Exit, Enter, or Ctrl-C."""
        if not math.isfinite(poll_period) or poll_period <= 0.0:
            raise ValueError("poll_period must be finite and positive")
        try:
            while self.poll_once():
                if sys.stdin.isatty():
                    readable, _writable, _errors = select.select(
                        [sys.stdin],
                        [],
                        [],
                        0.0,
                    )
                    if readable:
                        sys.stdin.readline()
                        break
                time.sleep(poll_period)
        except KeyboardInterrupt:
            pass
        finally:
            self.close()

    def close(self) -> None:
        """Remove editor controls without deleting the Meshcat scene."""
        if self._closed:
            return
        for button in (*self._body_buttons, *EDITOR_BUTTONS):
            self.meshcat.DeleteButton(button, strict=False)
        self._hide_edit_controls()
        self.meshcat.DeleteSlider(
            MODIFICATION_MODE_SLIDER,
            strict=False,
        )
        self._closed = True

    def _print_instructions(self) -> None:
        if hasattr(self.meshcat, "web_url"):
            print(f"Meshcat URL: {self.meshcat.web_url()}")
        print(
            "Set 'Modification mode' to 1, press a named body button, "
            "then use the sphere-index and X/Y/Z/radius sliders that appear. "
            "Only the selected body is shown, and its selected sphere is "
            "transparent yellow. Changes remain in memory until Save changes."
        )


def launch_sphere_proxy_editor(
    sphere_path: str | Path,
    *,
    meshcat: Any | None = None,
    proxy_opacity: float = 0.35,
    hold: bool = True,
) -> SphereProxyEditor:
    """Load a sphere YAML, publish it, and optionally run its editor loop."""
    path = Path(sphere_path).expanduser().resolve()
    result = load_collision_proxies(sphere_path=path)
    editor = SphereProxyEditor(
        result,
        path,
        meshcat=meshcat,
        proxy_opacity=proxy_opacity,
    )
    if hold:
        editor.run()
    return editor


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the command-line interface for the Meshcat sphere editor."""
    parser = argparse.ArgumentParser(
        description=(
            "Interactively add, delete, move, resize, and save sphere "
            "collision proxies using Drake Meshcat."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=(
            "Collision-proxy configuration used to locate the default "
            "sphere YAML."
        ),
    )
    parser.add_argument(
        "--sphere-yaml",
        type=Path,
        default=None,
        help=(
            "Sphere YAML to edit. Defaults to sphere_collisions.yaml in "
            "the configured output directory."
        ),
    )
    parser.add_argument(
        "--proxy-opacity",
        type=float,
        default=0.35,
        help="Opacity for unselected spheres.",
    )
    return parser


def main() -> None:
    """Load the selected proxy file and run the interactive editor."""
    parser = build_argument_parser()
    args = parser.parse_args()
    try:
        config = CollisionProxyConfig.from_yaml(args.config)
        sphere_path = _configured_sphere_path(
            config,
            args.sphere_yaml,
        )
        launch_sphere_proxy_editor(
            sphere_path,
            proxy_opacity=args.proxy_opacity,
        )
    except (
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
        yaml.YAMLError,
    ) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
