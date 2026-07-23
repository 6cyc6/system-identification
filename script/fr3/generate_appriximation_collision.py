#!/usr/bin/env python3
"""Generate and visualize robot collision sphere and capsule approximations."""

from __future__ import annotations

import argparse
import sys

from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sysid.model.collision_approximation_generator import (  # noqa: E402
    GeneratorConfig,
    generate_collision_approximations,
    load_collision_approximations,
    save_collision_approximations,
    visualize_collision_approximations,
)


def _group_override(value: str) -> tuple[str, tuple[str, ...]]:
    anchor, separator, members_text = value.partition("=")
    members = tuple(
        member.strip() for member in members_text.split(",") if member.strip()
    )
    if not separator or not anchor.strip() or not members:
        raise argparse.ArgumentTypeError(
            "Groups must use ANCHOR=LINK,LINK syntax"
        )
    return anchor.strip(), members


def _joint_position(value: str) -> tuple[str, float]:
    name, separator, position_text = value.partition("=")
    if not separator or not name.strip():
        raise argparse.ArgumentTypeError(
            "Joint positions must use JOINT=VALUE syntax"
        )
    try:
        position = float(position_text)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"Invalid joint position {position_text!r}"
        ) from error
    return name.strip(), position


def _dictionary(
    values: list[tuple[str, object]],
    *,
    label: str,
) -> dict[str, object]:
    result: dict[str, object] = {}
    for name, value in values:
        if name in result:
            raise ValueError(f"Duplicate {label}: {name}")
        result[name] = value
    return result


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate conservative collision spheres and capsules from a "
            "robot URDF, save YAML manifests, and inspect them in Meshcat."
        )
    )
    parser.add_argument("--robot", default="fr3")
    parser.add_argument(
        "--urdf",
        type=Path,
        help="Explicit URDF path; otherwise resolve from --robot.",
    )
    parser.add_argument(
        "--mode",
        choices=("both", "spheres", "capsules"),
        default="both",
    )
    parser.add_argument("--max-spheres", type=int, default=8)
    parser.add_argument("--max-capsules", type=int, default=3)
    parser.add_argument("--margin", type=float, default=0.002)
    parser.add_argument("--voxel-divisions", type=int, default=100)
    parser.add_argument("--precision", type=float, default=0.95)
    parser.add_argument(
        "--group",
        action="append",
        default=[],
        type=_group_override,
        metavar="ANCHOR=LINK,LINK",
        help="Explicitly merge collision links into an anchor-link frame.",
    )
    parser.add_argument(
        "--joint-position",
        action="append",
        default=[],
        type=_joint_position,
        metavar="JOINT=VALUE",
        help="Override the reference position used for grouped-link transforms.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help=(
            "Output directory. Defaults to "
            "robot_description/{robot}_description/collisions."
        ),
    )
    parser.add_argument(
        "--visualize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Publish source geometry and approximations to Meshcat.",
    )
    parser.add_argument(
        "--initial-view",
        choices=("spheres", "capsules", "both"),
        default="spheres",
    )
    parser.add_argument(
        "--hold",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep Meshcat alive until Enter or the Exit viewer button.",
    )
    return parser


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()
    try:
        group_overrides = _dictionary(args.group, label="group anchor")
        joint_positions = _dictionary(
            args.joint_position,
            label="joint-position override",
        )
        config = GeneratorConfig(
            max_spheres=args.max_spheres,
            max_capsules=args.max_capsules,
            margin=args.margin,
            voxel_divisions=args.voxel_divisions,
            precision=args.precision,
        )
        result = generate_collision_approximations(
            args.robot,
            urdf_path=args.urdf,
            config=config,
            mode=args.mode,
            group_overrides=group_overrides,
            joint_position_overrides=joint_positions,
        )
        paths = save_collision_approximations(
            result,
            args.output_dir,
            mode=args.mode,
        )
    except (FileNotFoundError, RuntimeError, TypeError, ValueError) as error:
        parser.error(str(error))

    for kind, path in paths.items():
        print(f"Saved {kind}: {path}")

    if not args.visualize:
        return

    loaded = load_collision_approximations(
        sphere_path=paths.get("spheres"),
        capsule_path=paths.get("capsules"),
    )
    visualize_collision_approximations(
        loaded,
        initial_view=args.initial_view,
        hold=args.hold,
    )


if __name__ == "__main__":
    main()
