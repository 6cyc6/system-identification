#!/usr/bin/env python3
"""CLI for visualizing saved collision proxies and source geometry."""

from __future__ import annotations

import argparse

from pathlib import Path

import yaml

from sysid.collision_proxy_generator import (
    DEFAULT_CONFIG_PATH,
    CollisionProxyConfig,
    visualize_collision_model,
)
from sysid.utils.path_utils import BASE_DIR


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize saved sphere, capsule, and ellipsoid collision proxies "
            "with the source URDF collision geometry in Meshcat."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=(
            "YAML configuration file. Defaults to "
            f"{DEFAULT_CONFIG_PATH.relative_to(BASE_DIR)}."
        ),
    )
    return parser


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()
    try:
        config = CollisionProxyConfig.from_yaml(args.config)
        visualize_collision_model(config)
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
