#!/usr/bin/env python3
"""CLI for generating robot collision proxies."""

from __future__ import annotations

import argparse

from pathlib import Path

import yaml

from sysid.collision_proxy_generator import (
    DEFAULT_CONFIG_PATH,
    CollisionProxyConfig,
    generate_configured_collision_proxies,
)
from sysid.utils.path_utils import BASE_DIR


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate conservative sphere, capsule, and ellipsoid collision "
            "proxies from a robot URDF and save YAML manifests."
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
        paths = generate_configured_collision_proxies(config)
    except (
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
        yaml.YAMLError,
    ) as error:
        parser.error(str(error))

    for kind, path in paths.items():
        print(f"Saved {kind}: {path}")


if __name__ == "__main__":
    main()
