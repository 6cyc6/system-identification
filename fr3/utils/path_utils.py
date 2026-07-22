"""Project paths derived from the environment initialized by ``env.sh``."""

import os
from pathlib import Path


def _load_base_dir():
    """Return the repository root exported by ``env.sh``."""
    raw_base_dir = os.environ.get("BASE_DIR")
    if not raw_base_dir:
        raise RuntimeError(
            "BASE_DIR is not set. Source env.sh or run the command through Pixi."
        )

    base_dir = Path(raw_base_dir).expanduser().resolve()
    if not base_dir.is_dir():
        raise RuntimeError(f"BASE_DIR does not point to a directory: {base_dir}")
    return base_dir


BASE_DIR = _load_base_dir()


def get_config_path(*parts):
    """Return a path below the repository's checked-in config directory."""
    return BASE_DIR.joinpath("config", *parts)


def resolve_repo_path(path):
    """Resolve a relative path against BASE_DIR; preserve absolute paths."""
    path = Path(path).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (BASE_DIR / path).resolve()


def _search_roots(path):
    """Build candidate roots for a file search.

    Relative paths are checked from both the current working directory and the
    repository exported by ``env.sh``.  The explicit robot-description fallback
    preserves support for older callers that pass ``../robot_description``.
    """
    raw_path = Path(path).expanduser()
    if raw_path.is_absolute():
        roots = [raw_path]
    else:
        roots = [Path.cwd() / raw_path, BASE_DIR / raw_path]
        if "robot_description" in raw_path.parts:
            roots.append(BASE_DIR / "robot_description")

    unique_roots = []
    for root in roots:
        root = root.resolve()
        if root not in unique_roots:
            unique_roots.append(root)
    return unique_roots


def find_path(name, path):
    """Recursively find ``name`` below one of the candidate search roots."""
    search_roots = _search_roots(path)
    for search_root in search_roots:
        if not search_root.exists():
            continue
        for root, _, files in os.walk(search_root):
            if name in files:
                return os.path.join(root, name)

    searched = ", ".join(str(root) for root in search_roots)
    raise FileNotFoundError(
        f"Can't find {name} in directory {path}. Searched: {searched}"
    )
