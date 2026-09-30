"""Resolve bundled experiment configuration without redirecting runtime writes."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG_MAP = json.loads((ROOT / 'configs/paths.json').read_text(encoding='utf-8'))


def resolve_config(path):
    """Prefer current run files; use a bundled configuration when one is absent.

    Only paths explicitly recorded in configs/paths.json can fall back. Missing
    checkpoints and other runtime products are never replaced with reference data.
    """
    path = Path(path)
    if path.exists():
        return path
    try:
        relative = path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path
    target = CONFIG_MAP.get(relative)
    return ROOT / target if target else path


def read_config_text(path, *args, **kwargs):
    """Read UTF-8 release text, preserving any explicitly supplied encoding."""
    if not args and 'encoding' not in kwargs:
        kwargs['encoding'] = 'utf-8'
    return resolve_config(path).read_text(*args, **kwargs)
