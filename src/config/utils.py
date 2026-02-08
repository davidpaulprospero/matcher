"""
Configuration utility functions.

Provides safe_get_config_value() to eliminate scattered dict/object
dual-access patterns throughout the codebase (CLAUDE.md Rule 6).
"""

from dataclasses import fields, is_dataclass
from typing import Any, Dict


def safe_get_config_value(obj: Any, key: str, default: Any = None) -> Any:
    """
    Safely get a config value from either a dict or a dataclass/object.

    Implements CLAUDE.md Rule 6: Handle both dict .get() and getattr() access.

    Args:
        obj: A dict, dataclass instance, or any object
        key: The attribute/key name to look up
        default: Default value if key is missing

    Returns:
        The value for the key, or default if not found

    Examples:
        >>> safe_get_config_value({'timeout': 30}, 'timeout', 10)
        30
        >>> safe_get_config_value(some_dataclass, 'timeout', 10)
        10  # if timeout not set
    """
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def config_diff(config_a: Any, config_b: Any, _prefix: str = "") -> Dict[str, Dict[str, Any]]:
    """
    Compare two Config instances and return a dict of differences.

    Recursively compares dataclass fields. Fields with equal values are excluded.

    Args:
        config_a: First config instance
        config_b: Second config instance
        _prefix: Internal use - dot-delimited field path prefix

    Returns:
        Dict mapping field paths to {'old': value_a, 'new': value_b}.
        Empty dict if configs are identical.

    Example:
        >>> diff = config_diff(default_config, modified_config)
        >>> # {'matching.min_confidence': {'old': 0.6, 'new': 0.8}}
    """
    diffs: Dict[str, Dict[str, Any]] = {}

    if not is_dataclass(config_a) or not is_dataclass(config_b):
        # Leaf comparison for non-dataclass values
        if config_a != config_b:
            diffs[_prefix] = {"old": config_a, "new": config_b}
        return diffs

    for f in fields(config_a):
        # Skip private/metadata fields (e.g., _loaded_at, _config_hash)
        if f.name.startswith("_"):
            continue
        path = f"{_prefix}.{f.name}" if _prefix else f.name
        val_a = getattr(config_a, f.name)
        val_b = getattr(config_b, f.name)

        if is_dataclass(val_a) and is_dataclass(val_b):
            diffs.update(config_diff(val_a, val_b, _prefix=path))
        elif val_a != val_b:
            diffs[path] = {"old": val_a, "new": val_b}

    return diffs
