"""
Configuration utility functions.

Provides safe_get_config_value() to eliminate scattered dict/object
dual-access patterns throughout the codebase (CLAUDE.md Rule 6).
"""

from typing import Any


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
