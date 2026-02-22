"""
Config Schema Validation - US-85-006

Fail-fast validation of raw YAML data against config section dataclasses.
Uses dataclasses.fields() to auto-derive expected schema, catching typos
and type mismatches before any pipeline work begins.

US-128-007: Enhanced error messages with YAML line numbers and field locations.
"""

from __future__ import annotations

import difflib
import logging
import typing
from dataclasses import fields, is_dataclass, MISSING
from typing import Any, Dict, List, Optional, Tuple, Type, get_type_hints

import yaml

from .base import Config, ConfigError

logger = logging.getLogger(__name__)

# Config validation log prefix
CONFIG_VALIDATION_PREFIX = "[CONFIG_VALIDATION]"


# =============================================================================
# YAML Parsing with Line Number Tracking
# =============================================================================

def _parse_yaml_with_lines(content: str) -> Tuple[Dict[str, Any], Dict[str, int]]:
    """Parse YAML content and return data with line number mapping.

    Uses SafeLoader and tracks line numbers for top-level keys only.
    This is sufficient for most error messages.

    Returns:
        Tuple of (parsed_data, line_number_map)
        line_number_map: {section_name: line_number, ...}
    """
    from yaml import SafeLoader

    class _LineTrackingLoader(SafeLoader):
        pass

    def track_lines(loader: SafeLoader, node: yaml.MappingNode) -> Dict:
        """Track line numbers for top-level keys only."""
        mapping = {}
        for key_node, value_node in node.value:
            key = loader.construct_object(key_node, deep=False)
            if isinstance(key, str):
                mapping[key] = loader.construct_object(value_node, deep=False)
                # Store line number in special key format for top-level only
                mapping[f'__line_{key}__'] = key_node.start_mark.line + 1
        return mapping

    _LineTrackingLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
        track_lines
    )

    loader = _LineTrackingLoader(content)
    data = loader.get_single_data()

    # Extract line numbers from special __line_*__ keys
    line_map = {}
    if isinstance(data, dict):
        keys_to_remove = []
        for key in list(data.keys()):
            if key.startswith('__line_') and key.endswith('__'):
                # Extract section name from __line_X__
                original_key = key[7:-2]  # Remove __line_ prefix and __ suffix
                line_map[original_key] = data[key]
                keys_to_remove.append(key)

        # Also remove any nested __line_* keys
        for key in list(data.keys()):
            if isinstance(data[key], dict):
                nested_keys_to_remove = []
                for nested_key in data[key]:
                    if isinstance(nested_key, str) and nested_key.startswith('__line_'):
                        nested_keys_to_remove.append(nested_key)
                for nk in nested_keys_to_remove:
                    del data[key][nk]

        # Remove the line tracking keys from data
        for key in keys_to_remove:
            del data[key]

    return data, line_map


def _get_line_number(key: str, line_map: Dict[str, int]) -> Optional[int]:
    """Get line number for a given key from the line map."""
    return line_map.get(key)


# =============================================================================
# Fuzzy Matching for Unknown Field Suggestions
# =============================================================================

def _find_closest_field(unknown_field: str, valid_fields: List[str], max_suggestions: int = 3) -> List[str]:
    """Find the closest matching valid fields using fuzzy matching.

    Args:
        unknown_field: The unknown field name
        valid_fields: List of valid field names to match against
        max_suggestions: Maximum number of suggestions to return

    Returns:
        List of suggested field names (sorted by similarity)
    """
    if not valid_fields:
        return []

    # Get all matches with their similarity ratios
    matches = []
    for valid in valid_fields:
        ratio = difflib.SequenceMatcher(None, unknown_field.lower(), valid.lower()).ratio()
        matches.append((valid, ratio))

    # Sort by similarity (highest first)
    matches.sort(key=lambda x: x[1], reverse=True)

    # Return top suggestions with similarity > 0.4
    suggestions = [m[0] for m in matches if m[1] > 0.4][:max_suggestions]
    return suggestions


def _get_all_valid_fields() -> List[str]:
    """Get all valid top-level section names."""
    return list(_KNOWN_SECTION_NAMES)


class ConfigValidationError(ConfigError):
    """Raised when config data fails schema validation (wrong types, missing required fields)."""
    pass


# Known top-level YAML keys that map to Config dataclass sections.
# Derived from Config._from_dict() section_mapping + special keys.
_KNOWN_SECTION_NAMES = frozenset({
    # From section_mapping in Config._from_dict()
    'project', 'transcription', 'embedding', 'indexing', 'vision',
    'scene_detection', 'audio_analysis', 'matching', 'negative_matching',
    'remix', 'zero_download_remix', 'image_search', 'keyword', 'llm',
    'enhanced', 'downloading', 'download', 'stock_footage', 'deduplication',
    'output', 'multi_style', 'logging', 'cache', 'pipeline', 'api_keys',
    'healing', 'iterative_matching', 'rate_limit', 'broll', 'global_cache',
    'silent_video',
    # Special keys handled outside section_mapping
    'duration_tiers', 'project_dir',
    # Legacy/convenience keys present in config.yaml but not mapped to dataclasses
    'defaults', 'video_search',
    # US-142-007: External config validation webhook
    'validation_webhook',
    # US-151-003: Test mode configuration
    'test_mode',
})


def _get_primitive_type(field_type: Any) -> Optional[type]:
    """Extract the base primitive type from a type annotation.

    Returns the primitive type (int, float, str, bool) if the field type
    is one of those, or None if it's a complex/composite type.
    Handles Optional[X] by unwrapping to X.
    """
    # Unwrap Optional[X] -> X, but not other generic types (List, Dict, etc.)
    origin = getattr(field_type, '__origin__', None)
    if origin is not None:
        import typing
        # Only unwrap Union/Optional, not List/Dict/Tuple/etc.
        if origin is typing.Union:
            args = getattr(field_type, '__args__', ())
            non_none = [a for a in args if a is not type(None)]
            if len(non_none) == 1:
                field_type = non_none[0]
            else:
                return None  # Union of multiple types — skip
        else:
            return None  # List, Dict, Tuple, etc. — not primitive

    if field_type in (int, float, str, bool):
        return field_type
    return None


def _check_field_type(
    section_name: str,
    field_name: str,
    value: Any,
    expected_type: type,
    line_number: Optional[int] = None,
) -> Optional[str]:
    """Check if a value matches the expected primitive type.

    Returns an error string if mismatched, None if OK.
    Allows int where float is expected (numeric promotion).

    Args:
        section_name: Name of the config section
        field_name: Name of the field
        value: The value to check
        expected_type: The expected type (int, float, str, bool)
        line_number: Optional YAML line number for error messages
    """
    if value is None:
        return None  # None is acceptable for Optional fields

    # Build location string with line number if available
    location = f"{section_name}.{field_name}"
    if line_number:
        location = f"{location} (line {line_number})"

    # bool is a subclass of int in Python, so check bool first
    if expected_type is bool:
        if not isinstance(value, bool):
            return (
                f"{location}: expected bool, "
                f"got {type(value).__name__} ({value!r})"
            )
    elif expected_type is int:
        # Allow int, reject float/str/bool
        if isinstance(value, bool) or not isinstance(value, int):
            return (
                f"{location}: expected int, "
                f"got {type(value).__name__} ({value!r})"
            )
    elif expected_type is float:
        # Allow int or float (numeric promotion)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return (
                f"{location}: expected number, "
                f"got {type(value).__name__} ({value!r})"
            )
    elif expected_type is str:
        if not isinstance(value, str):
            return (
                f"{location}: expected string, "
                f"got {type(value).__name__}"
            )

    return None


def _validate_section_fields(
    section_name: str,
    data: Dict[str, Any],
    dataclass_type: Type,
    line_map: Optional[Dict[str, int]] = None,
) -> List[str]:
    """Validate fields in a section dict against the dataclass type hints.

    Returns list of error strings for type mismatches.

    Args:
        section_name: Name of the config section
        data: Raw dict from YAML parsing
        dataclass_type: The dataclass type to validate against
        line_map: Optional dict mapping field names to YAML line numbers
    """
    errors: List[str] = []
    line_map = line_map or {}

    try:
        type_hints = get_type_hints(dataclass_type)
    except Exception:
        type_hints = {}

    dc_fields = {f.name: f for f in fields(dataclass_type)}

    for key, value in data.items():
        if key not in dc_fields:
            continue  # Unknown keys handled elsewhere (_build_dataclass)

        field_type = type_hints.get(key, dc_fields[key].type)

        # Skip string annotations we can't resolve
        if isinstance(field_type, str):
            continue

        # Get line number for this field
        field_line = line_map.get(key)

        # If value is a dict and field is a nested dataclass, recurse
        if isinstance(value, dict) and hasattr(field_type, '__dataclass_fields__'):
            nested_errors = _validate_section_fields(
                f"{section_name}.{key}", value, field_type, line_map
            )
            errors.extend(nested_errors)
            continue

        # Check primitive types
        prim_type = _get_primitive_type(field_type)
        if prim_type is not None:
            err = _check_field_type(section_name, key, value, prim_type, field_line)
            if err:
                errors.append(err)
            continue

        # Handle List[str] type - validate as list of non-empty strings
        if _is_list_of_strings(field_type):
            list_errors = _validate_list_of_strings(section_name, key, value, field_line)
            errors.extend(list_errors)
            continue

        # Handle Dict[str, List[str]] type - validate as dict with non-empty list values
        if _is_dict_of_string_lists(field_type):
            dict_errors = _validate_dict_of_string_lists(section_name, key, value, field_line)
            errors.extend(dict_errors)

    return errors


def _is_list_of_strings(field_type: Any) -> bool:
    """Check if field type is List[str] or Optional[List[str]]."""
    origin = getattr(field_type, '__origin__', None)
    if origin is list:
        args = getattr(field_type, '__args__', ())
        return len(args) == 1 and args[0] is str
    # Check Optional[List[str]]
    if origin is typing.Union:
        args = getattr(field_type, '__args__', ())
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            return _is_list_of_strings(non_none[0])
    return False


def _is_dict_of_string_lists(field_type: Any) -> bool:
    """Check if field type is Dict[str, List[str]] or Optional[Dict[str, List[str]]]."""
    origin = getattr(field_type, '__origin__', None)
    if origin is dict:
        args = getattr(field_type, '__args__', ())
        return len(args) == 2 and args[0] is str and _is_list_of_strings(args[1])
    # Check Optional[Dict[str, List[str]]]
    if origin is typing.Union:
        args = getattr(field_type, '__args__', ())
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            return _is_dict_of_string_lists(non_none[0])
    return False


def _validate_list_of_strings(
    section_name: str,
    field_name: str,
    value: Any,
    line_number: Optional[int] = None,
) -> List[str]:
    """Validate that value is a list of non-empty strings.

    Returns list of error strings.

    Args:
        section_name: Name of the config section
        field_name: Name of the field
        value: The value to check
        line_number: Optional YAML line number for error messages
    """
    errors: List[str] = []

    # Build location string with line number if available
    location = f"{section_name}.{field_name}"
    if line_number:
        location = f"{location} (line {line_number})"

    if value is None:
        return errors  # None is acceptable for Optional fields

    if not isinstance(value, list):
        errors.append(
            f"{location}: expected list, got {type(value).__name__}"
        )
        return errors

    for i, item in enumerate(value):
        if not isinstance(item, str):
            errors.append(
                f"{location}[{i}]: expected string, got {type(item).__name__}"
            )
        elif not item:  # empty string
            errors.append(
                f"{location}[{i}]: expected non-empty string, got empty string"
            )

    return errors


def _validate_dict_of_string_lists(
    section_name: str,
    field_name: str,
    value: Any,
    line_number: Optional[int] = None,
) -> List[str]:
    """Validate that value is a dict with non-empty string list values.

    Returns list of error strings.

    Args:
        section_name: Name of the config section
        field_name: Name of the field
        value: The value to check
        line_number: Optional YAML line number for error messages
    """
    errors: List[str] = []

    # Build location string with line number if available
    location = f"{section_name}.{field_name}"
    if line_number:
        location = f"{location} (line {line_number})"

    if value is None:
        return errors  # None is acceptable for Optional fields

    if not isinstance(value, dict):
        errors.append(
            f"{location}: expected dict, got {type(value).__name__}"
        )
        return errors

    for key, list_value in value.items():
        if not isinstance(list_value, list):
            errors.append(
                f"{location}[{key!r}]: expected list, got {type(list_value).__name__}"
            )
            continue

        # Check for empty list
        if len(list_value) == 0:
            errors.append(
                f"{location}[{key!r}]: expected non-empty list"
            )
            continue

        for i, item in enumerate(list_value):
            if not isinstance(item, str):
                errors.append(
                    f"{location}[{key!r}][{i}]: expected string, got {type(item).__name__}"
                )
            elif not item:  # empty string
                errors.append(
                    f"{location}[{key!r}][{i}]: expected non-empty string"
                )

    return errors


def _validate_youtube_api_config(
    data: Dict[str, Any],
    line_map: Optional[Dict[str, int]] = None,
) -> List[str]:
    """Validate YouTubeAPIConfig-specific constraints.

    Validates:
    - api_key format (non-empty string when enabled)
    - quota_limit is positive integer within allowed range (1-1000000)
    - warn_at_percent is 0-100
    - timeout_seconds is reasonable (5-120)
    - cache_ttl_seconds is reasonable (60-86400)

    Args:
        data: Raw dict from YAML parsing for youtube_api section.
        line_map: Optional dict mapping field names to YAML line numbers.

    Returns:
        List of error strings.
    """
    errors: List[str] = []
    line_map = line_map or {}

    if not data or not isinstance(data, dict):
        return errors

    # Get line number helper
    def get_line(field: str) -> Optional[int]:
        return line_map.get(f"youtube_api.{field}")

    # Validate api_key: non-empty string when enabled
    enabled = data.get('enabled', False)
    api_key = data.get('api_key', '')
    api_keys = data.get('api_keys', [])
    if enabled:
        # Check if api_key is provided (even if empty string)
        # Note: api_keys (plural) can be used instead of single api_key for key rotation
        has_api_keys = api_keys and isinstance(api_keys, list) and len(api_keys) > 0
        if 'api_key' in data:
            if not isinstance(api_key, str):
                errors.append(
                    f"download.youtube_api.api_key: expected string, got {type(api_key).__name__}"
                )
            elif not api_key.strip() and not has_api_keys:
                line = get_line('api_key')
                location = "download.youtube_api.api_key"
                if line:
                    location += f" (line {line})"
                errors.append(f"{location}: api_key must be non-empty when enabled")
        elif not has_api_keys:
            # api_key not provided but enabled, and no api_keys either
            line = get_line('api_key')
            location = "download.youtube_api.api_key"
            if line:
                location += f" (line {line})"
            errors.append(f"{location}: api_key or api_keys required when enabled")

    # Validate api_keys list: non-empty strings
    api_keys = data.get('api_keys', [])
    if api_keys:
        if not isinstance(api_keys, list):
            errors.append(
                f"download.youtube_api.api_keys: expected list, got {type(api_keys).__name__}"
            )
        else:
            for i, key in enumerate(api_keys):
                if not isinstance(key, str):
                    errors.append(
                        f"download.youtube_api.api_keys[{i}]: expected string, got {type(key).__name__}"
                    )
                elif not key.strip():
                    errors.append(
                        f"download.youtube_api.api_keys[{i}]: must be non-empty string"
                    )

    # Validate quota_limit: positive integer within allowed range (1-1000000)
    quota_limit = data.get('quota_limit')
    if quota_limit is not None:
        if not isinstance(quota_limit, int) or isinstance(quota_limit, bool):
            errors.append(
                f"download.youtube_api.quota_limit: expected int, got {type(quota_limit).__name__}"
            )
        else:
            if quota_limit < 1:
                line = get_line('quota_limit')
                location = "download.youtube_api.quota_limit"
                if line:
                    location += f" (line {line})"
                errors.append(f"{location}: must be >= 1, got {quota_limit}")
            elif quota_limit > 1000000:
                line = get_line('quota_limit')
                location = "download.youtube_api.quota_limit"
                if line:
                    location += f" (line {line})"
                errors.append(f"{location}: must be <= 1000000, got {quota_limit}")

    # Validate warn_at_percent: 0-100
    warn_at_percent = data.get('warn_at_percent')
    if warn_at_percent is not None:
        if not isinstance(warn_at_percent, int) or isinstance(warn_at_percent, bool):
            errors.append(
                f"download.youtube_api.warn_at_percent: expected int, got {type(warn_at_percent).__name__}"
            )
        else:
            if warn_at_percent < 0 or warn_at_percent > 100:
                line = get_line('warn_at_percent')
                location = "download.youtube_api.warn_at_percent"
                if line:
                    location += f" (line {line})"
                errors.append(f"{location}: must be 0-100, got {warn_at_percent}")

    # Validate timeout_seconds: reasonable range (5-120)
    timeout_seconds = data.get('timeout_seconds')
    if timeout_seconds is not None:
        if not isinstance(timeout_seconds, int) or isinstance(timeout_seconds, bool):
            errors.append(
                f"download.youtube_api.timeout_seconds: expected int, got {type(timeout_seconds).__name__}"
            )
        else:
            if timeout_seconds < 5 or timeout_seconds > 120:
                line = get_line('timeout_seconds')
                location = "download.youtube_api.timeout_seconds"
                if line:
                    location += f" (line {line})"
                errors.append(f"{location}: must be 5-120 seconds, got {timeout_seconds}")

    # Validate cache_ttl_seconds: reasonable range (60-86400)
    cache_ttl_seconds = data.get('cache_ttl_seconds')
    if cache_ttl_seconds is not None:
        if not isinstance(cache_ttl_seconds, int) or isinstance(cache_ttl_seconds, bool):
            errors.append(
                f"download.youtube_api.cache_ttl_seconds: expected int, got {type(cache_ttl_seconds).__name__}"
            )
        else:
            if cache_ttl_seconds < 60 or cache_ttl_seconds > 86400:
                line = get_line('cache_ttl_seconds')
                location = "download.youtube_api.cache_ttl_seconds"
                if line:
                    location += f" (line {line})"
                errors.append(f"{location}: must be 60-86400 seconds, got {cache_ttl_seconds}")

    # Validate channel_metadata_cache_ttl_seconds: reasonable range (60-604800)
    channel_cache_ttl = data.get('channel_metadata_cache_ttl_seconds')
    if channel_cache_ttl is not None:
        if not isinstance(channel_cache_ttl, int) or isinstance(channel_cache_ttl, bool):
            errors.append(
                f"download.youtube_api.channel_metadata_cache_ttl_seconds: expected int, got {type(channel_cache_ttl).__name__}"
            )
        else:
            if channel_cache_ttl < 60 or channel_cache_ttl > 604800:
                line = get_line('channel_metadata_cache_ttl_seconds')
                location = "download.youtube_api.channel_metadata_cache_ttl_seconds"
                if line:
                    location += f" (line {line})"
                errors.append(f"{location}: must be 60-604800 seconds, got {channel_cache_ttl}")

    # Validate cache_ttl_days: reasonable range (1-90)
    cache_ttl_days = data.get('cache_ttl_days')
    if cache_ttl_days is not None:
        if not isinstance(cache_ttl_days, int) or isinstance(cache_ttl_days, bool):
            errors.append(
                f"download.youtube_api.cache_ttl_days: expected int, got {type(cache_ttl_days).__name__}"
            )
        else:
            if cache_ttl_days < 1 or cache_ttl_days > 90:
                line = get_line('cache_ttl_days')
                location = "download.youtube_api.cache_ttl_days"
                if line:
                    location += f" (line {line})"
                errors.append(f"{location}: must be 1-90 days, got {cache_ttl_days}")

    # Validate min_subscriber_count: positive integer
    min_subs = data.get('min_subscriber_count')
    if min_subs is not None:
        if not isinstance(min_subs, int) or isinstance(min_subs, bool):
            errors.append(
                f"download.youtube_api.min_subscriber_count: expected int, got {type(min_subs).__name__}"
            )
        elif min_subs < 0:
            line = get_line('min_subscriber_count')
            location = "download.youtube_api.min_subscriber_count"
            if line:
                location += f" (line {line})"
            errors.append(f"{location}: must be >= 0, got {min_subs}")

    # Validate max_retries: non-negative integer
    max_retries = data.get('max_retries')
    if max_retries is not None:
        if not isinstance(max_retries, int) or isinstance(max_retries, bool):
            errors.append(
                f"download.youtube_api.max_retries: expected int, got {type(max_retries).__name__}"
            )
        elif max_retries < 0:
            line = get_line('max_retries')
            location = "download.youtube_api.max_retries"
            if line:
                location += f" (line {line})"
            errors.append(f"{location}: must be >= 0, got {max_retries}")

    # Validate retry_delay_seconds: positive number
    retry_delay = data.get('retry_delay_seconds')
    if retry_delay is not None:
        if not isinstance(retry_delay, (int, float)) or isinstance(retry_delay, bool):
            errors.append(
                f"download.youtube_api.retry_delay_seconds: expected number, got {type(retry_delay).__name__}"
            )
        elif retry_delay <= 0:
            line = get_line('retry_delay_seconds')
            location = "download.youtube_api.retry_delay_seconds"
            if line:
                location += f" (line {line})"
            errors.append(f"{location}: must be > 0, got {retry_delay}")

    # US-152-008: Validate rate_limit_rps: positive number
    rate_limit_rps = data.get('rate_limit_rps')
    if rate_limit_rps is not None:
        if not isinstance(rate_limit_rps, (int, float)) or isinstance(rate_limit_rps, bool):
            errors.append(
                f"download.youtube_api.rate_limit_rps: expected number, got {type(rate_limit_rps).__name__}"
            )
        elif rate_limit_rps <= 0:
            line = get_line('rate_limit_rps')
            location = "download.youtube_api.rate_limit_rps"
            if line:
                location += f" (line {line})"
            errors.append(f"{location}: must be > 0, got {rate_limit_rps}")

    # Validate quota_auto_scale_enabled is boolean
    auto_scale = data.get('quota_auto_scale_enabled')
    if auto_scale is not None:
        if not isinstance(auto_scale, bool):
            errors.append(
                f"download.youtube_api.quota_auto_scale_enabled: expected bool, got {type(auto_scale).__name__}"
            )

    # Validate quota_multiplier: positive number
    quota_mult = data.get('quota_multiplier')
    if quota_mult is not None:
        if not isinstance(quota_mult, (int, float)) or isinstance(quota_mult, bool):
            errors.append(
                f"download.youtube_api.quota_multiplier: expected number, got {type(quota_mult).__name__}"
            )
        elif quota_mult <= 0:
            line = get_line('quota_multiplier')
            location = "download.youtube_api.quota_multiplier"
            if line:
                location += f" (line {line})"
            errors.append(f"{location}: must be > 0, got {quota_mult}")

    # Validate quota_floor and quota_ceiling
    quota_floor = data.get('quota_floor')
    if quota_floor is not None:
        if not isinstance(quota_floor, int) or isinstance(quota_floor, bool):
            errors.append(
                f"download.youtube_api.quota_floor: expected int, got {type(quota_floor).__name__}"
            )
        elif quota_floor < 1:
            line = get_line('quota_floor')
            location = "download.youtube_api.quota_floor"
            if line:
                location += f" (line {line})"
            errors.append(f"{location}: must be >= 1, got {quota_floor}")

    quota_ceiling = data.get('quota_ceiling')
    if quota_ceiling is not None:
        if not isinstance(quota_ceiling, int) or isinstance(quota_ceiling, bool):
            errors.append(
                f"download.youtube_api.quota_ceiling: expected int, got {type(quota_ceiling).__name__}"
            )
        elif quota_ceiling < 1:
            line = get_line('quota_ceiling')
            location = "download.youtube_api.quota_ceiling"
            if line:
                location += f" (line {line})"
            errors.append(f"{location}: must be >= 1, got {quota_ceiling}")

    # Validate quota_floor <= quota_ceiling
    if quota_floor is not None and quota_ceiling is not None:
        if isinstance(quota_floor, int) and isinstance(quota_ceiling, int):
            if not isinstance(quota_floor, bool) and not isinstance(quota_ceiling, bool):
                if quota_floor > quota_ceiling:
                    errors.append(
                        "download.youtube_api.quota_floor must be <= quota_ceiling"
                    )

    # Validate rotation_strategy: must be one of valid options
    rotation_strategy = data.get('rotation_strategy')
    valid_rotation_strategies = ['sequential', 'random', 'least_used', 'smart']
    if rotation_strategy is not None:
        if not isinstance(rotation_strategy, str):
            errors.append(
                f"download.youtube_api.rotation_strategy: expected string, got {type(rotation_strategy).__name__}"
            )
        elif rotation_strategy not in valid_rotation_strategies:
            line = get_line('rotation_strategy')
            location = "download.youtube_api.rotation_strategy"
            if line:
                location += f" (line {line})"
            errors.append(
                f"{location}: must be one of {valid_rotation_strategies}, got '{rotation_strategy}'"
            )

    # US-155-009: Validate timestamp_precision
    timestamp_precision = data.get('timestamp_precision')
    valid_precisions = ['millisecond', 'second', '5_second']
    if timestamp_precision is not None:
        if not isinstance(timestamp_precision, str):
            errors.append(
                f"download.youtube_api.timestamp_precision: expected string, got {type(timestamp_precision).__name__}"
            )
        elif timestamp_precision not in valid_precisions:
            line = get_line('timestamp_precision')
            location = "download.youtube_api.timestamp_precision"
            if line:
                location += f" (line {line})"
            errors.append(
                f"{location}: must be one of {valid_precisions}, got '{timestamp_precision}'"
            )

    return errors


def validate_config_schema(
    data: Dict[str, Any],
    raise_on_error: bool = True,
    line_map: Optional[Dict[str, int]] = None,
) -> List[str]:
    """Validate raw YAML config data against the Config schema.

    Checks:
    1. All top-level section names are recognized (warns on unknown)
    2. Numeric fields contain numbers, booleans contain booleans, etc.
    3. Nested objects match expected dataclass structure

    US-128-007: Enhanced error messages include YAML line numbers and
    suggest closest matching fields for unknown field errors.

    Args:
        data: Raw dict from YAML parsing (before Config construction).
        raise_on_error: If True, raise ConfigValidationError on type errors.
        line_map: Optional dict mapping section names to YAML line numbers.

    Returns:
        List of warning/error strings (for logging).

    Raises:
        ConfigValidationError: If raise_on_error=True and type errors found.
    """
    logger.info(f"{CONFIG_VALIDATION_PREFIX} Starting config schema validation")

    if not data or not isinstance(data, dict):
        logger.info(f"{CONFIG_VALIDATION_PREFIX} Validation complete: no data to validate")
        return []

    warnings: List[str] = []
    errors: List[str] = []
    line_map = line_map or {}

    # --- Check 1: Unknown top-level section names ---
    valid_fields = _get_all_valid_fields()
    for key in data.keys():
        if key not in _KNOWN_SECTION_NAMES:
            # Get line number if available
            line_num = line_map.get(key)
            location = f"'{key}'"
            if line_num:
                location = f"'{key}' (line {line_num})"

            # Find suggestions using fuzzy matching
            suggestions = _find_closest_field(key, valid_fields)
            suggestion_msg = ""
            if suggestions:
                suggestion_msg = f" Did you mean: {', '.join(suggestions)}?"

            warnings.append(
                f"Unknown config section {location} — will be ignored. "
                f"Check spelling or remove from config.yaml.{suggestion_msg}"
            )

    # --- Check 2 & 3: Type validation per section ---
    # Build section_name -> dataclass_type mapping from Config._from_dict logic
    from .sections import (
        TranscriptionConfig, EmbeddingConfig, IndexingConfig,
        VisionConfig, SceneDetectionConfig, AudioAnalysisConfig,
        MatchingConfig, NegativeMatchingConfig,
        RemixConfig, ZeroDownloadRemixConfig,
        ImageSearchConfig, KeywordConfig, LLMConfig,
        EnhancedFeaturesConfig, DownloadingConfig, DownloadConfig,
        StockFootageConfig, SilentVideoConfig,
        DeduplicationConfig, OutputConfig, MultiStyleConfig,
        LoggingConfig, CacheConfig, GlobalCacheConfig,
        PipelineConfig, APIKeysConfig,
        HealingConfig, IterativeMatchingConfig, RateLimitConfig,
        BrollConfig, VideoSearchConfig,
        ValidationWebhookConfig,
        YouTubeAPIConfig,
        TestModeConfig,
    )
    from .sections.core import ProjectConfig

    section_types: Dict[str, Type] = {
        'project': ProjectConfig,
        'transcription': TranscriptionConfig,
        'embedding': EmbeddingConfig,
        'indexing': IndexingConfig,
        'vision': VisionConfig,
        'scene_detection': SceneDetectionConfig,
        'audio_analysis': AudioAnalysisConfig,
        'matching': MatchingConfig,
        'negative_matching': NegativeMatchingConfig,
        'remix': RemixConfig,
        'zero_download_remix': ZeroDownloadRemixConfig,
        'image_search': ImageSearchConfig,
        'keyword': KeywordConfig,
        'llm': LLMConfig,
        'enhanced': EnhancedFeaturesConfig,
        'downloading': DownloadingConfig,
        'download': DownloadConfig,
        'stock_footage': StockFootageConfig,
        'silent_video': SilentVideoConfig,
        'deduplication': DeduplicationConfig,
        'output': OutputConfig,
        'multi_style': MultiStyleConfig,
        'logging': LoggingConfig,
        'cache': CacheConfig,
        'global_cache': GlobalCacheConfig,
        'pipeline': PipelineConfig,
        'api_keys': APIKeysConfig,
        'healing': HealingConfig,
        'iterative_matching': IterativeMatchingConfig,
        'rate_limit': RateLimitConfig,
        'broll': BrollConfig,
        'video_search': VideoSearchConfig,
        'validation_webhook': ValidationWebhookConfig,
        'test_mode': TestModeConfig,
    }

    for section_name, dc_type in section_types.items():
        section_data = data.get(section_name)
        if not section_data or not isinstance(section_data, dict):
            continue

        # Get line number for this section
        section_line = line_map.get(section_name)

        section_errors = _validate_section_fields(section_name, section_data, dc_type, line_map)
        errors.extend(section_errors)

    # --- Check 4: YouTubeAPIConfig-specific validation ---
    download_data = data.get('download')
    if download_data and isinstance(download_data, dict):
        youtube_api_data = download_data.get('youtube_api')
        if youtube_api_data and isinstance(youtube_api_data, dict):
            # Build nested line map for youtube_api fields
            nested_line_map = {}
            for key, value in line_map.items():
                if key.startswith('download.youtube_api.'):
                    nested_line_map[key.replace('download.youtube_api.', '')] = value
            yt_errors = _validate_youtube_api_config(youtube_api_data, nested_line_map)
            errors.extend(yt_errors)

    # Log warnings (unknown sections)
    for w in warnings:
        logger.warning(f"{CONFIG_VALIDATION_PREFIX} {w}")

    # Handle errors with CFG-xxx error codes
    if errors:
        for e in errors:
            logger.error(f"[CFG-001] Config schema validation error: {e}")

        logger.error(f"{CONFIG_VALIDATION_PREFIX} Schema validation failed with {len(errors)} error(s)")

        if raise_on_error:
            raise ConfigValidationError(
                f"Config schema validation failed with {len(errors)} error(s):\n"
                + "\n".join(f"  - {e}" for e in errors)
            )

    logger.info(f"{CONFIG_VALIDATION_PREFIX} Validation complete: {len(warnings)} warnings, {len(errors)} errors")
    return warnings + errors
