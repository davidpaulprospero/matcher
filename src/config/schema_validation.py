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
    if not data or not isinstance(data, dict):
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
    }

    for section_name, dc_type in section_types.items():
        section_data = data.get(section_name)
        if not section_data or not isinstance(section_data, dict):
            continue

        # Get line number for this section
        section_line = line_map.get(section_name)

        section_errors = _validate_section_fields(section_name, section_data, dc_type, line_map)
        errors.extend(section_errors)

    # Log warnings (unknown sections)
    for w in warnings:
        logger.warning(f"Config schema: {w}")

    # Handle errors
    if errors:
        for e in errors:
            logger.error(f"Config schema: {e}")

        if raise_on_error:
            raise ConfigValidationError(
                f"Config schema validation failed with {len(errors)} error(s):\n"
                + "\n".join(f"  - {e}" for e in errors)
            )

    return warnings + errors
