"""
Config Schema Validation - US-85-006

Fail-fast validation of raw YAML data against config section dataclasses.
Uses dataclasses.fields() to auto-derive expected schema, catching typos
and type mismatches before any pipeline work begins.
"""

from __future__ import annotations

import logging
from dataclasses import fields, is_dataclass, MISSING
from typing import Any, Dict, List, Optional, Tuple, Type, get_type_hints

from .base import Config, ConfigError

logger = logging.getLogger(__name__)


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
) -> Optional[str]:
    """Check if a value matches the expected primitive type.

    Returns an error string if mismatched, None if OK.
    Allows int where float is expected (numeric promotion).
    """
    if value is None:
        return None  # None is acceptable for Optional fields

    # bool is a subclass of int in Python, so check bool first
    if expected_type is bool:
        if not isinstance(value, bool):
            return (
                f"{section_name}.{field_name}: expected bool, "
                f"got {type(value).__name__} ({value!r})"
            )
    elif expected_type is int:
        # Allow int, reject float/str/bool
        if isinstance(value, bool) or not isinstance(value, int):
            return (
                f"{section_name}.{field_name}: expected int, "
                f"got {type(value).__name__} ({value!r})"
            )
    elif expected_type is float:
        # Allow int or float (numeric promotion)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return (
                f"{section_name}.{field_name}: expected number, "
                f"got {type(value).__name__} ({value!r})"
            )
    elif expected_type is str:
        if not isinstance(value, str):
            return (
                f"{section_name}.{field_name}: expected str, "
                f"got {type(value).__name__} ({value!r})"
            )

    return None


def _validate_section_fields(
    section_name: str,
    data: Dict[str, Any],
    dataclass_type: Type,
) -> List[str]:
    """Validate fields in a section dict against the dataclass type hints.

    Returns list of error strings for type mismatches.
    """
    errors: List[str] = []

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

        # If value is a dict and field is a nested dataclass, recurse
        if isinstance(value, dict) and hasattr(field_type, '__dataclass_fields__'):
            nested_errors = _validate_section_fields(
                f"{section_name}.{key}", value, field_type
            )
            errors.extend(nested_errors)
            continue

        # Check primitive types
        prim_type = _get_primitive_type(field_type)
        if prim_type is not None:
            err = _check_field_type(section_name, key, value, prim_type)
            if err:
                errors.append(err)

    return errors


def validate_config_schema(
    data: Dict[str, Any],
    raise_on_error: bool = True,
) -> List[str]:
    """Validate raw YAML config data against the Config schema.

    Checks:
    1. All top-level section names are recognized (warns on unknown)
    2. Numeric fields contain numbers, booleans contain booleans, etc.
    3. Nested objects match expected dataclass structure

    Args:
        data: Raw dict from YAML parsing (before Config construction).
        raise_on_error: If True, raise ConfigValidationError on type errors.

    Returns:
        List of warning/error strings (for logging).

    Raises:
        ConfigValidationError: If raise_on_error=True and type errors found.
    """
    if not data or not isinstance(data, dict):
        return []

    warnings: List[str] = []
    errors: List[str] = []

    # --- Check 1: Unknown top-level section names ---
    for key in data.keys():
        if key not in _KNOWN_SECTION_NAMES:
            warnings.append(
                f"Unknown config section '{key}' — will be ignored. "
                f"Check spelling or remove from config.yaml."
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
        BrollConfig,
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
    }

    for section_name, dc_type in section_types.items():
        section_data = data.get(section_name)
        if not section_data or not isinstance(section_data, dict):
            continue

        section_errors = _validate_section_fields(section_name, section_data, dc_type)
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
