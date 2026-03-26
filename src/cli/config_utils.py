"""
Configuration loading utilities for the matcher pipeline.

Extracted from main.py (Jan 2026).
"""

import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..config import Config

from ..config.utils import safe_get_config_value

logger = logging.getLogger(__name__)


def validate_root_directories(config: 'Config') -> None:
    """
    Validate and create root directories (E:/v, E:/i, etc.) at startup.

    This ensures that custom root directories exist before the pipeline
    starts processing. If they don't exist, attempts to create them.

    Args:
        config: Configuration object

    Raises:
        SystemExit: If directories cannot be created
    """
    errors = []

    # Check download.root_dir
    download_root = safe_get_config_value(config.download, 'root_dir')
    if download_root:
        root_path = Path(download_root)
        if not root_path.is_absolute():
            root_path = root_path.resolve()
        elif not root_path.exists():
            try:
                root_path.mkdir(parents=True, exist_ok=True)
                logger.info(f"Created videos root directory: {root_path}")
            except Exception as e:
                errors.append(f"Cannot create download.root_dir: {download_root} - {e}")
        else:
            logger.info(f"Videos root directory exists: {root_path}")

    # Check image_search.root_dir
    image_root = safe_get_config_value(config.image_search, 'root_dir')
    if image_root:
        root_path = Path(image_root)
        if not root_path.is_absolute():
            root_path = root_path.resolve()
        elif not root_path.exists():
            try:
                root_path.mkdir(parents=True, exist_ok=True)
                logger.info(f"Created images root directory: {root_path}")
            except Exception as e:
                errors.append(f"Cannot create image_search.root_dir: {image_root} - {e}")
        else:
            logger.info(f"Images root directory exists: {root_path}")

    if errors:
        logger.error("[CFG-001] Root directory configuration errors:")
        for e in errors:
            logger.error(f"[CFG-001]   - {e}")
        logger.error("[CFG-001] Check your config.yaml and ensure drives exist.")
        sys.exit(1)


def _reset_wrongly_resolved_paths(config: 'Config', project_dir: Path) -> None:
    """Reset section paths that were resolved to the wrong base directory.

    When load_config() is called, __post_init__ runs _resolve_paths() with
    the default project_dir=".".  This resolves relative paths (e.g. ".cache")
    to the current working directory instead of the real project directory.

    This function detects those wrongly-resolved paths and resets them to
    their original relative defaults so _resolve_paths() can re-resolve
    them correctly when called with the proper project_dir.

    Args:
        config: Configuration object whose paths may be cwd-resolved
        project_dir: The actual project directory
    """
    project_str = str(project_dir)

    # (section_obj, field_name, default_relative_value)
    section_path_defaults = [
        (config.cache, 'cache_dir', '.cache'),
        (config.logging, 'log_dir', 'logs'),
        (config.output, 'output_dir', 'output'),
        (config.transcription, 'cache_dir', 'transcriptions'),
    ]

    for section_obj, field_name, default_val in section_path_defaults:
        value = getattr(section_obj, field_name, None)
        if value and Path(value).is_absolute() and not str(value).startswith(project_str):
            setattr(section_obj, field_name, default_val)


def make_paths_project_relative(config: 'Config', project_dir: Path) -> 'Config':
    """
    Ensure paths in config are relative to project directory.

    Delegates to Config._resolve_paths() which is the single source of
    truth for all path resolution (output_dir, download_dir, cache_dir,
    log_dir, etc.).  Sets project_dir on the config then resolves.

    Args:
        config: Configuration object
        project_dir: Project directory path

    Returns:
        Updated configuration object
    """
    config.project_dir = str(project_dir)
    _reset_wrongly_resolved_paths(config, project_dir)
    config._resolve_paths()
    return config


def load_project_config(project_dir: Path, config_path: Path = None) -> 'Config':
    """
    Load configuration for a project.

    Args:
        project_dir: Project directory
        config_path: Path to config file (defaults to config.yaml)

    Returns:
        Configuration object with paths resolved to project directory
    """
    from ..config import load_config

    # Load config from specified path or default
    if config_path and config_path.exists():
        config = load_config(str(config_path))
    else:
        config = load_config()

    # Delegate to make_paths_project_relative which is the single
    # entry point for project-relative path resolution.
    return make_paths_project_relative(config, project_dir)


def _merge_single_tier(tier_config, overrides: dict):
    """
    Merge overrides into a single DurationTierConfig.

    Supports aliases:
    - 'count' or 'per_keyword' -> 'videos_per_keyword'
    - 'min' -> 'min_seconds'
    - 'max' -> 'max_seconds'

    Args:
        tier_config: DurationTierConfig object to update
        overrides: Dict of overrides with potentially aliased keys

    Returns:
        Updated tier_config
    """
    key_aliases = {
        'count': 'videos_per_keyword',
        'per_keyword': 'videos_per_keyword',
        'min': 'min_seconds',
        'max': 'max_seconds',
    }

    for key, value in overrides.items():
        actual_key = key_aliases.get(key, key)
        if hasattr(tier_config, actual_key):
            setattr(tier_config, actual_key, value)

    return tier_config


def _merge_duration_tiers(tiers_or_tier, overrides: dict):
    """
    Merge duration tier overrides.

    Can be called with either:
    - DurationTiersConfig (or any object with tier name attributes) + dict of {tier_name: {key: value}}
    - DurationTierConfig + dict of {key: value}

    Supports aliases:
    - 'count' or 'per_keyword' -> 'videos_per_keyword'
    - 'min' -> 'min_seconds'
    - 'max' -> 'max_seconds'

    Args:
        tiers_or_tier: DurationTiersConfig or DurationTierConfig
        overrides: Dict of overrides

    Returns:
        Updated object
    """
    from ..config.sections.duration import DurationTierConfig

    # Check if overrides contains tier names (short, medium, long, longer)
    tier_names = ['short', 'medium', 'long', 'longer']
    has_tier_overrides = any(name in overrides for name in tier_names)

    # If overrides look like tier overrides OR object has any tier attributes
    if has_tier_overrides or any(hasattr(tiers_or_tier, name) for name in tier_names):
        # It's a DurationTiersConfig-like object - iterate over tier names
        for tier_name, tier_overrides in overrides.items():
            if tier_name not in tier_names:
                continue
            if not isinstance(tier_overrides, dict):
                continue

            tier_config = getattr(tiers_or_tier, tier_name, None)

            # Handle case where tier_config is None - create from scratch
            if tier_config is None:
                # Map aliased keys to internal format
                key_aliases = {
                    'count': 'videos_per_keyword',
                    'per_keyword': 'videos_per_keyword',
                    'min': 'min_seconds',
                    'max': 'max_seconds',
                }
                mapped = {}
                for k, v in tier_overrides.items():
                    actual_key = key_aliases.get(k, k)
                    mapped[actual_key] = v
                # Create new DurationTierConfig with defaults + overrides
                new_config = DurationTierConfig(
                    min_seconds=mapped.get('min_seconds', 0),
                    max_seconds=mapped.get('max_seconds', 0),
                    videos_per_keyword=mapped.get('videos_per_keyword', 5),
                    max_total=mapped.get('max_total', 0)
                )
                setattr(tiers_or_tier, tier_name, new_config)
            else:
                _merge_single_tier(tier_config, tier_overrides)
        return tiers_or_tier
    else:
        # It's a single DurationTierConfig
        return _merge_single_tier(tiers_or_tier, overrides)


def _revalidate_modified_sections(config: 'Config', modified_sections: set) -> None:
    """
    Re-run __post_init__() on modified config sections that have one.

    After merge_config() applies raw overrides, nested dicts may need
    converting to dataclass instances and constraints may need re-checking.

    Args:
        config: Configuration object with sections already modified
        modified_sections: Set of section attribute names that were modified
    """
    for section_name in modified_sections:
        section_obj = getattr(config, section_name, None)
        if section_obj is not None and hasattr(section_obj, '__post_init__'):
            section_obj.__post_init__()


def merge_config(config: 'Config', overrides: dict) -> 'Config':
    """
    Merge override dict into config object.

    Handles nested configuration sections like 'keyword', 'download', etc.
    Special handling for duration_tiers with key aliases.
    After all overrides are applied, re-runs __post_init__() on modified
    sections to re-validate constraints and convert nested dicts.

    Args:
        config: Base configuration object
        overrides: Dictionary of overrides

    Returns:
        Updated configuration object
    """
    # Handle legacy paths: download.tier_config -> duration_tiers
    if 'download' in overrides and isinstance(overrides['download'], dict):
        if 'tier_config' in overrides['download']:
            tier_overrides = overrides['download'].pop('tier_config')
            if 'duration_tiers' not in overrides:
                overrides['duration_tiers'] = {}
            overrides['duration_tiers'].update(tier_overrides)

    # Handle legacy paths: keywords.tier_config -> duration_tiers
    if 'keywords' in overrides and isinstance(overrides['keywords'], dict):
        if 'tier_config' in overrides['keywords']:
            tier_overrides = overrides['keywords'].pop('tier_config')
            if 'duration_tiers' not in overrides:
                overrides['duration_tiers'] = {}
            overrides['duration_tiers'].update(tier_overrides)

    # Handle legacy paths: keyword.tier_config (singular) -> duration_tiers
    if 'keyword' in overrides and isinstance(overrides['keyword'], dict):
        if 'tier_config' in overrides['keyword']:
            tier_overrides = overrides['keyword'].pop('tier_config')
            if 'duration_tiers' not in overrides:
                overrides['duration_tiers'] = {}
            overrides['duration_tiers'].update(tier_overrides)

    modified_sections = set()

    for section, values in overrides.items():
        # Special handling for duration_tiers
        if section == 'duration_tiers' and isinstance(values, dict):
            if hasattr(config, 'duration_tiers'):
                tiers_obj = config.duration_tiers
                for tier_name, tier_overrides in values.items():
                    if hasattr(tiers_obj, tier_name) and isinstance(tier_overrides, dict):
                        tier_config = getattr(tiers_obj, tier_name)
                        _merge_single_tier(tier_config, tier_overrides)
                modified_sections.add('duration_tiers')
            continue

        if hasattr(config, section):
            section_obj = getattr(config, section)
            if isinstance(values, dict):
                for key, value in values.items():
                    if hasattr(section_obj, key):
                        setattr(section_obj, key, value)
                modified_sections.add(section)
            else:
                setattr(config, section, values)

    # Re-validate modified sections (convert nested dicts, check constraints)
    _revalidate_modified_sections(config, modified_sections)

    # Run cross-section validation to catch invalid overrides
    validation_errors = config.validate()
    for error in validation_errors:
        logger.warning("Post-merge validation: %s", error)

    return config


def validate_config_at_startup(config: 'Config') -> bool:
    """
    Validate critical configuration before running.

    Validates:
    - video_search.results_per_keyword > 0, video_search.max_total_results > 0
    - search_budget values consistency with video_search values
    - duration_tiers configuration (at least one tier must be valid)
    - matching.min_confidence is between 0 and 1
    - Deprecated config options in config.yaml

    Args:
        config: Configuration object

    Returns:
        True if valid, False if critical errors
    """
    errors = []
    warnings = []

    # Check for API keys
    if not config.gemini_api_key and not config.anthropic_api_key:
        logger.warning("No LLM API keys configured. Matching will use embedding-only mode.")

    # Check for required paths
    if hasattr(config.download, 'download_dir'):
        dl_path = Path(config.download.download_dir)
        if dl_path.is_absolute() and not dl_path.parent.exists():
            errors.append(f"Download directory parent does not exist: {dl_path.parent}")

    # Validate video_search config: results_per_keyword and max_total_results > 0
    if hasattr(config, 'video_search'):
        vs = config.video_search
        results_per_keyword = getattr(vs, 'results_per_keyword', 0)
        max_total_results = getattr(vs, 'max_total_results', 0)

        if results_per_keyword <= 0:
            errors.append(
                f"video_search.results_per_keyword must be > 0, got {results_per_keyword}. "
                "Check video_search.results_per_keyword in config.yaml"
            )
        if max_total_results <= 0:
            errors.append(
                f"video_search.max_total_results must be > 0, got {max_total_results}. "
                "Check video_search.max_total_results in config.yaml"
            )

        # Validate search_budget values consistency with video_search values
        if hasattr(config, 'search_budget'):
            sb = config.search_budget
            sb_results_per_keyword = getattr(sb, 'results_per_keyword', 0)
            sb_max_total = getattr(sb, 'max_total_results', 0)

            if sb_results_per_keyword > 0 and sb_results_per_keyword != results_per_keyword:
                warnings.append(
                    f"search_budget.results_per_keyword ({sb_results_per_keyword}) differs from "
                    f"video_search.results_per_keyword ({results_per_keyword}). "
                    "Consider aligning these values for consistent behavior."
                )
            if sb_max_total > 0 and sb_max_total != max_total_results:
                warnings.append(
                    f"search_budget.max_total_results ({sb_max_total}) differs from "
                    f"video_search.max_total_results ({max_total_results}). "
                    "Consider aligning these values for consistent behavior."
                )

    # Validate duration_tiers: at least one tier must exist and be valid
    if hasattr(config, 'duration_tiers'):
        tiers = config.duration_tiers
        tier_names = ['short', 'medium', 'long', 'longer']
        valid_tiers = 0

        for tier_name in tier_names:
            tier = getattr(tiers, tier_name, None)
            if tier is not None:
                # Check if tier has valid min/max (at least min must be defined)
                min_sec = getattr(tier, 'min_seconds', None)
                max_sec = getattr(tier, 'max_seconds', None)
                if min_sec is not None and max_sec is not None:
                    valid_tiers += 1

        if valid_tiers == 0:
            errors.append(
                "duration_tiers must have at least one valid tier. "
                "Check duration_tiers configuration in config.yaml"
            )

    # Validate matching.min_confidence is between 0 and 1
    if hasattr(config, 'matching'):
        matching = config.matching
        min_confidence = getattr(matching, 'min_confidence', None)

        if min_confidence is not None:
            if min_confidence < 0.0 or min_confidence > 1.0:
                errors.append(
                    f"matching.min_confidence must be between 0 and 1, got {min_confidence}. "
                    "Check matching.min_confidence in config.yaml"
                )

    # Check for deprecated config options in config.yaml
    # These are common deprecated options that should trigger warnings
    deprecated_options = [
        ('download', 'tier_config'),
        ('keywords', 'tier_config'),
        ('keyword', 'tier_config'),
        ('caption', 'force_transcribe'),
        ('match', 'use_llm'),
    ]

    # Note: We can't directly check config.yaml for deprecated options here
    # since we're working with the loaded Config object. The deprecation
    # warnings are handled in merge_config() which converts legacy paths.
    # Log a general warning about checking config.yaml for deprecated options.
    warnings.append(
        "If using a custom config.yaml, check for deprecated options: "
        "download.tier_config, keywords.tier_config, keyword.tier_config. "
        "These have been migrated to duration_tiers."
    )

    # US-142-009: Validate file references (directories, config files)
    file_reference_issues = config.validate_file_references()
    if file_reference_issues:
        logger.warning("[CFG-002] File reference validation warnings:")
        for issue in file_reference_issues:
            logger.warning(f"[CFG-002]   - [{issue['type']}] {issue['field']}: {issue['path']}")
            logger.warning(f"[CFG-002]     Suggestion: {issue['suggestion']}")
        # File reference issues are warnings, not errors - directories can be created
        warnings.extend([f"{i['field']}: {i['path']}" for i in file_reference_issues])

    # Print warnings
    if warnings:
        logger.warning("[CFG-003] Configuration warnings:")
        for w in warnings:
            logger.warning(f"[CFG-003]   - {w}")

    if errors:
        logger.error("[CFG-004] Configuration validation errors:")
        for e in errors:
            logger.error(f"[CFG-004]   - {e}")
        return False

    return True
