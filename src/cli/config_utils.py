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
    download_root = getattr(config.download, 'root_dir', None)
    if download_root:
        root_path = Path(download_root)
        if not root_path.is_absolute():
            errors.append(f"download.root_dir must be an absolute path: {download_root}")
        elif not root_path.exists():
            try:
                root_path.mkdir(parents=True, exist_ok=True)
                print(f"  ✓ Created videos root directory: {root_path}")
            except Exception as e:
                errors.append(f"Cannot create download.root_dir: {download_root} - {e}")
        else:
            print(f"  ✓ Videos root directory: {root_path}")

    # Check image_search.root_dir
    image_root = getattr(config.image_search, 'root_dir', None)
    if image_root:
        root_path = Path(image_root)
        if not root_path.is_absolute():
            errors.append(f"image_search.root_dir must be an absolute path: {image_root}")
        elif not root_path.exists():
            try:
                root_path.mkdir(parents=True, exist_ok=True)
                print(f"  ✓ Created images root directory: {root_path}")
            except Exception as e:
                errors.append(f"Cannot create image_search.root_dir: {image_root} - {e}")
        else:
            print(f"  ✓ Images root directory: {root_path}")

    if errors:
        print("\n  ❌ Root directory configuration errors:")
        for e in errors:
            print(f"    - {e}")
        print("\n  Check your config.yaml and ensure drives exist.")
        sys.exit(1)


def make_paths_project_relative(config: 'Config', project_dir: Path) -> 'Config':
    """
    Ensure paths in config are relative to project directory.

    Args:
        config: Configuration object
        project_dir: Project directory path

    Returns:
        Updated configuration object
    """
    # Update output directory
    if hasattr(config.output, 'output_dir'):
        output_path = Path(config.output.output_dir)
        if not output_path.is_absolute():
            config.output.output_dir = str(project_dir / output_path)

    # Update video directory
    if hasattr(config.download, 'download_dir'):
        video_path = Path(config.download.download_dir)
        if not video_path.is_absolute():
            config.download.download_dir = str(project_dir / video_path)

    # Update cache directory
    if hasattr(config.transcription, 'cache_dir'):
        cache_path = Path(config.transcription.cache_dir)
        if not cache_path.is_absolute():
            config.transcription.cache_dir = str(project_dir / cache_path)

    return config


def load_project_config(project_dir: Path, config_path: Path = None) -> 'Config':
    """
    Load configuration with project-specific overrides.

    Loading order:
    1. Base config from config_path (or default config.yaml)
    2. Project-specific overrides from project_config.yaml

    Args:
        project_dir: Project directory
        config_path: Path to base config file

    Returns:
        Merged configuration object
    """
    from ..config import load_config

    # Determine project_config.yaml path first
    project_config_path = project_dir / 'project_config.yaml'

    # Load base config
    # If config_path points to project_config.yaml, use default config instead
    # (project_config should be used as overrides, not as base config)
    if config_path and config_path.exists():
        try:
            # Check if config_path is the same as project_config.yaml
            if config_path.resolve() == project_config_path.resolve():
                print(f"  ⚠ --config points to project_config.yaml, using default config as base")
                config = load_config()  # Use default config.yaml
            else:
                config = load_config(str(config_path))
        except (OSError, ValueError):
            # resolve() can fail on some paths, fall back to string comparison
            if str(config_path).endswith('project_config.yaml'):
                print(f"  ⚠ --config points to project_config.yaml, using default config as base")
                config = load_config()
            else:
                config = load_config(str(config_path))
    else:
        config = load_config()

    # Look for project-specific config (apply overrides)
    project_overrides = {}
    if project_config_path.exists():
        import yaml
        print(f"  ✓ Loading project config: {project_config_path}")
        try:
            with open(project_config_path, 'r', encoding='utf-8') as f:
                project_overrides = yaml.safe_load(f) or {}
            config = merge_config(config, project_overrides)
            # Re-convert nested configs after merge (per Rule 2 - dicts need conversion)
            if hasattr(config, '_convert_nested_configs'):
                config._convert_nested_configs()
            # Re-resolve paths after merge to pick up video_source_dir
            config._resolve_paths()
        except Exception as e:
            print(f"  ⚠ Failed to load project config: {e}")

    # Set project_dir and resolve all paths relative to it
    # Reset paths to relative values so _resolve_paths() re-resolves them correctly
    # (they may have been resolved to wrong base in __post_init__)
    config.project_dir = str(project_dir)

    # Reset cache_dir if it was resolved to wrong location
    # Preserve custom values from project_config.yaml
    if config.cache.cache_dir and not str(config.cache.cache_dir).startswith(str(project_dir)):
        custom_cache = project_overrides.get('cache', {}).get('cache_dir')
        config.cache.cache_dir = custom_cache if custom_cache else ".cache"

    # Reset log_dir if it was resolved to wrong location
    # Preserve custom values from project_config.yaml
    if config.logging.log_dir and not str(config.logging.log_dir).startswith(str(project_dir)):
        custom_log = project_overrides.get('logging', {}).get('log_dir')
        config.logging.log_dir = custom_log if custom_log else "logs"

    config._resolve_paths()

    return config


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


def merge_config(config: 'Config', overrides: dict) -> 'Config':
    """
    Merge override dict into config object.

    Handles nested configuration sections like 'keyword', 'download', etc.
    Special handling for duration_tiers with key aliases.

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

    for section, values in overrides.items():
        # Special handling for duration_tiers
        if section == 'duration_tiers' and isinstance(values, dict):
            if hasattr(config, 'duration_tiers'):
                tiers_obj = config.duration_tiers
                for tier_name, tier_overrides in values.items():
                    if hasattr(tiers_obj, tier_name) and isinstance(tier_overrides, dict):
                        tier_config = getattr(tiers_obj, tier_name)
                        _merge_single_tier(tier_config, tier_overrides)
            continue

        if hasattr(config, section):
            section_obj = getattr(config, section)
            if isinstance(values, dict):
                _deep_merge_section(section_obj, values)
            else:
                setattr(config, section, values)
    return config


def _deep_merge_section(section_obj, overrides: dict, depth: int = 0) -> None:
    """
    Recursively merge overrides into a config section object.

    This handles nested config sections like download.fallback.caption
    by merging at each level instead of replacing entire sub-sections.

    Args:
        section_obj: The config section object to merge into
        overrides: Dictionary of overrides to apply
        depth: Current recursion depth (for debugging)
    """
    max_depth = 5  # Prevent infinite recursion
    if depth > max_depth:
        return

    for key, value in overrides.items():
        if not hasattr(section_obj, key):
            continue

        existing = getattr(section_obj, key)

        # If both existing and new value are nested objects/dicts, merge recursively
        if isinstance(value, dict) and existing is not None:
            # Check if existing is a dataclass or has attributes to merge into
            if hasattr(existing, '__dataclass_fields__') or (
                hasattr(existing, '__dict__') and not isinstance(existing, (str, int, float, bool, list, tuple))
            ):
                _deep_merge_section(existing, value, depth + 1)
            else:
                # Existing is a simple type or dict, replace it
                setattr(section_obj, key, value)
        else:
            # Simple value or list, replace directly
            setattr(section_obj, key, value)


def validate_config_at_startup(config: 'Config') -> bool:
    """
    Validate critical configuration before running.

    Args:
        config: Configuration object

    Returns:
        True if valid, False if critical errors
    """
    errors = []

    # Check for API keys
    if not config.gemini_api_key and not config.anthropic_api_key:
        logger.warning("No LLM API keys configured. Matching will use embedding-only mode.")

    # Check for required paths
    if hasattr(config.download, 'download_dir'):
        dl_path = Path(config.download.download_dir)
        if dl_path.is_absolute() and not dl_path.parent.exists():
            errors.append(f"Download directory parent does not exist: {dl_path.parent}")

    if errors:
        print("\n  ❌ Configuration validation errors:")
        for e in errors:
            print(f"    - {e}")
        return False

    return True
