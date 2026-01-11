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
    1. Base config from config_path
    2. Project-specific overrides from project_config.yaml

    Args:
        project_dir: Project directory
        config_path: Path to base config file

    Returns:
        Merged configuration object
    """
    from ..config import load_config

    # Load base config
    if config_path and config_path.exists():
        config = load_config(str(config_path))
    else:
        config = load_config()

    # Look for project-specific config
    project_config_path = project_dir / 'project_config.yaml'
    if project_config_path.exists():
        import yaml
        print(f"  ✓ Loading project config: {project_config_path}")
        try:
            with open(project_config_path, 'r', encoding='utf-8') as f:
                project_overrides = yaml.safe_load(f) or {}
            config = merge_config(config, project_overrides)
        except Exception as e:
            print(f"  ⚠ Failed to load project config: {e}")

    # Make paths project-relative
    config = make_paths_project_relative(config, project_dir)

    return config


def merge_config(config: 'Config', overrides: dict) -> 'Config':
    """
    Merge override dict into config object.

    Handles nested configuration sections like 'keyword', 'download', etc.

    Args:
        config: Base configuration object
        overrides: Dictionary of overrides

    Returns:
        Updated configuration object
    """
    for section, values in overrides.items():
        if hasattr(config, section):
            section_obj = getattr(config, section)
            if isinstance(values, dict):
                for key, value in values.items():
                    if hasattr(section_obj, key):
                        setattr(section_obj, key, value)
            else:
                setattr(config, section, values)
    return config


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
