#!/usr/bin/env python3
"""
Voiceover-Matcher: End-to-End Documentary Footage Pipeline v3.0

One command to go from voiceover → matched timeline:
1. Extract keywords from voiceover (LLM)
2. Download footage from YouTube + Pexels + Pixabay
3. Transcribe & index videos (parallel + delta-aware)
4. Match to voiceover segments (with confidence enforcement)
5. Output DaVinci Resolve timeline (multiple styles)

Usage:
    python main.py --voiceover script.srt
    python main.py --voiceover script.srt --keywords 30
    python main.py --project "E:\\Projects\\MyDoc" --voiceover voiceover.srt
    python main.py --resume                    # Resume interrupted run
    python main.py --fresh                     # Force fresh start
    python main.py --use-keywords              # Use most recent saved keywords
    python main.py --use-keywords mypreset     # Use specific keyword preset
    python main.py --save-keywords             # Save keywords after extraction
    python main.py --list-keywords             # List saved keyword presets
    python main.py --match-only                # Skip download, just match existing
    python main.py --config custom_config.yaml # Use custom config file
"""
from __future__ import annotations

import os
import sys

# Suppress FFmpeg H.264 decoder warnings from OpenCV (must be set before cv2 import)
os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "-8"  # AV_LOG_QUIET
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"

# Fix Windows console encoding for Unicode characters
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# =============================================================================
# PATH SETUP
# =============================================================================

# Determine install directory (where this script lives)
INSTALL_DIR = Path(__file__).parent.resolve()

# Add src to path
sys.path.insert(0, str(INSTALL_DIR / 'src'))

# Project directory (set later via --project argument)
PROJECT_DIR = None

# =============================================================================
# IMPORTS FROM CLI PACKAGE
# =============================================================================

from src.cli import (
    parse_arguments,
    load_environment,
    validate_root_directories,
    load_project_config,
    validate_config_at_startup,
    setup_logging,
    find_voiceover_interactive,
)
from src.config import (
    Config, load_config, get_config, set_config,
    ensure_dirs,
)
from src.keywords import KeywordManager
from src.pipeline_events import VerboseTraceHandler, EventTracer

# Global config instance - loaded at startup
_config: Optional[Config] = None


def get_pipeline_config() -> Config:
    """Get the pipeline configuration (loads if not already loaded)"""
    global _config
    if _config is None:
        _config = get_config()
    return _config


def _build_effective_config(config):
    """
    Build effective config with all values shown (including defaults applied).

    This shows which config values would be used after defaults are applied,
    helpful for debugging and understanding actual configuration in use.
    """
    def get_value(obj, attr, default=None):
        """Safely get attribute value from object or return default."""
        val = getattr(obj, attr, default)
        # Filter out private/internal attributes
        if attr.startswith('_'):
            return default
        return val

    def serialize_section(section, section_name):
        """Recursively serialize a config section."""
        if section is None:
            return None

        result = {}
        # Check if it's a dataclass
        if hasattr(section, '__dataclass_fields__'):
            for field_name in section.__dataclass_fields__:
                value = getattr(section, field_name, None)
                if hasattr(value, '__dataclass_fields__'):
                    # Nested dataclass
                    result[field_name] = serialize_section(value, field_name)
                elif hasattr(value, '__dict__') and not isinstance(value, (str, int, float, bool, list, dict)):
                    # Object with attributes
                    nested = {}
                    for attr in dir(value):
                        if not attr.startswith('_') and not callable(getattr(value, attr, None)):
                            try:
                                nested[attr] = getattr(value, attr, None)
                            except Exception:
                                pass
                    result[field_name] = nested if nested else value
                else:
                    result[field_name] = value
        elif isinstance(section, dict):
            for key, value in section.items():
                if hasattr(value, '__dataclass_fields__'):
                    result[key] = serialize_section(value, key)
                else:
                    result[key] = value
        else:
            # Primitive or simple object
            result = section

        return result

    # Get key sections with their effective values
    effective = {
        "project": serialize_section(getattr(config, 'project', None), 'project'),
        "video_search": serialize_section(getattr(config, 'video_search', None), 'video_search'),
        "matching": serialize_section(getattr(config, 'matching', None), 'matching'),
        "iterative_matching": serialize_section(getattr(config, 'iterative_matching', None), 'iterative_matching'),
        "download": serialize_section(getattr(config, 'download', None), 'download'),
        "transcription": serialize_section(getattr(config, 'transcription', None), 'transcription'),
        "embedding": serialize_section(getattr(config, 'embedding', None), 'embedding'),
        "output": serialize_section(getattr(config, 'output', None), 'output'),
    }

    return effective


def _build_effective_config_with_sources(config):
    """
    Build effective config with source tracking (env var, yaml, or default).

    Returns a dict with structure:
    {
        'section_name': {
            'source': 'env|yaml|default',
            'fields': {
                'field_name': {
                    'value': ...,
                    'source': 'env|yaml|default'
                },
                ...
            },
            'nested': { ... }
        },
        ...
    }
    """
    import os
    from dataclasses import fields, is_dataclass
    from typing import get_type_hints

    def get_value(obj, attr, default=None):
        """Safely get attribute value from object or return default."""
        val = getattr(obj, attr, default)
        if attr.startswith('_'):
            return default
        return val

    def serialize_section_with_sources(section, section_name, source='yaml'):
        """Recursively serialize a config section with source tracking."""
        if section is None:
            return None

        result = {'source': source, 'fields': {}, 'nested': {}}

        if hasattr(section, '__dataclass_fields__'):
            for field_name in section.__dataclass_fields__:
                value = getattr(section, field_name, None)
                if hasattr(value, '__dataclass_fields__'):
                    # Nested dataclass
                    result['nested'][field_name] = serialize_section_with_sources(value, field_name, source)
                elif hasattr(value, '__dict__') and not isinstance(value, (str, int, float, bool, list, dict)):
                    # Object with attributes
                    nested = {}
                    for attr in dir(value):
                        if not attr.startswith('_') and not callable(getattr(value, attr, None)):
                            try:
                                nested[attr] = {'value': getattr(value, attr, None), 'source': source}
                            except Exception:
                                pass
                    result['fields'][field_name] = {'value': nested if nested else value, 'source': source}
                else:
                    result['fields'][field_name] = {'value': value, 'source': source}
        elif isinstance(section, dict):
            for key, value in section.items():
                if hasattr(value, '__dataclass_fields__'):
                    result['nested'][key] = serialize_section_with_sources(value, key, source)
                else:
                    result['fields'][key] = {'value': value, 'source': source}
        else:
            return {'value': section, 'source': source}

        return result

    # Track which sections were overridden by env vars
    env_sections = set()
    for env_name in os.environ:
        if env_name.startswith('MATCHER_'):
            # Extract section name
            suffix = env_name[len('MATCHER_'):]
            parts = suffix.split('_')
            if parts:
                env_sections.add(parts[0].lower())

    # Get key sections with their effective values and sources
    section_names = [
        'project', 'video_search', 'matching', 'iterative_matching',
        'download', 'transcription', 'embedding', 'output', 'cache',
        'pipeline', 'infrastructure', 'rate_limit', 'healing'
    ]

    effective = {}
    for section_name in section_names:
        section = getattr(config, section_name, None)
        if section is not None:
            # Check if this section has env override
            source = 'env' if section_name in env_sections else 'yaml'
            effective[section_name] = serialize_section_with_sources(section, section_name, source)

    return effective


def _print_config_with_sources(effective_config, indent=0):
    """Print config with sources in human-readable format."""
    prefix = "  " * (indent + 1)

    for section_name, section_data in effective_config.items():
        if not isinstance(section_data, dict):
            continue

        source = section_data.get('source', 'yaml')
        source_marker = _get_source_marker(source)

        print(f"{prefix}{section_name.upper()} [{source_marker}]")

        # Print fields
        fields = section_data.get('fields', {})
        for field_name, field_data in fields.items():
            if isinstance(field_data, dict) and 'value' in field_data:
                value = field_data['value']
                field_source = field_data.get('source', 'yaml')
                field_marker = _get_source_marker(field_source)

                # Truncate long values for display
                value_str = str(value)
                if len(value_str) > 50:
                    value_str = value_str[:47] + "..."

                print(f"{prefix}  {field_name}: {value_str} {field_marker}")
            else:
                # Direct value
                value_str = str(field_data)
                if len(value_str) > 50:
                    value_str = value_str[:47] + "..."
                print(f"{prefix}  {field_name}: {value_str}")

        # Print nested sections
        nested = section_data.get('nested', {})
        if nested:
            for nested_name, nested_data in nested.items():
                if isinstance(nested_data, dict):
                    nested_source = nested_data.get('source', 'yaml')
                    nested_marker = _get_source_marker(nested_source)
                    print(f"{prefix}  {nested_name.upper()} [{nested_marker}]:")
                    _print_nested_fields(nested_data, indent + 2)


def _print_nested_fields(nested_data, indent):
    """Print nested config fields."""
    prefix = "  " * indent

    fields = nested_data.get('fields', {})
    for field_name, field_data in fields.items():
        if isinstance(field_data, dict) and 'value' in field_data:
            value = field_data['value']
            field_source = field_data.get('source', 'yaml')
            field_marker = _get_source_marker(field_source)

            value_str = str(value)
            if len(value_str) > 50:
                value_str = value_str[:47] + "..."

            print(f"{prefix}{field_name}: {value_str} {field_marker}")
        else:
            value_str = str(field_data)
            if len(value_str) > 50:
                value_str = value_str[:47] + "..."
            print(f"{prefix}{field_name}: {value_str}")


def _get_source_marker(source):
    """Get visual marker for config source."""
    markers = {
        'env': '[ENV]',
        'yaml': '[YAML]',
        'default': '[DEFAULT]'
    }
    return markers.get(source, '[?]')


def _get_env_overrides_summary():
    """Get summary of environment variable overrides applied."""
    import os
    overrides = []
    prefix = "MATCHER_"

    for env_name, env_value in os.environ.items():
        if env_name.startswith(prefix):
            # Parse to human-readable format
            suffix = env_name[len(prefix):]
            overrides.append(f"{suffix}={env_value}")

    return overrides


def _preload_cached_data_for_match_only(pipeline, config):
    """
    Preload cached data for match-only mode when checkpoint data is incomplete.

    This handles the case where the pipeline completed but the checkpoint wasn't
    updated properly. It loads embeddings and text_metadata from disk cache.
    """
    import logging
    import numpy as np
    logger = logging.getLogger(__name__)

    state = pipeline.state
    cache_dir = Path(config.cache.cache_dir)

    # 1. Load audio files from disk if not in state
    if not getattr(state, 'downloaded_audio', None) and not getattr(state, 'downloaded_videos', None):
        from src.state import AudioDownload
        root_dir = getattr(config.download, 'root_dir', None)
        if root_dir:
            root_path = Path(root_dir)
            # Find project-specific subdirectory
            for subdir in root_path.iterdir():
                if subdir.is_dir():
                    audio_files = []
                    for audio_dir in subdir.rglob('*_audio'):
                        for mp3 in audio_dir.glob('*.mp3'):
                            audio_files.append(AudioDownload(
                                file=str(mp3),
                                url="",
                                video_id=mp3.stem,
                                title=mp3.stem,
                                duration=0.0,
                                keyword=audio_dir.name.replace('_audio', '').replace('_l', '').replace('_m', '')
                            ))
                    if audio_files:
                        state.downloaded_audio = audio_files
                        logger.info(f"Preloaded {len(audio_files)} audio files from disk")
                        break

    # 2. Load transcripts from cache
    transcripts_dir = cache_dir / 'transcriptions'
    if transcripts_dir.exists() and not getattr(state, 'transcripts', None):
        import json
        transcripts = {}
        for json_file in transcripts_dir.glob('*.json'):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    segments = None
                    if isinstance(data, list):
                        segments = data
                    elif isinstance(data, dict) and 'segments' in data:
                        segments = data['segments']

                    if segments:
                        # Extract actual video/audio path from segment data
                        # (cache files are named by hash, but contain source_file)
                        video_path = None
                        for seg in segments:
                            if isinstance(seg, dict):
                                video_path = seg.get('source_file') or seg.get('video_path')
                                if video_path:
                                    break

                        # Use actual path as key (fall back to filename if not found)
                        key = video_path if video_path else json_file.stem
                        transcripts[key] = segments
            except Exception:
                pass
        if transcripts:
            state.transcripts = transcripts
            logger.info(f"Preloaded {len(transcripts)} transcripts from cache")

    # 3. Embeddings are loaded during TranscribeStage.restore() via _compute_embeddings
    # NOTE: We intentionally do NOT preload text_metadata or embeddings from the embedding cache
    # here because the embedding cache files don't contain actual video paths - they only
    # have hash-based filenames. Loading them here would set incorrect video_path values
    # and prevent proper B-roll flag propagation. The TranscribeStage._rebuild_text_metadata()
    # method correctly extracts video paths from transcript cache files.
    # See: https://github.com/project/issues/XXX - B-roll propagation fix


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def _display_error_stats(pipeline_agg):
    """Display error frequency analytics to console (US-120-005).

    Args:
        pipeline_agg: PipelineErrorAggregator instance
    """
    report = pipeline_agg.generate_frequency_report(top_n=10)
    temporal = pipeline_agg.get_temporal_patterns()

    print("\n" + "=" * 60)
    print("ERROR FREQUENCY ANALYTICS")
    print("=" * 60)

    # Overall stats
    print(f"\nTotal Errors: {report['total_errors']}")
    print(f"Stages with errors: {len(report.get('stages_with_errors', []))}")

    # Top errors
    overall = report.get('overall', {})
    top_errors = overall.get('top_errors', [])
    if top_errors:
        print("\nTop Error Patterns:")
        for i, err in enumerate(top_errors, 1):
            print(f"  {i}. [{err['count']} errors, {err['percentage']}%] {err['sample'][:60]}")

    # Category breakdown
    category_breakdown = overall.get('category_breakdown', {})
    if category_breakdown:
        print("\nCategory Breakdown:")
        for cat, count in sorted(category_breakdown.items(), key=lambda x: x[1], reverse=True):
            pct = overall.get('percentages', {}).get(cat, 0)
            print(f"  {cat:<20}: {count:>4} ({pct:>5.1f}%)")

    # Temporal patterns
    if temporal.get('total_temporal_errors', 0) > 0:
        print("\nTemporal Patterns:")

        time_of_day = temporal.get('time_of_day', {})
        if time_of_day:
            print("  Time of Day:")
            for bucket, count in time_of_day.items():
                print(f"    {bucket:<12}: {count}")

        day_of_week = temporal.get('day_of_week', {})
        if day_of_week:
            print("  Day of Week:")
            for day, count in day_of_week.items():
                print(f"    {day:<12}: {count}")

    print("=" * 60)


def _export_error_stats(path: str, pipeline_agg, project_dir):
    """Export error stats as JSON file (US-120-005).

    Args:
        path: Output file path for JSON export
        pipeline_agg: PipelineErrorAggregator instance
        project_dir: Project directory
    """
    import json
    from pathlib import Path

    # Resolve output path
    output_path = Path(path)
    if not output_path.is_absolute():
        output_path = project_dir / path

    # Get error stats
    stats = pipeline_agg.export_as_json(include_temporal=True)

    # Add metadata
    stats['exported_at'] = __import__('datetime').datetime.now().isoformat()
    stats['project_dir'] = str(project_dir)

    # Write to file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, default=str)

    print(f"\n  ✓ Error stats exported to {output_path}")


# =============================================================================
# CONFIG TEMPLATE FUNCTIONS
# =============================================================================

def get_template_dir():
    """Get the config templates directory."""
    # Check in install dir first, then relative to current file
    install_dir = Path(__file__).parent
    template_dir = install_dir / "config" / "templates"
    if template_dir.exists():
        return template_dir

    # Also try current working directory
    cwd_template_dir = Path.cwd() / "config" / "templates"
    if cwd_template_dir.exists():
        return cwd_template_dir

    return None


def list_available_templates():
    """List all available config templates."""
    template_dir = get_template_dir()
    if template_dir is None or not template_dir.exists():
        return []

    templates = []
    for f in template_dir.glob("template-*.yaml"):
        name = f.stem.replace("template-", "")
        # Read header comment for description
        try:
            with open(f, 'r', encoding='utf-8') as fp:
                content = fp.read(500)
                # Extract description from comment
                desc = ""
                for line in content.split('\n'):
                    if line.startswith('# Purpose:'):
                        desc = line.replace('# Purpose:', '').strip()
                        break
                    if line.startswith('# Best for:'):
                        desc = line.replace('# Best for:', '').strip()
                        break
        except Exception:
            desc = ""

        templates.append({
            'name': name,
            'file': str(f),
            'description': desc
        })

    return templates


def print_available_templates():
    """Print available templates and exit."""
    templates = list_available_templates()

    print("\n  Available config templates:")
    print("  " + "-" * 50)

    if not templates:
        print("  No templates found in config/templates/")
        print("\n  Create templates as template-<name>.yaml in config/templates/")
        sys.exit(0)

    for t in templates:
        desc = t['description'] or "(no description)"
        print(f"  {t['name']:12} - {desc}")

    print("\n  Usage:")
    print("    python main.py --use-template fast")
    print("    python main.py --use-template quality --config custom.yaml")
    print("\n  Use --config with -c to override template values")
    sys.exit(0)


def get_profile_dir():
    """Find the profiles directory."""
    # Check relative to config.yaml location first
    config_dir = Path("config")
    if (config_dir / "profiles").exists():
        return config_dir / "profiles"

    # Check relative to current directory
    if Path("config/profiles").exists():
        return Path("config/profiles")

    # Check relative to script location
    script_dir = Path(__file__).parent
    if (script_dir / "config" / "profiles").exists():
        return script_dir / "config" / "profiles"

    return None


def list_available_profiles():
    """List available config profiles."""
    import yaml

    profile_dir = get_profile_dir()

    if profile_dir is None or not profile_dir.exists():
        return []

    profiles = []
    for f in sorted(profile_dir.glob("*.yaml")):
        try:
            with open(f, 'r', encoding='utf-8') as fp:
                data = yaml.safe_load(fp) or {}
                profiles.append({
                    'name': data.get('name', f.stem),
                    'description': data.get('description', ''),
                    'inherits_from': data.get('inherits_from', 'config.yaml'),
                    'file': str(f)
                })
        except Exception:
            continue

    return profiles


def print_available_profiles():
    """Print available profiles and exit."""
    profiles = list_available_profiles()

    print("\n  Available config profiles:")
    print("  " + "-" * 50)

    if not profiles:
        print("  No profiles found in config/profiles/")
        print("\n  Create profiles as <name>.yaml in config/profiles/")
        sys.exit(0)

    for p in profiles:
        desc = p['description'] or "(no description)"
        inherits = p['inherits_from']
        print(f"  {p['name']:12} - {desc} (inherits from: {inherits})")

    print("\n  Usage:")
    print("    python main.py --profile dev")
    print("    python main.py --profile prod")
    print("    python main.py --profile staging --config custom.yaml")
    print("\n  Use --config with -c to override profile values")
    sys.exit(0)


def load_template(template_name: str) -> dict:
    """
    Load a config template by name.

    Args:
        template_name: Name of template (e.g., 'fast', 'quality', 'debug')

    Returns:
        Dictionary of template config values

    Raises:
        FileNotFoundError: If template not found
    """
    # Use SafeLoader to properly resolve YAML anchors
    try:
        from yaml import CSafeLoader as SafeLoader
    except ImportError:
        from yaml import SafeLoader

    template_dir = get_template_dir()
    if template_dir is None:
        raise FileNotFoundError(f"Template directory not found")

    template_file = template_dir / f"template-{template_name}.yaml"
    if not template_file.exists():
        raise FileNotFoundError(
            f"Template '{template_name}' not found. "
            f"Available: {[t['name'] for t in list_available_templates()]}"
        )

    import yaml
    with open(template_file, 'r', encoding='utf-8') as f:
        return yaml.load(f, Loader=SafeLoader) or {}


def merge_configs(base: dict, override: dict) -> dict:
    """
    Merge two config dictionaries recursively.
    Override values take precedence over base values.

    Args:
        base: Base config (template)
        override: Override config (user config)

    Returns:
        Merged config dictionary
    """
    import copy
    result = copy.deepcopy(base)

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value

    return result


def load_config_with_template(config_path: str, template_name: str = None,
                              skip_final_validation: bool = False) -> "Config":
    """
    Load config with optional template as base.

    Args:
        config_path: Path to user config file
        template_name: Optional template name to use as base
        skip_final_validation: Skip API key and constraint validation

    Returns:
        Loaded Config object
    """
    from src.config import Config

    # If template specified, use template as base config (not merged with config.yaml)
    if template_name:
        template_dir = get_template_dir()
        if template_dir is None:
            raise FileNotFoundError("Template directory not found")

        template_file = template_dir / f"template-{template_name}.yaml"
        if not template_file.exists():
            raise FileNotFoundError(
                f"Template '{template_name}' not found. "
                f"Available: {[t['name'] for t in list_available_templates()]}"
            )

        # Load template directly as config
        config = Config.from_yaml(str(template_file),
                                  skip_final_validation=skip_final_validation)
        print(f"  Applied template '{template_name}'")
        return config
    else:
        # Standard config loading
        config = Config.from_yaml(config_path)
        return config


def main():
    """Main entry point"""
    global PROJECT_DIR, _config

    args = parse_arguments()

    # Handle --list-templates BEFORE loading config
    if getattr(args, 'list_templates', False):
        print_available_templates()

    # Handle --list-profiles BEFORE loading config
    if getattr(args, 'list_profiles', False):
        print_available_profiles()

    # Handle --use-template - prepare for template loading later
    template_name = getattr(args, 'use_template', None)

    # Handle --profile - prepare for profile loading later
    profile_name = getattr(args, 'profile', None)

    # Handle --dump-dependency-graph BEFORE loading config (US-89-011)
    if getattr(args, 'dump_dependency_graph', False):
        from src.stages import (
            generate_dot_graph,
            print_dependency_summary,
            validate_no_cycles
        )
        fmt = getattr(args, 'dependency_graph_format', 'dot')
        if fmt == 'dot':
            print(generate_dot_graph())
        else:
            print(print_dependency_summary())
        # Also check for cycles
        cycle_error = validate_no_cycles()
        if cycle_error:
            print(f"\n  ERROR: {cycle_error}", file=sys.stderr)
            sys.exit(1)
        sys.exit(0)

    # Handle --export-pipeline-graph BEFORE loading config (US-125-012)
    export_pipeline_graph_path = getattr(args, 'export_pipeline_graph', None)
    if export_pipeline_graph_path:
        # Determine project directory for relative path resolution
        if args.project:
            project_dir = Path(args.project).resolve()
        else:
            project_dir = Path.cwd()
        _export_pipeline_graph(export_pipeline_graph_path, project_dir)
        sys.exit(0)

    # Determine project directory
    if args.project:
        PROJECT_DIR = Path(args.project).resolve()
    else:
        PROJECT_DIR = Path.cwd()

    # Load environment variables
    print("\n  Loading environment...")
    load_environment(PROJECT_DIR)

    # Load configuration
    print(f"  Loading configuration from {args.config}...")

    config_path = Path(args.config)
    if not config_path.is_absolute():
        # Try project dir first, then install dir
        if (PROJECT_DIR / args.config).exists():
            config_path = PROJECT_DIR / args.config
        elif (INSTALL_DIR / args.config).exists():
            config_path = INSTALL_DIR / args.config

    # For --dry-run-config, skip final validation to show all config values and validation errors
    skip_final_validation = getattr(args, 'dry_run_config', False)

    if args.project:
        config = load_project_config(PROJECT_DIR, config_path)
    else:
        # Use profile if specified (overrides base config.yaml)
        if profile_name:
            from src.config import Config
            config = Config.load_with_profile(str(config_path), profile_name,
                                              skip_final_validation=skip_final_validation)
            print(f"  Applied profile '{profile_name}'")
        # Use template if specified (replaces entire config)
        elif template_name:
            config = load_config_with_template(str(config_path), template_name,
                                               skip_final_validation=skip_final_validation)
        else:
            config = load_config(str(config_path), skip_final_validation=skip_final_validation)

    _config = config
    set_config(config)

    # Validate root directories (E:/v, E:/i, etc.) and create them if needed
    # Skip for --health-check, --circuit-status, --escalation-status, --dry-run-config, --validate-youtube-api to allow diagnostics to run even without proper dirs
    skip_validation = getattr(args, 'health_check', False) or getattr(args, 'circuit_status', False) or getattr(args, 'escalation_status', False) or getattr(args, 'dry_run_config', False) or getattr(args, 'validate_youtube_api', False)
    if not skip_validation:
        validate_root_directories(config)

    # Apply --non-interactive flag
    if hasattr(args, 'non_interactive') and args.non_interactive:
        config.enhanced.non_interactive = True

    # Apply --reset-budget flag (US-42-012)
    if hasattr(args, 'reset_budget') and args.reset_budget:
        config.download.reset_budget = True

    # Apply --reset-section flag (US-142-008)
    if hasattr(args, 'reset_section') and args.reset_section:
        section_name = args.reset_section
        success = config.reset_section(section_name)
        if success:
            print(f"[CONFIG] Reset section '{section_name}' to defaults")
        else:
            print(f"[CONFIG] Unknown section '{section_name}'. Available sections: "
                  "matching, download, video_search, etc.")
            return 1

    # Apply caption-first mode CLI flags (handle dict or object config - Rule 6)
    caption_first = config.download.caption_first
    if getattr(args, 'caption_first', False):
        if isinstance(caption_first, dict):
            caption_first['enabled'] = True
        else:
            caption_first.enabled = True
        print("  Caption-first mode enabled via --caption-first")

    if getattr(args, 'caption_language', None):
        if isinstance(caption_first, dict):
            caption_first['preferred_language'] = args.caption_language
        else:
            caption_first.preferred_language = args.caption_language
        print(f"  Caption language set to '{args.caption_language}' via --caption-language")

    if getattr(args, 'no_caption_fallback', False):
        if isinstance(caption_first, dict):
            caption_first['fallback_to_transcription'] = False
        else:
            caption_first.fallback_to_transcription = False
        print("  Caption fallback disabled via --no-caption-fallback")

    # YouTube Data API CLI flags (US-146-010, US-148-005, US-154-011)
    youtube_api = config.download.youtube_api
    youtube_api_arg = getattr(args, 'youtube_api', False)
    youtube_api_enabled_arg = getattr(args, 'youtube_api_enabled', False)
    no_youtube_api_arg = getattr(args, 'no_youtube_api', False)
    force_yt_dlp_arg = getattr(args, 'force_yt_dlp', False)
    youtube_api_key_arg = getattr(args, 'youtube_api_key', None)

    # Combine --youtube-api and --youtube-api-enabled flags
    enable_flag = youtube_api_arg or youtube_api_enabled_arg

    # US-154-011: Handle --force-yt-dlp flag (same as --no-youtube-api)
    if force_yt_dlp_arg:
        no_youtube_api_arg = True

    if enable_flag or no_youtube_api_arg:
        if isinstance(youtube_api, dict):
            youtube_api['enabled'] = enable_flag and not no_youtube_api_arg
        else:
            youtube_api.enabled = enable_flag and not no_youtube_api_arg
        if enable_flag:
            print("  YouTube Data API enabled via --youtube-api")
        elif force_yt_dlp_arg:
            print("  YouTube Data API disabled via --force-yt-dlp")
        else:
            print("  YouTube Data API disabled via --no-youtube-api")

    if youtube_api_key_arg:
        if isinstance(youtube_api, dict):
            youtube_api['api_key'] = youtube_api_key_arg
        else:
            youtube_api.api_key = youtube_api_key_arg
        print(f"  YouTube Data API key set via --youtube-api-key")

    # US-150-003: Handle --youtube-api-keys for multi-key rotation
    youtube_api_keys_arg = getattr(args, 'youtube_api_keys', None)
    if youtube_api_keys_arg:
        # Parse comma-separated keys
        keys = [k.strip() for k in youtube_api_keys_arg.split(',') if k.strip()]
        if keys:
            if isinstance(youtube_api, dict):
                youtube_api['api_keys'] = keys
            else:
                youtube_api.api_keys = keys
            print(f"  YouTube Data API keys set via --youtube-api-keys ({len(keys)} keys)")

    # US-148-007: Handle --reset-youtube-quota flag
    reset_youtube_quota_arg = getattr(args, 'reset_youtube_quota', False)
    if reset_youtube_quota_arg:
        from src.downloader.api_fallback_handler import set_youtube_quota_reset_requested
        set_youtube_quota_reset_requested(True)
        print(f"  YouTube API quota reset requested via --reset-youtube-quota")

    # US-148-005: Support environment variable for YouTube API key (MATCHER_YOUTUBE_API_KEY)
    # This is a shortcut env var that doesn't require the full download.youtube_api.api_key path
    env_youtube_api_key = os.environ.get('MATCHER_YOUTUBE_API_KEY')
    env_youtube_api_enabled = os.environ.get('MATCHER_YOUTUBE_API_ENABLED')

    # Apply env var overrides (CLI args take precedence)
    if env_youtube_api_key and not youtube_api_key_arg:
        if isinstance(youtube_api, dict):
            youtube_api['api_key'] = env_youtube_api_key
        else:
            youtube_api.api_key = env_youtube_api_key
        print(f"  YouTube Data API key set via MATCHER_YOUTUBE_API_KEY env var")

    if env_youtube_api_enabled is not None:
        enabled = env_youtube_api_enabled.lower() in ('true', '1', 'yes')
        # Only override if CLI flags weren't used
        if not youtube_api_arg and not no_youtube_api_arg:
            if isinstance(youtube_api, dict):
                youtube_api['enabled'] = enabled
            else:
                youtube_api.enabled = enabled
            if enabled:
                print(f"  YouTube Data API enabled via MATCHER_YOUTUBE_API_ENABLED env var")
            else:
                print(f"  YouTube Data API disabled via MATCHER_YOUTUBE_API_ENABLED env var")

    # Model version pinning (US-124-010)
    if getattr(args, 'transcription_model_version', None):
        transcription_cfg = config.transcription
        if isinstance(transcription_cfg, dict):
            transcription_cfg['model_version'] = args.transcription_model_version
        else:
            transcription_cfg.model_version = args.transcription_model_version
        print(f"  Model version pinned to '{args.transcription_model_version}' via --transcription-model-version")

    # US-130-007: Handle --hot-backup CLI flag
    if hasattr(args, 'hot_backup'):
        hot_backup_arg = args.hot_backup
        if hot_backup_arg is not None:
            # Get pipeline config (handle dict/object compatibility)
            pipeline_config = getattr(config, 'pipeline_config', None)
            if not pipeline_config:
                pipeline_config = getattr(config, 'pipeline', None)

            if pipeline_config:
                # Get or create hot_backup config
                hot_backup_cfg = getattr(pipeline_config, 'hot_backup', None)
                if isinstance(hot_backup_cfg, dict):
                    # Convert to object if needed
                    from src.config.sections.infrastructure import CheckpointHotBackupConfig
                    hot_backup_cfg = CheckpointHotBackupConfig(**hot_backup_cfg)
                    pipeline_config.hot_backup = hot_backup_cfg

                if hot_backup_arg == 'off':
                    # Disable hot backup
                    hot_backup_cfg.enabled = False
                    print("  Hot backup disabled via --hot-backup off")
                elif hot_backup_arg == 'enabled':
                    # Enable using config path (or empty = use as-is)
                    if hot_backup_cfg.hot_backup_path:
                        hot_backup_cfg.enabled = True
                        print(f"  Hot backup enabled via --hot-backup (path: {hot_backup_cfg.hot_backup_path})")
                    else:
                        print("  Warning: --hot-backup enabled but no hot_backup_path configured")
                else:
                    # Use provided path
                    hot_backup_cfg.enabled = True
                    hot_backup_cfg.hot_backup_path = hot_backup_arg
                    print(f"  Hot backup enabled via --hot-backup {hot_backup_arg}")

    # Setup logging with dual log files (normal + verbose)
    if hasattr(args, 'project') and args.project:
        log_output_dir = Path(args.project)
    else:
        log_output_dir = Path(config.output.output_dir)

    logger = setup_logging(
        config,
        output_dir=log_output_dir,
        json_logs=getattr(args, 'json_logs', False)
    )

    # Print log file locations
    if hasattr(logger, 'log_paths'):
        print(f"\n  📝 Log files:")
        print(f"    Normal:  {Path(logger.log_paths['normal']).name}")
        print(f"    Verbose: {Path(logger.log_paths['verbose']).name}")
        if logger.log_paths.get('json'):
            print(f"    JSON:    {Path(logger.log_paths['json']).name}")

    # Validate config (--validate-config or --validate-config-json)
    if args.validate_config or args.validate_config_json:
        import json as json_module

        # Check for JSON output mode
        json_output = args.validate_config_json

        # Collect validation results
        errors = config.validate()
        warnings = []

        # Run startup validation to get warnings about deprecated options
        # (but don't exit on errors, just collect them)
        try:
            # Temporarily redirect print to capture warnings
            import io
            old_stdout = sys.stdout
            sys.stdout = io.StringIO()
            try:
                # This will print warnings but we capture them
                validate_config_at_startup(config)
            finally:
                warnings_output = sys.stdout.getvalue()
                sys.stdout = old_stdout

            # Extract warnings from the captured output
            if "deprecated" in warnings_output.lower() or "warning" in warnings_output.lower():
                for line in warnings_output.strip().split('\n'):
                    if line.strip():
                        warnings.append(line.strip())
        except SystemExit:
            pass  # Ignore sys.exit from validate_config_at_startup

        # Build effective config (values with defaults applied)
        effective_config = _build_effective_config(config)

        # US-142-009: Get file reference validation results
        file_references = config.validate_file_references()

        # Prepare result - use "valid" as bool per acceptance criteria
        is_valid = len(errors) == 0
        result = {
            "valid": is_valid,
            "config_loaded": True,
            "errors": errors,
            "warnings": warnings,
            "file_references": file_references,
            "effective_config": effective_config,
        }

        if json_output:
            # JSON output mode
            print(json_module.dumps(result, indent=2))
            # Exit codes: 0 if valid, 1 if errors, 2 if warnings only
            if errors:
                sys.exit(1)
            elif warnings:
                sys.exit(2)
            else:
                sys.exit(0)
        else:
            # Human-readable output
            print("\n  Configuration Validation:")
            print("    ✓ Config loaded successfully")
            print(f"    Validation: {'✓ Valid' if is_valid else '⚠ Invalid'}")

            if warnings:
                print("\n  Warnings:")
                for w in warnings:
                    print(f"    - {w}")

            # US-142-009: Show file reference validation results
            if file_references:
                print("\n  File References:")
                for fr in file_references:
                    print(f"    - [{fr['type']}] {fr['field']}: {fr['path']}")
                    print(f"      {fr['suggestion']}")

            if errors:
                print("\n  Errors:")
                for e in errors:
                    print(f"    - {e}")
                sys.exit(1)
            elif warnings:
                print("\n  ⚠ Configuration has warnings but no errors")
                sys.exit(2)
            else:
                print("\n  ✓ Configuration is valid")
                sys.exit(0)

    # Handle --dry-run-config (US-128-009): Show effective config values with sources
    if getattr(args, 'dry_run_config', False):
        import json as json_module

        # Collect validation results
        errors = config.validate()
        warnings = []

        # Run startup validation to get warnings
        try:
            import io
            old_stdout = sys.stdout
            sys.stdout = io.StringIO()
            try:
                validate_config_at_startup(config)
            finally:
                warnings_output = sys.stdout.getvalue()
                sys.stdout = old_stdout

            if "deprecated" in warnings_output.lower() or "warning" in warnings_output.lower():
                for line in warnings_output.strip().split('\n'):
                    if line.strip():
                        warnings.append(line.strip())
        except SystemExit:
            pass

        # Build effective config with source tracking
        effective_config = _build_effective_config_with_sources(config)

        is_valid = len(errors) == 0

        # Print human-readable output
        print("\n" + "=" * 60)
        print("  CONFIG DRY-RUN: Effective Values with Sources")
        print("=" * 60)

        print("\n  Configuration Sections:")
        print("-" * 40)

        # Print each section with its values and sources
        _print_config_with_sources(effective_config)

        print("\n" + "-" * 40)
        print("\n  Validation Results:")
        print(f"    Status: {'✓ Valid' if is_valid else '✗ Invalid'}")

        if warnings:
            print("\n  Warnings:")
            for w in warnings:
                print(f"    - {w}")

        if errors:
            print("\n  Errors:")
            for e in errors:
                print(f"    - {e}")

        # Print summary of env overrides if any
        env_overrides = _get_env_overrides_summary()
        if env_overrides:
            print("\n  Environment Variable Overrides Applied:")
            for override in env_overrides:
                print(f"    - {override}")

        print("\n" + "=" * 60)

        # Exit codes: 0 if valid, 1 if errors
        if errors:
            sys.exit(1)
        else:
            sys.exit(0)

    # Handle --show-precedence (US-142-010): Show config value sources
    show_precedence = getattr(args, 'show_precedence', None)
    if show_precedence is not None:
        import json as json_module

        # Validate config loads
        errors = config.validate()
        if errors:
            print("\n  Config validation errors:")
            for e in errors:
                print(f"    - {e}")
            sys.exit(1)

        # Print header
        print("\n" + "=" * 60)
        print("  CONFIG VALUE PRECEDENCE")
        print("  Precedence order: env > profile > yaml > default")
        print("=" * 60)

        # If specific field requested
        field_path = show_precedence if show_precedence else None
        if field_path:
            # Show specific field
            source_info = config.get_value_source(field_path)
            if source_info:
                print(f"\n  Field: {field_path}")
                print(f"    Value: {source_info['value']}")
                print(f"    Source: {source_info['source']}")
                print(f"    Section: {source_info['section']}")
            else:
                print(f"\n  Field '{field_path}' not found")
                # Suggest similar fields
                all_sources = config.get_all_sources()
                similar = [k for k in all_sources.keys() if field_path.lower() in k.lower()]
                if similar:
                    print(f"  Did you mean: {', '.join(similar[:5])}")
                sys.exit(1)
        else:
            # Show all sources grouped by source type
            all_sources = config.get_all_sources()

            # Group by source
            by_source = {'env': [], 'profile': [], 'yaml': [], 'default': []}
            for field_path_key, info in all_sources.items():
                source = info.get('source', 'default')
                if source in by_source:
                    by_source[source].append((field_path_key, info['value']))

            # Print each source group
            for source_type in ['env', 'profile', 'yaml', 'default']:
                fields = by_source[source_type]
                if fields:
                    print(f"\n  [{source_type.upper()}] ({len(fields)} fields)")
                    print("  " + "-" * 40)
                    # Sort and show first 20
                    for field_key, value in sorted(fields)[:20]:
                        # Truncate long values
                        value_str = str(value)
                        if len(value_str) > 50:
                            value_str = value_str[:47] + "..."
                        print(f"    {field_key}: {value_str}")
                    if len(fields) > 20:
                        print(f"    ... and {len(fields) - 20} more")

            print("\n  " + "=" * 60)
            print(f"  Total tracked fields: {len(all_sources)}")
            print("=" * 60)

        sys.exit(0)

    # Handle --config-load-stats (US-142-011): Show config loading performance metrics
    config_load_stats = getattr(args, 'config_load_stats', False)
    if config_load_stats:
        # Validate config loads
        errors = config.validate()
        if errors:
            print("\n  Config validation errors:")
            for e in errors:
                print(f"    - {e}")
            sys.exit(1)

        # Print header
        print("\n" + "=" * 60)
        print("  CONFIG LOADING PERFORMANCE METRICS")
        print("  Per-phase timing (milliseconds)")
        print("=" * 60)

        # Get load metrics
        metrics = config.get_load_metrics()

        # Print each phase
        phases = ['parse', 'line_tracking', 'interpolation', 'validation',
                  'convert', 'env_overrides', 'final_validation', 'total']
        for phase in phases:
            if phase in metrics:
                duration = metrics[phase]
                bar_len = int(duration / 2)  # Scale for display
                bar = '█' * min(bar_len, 30)
                print(f"  {phase:20s}: {duration:8.2f}ms {bar}")

        print("\n  " + "=" * 60)
        print(f"  Total load time: {metrics.get('total', 0):.2f}ms")
        print("=" * 60)

        sys.exit(0)

    # Handle --checkpoint-export and --checkpoint-import (US-115-009, US-130-011)
    checkpoint_export = getattr(args, 'checkpoint_export', None)
    checkpoint_import = getattr(args, 'checkpoint_import', None)

    if checkpoint_export or checkpoint_import:
        from src.checkpoint import CheckpointManager

        if checkpoint_export:
            # Export checkpoint
            print(f"\n  Exporting checkpoint to: {checkpoint_export}")

            # Parse include_stages
            include_stages = None
            export_stages = getattr(args, 'export_stages', None)
            if export_stages:
                include_stages = [s.strip() for s in export_stages.split(',') if s.strip()]
                print(f"  Exporting stages: {include_stages}")

            # Check for redaction flag (US-130-011)
            redact = getattr(args, 'redact', False)
            if redact:
                print(f"  Redacting sensitive data: URLs, paths, tokens")

            # Create checkpoint manager
            checkpoint_mgr = CheckpointManager(PROJECT_DIR, config_hash=config.hash)

            # Export (with or without redaction)
            if redact:
                success = checkpoint_mgr.export_checkpoint_redacted(
                    checkpoint_export, include_stages=include_stages
                )
            else:
                success = checkpoint_mgr.export_checkpoint(
                    checkpoint_export, include_stages=include_stages
                )
            if success:
                print(f"\n  ✓ Checkpoint exported successfully")
                sys.exit(0)
            else:
                print(f"\n  ✗ Failed to export checkpoint")
                sys.exit(1)

        if checkpoint_import:
            # Import checkpoint
            print(f"\n  Importing checkpoint from: {checkpoint_import}")

            # Determine target directory
            target_dir = getattr(args, 'checkpoint_import_target', None)
            if target_dir:
                print(f"  Target project: {target_dir}")
            else:
                target_dir = str(PROJECT_DIR)

            # Create checkpoint manager
            checkpoint_mgr = CheckpointManager(PROJECT_DIR, config_hash=config.hash)

            # Import
            success = checkpoint_mgr.import_checkpoint(checkpoint_import, target_project_dir=target_dir)
            if success:
                print(f"\n  ✓ Checkpoint imported successfully")
                sys.exit(0)
            else:
                print(f"\n  ✗ Failed to import checkpoint")
                sys.exit(1)

    # Handle --health-check (US-108-005)
    if getattr(args, 'health_check', False):
        from src.health_checker import HealthChecker, HealthStatus
        import json as json_module

        print("\n  Pipeline Health Check")
        print("  " + "=" * 40)

        # Create health checker
        checker = HealthChecker(config)

        # Run all health checks
        results = checker.run_all_checks(project_path=str(PROJECT_DIR) if PROJECT_DIR else None)

        # Print results by category
        critical_failures = []
        warnings = []

        for result in results:
            if result.status == HealthStatus.FAILED:
                critical_failures.append(result)
            elif result.status == HealthStatus.WARNING:
                warnings.append(result)

        # Print results
        print("\n  Critical Checks:")
        if critical_failures:
            for result in critical_failures:
                print(f"    ✗ {result.name}: {result.message}")
        else:
            print("    ✓ All critical checks passed")

        if warnings:
            print("\n  Warnings:")
            for result in warnings:
                print(f"    ⚠ {result.name}: {result.message}")

        # Summary
        ok_count = sum(1 for r in results if r.status == HealthStatus.OK)
        total_count = len(results)
        print(f"\n  Summary: {ok_count}/{total_count} checks passed")

        # Exit with appropriate code
        if critical_failures:
            print("\n  ✗ Health check FAILED - pipeline cannot proceed")
            sys.exit(1)
        elif warnings:
            print("\n  ⚠ Health check passed with warnings - proceeding with caution")
            sys.exit(0)
        else:
            print("\n  ✓ Health check passed - pipeline ready")
            sys.exit(0)

    # Handle --validate-youtube-api (US-150-008)
    if getattr(args, 'validate_youtube_api', False):
        from src.health_checker import HealthChecker, HealthStatus
        from src.downloader.youtube_api_client import YouTubeAPIClient
        import json as json_module
        import time as time_module

        print("\n  YouTube API Validation")
        print("  " + "=" * 40)

        # Get API key from config
        api_key = None
        download_config = getattr(config, 'download', None)
        if download_config:
            if isinstance(download_config, dict):
                api_key = download_config.get('youtube_api_key')
                if not api_key:
                    api_keys_list = download_config.get('youtube_api_keys', [])
                    if api_keys_list:
                        api_key = api_keys_list[0] if api_keys_list else None
            elif hasattr(download_config, 'youtube_api_key'):
                api_key = download_config.youtube_api_key
                if not api_key:
                    api_keys = getattr(download_config, 'youtube_api_keys', [])
                    if api_keys:
                        api_key = api_keys[0] if api_keys else None

        # Also check environment variable as fallback
        if not api_key:
            import os
            api_key = os.environ.get('YOUTUBE_API_KEY')

        if not api_key:
            print("\n  ✗ No YouTube API key configured")
            print("    Please set youtube_api_key in config or YOUTUBE_API_KEY environment variable")
            sys.exit(1)

        print(f"\n  Testing API key...")

        # Create client and run health check
        client = YouTubeAPIClient(
            api_key=api_key,
            quota_limit=10000,
            timeout=15,
            auto_scale_quota=False,
        )

        is_valid, error_message, quota_info = client.health_check()
        client.close()

        if is_valid:
            print(f"\n  ✓ API key is VALID")
            print(f"\n  Quota Status:")
            print(f"    Used: {quota_info.get('quota_used', 0):,}")
            print(f"    Limit: {quota_info.get('quota_limit', 0):,}")
            print(f"    Percent Used: {quota_info.get('percent_used', 0):.1f}%")
            print(f"    Keys Available: {quota_info.get('keys_available', 'N/A')}")
            sys.exit(0)
        else:
            print(f"\n  ✗ API key validation FAILED")
            print(f"    Error: {error_message}")
            if 'quota' in error_message.lower() or 'exceeded' in error_message.lower():
                print(f"\n    Quota Status:")
                print(f"      Used: {quota_info.get('quota_used', 0):,}")
                print(f"      Limit: {quota_info.get('quota_limit', 0):,}")
                print(f"      Percent Used: {quota_info.get('percent_used', 0):.1f}%")
            sys.exit(1)

    # Handle --check-api-health (US-155-006)
    if getattr(args, 'check_api_health', False):
        from src.downloader.youtube_api_client import YouTubeAPIClient

        print("\n  YouTube API Health Check")
        print("  " + "=" * 40)

        # Get API key from config
        api_key = None
        download_config = getattr(config, 'download', None)
        if download_config:
            if isinstance(download_config, dict):
                api_key = download_config.get('youtube_api_key')
                if not api_key:
                    api_keys_list = download_config.get('youtube_api_keys', [])
                    if api_keys_list:
                        api_key = api_keys_list[0] if api_keys_list else None
            elif hasattr(download_config, 'youtube_api_key'):
                api_key = download_config.youtube_api_key
                if not api_key:
                    api_keys = getattr(download_config, 'youtube_api_keys', [])
                    if api_keys:
                        api_key = api_keys[0] if api_keys else None

        # Also check environment variable as fallback
        if not api_key:
            import os
            api_key = os.environ.get('YOUTUBE_API_KEY')

        if not api_key:
            print("\n  ✗ No YouTube API key configured")
            print("    Please set youtube_api_key in config or YOUTUBE_API_KEY environment variable")
            sys.exit(1)

        print(f"\n  Running connectivity health check...")

        # Create client and run health check
        client = YouTubeAPIClient(
            api_key=api_key,
            quota_limit=10000,
            timeout=15,
            auto_scale_quota=False,
        )

        is_valid, error_message, quota_info = client.health_check()
        client.close()

        if is_valid:
            print(f"\n  ✓ API health check PASSED")
            print(f"\n  Connectivity Status: OK")
            print(f"\n  Quota Status:")
            print(f"    Used: {quota_info.get('quota_used', 0):,}")
            print(f"    Limit: {quota_info.get('quota_limit', 0):,}")
            print(f"    Percent Used: {quota_info.get('percent_used', 0):.1f}%")
            print(f"    Keys Available: {quota_info.get('keys_available', 'N/A')}")
            sys.exit(0)
        else:
            print(f"\n  ✗ API health check FAILED")
            print(f"    Error: {error_message}")
            if 'quota' in error_message.lower() or 'exceeded' in error_message.lower():
                print(f"\n    Quota Status:")
                print(f"      Used: {quota_info.get('quota_used', 0):,}")
                print(f"      Limit: {quota_info.get('quota_limit', 0):,}")
                print(f"      Percent Used: {quota_info.get('percent_used', 0):.1f}%")
            elif 'network' in error_message.lower() or 'timeout' in error_message.lower():
                print(f"\n    Please check your network connection and try again.")
            sys.exit(1)

    # Handle --circuit-status (US-120-006)
    if getattr(args, 'circuit_status', False):
        from src.downloader.circuit_breaker import CircuitBreaker, CircuitBreakerConfig
        from src.caption.circuit_breaker import CaptionCircuitBreaker
        import time as time_module

        print("\n  Circuit Breaker Status")
        print("  " + "=" * 40)

        # Show search circuit breaker
        print("\n  [Search Circuit Breaker]")
        search_config = CircuitBreakerConfig()
        search_cb = CircuitBreaker(config=search_config)
        state = search_cb.get_state()
        history = search_cb.get_failure_history()

        print(f"    State: {state.upper()}")
        print(f"    Enabled: {search_cb.config.enabled}")
        print(f"    Consecutive Failures: {search_cb.state.consecutive_failures}")
        print(f"    Total Trips: {search_cb.state.total_trips}")
        print(f"    Total Paused: {search_cb.state.total_paused_seconds:.1f}s")
        print(f"    Threshold: {search_cb.config.consecutive_failures_threshold}")
        print(f"    Pause Duration: {search_cb.config.pause_seconds}s")

        if history:
            print(f"\n    Recent Failures (last {min(10, len(history))}):")
            for i, record in enumerate(history[-10:], 1):
                ts = record.get('timestamp', 0)
                failures = record.get('consecutive_failures', '?')
                readable_time = time_module.strftime('%H:%M:%S', time_module.localtime(ts))
                print(f"      {i}. {readable_time} - {failures} failures")

        # Show caption circuit breaker
        print("\n  [Caption Circuit Breaker]")
        caption_cb = CaptionCircuitBreaker()
        caption_state = caption_cb.get_state()
        caption_history = caption_cb.get_failure_history()

        print(f"    State: {caption_state.upper()}")
        print(f"    Enabled: {caption_cb.is_enabled}")
        print(f"    Consecutive Failures: {caption_cb.state.consecutive_failures}")
        print(f"    Total Trips: {caption_cb.state.total_trips}")

        if caption_history:
            print(f"\n    Recent Failures (last {min(10, len(caption_history))}):")
            for i, record in enumerate(caption_history[-10:], 1):
                ts = record.get('timestamp', 0)
                failures = record.get('consecutive_failures', '?')
                readable_time = time_module.strftime('%H:%M:%S', time_module.localtime(ts))
                print(f"      {i}. {readable_time} - {failures} failures")

        print("\n  ✓ Circuit breaker status displayed")
        sys.exit(0)

    # Handle --escalation-status (US-120-012)
    if getattr(args, 'escalation_status', False):
        from src.health_checker import get_circuit_breaker_health, get_tier_effectiveness_health
        from src.downloader.circuit_breaker import CircuitBreaker, CircuitBreakerConfig
        from src.downloader.per_keyword_circuit_breaker import PerKeywordCircuitBreaker, PerKeywordCircuitBreakerConfig
        import time as time_module

        print("\n  Escalation Health Dashboard")
        print("  " + "=" * 50)

        # Get circuit breaker health
        print("\n  [Circuit Breaker Status]")
        cb_config = CircuitBreakerConfig()
        cb = CircuitBreaker(config=cb_config)
        pkb_config = PerKeywordCircuitBreakerConfig()
        pkb = PerKeywordCircuitBreaker(config=pkb_config)

        cb_health = get_circuit_breaker_health(cb, pkb)

        if cb_health:
            # Main breaker
            if cb_health.get('main_breaker'):
                mb = cb_health['main_breaker']
                print(f"\n    Main Circuit Breaker:")
                print(f"      State:        {mb.get('current_state', 'unknown').upper()}")
                print(f"      Trip Count:   {mb.get('trip_count', 0)}")
                print(f"      Failures:     {mb.get('consecutive_failures', 0)}")
                print(f"      Success:      {mb.get('success_count', 0)}")
                print(f"      Fail Rate:    {mb.get('failure_rate', 0)*100:.1f}%")

            # Per-keyword breaker
            if cb_health.get('per_keyword_breaker'):
                pkb_health = cb_health['per_keyword_breaker']
                print(f"\n    Per-Keyword Circuit Breaker:")
                print(f"      Active Keywords:     {pkb_health.get('active_keywords', 0)}")
                print(f"      Rate-Limited:       {pkb_health.get('rate_limited_keywords', 0)}")
                print(f"      Rate-Limit %:       {pkb_health.get('rate_limit_percentage', 0)*100:.1f}%")
                print(f"      Global Tripped:     {'YES' if pkb_health.get('global_fallback_tripped') else 'No'}")

                # Show individual keyword states (limit to 10)
                kw_states = pkb_health.get('keyword_states', {})
                if kw_states:
                    print(f"\n      Keyword States (showing up to 10):")
                    for i, (kw, state) in enumerate(list(kw_states.items())[:10]):
                        status = "OPEN" if state.get('is_open') else "closed"
                        print(f"        {kw[:30]:30s} [{status:5s}] trips={state.get('total_trips', 0)}")

            # Recommendations
            recommendations = cb_health.get('recommendations', [])
            if recommendations:
                print(f"\n    Recommendations:")
                for rec in recommendations:
                    severity_icon = "✗" if rec.get('severity') == 'critical' else "⚠"
                    print(f"      {severity_icon} {rec.get('message', '')}")
                    print(f"         → {rec.get('action', '')}")
        else:
            print("    No circuit breaker data available")

        # Get tier effectiveness - requires full pipeline context
        print("\n  [Tier Effectiveness]")
        print("    Note: Tier effectiveness metrics require running pipeline")
        print("    with an active EscalationManager. Run the pipeline first")
        print("    to populate tier effectiveness data.")

        print("\n  ✓ Escalation status displayed")
        sys.exit(0)

    # Handle --validate-checkpoint (US-120-007)
    if getattr(args, 'validate_checkpoint', False):
        from src.checkpoint import validate_checkpoint

        if not PROJECT_DIR:
            print("  Error: --validate-checkpoint requires --project")
            sys.exit(1)

        checkpoint_path = PROJECT_DIR / "checkpoint.json"

        if not checkpoint_path.exists():
            print(f"  Error: Checkpoint not found: {checkpoint_path}")
            sys.exit(1)

        # Run validation
        result = validate_checkpoint(str(checkpoint_path))

        # Output format
        output_json = getattr(args, 'validate_checkpoint_json', False)

        if output_json:
            import json
            print(json.dumps(result, indent=2))
        else:
            print("\n  Checkpoint Validation Results")
            print("  " + "=" * 40)
            print(f"\n  Overall: {'✓ VALID' if result['is_valid'] else '✗ INVALID'}")

            print(f"\n  Checks:")
            print(f"    JSON:         {'✓' if result['json_valid'] else '✗'}")
            print(f"    Schema:       {'✓' if result['schema_valid'] else '✗'}")
            print(f"    Stage Order:  {'✓' if result['stage_order_valid'] else '✗'}")
            print(f"    Age:          {'⚠ STALE' if result['is_stale'] else '✓ OK'} ({result['age_hours']:.1f}h)")

            if result['issues']:
                print(f"\n  Issues:")
                for issue in result['issues']:
                    print(f"    ✗ {issue}")

            if result['warnings']:
                print(f"\n  Warnings:")
                for warning in result['warnings']:
                    print(f"    ⚠ {warning}")

            print(f"\n  Details:")
            print(f"    Version:     {result['version']}")
            print(f"    Last Stage:  {result['last_stage']}")
            print(f"    Created:     {result['created_at']}")

        # Exit with appropriate code
        sys.exit(0 if result['is_valid'] else 1)

    # Handle --diagnostic-view (US-120-010): Launch diagnostic log viewer
    if getattr(args, 'diagnostic_view', False):
        import subprocess

        # Get logs directory - check common locations
        logs_dir = './logs'
        if PROJECT_DIR:
            # Check if there's a logs dir in the project
            project_logs = PROJECT_DIR / 'logs'
            if project_logs.exists():
                logs_dir = str(project_logs)

        # Run the diagnostic viewer script
        viewer_script = Path(__file__).parent / 'scripts' / 'diagnostic_viewer.py'

        if not viewer_script.exists():
            print(f"  Error: Diagnostic viewer script not found: {viewer_script}")
            sys.exit(1)

        # Pass through arguments to the viewer
        viewer_args = [
            sys.executable,
            str(viewer_script),
            '--logs-dir', logs_dir,
        ]

        # Add any additional viewer-specific args if passed
        if hasattr(args, 'diagnostic_level') and args.diagnostic_level:
            viewer_args.extend(['--level', args.diagnostic_level])
        if hasattr(args, 'diagnostic_category') and args.diagnostic_category:
            viewer_args.extend(['--category', args.diagnostic_category])
        if hasattr(args, 'diagnostic_search') and args.diagnostic_search:
            viewer_args.extend(['--search', args.diagnostic_search])

        # Run viewer and pass through its exit code
        result = subprocess.run(
            viewer_args,
            encoding='utf-8',
            errors='replace',
        )
        sys.exit(result.returncode)

    # Handle --config-diff (US-120-011): Show config changes since startup
    if getattr(args, 'config_diff', False):
        from src.config.base import Config

        # Load config to get startup hashes
        config = Config.from_yaml(CONFIG_PATH)

        print("\n  Config Drift Detection")
        print("  " + "=" * 40)

        # Check for drift
        drift_results = config.check_drift(force_warn=True)

        if not drift_results:
            print("\n  ✓ No config drift detected")
            print("\n  Startup field hashes:")
            for section, field_hash in config._field_hashes.items():
                print(f"    {section}: {field_hash}")
            print(f"\n  Config hash: {config._config_hash}")
            print(f"  Loaded at: {config._loaded_at}")
            sys.exit(0)

        print(f"\n  ⚠ Config drift detected in {len(drift_results)} section(s):")
        for drift in drift_results:
            print(f"\n    [{drift['section']}]")
            print(f"      Old hash: {drift['old_hash']}")
            print(f"      New hash: {drift['new_hash']}")
            if drift['changed_fields']:
                print(f"      Changed fields: {', '.join(drift['changed_fields'])}")

        print("\n  Startup hashes (all sections):")
        for section, field_hash in config._field_hashes.items():
            print(f"    {section}: {field_hash}")

        sys.exit(0)

    # Handle --checkpoint-info (US-108-010, US-115-011)
    if getattr(args, 'checkpoint_info', False):
        from src.checkpoint import CheckpointManager

        if not PROJECT_DIR:
            print("  Error: --checkpoint-info requires --project")
            sys.exit(1)

        verbose = getattr(args, 'checkpoint_info_verbose', False)

        print("\n  Checkpoint Information")
        print("  " + "=" * 40)

        # Create checkpoint manager
        cm = CheckpointManager(PROJECT_DIR, config_hash="")

        # Load checkpoint if it exists
        if cm.exists():
            cm.load()

            # Get integrity info
            integrity = cm.verify_integrity()

            # US-115-012: Get integrity check status (includes last_verified_at, all_backups_valid)
            integrity_status = cm.get_integrity_status()

            # Get backup info
            backups = cm.get_backup_info()

            # Display information
            print(f"\n  Version: {integrity['version']}")
            print(f"  Last Stage: {integrity['last_stage']}")
            print(f"  Backup Count: {integrity['backup_count']}")
            print(f"  Integrity Status: {'VALID' if integrity['is_valid'] else 'INVALID'}")

            # US-115-012: Display integrity check results (last_verified_at, all_backups_valid)
            print(f"\n  Integrity Check Result:")
            print(f"    Last Verified: {integrity_status.get('last_verified_at', 'never')}")
            print(f"    All Backups Valid: {integrity_status.get('all_backups_valid', 'N/A')}")
            print(f"    Verify On Save: {'enabled' if integrity_status.get('verify_on_save_enabled') else 'disabled'}")
            print(f"    Scheduled Verification: {'running' if integrity_status.get('schedule_verification_running') else 'disabled'}")
            print(f"    Verification History: {integrity_status.get('history_count', 0)} checks")

            if integrity['issues']:
                print(f"\n  Issues:")
                for issue in integrity['issues']:
                    print(f"    - {issue}")

            print(f"\n  Available Backups:")
            if backups:
                for b in backups:
                    print(f"    - {b['path']}: {b['size_bytes']} bytes (modified: {b['modified'][:19]})")
            else:
                print(f"    (none)")

            # US-130-002: Display checkpoint statistics
            print(f"\n  Checkpoint Statistics:")
            stats = cm.get_checkpoint_stats()
            print(f"    Total Size: {stats.get('total_size_bytes', 0):,} bytes")
            print(f"    Save Count: {stats.get('save_count', 0)}")
            print(f"    Avg Save Time: {stats.get('avg_save_time_ms', 0):.2f} ms")
            print(f"    Oldest: {stats.get('oldest_checkpoint', 'N/A') or 'N/A'}")
            print(f"    Newest: {stats.get('newest_checkpoint', 'N/A') or 'N/A'}")
            print(f"    Backups: {stats.get('backup_count', 0)}")

            # Per-stage sizes
            per_stage = stats.get('per_stage_sizes', {})
            if per_stage:
                print(f"\n  Per-Stage Sizes:")
                for stage, size in sorted(per_stage.items(), key=lambda x: x[1], reverse=True):
                    print(f"    {stage}: {size:,} bytes")

            if cm.data:
                print(f"\n  Stage Progress:")
                for stage in ['analyze', 'video_search', 'caption', 'match', 'iterative_match', 'download_segments']:
                    data = getattr(cm.data, stage, {})
                    if data:
                        print(f"    ✓ {stage.upper()}: {len(data) if isinstance(data, dict) else 'has data'}")
                    else:
                        print(f"    ○ {stage.upper()}: not completed")

            # US-115-011: Enhanced detailed state analysis (--checkpoint-info-verbose)
            if verbose and cm.data:
                print(f"\n  Video Stats:")
                # Total videos found
                total_videos = 0
                if cm.data.video_search:
                    total_videos = cm.data.video_search.get('video_count', 0)
                # Downloaded segments
                downloaded = 0
                failed = 0
                cached = 0
                if cm.data.download_segments:
                    downloaded = cm.data.download_segments.get('segment_count', 0)
                    failed = len(cm.data.download_segments.get('failed_items', []))
                # Caption status (can infer cached from caption count)
                if cm.data.caption:
                    caption_count = cm.data.caption.get('caption_count', 0)
                    cached = caption_count if caption_count > 0 else 0

                print(f"    Total videos found: {total_videos}")
                print(f"    Downloaded: {downloaded}")
                print(f"    Failed: {failed}")
                print(f"    Cached (captions): {cached}")

                print(f"\n  Match Quality:")
                # Get match quality metrics
                match_count = 0
                avg_confidence = 0.0
                unique_sources = 0
                source_diversity_score = 0.0

                if cm.data.match:
                    match_count = cm.data.match.get('match_count', 0)
                    avg_confidence = cm.data.match.get('avg_confidence', 0.0)

                    # Calculate unique sources from matches
                    matches = cm.data.match.get('matches', [])
                    if matches:
                        source_ids = set()
                        for m in matches:
                            if isinstance(m, dict):
                                vid = m.get('video_file', m.get('video_id', ''))
                                if vid:
                                    source_ids.add(vid)
                            elif hasattr(m, 'video_file'):
                                source_ids.add(m.video_file)
                        unique_sources = len(source_ids)

                        # Source diversity score: unique_sources / match_count
                        if match_count > 0:
                            source_diversity_score = unique_sources / match_count

                print(f"    Total matches: {match_count}")
                print(f"    Avg confidence: {avg_confidence:.1%}" if avg_confidence > 0 else "    Avg confidence: N/A")
                print(f"    Unique sources: {unique_sources}")
                print(f"    Source diversity score: {source_diversity_score:.2f}" if source_diversity_score > 0 else "    Source diversity score: N/A")

                print(f"\n  Stage Timing Comparison:")
                # Get current stage metrics
                current_metrics = cm.data.stage_metrics if cm.data else {}

                # Get history for previous run timing
                history = cm.get_checkpoint_history(limit=10)
                previous_timing = {}

                if history:
                    # Find the most recent completed run (not intermediate)
                    for entry in reversed(history):
                        if entry.get('last_completed_stage'):
                            prev_stage = entry.get('last_completed_stage', '')
                            # Get timing from previous run's stage_metrics
                            prev_metrics = entry.get('stage_metrics', {})
                            for stage_name, metrics in prev_metrics.items():
                                if isinstance(metrics, dict) and 'duration_seconds' in metrics:
                                    if stage_name not in previous_timing:
                                        previous_timing[stage_name] = metrics['duration_seconds']
                            break

                # Display timing comparison
                stages_to_show = ['ANALYZE', 'VIDEO_SEARCH', 'CAPTION', 'MATCH', 'ITERATIVE_MATCH', 'DOWNLOAD_SEGMENTS']
                has_timing_data = False

                for stage_name in stages_to_show:
                    current_duration = 0.0
                    if stage_name in current_metrics:
                        current_duration = current_metrics[stage_name].get('duration_seconds', 0.0)

                    prev_duration = previous_timing.get(stage_name, 0.0)

                    if current_duration > 0 or prev_duration > 0:
                        has_timing_data = True
                        if prev_duration > 0:
                            diff = current_duration - prev_duration
                            diff_pct = (diff / prev_duration) * 100
                            sign = '+' if diff >= 0 else ''
                            print(f"    {stage_name}: {current_duration:.1f}s (prev: {prev_duration:.1f}s, {sign}{diff_pct:.1f}%)")
                        else:
                            print(f"    {stage_name}: {current_duration:.1f}s (no previous run)")

                if not has_timing_data:
                    print(f"    (no timing data available)")
        else:
            print("\n  No checkpoint found in project directory")

        sys.exit(0)

    # Handle --checkpoint-diff (US-115-004)
    checkpoint_diff_args = getattr(args, 'checkpoint_diff', None)
    if checkpoint_diff_args:
        from src.checkpoint_diff import compute_checkpoint_diff, format_diff_human, format_diff_json

        checkpoint1, checkpoint2 = checkpoint_diff_args

        print("\n  Checkpoint Diff")
        print("  " + "=" * 40)

        try:
            diff = compute_checkpoint_diff(checkpoint1, checkpoint2)

            if getattr(args, 'checkpoint_diff_json', False):
                print(format_diff_json(diff))
            else:
                print(format_diff_human(diff))

            sys.exit(0)
        except FileNotFoundError as e:
            print(f"\n  Error: {e}")
            sys.exit(1)
        except Exception as e:
            print(f"\n  Error computing diff: {e}")
            sys.exit(1)

    # Handle --checkpoint-history (US-115-007)
    checkpoint_history_limit = getattr(args, 'checkpoint_history', None)
    if checkpoint_history_limit is not None:
        from src.checkpoint import CheckpointManager

        if not PROJECT_DIR:
            print("  Error: --checkpoint-history requires --project")
            sys.exit(1)

        print("\n  Checkpoint Run History")
        print("  " + "=" * 40)

        # Create checkpoint manager
        cm = CheckpointManager(PROJECT_DIR, config_hash="")

        # Get history
        history = cm.get_checkpoint_history(limit=checkpoint_history_limit)

        if not history:
            print("\n  No checkpoint history found")
            sys.exit(0)

        print(f"\n  Showing last {len(history)} entries:\n")

        # Print header
        print(f"  {'Timestamp':<25} {'Stage':<18} {'Videos':<8} {'Matches':<8} {'Size':<10}")
        print("  " + "-" * 70)

        for entry in history:
            timestamp = entry.get('timestamp', 'N/A')[:19]  # Trim to just date/time
            stage = entry.get('stage', 'N/A')
            videos = entry.get('video_count', 0)
            matches = entry.get('match_count', 0)
            size = entry.get('size_bytes', 0)

            # Format size nicely
            if size < 1024:
                size_str = f"{size}B"
            elif size < 1024 * 1024:
                size_str = f"{size/1024:.1f}KB"
            else:
                size_str = f"{size/(1024*1024):.1f}MB"

            print(f"  {timestamp:<25} {stage:<18} {videos:<8} {matches:<8} {size_str:<10}")

        sys.exit(0)

    # Handle --validate-only (US-108-011) - Stage contract validation
    if getattr(args, 'validate_only', False):
        from src.pipeline_validator import PipelineValidator
        from src.stages import get_all_stages

        print("\n  Stage Contract Validation")
        print("  " + "=" * 40)

        # Get stages
        stages = get_all_stages()
        stage_list = list(stages.values())

        # Create validator
        validator = PipelineValidator(config, stage_list)

        # Try to load checkpoint data if project directory exists
        checkpoint_data = None
        if PROJECT_DIR:
            checkpoint_path = PROJECT_DIR / "checkpoint.json"
            if checkpoint_path.exists():
                import json
                with open(checkpoint_path, 'r', encoding='utf-8') as f:
                    checkpoint_data = json.load(f)

        if checkpoint_data:
            # Validate all stages I/O
            violations = validator.validate_all_stages_io(checkpoint_data)

            if violations:
                print(f"\n  Contract Violations Found:")
                error_count = 0
                for stage_name, stage_violations in violations.items():
                    print(f"\n  {stage_name}:")
                    for v in stage_violations:
                        icon = "✗" if v.severity == "error" else "⚠"
                        print(f"    {icon} [{v.violation_type}] {v.field_name}")
                        print(f"         Expected: {v.expected}")
                        print(f"         Actual: {v.actual}")
                        if v.severity == "error":
                            error_count += 1

                print(f"\n  Summary: {error_count} errors, {sum(len(v) for v in violations.values()) - error_count} warnings")
                print("\n  ✗ Contract validation FAILED")
                sys.exit(1)
            else:
                print("\n  ✓ All stage contracts validated successfully")
                sys.exit(0)
        else:
            print("\n  No checkpoint found - cannot validate stage I/O contracts")
            print("  Run pipeline first to generate checkpoint, then use --validate-only")
            sys.exit(1)

    # Handle --validate-captions (US-005 Sprint 7)
    if getattr(args, 'validate_captions', False):
        from src.caption_fetcher import (
            validate_caption_config, run_caption_test_fetch
        )

        print("\n  Caption Configuration Validation")
        print("  " + "=" * 40)

        # Run validation
        validation_result = validate_caption_config(config)

        # Print checks performed
        for check_name, status in validation_result.checks_performed.items():
            icon = "✓" if status == "passed" else ("⚠" if status == "warning" else ("○" if status == "skipped" else "✗"))
            print(f"    {icon} {check_name}: {status}")

        # Print errors and warnings
        if validation_result.errors:
            print("\n  Errors:")
            for error in validation_result.errors:
                print(f"    ✗ {error}")

        if validation_result.warnings:
            print("\n  Warnings:")
            for warning in validation_result.warnings:
                print(f"    ⚠ {warning}")

        # Handle --test-fetch N
        test_fetch_count = getattr(args, 'test_fetch', None)
        if test_fetch_count and test_fetch_count > 0:
            if not validation_result.is_valid:
                print("\n  ⚠ Skipping test fetch due to validation errors")
                sys.exit(1)

            print(f"\n  Test Fetch ({test_fetch_count} videos)")
            print("  " + "-" * 30)

            # Get video IDs from checkpoint if available
            video_ids = []
            checkpoint_file = PROJECT_DIR / "checkpoint.json"
            if checkpoint_file.exists():
                import json
                try:
                    with open(checkpoint_file, 'r', encoding='utf-8') as f:
                        checkpoint = json.load(f)
                    # Try to get video IDs from video_candidates or text_metadata
                    stages = checkpoint.get('stages', {})
                    download_stage = stages.get('DOWNLOAD', {})
                    video_candidates = download_stage.get('video_candidates', [])
                    if video_candidates:
                        # Extract video IDs from candidates
                        for vc in video_candidates:
                            vid = vc.get('video_id') or vc.get('id')
                            if vid and len(vid) == 11:  # YouTube video IDs are 11 chars
                                video_ids.append(vid)
                except Exception as e:
                    print(f"    ⚠ Could not load checkpoint: {e}")

            if not video_ids:
                print("    ⚠ No video IDs found in checkpoint")
                print("    Run the pipeline first to collect video candidates")
                print("\n  Config valid. Test fetch: skipped (no videos)")
                sys.exit(0)

            # Run test fetch
            summary = run_caption_test_fetch(
                video_ids, config, max_videos=test_fetch_count
            )

            # Print results
            print(f"    {summary}")

            if summary.results:
                print("\n  Individual Results:")
                for result in summary.results:
                    status = "✓" if result.success else "✗"
                    if result.success:
                        print(f"    {status} {result.video_id}: {result.format_used}, "
                              f"{result.segment_count} segments, {result.elapsed_seconds:.1f}s")
                    else:
                        print(f"    {status} {result.video_id}: {result.error}")

            # Exit code 2 if any fetches failed
            if summary.failures > 0:
                print(f"\n  ⚠ {summary.failures}/{summary.total} test fetches failed")
                sys.exit(2)

        # Final summary
        if validation_result.is_valid:
            print(f"\n  ✓ Config valid." +
                  (f" {test_fetch_count} test fetch(es) succeeded." if test_fetch_count else ""))
            sys.exit(0)
        else:
            print(f"\n  ✗ Config invalid: {len(validation_result.errors)} error(s)")
            sys.exit(1)

    # Handle --cleanup-caption-cache (US-004 Sprint 8)
    if getattr(args, 'cleanup_caption_cache', False):
        from src.caption_fetcher import CaptionCache

        print("\n  Caption Cache Cleanup")
        print("  " + "=" * 40)

        # Get caption-first config
        caption_config = getattr(config.download, 'caption_first', None)
        cache = CaptionCache(caption_config)

        # Get override max_age_days if provided
        max_age_days = getattr(args, 'cleanup_caption_cache_days', None)
        if max_age_days is None:
            max_age_days = cache.max_age_days

        dry_run = getattr(args, 'cleanup_caption_cache_dry_run', False)

        print(f"    Cache directory: {cache.cache_dir}")
        print(f"    Max age threshold: {max_age_days} days")
        print(f"    Dry run: {dry_run}")
        print()

        # Get current stats first
        stats = cache.get_stats()
        print(f"    Current entries: {stats['total_entries']:,}")
        print(f"    Current size: {stats['cache_size_mb']:.2f} MB")
        print()

        # Run cleanup
        result = cache.cleanup_stale_entries(
            max_age_days=max_age_days,
            dry_run=dry_run
        )

        action = "Would remove" if dry_run else "Removed"
        print(f"    {action}: {result['entries_removed']:,} stale entries")
        print(f"    Bytes freed: {result['bytes_freed'] / (1024 * 1024):.2f} MB")
        if result['entries_removed'] > 0:
            print(f"    Oldest entry: {result['oldest_removed_days']:.1f} days old")

        if dry_run:
            print("\n  Run without --cleanup-caption-cache-dry-run to actually remove entries.")

        sys.exit(0)

    # Handle --rebuild-transcript-cache (US-124-008)
    if getattr(args, 'rebuild_transcript_cache', False):
        from src.transcription.cache import TranscriptCache

        print("\n  Transcript Cache Rebuild")
        print("  " + "=" * 40)

        # Get project directory for cache location
        project_dir = getattr(args, 'project', None)
        if project_dir:
            cache_base = Path(project_dir) / ".cache"
        else:
            cache_base = Path.cwd() / ".cache"

        print(f"    Cache directory: {cache_base}")

        # Create cache with force rebuild
        cache = TranscriptCache(str(cache_base), force_rebuild=True)

        stats = cache.get_stats()
        print(f"    Entries indexed: {stats['source_map_size']:,}")
        print(f"    Video IDs indexed: {stats['video_id_map_size']:,}")
        print(f"    Total cache entries: {stats['total_entries']:,}")

        print("\n  Force rebuild complete.")
        sys.exit(0)

    # Handle --benchmark-transcript-cache (US-124-008)
    if getattr(args, 'benchmark_transcript_cache', False):
        from src.transcription.cache import TranscriptCache

        print("\n  Transcript Cache Benchmark")
        print("  " + "=" * 40)

        # Get project directory for cache location
        project_dir = getattr(args, 'project', None)
        if project_dir:
            cache_base = Path(project_dir) / ".cache"
        else:
            cache_base = Path.cwd() / ".cache"

        print(f"    Cache directory: {cache_base}")
        print()

        # Run benchmark
        cache = TranscriptCache(str(cache_base))
        results = cache.benchmark_startup()

        print(f"    Incremental rebuild: {results['incremental_time_ms']:.1f} ms")
        print(f"    Full rebuild: {results['full_time_ms']:.1f} ms")
        print(f"    Improvement: {results['improvement_percent']:.1f}%")
        print(f"    Entries indexed: {results['entries_indexed']:,}")
        print()

        if results['improvement_percent'] > 0:
            print(f"  ✓ Incremental rebuild is {results['improvement_percent']:.1f}% faster")
        else:
            print("  Note: Run with existing cache entries to see improvement")

        sys.exit(0)

    # Validate at startup (skip for --health-check and --validate-youtube-api to allow diagnostics to run)
    if not getattr(args, 'health_check', False) and not getattr(args, 'validate_youtube_api', False):
        if not validate_config_at_startup(config):
            print("\n  ⚠ Configuration has critical errors. Fix and retry.")
            sys.exit(1)

    # US-150-008: Validate YouTube API at startup when enabled
    if not getattr(args, 'health_check', False) and not getattr(args, 'validate_youtube_api', False):
        # Check if youtube_api is enabled in config
        download_config = getattr(config, 'download', None)
        youtube_api_enabled = False
        if download_config:
            if isinstance(download_config, dict):
                yt_config = download_config.get('youtube_api', {})
                if isinstance(yt_config, dict):
                    youtube_api_enabled = yt_config.get('enabled', False)
            elif hasattr(download_config, 'youtube_api'):
                yt_config = getattr(download_config, 'youtube_api', None)
                if yt_config:
                    youtube_api_enabled = getattr(yt_config, 'enabled', False)

        if youtube_api_enabled:
            # Check if we should skip validation (configurable)
            skip_api_validation = False
            if download_config:
                if isinstance(download_config, dict):
                    yt_config = download_config.get('youtube_api', {})
                    if isinstance(yt_config, dict):
                        skip_api_validation = yt_config.get('skip_startup_validation', False)
                elif hasattr(download_config, 'youtube_api'):
                    yt_config = getattr(download_config, 'youtube_api', None)
                    if yt_config:
                        skip_api_validation = getattr(yt_config, 'skip_startup_validation', False)

            if not skip_api_validation:
                # Get API key
                api_key = None
                if download_config:
                    if isinstance(download_config, dict):
                        api_key = download_config.get('youtube_api_key')
                        if not api_key:
                            api_keys_list = download_config.get('youtube_api_keys', [])
                            if api_keys_list:
                                api_key = api_keys_list[0]
                    elif hasattr(download_config, 'youtube_api_key'):
                        api_key = download_config.youtube_api_key
                        if not api_key:
                            api_keys = getattr(download_config, 'youtube_api_keys', [])
                            if api_keys:
                                api_key = api_keys[0]

                # Also check environment variable
                if not api_key:
                    import os
                    api_key = os.environ.get('YOUTUBE_API_KEY')

                if api_key:
                    print("\n  Validating YouTube API key...")
                    from src.downloader.youtube_api_client import YouTubeAPIClient
                    import time as time_module

                    start = time_module.perf_counter()
                    client = YouTubeAPIClient(
                        api_key=api_key,
                        quota_limit=10000,
                        timeout=15,
                        auto_scale_quota=False,
                    )
                    is_valid, error_message, quota_info = client.health_check()
                    duration_ms = (time_module.perf_counter() - start) * 1000
                    client.close()

                    if is_valid:
                        percent_used = quota_info.get('percent_used', 0)
                        warn_at = 80
                        if download_config:
                            if isinstance(download_config, dict):
                                yt_config = download_config.get('youtube_api', {})
                                if isinstance(yt_config, dict):
                                    warn_at = yt_config.get('warn_at_percent', 80)
                            elif hasattr(download_config, 'youtube_api'):
                                yt_config = getattr(download_config, 'youtube_api', None)
                                if yt_config:
                                    warn_at = getattr(yt_config, 'warn_at_percent', 80)

                        if percent_used >= warn_at:
                            print(f"\n  ⚠ YouTube API quota warning: {percent_used:.1f}% used ({quota_info.get('quota_used', 0):,} / {quota_info.get('quota_limit', 0):,})")
                        else:
                            print(f"\n  ✓ YouTube API validated ({percent_used:.1f}% quota used)")
                    else:
                        if 'quota' in error_message.lower() or 'exceeded' in error_message.lower():
                            print(f"\n  ⚠ YouTube API quota issue: {error_message}")
                            print(f"    Quota: {quota_info.get('percent_used', 0):.1f}% used - will fallback to yt-dlp")
                        else:
                            print(f"\n  ✗ YouTube API validation failed: {error_message}")
                            print(f"    Will fallback to yt-dlp for searches")

    # Ensure directories exist
    ensure_dirs(config)

    # Handle --list-keywords
    if getattr(args, 'list_keywords', False):
        keyword_manager = KeywordManager(PROJECT_DIR)
        if keyword_manager.has_presets():
            print("\n" + keyword_manager.get_summary())
        else:
            print("\n  No saved keyword presets found.")
            print(f"  Use --save-keywords to save keywords after extraction.")
        sys.exit(0)

    # US-130-009: Handle checkpoint index queries
    from src.checkpoint_index import get_or_create_index
    index = get_or_create_index(PROJECT_DIR)

    if getattr(args, 'list_checkpoints', False):
        entries = index.get_all()
        if not entries:
            print("\n  No checkpoint index entries found.")
            sys.exit(0)
        print(f"\n  Checkpoint Index ({len(entries)} entries):")
        print("-" * 80)
        for e in entries:
            size_kb = e.size / 1024
            print(f"  {e.timestamp[:19]} | {e.stage:20} | {size_kb:8.1f} KB | {e.config_hash[:8] if e.config_hash else 'none'}")
        sys.exit(0)

    if getattr(args, 'query_checkpoints_by_stage', None):
        stage_name = args.query_checkpoints_by_stage
        entries = index.query_by_stage(stage_name)
        if not entries:
            print(f"\n  No checkpoint entries found for stage: {stage_name}")
            sys.exit(0)
        print(f"\n  Checkpoints for stage '{stage_name}' ({len(entries)} entries):")
        print("-" * 80)
        for e in entries:
            size_kb = e.size / 1024
            print(f"  {e.timestamp[:19]} | {size_kb:8.1f} KB | {e.config_hash[:8] if e.config_hash else 'none'}")
        sys.exit(0)

    if getattr(args, 'query_checkpoints_by_date', None):
        start_date_str, end_date_str = args.query_checkpoints_by_date
        from datetime import datetime
        try:
            # Try parsing ISO format first
            start_date = datetime.fromisoformat(start_date_str.replace('Z', '+00:00'))
        except ValueError:
            try:
                start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
            except ValueError:
                print(f"\n  Error: Invalid start date format: {start_date_str}")
                print("  Use ISO format (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS)")
                sys.exit(1)
        try:
            end_date = datetime.fromisoformat(end_date_str.replace('Z', '+00:00'))
        except ValueError:
            try:
                end_date = datetime.strptime(end_date_str, '%Y-%m-%d')
                # Include entire end day
                from datetime import timedelta
                end_date = end_date + timedelta(days=1)
            except ValueError:
                print(f"\n  Error: Invalid end date format: {end_date_str}")
                print("  Use ISO format (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS)")
                sys.exit(1)

        entries = index.query_by_date_range(start=start_date, end=end_date)
        if not entries:
            print(f"\n  No checkpoint entries found between {start_date_str} and {end_date_str}")
            sys.exit(0)
        print(f"\n  Checkpoints from {start_date_str} to {end_date_str} ({len(entries)} entries):")
        print("-" * 80)
        for e in entries:
            size_kb = e.size / 1024
            print(f"  {e.timestamp[:19]} | {e.stage:20} | {size_kb:8.1f} KB | {e.config_hash[:8] if e.config_hash else 'none'}")
        sys.exit(0)

    # Check for voiceover file
    if not args.voiceover:
        args.voiceover = find_voiceover_interactive(PROJECT_DIR)
        if not args.voiceover:
            sys.exit(1)

    # Resolve voiceover path
    vo_path = Path(args.voiceover)
    if not vo_path.is_absolute():
        vo_path = PROJECT_DIR / vo_path

    if not vo_path.exists():
        print(f"\n  Error: Voiceover file not found: {vo_path}")
        sys.exit(1)

    # Run pipeline using modular architecture
    from src.pipeline import (
        create_default_pipeline,
        create_match_only_pipeline,
        create_output_only_pipeline,
        create_pipeline_variant,
        PipelineVariantOptions,
    )
    from src.agents import ResilientRunner, HealingOrchestrator, HealingStrategy

    # Handle keyword presets/selection
    keyword_manager = KeywordManager(PROJECT_DIR)
    use_keywords = getattr(args, 'use_keywords', None)
    save_keywords = getattr(args, 'save_keywords', None)

    # Validate mutually exclusive flags
    if getattr(args, 'match_only', False) and getattr(args, 'output_only', False):
        print("\n  Error: --match-only and --output-only cannot be used together.")
        sys.exit(1)

    # Get verbose_progress flag (US-108-003)
    verbose_progress = getattr(args, 'verbose_progress', False)

    # US-155-011: Get show_quota flag
    show_quota = getattr(args, 'show_quota', False)

    # Create pipeline
    if getattr(args, 'output_only', False):
        pipeline = create_output_only_pipeline(config, PROJECT_DIR, verbose_progress, show_quota)
        # Output-only requires checkpoint data - force resume mode
        if not args.resume:
            args.resume = True

        # Validate checkpoint exists
        if not pipeline.checkpoint.exists():
            print("\n  Error: --output-only requires existing checkpoint data.")
            print("  Run the full pipeline first.")
            sys.exit(1)

        # Check checkpoint has sufficient data (at least DOWNLOAD_SEGMENTS completed)
        checkpoint_data = pipeline.checkpoint.load()
        if checkpoint_data:
            last_stage = checkpoint_data.last_completed_stage
            if last_stage:
                from src.checkpoint import STAGE_ORDER
                try:
                    last_idx = STAGE_ORDER.index(last_stage)
                    dl_idx = STAGE_ORDER.index('DOWNLOAD_SEGMENTS')
                    if last_idx < dl_idx:
                        print(f"\n  Error: Checkpoint incomplete for output-only mode.")
                        print(f"  Last completed: {last_stage}")
                        print(f"  Required: at least DOWNLOAD_SEGMENTS (run full pipeline first)")
                        sys.exit(1)

                    # Reset so only OUTPUT re-runs
                    if last_stage in ('DOWNLOAD_SEGMENTS', 'OUTPUT'):
                        pipeline.checkpoint.data.last_completed_stage = 'DOWNLOAD_SEGMENTS'
                        pipeline.checkpoint._atomic_save()
                        if last_stage == 'OUTPUT':
                            print(f"  Reset checkpoint from OUTPUT to DOWNLOAD_SEGMENTS for re-generation")
                except ValueError:
                    pass  # Unknown stage, let it proceed

        print(f"\n  Output-only mode: Regenerating OTIO/EDL/XML from checkpoint")
        print(f"  Will skip: ANALYZE through DOWNLOAD_SEGMENTS")
        print(f"  Will run: OUTPUT")

    elif args.match_only:
        pipeline = create_match_only_pipeline(config, PROJECT_DIR, verbose_progress, show_quota)
        # Match-only requires checkpoint data - force resume mode
        if not args.resume:
            args.resume = True

        # Validate checkpoint exists
        if not pipeline.checkpoint.exists():
            print("\n  Error: --match-only requires existing checkpoint data.")
            print("  Run the full pipeline first to download and transcribe videos.")
            sys.exit(1)

        # Check checkpoint has sufficient data (at least SCENE_DETECTION completed)
        checkpoint_data = pipeline.checkpoint.load()
        if checkpoint_data:
            last_stage = checkpoint_data.last_completed_stage
            if last_stage:
                from src.checkpoint import STAGE_ORDER
                try:
                    last_idx = STAGE_ORDER.index(last_stage)
                    scene_idx = STAGE_ORDER.index('SCENE_DETECTION')
                    if last_idx < scene_idx:
                        print(f"\n  Error: Checkpoint incomplete for match-only mode.")
                        print(f"  Last completed: {last_stage}")
                        print(f"  Required: SCENE_DETECTION (run full pipeline first)")
                        sys.exit(1)

                    # If pipeline completed past SCENE_DETECTION, reset so
                    # MATCH, BROLL_MATCH, and OUTPUT can re-run
                    if last_stage in ('MATCH', 'BROLL_MATCH', 'DOWNLOAD_SEGMENTS', 'OUTPUT'):
                        pipeline.checkpoint.data.last_completed_stage = 'SCENE_DETECTION'
                        pipeline.checkpoint._atomic_save()  # Save without stage update
                        print(f"  Reset checkpoint from {last_stage} to SCENE_DETECTION for re-matching")
                except ValueError:
                    pass  # Unknown stage, let it proceed

        # Print confirmation
        print(f"\n  Match-only mode: Using checkpoint data from previous run")
        print(f"  Will skip: ANALYZE through SCENE_DETECTION")
        print(f"  Will run: MATCH, OUTPUT")

        # Preload cached data for match-only mode (fallback when checkpoint is incomplete)
        _preload_cached_data_for_match_only(pipeline, config)
    else:
        # US-108-008: Use pipeline variant based on --pipeline-mode flag
        # US-151-002: --test-mode flag overrides --pipeline-mode
        if getattr(args, 'test_mode', False):
            pipeline_mode = 'test'
        else:
            pipeline_mode = getattr(args, 'pipeline_mode', 'full')

        # Handle mutually exclusive: --match-only/--output-only take precedence
        if getattr(args, 'match_only', False) or getattr(args, 'output_only', False):
            # Already handled above, just use default
            pipeline = create_default_pipeline(config, PROJECT_DIR, verbose_progress, show_quota)
        elif pipeline_mode != 'full':
            # Use pipeline variant factory for fast/test modes
            variant_options = PipelineVariantOptions(
                mode=pipeline_mode,
                parallel_execution=getattr(args, 'parallel', False),
                max_videos=getattr(args, 'max_videos', None),
                max_voiceover_segments=getattr(args, 'max_segments', None),
                max_downloads=getattr(args, 'max_downloads', None),
                # US-151-012: Set test mode specific options for display in dry-run
                skip_embeddings=getattr(args, 'skip_embeddings', True) if pipeline_mode == 'test' else False,
                skip_iterative_match=getattr(args, 'skip_iterative', True) if pipeline_mode == 'test' else False,
            )

            # Print variant info with actual limits
            max_videos = getattr(args, 'max_videos', None) or 3
            max_segments = getattr(args, 'max_segments', None) or 10
            max_downloads = getattr(args, 'max_downloads', None) or 3
            variant_descriptions = {
                'fast': 'Fast mode: skips iterative_match, reduces search results, skips embeddings',
                'test': f'Test mode: max {max_videos} videos, max {max_segments} segments, max {max_downloads} downloads',
            }
            print(f"\n  Pipeline variant: {variant_descriptions.get(pipeline_mode, pipeline_mode)}")

            # US-151-012: Pass dry_run to skip applying limits (show what would be limited)
            dry_run = getattr(args, 'dry_run', False)
            # US-155-011: Pass show_quota to pipeline variant
            pipeline = create_pipeline_variant(
                config, PROJECT_DIR, variant_options, verbose_progress, dry_run=dry_run, show_quota=show_quota
            )
        else:
            pipeline = create_default_pipeline(config, PROJECT_DIR, verbose_progress, show_quota)

    # Initialize state
    pipeline.state.voiceover_path = str(vo_path)
    pipeline.state.num_keywords = args.keywords
    pipeline.state.topic_context = ""

    # Handle keyword preset loading
    if use_keywords:
        loaded_keywords = keyword_manager.get_preset(use_keywords)
        if loaded_keywords:
            pipeline.state.keywords = loaded_keywords['keywords']
            pipeline.state.topic_context = loaded_keywords.get('topic', '')
            print(f"\n  ✓ Loaded keyword preset: {use_keywords}")
            print(f"    Keywords ({len(pipeline.state.keywords)}): {', '.join(pipeline.state.keywords[:5])}")
            if len(pipeline.state.keywords) > 5:
                print(f"    ... and {len(pipeline.state.keywords) - 5} more")

    # Handle fresh start
    if getattr(args, 'fresh', False):
        if pipeline.checkpoint.exists():
            import shutil
            checkpoint_file = pipeline.checkpoint.checkpoint_path
            backup_file = checkpoint_file.with_suffix('.backup.json')
            shutil.copy2(checkpoint_file, backup_file)
            checkpoint_file.unlink()
            print(f"\n  🔄 Fresh start: Deleted checkpoint (backup saved)")

    # Handle force rematch
    if getattr(args, 'force_rematch', False):
        pipeline.checkpoint.mark_stage_incomplete('MATCH')
        print(f"\n  🔄 Force rematch: Marked MATCH stage incomplete")

    # Handle refresh entities
    if getattr(args, 'refresh_entities', False):
        pipeline.checkpoint.refresh_entities = True
        print(f"\n  🔄 Refresh entities: Will re-download entity images")

    # Handle save matching fixtures
    if getattr(args, 'save_matching_fixtures', None):
        pipeline.state.save_matching_fixtures = args.save_matching_fixtures

    # Setup verbose trace handler (US-120-004)
    trace_handler = None
    if getattr(args, 'trace', False):
        trace_handler = VerboseTraceHandler(include_memory=True)
        pipeline.register_hook('before_stage', trace_handler.on_trace_event)
        pipeline.register_hook('after_stage', trace_handler.on_trace_event)
        print("\n  📊 Verbose tracing enabled (--trace)")

    # Setup JSON trace file (US-108-007)
    trace_file_path = None
    if getattr(args, 'trace_events', False):
        trace_file_path = PROJECT_DIR / 'pipeline_trace.jsonl'
        tracer = EventTracer(str(trace_file_path))
        tracer.start()
        pipeline.register_hook('before_stage', tracer.on_event)
        pipeline.register_hook('after_stage', tracer.on_event)
        pipeline.register_hook('on_stage_error', tracer.on_event)
        print(f"\n  📊 Event tracing enabled: {trace_file_path}")

    # Setup self-healing if enabled
    healing_config = getattr(config, 'healing', None)
    healing_enabled = healing_config and getattr(healing_config, 'enabled', True)
    orchestrator = None
    runner = None

    if healing_enabled:
        # Build strategy from config
        strategy_name = getattr(healing_config, 'strategy', 'conservative')
        strategy_map = {
            'aggressive': HealingStrategy.aggressive,
            'conservative': HealingStrategy.conservative,
            'interactive': HealingStrategy.interactive,
            'minimal': HealingStrategy.minimal,
        }
        strategy = strategy_map.get(strategy_name, HealingStrategy.conservative)()

        # Override strategy settings from config
        if hasattr(healing_config, 'max_attempts_per_stage'):
            strategy.max_attempts_per_stage = healing_config.max_attempts_per_stage
        if hasattr(healing_config, 'max_total_heals'):
            strategy.max_total_heals = healing_config.max_total_heals
        if hasattr(healing_config, 'heal_delay'):
            strategy.heal_delay = healing_config.heal_delay

        orchestrator = HealingOrchestrator(config, PROJECT_DIR, strategy)
        runner = ResilientRunner(config, PROJECT_DIR, orchestrator=orchestrator)

        print(f"\n  🛡️  Self-healing enabled: strategy={strategy_name}, max_attempts={strategy.max_attempts_per_stage}")

        # Run preflight checks if configured
        if getattr(healing_config, 'run_preflight', True):
            print(f"  Running preflight checks...")
            issues = orchestrator.run_preflight(pipeline.state)
            if issues:
                print(f"  Found {len(issues)} preflight issue(s)")
                if getattr(healing_config, 'auto_fix_preflight', True):
                    fixed_count, remaining_count = orchestrator.fix_preflight_issues(issues, pipeline.state)
                    if fixed_count:
                        print(f"  ✓ Auto-fixed {fixed_count} issue(s)")
                    if remaining_count:
                        print(f"  ⚠ {remaining_count} issue(s) could not be auto-fixed")
            else:
                print(f"  ✓ All preflight checks passed")

    # Run the pipeline
    dry_run = getattr(args, 'dry_run', False)
    if dry_run:
        # Dry-run mode bypasses self-healing; validate directly
        success = pipeline.run(resume=args.resume, dry_run=True)
    elif runner:
        success = runner.run_pipeline(pipeline, resume=args.resume)
    else:
        success = pipeline.run(resume=args.resume)

    # Print healing report if enabled
    if orchestrator and getattr(healing_config, 'print_report', True):
        orchestrator.print_report()

    # Print error summary with suggestions if --error-summary flag is set
    error_summary_flag = getattr(args, 'error_summary', False)
    if error_summary_flag and hasattr(pipeline, '_error_aggregator'):
        from src.stages.error_aggregator import PipelineErrorSuggestionEngine
        engine = PipelineErrorSuggestionEngine(pipeline._error_aggregator)
        print("\n" + engine.format_summary())

    # Handle --error-stats flag (US-120-005): Display error frequency analytics
    error_stats_flag = getattr(args, 'error_stats', False)
    error_stats_json_path = getattr(args, 'error_stats_json', None)

    if error_stats_flag or error_stats_json_path:
        if not hasattr(pipeline, '_error_aggregator'):
            print("\n  Error: No error aggregator found. Run pipeline first.")
        else:
            from src.stages.error_aggregator import PipelineErrorAggregator
            pipeline_agg = pipeline._error_aggregator

            if not isinstance(pipeline_agg, PipelineErrorAggregator):
                print("\n  Error: Invalid error aggregator type.")
            elif pipeline_agg.total_errors == 0:
                print("\n  No errors recorded in pipeline.")
            else:
                # Display error stats if flag is set
                if error_stats_flag:
                    _display_error_stats(pipeline_agg)

                # Export to JSON if path is specified
                if error_stats_json_path:
                    _export_error_stats(error_stats_json_path, pipeline_agg, PROJECT_DIR)

    # Save keywords if requested
    if save_keywords and pipeline.state.keywords:
        keyword_manager.save_keywords(
            name=save_keywords,
            keywords=pipeline.state.keywords,
            topic_context=pipeline.state.topic_context or "",
            entities=pipeline.state.extracted_entities or []
        )
        print(f"\n  ✓ Saved keyword preset: {save_keywords}")

    # Handle --export-metrics flag (export rate limit metrics to JSON)
    export_metrics_path = getattr(args, 'export_metrics', None)
    if export_metrics_path:
        _export_rate_limit_metrics(export_metrics_path, config, PROJECT_DIR)

    # Handle --export-caption-metrics flag (export caption fetch metrics to JSON)
    export_caption_metrics_path = getattr(args, 'export_caption_metrics', None)
    if export_caption_metrics_path:
        _export_caption_metrics(export_caption_metrics_path, config, PROJECT_DIR, checkpoint_manager)

    # Handle --export-resource-metrics flag (US-125-003)
    export_resource_metrics_path = getattr(args, 'export_resource_metrics', None)
    if export_resource_metrics_path:
        _export_resource_metrics(export_resource_metrics_path, config, PROJECT_DIR, pipeline)

    # Handle --export-download-metrics flag (US-129-006)
    export_download_metrics_path = getattr(args, 'export_download_metrics', None)
    if export_download_metrics_path:
        _export_download_metrics(export_download_metrics_path, config, PROJECT_DIR)

    # Handle --export-youtube-api-metrics / --export-api-metrics flag (US-149-006, US-157-010)
    # Check both forms since --export-api-metrics is an alias
    export_youtube_api_metrics_path = getattr(args, 'export_youtube_api_metrics', None) or getattr(args, 'export_api_metrics', None)
    if export_youtube_api_metrics_path:
        time_window = getattr(args, 'api_metrics_window', 'all')
        _export_youtube_api_metrics(export_youtube_api_metrics_path, config, PROJECT_DIR, time_window)

    # Handle --show-key-quota-status flag (US-155-009)
    if getattr(args, 'show_key_quota_status', False):
        _display_per_key_quota_status(config)
        sys.exit(0)

    # Handle --show-key-health flag (US-158-011)
    if getattr(args, 'show_key_health', False):
        _display_key_health_dashboard(config)
        sys.exit(0)

    # Handle --rate-limit-stats flag (US-120-009)
    if getattr(args, 'rate_limit_stats', False):
        _display_rate_limit_stats(config, checkpoint_manager)
        sys.exit(0)

    # Handle --export-config flag (US-112-012, US-128-006)
    export_config_path = getattr(args, 'export_config', None)
    if export_config_path:
        redact_sensitive = not getattr(args, 'export_config_include_sensitive', False)
        sections_raw = getattr(args, 'export_config_sections', None)
        sections = None
        if sections_raw:
            sections = [s.strip() for s in sections_raw.split(',')]

        export_format = getattr(args, 'export_config_format', 'yaml')

        if export_format == 'yaml':
            config.to_yaml(export_config_path, redact_sensitive=redact_sensitive, sections=sections)
        elif export_format == 'json':
            config.to_json(export_config_path, redact_sensitive=redact_sensitive, sections=sections)
        elif export_format == 'diff':
            config.to_diff(export_config_path, redact_sensitive=redact_sensitive, sections=sections)

        print(f"\n  ✓ Config exported to {export_config_path} (format: {export_format})")
        if redact_sensitive:
            print("  (Sensitive fields redacted with ***REDACTED***)")
        else:
            print("  (WARNING: Sensitive fields included)")
        sys.exit(0)

    # Handle --config-history flag (US-128-012)
    if getattr(args, 'config_history', False):
        history = config.get_version_history()
        if not history:
            print("\n  No version history found.")
            print("  Version history is recorded when:")
            print("    - Config is migrated to a new version")
            print("    - set_version_notes() is called to record a change")
            sys.exit(0)

        print("\n  Config Version History:")
        print("  " + "=" * 50)
        for entry in history:
            print(f"  Version: {entry.get('version', 'unknown')}")
            print(f"  Date:    {entry.get('date', 'unknown')}")
            print(f"  Notes:   {entry.get('notes', 'N/A')}")
            print("  " + "-" * 50)
        sys.exit(0)

    # Handle --config-usage flag (US-142-003)
    if getattr(args, 'config_usage', False):
        usage_stats = config.get_usage_stats()
        if not usage_stats:
            print("\n  No config usage statistics found.")
            print("  Usage tracking records which config fields are accessed during pipeline run.")
            sys.exit(0)

        print("\n  Config Usage Statistics:")
        print("  " + "=" * 50)
        # Sort by access count descending
        sorted_stats = sorted(usage_stats.items(), key=lambda x: x[1], reverse=True)
        for field, count in sorted_stats:
            print(f"  {field}: {count} accesses")
        print("  " + "-" * 50)
        print(f"  Total unique fields accessed: {len(usage_stats)}")
        print(f"  Total field accesses: {sum(usage_stats.values())}")
        sys.exit(0)

    if not success:
        print("\n  ❌ Pipeline failed")
        sys.exit(1)
    else:
        print("\n  ✅ Pipeline completed successfully")

        # US-146-012: Display YouTube API usage summary (API vs yt-dlp ratio)
        _display_youtube_api_usage_summary(pipeline)


def _display_youtube_api_usage_summary(pipeline) -> None:
    """Display YouTube API vs yt-dlp usage summary after pipeline completion (US-146-012).

    Args:
        pipeline: PipelineOrchestrator instance
    """
    try:
        from src.downloader.api_fallback_handler import get_youtube_api_client

        client = get_youtube_api_client()
        if client is None:
            return

        summary = client.get_usage_summary()
        print(f"  {summary}")

    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.debug(f"Could not display YouTube API usage summary: {e}")


def _get_youtube_api_metrics_for_export() -> Optional[Dict[str, Any]]:
    """Get YouTube API metrics for standard metrics export (US-150-010).

    Returns:
        Dictionary with YouTube API metrics or None if not available.
    """
    try:
        from src.downloader.api_fallback_handler import get_youtube_api_client

        client = get_youtube_api_client()
        if client is None:
            return None

        metrics = client.get_api_metrics()
        if metrics is None:
            return None

        # US-155-012: Add retry budget stats to metrics export
        try:
            retry_budget_stats = client.get_retry_budget_stats()
            if retry_budget_stats:
                metrics['retry_budget'] = retry_budget_stats
        except Exception:
            pass  # Don't fail entire export if retry budget stats unavailable

        return metrics
    except Exception:
        return None


def _display_youtube_api_metrics_summary(metrics: Dict[str, Any]) -> None:
    """Display YouTube API metrics summary to CLI (US-150-010).

    Args:
        metrics: YouTube API metrics dictionary
    """
    api_calls = metrics.get('api_calls', {})
    total_calls = api_calls.get('total', 0) if api_calls else 0

    if total_calls > 0:
        print(f"     YouTube API: {total_calls} calls")
        print(f"       - Search: {api_calls.get('search', 0)}")
        print(f"       - Videos: {api_calls.get('videos', 0)}")
        print(f"       - Channels: {api_calls.get('channels', 0)}")
        print(f"       - Captions: {api_calls.get('captions', 0)}")

        # Show quota info
        quota_used = metrics.get('quota_used', 0)
        quota_limit = metrics.get('quota_limit', 0)
        if quota_limit > 0:
            quota_pct = (quota_used / quota_limit) * 100
            print(f"       - Quota: {quota_used}/{quota_limit} ({quota_pct:.1f}%)")

        # Show cache metrics
        cache_metrics = metrics.get('cache_metrics', {})
        cache_hits = cache_metrics.get('cache_hits', 0)
        cache_misses = cache_metrics.get('cache_misses', 0)
        if cache_hits > 0 or cache_misses > 0:
            print(f"       - Cache: {cache_hits} hits, {cache_misses} misses")

        # Show fallback events
        fallback_events = metrics.get('fallback_events', {})
        total_fallbacks = fallback_events.get('total', 0) if fallback_events else 0
        if total_fallbacks > 0:
            print(f"       - Fallbacks: {total_fallbacks} (to yt-dlp)")


def _export_rate_limit_metrics(path: str, config, project_dir: Path):
    """Export rate limit metrics to JSON file.

    Loads metrics from the download checkpoint and exports to the specified path.
    Also includes YouTube API metrics if available (US-150-010).

    Args:
        path: Output file path for JSON export
        config: Pipeline config
        project_dir: Project directory
    """
    import logging
    logger = logging.getLogger(__name__)

    try:
        from src.downloader.rate_limit_metrics import RateLimitMetrics

        # Load download checkpoint to get rate limit metrics
        cache_dir = Path(getattr(config.cache, 'cache_dir', '.cache'))
        if not cache_dir.is_absolute():
            cache_dir = project_dir / cache_dir

        checkpoint_file = cache_dir / "download_checkpoint.json"

        if not checkpoint_file.exists():
            print(f"\n  ⚠ No download checkpoint found at {checkpoint_file}")
            print("  Cannot export metrics - no download data available.")
            return

        import json
        with open(checkpoint_file, 'r', encoding='utf-8') as f:
            checkpoint_data = json.load(f)

        # US-155-006: Also load main checkpoint for API health check
        main_checkpoint_file = project_dir / "checkpoint.json"
        main_checkpoint_data = {}
        if main_checkpoint_file.exists():
            with open(main_checkpoint_file, 'r', encoding='utf-8') as f:
                main_checkpoint_data = json.load(f)

        # Extract rate_limit_metrics from checkpoint
        rate_limit_metrics_data = checkpoint_data.get('rate_limit_metrics')
        if not rate_limit_metrics_data:
            print("\n  ⚠ No rate limit metrics in checkpoint.")
            print("  Run a download stage to collect metrics.")
            return

        # Reconstruct RateLimitMetrics from checkpoint data
        metrics = RateLimitMetrics.from_dict(rate_limit_metrics_data)

        # Resolve output path
        output_path = Path(path)
        if not output_path.is_absolute():
            output_path = project_dir / output_path

        # Export to JSON with config snapshot
        metrics.export_to_json_file(str(output_path), config)

        # US-150-010: Also export YouTube API metrics if available
        youtube_api_metrics = _get_youtube_api_metrics_for_export()
        if youtube_api_metrics:
            # Load the exported file and add youtube_api section
            with open(output_path, 'r', encoding='utf-8') as f:
                exported_data = json.load(f)
            exported_data['youtube_api'] = youtube_api_metrics
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(exported_data, f, indent=2)

        # US-155-006: Also export API health check result if available
        # Get from main checkpoint, not download checkpoint
        api_health_check = main_checkpoint_data.get('state', {}).get('api_health_check', {})
        if api_health_check:
            # Load the exported file and add api_health_check section
            with open(output_path, 'r', encoding='utf-8') as f:
                exported_data = json.load(f)
            exported_data['api_health_check'] = api_health_check
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(exported_data, f, indent=2)

        print(f"\n  📊 Exported rate limit metrics to {output_path}")
        print(f"     Downloads: {metrics.total_downloads} ({metrics.success_rate}% success)")
        print(f"     Rate limits: {metrics.rate_limit_events} events")

        # US-150-010: Display YouTube API metrics summary
        if youtube_api_metrics:
            _display_youtube_api_metrics_summary(youtube_api_metrics)

    except Exception as e:
        logger.warning(f"Failed to export rate limit metrics: {e}")
        print(f"\n  ⚠ Failed to export metrics: {e}")


def _export_download_metrics(path: str, config, project_dir: Path):
    """Export download speed analytics to JSON file (US-129-006).

    Loads speed tracker data from the download checkpoint and exports speed analytics
    including avg/median speed, speed trends, slow video detection, and rate limit warnings.

    Args:
        path: Output file path for JSON export
        config: Pipeline config
        project_dir: Project directory
    """
    import logging
    import statistics
    logger = logging.getLogger(__name__)

    try:
        from src.downloader.speed_tracker import DownloadSpeedTracker, DownloadSpeedConfig

        # Load download checkpoint to get speed tracker state
        cache_dir = Path(getattr(config.cache, 'cache_dir', '.cache'))
        if not cache_dir.is_absolute():
            cache_dir = project_dir / cache_dir

        checkpoint_file = cache_dir / "download_checkpoint.json"

        if not checkpoint_file.exists():
            print(f"\n  ⚠ No download checkpoint found at {checkpoint_file}")
            print("  Cannot export download metrics - no download data available.")
            return

        import json
        with open(checkpoint_file, 'r', encoding='utf-8') as f:
            checkpoint_data = json.load(f)

        # Extract speed_tracker_state from checkpoint
        speed_tracker_state = checkpoint_data.get('speed_tracker_state')
        if not speed_tracker_state:
            print("\n  ⚠ No speed tracker data in checkpoint.")
            print("  Run a download stage to collect speed metrics.")
            return

        # Reconstruct DownloadSpeedTracker from checkpoint data
        speed_tracker = DownloadSpeedTracker(DownloadSpeedConfig())
        speed_tracker.from_checkpoint_dict(speed_tracker_state)

        # Get speed stats
        stats = speed_tracker.get_speed_stats()

        if not stats or stats.get('samples', 0) == 0:
            print("\n  ⚠ No speed data available in checkpoint.")
            return

        # Calculate speed analytics
        speeds = [r['speed_mbps'] for r in stats.get('records', [])]

        # Calculate median
        median_speed = round(statistics.median(speeds), 2) if speeds else 0.0

        # Determine speed trend
        trend = _calculate_speed_trend(speeds)

        # Detect slow videos (below 0.5 MB/s)
        slow_threshold = 0.5
        slow_videos = [
            {
                "video_id": r['video_id'],
                "speed_mbps": r['speed_mbps'],
                "tier": r['tier']
            }
            for r in stats.get('records', [])
            if r['speed_mbps'] < slow_threshold
        ]

        # Check for sustained degradation (rate limit early warning)
        degradation_signal = None
        try:
            deg = speed_tracker.detect_sustained_degradation()
            if deg.detected:
                degradation_signal = {
                    "detected": True,
                    "degradation_percentage": deg.degradation_percentage,
                    "trend": deg.trend,
                    "message": deg.message,
                    "peak_speed_mbps": deg.peak_speed_mbps,
                    "current_speed_mbps": deg.current_speed_mbps
                }
        except Exception:
            pass

        # Check for rate limit signals
        rate_limit_signal = None
        try:
            rls = speed_tracker.detect_rate_limit_signals()
            if rls.detected:
                rate_limit_signal = {
                    "detected": True,
                    "consecutive_slow_count": rls.consecutive_slow_count,
                    "threshold": rls.threshold,
                    "message": rls.message
                }
        except Exception:
            pass

        # Build export data
        export_data = {
            "format": "json",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "speed_analytics": {
                "samples": stats.get('samples', 0),
                "avg_speed_mbps": stats.get('avg_speed_mbps', 0.0),
                "median_speed_mbps": median_speed,
                "speed_std_dev": stats.get('std_dev_mbps', 0.0),
                "min_speed_mbps": stats.get('min_speed_mbps', 0.0),
                "max_speed_mbps": stats.get('max_speed_mbps', 0.0),
                "speed_trend": trend,
                "slow_videos": slow_videos,
                "slow_video_count": len(slow_videos),
                "sustained_degradation_warning": degradation_signal,
                "rate_limit_signal": rate_limit_signal,
                "variance_category": stats.get('variance_category', 'unknown'),
                "coefficient_of_variation": stats.get('coefficient_of_variation', 0.0),
            },
            "per_video": [
                {
                    "video_id": r['video_id'],
                    "speed_mbps": r['speed_mbps'],
                    "bytes": r['bytes'],
                    "duration": r['duration'],
                    "tier": r['tier']
                }
                for r in stats.get('records', [])
            ],
            "project_path": str(project_dir),
            "config_snapshot": {
                "version": getattr(config.project, 'version', 'unknown'),
            },
        }

        # Resolve output path
        output_path = Path(path)
        if not output_path.is_absolute():
            output_path = project_dir / output_path

        # Ensure parent directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Export to JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2)

        # Print summary
        analytics = export_data['speed_analytics']
        print(f"\n  📊 Exported download speed analytics to {output_path}")
        print(f"     Samples: {analytics['samples']}")
        print(f"     Avg speed: {analytics['avg_speed_mbps']} Mbps")
        print(f"     Median speed: {analytics['median_speed_mbps']} Mbps")
        print(f"     Trend: {analytics['speed_trend']}")
        if analytics['slow_video_count'] > 0:
            print(f"     Slow videos: {analytics['slow_video_count']} (below {slow_threshold} Mbps)")
        if degradation_signal:
            print(f"     ⚠ Sustained degradation detected: {degradation_signal['degradation_percentage']*100:.1f}% drop")
        if rate_limit_signal:
            print(f"     ⚠ Rate limit signal: {rate_limit_signal['consecutive_slow_count']} consecutive slow")

    except Exception as e:
        logger.warning(f"Failed to export download metrics: {e}")
        print(f"\n  ⚠ Failed to export download metrics: {e}")


def _calculate_speed_trend(speeds: list) -> str:
    """Calculate speed trend over recent downloads.

    Args:
        speeds: List of speeds in chronological order (oldest first)

    Returns:
        Trend string: 'increasing', 'decreasing', 'stable', or 'insufficient_data'
    """
    if len(speeds) < 3:
        return "insufficient_data"

    # Use first half vs second half comparison
    mid = len(speeds) // 2
    first_half = speeds[:mid]
    second_half = speeds[mid:]

    if not first_half or not second_half:
        return "stable"

    first_avg = sum(first_half) / len(first_half)
    second_avg = sum(second_half) / len(second_half)

    # 10% threshold for trend detection
    if second_avg > first_avg * 1.1:
        return "increasing"
    elif second_avg < first_half[0] * 0.9:
        # More strict check for decreasing - compare to first speed
        return "decreasing"
    else:
        return "stable"


def _generate_youtube_api_prometheus_format(metrics: dict) -> str:
    """Generate Prometheus-format metrics from YouTube API metrics (US-157-010).

    Args:
        metrics: YouTube API metrics dictionary

    Returns:
        Prometheus-format string with HELP and TYPE declarations
    """
    lines = []

    # API calls by endpoint
    api_calls = metrics.get('api_calls', {})
    lines.extend([
        "# HELP youtube_api_calls_total Total number of YouTube API calls",
        "# TYPE youtube_api_calls_total counter",
        f'youtube_api_calls_total {api_calls.get("total", 0)}',
        "",
        "# HELP youtube_api_calls_by_endpoint Total calls per endpoint",
        "# TYPE youtube_api_calls_by_endpoint counter",
        f'youtube_api_calls_by_endpoint{{endpoint="search"}} {api_calls.get("search", 0)}',
        f'youtube_api_calls_by_endpoint{{endpoint="videos"}} {api_calls.get("videos", 0)}',
        f'youtube_api_calls_by_endpoint{{endpoint="channels"}} {api_calls.get("channels", 0)}',
        f'youtube_api_calls_by_endpoint{{endpoint="captions"}} {api_calls.get("captions", 0)}',
        "",
    ])

    # API errors by type
    error_rate_by_type = metrics.get('error_rate_by_type', {})
    lines.extend([
        "# HELP youtube_api_errors_total Total number of YouTube API errors",
        "# TYPE youtube_api_errors_total counter",
        f'youtube_api_errors_total {error_rate_by_type.get("403_forbidden", 0) + error_rate_by_type.get("429_rate_limit", 0) + error_rate_by_type.get("500_server_error", 0) + error_rate_by_type.get("other", 0)}',
        "",
        "# HELP youtube_api_error_rate_by_type Errors grouped by HTTP status code",
        "# TYPE youtube_api_error_rate_by_type counter",
        f'youtube_api_error_rate_by_type{{type="403_forbidden"}} {error_rate_by_type.get("403_forbidden", 0)}',
        f'youtube_api_error_rate_by_type{{type="429_rate_limit"}} {error_rate_by_type.get("429_rate_limit", 0)}',
        f'youtube_api_error_rate_by_type{{type="500_server_error"}} {error_rate_by_type.get("500_server_error", 0)}',
        f'youtube_api_error_rate_by_type{{type="other"}} {error_rate_by_type.get("other", 0)}',
        "",
    ])

    # Success rate
    success_rate = metrics.get('success_rate', {})
    lines.extend([
        "# HELP youtube_api_success_rate_percent Success rate percentage per endpoint",
        "# TYPE youtube_api_success_rate_percent gauge",
        f'youtube_api_success_rate_percent{{endpoint="overall"}} {success_rate.get("overall", 0)}',
        f'youtube_api_success_rate_percent{{endpoint="search"}} {success_rate.get("search", 0)}',
        f'youtube_api_success_rate_percent{{endpoint="videos"}} {success_rate.get("videos", 0)}',
        f'youtube_api_success_rate_percent{{endpoint="channels"}} {success_rate.get("channels", 0)}',
        f'youtube_api_success_rate_percent{{endpoint="captions"}} {success_rate.get("captions", 0)}',
        "",
    ])

    # Latency percentiles
    latency_percentiles = metrics.get('latency_percentiles', {})
    for endpoint in ['search', 'videos', 'channels', 'captions']:
        if endpoint in latency_percentiles:
            ep_latency = latency_percentiles[endpoint]
            lines.extend([
                f"# HELP youtube_api_latency_p50_ms {endpoint} endpoint p50 latency",
                f"# TYPE youtube_api_latency_p50_ms gauge",
                f'youtube_api_latency_p50_ms{{endpoint="{endpoint}"}} {ep_latency.get("p50", 0)}',
                f"# HELP youtube_api_latency_p95_ms {endpoint} endpoint p95 latency",
                f"# TYPE youtube_api_latency_p95_ms gauge",
                f'youtube_api_latency_p95_ms{{endpoint="{endpoint}"}} {ep_latency.get("p95", 0)}',
                f"# HELP youtube_api_latency_p99_ms {endpoint} endpoint p99 latency",
                f"# TYPE youtube_api_latency_p99_ms gauge",
                f'youtube_api_latency_p99_ms{{endpoint="{endpoint}"}} {ep_latency.get("p99", 0)}',
                "",
            ])

    # Average response time
    avg_response_time_ms = metrics.get('avg_response_time_ms', 0)
    lines.extend([
        "# HELP youtube_api_avg_response_time_ms Average response time across all endpoints",
        "# TYPE youtube_api_avg_response_time_ms gauge",
        f'youtube_api_avg_response_time_ms {avg_response_time_ms}',
        "",
    ])

    # Cache metrics
    cache_hit_rate = metrics.get('cache_hit_rate', 0)
    cache_hits = metrics.get('cache_hits', 0)
    cache_misses = metrics.get('cache_misses', 0)
    lines.extend([
        "# HELP youtube_api_cache_hit_rate_percent Cache hit rate percentage",
        "# TYPE youtube_api_cache_hit_rate_percent gauge",
        f'youtube_api_cache_hit_rate_percent {cache_hit_rate}',
        "",
        "# HELP youtube_api_cache_hits Total cache hits",
        "# TYPE youtube_api_cache_hits counter",
        f'youtube_api_cache_hits {cache_hits}',
        "",
        "# HELP youtube_api_cache_misses Total cache misses",
        "# TYPE youtube_api_cache_misses counter",
        f'youtube_api_cache_misses {cache_misses}',
        "",
    ])

    # Fallback count
    fallback_triggered_count = metrics.get('fallback_triggered_count', 0)
    lines.extend([
        "# HELP youtube_api_fallback_triggered_total Total number of API to yt-dlp fallbacks",
        "# TYPE youtube_api_fallback_triggered_total counter",
        f'youtube_api_fallback_triggered_total {fallback_triggered_count}',
        "",
    ])

    # Quota usage
    quota_aggregates = metrics.get('quota_aggregates', {})
    quota_used = quota_aggregates.get('quota_used', 0)
    quota_limit = quota_aggregates.get('quota_limit', 0)
    quota_usage_percent = quota_aggregates.get('quota_usage_percent', 0)
    lines.extend([
        "# HELP youtube_api_quota_used Units of quota used",
        "# TYPE youtube_api_quota_used gauge",
        f'youtube_api_quota_used {quota_used}',
        "",
        "# HELP youtube_api_quota_limit Total quota limit",
        "# TYPE youtube_api_quota_limit gauge",
        f'youtube_api_quota_limit {quota_limit}',
        "",
        "# HELP youtube_api_quota_usage_percent Percentage of quota used",
        "# TYPE youtube_api_quota_usage_percent gauge",
        f'youtube_api_quota_usage_percent {quota_usage_percent}',
        "",
    ])

    return "\n".join(lines)


def _export_youtube_api_metrics(path: str, config, project_dir: Path, time_window: str = None):
    """Export YouTube API usage metrics to JSON file (US-149-006, US-153-010, US-157-010).

    Exports API call counts, errors, fallbacks, quota aggregates, success rates,
    latency percentiles, circuit breaker state, and key rotation events.
    Also exports Prometheus-format metrics and time-window aggregated data.

    JSON Schema:
    {
        "schema_version": "1.2",
        "generated_at": "ISO8601 timestamp",
        "time_window": "hourly|daily|sprint|all",
        "metrics": {
            "api_calls": {
                "search": <int>,
                "videos": <int>,
                "channels": <int>,
                "captions": <int>,
                "total": <int>
            },
            "api_calls_total": <int>,
            "api_calls_by_endpoint": {...},
            "api_errors": {
                "search": <int>,
                "videos": <int>,
                "channels": <int>,
                "captions": <int>,
                "total": <int>
            },
            "error_rate_by_type": {
                "403_forbidden": <int>,
                "429_rate_limit": <int>,
                "500_server_error": <int>,
                "other": <int>,
                "rate_percent": <float>
            },
            "success_rate": {
                "search": <float>,
                "videos": <float>,
                "channels": <float>,
                "captions": <float>,
                "overall": <float>
            },
            "latency_percentiles": {
                "search": {"p50": <float>, "p95": <float>, "p99": <float>},
                "videos": {"p50": <float>, "p95": <float>, "p99": <float>},
                "channels": {"p50": <float>, "p95": <float>, "p99": <float>},
                "captions": {"p50": <float>, "p95": <float>, "p99": <float>}
            },
            "avg_response_time_ms": <float>,
            "cache_hit_rate": <float>,
            "cache_hits": <int>,
            "cache_misses": <int>,
            "fallback_triggered_count": <int>,
            "key_rotation_events": [<list of rotation events>],
            "circuit_breaker_state": {<per-endpoint state>},
            "fallback_events": {
                "total": <int>,
                "events": [<list of fallback events>]
            },
            "quota_aggregates": {
                "quota_used": <int>,
                "quota_limit": <int>,
                "quota_usage_percent": <float>,
                "keys": [<list of key stats>]
            }
        },
        "aggregations": {
            "hourly": {...},
            "daily": {...},
            "sprint": {...}
        },
        "prometheus_format": "# HELP lines..."
    }

    Args:
        path: Output file path for JSON export
        config: Pipeline config
        project_dir: Project directory
        time_window: Optional time window filter (hourly, daily, sprint, all)
    """
    import logging
    from datetime import datetime, timezone, timedelta
    import json
    import statistics
    logger = logging.getLogger(__name__)

    try:
        from src.downloader.api_fallback_handler import get_youtube_api_client

        client = get_youtube_api_client()
        if client is None:
            print("\n  ⚠ YouTube API client not available.")
            print("  Ensure YouTube API is configured in config.yaml.")
            return

        # Get metrics from API client
        metrics = client.get_api_metrics()

        if not metrics:
            print("\n  ⚠ No YouTube API metrics available.")
            print("  Run a pipeline with YouTube API enabled to collect metrics.")
            return

        # Validate schema structure (US-153-010: added success_rate, latency_percentiles, key_rotation_events, circuit_breaker_state)
        required_keys = ['api_calls', 'api_errors', 'fallback_events', 'quota_aggregates', 'success_rate', 'latency_percentiles', 'key_rotation_events', 'circuit_breaker_state']
        for key in required_keys:
            if key not in metrics:
                logger.warning(f"Invalid metrics schema: missing '{key}'")
                print(f"\n  ⚠ Invalid metrics format: missing '{key}'")
                return

        # US-157-010: Calculate additional metrics
        api_calls = metrics.get('api_calls', {})
        api_errors = metrics.get('api_errors', {})
        fallback_events = metrics.get('fallback_events', {})
        quota_aggregates = metrics.get('quota_aggregates', {})
        cache_metrics = metrics.get('cache_metrics', {})
        latency_percentiles = metrics.get('latency_percentiles', {})

        # api_calls_total
        api_calls_total = api_calls.get('total', 0) if api_calls else 0
        # api_calls_by_endpoint is already in api_calls

        # quota_usage_percent
        quota_used = quota_aggregates.get('quota_used', 0) if quota_aggregates else 0
        quota_limit = quota_aggregates.get('quota_limit', 0) if quota_aggregates else 0
        quota_usage_percent = round((quota_used / quota_limit) * 100, 2) if quota_limit > 0 else 0.0

        # error_rate_by_type - get from metrics
        error_403 = metrics.get('errors_403', 0)
        error_429 = metrics.get('errors_429', 0)
        error_500 = metrics.get('errors_500', 0)
        error_other = metrics.get('errors_other', 0)
        total_errors = error_403 + error_429 + error_500 + error_other
        error_rate_percent = round((total_errors / api_calls_total) * 100, 2) if api_calls_total > 0 else 0.0
        error_rate_by_type = {
            "403_forbidden": error_403,
            "429_rate_limit": error_429,
            "500_server_error": error_500,
            "other": error_other,
            "rate_percent": error_rate_percent
        }

        # avg_response_time_ms - calculate from latency percentiles
        all_latencies = []
        for endpoint_latencies in latency_percentiles.values():
            if isinstance(endpoint_latencies, dict):
                # Use p50 as representative
                p50 = endpoint_latencies.get('p50', 0)
                if p50 > 0:
                    all_latencies.append(p50)
        avg_response_time_ms = round(statistics.mean(all_latencies), 2) if all_latencies else 0.0

        # cache_hit_rate
        cache_hits = cache_metrics.get('cache_hits', 0) if cache_metrics else 0
        cache_misses = cache_metrics.get('cache_misses', 0) if cache_metrics else 0
        total_cache_ops = cache_hits + cache_misses
        cache_hit_rate = round((cache_hits / total_cache_ops) * 100, 2) if total_cache_ops > 0 else 0.0

        # fallback_triggered_count
        fallback_triggered_count = fallback_events.get('total', 0) if fallback_events else 0

        # Add new metrics to the metrics dict (US-157-010)
        metrics['api_calls_total'] = api_calls_total
        metrics['api_calls_by_endpoint'] = {k: v for k, v in api_calls.items() if k != 'total'}
        metrics['error_rate_by_type'] = error_rate_by_type
        metrics['avg_response_time_ms'] = avg_response_time_ms
        metrics['cache_hit_rate'] = cache_hit_rate
        metrics['cache_hits'] = cache_hits
        metrics['cache_misses'] = cache_misses
        metrics['fallback_triggered_count'] = fallback_triggered_count

        # Add quota_usage_percent to quota_aggregates
        if quota_aggregates:
            metrics['quota_aggregates']['quota_usage_percent'] = quota_usage_percent

        # US-157-010: Calculate time-window aggregations
        now = datetime.now(timezone.utc)
        hour_ago = now - timedelta(hours=1)
        day_ago = now - timedelta(days=1)
        sprint_ago = now - timedelta(days=14)  # Sprint = 2 weeks

        def filter_by_time(events, time_field='timestamp'):
            """Filter events by time window."""
            result = {"hourly": [], "daily": [], "sprint": [], "all": events}
            if not events:
                return result
            for event in events:
                event_time = event.get(time_field)
                if not event_time:
                    continue
                try:
                    if isinstance(event_time, str):
                        event_dt = datetime.fromisoformat(event_time.replace('Z', '+00:00'))
                    else:
                        event_dt = event_time
                    if event_dt >= hour_ago:
                        result["hourly"].append(event)
                    if event_dt >= day_ago:
                        result["daily"].append(event)
                    if event_dt >= sprint_ago:
                        result["sprint"].append(event)
                except Exception:
                    pass
            return result

        # Aggregate fallback events by time window
        fallback_event_list = fallback_events.get('events', []) if fallback_events else []
        fallback_by_time = filter_by_time(fallback_event_list)

        # Aggregate key rotation events by time window
        key_rotation_events = metrics.get('key_rotation_events', [])
        key_rotations_by_time = filter_by_time(key_rotation_events, 'time')

        # Build aggregations (US-157-010)
        aggregations = {}
        for window in ['hourly', 'daily', 'sprint', 'all']:
            window_fallbacks = fallback_by_time.get(window, [])
            window_key_rotations = key_rotations_by_time.get(window, [])
            aggregations[window] = {
                "fallback_count": len(window_fallbacks),
                "key_rotation_count": len(window_key_rotations),
                "api_calls": api_calls_total if window == 'all' else 0,  # Would need per-event tracking for accurate counts
            }

        # US-157-010: Generate Prometheus format
        prometheus_lines = _generate_youtube_api_prometheus_format(metrics)

        # Resolve output path
        output_path = Path(path)
        if not output_path.is_absolute():
            output_path = project_dir / output_path

        # Add metadata to the export with schema version (US-157-010: updated to 1.2)
        export_data = {
            "schema_version": "1.2",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "time_window": time_window or "all",
            "metrics": metrics,
            "aggregations": aggregations,
            "prometheus_format": prometheus_lines
        }

        # Export to JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2)

        # Also export Prometheus format to separate file if requested
        prometheus_path = str(output_path).replace('.json', '.prom')
        with open(prometheus_path, 'w', encoding='utf-8') as f:
            f.write(prometheus_lines)

        # Extract summary stats for display
        key_rotation_events = metrics.get('key_rotation_events', [])
        circuit_breaker_state = metrics.get('circuit_breaker_state', {})
        success_rate = metrics.get('success_rate', {})

        total_calls = api_calls_total
        total_errors = error_403 + error_429 + error_500 + error_other
        total_fallbacks = fallback_triggered_count

        print(f"\n  ✓ Exported YouTube API metrics to {output_path}")
        print(f"  ✓ Exported Prometheus format to {prometheus_path}")
        print(f"     Total API calls: {total_calls}")
        print(f"     Total errors: {total_errors}")
        print(f"     Total fallbacks: {total_fallbacks}")

        # US-157-010: Display new metrics
        print(f"     Avg response time: {avg_response_time_ms}ms")
        print(f"     Cache hit rate: {cache_hit_rate}%")
        print(f"     Quota usage: {quota_usage_percent}%")

        # US-157-010: Display error rate by type
        if error_rate_by_type:
            print(f"     Error breakdown:")
            print(f"       - 403 Forbidden: {error_rate_by_type.get('403_forbidden', 0)}")
            print(f"       - 429 Rate Limit: {error_rate_by_type.get('429_rate_limit', 0)}")
            print(f"       - 500 Server Error: {error_rate_by_type.get('500_server_error', 0)}")
            print(f"       - Other: {error_rate_by_type.get('other', 0)}")

        # US-153-010: Display success rates per endpoint
        if success_rate:
            print(f"     Success rates:")
            for endpoint in ['search', 'videos', 'channels', 'captions']:
                rate = success_rate.get(endpoint, 0)
                print(f"       - {endpoint}: {rate:.1f}%")

        # US-153-010: Display latency percentiles
        if latency_percentiles:
            print(f"     Latency percentiles (ms):")
            for endpoint in ['search', 'videos', 'channels', 'captions']:
                if endpoint in latency_percentiles:
                    p50 = latency_percentiles[endpoint].get('p50', 0)
                    p95 = latency_percentiles[endpoint].get('p95', 0)
                    p99 = latency_percentiles[endpoint].get('p99', 0)
                    print(f"       - {endpoint}: p50={p50:.1f}, p95={p95:.1f}, p99={p99:.1f}")

        # US-153-010: Display key rotation events
        if key_rotation_events:
            print(f"     Key rotations: {len(key_rotation_events)}")

        # US-153-010: Display circuit breaker state
        if circuit_breaker_state:
            open_count = sum(1 for state in circuit_breaker_state.values() if state.get('state') == 'open')
            if open_count > 0:
                print(f"     Circuit breakers open: {open_count}")

        if quota_aggregates:
            quota_used = quota_aggregates.get('quota_used', 0)
            quota_limit = quota_aggregates.get('quota_limit', 0)
            if quota_limit > 0:
                quota_pct = (quota_used / quota_limit) * 100
                print(f"     Quota: {quota_used}/{quota_limit} ({quota_pct:.1f}%)")

    except Exception as e:
        logger.warning(f"Failed to export YouTube API metrics: {e}")
        print(f"\n  ⚠ Failed to export YouTube API metrics: {e}")


def _export_caption_metrics(path: str, config, project_dir: Path, checkpoint_manager):
    """Export caption fetch metrics to JSON file (US-004 Sprint 7).

    Loads caption metrics from the pipeline checkpoint and exports to the specified path.

    Args:
        path: Output file path for JSON export
        config: Pipeline config
        project_dir: Project directory
        checkpoint_manager: CheckpointManager instance with loaded checkpoint
    """
    import logging
    logger = logging.getLogger(__name__)

    try:
        from src.caption_fetcher import CaptionMetrics

        # Try to get caption metrics from checkpoint manager's stages data
        caption_data = None
        if checkpoint_manager:
            stages = getattr(checkpoint_manager, '_stages', {})
            caption_data = stages.get('CAPTION', {})

        if not caption_data:
            # Fallback: try to load from checkpoint file directly
            checkpoint_file = project_dir / "checkpoint.json"
            if checkpoint_file.exists():
                import json
                with open(checkpoint_file, 'r', encoding='utf-8') as f:
                    checkpoint = json.load(f)
                caption_data = checkpoint.get('stages', {}).get('CAPTION', {})

        if not caption_data:
            print("\n  ⚠ No caption stage data found in checkpoint.")
            print("  Run a CAPTION stage to collect metrics.")
            return

        # Extract caption_metrics from stage data
        caption_metrics_data = caption_data.get('caption_metrics')
        if not caption_metrics_data:
            print("\n  ⚠ No caption metrics in checkpoint.")
            print("  Run caption-first mode to collect metrics.")
            return

        # Reconstruct CaptionMetrics from checkpoint data
        metrics = CaptionMetrics.from_dict(caption_metrics_data)

        # Resolve output path
        output_path = Path(path)
        if not output_path.is_absolute():
            output_path = project_dir / output_path

        # Get video count from caption data
        video_count = caption_data.get('total_videos', len(caption_data.get('caption_results', {})))

        # Export to JSON with run metadata
        metrics.export_json(
            str(output_path),
            project_path=str(project_dir),
            config=config,
            video_count=video_count
        )

        print(f"\n  📊 Exported caption metrics to {output_path}")
        print(f"     Fetch attempts: {metrics.fetch_attempts}, Successes: {metrics.successes}")
        print(f"     Success rate: {metrics.success_rate}%, Cache hit rate: {metrics.cache_hit_rate}%")

    except Exception as e:
        logger.warning(f"Failed to export caption metrics: {e}")
        print(f"\n  ⚠ Failed to export caption metrics: {e}")


def _export_resource_metrics(path: str, config, project_dir: Path, pipeline):
    """Export pipeline resource monitoring metrics to JSON file (US-125-003).

    Exports CPU/memory usage per stage in structured JSON format for external
    monitoring dashboards. Supports optional stage filter via path:STAGE format.

    Args:
        path: Output file path for JSON export, optionally with :STAGE suffix
              (e.g., 'resource_metrics.json' or 'resource_metrics.json:DOWNLOAD_SEGMENTS')
        config: Pipeline config
        project_dir: Project directory
        pipeline: PipelineOrchestrator instance with resource history
    """
    import logging
    import json

    logger = logging.getLogger(__name__)

    try:
        # Parse stage filter from path (format: path:STAGE)
        stage_filter = None
        if ':' in path and not path.startswith(':'):
            path_parts = path.rsplit(':', 1)
            path = path_parts[0]
            stage_filter = path_parts[1]

        # Get resource metrics from pipeline
        if not hasattr(pipeline, 'export_resource_metrics'):
            print("\n  ⚠ Pipeline does not support resource metrics export.")
            print("  Upgrade to the latest version of the pipeline.")
            return

        resource_data = pipeline.export_resource_metrics(stage_filter=stage_filter)

        # Resolve output path
        output_path = Path(path)
        if not output_path.is_absolute():
            output_path = project_dir / output_path

        # Ensure parent directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Export to JSON with metadata
        export_content = {
            'resource_metrics': resource_data,
            'project_path': str(project_dir),
            'config_snapshot': {
                'version': getattr(config.project, 'version', 'unknown'),
            },
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_content, f, indent=2)

        # Print summary
        summary = resource_data.get('summary', {})
        print(f"\n  📊 Exported resource metrics to {output_path}")
        if stage_filter:
            print(f"     Stage filter: {stage_filter}")
        print(f"     Stages tracked: {summary.get('stages_tracked', 0)}")
        print(f"     Total measurements: {summary.get('total_measurements', 0)}")
        if summary.get('memory_percent_max'):
            print(f"     Peak memory: {summary.get('memory_percent_max')}%")
        if summary.get('cpu_percent_max'):
            print(f"     Peak CPU: {summary.get('cpu_percent_max')}%")

        # Print download-specific metrics if available (US-129-012)
        download_metrics = resource_data.get('download_metrics')
        if download_metrics:
            print(f"     Download metrics:")
            print(f"       - Total bandwidth: {download_metrics.get('bandwidth_bytes_total', 0)} bytes")
            print(f"       - Avg bandwidth: {download_metrics.get('bandwidth_mbps_avg', 0.0)} Mbps")
            print(f"       - Peak bandwidth: {download_metrics.get('bandwidth_mbps_peak', 0.0)} Mbps")
            print(f"       - Avg concurrent: {download_metrics.get('concurrent_downloads_avg', 0.0)}")
            print(f"       - Max concurrent: {download_metrics.get('concurrent_downloads_max', 0)}")

    except Exception as e:
        logger.warning(f"Failed to export resource metrics: {e}")
        print(f"\n  ⚠ Failed to export resource metrics: {e}")


def _export_pipeline_graph(path: str, project_dir: Path):
    """Export pipeline dependency graph to file (US-125-012).

    Generates DOT representation and optionally renders to PNG/SVG using graphviz.

    Args:
        path: Output file path (.dot, .png, or .svg)
        project_dir: Project directory for relative path resolution
    """
    import logging
    import os
    import subprocess

    logger = logging.getLogger(__name__)

    try:
        from src.stages import generate_dot_graph

        # Resolve output path
        output_path = Path(path)
        if not output_path.is_absolute():
            output_path = project_dir / output_path

        # Ensure parent directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Get file extension to determine format
        ext = output_path.suffix.lower()

        if ext == '.dot':
            # Write DOT file directly
            dot_content = generate_dot_graph()
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(dot_content)
            print(f"\n  📊 Exported pipeline graph to {output_path}")

        elif ext in ('.png', '.svg'):
            # Try to render with graphviz
            try:
                import graphviz
            except ImportError:
                print(f"\n  ⚠ Graphviz Python library not installed.")
                print(f"  Install with: pip install graphviz")
                print(f"  Also ensure graphviz is installed on your system.")
                print(f"  Falling back to DOT format...")
                # Fall back to DOT
                output_path = output_path.with_suffix('.dot')
                dot_content = generate_dot_graph()
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(dot_content)
                print(f"  📊 Exported pipeline graph to {output_path}")
                return

            # Render with graphviz
            dot_content = generate_dot_graph()
            graph = graphviz.from_dot(dot_content)

            # Render to requested format
            renderer = 'png' if ext == '.png' else 'svg'
            try:
                graph.render(
                    filename=output_path.stem,
                    directory=output_path.parent,
                    format=renderer,
                    cleanup=True
                )
                print(f"\n  📊 Exported pipeline graph to {output_path}")
            except Exception as render_err:
                logger.warning(f"Graphviz render failed: {render_err}")
                # Fall back to DOT
                output_path = output_path.with_suffix('.dot')
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(dot_content)
                print(f"  📊 Exported pipeline graph (DOT) to {output_path}")
                print(f"  (Graphviz rendering failed: {render_err})")

        else:
            print(f"\n  ⚠ Unsupported format: {ext}")
            print(f"  Supported formats: .dot, .png, .svg")
            return

    except Exception as e:
        logger.warning(f"Failed to export pipeline graph: {e}")
        print(f"\n  ⚠ Failed to export pipeline graph: {e}")


def _display_rate_limit_stats(config, checkpoint_manager):
    """Display rate limit budget diagnostics (US-120-009).

    Shows current state of rate limit budgets including:
    - Rotations, VPN switches, and backoff time remaining
    - Predictive rate limit likelihood from historical patterns
    - Tier status (if tier isolation is enabled)
    - Keywords affected by rate limiting

    Args:
        config: Pipeline config
        checkpoint_manager: CheckpointManager instance with loaded checkpoint
    """
    from src.downloader.rate_limit_budget import RateLimitBudget
    from src.downloader.rate_limit_predictor import RateLimitPredictor

    print("\n" + "=" * 60)
    print("  RATE LIMIT BUDGET DIAGNOSTICS")
    print("=" * 60)

    # Load rate limit budget from checkpoint if available
    budget = None
    predictor = None

    if checkpoint_manager:
        stages = getattr(checkpoint_manager, '_stages', {})
        download_data = stages.get('DOWNLOAD', {})
        budget_data = download_data.get('rate_limit_budget')

        if budget_data:
            budget = RateLimitBudget.from_dict(budget_data)
            print(f"\n  ✓ Loaded budget from checkpoint")
        else:
            budget = RateLimitBudget.from_config(getattr(config.download, 'rate_limit_budget', None))
            print(f"\n  ✓ Using fresh budget from config")

        # Try to load predictor state
        predictor_data = download_data.get('rate_limit_predictor')
        if predictor_data:
            predictor = RateLimitPredictor.from_dict(predictor_data)
            print(f"  ✓ Loaded predictor from checkpoint")
    else:
        # No checkpoint, use config defaults
        budget = RateLimitBudget.from_config(getattr(config.download, 'rate_limit_budget', None))
        predictor = RateLimitPredictor()
        print(f"\n  ✓ Using fresh budget from config (no checkpoint)")

    # Display budget status
    print("\n  --- Budget Status ---")
    status = budget.get_budget_status()

    # Rotations
    rot = status['rotations']
    if rot['unlimited']:
        print(f"    Rotations: {rot['used']} used (unlimited)")
    else:
        print(f"    Rotations: {rot['used']} used, {rot['remaining']} remaining (max: {rot['max']})")

    # VPN switches
    vpn = status['vpn_switches']
    if vpn['unlimited']:
        print(f"    VPN switches: {vpn['used']} used (unlimited)")
    else:
        print(f"    VPN switches: {vpn['used']} used, {vpn['remaining']} remaining (max: {vpn['max']})")

    # Backoff time
    bo = status['backoff_time']
    if bo['unlimited']:
        print(f"    Backoff time: {bo['spent']}s spent (unlimited)")
    else:
        print(f"    Backoff time: {bo['spent']}s spent, {bo['remaining']}s remaining (max: {bo['max']}s)")

    # Exhaustion status
    print(f"\n    Exhaustion status: {'EXHAUSTED' if status['is_exhausted'] else 'OK'}")
    print(f"    Nearly exhausted (>80%): {'YES' if status['is_nearly_exhausted'] else 'NO'}")

    # Success rate
    if status.get('success_rate') is not None:
        print(f"    Success rate: {status['success_rate']*100:.1f}%")

    # Tier status (if enabled)
    tier_status = status.get('tier_status', {})
    if tier_status:
        print(f"\n  --- Tier Status ---")
        for tier_name, tier_info in tier_status.items():
            remaining = tier_info.get('attempts_remaining')
            max_attempts = tier_info.get('max_attempts')
            if remaining is not None:
                print(f"    {tier_name}: {remaining}/{max_attempts} attempts remaining")
            else:
                print(f"    {tier_name}: unlimited")
            if tier_info.get('borrowed_from'):
                print(f"      (borrowed from {tier_info['borrowed_from']})")
            if tier_info.get('lent_to'):
                print(f"      (lent to {tier_info['lent_to']})")

    # Predictive likelihood
    if predictor:
        print(f"\n  --- Predictive Rate Limit Likelihood ---")
        likelihood = predictor.predict_rate_limit_likelihood()
        pred_details = predictor.get_time_window_prediction()

        print(f"    Current likelihood: {likelihood*100:.1f}%")
        print(f"    Time window: {pred_details['time_window']} on {pred_details['day_of_week']}")
        print(f"    Historical attempts: {pred_details['total_attempts']}")
        print(f"    Historical rate limits: {pred_details['rate_limit_count']}")
        print(f"    Data sufficient: {'Yes' if pred_details['data_sufficient'] else 'No (using estimates)'}")
        print(f"    Should increase budget: {'Yes' if pred_details['should_increase_budget'] else 'No'}")

    # Keywords affected
    print(f"\n  --- Keywords ---")
    print(f"    Keywords affected by rate limits: {status['keywords_affected']}")

    print("\n" + "=" * 60)


def _display_per_key_quota_status(config):
    """Display per-key quota status for all API keys (US-155-009).

    Shows health status, quota used, remaining quota, and reset time for each key.
    """
    from src.downloader.youtube_api_client import YouTubeAPIClient

    # Get API keys from config
    yt_config = config.download.youtube_api
    api_keys = getattr(yt_config, 'api_keys', None)
    api_key = getattr(yt_config, 'api_key', '')

    if not api_keys and not api_key:
        print("Error: No YouTube API keys configured")
        return

    # Use api_keys list if available, otherwise wrap single api_key
    keys_list = api_keys if api_keys else [api_key]

    # Create client to get status
    client = YouTubeAPIClient(
        api_keys=keys_list,
        quota_limit=getattr(yt_config, 'quota_limit', 10000),
        warn_at_percent=getattr(yt_config, 'warn_at_percent', 80),
        rotation_strategy=getattr(yt_config, 'rotation_strategy', 'sequential'),
    )

    # Get per-key health status
    health_data = client.get_per_key_health()

    print("\n" + "=" * 60)
    print("  PER-KEY QUOTA STATUS")
    print("=" * 60)
    print(f"  Total keys: {len(keys_list)}")
    print(f"  Rotation strategy: {client.rotation_strategy}")
    print()

    for key_idx, data in health_data.items():
        key_preview = keys_list[key_idx][:8] + "..."
        status = data['health_status']
        quota_used = data['quota_used']
        quota_remaining = data['quota_remaining']
        percent_used = data['percent_used']
        reset_in_hours = data.get('reset_in_hours')

        # Status emoji and color
        if status == 'healthy':
            status_str = "✓ HEALTHY"
        elif status == 'quota_warning':
            status_str = "⚠ QUOTA WARNING"
        else:
            status_str = "✗ EXHAUSTED"

        print(f"  Key #{key_idx + 1} ({key_preview}):")
        print(f"    Status: {status_str}")
        print(f"    Quota: {quota_used:,} / {client._total_quota_limit:,} ({percent_used:.1f}% used)")
        print(f"    Remaining: {quota_remaining:,}")

        if reset_in_hours is not None and reset_in_hours > 0:
            print(f"    Reset in: {reset_in_hours:.1f} hours")
        elif status == 'exhausted':
            print(f"    Reset at: midnight UTC")
        print()

    # Summary
    healthy_count = sum(1 for d in health_data.values() if d['health_status'] == 'healthy')
    warning_count = sum(1 for d in health_data.values() if d['health_status'] == 'quota_warning')
    exhausted_count = sum(1 for d in health_data.values() if d['health_status'] == 'exhausted')

    print(f"  Summary: {healthy_count} healthy, {warning_count} warning, {exhausted_count} exhausted")
    print("=" * 60)


def _display_key_health_dashboard(config):
    """Display API key health dashboard with error rates and rotation recommendations (US-158-011).

    Shows per-key error counts, error rates, health status based on error rate,
    and recommends the best key to use.
    """
    from src.downloader.youtube_api_client import YouTubeAPIClient

    # Get API keys from config
    yt_config = config.download.youtube_api
    api_keys = getattr(yt_config, 'api_keys', None)
    api_key = getattr(yt_config, 'api_key', '')

    if not api_keys and not api_key:
        print("Error: No YouTube API keys configured")
        return

    # Use api_keys list if available, otherwise wrap single api_key
    keys_list = api_keys if api_keys else [api_key]

    # Create client to get status
    client = YouTubeAPIClient(
        api_keys=keys_list,
        quota_limit=getattr(yt_config, 'quota_limit', 10000),
        warn_at_percent=getattr(yt_config, 'warn_at_percent', 80),
        rotation_strategy=getattr(yt_config, 'rotation_strategy', 'sequential'),
    )

    print("\n" + "=" * 60)
    print("  API KEY HEALTH DASHBOARD")
    print("=" * 60)
    print(f"  Total keys: {len(keys_list)}")
    print()

    # Build health data for each key
    key_health_data = []
    best_key = None
    best_score = -1

    for key_idx in range(len(keys_list)):
        key_preview = keys_list[key_idx][:8] + "..." if len(keys_list[key_idx]) > 8 else keys_list[key_idx]

        # Get error counts from the client
        errors = client._key_errors.get(key_idx, {"403": 0, "429": 0, "other": 0, "total": 0})
        successes = client._key_successes.get(key_idx, 0)
        total_requests = errors["total"] + successes

        # Calculate error rate
        error_rate = (errors["total"] / total_requests * 100) if total_requests > 0 else 0.0

        # Get quota info
        quota_used = client._key_quota_used.get(key_idx, 0)
        quota_remaining = max(0, client._total_quota_limit - quota_used)
        percent_used = (quota_used / client._total_quota_limit * 100) if client._total_quota_limit > 0 else 0

        # Determine health status based on error rate
        if error_rate >= 50:
            status = "CRITICAL"
        elif error_rate >= 20:
            status = "WARNING"
        elif error_rate >= 5:
            status = "DEGRADED"
        else:
            status = "HEALTHY"

        # Calculate health score (higher is better)
        # Score = (1 - error_rate/100) * 0.7 + (quota_remaining/quota_limit) * 0.3
        error_score = (1 - error_rate / 100) * 70
        quota_score = (quota_remaining / client._total_quota_limit) * 30 if client._total_quota_limit > 0 else 30
        health_score = error_score + quota_score

        key_data = {
            "key_idx": key_idx,
            "key_preview": key_preview,
            "errors_403": errors["403"],
            "errors_429": errors["429"],
            "errors_other": errors["other"],
            "total_errors": errors["total"],
            "successes": successes,
            "total_requests": total_requests,
            "error_rate": error_rate,
            "quota_used": quota_used,
            "quota_remaining": quota_remaining,
            "percent_used": percent_used,
            "status": status,
            "health_score": health_score,
        }
        key_health_data.append(key_data)

        # Track best key
        if health_score > best_score:
            best_score = health_score
            best_key = key_idx

    # Display table header
    print(f"  {'Key':<10} {'Status':<10} {'Error Rate':<12} {'Errors':<20} {'Quota':<15} {'Score':<8}")
    print(f"  {'-'*10} {'-'*10} {'-'*12} {'-'*20} {'-'*15} {'-'*8}")

    # Display each key
    for kd in key_health_data:
        status_icon = {
            "HEALTHY": "✓",
            "DEGRADED": "⚠",
            "WARNING": "⚠",
            "CRITICAL": "✗",
        }.get(kd["status"], "?")

        error_detail = f"{kd['total_errors']} ({kd['errors_403']}xx/{kd['errors_429']}xx)"
        quota_str = f"{kd['quota_remaining']:,}/{client._total_quota_limit:,}"

        flag = ""
        if kd["status"] == "CRITICAL":
            flag = " [ROTATE]"
        elif kd["status"] == "WARNING":
            flag = " [CHECK]"

        print(f"  #{kd['key_idx']+1:<8} {status_icon} {kd['status']:<8} {kd['error_rate']:>6.1f}%     {error_detail:<20} {quota_str:<15} {kd['health_score']:>5.1f}{flag}")

    print()

    # Recommendations
    print("  RECOMMENDATIONS:")
    print("  " + "-" * 50)

    # Best key recommendation
    if best_key is not None:
        print(f"  Best key to use: #{best_key + 1} (score: {best_score:.1f})")

    # Flag keys needing attention
    critical_keys = [kd["key_idx"] + 1 for kd in key_health_data if kd["status"] == "CRITICAL"]
    warning_keys = [kd["key_idx"] + 1 for kd in key_health_data if kd["status"] == "WARNING"]

    if critical_keys:
        print(f"  Keys needing rotation: {', '.join(f'#{k}' for k in critical_keys)}")
    if warning_keys:
        print(f"  Keys to monitor: {', '.join(f'#{k}' for k in warning_keys)}")

    if not critical_keys and not warning_keys:
        print("  All keys are healthy - no action needed.")

    print("=" * 60)


if __name__ == '__main__':
    main()
