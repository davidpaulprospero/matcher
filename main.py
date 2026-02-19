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
    # Skip for --health-check, --circuit-status, --escalation-status, --dry-run-config to allow diagnostics to run even without proper dirs
    skip_validation = getattr(args, 'health_check', False) or getattr(args, 'circuit_status', False) or getattr(args, 'escalation_status', False) or getattr(args, 'dry_run_config', False)
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

    logger = setup_logging(config, output_dir=log_output_dir)

    # Print log file locations
    if hasattr(logger, 'log_paths'):
        print(f"\n  📝 Log files:")
        print(f"    Normal:  {Path(logger.log_paths['normal']).name}")
        print(f"    Verbose: {Path(logger.log_paths['verbose']).name}")

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

    # Validate at startup (skip for --health-check to allow diagnostics to run)
    if not getattr(args, 'health_check', False):
        if not validate_config_at_startup(config):
            print("\n  ⚠ Configuration has critical errors. Fix and retry.")
            sys.exit(1)

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

    # Create pipeline
    if getattr(args, 'output_only', False):
        pipeline = create_output_only_pipeline(config, PROJECT_DIR, verbose_progress)
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
        pipeline = create_match_only_pipeline(config, PROJECT_DIR, verbose_progress)
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
        pipeline_mode = getattr(args, 'pipeline_mode', 'full')

        # Handle mutually exclusive: --match-only/--output-only take precedence
        if getattr(args, 'match_only', False) or getattr(args, 'output_only', False):
            # Already handled above, just use default
            pipeline = create_default_pipeline(config, PROJECT_DIR, verbose_progress)
        elif pipeline_mode != 'full':
            # Use pipeline variant factory for fast/test modes
            variant_options = PipelineVariantOptions(
                mode=pipeline_mode,
                parallel_execution=getattr(args, 'parallel', False),
            )

            # Print variant info
            variant_descriptions = {
                'fast': 'Fast mode: skips iterative_match, reduces search results, skips embeddings',
                'test': f'Test mode: max 3 videos, max 10 voiceover segments',
            }
            print(f"\n  Pipeline variant: {variant_descriptions.get(pipeline_mode, pipeline_mode)}")

            pipeline = create_pipeline_variant(
                config, PROJECT_DIR, variant_options, verbose_progress
            )
        else:
            pipeline = create_default_pipeline(config, PROJECT_DIR, verbose_progress)

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


def _export_rate_limit_metrics(path: str, config, project_dir: Path):
    """Export rate limit metrics to JSON file.

    Loads metrics from the download checkpoint and exports to the specified path.

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

        print(f"\n  📊 Exported rate limit metrics to {output_path}")
        print(f"     Downloads: {metrics.total_downloads} ({metrics.success_rate}% success)")
        print(f"     Rate limits: {metrics.rate_limit_events} events")

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


if __name__ == '__main__':
    main()
