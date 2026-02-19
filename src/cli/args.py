"""
Command-line argument parsing for the matcher pipeline.

Extracted from main.py (Jan 2026).
Updated Feb 2026: Simplified 7-stage pipeline with caption-first default.
"""

import argparse
import warnings


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Voiceover-to-Footage Matching Pipeline v4.0 (Caption-First)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Pipeline Stages (7-stage caption-first):
    ANALYZE → VIDEO_SEARCH → CAPTION → MATCH → ITERATIVE_MATCH → DOWNLOAD_SEGMENTS → OUTPUT

Examples:
    python main.py --voiceover script.srt
    python main.py --voiceover script.srt --keywords 30
    python main.py --project "E:\\Projects\\MyDoc" --voiceover voiceover.srt

    # Checkpoint & Resume
    python main.py --resume                    # Resume interrupted run
    python main.py --fresh                     # Force fresh start

    # Saved Keywords (reproducible runs)
    python main.py --save-keywords             # Save keywords after extraction
    python main.py --save-keywords mypreset    # Save with custom name
    python main.py --use-keywords              # Use most recent saved keywords
    python main.py --use-keywords mypreset     # Use specific preset
    python main.py --list-keywords             # List all saved presets

    # Other options
    python main.py --match-only                # Re-run matching only
    python main.py --output-only               # Regenerate OTIO only (needs checkpoint)
    python main.py --config custom_config.yaml # Use custom config
        """
    )

    parser.add_argument(
        '--voiceover', '-v',
        type=str,
        help='Path to voiceover file (SRT, MP3, WAV, MP4)'
    )

    parser.add_argument(
        '--keywords', '-k',
        type=int,
        default=None,
        help='Number of keywords to extract (default: from config.yaml)'
    )

    parser.add_argument(
        '--project', '-p',
        type=str,
        default=None,
        help='Project directory (contains videos, outputs)'
    )

    parser.add_argument(
        '--config', '-c',
        type=str,
        default='config.yaml',
        help='Path to config file (default: config.yaml)'
    )

    parser.add_argument(
        '--use-template',
        type=str,
        metavar='NAME',
        help='Use a config template as base (fast, quality, debug). '
             'Templates are in config/templates/. Use with -c to override template values.'
    )

    parser.add_argument(
        '--list-templates',
        action='store_true',
        help='List available config templates and exit'
    )

    parser.add_argument(
        '--profile',
        type=str,
        metavar='NAME',
        help='Use a config profile that inherits from config.yaml and overrides specific values. '
             'Profiles are in config/profiles/. Use with -c to override profile values.'
    )

    parser.add_argument(
        '--list-profiles',
        action='store_true',
        help='List available config profiles and exit'
    )

    parser.add_argument(
        '--match-only',
        action='store_true',
        help='Skip download, only match existing footage'
    )

    parser.add_argument(
        '--output-only',
        action='store_true',
        help='Regenerate OTIO/EDL/XML only (fastest, requires checkpoint with stage data)'
    )

    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume interrupted pipeline run'
    )

    parser.add_argument(
        '--fresh',
        action='store_true',
        help='Force fresh start, ignore any existing checkpoint'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Preview pipeline execution plan without running stages. '
             'Shows which stages would execute and validates inputs.'
    )

    parser.add_argument(
        '--force-rematch',
        action='store_true',
        help='Force rematch all videos, ignoring cached matches (delta matching)'
    )

    parser.add_argument(
        '--use-keywords',
        type=str,
        nargs='?',
        const='latest',
        metavar='PRESET',
        help='Use saved keywords (specify preset name, or "latest" for most recent)'
    )

    parser.add_argument(
        '--save-keywords',
        type=str,
        nargs='?',
        const='auto',
        metavar='NAME',
        help='Save extracted keywords as a preset (auto-generates name if not specified)'
    )

    parser.add_argument(
        '--list-keywords',
        action='store_true',
        help='List saved keyword presets and exit'
    )

    parser.add_argument(
        '--validate-config',
        action='store_true',
        help='Validate config file and exit'
    )

    parser.add_argument(
        '--validate-config-json',
        action='store_true',
        help='Validate config file and output results as JSON'
    )

    parser.add_argument(
        '--dry-run-config',
        action='store_true',
        help='Load and validate config without running pipeline. '
             'Shows effective values with their sources (env var, yaml, or default).'
    )

    # US-142-010: Show config value precedence
    parser.add_argument(
        '--show-precedence',
        nargs='?',
        const='',
        metavar='FIELD',
        help='Show config value precedence (env > profile > yaml > default). '
             'Optional FIELD argument shows specific field, otherwise shows all.'
    )

    # US-142-011: Show config loading performance metrics
    parser.add_argument(
        '--config-load-stats',
        action='store_true',
        help='Show config loading performance metrics (timing per phase: parse, validate, migrate, convert).'
    )

    # US-130-007: Hot backup CLI override
    parser.add_argument(
        '--hot-backup',
        type=str,
        nargs='?',
        const='enabled',
        metavar='PATH',
        help='Enable hot backup to specified path. '
             'Use --hot-backup without argument to enable using config path, '
             'or specify a path directly. Use --hot-backup off to disable.'
    )

    # US-130-009: Checkpoint index query arguments
    parser.add_argument(
        '--list-checkpoints',
        action='store_true',
        help='List all checkpoint index entries for the project'
    )

    parser.add_argument(
        '--query-checkpoints-by-stage',
        type=str,
        metavar='STAGE',
        help='Query checkpoint index by stage name (e.g., MATCH, DOWNLOAD_SEGMENTS)'
    )

    parser.add_argument(
        '--query-checkpoints-by-date',
        type=str,
        nargs=2,
        metavar=('START_DATE', 'END_DATE'),
        help='Query checkpoint index by date range (ISO format: YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS)'
    )

    parser.add_argument(
        '--non-interactive',
        action='store_true',
        help='Run in non-interactive mode (skip all prompts, use defaults)'
    )

    parser.add_argument(
        '--save-matching-fixtures',
        type=str,
        metavar='PATH',
        help='Save matching inputs to fixture file for testing (e.g., fixtures/test.json)'
    )

    parser.add_argument(
        '--refresh-entities',
        action='store_true',
        help='Force re-download entity images (ignore local cache)'
    )

    parser.add_argument(
        '--export-metrics',
        type=str,
        metavar='PATH',
        help='Export rate limit metrics to JSON file after pipeline (e.g., metrics.json)'
    )

    parser.add_argument(
        '--export-caption-metrics',
        type=str,
        metavar='PATH',
        help='Export caption fetch metrics to JSON file after pipeline '
             '(e.g., caption_metrics.json). Writes to project directory by default.'
    )

    parser.add_argument(
        '--export-resource-metrics',
        type=str,
        metavar='PATH[:STAGE]',
        help='Export pipeline resource monitoring metrics to JSON file after pipeline '
             '(e.g., resource_metrics.json). Writes CPU/memory usage per stage. '
             'Optional :STAGE suffix to filter by stage (e.g., :DOWNLOAD_SEGMENTS). '
             'Writes to project directory by default.'
    )

    # Download speed analytics export (US-129-006)
    parser.add_argument(
        '--export-download-metrics',
        type=str,
        metavar='PATH',
        help='Export download speed analytics to JSON file after pipeline '
             '(e.g., download_metrics.json). Includes avg/median speed, speed trends, '
             'slow video detection, and rate limit early warnings. '
             'Writes to project directory by default.'
    )

    # Caption-first mode flags (caption-first is now default in v4.0)
    parser.add_argument(
        '--caption-first',
        action='store_true',
        help='DEPRECATED: Caption-first is now the default behavior. '
             'This flag is kept for backward compatibility but has no effect.'
    )

    parser.add_argument(
        '--caption-language',
        type=str,
        metavar='CODE',
        help='Preferred caption language code (ISO 639-1, e.g., "en", "es", "fr"). '
             'Overrides config.yaml caption_first.preferred_language.'
    )

    parser.add_argument(
        '--no-caption-fallback',
        action='store_true',
        help='DEPRECATED: Transcription fallback has been removed in v4.0. '
             'Videos without captions are skipped. This flag has no effect.'
    )

    # Caption validation CLI (US-005 Sprint 7)
    parser.add_argument(
        '--validate-captions',
        action='store_true',
        help='Validate caption-first configuration and exit. '
             'Checks: language codes (ISO 639-1), format preferences, '
             'timeout values, and cache path writability. '
             'Exit codes: 0=valid, 1=validation errors.'
    )

    parser.add_argument(
        '--test-fetch',
        type=int,
        metavar='N',
        default=None,
        help='With --validate-captions: fetch captions for N sample videos to test. '
             'Requires --project with existing video candidates. '
             'Exit code 2 if any test fetches fail.'
    )

    # Cache cleanup (US-004 Sprint 8)
    parser.add_argument(
        '--cleanup-caption-cache',
        action='store_true',
        help='Remove stale caption cache entries older than max_cache_age_days '
             '(default: 30 days, configurable in config.yaml). '
             'Reports freed space and exits.'
    )

    parser.add_argument(
        '--cleanup-caption-cache-days',
        type=int,
        metavar='DAYS',
        default=None,
        help='Override max_cache_age_days for cleanup. '
             'Use with --cleanup-caption-cache to remove entries older than DAYS. '
             'Example: --cleanup-caption-cache --cleanup-caption-cache-days 7'
    )

    parser.add_argument(
        '--cleanup-caption-cache-dry-run',
        action='store_true',
        help='With --cleanup-caption-cache: show what would be removed without deleting. '
             'Useful for previewing cleanup impact.'
    )

    # Transcript cache rebuild (US-124-008)
    parser.add_argument(
        '--rebuild-transcript-cache',
        action='store_true',
        help='Force full rebuild of transcript cache source map. '
             'Use this if the cache index is stale or corrupted. '
             'Exit after rebuild completes.'
    )

    parser.add_argument(
        '--benchmark-transcript-cache',
        action='store_true',
        help='Benchmark transcript cache startup time. '
             'Measures incremental vs full rebuild time and reports improvement. '
             'Exit after benchmark completes.'
    )

    # Model version pinning (US-124-010)
    parser.add_argument(
        '--transcription-model-version',
        type=str,
        metavar='VERSION',
        default=None,
        help='Pin Whisper model to specific version (e.g., "v3" for large-v3, "v2" for large-v2). '
             'Overrides config.yaml transcription.model_version. '
             'Examples: "v3", "v2", "v3-turbo"'
    )

    # Retry budget control (US-42-012)
    parser.add_argument(
        '--reset-budget',
        action='store_true',
        help='Reset retry budget counters when resuming from checkpoint. '
             'Use with --resume when rate limiting has subsided and you want '
             'to retry with a fresh budget instead of continuing from where it left off.'
    )

    # Config section reset (US-142-008)
    parser.add_argument(
        '--reset-section',
        type=str,
        metavar='NAME',
        help='Reset a specific config section to its default values. '
             'Useful for testing or reverting changes. '
             'Examples: matching, download, video_search'
    )

    # Stage dependency graph (US-89-011)
    parser.add_argument(
        '--dump-dependency-graph',
        action='store_true',
        help='Output DOT graph representation of stage dependencies and exit. '
             'Use with --format to specify output format (dot, summary).'
    )

    parser.add_argument(
        '--dependency-graph-format',
        type=str,
        choices=['dot', 'summary'],
        default='dot',
        help='Output format for --dump-dependency-graph (default: dot)'
    )

    # Pipeline graph export (US-125-012)
    parser.add_argument(
        '--export-pipeline-graph',
        type=str,
        metavar='PATH',
        help='Export pipeline dependency graph to PATH. '
             'Supports .dot (default), .png, .svg formats. '
             'Uses graphviz to render image formats if available. '
             'Example: --export-pipeline-graph pipeline.png'
    )

    # Progress reporting flags (US-108-003, US-120-008)
    parser.add_argument(
        '--verbose-progress',
        action='store_true',
        help='Enable detailed per-stage progress output including '
             'stage-level percentage, ETA, and resource usage (memory, CPU).'
    )

    # Rate limit budget diagnostics (US-120-009)
    parser.add_argument(
        '--rate-limit-stats',
        action='store_true',
        help='Display rate limit budget diagnostics including remaining budget, '
             'predictive likelihood, and tier status. Shows current state of '
             'rotations, VPN switches, and backoff time budgets.'
    )

    parser.add_argument(
        '--progress',
        action='store_true',
        help='Show real-time progress with ETA and stage timeout detection. '
             'Displays current stage, progress percentage, elapsed time, ETA, '
             'and warns when stages appear stuck or exceed expected duration.'
    )

    # Health check flag (US-108-005, US-120-003)
    parser.add_argument(
        '--health-check',
        action='store_true',
        help='Run pipeline health diagnostics without executing stages. '
             'Checks disk space, memory, network, FFmpeg, yt-dlp, and LLM provider connectivity.'
    )

    # Error summary flag (US-108-006)
    parser.add_argument(
        '--error-summary',
        action='store_true',
        help='Display aggregated errors with actionable suggestions after pipeline completion. '
             'Shows error categories, patterns, and fix recommendations.'
    )

    # Error stats flag (US-120-005)
    parser.add_argument(
        '--error-stats',
        action='store_true',
        help='Display error frequency analytics with top errors, counts, percentages, '
             'and temporal patterns (time of day, day of week). Requires --project with checkpoint.'
    )

    parser.add_argument(
        '--error-stats-json',
        type=str,
        metavar='PATH',
        default=None,
        help='Export error stats as JSON to PATH. Use with --error-stats or standalone. '
             'Example: --error-stats-json ./error_stats.json'
    )

    # Escalation status flag (US-120-012)
    parser.add_argument(
        '--escalation-status',
        action='store_true',
        help='Display tier health dashboard showing circuit breaker state, per-keyword '
             'circuit breaker status, tier effectiveness metrics, and actionable recommendations.'
    )

    # Event tracing flag (US-108-007)
    parser.add_argument(
        '--trace-events',
        action='store_true',
        help='Enable JSON-lines event log to pipeline_trace.jsonl for debugging and observability. '
             'Logs all pipeline events with timestamps, stage names, and metadata.'
    )

    # Verbose tracing flag (US-120-004)
    parser.add_argument(
        '--trace',
        action='store_true',
        help='Enable verbose console tracing output showing stage start/end, duration, '
             'memory usage, and checkpoint state. Useful for debugging pipeline performance.'
    )

    # Pipeline variant mode (US-108-008)
    parser.add_argument(
        '--pipeline-mode',
        type=str,
        choices=['fast', 'full', 'test'],
        default='full',
        help='Pipeline execution mode: '
             'fast=skips iterative_match, reduces search results, skips embeddings; '
             'full=standard 7-stage pipeline (default); '
             'test=limited to 3 videos, 10 voiceover segments'
    )

    # Checkpoint info flag (US-108-010)
    parser.add_argument(
        '--checkpoint-info',
        action='store_true',
        help='Display checkpoint information: version, last_stage, backup_count, '
             'integrity_status, and available backups. Requires --project.'
    )

    # Verbose flag for checkpoint-info (US-115-011)
    parser.add_argument(
        '--checkpoint-info-verbose',
        action='store_true',
        help='Display detailed checkpoint information including video_stats, match_quality, '
             'and stage_timing_comparison. Requires --checkpoint-info.'
    )

    # Checkpoint diff tool (US-115-004)
    parser.add_argument(
        '--checkpoint-diff',
        nargs=2,
        metavar=('CHECKPOINT1', 'CHECKPOINT2'),
        help='Compare two checkpoint files and show staged changes. '
             'Displays added/removed/modified stage data with field-level diff.'
    )

    parser.add_argument(
        '--checkpoint-diff-json',
        action='store_true',
        help='Output checkpoint diff in JSON format for programmatic use. '
             'Use with --checkpoint-diff.'
    )

    # Checkpoint history (US-115-007)
    parser.add_argument(
        '--checkpoint-history',
        type=int,
        nargs='?',
        const=10,
        default=None,
        metavar='LIMIT',
        help='Display checkpoint run history: timestamps, stages, video/match counts, '
             'and file sizes. Default shows last 10 entries. Requires --project.'
    )

    # Stage contract validation flag (US-108-011)
    parser.add_argument(
        '--validate-only',
        action='store_true',
        help='Run stage input/output contract validation without executing pipeline. '
             'Validates checkpoint data against registered stage schemas and reports violations.'
    )

    # Config export/import (US-112-012)
    parser.add_argument(
        '--export-config',
        type=str,
        metavar='PATH',
        help='Export current config to YAML file with sensitive fields (API keys) '
             'redacted by default. Use --export-config-include-sensitive to include actual values.'
    )

    parser.add_argument(
        '--export-config-include-sensitive',
        action='store_true',
        help='Include sensitive fields (API keys) in --export-config output. '
             'Default is to redact with ***REDACTED***'
    )

    parser.add_argument(
        '--export-config-sections',
        type=str,
        metavar='SECTIONS',
        help='Comma-separated list of sections to export with --export-config. '
             'Example: --export-config-sections matching,download,output. '
             'If not specified, exports all sections.'
    )

    parser.add_argument(
        '--export-config-format',
        type=str,
        choices=['yaml', 'json', 'diff'],
        default='yaml',
        help='Output format for --export-config: yaml (default), json, or diff. '
             'Diff shows only non-default values with their defaults.'
    )

    # Config history (US-128-012)
    parser.add_argument(
        '--config-history',
        action='store_true',
        help='Show config version history and exit. Displays version progression '
             'with dates and notes from --export-config --format json output.'
    )

    # Config usage tracking (US-142-003)
    parser.add_argument(
        '--config-usage',
        action='store_true',
        help='Show config usage statistics after pipeline run. Displays which config '
             'fields were accessed and how many times.'
    )

    # Checkpoint export/import (US-115-009)
    parser.add_argument(
        '--checkpoint-export',
        type=str,
        metavar='PATH',
        default=None,
        help='Export checkpoint to PATH. Use --export-stages to select specific stages. '
             'Example: --checkpoint-export ./exported_checkpoint.json '
             '--export-stages video_search,caption'
    )

    parser.add_argument(
        '--checkpoint-import',
        type=str,
        metavar='PATH',
        default=None,
        help='Import checkpoint from PATH to current project. '
             'Example: --checkpoint-import ./exported_checkpoint.json'
    )

    parser.add_argument(
        '--export-stages',
        type=str,
        metavar='STAGES',
        default=None,
        help='Comma-separated list of stages to export (with --checkpoint-export). '
             'Example: --export-stages video_search,caption,match. '
             'If not specified, exports all stages. '
             'Valid stages: ANALYZE, VIDEO_SEARCH, CAPTION, MATCH, ITERATIVE_MATCH, DOWNLOAD_SEGMENTS'
    )

    # Checkpoint redaction (US-130-011)
    parser.add_argument(
        '--redact',
        action='store_true',
        help='Redact sensitive data when exporting checkpoint (URLs, paths, tokens). '
             'Use with --checkpoint-export to create privacy-safe exports for debugging.'
    )

    parser.add_argument(
        '--checkpoint-import-target',
        type=str,
        metavar='DIR',
        default=None,
        help='Target project directory for --checkpoint-import. '
             'If not specified, imports to the current --project directory.'
    )

    # Circuit breaker state visibility (US-120-006)
    parser.add_argument(
        '--circuit-status',
        action='store_true',
        help='Display circuit breaker state including open/closed/half-open status, '
             'failure history, and health metrics. Does not run pipeline.'
    )

    # Checkpoint validation (US-120-007)
    parser.add_argument(
        '--validate-checkpoint',
        action='store_true',
        help='Validate checkpoint file integrity without running pipeline. '
             'Checks: valid JSON, required fields, stage order correctness, and age. '
             'Requires --project with existing checkpoint.json.'
    )

    parser.add_argument(
        '--validate-checkpoint-json',
        action='store_true',
        help='Output validation results as JSON. Use with --validate-checkpoint.'
    )

    # Diagnostic log viewer (US-120-010)
    parser.add_argument(
        '--diagnostic-view',
        action='store_true',
        help='Launch diagnostic log viewer to view and filter pipeline logs. '
             'Supports --logs-dir, --level, --category, --from, --to, --search flags. '
             'Run with --diagnostic-view --help for full options.'
    )

    # Config drift detection (US-120-011)
    parser.add_argument(
        '--config-diff',
        action='store_true',
        help='Show config changes since startup. Displays sections that have '
             'changed during pipeline execution with field-level details.'
    )

    return parser.parse_args()
