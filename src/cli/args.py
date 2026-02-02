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

    # Retry budget control (US-42-012)
    parser.add_argument(
        '--reset-budget',
        action='store_true',
        help='Reset retry budget counters when resuming from checkpoint. '
             'Use with --resume when rate limiting has subsided and you want '
             'to retry with a fresh budget instead of continuing from where it left off.'
    )

    return parser.parse_args()
