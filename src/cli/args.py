"""
Command-line argument parsing for the matcher pipeline.

Extracted from main.py (Jan 2026).
"""

import argparse


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Voiceover-to-Footage Matching Pipeline v3.0",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
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
    python main.py --match-only                # Skip download, match existing
    python main.py --config custom_config.yaml # Use custom config

    # Keyword Mode (no voiceover required)
    python main.py --keyword-list "sunset,ocean,beach" --duration 60
    python main.py --keyword-list "coral reef,marine life" --keyword-mode script --style documentary
    python main.py --keyword-list "sunset" --keyword-mode montage --duration 30
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
        help='Project directory (contains videos, outputs, project_config.yaml)'
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

    # Keyword Mode arguments (pipeline without voiceover)
    parser.add_argument(
        '--keyword-list',
        type=str,
        help='Comma-separated keywords for keyword mode (no voiceover required)'
    )

    parser.add_argument(
        '--keyword-mode',
        type=str,
        choices=['montage', 'script', 'collection'],
        default=None,
        help='Keyword mode type: montage (equal segments), script (LLM narration), collection (organize only)'
    )

    parser.add_argument(
        '--duration',
        type=float,
        default=None,
        help='Target duration in seconds (keyword mode)'
    )

    parser.add_argument(
        '--style',
        type=str,
        choices=['documentary', 'promotional', 'narrative', 'listicle', 'poetic', 'minimal'],
        default=None,
        help='Script style for keyword mode script generation'
    )

    parser.add_argument(
        '--tone',
        type=str,
        choices=['inspiring', 'serious', 'playful', 'urgent', 'contemplative'],
        default=None,
        help='Script tone for keyword mode script generation'
    )

    return parser.parse_args()
