#!/usr/bin/env python3
"""
Generate SRT from voiceover audio/video file.

Usage:
    python scripts/generate_srt.py "path/to/voiceover.mp3"
    python scripts/generate_srt.py "path/to/voiceover.mp3" -o "output/subtitles.srt"
    python scripts/generate_srt.py "path/to/voiceover.mp3" --model large-v3
"""

import argparse
import logging
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.transcription import transcribe_voiceover_media

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description='Generate SRT subtitles from voiceover audio/video file'
    )
    parser.add_argument(
        'input',
        help='Path to voiceover audio or video file (mp3, wav, m4a, mp4, etc.)'
    )
    parser.add_argument(
        '-o', '--output',
        help='Output SRT path (default: same as input with .srt extension)'
    )
    parser.add_argument(
        '--model',
        default='base',
        choices=['tiny', 'base', 'small', 'medium', 'large-v2', 'large-v3'],
        help='Whisper model to use (default: base)'
    )
    parser.add_argument(
        '--language',
        default=None,
        help='Language code (e.g., en, es, fr) - auto-detect if not specified'
    )
    parser.add_argument(
        '--no-vad',
        action='store_true',
        help='Disable Voice Activity Detection (VAD)'
    )
    parser.add_argument(
        '--no-word-timestamps',
        action='store_true',
        help='Disable word-level timestamp generation'
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)

    logger.info(f"Transcribing: {input_path}")
    logger.info(f"Model: {args.model}")

    try:
        srt_path = transcribe_voiceover_media(
            media_path=str(input_path),
            output_srt_path=args.output,
            model_name=args.model,
            language=args.language,
            word_timestamps=not args.no_word_timestamps,
            vad_filter=not args.no_vad
        )
        logger.info(f"SRT generated: {srt_path}")
        print(f"\nSuccess! SRT saved to: {srt_path}")
    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
