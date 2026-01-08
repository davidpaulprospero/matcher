"""
Utility functions for transcription.

Provides helpers for:
- Audio extraction from videos
- Subtitle file generation (SRT format)
- Helper functions for transcription processing
"""

import subprocess
import hashlib
import logging
from pathlib import Path
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


def extract_audio(video_path: str, output_dir: str = None) -> Optional[str]:
    """
    Extract audio from video file using ffmpeg.

    Args:
        video_path: Path to video file
        output_dir: Directory for output audio file (default: same as video)

    Returns:
        Path to extracted audio file, or None if extraction fails
    """
    video_path = Path(video_path)

    # Use hash of full path to avoid collisions with similar filenames
    path_hash = hashlib.md5(str(video_path).encode()).hexdigest()[:8]
    audio_filename = f"{video_path.stem[:80]}_{path_hash}.wav"

    if output_dir:
        audio_path = Path(output_dir) / audio_filename
    else:
        audio_path = video_path.parent / audio_filename

    # Skip if already extracted
    if audio_path.exists():
        return str(audio_path)

    try:
        cmd = [
            'ffmpeg', '-loglevel', 'error',
            '-i', str(video_path),
            '-vn', '-acodec', 'pcm_s16le',
            '-ar', '16000', '-ac', '1',
            '-y', str(audio_path)
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120
        )

        if result.returncode == 0 and audio_path.exists():
            return str(audio_path)
        else:
            logger.debug(f"FFmpeg error: {result.stderr}")
            return None

    except Exception as e:
        logger.debug(f"Audio extraction error: {e}")
        return None


def write_srt(segments: List[dict], srt_path: str):
    """
    Write segments to SRT subtitle format.

    Args:
        segments: List of segment dicts with 'start', 'end', 'text' keys
        srt_path: Path for output SRT file
    """
    def format_timestamp(seconds: float) -> str:
        """Convert seconds to SRT timestamp format (HH:MM:SS,mmm)"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

    with open(srt_path, 'w', encoding='utf-8') as f:
        for i, seg in enumerate(segments, 1):
            start = seg.get('start', 0)
            end = seg.get('end', 0)
            text = seg.get('text', '').strip()

            f.write(f"{i}\n")
            f.write(f"{format_timestamp(start)} --> {format_timestamp(end)}\n")
            f.write(f"{text}\n\n")


def extract_video_id(filename: str) -> Optional[str]:
    """
    Extract YouTube video ID from filename.

    Handles formats:
    - video_id.mp4
    - video_id.mp3
    - video_id_0045.mp4 (segments)

    YouTube IDs are 11 characters: [A-Za-z0-9_-]

    Args:
        filename: Video filename

    Returns:
        Video ID string, or None if not found
    """
    import re
    stem = Path(filename).stem

    # Pattern 1: Segment file like "abc12345678_0045" -> extract "abc12345678"
    segment_match = re.match(r'^([A-Za-z0-9_-]{11})_\d+$', stem)
    if segment_match:
        return segment_match.group(1)

    # Pattern 2: Regular file with 11-char ID at start
    if len(stem) >= 11:
        potential_id = stem[:11]
        if re.match(r'^[A-Za-z0-9_-]{11}$', potential_id):
            return potential_id

    # Pattern 3: Try to find 11-char sequence anywhere (less reliable)
    match = re.search(r'[A-Za-z0-9_-]{11}', stem)
    if match:
        return match.group(0)

    return None


def format_timestamp_srt(seconds: float) -> str:
    """
    Convert seconds to SRT timestamp format (HH:MM:SS,mmm).

    Args:
        seconds: Time in seconds

    Returns:
        Formatted timestamp string
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
