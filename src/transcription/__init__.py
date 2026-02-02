"""
Transcription package - Whisper-based video transcription.

Provides parallel transcription with GPU-optimized shared model,
multi-strategy caching, and delta-aware indexing.

This package provides:
- WhisperClient: Thread-safe Whisper model client with GPU locking
- TranscriptCache: Multi-strategy caching for transcriptions
- DeltaAwareIndex: Delta-aware video tracking
- Utility functions: Audio extraction, subtitle generation
- Orchestration functions: Parallel and single-video transcription

Public API:
- transcribe_videos_parallel() - Main entry point for parallel transcription
- transcribe_voiceover_audio() - Transcribe voiceover audio file
- transcribe_voiceover_media() - Transcribe and generate SRT from media
- get_transcript_segments() - Get transcripts for single video
- transcribe_video() - Transcribe single video with cache
- cleanup_model() - Cleanup GPU resources
- WhisperClient - Thread-safe Whisper model client
- TranscriptCache - Multi-strategy caching
- DeltaAwareIndex - Delta-aware video tracking
- TranscriptSegment - Transcript segment dataclass (from src.state)

Example:
    from src.transcription import transcribe_videos_parallel

    transcripts = transcribe_videos_parallel(
        videos=downloaded_videos,
        project_dir=project_dir,
        config=config
    )
"""

# Import modular components
from .whisper_client import WhisperClient, cleanup_model
from .cache import TranscriptCache
from .delta_index import DeltaAwareIndex
from .utils import extract_audio, write_srt, extract_video_id, format_timestamp_srt
from .exceptions import (
    TranscriptionError,
    TransientTranscriptionError,
    PermanentTranscriptionError,
    is_transient_error
)

# Import orchestration functions from parallel_processor
from .parallel_processor import (
    transcribe_videos_parallel,
    transcribe_video,
    transcribe_voiceover_audio,
    transcribe_voiceover_media,
    get_transcript_segments
)

# Import TranscriptSegment from canonical location (src.state)
from src.state import TranscriptSegment

# Public API
__all__ = [
    # Main orchestration (from parallel_processor)
    'transcribe_videos_parallel',
    'transcribe_video',
    'transcribe_voiceover_audio',
    'transcribe_voiceover_media',
    'get_transcript_segments',

    # Dataclasses (from src.state)
    'TranscriptSegment',

    # Modular components
    'WhisperClient',
    'TranscriptCache',
    'DeltaAwareIndex',

    # Exceptions (from exceptions)
    'TranscriptionError',
    'TransientTranscriptionError',
    'PermanentTranscriptionError',
    'is_transient_error',

    # Cleanup
    'cleanup_model',

    # Utilities
    'extract_audio',
    'write_srt',
    'extract_video_id',
    'format_timestamp_srt',
]
