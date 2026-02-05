"""Core configuration: Project, transcription, embedding, indexing.

Extracted from monolithic config.py during refactoring (Jan 7, 2026).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

__all__ = [
    'ProjectConfig',
    'PauseSplitConfig',
    'TranscriptionConfig',
    'EmbeddingConfig',
    'IndexingConfig',
]


@dataclass
class ProjectConfig:
    """Project identification settings"""
    name: str = "matcher-alt"
    version: str = "3.0.0"
    description: str = ""
    max_name_display_length: int = 15  # Truncation length for project name in short paths


@dataclass
class PauseSplitConfig:
    """Pause-based segment splitting settings

    Aggressive splitting for better video matching:
    - Sentence boundaries: Every sentence becomes its own segment
    - List markers: "1.", "2.", "8." isolates countdown items
    - Location patterns: "Atlanta, Georgia," splits at city/state pairs
    """
    enabled: bool = True
    min_gap_ms: int = 500  # Minimum gap to consider a pause (ms)
    split_at_sentences: bool = True  # Split at sentence boundaries (. ! ?)
    split_at_list_markers: bool = True  # Split before "1.", "2.", "Number 10", etc.
    split_at_locations: bool = True  # Split at "City, State" patterns
    min_phrase_words: int = 2  # Minimum words to keep in a segment
    min_segment_duration: float = 0.5  # Minimum duration for split segments (seconds)


@dataclass
class TranscriptionConfig:
    """Transcription settings (faster-whisper)

    Chain-of-thought: GPU acceleration critical for performance
    Reasoning: Parallel audio extraction (CPU) + sequential GPU transcription
    Decision: max_workers controls CPU parallelism, GPU uses shared model
    """
    provider: str = "faster-whisper"
    model: str = "base"  # tiny, base, small, medium, large, large-v2, large-v3
    language: str = "en"
    use_gpu: bool = True
    compute_type: str = "auto"  # auto, float16, int8, float32

    # GPU memory settings (US-60-007)
    # Minimum free GPU memory required before loading model (MB)
    # Model sizes (approximate VRAM at float16):
    #   tiny: ~400MB, base: ~500MB, small: ~1GB, medium: ~2GB, large: ~3GB
    # Default 2000MB accommodates base model with comfortable margin
    minimum_gpu_memory_mb: int = 2000
    # Automatically downgrade model if insufficient GPU memory
    auto_downgrade_model: bool = True

    # Parallel processing
    max_workers: int = 4
    batch_size: int = 10

    # Audio extraction parallelism (US-60-010)
    # Number of parallel workers for FFmpeg audio extraction (Phase 1)
    # Default: min(4, cpu_count()) for sensible default on various machines
    # Set to 0 or None to use the default calculation
    # Validated in __post_init__ to ensure <= cpu_count()
    audio_extraction_workers: int = 0

    # Automatic cleanup after batch (US-60-011)
    # When True, automatically calls whisper_client.cleanup() after batch transcription
    # to free GPU memory for subsequent stages (embedding, matching)
    auto_cleanup_after_batch: bool = True

    # VAD settings
    vad_filter: bool = False  # Default False - YouTube audio quality varies, VAD too aggressive
    min_silence_duration_ms: int = 200
    speech_pad_ms: int = 10

    # Segment splitting (for voiceover optimization)
    split_threshold_multiplier: float = 1.5  # Split if > multiplier * median
    long_median_threshold: float = 8.0  # If median > this, use absolute threshold
    absolute_split_threshold: float = 12.0  # Absolute threshold for very long segments
    min_split_duration: float = 4.0  # Minimum duration to consider splitting

    # Pause-based splitting
    pause_split: PauseSplitConfig = None

    # Caching
    cache_transcriptions: bool = True
    cache_dir: str = "transcriptions"

    # Audio extraction timeout (seconds)
    # FFmpeg subprocess killed if extraction exceeds this limit
    audio_extraction_timeout: int = 60

    def __post_init__(self):
        if self.pause_split is None:
            self.pause_split = PauseSplitConfig()

        # Validate and set audio_extraction_workers (US-60-010)
        cpu_count = os.cpu_count() or 4  # Fallback to 4 if cpu_count() returns None
        default_workers = min(4, cpu_count)

        if self.audio_extraction_workers <= 0:
            # Use default: min(4, cpu_count())
            self.audio_extraction_workers = default_workers
        elif self.audio_extraction_workers > cpu_count:
            # Cap at cpu_count() to prevent over-subscription
            self.audio_extraction_workers = cpu_count


@dataclass
class EmbeddingConfig:
    """Embedding generation settings

    Chain-of-thought: Gemini embeddings best quality, local for offline
    Reasoning: Batch processing reduces API calls by 100x
    Decision: batch_size=100 optimal for Gemini API limits
    """
    provider: str = "gemini"  # gemini, voyage, local

    # Model settings per provider
    gemini_model: str = "models/text-embedding-004"
    voyage_model: str = "voyage-2"
    local_model: str = "all-MiniLM-L6-v2"

    # Batch processing (Gemini supports up to 100)
    batch_size: int = 100
    max_retries: int = 3
    retry_delay: float = 2.0

    # Parallel processing (for large text sets)
    # ThreadPoolExecutor workers for batch processing
    max_workers: int = 4

    # Caching
    cache_embeddings: bool = True
    cache_batch_results: bool = True

    # Known embedding providers
    KNOWN_PROVIDERS = {'gemini', 'openai', 'local', 'sentence_transformers'}

    def __post_init__(self):
        # Validate batch_size >= 1
        if self.batch_size < 1:
            logger.warning(
                "EmbeddingConfig.batch_size=%s is invalid (must be >= 1), "
                "correcting to 1", self.batch_size
            )
            self.batch_size = 1

        # Validate max_retries >= 0
        if self.max_retries < 0:
            logger.warning(
                "EmbeddingConfig.max_retries=%s is invalid (must be >= 0), "
                "correcting to 0", self.max_retries
            )
            self.max_retries = 0

        # Validate max_workers >= 1
        if self.max_workers < 1:
            logger.warning(
                "EmbeddingConfig.max_workers=%s is invalid (must be >= 1), "
                "correcting to 1", self.max_workers
            )
            self.max_workers = 1

        # Warn on unknown provider
        if self.provider not in self.KNOWN_PROVIDERS:
            logger.warning(
                "EmbeddingConfig.provider='%s' is not a known provider "
                "(known: %s)", self.provider, sorted(self.KNOWN_PROVIDERS)
            )


@dataclass
class IndexingConfig:
    """FAISS indexing settings

    Chain-of-thought: Vector index for fast similarity search
    Reasoning: Flat index exact but O(n), IVF approximate but O(sqrt(n))
    Decision: Use flat for <10k vectors, IVF for larger
    """
    index_type: str = "flat"  # flat, ivf
    use_faiss: bool = True

    # IVF settings (for large datasets)
    ivf_nlist: int = 100  # Number of clusters
    ivf_nprobe: int = 10  # Clusters to search

    # Similarity settings
    normalize_embeddings: bool = True
    similarity_metric: str = "cosine"  # cosine, dot, euclidean
