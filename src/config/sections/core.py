"""Core configuration: Project, transcription, embedding, indexing.

Extracted from monolithic config.py during refactoring (Jan 7, 2026).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger(__name__)

__all__ = [
    'ProjectConfig',
    'PauseSplitConfig',
    'SegmentPostProcessingConfig',
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
class SegmentPostProcessingConfig:
    """Segment post-processing settings (US-124-011)

    Controls intelligent segmentation after Whisper transcription:
    - Merge segments with same speaker (based on speaker labels if available)
    - Split segments at natural language boundaries (punctuation)
    """
    # Enable/disable all post-processing
    enabled: bool = True

    # Merge segments with same speaker (if speaker labels available)
    # Whisper can output speaker diarization with word-level timestamps
    merge_same_speaker: bool = True

    # Maximum gap between segments to consider merging (seconds)
    # Only applies when merge_same_speaker is True
    merge_max_gap_seconds: float = 0.5

    # Minimum duration for merged segment (seconds)
    # Prevents merging very short segments
    merge_min_duration: float = 1.0

    # Split segments at natural language boundaries (punctuation)
    split_at_punctuation: bool = True

    # Punctuation marks that trigger splits (in order of precedence)
    # Each is treated as end-of-sentence boundary
    split_punctuation: str = ".!?"

    # Don't split at abbreviations (e.g., "Mr.", "Dr.", "U.S.")
    split_ignore_abbreviations: bool = True

    # Common abbreviations to ignore when splitting
    abbreviations: List[str] = None

    # Minimum length of text after punctuation to create new segment
    # Prevents splitting on "e.g. something" where "e.g." shouldn't split
    split_min_after_text: int = 2

    # Minimum duration for split segments (seconds)
    split_min_duration: float = 0.5

    # Enable speaker-aware merging using VAD segment analysis (US-137-008)
    # When True, detects speaker changes based on VAD segments and merges
    # segments by detected speaker rather than relying on explicit speaker labels
    enable_speaker_aware_merging: bool = False

    # Minimum confidence for speaker change detection (0.0-1.0)
    # Higher values require stronger evidence for speaker boundaries
    speaker_change_confidence_threshold: float = 0.7

    def __post_init__(self):
        if self.abbreviations is None:
            self.abbreviations = [
                "mr", "mrs", "ms", "dr", "prof", "sr", "jr",
                "vs", "etc", "eg", "ie", "al",
                "us", "usa", "uk", "eu", "un", "nato",
                "a", "b", "c", "d", "e", "f", "g", "h",  # Single letters
            ]


@dataclass
class TranscriptionConfig:
    """Transcription settings (faster-whisper)

    Chain-of-thought: GPU acceleration critical for performance
    Reasoning: Parallel audio extraction (CPU) + sequential GPU transcription
    Decision: max_workers controls CPU parallelism, GPU uses shared model
    """
    provider: str = "faster-whisper"
    model: str = "base"  # tiny, base, small, medium, large, large-v2, large-v3
    # Model version to pin (e.g., "v3" for large-v3, "v2" for large-v2)
    # faster-whisper uses format: model@version (e.g., "small@v3")
    # Set to None or empty to use latest available version
    model_version: Optional[str] = None
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

    # Batch processing limit (US-110-005)
    # Maximum videos to process in a single batch before pausing
    # Helps manage memory and allows for progress checkpointing
    # Set to 0 to process all videos in one batch (no batching)
    batch_size: int = 50
    # Pause duration between batches (seconds)
    # Allows GPU to cool down and system to stabilize between batches
    batch_wait_seconds: int = 5

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
    pause_split: Optional[PauseSplitConfig] = None

    # Segment post-processing (US-124-011)
    # Intelligent segmentation after Whisper transcription
    post_processing: Optional[SegmentPostProcessingConfig] = None

    # Caching
    cache_transcriptions: bool = True
    cache_dir: str = "transcriptions"

    # Audio extraction timeout (seconds)
    # FFmpeg subprocess killed if extraction exceeds this limit
    audio_extraction_timeout: int = 60

    # GPU transcription timeout (seconds) (US-79-002)
    # Maximum time allowed for a single model.transcribe() call
    # Prevents hung GPU calls from blocking the entire pipeline
    # Must be >= 30 seconds to allow for large audio files
    gpu_transcription_timeout: int = 300

    # Transcription retry attempts (US-79-004)
    # Maximum retry attempts for transient errors (e.g., GPU OOM)
    # 0 = no retries (single attempt only), max 5
    max_retries: int = 2

    # Cache max age in days (US-79-005)
    # Cache entries older than this are removed by cleanup_stale_entries()
    # Must be >= 1 day
    cache_max_age_days: int = 30

    # Whisper model threading (US-79-007)
    # num_workers: Number of workers for WhisperModel batched decoding
    # Keep at 1 for GPU to avoid CUDA serialization overhead; increase only for CPU-only mode
    whisper_num_workers: int = 1
    # cpu_threads: Number of CPU threads for WhisperModel (ctranslate2)
    # Higher values benefit CPU-bound operations (beam search, preprocessing)
    # Default 4 is conservative; systems with many cores can increase
    whisper_cpu_threads: int = 4

    # Progress logging interval (US-79-008)
    # Log progress every N items during batch transcription phases
    # Must be >= 1
    progress_log_interval: int = 10

    # Batch retry budget (US-79-010)
    # Maximum total transcription attempts across all videos in a batch
    # Prevents infinite retry loops when many videos fail
    # Set to 0 for unlimited attempts
    retry_budget_max_attempts: int = 50
    # Maximum cumulative backoff time (seconds) before budget exhaustion
    # Set to 0 for unlimited backoff
    retry_budget_max_backoff_seconds: float = 180.0

    # Language detection (US-110-003, US-124-006)
    # Minimum confidence threshold for Whisper language detection
    # When Whisper confidence is below this, fallback to langdetect library
    # Range: 0.0 to 1.0, default 0.8
    min_language_confidence: float = 0.8

    # Language detection library preference for fallback (US-124-006)
    # Which library to use when Whisper confidence is below threshold.
    # Options: "langdetect" (default), "langid", "fasttext"
    # Set to None to disable fallback detection
    language_detection_library: str = "langdetect"

    # Prefer Whisper over fallback library (US-124-006)
    # When True, always use Whisper's language detection even if confidence is below threshold
    # When False, fallback to configured library when Whisper confidence is low
    # Default False - fallback improves accuracy for multilingual content
    prefer_whisper_over_fallback: bool = False

    # Use consensus-based language detection (US-124-006)
    # When True, runs multiple detection libraries and uses consensus voting
    # Requires at least 2 libraries to be available
    # Default False - use single library for performance
    use_consensus_detection: bool = False

    # Weighted confidence fallback (US-137-009)
    # When True, combines Whisper, langdetect, and langid using weighted confidence:
    #   - Whisper: 60% weight
    #   - langdetect: 25% weight
    #   - langid: 15% weight
    # This provides more robust language detection than single-source
    language_detection_weighted_fallback: bool = True

    # Minimum combined confidence threshold for weighted fallback (US-137-009)
    # When weighted fallback is enabled, the combined confidence must exceed this
    # to accept the result; otherwise falls back to Whisper alone
    # Range: 0.0 to 1.0, default 0.7
    min_combined_confidence: float = 0.7

    # GPU-to-CPU automatic fallback (US-110-004)
    # When True, automatically falls back to CPU compute type when GPU fails
    # (e.g., CUDA out of memory, GPU not available)
    # Preserves retry attempts by falling back before retry budget is exhausted
    auto_fallback_to_cpu: bool = True

    # Cache compression (US-110-008)
    # When True, compresses transcript cache files using gzip
    # Reduces disk space usage significantly for large transcript caches
    # Default True - backward compatible with uncompressed files on read
    compress_cache: bool = True

    # Segment quality filtering (US-110-009)
    # Minimum number of words required in a segment to be considered valid
    # Segments with fewer words are filtered out before caching
    # Set to 0 to disable filtering, recommended minimum is 1-3
    min_segment_words: int = 3

    # Auto model selection based on audio duration (US-110-010)
    # When True, automatically selects optimal Whisper model based on audio duration:
    #   - tiny:  audio < 5 minutes
    #   - base:  5-30 minutes
    #   - small: 30+ minutes
    # When False, uses the manually specified model from config
    # Default True - enables faster transcription for short audio
    auto_model_selection: bool = True

    # FFmpeg pipelining (US-124-012)
    # Pipeline depth: number of videos to extract audio for ahead of transcription
    # When > 0, audio extraction starts while previous videos are being transcribed
    # This overlaps I/O (audio extraction) with GPU computation (transcription)
    # Set to 0 to disable pipelining (traditional two-phase approach)
    # Default 3 provides good overlap for most systems (US-137-006)
    pipeline_depth: int = 3

    # Dynamic pipeline depth adjustment (US-137-006)
    # When True, automatically adjusts pipeline_depth based on GPU utilization
    # Higher GPU utilization -> reduce pipeline depth to avoid queue buildup
    # Lower GPU utilization -> increase pipeline depth for better overlap
    dynamic_pipeline_depth: bool = True
    # GPU utilization threshold for reducing pipeline depth (%)
    # When GPU utilization exceeds this, pipeline depth is reduced
    gpu_utilization_threshold_high: float = 85.0
    # GPU utilization threshold for increasing pipeline depth (%)
    # When GPU utilization is below this, pipeline depth can be increased
    gpu_utilization_threshold_low: float = 50.0
    # Minimum pipeline depth to maintain even with high GPU utilization
    min_pipeline_depth: int = 1
    # Maximum pipeline depth to allow with low GPU utilization
    max_pipeline_depth: int = 6

    # Auto-tune max_workers based on CPU cores and GPU (US-137-006)
    # When True, automatically adjusts audio_extraction_workers based on
    # available CPU cores and GPU memory
    auto_tune_workers: bool = True
    # Base worker count multiplier (0.5 = half of available cores, 1.0 = all cores)
    worker_multiplier: float = 0.5

    # Metrics export settings (US-137-003)
    # Directory to export transcription metrics after batch completion
    # Set to empty string to disable metrics export
    metrics_export_path: str = "output/transcription_metrics"
    # Export formats to enable: "json", "prometheus", "csv", or comma-separated combination
    metrics_export_formats: str = "json"

    # Predictive cache warming (US-137-004)
    # When True, warms transcription cache with videos from VIDEO_SEARCH stage
    # before CAPTION stage runs, enabling cache hits for videos that will be downloaded
    # This reduces unnecessary transcription for videos already in global cache
    predictive_cache_warming: bool = True

    # Quality gate (US-137-005)
    # When True, enables quality threshold gates for transcription
    # Transcriptions below min_quality_threshold will be rejected and retried
    quality_gate_enabled: bool = True
    # Minimum average word confidence required for transcription to pass quality gate
    # Value between 0.0 and 1.0. Default 0.5 is reasonable for most use cases
    # Transcriptions with avg_word_confidence below this will be rejected
    min_quality_threshold: float = 0.5

    # GPU memory management (US-137-010)
    # Enable GPU memory monitoring and preemption
    gpu_memory_monitoring_enabled: bool = True
    # Memory usage threshold (%) at which to trigger preemption
    # When GPU memory exceeds this threshold, transcription is paused
    gpu_memory_threshold_percent: float = 85.0
    # Memory recovery strategy when threshold is exceeded:
    #   - "wait_and_retry": Wait for memory to free up, then retry
    #   - "reduce_batch_size": Reduce batch size and continue
    #   - "fallback_to_cpu": Fall back to CPU processing
    memory_recovery_strategy: str = "wait_and_retry"
    # How often to check GPU memory (seconds)
    memory_check_interval_seconds: float = 2.0
    # Maximum time to wait for memory to free up before taking action (seconds)
    memory_recovery_max_wait_seconds: float = 60.0
    # Batch size reduction factor when memory is under pressure (0.1-1.0)
    # Lower values reduce batch size more aggressively
    memory_batch_reduction_factor: float = 0.5
    # Minimum batch size to maintain even under memory pressure
    min_batch_size_under_pressure: int = 5

    # Jitter correlation backoff (US-137-011)
    # Backoff strategy to use for retries:
    #   - "standard": Basic exponential backoff (delay * 2^attempt)
    #   - "jitter": Exponential backoff with uniform jitter to avoid thundering herd
    #   - "correlated": Jitter + correlation offset to stagger retries across workers
    #   - "adaptive": Auto-select best strategy based on historical success rate
    backoff_strategy: str = "jitter"
    # Jitter range as fraction of delay (0.0-1.0)
    # Higher values spread retries more but increase overall wait time
    backoff_jitter_factor: float = 0.3
    # Correlation factor for staggering retries across workers (0.0-1.0)
    # Higher values create more separation between worker retry times
    backoff_correlation_factor: float = 0.5
    # Maximum jitter cap in seconds to prevent excessive delays
    backoff_max_jitter_cap: float = 10.0

    # Known Whisper model names
    KNOWN_MODELS = {'tiny', 'base', 'small', 'medium', 'large', 'large-v2', 'large-v3'}
    # Known compute types for faster-whisper
    KNOWN_COMPUTE_TYPES = {'auto', 'float16', 'int8', 'float32', 'int8_float16'}

    def __post_init__(self):
        # Handle nested dataclass conversion (Rule 2)
        if self.pause_split is None:
            self.pause_split = PauseSplitConfig()
        elif isinstance(self.pause_split, dict):
            self.pause_split = PauseSplitConfig(**self.pause_split)

        # Handle post-processing config (US-124-011)
        if self.post_processing is None:
            self.post_processing = SegmentPostProcessingConfig()
        elif isinstance(self.post_processing, dict):
            self.post_processing = SegmentPostProcessingConfig(**self.post_processing)

        # Validate model name (US-66-008)
        if self.model not in self.KNOWN_MODELS:
            raise ValueError(
                f"TranscriptionConfig.model='{self.model}' is not a known Whisper model. "
                f"Valid models: {sorted(self.KNOWN_MODELS)}. "
                f"Check transcription.model in config.yaml"
            )

        # Validate compute_type (US-66-008)
        if self.compute_type not in self.KNOWN_COMPUTE_TYPES:
            raise ValueError(
                f"TranscriptionConfig.compute_type='{self.compute_type}' is not valid. "
                f"Valid types: {sorted(self.KNOWN_COMPUTE_TYPES)}. "
                f"Check transcription.compute_type in config.yaml"
            )

        # Validate minimum_gpu_memory_mb >= 100 (US-66-008)
        if self.minimum_gpu_memory_mb < 100:
            raise ValueError(
                f"TranscriptionConfig.minimum_gpu_memory_mb={self.minimum_gpu_memory_mb} "
                f"must be >= 100. Check transcription.minimum_gpu_memory_mb in config.yaml"
            )

        # Validate max_workers >= 1 (US-66-008)
        if self.max_workers < 1:
            raise ValueError(
                f"TranscriptionConfig.max_workers={self.max_workers} must be >= 1. "
                f"Check transcription.max_workers in config.yaml"
            )

        # Validate batch_size >= 1 (US-66-008, US-110-005)
        if self.batch_size < 1:
            raise ValueError(
                f"TranscriptionConfig.batch_size={self.batch_size} must be >= 1. "
                f"Check transcription.batch_size in config.yaml"
            )

        # Validate batch_wait_seconds >= 0 (US-110-005)
        if self.batch_wait_seconds < 0:
            raise ValueError(
                f"TranscriptionConfig.batch_wait_seconds={self.batch_wait_seconds} must be >= 0. "
                f"Check transcription.batch_wait_seconds in config.yaml"
            )

        # Validate gpu_transcription_timeout >= 30 (US-79-002)
        if self.gpu_transcription_timeout < 30:
            raise ValueError(
                f"TranscriptionConfig.gpu_transcription_timeout={self.gpu_transcription_timeout} "
                f"must be >= 30. Check transcription.gpu_transcription_timeout in config.yaml"
            )

        # Validate max_retries (US-79-004)
        if self.max_retries < 0 or self.max_retries > 5:
            raise ValueError(
                f"TranscriptionConfig.max_retries={self.max_retries} "
                f"must be >= 0 and <= 5. Check transcription.max_retries in config.yaml"
            )

        # Validate cache_max_age_days (US-79-005)
        if self.cache_max_age_days < 1:
            raise ValueError(
                f"TranscriptionConfig.cache_max_age_days={self.cache_max_age_days} "
                f"must be >= 1. Check transcription.cache_max_age_days in config.yaml"
            )

        # Validate whisper_num_workers (US-79-007)
        if self.whisper_num_workers < 1:
            raise ValueError(
                f"TranscriptionConfig.whisper_num_workers={self.whisper_num_workers} "
                f"must be >= 1. Check transcription.whisper_num_workers in config.yaml"
            )

        # Validate whisper_cpu_threads (US-79-007)
        if self.whisper_cpu_threads < 1:
            raise ValueError(
                f"TranscriptionConfig.whisper_cpu_threads={self.whisper_cpu_threads} "
                f"must be >= 1. Check transcription.whisper_cpu_threads in config.yaml"
            )

        # Validate progress_log_interval (US-79-008)
        if self.progress_log_interval < 1:
            raise ValueError(
                f"TranscriptionConfig.progress_log_interval={self.progress_log_interval} "
                f"must be >= 1. Check transcription.progress_log_interval in config.yaml"
            )

        # Validate retry_budget_max_attempts (US-79-010)
        if self.retry_budget_max_attempts < 0:
            raise ValueError(
                f"TranscriptionConfig.retry_budget_max_attempts={self.retry_budget_max_attempts} "
                f"must be >= 0. Check transcription.retry_budget_max_attempts in config.yaml"
            )

        # Validate retry_budget_max_backoff_seconds (US-79-010)
        if self.retry_budget_max_backoff_seconds < 0:
            raise ValueError(
                f"TranscriptionConfig.retry_budget_max_backoff_seconds={self.retry_budget_max_backoff_seconds} "
                f"must be >= 0. Check transcription.retry_budget_max_backoff_seconds in config.yaml"
            )

        # Validate compress_cache (US-110-008) - boolean, no validation needed
        # Just ensure it's a bool (YAML may load as string)
        if not isinstance(self.compress_cache, bool):
            self.compress_cache = bool(self.compress_cache)

        # Validate min_segment_words (US-110-009) - must be non-negative integer
        if not isinstance(self.min_segment_words, int):
            try:
                self.min_segment_words = int(self.min_segment_words)
            except (TypeError, ValueError):
                self.min_segment_words = 3
        if self.min_segment_words < 0:
            self.min_segment_words = 0

        # Validate auto_model_selection (US-110-010) - boolean
        # Ensure it's a bool (YAML may load as string)
        if not isinstance(self.auto_model_selection, bool):
            self.auto_model_selection = bool(self.auto_model_selection)

        # Validate and set audio_extraction_workers (US-60-010)
        cpu_count = os.cpu_count() or 4  # Fallback to 4 if cpu_count() returns None
        default_workers = min(4, cpu_count)

        if self.audio_extraction_workers <= 0:
            # Use default: min(4, cpu_count())
            self.audio_extraction_workers = default_workers
        elif self.audio_extraction_workers > cpu_count:
            # Cap at cpu_count() to prevent over-subscription
            self.audio_extraction_workers = cpu_count

        # Validate GPU memory management settings (US-137-010)
        valid_recovery_strategies = {'wait_and_retry', 'reduce_batch_size', 'fallback_to_cpu'}
        if self.memory_recovery_strategy not in valid_recovery_strategies:
            raise ValueError(
                f"TranscriptionConfig.memory_recovery_strategy='{self.memory_recovery_strategy}' "
                f"must be one of: {sorted(valid_recovery_strategies)}. "
                f"Check transcription.memory_recovery_strategy in config.yaml"
            )

        # Validate gpu_memory_threshold_percent (50-99)
        if not 50.0 <= self.gpu_memory_threshold_percent <= 99.0:
            raise ValueError(
                f"TranscriptionConfig.gpu_memory_threshold_percent={self.gpu_memory_threshold_percent} "
                f"must be between 50.0 and 99.0. "
                f"Check transcription.gpu_memory_threshold_percent in config.yaml"
            )

        # Validate memory_check_interval_seconds (> 0)
        if self.memory_check_interval_seconds <= 0:
            raise ValueError(
                f"TranscriptionConfig.memory_check_interval_seconds={self.memory_check_interval_seconds} "
                f"must be > 0. Check transcription.memory_check_interval_seconds in config.yaml"
            )

        # Validate memory_recovery_max_wait_seconds (> 0)
        if self.memory_recovery_max_wait_seconds <= 0:
            raise ValueError(
                f"TranscriptionConfig.memory_recovery_max_wait_seconds={self.memory_recovery_max_wait_seconds} "
                f"must be > 0. Check transcription.memory_recovery_max_wait_seconds in config.yaml"
            )

        # Validate memory_batch_reduction_factor (0.1-1.0)
        if not 0.1 <= self.memory_batch_reduction_factor <= 1.0:
            raise ValueError(
                f"TranscriptionConfig.memory_batch_reduction_factor={self.memory_batch_reduction_factor} "
                f"must be between 0.1 and 1.0. "
                f"Check transcription.memory_batch_reduction_factor in config.yaml"
            )

        # Validate min_batch_size_under_pressure (>= 1)
        if self.min_batch_size_under_pressure < 1:
            raise ValueError(
                f"TranscriptionConfig.min_batch_size_under_pressure={self.min_batch_size_under_pressure} "
                f"must be >= 1. Check transcription.min_batch_size_under_pressure in config.yaml"
            )

        # Validate backoff_strategy (US-137-011)
        valid_strategies = {'standard', 'jitter', 'correlated', 'adaptive'}
        if self.backoff_strategy not in valid_strategies:
            raise ValueError(
                f"TranscriptionConfig.backoff_strategy='{self.backoff_strategy}' "
                f"must be one of: {sorted(valid_strategies)}. "
                f"Check transcription.backoff_strategy in config.yaml"
            )

        # Validate backoff_jitter_factor (0.0-1.0)
        if not 0.0 <= self.backoff_jitter_factor <= 1.0:
            raise ValueError(
                f"TranscriptionConfig.backoff_jitter_factor={self.backoff_jitter_factor} "
                f"must be between 0.0 and 1.0. "
                f"Check transcription.backoff_jitter_factor in config.yaml"
            )

        # Validate backoff_correlation_factor (0.0-1.0)
        if not 0.0 <= self.backoff_correlation_factor <= 1.0:
            raise ValueError(
                f"TranscriptionConfig.backoff_correlation_factor={self.backoff_correlation_factor} "
                f"must be between 0.0 and 1.0. "
                f"Check transcription.backoff_correlation_factor in config.yaml"
            )

        # Validate backoff_max_jitter_cap (> 0)
        if self.backoff_max_jitter_cap <= 0:
            raise ValueError(
                f"TranscriptionConfig.backoff_max_jitter_cap={self.backoff_max_jitter_cap} "
                f"must be > 0. Check transcription.backoff_max_jitter_cap in config.yaml"
            )


@dataclass
class EmbeddingConfig:
    """Embedding generation settings

    Chain-of-thought: Gemini embeddings best quality, local for offline
    Reasoning: Batch processing reduces API calls by 100x
    Decision: batch_size=100 optimal for Gemini API limits
    """
    provider: str = "gemini"  # gemini, voyage, local

    # Model settings per provider
    gemini_model: str = "models/gemini-embedding-001"
    voyage_model: str = "voyage-2"
    local_model: str = "all-MiniLM-L6-v2"

    # Ollama settings
    ollama_model: str = "nomic-embed-text"
    ollama_base_url: str = "http://localhost:11434"

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
    KNOWN_PROVIDERS = {'gemini', 'openai', 'local', 'sentence_transformers', 'ollama'}

    def __post_init__(self):
        # ValueError for impossible values (negative); warn+clamp for soft limits
        if self.batch_size < 1:
            raise ValueError(
                f"EmbeddingConfig.batch_size={self.batch_size} must be >= 1. "
                f"Check embedding.batch_size in config.yaml"
            )

        if self.max_retries < 0:
            raise ValueError(
                f"EmbeddingConfig.max_retries={self.max_retries} must be >= 0. "
                f"Check embedding.max_retries in config.yaml"
            )

        if self.retry_delay < 0:
            raise ValueError(
                f"EmbeddingConfig.retry_delay={self.retry_delay} must be >= 0. "
                f"Check embedding.retry_delay in config.yaml"
            )

        if self.max_workers < 1:
            raise ValueError(
                f"EmbeddingConfig.max_workers={self.max_workers} must be >= 1. "
                f"Check embedding.max_workers in config.yaml"
            )

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
