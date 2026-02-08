"""Infrastructure configuration: Logging, caching, pipeline, API keys.

Extracted from monolithic config.py during refactoring (Jan 7, 2026).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

__all__ = [
    'LoggingConfig',
    'CacheConfig',
    'GlobalCacheConfig',
    'QualityGatesConfig',
    'PipelineConfig',
    'APIKeysConfig',
    'HealingConfig',
    'HealingLoggingConfig',
    'WatcherConfig',
    'LLMHealerConfig',
]


@dataclass
class LoggingConfig:
    """Logging settings

    Chain-of-thought: Comprehensive logging aids debugging and auditing
    Reasoning: Dual output (file + JSON) for human and machine consumption
    Decision: Track API costs for budget management
    """
    enabled: bool = True
    log_dir: str = "logs"
    log_level: str = "INFO"

    # Output targets
    log_to_file: bool = True
    log_to_console: bool = True
    generate_json_log: bool = True

    # Log content
    log_api_calls: bool = True
    log_match_decisions: bool = True
    log_performance: bool = True

    # Cost tracking
    track_api_costs: bool = True

    # Config debugging
    log_config_access: bool = False
    warn_on_hardcoded: bool = True


@dataclass
class CacheConfig:
    """Cache settings"""
    cache_dir: str = ".cache"
    cache_transcriptions: bool = True
    cache_embeddings: bool = True
    cache_scenes: bool = True
    cache_llm_responses: bool = True
    cache_vision: bool = True

    # Cross-project cache (legacy - use GlobalCacheConfig)
    cross_project_cache: bool = False
    cross_project_cache_dir: str = ""

    # Validation
    validate_cache_on_load: bool = True
    use_file_hash: bool = True


@dataclass
class GlobalCacheConfig:
    """Global cache settings for cross-project video reuse

    Enables sharing of video cache (transcripts, embeddings, scenes)
    across multiple projects. Videos from past projects can be reused
    if they match the current project's keywords/topics.
    """
    enabled: bool = True
    cache_dir: str = "~/.matcher_global_cache"  # Expands ~ to home dir

    # Pre-download optimization
    check_before_download: bool = True  # Query cache before downloading
    redownload_deleted: bool = True     # Re-download if cached video was deleted
    prompt_reuse: bool = True           # Ask user before reusing (false = auto-reuse)

    # Relevance thresholds
    min_keyword_similarity: float = 0.8  # Fuzzy match threshold for keywords
    min_topic_overlap: float = 0.3       # Topic relevance threshold

    # Limits
    max_reuse_videos: int = 50           # Max videos to reuse from cache per project
    max_redownload: int = 10             # Max deleted videos to re-download

    # What to share globally
    share_transcripts: bool = True
    share_embeddings: bool = True
    share_scenes: bool = True
    share_face_detection: bool = True

    # Priority boost for current project videos in matching
    current_project_boost: float = 0.1


@dataclass
class QualityGatesConfig:
    """Quality gate thresholds checked between pipeline stages.

    Quality gates validate intermediate results and emit warnings
    when coverage or quality falls below expected thresholds.
    Gates are non-blocking by default — they log warnings but
    do not abort the pipeline.
    """
    min_match_coverage: float = 0.5  # Minimum fraction of segments that must have a match


@dataclass
class PipelineConfig:
    """Pipeline automation settings"""
    # Stage control - skip individual stages
    skip_download: bool = False        # Skip video download (use existing videos)
    skip_image_search: bool = False    # Skip entity image search
    skip_transcription: bool = False   # Skip video transcription (use cached)
    skip_scene_detection: bool = False # Skip scene detection
    skip_matching: bool = False        # Skip matching stage

    # Video source directory (used when skip_download=true)
    # Set to absolute path of folder containing videos
    video_source_dir: str = ""

    # Resume/retry
    resume_enabled: bool = True
    max_retries: int = 3

    # Checkpoint backup rotation (US-51-007)
    checkpoint_backup_count: int = 3  # Number of rotated backup files to keep

    # Parallel processing
    parallel_transcription: bool = True
    parallel_embedding: bool = True

    # Quality gates (US-81-005)
    quality_gates: QualityGatesConfig = field(default_factory=QualityGatesConfig)

    def __post_init__(self):
        """Convert dict configs to dataclass instances (per Rule 2)."""
        if isinstance(self.quality_gates, dict):
            self.quality_gates = QualityGatesConfig(**self.quality_gates)


@dataclass
class APIKeysConfig:
    """API keys (loaded from environment)"""
    gemini_api_key: str = ""
    anthropic_api_key: str = ""
    voyage_api_key: str = ""
    pexels_api_key: str = ""
    pixabay_api_key: str = ""
    unsplash_api_key: str = ""

    def __post_init__(self):
        """Load from environment if not set"""
        self.gemini_api_key = self.gemini_api_key or os.getenv("GEMINI_API_KEY", "")
        self.anthropic_api_key = self.anthropic_api_key or os.getenv("ANTHROPIC_API_KEY", "")
        self.voyage_api_key = self.voyage_api_key or os.getenv("VOYAGE_API_KEY", "")
        self.pexels_api_key = self.pexels_api_key or os.getenv("PEXELS_API_KEY", "")
        self.pixabay_api_key = self.pixabay_api_key or os.getenv("PIXABAY_API_KEY", "")
        self.unsplash_api_key = self.unsplash_api_key or os.getenv("UNSPLASH_API_KEY", "")


@dataclass
class HealingLoggingConfig:
    """Logging settings for the healing system."""
    enabled: bool = True
    log_dir: str = "logs"
    json_log: bool = True
    console_format: str = "box"  # "box", "simple", "minimal"
    include_prompts: bool = False  # Include full LLM prompts (verbose)
    include_responses: bool = False  # Include full LLM responses (verbose)


@dataclass
class WatcherConfig:
    """Configuration for the watcher agent (local LLM for error classification).

    The watcher uses a local Ollama model for fast error triage,
    deciding which healer to try first and whether to escalate to LLM healer.
    """
    enabled: bool = True
    provider: str = "ollama"  # ollama, anthropic, gemini
    model: str = "llama3.2"   # Local model for fast triage
    fallback_model: str = "llama3.1"  # Try if primary unavailable
    host: str = "http://localhost:11434"  # Ollama server URL
    timeout: float = 30.0  # Max time for classification
    escalate_threshold: float = 0.7  # Confidence below this escalates to LLM healer
    max_failures: int = 3  # Disable after N consecutive failures
    recheck_interval_seconds: float = 300.0  # Re-check availability every 5 min
    warmup_on_preflight: bool = True  # Pre-warm model during preflight


@dataclass
class LLMHealerConfig:
    """Configuration for the LLM healer (Claude for complex error analysis).

    The LLM healer uses Claude (or fallback providers) to analyze errors
    that standard pattern-based healers cannot handle.
    """
    enabled: bool = True
    provider: str = "anthropic"  # anthropic, gemini, ollama
    model: str = "claude-sonnet-4-20250514"  # Claude Sonnet for cost-effectiveness
    max_tokens: int = 4096
    timeout: float = 60.0
    max_retries: int = 3  # Self-heal retries before giving up
    include_stack_trace: bool = True  # Include stack trace in context
    include_config_context: bool = True  # Include relevant config in context
    include_file_snippets: bool = True  # Include code snippets in context
    max_snippet_lines: int = 50  # Max lines of code per snippet
    max_context_chars: int = 12000  # Max chars for LLM context (~3000 tokens)
    recheck_interval_seconds: float = 300.0  # Re-check availability every 5 min
    max_failures: int = 3  # Disable after N consecutive failures


@dataclass
class HealingConfig:
    """Self-healing pipeline configuration.

    Controls automatic error recovery during pipeline execution.
    Healers detect specific error categories and attempt automatic fixes.

    The two-tier LLM system consists of:
    - Watcher (local Ollama): Fast error classification
    - LLM Healer (Claude): Complex error analysis when standard healers fail
    """
    # Enable/disable self-healing
    enabled: bool = True

    # Healing strategy: aggressive, conservative, interactive, minimal
    strategy: str = "conservative"

    # Maximum heal attempts per stage
    max_attempts_per_stage: int = 3

    # Maximum total heals before giving up
    max_total_heals: int = 20

    # Delay between heal attempts (seconds)
    heal_delay: float = 2.0

    # Run preflight checks before pipeline
    run_preflight: bool = True

    # Auto-fix issues found in preflight
    auto_fix_preflight: bool = True

    # Enable config rollback on failure
    enable_rollback: bool = True

    # Print healing report after pipeline completes
    print_report: bool = True

    # Nested configs for two-tier LLM delegation
    logging: HealingLoggingConfig = field(default_factory=HealingLoggingConfig)
    watcher: WatcherConfig = field(default_factory=WatcherConfig)
    llm_healer: LLMHealerConfig = field(default_factory=LLMHealerConfig)

    def __post_init__(self):
        """Convert dict configs to dataclass instances (per Rule 2)."""
        if isinstance(self.logging, dict):
            self.logging = HealingLoggingConfig(**self.logging)
        if isinstance(self.watcher, dict):
            self.watcher = WatcherConfig(**self.watcher)
        if isinstance(self.llm_healer, dict):
            self.llm_healer = LLMHealerConfig(**self.llm_healer)
