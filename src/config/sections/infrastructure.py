"""Infrastructure configuration: Logging, caching, pipeline, API keys.

Extracted from monolithic config.py during refactoring (Jan 7, 2026).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional

__all__ = [
    'LoggingConfig',
    'CacheConfig',
    'GlobalCacheConfig',
    'QualityGatesConfig',
    'PipelineConfig',
    'MetricsExportConfig',
    'RetryStrategyConfig',
    'StageRetryConfig',
    'StageTimeoutConfig',
    'DriftRuleConfig',
    'DriftRulesConfig',
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
class RetryStrategyConfig:
    """Configuration for retry strategy per stage.

    Supports different backoff strategies:
    - exponential: Delay doubles each retry (2, 4, 8, 16...)
    - linear: Delay increases by fixed amount each retry (2, 4, 6, 8...)
    - fixed: Same delay each retry (2, 2, 2, 2...)
    """
    strategy: str = "exponential"  # exponential, linear, fixed
    base_delay: float = 2.0  # Base delay in seconds
    max_delay: float = 60.0  # Maximum delay cap in seconds


@dataclass
class StageRetryConfig:
    """Retry configuration for a specific pipeline stage."""
    max_attempts: int = 3  # Maximum retry attempts (1 = no retry)
    enabled: bool = True  # Whether retries are enabled for this stage
    strategy: RetryStrategyConfig = field(default_factory=RetryStrategyConfig)

    def __post_init__(self):
        """Convert dict configs to dataclass instances (per Rule 2)."""
        if isinstance(self.strategy, dict):
            self.strategy = RetryStrategyConfig(**self.strategy)


@dataclass
class StageTimeoutConfig:
    """Timeout configuration for a specific pipeline stage (US-88-007).

    Controls maximum execution time for a stage before it's terminated
    with partial progress saved.
    """
    timeout_seconds: int = 3600  # Default 1 hour timeout for long-running stages
    enabled: bool = True  # Whether timeout enforcement is enabled
    save_on_timeout: bool = True  # Save partial progress before timeout

    def __post_init__(self):
        """Validate timeout configuration."""
        if self.timeout_seconds < 1:
            raise ValueError(
                f"timeout_seconds must be >= 1, got {self.timeout_seconds}"
            )


# Threshold type for drift detection
DriftThresholdType = Literal["ratio", "absolute_count", "percentage"]

# Severity level for drift detection
DriftSeverity = Literal["warning", "error"]


@dataclass
class DriftRuleConfig:
    """Configuration for a single data drift detection rule.

    Defines what to check after a stage completes - comparing output field
    counts against source field counts to detect data loss or unexpected
    reduction in processed items.
    """
    # Stage that triggers this check (after this stage completes)
    trigger_stage: str = ""
    # Source field in pipeline state (the input)
    source_field: str = ""
    # Target field in pipeline state (the output to check)
    target_field: str = ""
    # Threshold type: ratio (0.0-1.0), absolute_count (exact number), percentage (0-100)
    threshold_type: DriftThresholdType = "ratio"
    # Threshold value - interpretation depends on threshold_type
    # - ratio: minimum ratio of target/source (e.g., 0.8 = 80%)
    # - absolute_count: minimum absolute count for target
    # - percentage: minimum percentage of source (e.g., 80 = 80%)
    threshold: float = 0.8
    # Severity: warning (non-blocking) or error (blocks pipeline)
    severity: DriftSeverity = "warning"
    # Whether this rule is enabled
    enabled: bool = True
    # Custom remediation message (optional)
    remediation: str = ""

    def get_remediation_message(self, expected: int, actual: int, ratio: float) -> str:
        """Generate actionable remediation message.

        Args:
            expected: Expected count (from source field)
            actual: Actual count (from target field)
            ratio: Computed ratio

        Returns:
            Remediation message with actionable suggestions.
        """
        if self.remediation:
            return self.remediation

        # Generate default message based on trigger stage
        stage_messages = {
            "CAPTION": "Check caption fetch failures, rate limiting, or video availability.",
            "MATCH": "Review matching criteria, search budget, or embedding quality.",
            "OUTPUT": "Verify match quality thresholds or segment extraction issues.",
            "VIDEO_SEARCH": "Check YouTube API quota, search terms, or network issues.",
            "ITERATIVE_MATCH": "Review iterative match budget or query refinement.",
            "DOWNLOAD_SEGMENTS": "Check download permissions, video availability, or segment extraction.",
        }

        base_msg = stage_messages.get(
            self.trigger_stage,
            "Review stage configuration and check for errors in previous logs."
        )

        return (
            f"Data drift detected: {self.target_field} has {actual} items but "
            f"{self.source_field} has {expected} (ratio: {ratio:.1%}). "
            f"{base_msg} "
            f"To adjust: set drift_rules.{self.trigger_stage}_{self.target_field}.threshold "
            f"or disable with enabled: false."
        )


@dataclass
class DriftRulesConfig:
    """Container for all drift detection rules.

    Provides convenient access to rules by trigger stage and
    supports both list-based and dict-based configuration.
    """
    # List of drift rules
    rules: List[DriftRuleConfig] = field(default_factory=list)
    # Whether drift detection is globally enabled
    enabled: bool = True
    # Whether to track drift history for trend analysis
    track_history: bool = True
    # Maximum number of historical drift events to keep
    max_history: int = 100

    def __post_init__(self):
        """Convert dict configs to dataclass instances (per Rule 2)."""
        if self.rules:
            converted = []
            for rule in self.rules:
                if isinstance(rule, dict):
                    converted.append(DriftRuleConfig(**rule))
                elif isinstance(rule, DriftRuleConfig):
                    converted.append(rule)
            self.rules = converted
        else:
            # Default rules if none provided
            self.rules = self._get_default_rules()

    def _get_default_rules(self) -> List[DriftRuleConfig]:
        """Return default drift rules matching hardcoded DRIFT_RULES."""
        return [
            DriftRuleConfig(
                trigger_stage="CAPTION",
                source_field="video_ids",
                target_field="caption_results",
                threshold_type="ratio",
                threshold=0.8,
                severity="warning",
                remediation="Check caption fetch failures, rate limiting, or video availability. "
                            "Consider increasing caption_first.retry_budget.max_attempts in config.",
            ),
            DriftRuleConfig(
                trigger_stage="MATCH",
                source_field="video_ids",
                target_field="text_metadata",
                threshold_type="ratio",
                threshold=0.8,
                severity="warning",
                remediation="Review matching criteria, search budget, or embedding quality. "
                            "Check iterative_match settings for additional search rounds.",
            ),
            DriftRuleConfig(
                trigger_stage="OUTPUT",
                source_field="voiceover_segments",
                target_field="matches",
                threshold_type="ratio",
                threshold=0.5,
                severity="warning",
                remediation="Verify match quality thresholds or segment extraction issues. "
                            "Check quality_gates.match_coverage threshold in config.",
            ),
        ]

    def get_rules_for_stage(self, stage_name: str) -> List[DriftRuleConfig]:
        """Get all drift rules for a specific stage.

        Args:
            stage_name: Name of the stage

        Returns:
            List of DriftRuleConfig for the given stage
        """
        return [r for r in self.rules if r.trigger_stage == stage_name and r.enabled]

    def to_list(self) -> List[Dict]:
        """Convert to list of dicts for serialization."""
        return [
            {
                "trigger_stage": r.trigger_stage,
                "source_field": r.source_field,
                "target_field": r.target_field,
                "threshold_type": r.threshold_type,
                "threshold": r.threshold,
                "severity": r.severity,
                "enabled": r.enabled,
                "remediation": r.remediation,
            }
            for r in self.rules
        ]


@dataclass
class MetricsExportConfig:
    """Pipeline metrics export settings (US-88-008)."""
    # Output directory for exported metrics
    output_dir: str = "output/metrics"

    # Export formats
    export_json: bool = True
    export_prometheus: bool = False

    # Filename options
    include_timestamp: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "output_dir": self.output_dir,
            "export_json": self.export_json,
            "export_prometheus": self.export_prometheus,
            "include_timestamp": self.include_timestamp,
        }


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

    # Per-stage retry policies (US-88-003)
    # Maps stage name to retry configuration
    # Example: {"CAPTION": {"max_attempts": 5, "strategy": "exponential"}}
    retry_policy: dict = field(default_factory=dict)

    # Per-stage timeout policies (US-88-007)
    # Maps stage name to timeout configuration
    # Default: 3600s (1 hour) for long-running stages
    # Example: {"DOWNLOAD_SEGMENTS": {"timeout_seconds": 7200, "enabled": true}}
    timeout_policy: dict = field(default_factory=dict)

    # Checkpoint backup rotation (US-51-007)
    checkpoint_backup_count: int = 3  # Number of rotated backup files to keep

    # US-85-003: Minimum seconds between backup rotations for intermediate saves.
    # Stage-boundary saves always rotate regardless of this interval.
    min_rotation_interval_seconds: int = 60

    # Parallel processing
    parallel_transcription: bool = True
    parallel_embedding: bool = True

    # Quality gates (US-81-005)
    quality_gates: QualityGatesConfig = field(default_factory=QualityGatesConfig)

    # US-81-009: Batch failure threshold — abort batch stages when failure rate exceeds this
    # 0.5 = abort when >50% of processed items have failed. Set to 1.0 to disable.
    batch_failure_threshold: float = 0.5

    # US-88-006: Configurable cross-stage data drift detection
    # Detects unexpected data loss between pipeline stages
    drift_rules: DriftRulesConfig = field(default_factory=DriftRulesConfig)

    # US-88-008: Pipeline metrics export configuration
    # Exports stage timings, throughput, and error rates to JSON/Prometheus
    metrics_export: MetricsExportConfig = field(default_factory=MetricsExportConfig)

    def __post_init__(self):
        """Convert dict configs to dataclass instances (per Rule 2)."""
        if isinstance(self.quality_gates, dict):
            self.quality_gates = QualityGatesConfig(**self.quality_gates)
        # Convert retry_policy dict values to StageRetryConfig
        if self.retry_policy:
            converted = {}
            for stage_name, config in self.retry_policy.items():
                if isinstance(config, dict):
                    converted[stage_name] = StageRetryConfig(**config)
                elif isinstance(config, StageRetryConfig):
                    converted[stage_name] = config
            self.retry_policy = converted
        # US-88-006: Convert drift_rules to DriftRulesConfig
        if isinstance(self.drift_rules, dict):
            self.drift_rules = DriftRulesConfig(**self.drift_rules)
        # US-88-008: Convert metrics_export to MetricsExportConfig
        if isinstance(self.metrics_export, dict):
            self.metrics_export = MetricsExportConfig(**self.metrics_export)
        # US-88-003: Validate retry config values
        self._validate_retry_config()
        # US-88-007: Convert timeout_policy dict values to StageTimeoutConfig
        self._convert_timeout_policy()
        # US-88-007: Validate timeout config values
        self._validate_timeout_config()

    def _convert_timeout_policy(self) -> None:
        """Convert timeout_policy dict values to StageTimeoutConfig instances."""
        if self.timeout_policy:
            converted = {}
            for stage_name, config in self.timeout_policy.items():
                if isinstance(config, dict):
                    converted[stage_name] = StageTimeoutConfig(**config)
                elif isinstance(config, StageTimeoutConfig):
                    converted[stage_name] = config
            self.timeout_policy = converted

    def _validate_timeout_config(self) -> None:
        """Validate timeout policy configuration.

        Ensures timeout_seconds is >= 1.
        Raises ValueError if validation fails.
        """
        for stage_name, timeout_config in self.timeout_policy.items():
            if timeout_config.timeout_seconds < 1:
                raise ValueError(
                    f"timeout_policy.{stage_name}.timeout_seconds must be >= 1, "
                    f"got {timeout_config.timeout_seconds}"
                )

    def _validate_retry_config(self) -> None:
        """Validate retry policy configuration.

        Ensures max_attempts is >= 1 and strategy is valid.
        Raises ValueError if validation fails.
        """
        valid_strategies = {'exponential', 'linear', 'fixed'}
        for stage_name, retry_config in self.retry_policy.items():
            if retry_config.max_attempts < 1:
                raise ValueError(
                    f"retry_policy.{stage_name}.max_attempts must be >= 1, got {retry_config.max_attempts}"
                )
            if retry_config.strategy.strategy not in valid_strategies:
                raise ValueError(
                    f"retry_policy.{stage_name}.strategy.strategy must be one of {valid_strategies}, "
                    f"got '{retry_config.strategy.strategy}'"
                )
            if retry_config.strategy.base_delay < 0:
                raise ValueError(
                    f"retry_policy.{stage_name}.strategy.base_delay must be >= 0, "
                    f"got {retry_config.strategy.base_delay}"
                )
            if retry_config.strategy.max_delay < 0:
                raise ValueError(
                    f"retry_policy.{stage_name}.strategy.max_delay must be >= 0, "
                    f"got {retry_config.strategy.max_delay}"
                )

    def get_retry_config(self, stage_name: str) -> StageRetryConfig:
        """Get retry configuration for a specific stage.

        Args:
            stage_name: Name of the stage (e.g., 'CAPTION', 'MATCH')

        Returns:
            StageRetryConfig with settings for the stage,
            or default config if stage not configured.
        """
        if stage_name in self.retry_policy:
            return self.retry_policy[stage_name]
        # Return default config with current max_retries behavior
        return StageRetryConfig(
            max_attempts=self.max_retries,
            enabled=True,
            strategy=RetryStrategyConfig(strategy="exponential", base_delay=2.0)
        )

    def get_timeout_config(self, stage_name: str) -> StageTimeoutConfig:
        """Get timeout configuration for a specific stage (US-88-007).

        Args:
            stage_name: Name of the stage (e.g., 'CAPTION', 'MATCH')

        Returns:
            StageTimeoutConfig with settings for the stage,
            or default config if stage not configured.
        """
        if stage_name in self.timeout_policy:
            return self.timeout_policy[stage_name]
        # Return default config: 3600s (1 hour) timeout
        return StageTimeoutConfig(
            timeout_seconds=3600,
            enabled=True,
            save_on_timeout=True
        )


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
