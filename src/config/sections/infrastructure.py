"""Infrastructure configuration: Logging, caching, pipeline, API keys.

Extracted from monolithic config.py during refactoring (Jan 7, 2026).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple

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
    'StageTimeoutSettingsConfig',
    'DriftRuleConfig',
    'DriftRulesConfig',
    'APIKeysConfig',
    'HealingConfig',
    'HealingLoggingConfig',
    'WatcherConfig',
    'LLMHealerConfig',
    'CheckpointCompressionConfig',
    'CheckpointAutoRepairConfig',
    'CheckpointAutoCleanupConfig',
    'UnifiedErrorAggregationConfig',
    'CrossKeywordRetryLearningConfig',
    'WebhookConfig',
    'ValidationWebhookConfig',
    'ResourcePredictionConfig',
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

    # Intelligent cache invalidation (US-137-012)
    auto_invalidate_stale: bool = True  # Auto-invalidate when video metadata changes


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

    Jitter factor (0.0-1.0) adds randomization to prevent thundering herd:
    - 0.0 = no jitter (deterministic)
    - 0.2 = ±20% randomization (default)
    - 1.0 = full randomization (delay * random(0, 1))
    """
    strategy: str = "exponential"  # exponential, linear, fixed
    base_delay: float = 2.0  # Base delay in seconds
    max_delay: float = 60.0  # Maximum delay cap in seconds
    jitter_factor: float = 0.2  # Jitter to prevent thundering herd (0.0-1.0)


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


@dataclass
class StageTimeoutSettingsConfig:
    """Global timeout settings for all pipeline stages (US-108-002).

    Provides a centralized way to configure timeouts across all stages,
    with per-stage overrides available via stage_overrides.
    """
    default_timeout_seconds: int = 3600  # Default 1 hour timeout for all stages
    stage_overrides: Dict[str, int] = field(default_factory=dict)  # Per-stage overrides (seconds)
    enable_timeout: bool = True  # Whether timeout enforcement is globally enabled

    def __post_init__(self):
        """Validate timeout configuration."""
        if self.default_timeout_seconds < 1:
            raise ValueError(
                f"default_timeout_seconds must be >= 1, got {self.default_timeout_seconds}"
            )
        # Convert stage_overrides values to int if needed
        if self.stage_overrides:
            self.stage_overrides = {
                k: int(v) for k, v in self.stage_overrides.items()
            }

    def get_timeout_for_stage(self, stage_name: str) -> int:
        """Get the timeout value for a specific stage.

        Args:
            stage_name: Name of the stage

        Returns:
            Timeout in seconds (either override or default)
        """
        return self.stage_overrides.get(stage_name, self.default_timeout_seconds)


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
class CheckpointCompressionConfig:
    """US-115-003: Checkpoint compression configuration.

    Enables gzip compression for main checkpoint files to reduce
    disk space usage for large projects.

    Configure in config.yaml under pipeline.checkpoint_compression.
    """
    # Enable/disable checkpoint compression
    # When True, checkpoint files are gzip compressed
    # When False, checkpoint files are saved as plain JSON
    enabled: bool = True

    # Compression level (1-9, where 1 is fastest, 9 is best compression)
    # 6 is a good balance between speed and compression ratio
    compression_level: int = 6

    def __post_init__(self):
        if not 1 <= self.compression_level <= 9:
            raise ValueError(
                f"CheckpointCompressionConfig.compression_level must be in range 1-9, got {self.compression_level}"
            )


@dataclass
class CheckpointAutoCleanupConfig:
    """US-115-010: Automatic checkpoint cleanup configuration.

    Enables automatic cleanup of stale checkpoint backup files and orphaned
    temporary files from crashed saves during pipeline startup.

    Configure in config.yaml under pipeline.auto_cleanup.
    """
    # Enable/disable automatic checkpoint cleanup on pipeline startup
    enabled: bool = True

    # Maximum age in days for checkpoint backup files
    # Files older than this will be deleted
    max_age_days: int = 7

    def __post_init__(self):
        if self.max_age_days < 1:
            raise ValueError(
                f"CheckpointAutoCleanupConfig.max_age_days must be >= 1, got {self.max_age_days}"
            )


@dataclass
class CheckpointHotBackupConfig:
    """US-130-007: Periodic hot backup configuration.

    Enables automatic copying of checkpoint to a secondary location after each
    successful stage save. Hot backups are named with timestamp and preserved
    indefinitely (or per retention settings).

    Configure in config.yaml under pipeline.hot_backup.
    """
    # Enable/disable hot backup to secondary location
    enabled: bool = False

    # Secondary location for hot backups (local path or network share)
    # Examples: "E:/Backups/checkpoints", "//server/share/checkpoints"
    hot_backup_path: str = ""

    # Retention: number of hot backup files to keep (0 = keep all)
    retention_count: int = 0

    # Filename template for hot backups
    # {timestamp} is replaced with ISO format datetime
    filename_template: str = "checkpoint_{timestamp}.json"

    def __post_init__(self):
        if self.enabled and not self.hot_backup_path:
            raise ValueError(
                "CheckpointHotBackupConfig.enabled requires hot_backup_path to be set"
            )


@dataclass
class CloudBackupConfig:
    """US-130-010: Cloud/remote backup configuration.

    Enables automatic uploading of checkpoints to S3-compatible cloud storage
    (AWS S3, MinIO, Backblaze B2, etc.) for offsite disaster recovery.

    Configure in config.yaml under pipeline.cloud_backup.
    """
    # Enable/disable cloud backup
    enabled: bool = False

    # S3-compatible storage path
    # Format: s3://bucket/path or s3://bucket/path?endpoint_override=...
    # For MinIO: s3://bucket/path?endpoint_override=http://localhost:9000
    # For Backblaze: s3://bucket/path?endpoint_override=https://s3.us-west-002.backblazeb2.com
    cloud_backup_path: str = ""

    # Credentials from environment variables (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)
    # Or use AWS_PROFILE for named profile authentication
    # For S3-compatible: set AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, and optionally AWS_ENDPOINT_URL
    region_name: str = "us-east-1"

    # Upload trigger: "schedule" (interval_seconds) or "per_save" (every N saves)
    upload_trigger: str = "per_save"

    # For upload_trigger="schedule": upload every N seconds (default: 3600 = 1 hour)
    interval_seconds: int = 3600

    # For upload_trigger="per_save": upload every N saves (default: 5)
    saves_per_upload: int = 5

    # Retry configuration for transient network failures
    max_retries: int = 3
    retry_backoff_seconds: float = 1.0

    # Filename template for cloud backups
    # {project} = project name, {timestamp} = ISO datetime
    filename_template: str = "{project}/checkpoint_{timestamp}.json"

    def __post_init__(self):
        if self.enabled and not self.cloud_backup_path:
            raise ValueError(
                "CloudBackupConfig.enabled requires cloud_backup_path to be set"
            )
        if self.upload_trigger not in ("schedule", "per_save"):
            raise ValueError(
                f"CloudBackupConfig.upload_trigger must be 'schedule' or 'per_save', got {self.upload_trigger}"
            )


@dataclass
class CheckpointAutoRepairConfig:
    """US-115-006: Automatic checkpoint repair configuration.

    Enables automatic repair of common checkpoint corruption patterns
    during load, including truncated JSON, missing fields, and type mismatches.

    Configure in config.yaml under pipeline.checkpoint_auto_repair.
    """
    # Enable/disable automatic checkpoint repair
    # When True, corrupted checkpoints are automatically repaired on load
    enabled: bool = True

    # Log repair actions with before/after state for debugging
    log_repairs: bool = True

    # Maximum repair attempts before giving up
    max_repair_attempts: int = 3

    def __post_init__(self):
        if self.max_repair_attempts < 1:
            raise ValueError(
                f"CheckpointAutoRepairConfig.max_repair_attempts must be >= 1, got {self.max_repair_attempts}"
            )


@dataclass
class CheckpointIntegrityConfig:
    """US-115-012: Checkpoint integrity verification configuration.

    Enables proactive integrity monitoring for checkpoint files including
    scheduled verification, on-save verification, and backup validation.

    Configure in config.yaml under pipeline.checkpoint_integrity.
    """
    # Enable/disable integrity verification after each checkpoint save
    # When True, checksum is computed and verified after each save
    verify_on_save: bool = False

    # Enable/disable scheduled periodic integrity checks
    # When True, integrity checks run on a schedule defined by cron_expression
    schedule_verification: bool = False

    # Cron expression for periodic integrity verification
    # Default: "0 * * * *" = every hour
    # Format: minute hour day month weekday
    cron_expression: str = "0 * * * *"

    # Enable/disable verification of all rotated backups
    # When True, verify_all_backups() checks each backup file
    verify_backups: bool = True

    def __post_init__(self):
        if not isinstance(self.cron_expression, str) or not self.cron_expression:
            raise ValueError(
                f"CheckpointIntegrityConfig.cron_expression must be a non-empty string, got {self.cron_expression}"
            )


# US-125-010: Webhook notification configuration
@dataclass
class WebhookConfig:
    """Configuration for pipeline event webhook notifications (US-125-010).

    Enables sending HTTP POST notifications to configurable URLs when
    pipeline events occur. Supports different webhook URLs per event type
    and includes retry logic for failed deliveries.
    """
    # Enable/disable webhook notifications
    enabled: bool = False

    # Default webhook URL (used if event-specific URLs not configured)
    default_url: str = ""

    # Timeout for webhook requests in seconds
    timeout_seconds: float = 10.0

    # Maximum number of retry attempts for failed deliveries
    max_retries: int = 3

    # Base delay between retries in seconds
    retry_base_delay: float = 1.0

    # Maximum delay between retries in seconds
    retry_max_delay: float = 30.0

    # Event-specific webhook URLs
    # Key: event type (e.g., 'after_stage', 'on_pipeline_complete')
    # Value: webhook URL to notify
    # If not set, uses default_url
    event_urls: Dict[str, str] = field(default_factory=dict)

    # Whether to include full event data in webhook payload
    include_full_data: bool = True

    # Whether to include error details in webhook payload
    include_errors: bool = True

    # Custom headers to include in webhook requests (e.g., for auth)
    headers: Dict[str, str] = field(default_factory=dict)

    # US-138-012: Stage name filter - only trigger webhooks for these stages
    # Empty list means all stages are included
    # Example: ['MATCH', 'VIDEO_SEARCH'] only triggers for those stages
    stage_filter: List[str] = field(default_factory=list)

    # US-138-012: Event type filter - only trigger webhooks for these event types
    # Empty list means all event types are included
    # Example: ['after_stage', 'on_pipeline_complete'] only triggers for those events
    event_type_filter: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Validate webhook configuration."""
        if self.timeout_seconds < 0.1:
            raise ValueError(
                f"WebhookConfig.timeout_seconds must be >= 0.1, got {self.timeout_seconds}"
            )
        if self.max_retries < 0:
            raise ValueError(
                f"WebhookConfig.max_retries must be >= 0, got {self.max_retries}"
            )
        if self.retry_base_delay < 0:
            raise ValueError(
                f"WebhookConfig.retry_base_delay must be >= 0, got {self.retry_base_delay}"
            )
        if self.retry_max_delay < self.retry_base_delay:
            raise ValueError(
                f"WebhookConfig.retry_max_delay ({self.retry_max_delay}) must be >= "
                f"retry_base_delay ({self.retry_base_delay})"
            )

    def get_url_for_event(self, event_type: str) -> Optional[str]:
        """Get the webhook URL for a specific event type.

        Args:
            event_type: The event type to get URL for

        Returns:
            The configured URL for the event, or default_url if event-specific not set,
            or None if no URLs configured at all.
        """
        if event_type in self.event_urls and self.event_urls[event_type]:
            return self.event_urls[event_type]
        return self.default_url if self.default_url else None

    def should_send_event(self, event_type: str, stage_name: str) -> bool:
        """Check if an event should be sent based on filters.

        Args:
            event_type: The event type to check
            stage_name: The stage name to check

        Returns:
            True if the event should be sent, False if filtered out.
        """
        # Check event type filter
        if self.event_type_filter and event_type not in self.event_type_filter:
            return False

        # Check stage filter (empty means all stages allowed)
        if self.stage_filter and stage_name not in self.stage_filter:
            return False

        return True


# US-142-007: External config validation webhook
@dataclass
class ValidationWebhookConfig:
    """Configuration for external config validation webhook (US-142-007).

    Enables calling an external HTTP endpoint to validate the config
    before pipeline execution. Useful for enterprise policy enforcement.

    The webhook receives the config dict as JSON POST body and should return:
    {
        "valid": bool,       # Whether config is valid
        "errors": [],        # List of error messages (if invalid)
        "warnings": []       # List of warning messages (non-blocking)
    }

    Webhook failures are logged but don't block pipeline execution.
    """
    # Enable/disable external validation webhook
    enabled: bool = False

    # Webhook URL to call for config validation
    url: str = ""

    # Timeout for webhook requests in seconds
    timeout_seconds: float = 10.0

    # Custom headers to include in webhook requests (e.g., for auth)
    headers: Dict[str, str] = field(default_factory=dict)

    # Whether to fail validation on webhook error (default: False - log only)
    fail_on_error: bool = False

    def __post_init__(self):
        """Validate webhook configuration."""
        if self.enabled and not self.url:
            raise ValueError(
                "ValidationWebhookConfig.enabled=True requires a URL to be configured"
            )
        if self.timeout_seconds < 0.1:
            raise ValueError(
                f"ValidationWebhookConfig.timeout_seconds must be >= 0.1, got {self.timeout_seconds}"
            )


@dataclass
class ResourcePredictionConfig:
    """US-138-009: Resource usage prediction configuration.

    Enables prediction of memory requirements based on historical data
    for upcoming pipeline runs. Uses voiceover duration and video count
    to predict peak memory usage and warn if predicted usage is high.
    """
    # Enable/disable resource usage predictions
    enabled: bool = True

    # Memory prediction warning threshold (percentage of available memory)
    # Emit warning when predicted memory usage exceeds this threshold
    warning_threshold_percent: float = 75.0

    # Memory prediction critical threshold (percentage of available memory)
    # Emit critical warning when predicted memory usage exceeds this threshold
    critical_threshold_percent: float = 90.0

    # Number of historical runs to use for prediction averaging
    history_window: int = 5

    # Base memory per voiceover minute (MB) - used as baseline
    base_memory_per_vo_minute: float = 50.0

    # Additional memory per video (MB) - for video metadata processing
    memory_per_video: float = 5.0

    # Additional memory per segment (MB) - for matching operations
    memory_per_segment: float = 2.0

    def __post_init__(self):
        """Validate prediction configuration."""
        if not 0.0 <= self.warning_threshold_percent <= 100.0:
            raise ValueError(
                f"warning_threshold_percent must be 0-100, got {self.warning_threshold_percent}"
            )
        if not 0.0 <= self.critical_threshold_percent <= 100.0:
            raise ValueError(
                f"critical_threshold_percent must be 0-100, got {self.critical_threshold_percent}"
            )
        if self.warning_threshold_percent > self.critical_threshold_percent:
            raise ValueError(
                f"warning_threshold_percent ({self.warning_threshold_percent}) must be <= "
                f"critical_threshold_percent ({self.critical_threshold_percent})"
            )
        if self.history_window < 1:
            raise ValueError(f"history_window must be >= 1, got {self.history_window}")
        if self.base_memory_per_vo_minute < 0:
            raise ValueError(f"base_memory_per_vo_minute must be >= 0, got {self.base_memory_per_vo_minute}")
        if self.memory_per_video < 0:
            raise ValueError(f"memory_per_video must be >= 0, got {self.memory_per_video}")
        if self.memory_per_segment < 0:
            raise ValueError(f"memory_per_segment must be >= 0, got {self.memory_per_segment}")


# US-138-010: Per-stage error rate tracking and alerting config
@dataclass
class ErrorRateThresholdConfig:
    """Configuration for a single stage's error rate threshold.

    US-138-010: Configurable error rate thresholds per stage type.
    """
    # Error rate threshold (0.0-1.0). Emit warning when exceeded.
    warning_threshold: float = 0.2

    # Error rate threshold (0.0-1.0). Emit critical warning when exceeded.
    critical_threshold: float = 0.5


@dataclass
class ErrorRateTrackingConfig:
    """Configuration for per-stage error rate tracking and alerting.

    US-138-010: Tracks error rates per stage and emits warnings when
    configurable thresholds are exceeded.
    """
    # Enable/disable error rate tracking
    enabled: bool = True

    # Default warning threshold (0.0-1.0)
    default_warning_threshold: float = 0.2

    # Default critical threshold (0.0-1.0)
    default_critical_threshold: float = 0.5

    # Per-stage-type threshold overrides
    stage_thresholds: Dict[str, ErrorRateThresholdConfig] = field(default_factory=dict)

    def __post_init__(self):
        """Ensure stage_thresholds is a dict of ErrorRateThresholdConfig."""
        if isinstance(self.stage_thresholds, dict):
            for key, value in self.stage_thresholds.items():
                if isinstance(value, dict):
                    self.stage_thresholds[key] = ErrorRateThresholdConfig(**value)

    def get_threshold(self, stage_type: str) -> ErrorRateThresholdConfig:
        """Get threshold config for a stage type.

        Args:
            stage_type: Stage type name (e.g., 'DOWNLOAD', 'CAPTION', 'MATCH')

        Returns:
            ErrorRateThresholdConfig for the stage type (or defaults if not configured)
        """
        if stage_type in self.stage_thresholds:
            return self.stage_thresholds[stage_type]
        return ErrorRateThresholdConfig(
            warning_threshold=self.default_warning_threshold,
            critical_threshold=self.default_critical_threshold
        )

    def check_threshold(self, stage_type: str, error_rate: float) -> Tuple[str, float]:
        """Check if error rate exceeds thresholds for a stage.

        Args:
            stage_type: Stage type name
            error_rate: Current error rate (0.0-1.0)

        Returns:
            Tuple of (status, threshold): status is 'critical', 'warning', or 'ok',
            threshold is the threshold that was exceeded (or 0.0 if ok)
        """
        threshold_config = self.get_threshold(stage_type)

        if error_rate >= threshold_config.critical_threshold:
            return ('critical', threshold_config.critical_threshold)
        elif error_rate >= threshold_config.warning_threshold:
            return ('warning', threshold_config.warning_threshold)
        return ('ok', 0.0)


@dataclass
class PipelineConfig:
    """Pipeline automation settings"""
    # Stage control - skip individual stages
    skip_download: bool = False        # Skip video download (use existing videos)
    skip_image_search: bool = False    # Skip entity image search
    skip_transcription: bool = False   # Skip video transcription (use cached)
    skip_scene_detection: bool = False # Skip scene detection
    skip_matching: bool = False        # Skip matching stage

    # Pipeline execution mode: 'full' (default 10-stage) or 'entity_only' (V9/V10/V11 only, no yt-dlp)
    mode: str = "full"

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

    # US-125-011: Health check interval configuration
    # Controls how often health checks run during pipeline execution
    # Default: 300 seconds (5 minutes) - run health checks every 5 minutes
    # Set to 0 to disable interval-based health checks (runs before each stage only)
    # Example: {"default": 300, "VIDEO_SEARCH": 120, "DOWNLOAD_SEGMENTS": 600}
    health_check_interval: dict = field(default_factory=lambda: {"default": 300})

    # Checkpoint backup rotation (US-51-007)
    checkpoint_backup_count: int = 3  # Number of rotated backup files to keep

    # US-85-003: Minimum seconds between backup rotations for intermediate saves.
    # Stage-boundary saves always rotate regardless of this interval.
    min_rotation_interval_seconds: int = 60

    # US-115-003: Checkpoint compression configuration
    # Enables gzip compression for main checkpoint files to reduce disk space
    checkpoint_compression: CheckpointCompressionConfig = field(default_factory=CheckpointCompressionConfig)

    # US-115-006: Checkpoint auto-repair configuration
    # Enables automatic repair of common checkpoint corruption patterns
    checkpoint_auto_repair: CheckpointAutoRepairConfig = field(default_factory=CheckpointAutoRepairConfig)

    # US-115-010: Automatic checkpoint cleanup configuration
    # Enables cleanup of stale checkpoint backup files on pipeline startup
    auto_cleanup: CheckpointAutoCleanupConfig = field(default_factory=CheckpointAutoCleanupConfig)

    # US-130-007: Hot backup configuration
    # Copies checkpoint to secondary location after each stage save
    hot_backup: CheckpointHotBackupConfig = field(default_factory=CheckpointHotBackupConfig)

    # US-130-010: Cloud/remote backup configuration
    # Uploads checkpoints to S3-compatible storage for offsite disaster recovery
    cloud_backup: CloudBackupConfig = field(default_factory=CloudBackupConfig)

    # US-115-012: Checkpoint integrity verification configuration
    # Enables proactive integrity monitoring for checkpoint files
    checkpoint_integrity: CheckpointIntegrityConfig = field(default_factory=CheckpointIntegrityConfig)

    # Parallel processing
    parallel_transcription: bool = True
    parallel_embedding: bool = True

    # US-106-007: Enable parallel stage execution (default: false for safety)
    # When enabled, stages with the same dependencies run concurrently via ThreadPoolExecutor
    parallel_execution: bool = False

    # US-125-007: Maximum number of concurrent stages in parallel execution
    # Only applies when parallel_execution is true. Default is 4.
    max_concurrent_stages: int = 4

    # Quality gates (US-81-005)
    quality_gates: QualityGatesConfig = field(default_factory=QualityGatesConfig)

    # US-81-009: Batch failure threshold — abort batch stages when failure rate exceeds this
    # 0.5 = abort when >50% of processed items have failed. Set to 1.0 to disable.
    batch_failure_threshold: float = 0.5
    batch_failure_min_sample: int = 5

    # US-88-006: Configurable cross-stage data drift detection
    # Detects unexpected data loss between pipeline stages
    drift_rules: DriftRulesConfig = field(default_factory=DriftRulesConfig)

    # US-88-008: Pipeline metrics export configuration
    # Exports stage timings, throughput, and error rates to JSON/Prometheus
    metrics_export: MetricsExportConfig = field(default_factory=MetricsExportConfig)

    # US-108-009: Resource monitoring thresholds
    # Warning threshold - emit warning when memory exceeds this percentage
    memory_threshold: float = 80.0
    # Critical threshold - pause pipeline when memory exceeds this percentage
    critical_memory_threshold: float = 90.0
    # Critical threshold - pause pipeline when CPU exceeds this percentage
    critical_cpu_threshold: float = 95.0

    # US-125-010: Webhook notification configuration
    # Enables HTTP POST notifications for pipeline events
    webhook: WebhookConfig = field(default_factory=WebhookConfig)

    # US-138-009: Resource usage prediction configuration
    # Predicts memory requirements based on voiceover duration and video count
    resource_prediction: ResourcePredictionConfig = field(default_factory=ResourcePredictionConfig)

    # US-138-010: Per-stage error rate tracking and alerting configuration
    # Tracks error rates per stage and emits warnings when thresholds exceeded
    error_rate_tracking: ErrorRateTrackingConfig = field(default_factory=ErrorRateTrackingConfig)

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
        # US-115-003: Convert checkpoint_compression to CheckpointCompressionConfig
        if isinstance(self.checkpoint_compression, dict):
            self.checkpoint_compression = CheckpointCompressionConfig(**self.checkpoint_compression)
        # US-115-006: Convert checkpoint_auto_repair to CheckpointAutoRepairConfig
        if isinstance(self.checkpoint_auto_repair, dict):
            self.checkpoint_auto_repair = CheckpointAutoRepairConfig(**self.checkpoint_auto_repair)
        # US-115-010: Convert auto_cleanup to CheckpointAutoCleanupConfig
        if isinstance(self.auto_cleanup, dict):
            self.auto_cleanup = CheckpointAutoCleanupConfig(**self.auto_cleanup)
        # US-130-007: Convert hot_backup to CheckpointHotBackupConfig
        if isinstance(self.hot_backup, dict):
            self.hot_backup = CheckpointHotBackupConfig(**self.hot_backup)
        # US-115-012: Convert checkpoint_integrity to CheckpointIntegrityConfig
        if isinstance(self.checkpoint_integrity, dict):
            self.checkpoint_integrity = CheckpointIntegrityConfig(**self.checkpoint_integrity)
        # US-125-010: Convert webhook to WebhookConfig
        if isinstance(self.webhook, dict):
            self.webhook = WebhookConfig(**self.webhook)
        # US-138-009: Convert resource_prediction to ResourcePredictionConfig
        if isinstance(self.resource_prediction, dict):
            self.resource_prediction = ResourcePredictionConfig(**self.resource_prediction)
        # US-138-010: Convert error_rate_tracking to ErrorRateTrackingConfig
        if isinstance(self.error_rate_tracking, dict):
            self.error_rate_tracking = ErrorRateTrackingConfig(**self.error_rate_tracking)
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
            if not 0.0 <= retry_config.strategy.jitter_factor <= 1.0:
                raise ValueError(
                    f"retry_policy.{stage_name}.strategy.jitter_factor must be between 0.0 and 1.0, "
                    f"got {retry_config.strategy.jitter_factor}"
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


# US-123-011: Multi-source error aggregation config
@dataclass
class UnifiedErrorAggregationConfig:
    """Configuration for unified error aggregation across pipeline sources.

    Enables tracking and correlation of errors across DOWNLOAD, CAPTION,
    and TRANSCRIPTION pipeline stages for unified rate limit analysis.

    When enabled, aggregates error statistics from all sources to identify:
    - Cross-source rate limit patterns
    - Common infrastructure issues (IP, cookies)
    - Overall pipeline health metrics
    """
    # Enable/disable unified error aggregation
    enabled: bool = True

    # Enable cross-source correlation detection
    enable_correlation: bool = True

    # Time window for correlation detection (minutes)
    correlation_window_minutes: int = 5

    # Minimum number of sources required for correlation
    min_sources_for_correlation: int = 2

    # Enable unified rate limit stats reporting
    report_rate_limits: bool = True

    # Include in pipeline summary logs
    log_summary: bool = True


class CrossKeywordRetryLearningConfig:

    def __post_init__(self):
        """Set default categories if not provided."""
        if self.known_categories is None:
            self.known_categories = [
                'stock_footage', 'documentary', 'travel', 'cinematic',
                'interview', 'general', 'tech', 'news', 'tutorial',
                'music', 'gaming', 'sports', 'cooking', 'fitness',
                'education', 'entertainment', 'science', 'nature'
            ]
        elif isinstance(self.known_categories, list):
            # Already a list, ensure it's proper
            pass

