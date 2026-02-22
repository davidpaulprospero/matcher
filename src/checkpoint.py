"""
Checkpoint Manager for Pipeline Resume Functionality

Saves pipeline state after each stage so runs can be resumed if interrupted.
Keyword preset management has been extracted to src/keywords/manager.py.
"""

import gzip
import hashlib
import hmac
import json
import mmap
import os
import platform
import shutil
import sys
import time
import threading
import queue
import urllib.parse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List, TYPE_CHECKING
from dataclasses import dataclass, field, asdict
import logging

# US-130-004: Memory-mapped file loading threshold (10MB)
MMAP_THRESHOLD_BYTES = 10 * 1024 * 1024  # 10MB in bytes

# Cross-platform file locking imports
if sys.platform == 'win32':
    import msvcrt
else:
    import fcntl

if TYPE_CHECKING:
    from .config import Config
    from .stages import StageMetrics, StageType

# US-130-009: Checkpoint metadata indexing
from .checkpoint_index import CheckpointIndex, get_or_create_index
from .logging_templates import (
    log_stage_start,
    log_stage_complete,
    log_stage_skip,
    log_error_with_context,
    log_progress,
    get_correlation_id,
)

logger = logging.getLogger(__name__)


# Stage order for resume logic (7-stage simplified pipeline)
STAGE_ORDER = [
    "ANALYZE",
    "VIDEO_SEARCH",  # Search for videos without downloading
    "CAPTION",       # Fetch YouTube captions for video IDs
    "MATCH",
    "ITERATIVE_MATCH",  # Multi-pass gap filling after initial match
    "DOWNLOAD_SEGMENTS",
    "OUTPUT"
]

# Mapping from stage names to StageType for metrics validation
# US-106-005: Maps stage names to their type for completeness validation
_STAGE_NAME_TO_TYPE = {
    "ANALYZE": "processing",
    "VIDEO_SEARCH": "analysis",
    "CAPTION": "processing",
    "MATCH": "analysis",
    "ITERATIVE_MATCH": "analysis",
    "DOWNLOAD_SEGMENTS": "download",
    "OUTPUT": "output",
}

# Current checkpoint version — single source of truth for write and validation
CURRENT_CHECKPOINT_VERSION = "2.1"

# Legacy stage names for checkpoint migration
LEGACY_STAGES = [
    "DOWNLOAD", "STOCK", "BROLL_DOWNLOAD", "REMIX",
    "TRANSCRIBE", "SCENE_DETECTION", "BROLL_MATCH"
]

# STAGE_FIELD_MAP is auto-generated after CheckpointData is defined (see below).
# Terminal stages with no checkpoint data field:
_STAGE_FIELD_MAP_TERMINAL = {"OUTPUT"}


# Backward-compatible re-exports from src.keywords.manager
from src.keywords.manager import SavedKeywords, KeywordManager, format_keyword_prompt

# Re-export CheckpointMigrator for convenience (US-108-004)
from .checkpoint_migrator import CheckpointMigrator


class CloudBackupManager:
    """US-130-010: Manages cloud/remote backup of checkpoints to S3-compatible storage.

    Supports AWS S3, MinIO, Backblaze B2, and other S3-compatible providers.
    Provides configurable upload triggers (schedule or per-save) and retry logic.
    """

    def __init__(self, config: 'CloudBackupConfig', project_name: str):
        """Initialize CloudBackupManager.

        Args:
            config: CloudBackupConfig from pipeline configuration
            project_name: Name of the project (used in backup filename)
        """
        self.config = config
        self.project_name = project_name
        self._s3_client = None
        self._bucket = None
        self._key_prefix = None
        self._last_upload_time = 0.0
        self._save_count = 0
        self._parse_s3_path()

    def _parse_s3_path(self) -> None:
        """Parse S3 path from cloud_backup_path config.

        Extracts bucket name, key prefix, and endpoint override from URL.
        """
        cloud_path = self.config.cloud_backup_path
        if not cloud_path:
            return

        # Parse s3://bucket/path?query
        if not cloud_path.startswith("s3://"):
            raise ValueError(f"Invalid S3 path: {cloud_path}. Must start with s3://")

        # Extract path without scheme
        path = cloud_path[5:]  # Remove "s3://"

        # Split path and query
        if "?" in path:
            path_part, query_part = path.split("?", 1)
            # Parse query params
            params = urllib.parse.parse_qs(query_part)
            self._endpoint_override = params.get("endpoint_override", [None])[0]
        else:
            path_part = path
            self._endpoint_override = None

        # Split bucket and key prefix
        if "/" in path_part:
            self._bucket, self._key_prefix = path_part.split("/", 1)
            # Ensure key_prefix ends with / for proper path construction
            if not self._key_prefix.endswith("/"):
                self._key_prefix += "/"
        else:
            self._bucket = path_part
            self._key_prefix = ""

    def _get_s3_client(self):
        """Get or create S3 client with retry logic."""
        if self._s3_client is not None:
            return self._s3_client

        try:
            import boto3
            from botocore.config import Config
            from botocore.exceptions import ClientError, BotoCoreError
        except ImportError:
            logger.warning("boto3 not installed - cloud backup unavailable")
            return None

        # Build client config with retries
        client_config = Config(
            retries={
                'max_attempts': self.config.max_retries,
                'mode': 'adaptive'
            },
            connect_timeout=5,
            read_timeout=30,
        )

        # Build kwargs
        kwargs = {
            'service_name': 's3',
            'region_name': self.config.region_name,
            'config': client_config,
        }

        # Add endpoint URL for S3-compatible storage
        if self._endpoint_override:
            kwargs['endpoint_url'] = self._endpoint_override

        # Credentials from environment variables (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)
        # boto3 automatically reads these from environment

        try:
            self._s3_client = boto3.client(**kwargs)
            # Test connection
            self._s3_client.head_bucket(Bucket=self._bucket)
            logger.debug(f"Connected to S3 bucket: {self._bucket}")
        except Exception as e:
            logger.warning(f"Failed to connect to S3 bucket {self._bucket}: {e}")
            self._s3_client = None

        return self._s3_client

    def should_upload(self) -> bool:
        """Check if an upload should be triggered based on configured trigger.

        Returns:
            True if upload should be performed
        """
        if not self.config.enabled:
            return False

        if self.config.upload_trigger == "schedule":
            current_time = time.time()
            interval = self.config.interval_seconds
            if current_time - self._last_upload_time >= interval:
                return True
        elif self.config.upload_trigger == "per_save":
            self._save_count += 1
            if self._save_count >= self.config.saves_per_upload:
                self._save_count = 0
                return True

        return False

    def upload_checkpoint(self, checkpoint_path: Path) -> Dict[str, Any]:
        """Upload checkpoint to cloud storage.

        Args:
            checkpoint_path: Path to the checkpoint file to upload

        Returns:
            Dict with upload report:
            - success: Whether upload succeeded
            - remote_key: S3 key of uploaded file (if successful)
            - bytes_uploaded: Size of uploaded file
            - errors: List of error messages (if any)
        """
        report = {
            "success": False,
            "remote_key": None,
            "bytes_uploaded": 0,
            "errors": [],
        }

        if not self.config.enabled:
            report["errors"].append("Cloud backup is disabled")
            return report

        if not self._bucket:
            report["errors"].append("Invalid S3 configuration - no bucket specified")
            return report

        if not checkpoint_path.exists():
            report["errors"].append(f"Checkpoint file not found: {checkpoint_path}")
            logger.warning(f"Cloud backup skipped - checkpoint not found: {checkpoint_path}")
            return report

        # Get S3 client
        s3_client = self._get_s3_client()
        if not s3_client:
            report["errors"].append("Failed to connect to S3 - check credentials and network")
            return report

        try:
            # Generate key name from template
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename_template = self.config.filename_template
            filename = filename_template.replace("{project}", self.project_name)
            filename = filename.replace("{timestamp}", timestamp)
            remote_key = self._key_prefix + filename

            # Read checkpoint file
            with open(checkpoint_path, 'rb') as f:
                file_data = f.read()

            # Upload with retry logic
            max_retries = self.config.max_retries
            last_error = None

            for attempt in range(max_retries):
                try:
                    s3_client.put_object(
                        Bucket=self._bucket,
                        Key=remote_key,
                        Body=file_data,
                        ContentType='application/json',
                    )
                    report["success"] = True
                    report["remote_key"] = remote_key
                    report["bytes_uploaded"] = len(file_data)
                    self._last_upload_time = time.time()
                    logger.info(f"Cloud backup uploaded: s3://{self._bucket}/{remote_key}")
                    break
                except Exception as e:
                    last_error = e
                    if attempt < max_retries - 1:
                        backoff = self.config.retry_backoff_seconds * (2 ** attempt)
                        logger.warning(
                            f"Cloud backup upload attempt {attempt + 1} failed: {e}. "
                            f"Retrying in {backoff:.1f}s..."
                        )
                        time.sleep(backoff)

            if not report["success"]:
                report["errors"].append(f"Failed after {max_retries} attempts: {last_error}")
                logger.error(f"Cloud backup failed after {max_retries} attempts: {last_error}")

        except Exception as e:
            report["errors"].append(str(e))
            logger.warning(f"Cloud backup failed: {e}")

        return report

    def download_latest(self, checkpoint_path: Path) -> Dict[str, Any]:
        """Download latest checkpoint from cloud storage.

        Args:
            checkpoint_path: Local path where checkpoint should be saved

        Returns:
            Dict with download report:
            - success: Whether download succeeded
            - remote_key: S3 key of downloaded file
            - bytes_downloaded: Size of downloaded file
            - errors: List of error messages (if any)
        """
        report = {
            "success": False,
            "remote_key": None,
            "bytes_downloaded": 0,
            "errors": [],
        }

        if not self.config.enabled:
            report["errors"].append("Cloud backup is disabled")
            return report

        if not self._bucket:
            report["errors"].append("Invalid S3 configuration - no bucket specified")
            return report

        # Get S3 client
        s3_client = self._get_s3_client()
        if not s3_client:
            report["errors"].append("Failed to connect to S3 - check credentials and network")
            return report

        try:
            # List objects with prefix to find latest
            prefix = self._key_prefix + f"{self.project_name}/checkpoint_"
            if not prefix.endswith("_"):
                prefix = prefix.rstrip("/") + "_"

            response = s3_client.list_objects_v2(
                Bucket=self._bucket,
                Prefix=prefix,
            )

            if 'Contents' not in response or not response['Contents']:
                report["errors"].append(f"No checkpoint backups found in s3://{self._bucket}/{prefix}")
                logger.warning(f"No cloud backups found for project: {self.project_name}")
                return report

            # Get most recent object
            latest = max(response['Contents'], key=lambda x: x['LastModified'])
            remote_key = latest['Key']

            # Download the file
            last_error = None
            max_retries = self.config.max_retries

            for attempt in range(max_retries):
                try:
                    s3_response = s3_client.get_object(Bucket=self._bucket, Key=remote_key)
                    file_data = s3_response['Body'].read()

                    # Ensure parent directory exists
                    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

                    # Write to temp file first, then move
                    temp_path = checkpoint_path.with_suffix('.json.tmp')
                    with open(temp_path, 'wb') as f:
                        f.write(file_data)
                    temp_path.replace(checkpoint_path)

                    report["success"] = True
                    report["remote_key"] = remote_key
                    report["bytes_downloaded"] = len(file_data)
                    logger.info(f"Cloud backup downloaded: s3://{self._bucket}/{remote_key}")
                    break
                except Exception as e:
                    last_error = e
                    if attempt < max_retries - 1:
                        backoff = self.config.retry_backoff_seconds * (2 ** attempt)
                        logger.warning(
                            f"Cloud backup download attempt {attempt + 1} failed: {e}. "
                            f"Retrying in {backoff:.1f}s..."
                        )
                        time.sleep(backoff)

            if not report["success"]:
                report["errors"].append(f"Failed after {max_retries} attempts: {last_error}")
                logger.error(f"Cloud download failed after {max_retries} attempts: {last_error}")

        except Exception as e:
            report["errors"].append(str(e))
            logger.warning(f"Cloud backup download failed: {e}")

        return report


# Store exception classes at module level for type checking
BotoCoreError = None
ClientError = None


def _init_cloud_errors():
    """Initialize boto error classes (lazy import)."""
    global BotoCoreError, ClientError
    try:
        from botocore.exceptions import BotoCoreError, ClientError
    except ImportError:
        pass


_init_cloud_errors()


@dataclass
class CheckpointData:
    """Data saved at each checkpoint"""
    version: str = CURRENT_CHECKPOINT_VERSION
    created_at: str = ""
    updated_at: str = ""
    last_completed_stage: str = ""
    config_hash: str = ""
    # US-130-008: HMAC-SHA256 signature for checkpoint authenticity verification
    signature: str = ""
    voiceover_path: str = ""
    voiceover_hash: str = ""

    # Stage outputs (7-stage simplified pipeline)
    analyze: Dict[str, Any] = field(default_factory=dict)
    video_search: Dict[str, Any] = field(default_factory=dict)  # NEW: search results without download
    caption: Dict[str, Any] = field(default_factory=dict)
    match: Dict[str, Any] = field(default_factory=dict)
    iterative_match: Dict[str, Any] = field(default_factory=dict)
    download_segments: Dict[str, Any] = field(default_factory=dict)

    # US-71-009: Chapter detection results persisted after match stage.
    # Contains 'chapters' (list of ChapterCandidate dicts) and
    # 'listicle_groups' (list of ListicleGroup dicts).
    # Also stored inside match stage data for co-locality.
    chapter_data: Dict[str, Any] = field(default_factory=dict)

    # Stage metrics for pipeline observability (US-49-012)
    # Maps stage name -> serialized StageMetrics dict
    stage_metrics: Dict[str, Any] = field(default_factory=dict)

    # US-79-012: Transcription metrics summary from batch processing
    # Stores TranscriptionMetrics.get_summary_dict() output for cross-run comparison
    transcription_metrics: Dict[str, Any] = field(default_factory=dict)

    # US-88-009: Stage input validation cache for faster resume
    # Maps stage name -> {config_hash, validated_at, result}
    validation_cache: Dict[str, Any] = field(default_factory=dict)

    # US-89-003: Escalation state persistence for resume capability
    # Stores serialized escalation manager state (keyword tiers, metrics, timeline)
    escalation_state: Dict[str, Any] = field(default_factory=dict)

    # US-89-005: Circuit breaker health metrics for debugging and observability
    # Stores circuit breaker health metrics (trip count, state, pause duration)
    circuit_breaker_health: Dict[str, Any] = field(default_factory=dict)

    # US-109-010: Circuit breaker state persistence for faster recovery
    # Stores circuit breaker state (is_open, consecutive_failures, total_trips)
    # for resume across pipeline restarts
    circuit_breaker_state: Dict[str, Any] = field(default_factory=dict)

    # US-123-007: Cross-session rate limit state persistence
    # Stores serialized rate limit budget state (rotations, vpn_switches, backoff_time)
    # for resume across pipeline restarts
    rate_limit_state: Dict[str, Any] = field(default_factory=dict)

    # US-115-003: Dual-format storage during migration
    # Stores compressed checkpoint data during format migration
    # This allows for seamless migration between compressed and uncompressed formats
    compressed_checkpoint: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> 'CheckpointData':
        filtered = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}

        # Validate that stage data fields are dicts (not strings, lists, etc.)
        for stage_name, field_name in STAGE_FIELD_MAP.items():
            if field_name in filtered and not isinstance(filtered[field_name], dict):
                logger.warning(
                    f"CheckpointData.from_dict: stage '{field_name}' data is "
                    f"{type(filtered[field_name]).__name__}, expected dict — resetting to empty"
                )
                filtered[field_name] = {}

        return cls(**filtered)

    def validate(self, for_migration: bool = False) -> List[str]:
        """
        Validate internal consistency of checkpoint data.

        Checks:
        - Version compatibility with CURRENT_CHECKPOINT_VERSION
        - All stages before last_completed_stage have non-empty data
        - Stage data fields are non-empty dicts when stage is marked complete

        Args:
            for_migration: If True, creates backup before validation for safety.

        Returns a list of warning messages (empty if fully consistent).
        """
        warnings = []

        # AC1: Add validation for checkpoint version compatibility
        if self.version and self.version != CURRENT_CHECKPOINT_VERSION:
            # Check if it's a migratable version
            migratable_versions = ["0.9", "1.0", "2.0", "2.1"]
            if self.version not in migratable_versions:
                warnings.append(
                    f"Checkpoint version '{self.version}' is not compatible with "
                    f"current version '{CURRENT_CHECKPOINT_VERSION}' and cannot be migrated"
                )
            else:
                warnings.append(
                    f"Checkpoint version '{self.version}' will be migrated to "
                    f"'{CURRENT_CHECKPOINT_VERSION}'"
                )

        if not self.last_completed_stage:
            return warnings

        if self.last_completed_stage not in STAGE_ORDER:
            warnings.append(
                f"Unknown last_completed_stage '{self.last_completed_stage}'"
            )
            return warnings

        completed_idx = STAGE_ORDER.index(self.last_completed_stage)

        # Check all stages up to and including last_completed_stage
        for i in range(completed_idx + 1):
            stage_name = STAGE_ORDER[i]
            field_name = STAGE_FIELD_MAP.get(stage_name)
            if field_name is None:
                # No checkpoint field for this stage (e.g., OUTPUT)
                continue
            stage_data = getattr(self, field_name, {})

            # AC2: Validate that stage data fields are non-empty dicts when stage is marked complete
            if not stage_data:
                warnings.append(
                    f"Stage {stage_name} is before last_completed_stage "
                    f"({self.last_completed_stage}) but has no data"
                )
            elif not isinstance(stage_data, dict):
                warnings.append(
                    f"Stage {stage_name} data is {type(stage_data).__name__}, expected dict"
                )
            elif len(stage_data) == 0:
                warnings.append(
                    f"Stage {stage_name} is marked complete but has empty data dict"
                )

        return warnings


def _build_stage_field_map():
    """Auto-generate STAGE_FIELD_MAP from STAGE_ORDER and CheckpointData fields.

    Convention: stage name lowercased (e.g., "VIDEO_SEARCH" -> "video_search").
    Terminal stages listed in _STAGE_FIELD_MAP_TERMINAL are excluded.
    """
    field_map = {}
    checkpoint_fields = set(CheckpointData.__dataclass_fields__.keys())
    for stage in STAGE_ORDER:
        if stage in _STAGE_FIELD_MAP_TERMINAL:
            continue
        field_name = stage.lower()
        if field_name in checkpoint_fields:
            field_map[stage] = field_name
    return field_map


def _validate_stage_field_map():
    """Validate every non-terminal STAGE_ORDER entry has a CheckpointData field.

    Raises ImportError at module load time if a stage is missing its field,
    preventing silent drift between STAGE_ORDER and CheckpointData.
    """
    checkpoint_fields = set(CheckpointData.__dataclass_fields__.keys())
    missing = []
    for stage in STAGE_ORDER:
        if stage in _STAGE_FIELD_MAP_TERMINAL:
            continue
        field_name = stage.lower()
        if field_name not in checkpoint_fields:
            missing.append(f"{stage} -> {field_name}")
    if missing:
        raise ImportError(
            f"STAGE_ORDER / CheckpointData drift detected. "
            f"The following stages have no corresponding CheckpointData field: "
            f"{', '.join(missing)}. "
            f"Add a Dict[str, Any] field named after the stage (lowercased) to CheckpointData."
        )


# Build and validate at module load time
STAGE_FIELD_MAP = _build_stage_field_map()
_validate_stage_field_map()


class CheckpointManager:
    """Manages pipeline checkpoints for resume functionality"""
    
    CHECKPOINT_FILE = "checkpoint.json"
    CHECKPOINT_BACKUP = "checkpoint.backup.json"
    CHECKPOINT_HISTORY = "checkpoint.history.json"

    def __init__(self, project_dir: Path, config_hash: str = "", config: 'Config' = None):
        self.project_dir = Path(project_dir)
        self.checkpoint_path = self.project_dir / self.CHECKPOINT_FILE
        self.backup_path = self.project_dir / self.CHECKPOINT_BACKUP
        self.history_path = self.project_dir / self.CHECKPOINT_HISTORY
        self.config_hash = config_hash
        self.data: Optional[CheckpointData] = None
        self._config = config  # Store config for stage restore
        # US-51-007: Backup rotation count from config (default 5 for US-108-010)
        self._backup_count = getattr(
            getattr(config, 'pipeline', None), 'checkpoint_backup_count', 5
        ) if config else 5
        # US-85-003: Interval-gated backup rotation for intermediate saves
        self._min_rotation_interval = getattr(
            getattr(config, 'pipeline', None), 'min_rotation_interval_seconds', 60
        ) if config else 60
        self._last_rotation_time: float = 0.0  # epoch seconds of last rotation
        # US-115-002: Track last saved data for incremental saving
        self._last_saved_data: Optional[Dict[str, Any]] = None
        # US-115-003: Checkpoint compression configuration
        pipeline_config = getattr(config, 'pipeline', None) if config else None
        compression_config = getattr(pipeline_config, 'checkpoint_compression', None) if pipeline_config else None
        self._compression_enabled = getattr(compression_config, 'enabled', True) if compression_config else True
        self._compression_level = getattr(compression_config, 'compression_level', 6) if compression_config else 6
        # US-115-003: Track compression stats
        self._compression_stats: Dict[str, Any] = {}
        # US-115-006: Checkpoint auto-repair configuration
        auto_repair_config = getattr(pipeline_config, 'checkpoint_auto_repair', None) if pipeline_config else None
        self._auto_repair_enabled = getattr(auto_repair_config, 'enabled', True) if auto_repair_config else True
        self._auto_repair_log = getattr(auto_repair_config, 'log_repairs', True) if auto_repair_config else True
        self._auto_repair_max_attempts = getattr(auto_repair_config, 'max_repair_attempts', 3) if auto_repair_config else 3
        # US-115-006: Track repair stats for health diagnostics
        self._repair_stats: Dict[str, Any] = {
            "total_repair_attempts": 0,
            "successful_repairs": 0,
            "failed_repairs": 0,
            "repair_history": [],  # List of repair actions with before/after
        }
        # US-115-005: Lazy loading infrastructure
        self._raw_checkpoint_data: Optional[Dict[str, Any]] = None  # Raw dict before CheckpointData conversion
        self._stage_data_cache: Dict[str, Any] = {}  # Cache for lazily loaded stage data
        self._lazy_load_stats: Dict[str, Any] = {  # Stats tracking
            "stages_loaded": [],
            "stages_cached": 0,
            "memory_saved_bytes": 0,
            "total_load_calls": 0,
        }
        self._full_checkpoint_loaded: bool = False  # True if full checkpoint loaded
        # US-115-007: Checkpoint history tracking for metadata indexing
        self._history: List[Dict[str, Any]] = []
        self._history_loaded: bool = False
        # US-115-008: Async checkpoint saving with coalesced writes
        self._async_save_queue: queue.Queue = queue.Queue()
        self._async_save_thread: Optional[threading.Thread] = None
        self._pending_save: bool = False  # Track if async save in progress
        self._async_save_metrics: Dict[str, Any] = {
            "saves_coalesced": 0,
            "avg_queue_delay_ms": 0.0,
            "total_async_saves": 0,
            "queue_delays": [],  # Track delays for averaging
        }
        self._coalesce_window_ms: float = 50.0  # Max time to wait for coalescing (50ms)
        # US-115-010: Automatic checkpoint cleanup configuration
        auto_cleanup_config = getattr(pipeline_config, 'auto_cleanup', None) if pipeline_config else None
        self._auto_cleanup_enabled = getattr(auto_cleanup_config, 'enabled', True) if auto_cleanup_config else True
        self._auto_cleanup_max_age_days = getattr(auto_cleanup_config, 'max_age_days', 7) if auto_cleanup_config else 7
        # US-115-012: Checkpoint integrity verification configuration
        integrity_config = getattr(pipeline_config, 'checkpoint_integrity', None) if pipeline_config else None
        self._verify_on_save = getattr(integrity_config, 'verify_on_save', False) if integrity_config else False
        self._schedule_verification = getattr(integrity_config, 'schedule_verification', False) if integrity_config else False
        self._cron_expression = getattr(integrity_config, 'cron_expression', '0 * * * *') if integrity_config else '0 * * * *'
        self._verify_backups = getattr(integrity_config, 'verify_backups', True) if integrity_config else True
        # Integrity check tracking
        self._last_verified_at: Optional[str] = None
        self._all_backups_valid: bool = True
        self._integrity_check_history: List[Dict[str, Any]] = []  # Track verification history
        self._scheduled_verification_thread: Optional[threading.Thread] = None
        self._stop_scheduled_verification: threading.Event = threading.Event()
        # US-130-002: Checkpoint statistics tracking
        self._stats: Dict[str, Any] = {
            "save_count": 0,
            "total_save_time_ms": 0.0,
            "save_times_ms": [],  # Track recent save times for averaging
            "max_save_times": 50,  # Keep last 50 save times for rolling average
            "total_size_bytes": 0,
            "last_save_size_bytes": 0,
            "oldest_checkpoint": None,
            "newest_checkpoint": None,
            "per_stage_sizes": {},  # Track size per stage field
        }
        # US-130-003: File locking for concurrent process safety
        self._lock_file_path: Optional[Path] = None
        self._lock_file_handle: Any = None
        self._lock_acquired: bool = False
        self._lock_timeout_seconds: float = 5.0  # Default timeout for lock acquisition
        self._lock_stats: Dict[str, Any] = {
            "acquired_count": 0,
            "failed_count": 0,
            "release_count": 0,
        }
        # US-130-004: Memory-mapped file loading configuration
        self._mmap_enabled: bool = True  # Enable mmap for large files
        self._mmap_threshold_bytes: int = MMAP_THRESHOLD_BYTES  # Threshold for mmap (10MB default)
        self._mmap_stats: Dict[str, Any] = {
            "mmap_loads": 0,
            "standard_loads": 0,
            "mmap_fallbacks": 0,
            "total_mmap_bytes": 0,
        }
        # US-130-005: Checkpoint data deduplication for repeated data structures
        self._dedup_enabled: bool = True  # Enable content-addressable deduplication
        self._dedup_store: Dict[str, Any] = {}  # Stores unique data blocks by hash
        self._dedup_ref_prefix: str = "$ref:"  # Reference marker for deduplicated data
        self._dedup_stats: Dict[str, Any] = {
            "dedup_saves": 0,
            "total_original_bytes": 0,
            "total_deduped_bytes": 0,
            "unique_blocks": 0,
            "references_replaced": 0,
        }
        # US-130-006: Differential checkpointing configuration
        self._diff_enabled: bool = True  # Enable differential checkpointing
        self._diff_full_interval: int = 10  # Save full checkpoint every N saves
        self._diff_save_count: int = 0  # Counter for saves since last full
        self._diff_base_data: Optional[Dict[str, Any]] = None  # Reference data for diff computation
        self._diff_stats: Dict[str, Any] = {
            "diff_saves": 0,
            "full_saves": 0,
            "total_original_bytes": 0,
            "total_diff_bytes": 0,
            "space_saved_bytes": 0,
        }
        # US-130-008: Cryptographic signature configuration
        # Key can be provided via config or environment variable
        sig_config = getattr(pipeline_config, 'checkpoint_signature', None) if pipeline_config else None
        self._signature_key: Optional[str] = getattr(sig_config, 'key', None) if sig_config else None
        # Fallback to environment variable if not provided in config
        if not self._signature_key:
            self._signature_key = os.environ.get('MATCHER_CHECKPOINT_SECRET', None)
        self._signature_enabled: bool = (
            getattr(sig_config, 'enabled', True) if sig_config else True
        )
        self._signature_stats: Dict[str, Any] = {
            "signatures_computed": 0,
            "signatures_verified": 0,
            "verification_failures": 0,
        }
        # US-138-005: Corruption metrics tracking for checkpoint integrity monitoring
        self._corruption_stats: Dict[str, Any] = {
            "total_corruptions_detected": 0,
            "corruption_types": {},  # Maps corruption type -> count
            "corruption_sources": {},  # Maps source (primary/backup/rotated) -> count
            "auto_repairs_attempted": 0,
            "auto_repairs_successful": 0,
            "graceful_recoveries": 0,  # Count of graceful recoveries from corruption
            "total_data_loss_events": 0,  # Events where some data was lost
            "corruption_history": [],  # List of recent corruption events
        }
        # US-138-005: Graceful degradation mode - continue with partial data
        self._graceful_degradation_enabled: bool = True  # Continue with partial data on corruption
        self._degradation_warnings: List[str] = []  # Warnings from last load attempt

        # US-130-010: Cloud backup manager for S3-compatible storage
        cloud_config = getattr(pipeline_config, 'cloud_backup', None) if pipeline_config else None
        if isinstance(cloud_config, dict):
            from src.config.sections.infrastructure import CloudBackupConfig
            cloud_config = CloudBackupConfig(**cloud_config)
        self._cloud_backup_manager: Optional[CloudBackupManager] = None
        if cloud_config and getattr(cloud_config, 'enabled', False):
            # Get project name from config or project_dir
            project_name = ""
            if config and hasattr(config, 'project'):
                project_name = getattr(config.project, 'name', '') or ''
            if not project_name:
                project_name = self.project_dir.name
            self._cloud_backup_manager = CloudBackupManager(cloud_config, project_name)
            logger.info(f"Cloud backup enabled: {cloud_config.cloud_backup_path}")

    def _compute_signature(self, data: Dict[str, Any]) -> str:
        """US-130-008: Compute HMAC-SHA256 signature for checkpoint data.

        Args:
            data: Checkpoint data dict (JSON-serializable)

        Returns:
            HMAC-SHA256 signature as hex string, or empty string if signature disabled
        """
        if not self._signature_enabled or not self._signature_key:
            logger.debug("Signature computation skipped (disabled or no key)")
            return ""

        # Create a copy of data without the signature field for signing
        sign_data = {k: v for k, v in data.items() if k != 'signature'}
        # Serialize consistently for deterministic signature
        try:
            content = json.dumps(sign_data, sort_keys=True, default=str)
        except (TypeError, ValueError):
            content = str(sign_data)

        # Compute HMAC-SHA256
        signature = hmac.new(
            self._signature_key.encode('utf-8'),
            content.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()

        self._signature_stats["signatures_computed"] = (
            self._signature_stats.get("signatures_computed", 0) + 1
        )
        logger.debug(f"Computed checkpoint signature: {signature[:16]}...")
        return signature

    def verify_signature(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """US-130-008: Verify HMAC-SHA256 signature of checkpoint data.

        Args:
            data: Checkpoint data dict loaded from file

        Returns:
            Dict with:
            - is_valid: bool (True if signature valid or not present)
            - signature_present: bool
            - expected_signature: str (computed signature for comparison)
            - issue: str (error message if verification fails)
        """
        result = {
            "is_valid": False,
            "signature_present": False,
            "expected_signature": "",
            "issue": "",
        }

        stored_signature = data.get('signature', '')

        # No signature stored - check if we have a key (legacy checkpoint)
        if not stored_signature:
            if self._signature_key:
                result["issue"] = "No signature present in checkpoint (legacy or missing key)"
                result["is_valid"] = False  # Cannot verify - treat as invalid if key configured
                logger.warning("Checkpoint has no signature but signature key is configured")
            else:
                # No key configured - legacy checkpoint, skip verification
                result["is_valid"] = True
                result["issue"] = "No signature key configured, skipping verification"
            return result

        result["signature_present"] = True

        # No key to verify with - cannot verify
        if not self._signature_key:
            result["issue"] = "No signature key configured, cannot verify"
            result["is_valid"] = False
            return result

        # Compute expected signature
        expected = self._compute_signature(data)
        result["expected_signature"] = expected

        # Compare signatures (constant-time to prevent timing attacks)
        if hmac.compare_digest(stored_signature, expected):
            result["is_valid"] = True
            self._signature_stats["signatures_verified"] = (
                self._signature_stats.get("signatures_verified", 0) + 1
            )
            logger.debug("Checkpoint signature verified successfully")
        else:
            result["is_valid"] = False
            result["issue"] = "Signature mismatch - checkpoint may be tampered"
            self._signature_stats["verification_failures"] = (
                self._signature_stats.get("verification_failures", 0) + 1
            )
            logger.error(f"Checkpoint signature verification FAILED: stored={stored_signature[:16]}..., expected={expected[:16]}...")

        return result

    def _compute_content_hash(self, data: Any) -> str:
        """US-130-005: Compute content hash for detecting duplicate data.

        Args:
            data: Any JSON-serializable data structure

        Returns:
            SHA256 hash of the data as a hex string
        """
        # Serialize consistently (sorted keys for dicts)
        try:
            content = json.dumps(data, sort_keys=True, default=str)
        except (TypeError, ValueError):
            # Fallback for non-JSON-serializable data
            content = str(data)
        return hashlib.sha256(content.encode('utf-8')).hexdigest()

    def _deduplicate_data(self, data: Dict[str, Any]) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """US-130-005: Deduplicate repeated data structures in checkpoint.

        Replaces duplicate data blocks with references to stored unique blocks.

        Args:
            data: The checkpoint data dict to deduplicate

        Returns:
            Tuple of (deduplicated_data, dedup_store)
        """
        if not self._dedup_enabled:
            return data, {}

        dedup_store: Dict[str, Any] = {}
        deduped_data = self._process_for_dedup(data, dedup_store)

        # Track stats
        original_size = len(json.dumps(data, default=str))
        deduped_size = len(json.dumps(deduped_data, default=str))
        self._dedup_stats["total_original_bytes"] += original_size
        self._dedup_stats["total_deduped_bytes"] += deduped_size
        self._dedup_stats["unique_blocks"] = len(dedup_store)
        self._dedup_stats["dedup_saves"] += 1

        return deduped_data, dedup_store

    def _process_for_dedup(self, data: Any, dedup_store: Dict[str, Any]) -> Any:
        """Recursively process data structures for deduplication.

        Args:
            data: Current data item
            dedup_store: Store for unique data blocks

        Returns:
            Data with duplicates replaced by references
        """
        if isinstance(data, dict):
            # Don't deduplicate metadata fields or special structures
            if data.get("$ref", "").startswith(self._dedup_ref_prefix):
                return data  # Already a reference

            result = {}
            for key, value in data.items():
                result[key] = self._process_for_dedup(value, dedup_store)
            return result
        elif isinstance(data, list):
            return [self._process_for_dedup(item, dedup_store) for item in data]
        elif isinstance(data, str):
            # Don't deduplicate short strings (not worth the overhead)
            if len(data) < 50:
                return data

            # Check if this string is a duplicate
            content_hash = self._compute_content_hash(data)
            if content_hash in dedup_store:
                self._dedup_stats["references_replaced"] += 1
                return {"$ref": f"{self._dedup_ref_prefix}{content_hash}"}
            else:
                dedup_store[content_hash] = data
                return data
        elif isinstance(data, (int, float, bool, type(None))):
            return data
        else:
            # For other types, try to hash them
            try:
                content_hash = self._compute_content_hash(data)
                if content_hash in dedup_store:
                    self._dedup_stats["references_replaced"] += 1
                    return {"$ref": f"{self._dedup_ref_prefix}{content_hash}"}
                else:
                    dedup_store[content_hash] = data
                    return data
            except Exception:
                return data

    def _restore_deduplicated_data(self, data: Dict[str, Any], dedup_store: Dict[str, Any]) -> Dict[str, Any]:
        """US-130-005: Restore deduplicated data from references.

        Replaces reference objects with the actual data from dedup_store.

        Args:
            data: The checkpoint data with references
            dedup_store: The store containing unique data blocks

        Returns:
            Data with references restored to original values
        """
        if not dedup_store:
            return data

        return self._process_for_restore(data, dedup_store)

    def _process_for_restore(self, data: Any, dedup_store: Dict[str, Any]) -> Any:
        """Recursively restore deduplicated references.

        Args:
            data: Current data item
            dedup_store: Store containing unique data blocks

        Returns:
            Data with references resolved
        """
        if isinstance(data, dict):
            # Check if this is a reference
            ref_value = data.get("$ref", "")
            if ref_value.startswith(self._dedup_ref_prefix):
                content_hash = ref_value[len(self._dedup_ref_prefix):]
                return dedup_store.get(content_hash, data)  # Return original if not found

            # Recursively process
            return {key: self._process_for_restore(value, dedup_store) for key, value in data.items()}
        elif isinstance(data, list):
            return [self._process_for_restore(item, dedup_store) for item in data]
        else:
            return data

    def _compute_diff(self, current_data: Dict[str, Any], base_data: Dict[str, Any]) -> Dict[str, Any]:
        """US-130-006: Compute diff between current and base checkpoint data.

        Creates a delta-encoded representation that stores only the differences
        between current and base checkpoint. Uses simple field-level diffing.

        Args:
            current_data: The current checkpoint data dict
            base_data: The base checkpoint data dict to compare against

        Returns:
            Dict containing the diff with metadata for restoration
        """
        diff = {
            "_diff_version": 1,
            "_diff_from": base_data.get("updated_at", ""),
            "_diff_timestamp": current_data.get("updated_at", ""),
            "_diff_type": "incremental",
            "_diff_fields": {},
        }

        # Track which fields changed
        all_keys = set(current_data.keys()) | set(base_data.keys())

        for key in all_keys:
            current_value = current_data.get(key)
            base_value = base_data.get(key)

            # Check if values are different
            if current_value != base_value:
                diff["_diff_fields"][key] = current_value

        return diff

    def _apply_diff(self, base_data: Dict[str, Any], diff: Dict[str, Any]) -> Dict[str, Any]:
        """US-130-006: Apply diff to reconstruct full checkpoint data.

        Takes a base checkpoint and applies a diff to reconstruct the full state.

        Args:
            base_data: The base checkpoint data dict
            diff: The diff dict containing changes

        Returns:
            Reconstructed full checkpoint data
        """
        if not diff:
            return base_data

        # Start with base data
        result = self._deep_copy_dict(base_data)

        # Apply each changed field
        diff_fields = diff.get("_diff_fields", {})
        for key, value in diff_fields.items():
            result[key] = value

        # Copy diff metadata to result
        result["_diff_version"] = diff.get("_diff_version", 1)
        result["_diff_from"] = diff.get("_diff_from", "")
        result["_diff_applied"] = diff.get("_diff_timestamp", "")

        return result

    def _deep_copy_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a deep copy of a dict, handling nested structures."""
        import copy
        try:
            return copy.deepcopy(data)
        except Exception:
            # Fallback to json round-trip for complex objects
            return json.loads(json.dumps(data, default=str))

    def cleanup_stale_backups(self) -> Dict[str, Any]:
        """US-115-010: Clean up stale checkpoint backups and orphaned temp files.

        Called on pipeline startup to prevent disk bloat from old backup files
        and orphaned .tmp files from crashed saves.

        Returns:
            Dict with cleanup report:
            - files_removed: Number of files removed
            - bytes_freed: Total bytes freed
            - backup_files_removed: Count of old backup files
            - temp_files_removed: Count of orphaned temp files
            - errors: List of error messages (if any)
        """
        report = {
            "files_removed": 0,
            "bytes_freed": 0,
            "backup_files_removed": 0,
            "temp_files_removed": 0,
            "errors": [],
        }

        if not self._auto_cleanup_enabled:
            logger.debug("Auto cleanup disabled, skipping")
            return report

        try:
            # Calculate cutoff time
            import time
            max_age_seconds = self._auto_cleanup_max_age_days * 24 * 3600
            cutoff_time = time.time() - max_age_seconds

            # Clean up old checkpoint.backup.*.json files
            backup_pattern = "checkpoint.backup.*.json"
            for backup_file in self.project_dir.glob(backup_pattern):
                try:
                    if backup_file.stat().st_mtime < cutoff_time:
                        file_size = backup_file.stat().st_size
                        backup_file.unlink()
                        report["backup_files_removed"] += 1
                        report["files_removed"] += 1
                        report["bytes_freed"] += file_size
                        logger.info(f"Removed old checkpoint backup: {backup_file.name}")
                except OSError as e:
                    report["errors"].append(f"Failed to remove {backup_file.name}: {e}")

            # Clean up orphaned .tmp files (from crashed saves)
            for temp_file in self.project_dir.glob("*.tmp"):
                try:
                    file_size = temp_file.stat().st_size
                    temp_file.unlink()
                    report["temp_files_removed"] += 1
                    report["files_removed"] += 1
                    report["bytes_freed"] += file_size
                    logger.info(f"Removed orphaned temp file: {temp_file.name}")
                except OSError as e:
                    report["errors"].append(f"Failed to remove {temp_file.name}: {e}")

            logger.info(
                f"Checkpoint cleanup: removed {report['files_removed']} files, "
                f"freed {report['bytes_freed']} bytes"
            )

        except Exception as e:
            report["errors"].append(f"Cleanup failed: {e}")
            logger.warning(f"Checkpoint cleanup failed: {e}")

        return report

    def _do_hot_backup(self) -> Dict[str, Any]:
        """US-130-007: Copy checkpoint to secondary location after stage save.

        Hot backups are triggered after each successful stage save and stored
        with timestamp in the configured hot_backup_path.

        Returns:
            Dict with hot backup report:
            - success: Whether backup succeeded
            - backup_path: Path to the backup file (if successful)
            - bytes_copied: Size of backup file
            - errors: List of error messages (if any)
        """
        report = {
            "success": False,
            "backup_path": None,
            "bytes_copied": 0,
            "errors": [],
        }

        # Get hot backup config
        config = getattr(self, '_config', None)
        if not config:
            logger.debug("No config available for hot backup")
            return report

        # Get pipeline config with fallback for dict/object compatibility
        pipeline_config = getattr(config, 'pipeline_config', None)
        if not pipeline_config:
            pipeline_config = getattr(config, 'pipeline', None)
        if not pipeline_config:
            logger.debug("No pipeline config available for hot backup")
            return report

        # Get hot_backup config (handle dict/object)
        hot_backup_config = getattr(pipeline_config, 'hot_backup', None)
        if isinstance(hot_backup_config, dict):
            from src.config.sections.infrastructure import CheckpointHotBackupConfig
            hot_backup_config = CheckpointHotBackupConfig(**hot_backup_config)

        if not hot_backup_config:
            logger.debug("No hot_backup config available")
            return report

        if not getattr(hot_backup_config, 'enabled', False):
            logger.debug("Hot backup disabled")
            return report

        hot_backup_path = getattr(hot_backup_config, 'hot_backup_path', '')
        if not hot_backup_path:
            logger.warning("Hot backup enabled but hot_backup_path not set")
            return report

        # Check if checkpoint file exists
        if not self.checkpoint_path.exists():
            report["errors"].append(f"Checkpoint file not found: {self.checkpoint_path}")
            logger.warning(f"Hot backup skipped - checkpoint not found: {self.checkpoint_path}")
            return report

        try:
            # Create target directory if it doesn't exist
            target_dir = Path(hot_backup_path)
            target_dir.mkdir(parents=True, exist_ok=True)

            # Generate timestamped filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename_template = getattr(hot_backup_config, 'filename_template', 'checkpoint_{timestamp}.json')
            filename = filename_template.replace('{timestamp}', timestamp)
            target_path = target_dir / filename

            # Copy checkpoint to secondary location
            shutil.copy2(self.checkpoint_path, target_path)
            report["success"] = True
            report["backup_path"] = str(target_path)
            report["bytes_copied"] = target_path.stat().st_size
            logger.info(f"Hot backup created: {target_path}")

            # Handle retention (if configured)
            retention_count = getattr(hot_backup_config, 'retention_count', 0)
            if retention_count > 0:
                # Find all hot backup files sorted by modification time
                backup_files = sorted(
                    target_dir.glob("checkpoint_*.json"),
                    key=lambda p: p.stat().st_mtime
                )
                # Remove oldest files beyond retention count
                files_to_remove = backup_files[:-retention_count]
                for old_file in files_to_remove:
                    try:
                        old_file.unlink()
                        logger.debug(f"Removed old hot backup: {old_file.name}")
                    except OSError as e:
                        logger.warning(f"Failed to remove old hot backup {old_file.name}: {e}")

        except Exception as e:
            report["errors"].append(str(e))
            # Graceful failure: log warning and continue (don't fail the save)
            logger.warning(f"Hot backup failed: {e}")

        return report

    def _do_cloud_backup(self) -> Dict[str, Any]:
        """US-130-010: Upload checkpoint to cloud storage after stage save.

        Cloud backups are triggered based on configured schedule (interval) or
        per-save count (per_save). Supports AWS S3, MinIO, Backblaze B2.

        Returns:
            Dict with cloud backup report:
            - success: Whether upload succeeded
            - remote_key: S3 key of uploaded file (if successful)
            - bytes_uploaded: Size of uploaded file
            - errors: List of error messages (if any)
        """
        report = {
            "success": False,
            "remote_key": None,
            "bytes_uploaded": 0,
            "errors": [],
        }

        # Check if cloud backup manager is initialized
        if not self._cloud_backup_manager:
            logger.debug("No cloud backup manager configured")
            return report

        # Check if upload should be triggered
        if not self._cloud_backup_manager.should_upload():
            logger.debug("Cloud backup not triggered yet (waiting for trigger condition)")
            return report

        # Perform upload
        return self._cloud_backup_manager.upload_checkpoint(self.checkpoint_path)

    def _update_index(self, stage: str) -> None:
        """US-130-009: Update checkpoint metadata index after stage save.

        This method adds an entry to the checkpoint index tracking:
        - timestamp: ISO timestamp of save
        - stage: Stage that was completed
        - size: Checkpoint file size in bytes
        - config_hash: Hash of config at time of save
        - voiceover_hash: Hash of voiceover file

        Graceful failure: logs warning but doesn't fail the main save.
        """
        try:
            # Check if checkpoint file exists
            if not self.checkpoint_path.exists():
                logger.warning(f"Index update skipped - checkpoint not found: {self.checkpoint_path}")
                return

            # Get checkpoint file size
            size_bytes = self.checkpoint_path.stat().st_size

            # Get config_hash and voiceover_hash
            config_hash = self.config_hash
            voiceover_hash = getattr(self.data, 'voiceover_hash', '') if self.data else ""

            # Create/update index and add entry
            index = get_or_create_index(self.project_dir)
            index.add_entry_and_save(
                stage=stage,
                size=size_bytes,
                config_hash=config_hash,
                voiceover_hash=voiceover_hash,
                checkpoint_file=self.checkpoint_path.name
            )

            logger.debug(f"Updated checkpoint index for stage {stage}")

        except Exception as e:
            # Graceful failure: log warning but don't fail the main save
            logger.warning(f"Index update failed: {e}")

    def exists(self) -> bool:
        """Check if a checkpoint exists"""
        return self.checkpoint_path.exists()

    def is_stale(self, max_age_hours: float = 24.0) -> bool:
        """
        Check if checkpoint is stale (older than max_age_hours).

        A stale checkpoint should be auto-cleared to avoid resuming
        from an outdated or corrupted state.

        Args:
            max_age_hours: Maximum age in hours before checkpoint is stale

        Returns:
            True if checkpoint is stale, False otherwise
        """
        if not self.data:
            return False

        try:
            # Use updated_at if available, otherwise created_at
            timestamp_str = self.data.updated_at or self.data.created_at
            if not timestamp_str:
                return True  # No timestamp = stale

            checkpoint_time = datetime.fromisoformat(timestamp_str)
            age = datetime.now() - checkpoint_time
            age_hours = age.total_seconds() / 3600

            return age_hours > max_age_hours
        except Exception:
            return True  # Can't parse timestamp = stale

    def get_age_hours(self) -> float:
        """Get checkpoint age in hours"""
        if not self.data:
            return 0.0

        try:
            timestamp_str = self.data.updated_at or self.data.created_at
            if not timestamp_str:
                return 0.0

            checkpoint_time = datetime.fromisoformat(timestamp_str)
            age = datetime.now() - checkpoint_time
            return age.total_seconds() / 3600
        except Exception:
            return 0.0
    
    def load(self) -> Optional[CheckpointData]:
        """
        Load existing checkpoint with corruption detection and timestamp comparison.

        If the main checkpoint is corrupt, attempts to restore from backup.
        If both are valid, compares timestamps and uses the newer one.
        Validates required fields and version compatibility.

        US-130-004: Uses memory-mapped file I/O for large checkpoints (>10MB).
        """
        start_time = time.perf_counter()
        correlation_id = get_correlation_id()
        checkpoint_exists = self.exists()
        log_stage_start(logger, "CHECKPOINT_LOAD", correlation_id=correlation_id, checkpoint_exists=checkpoint_exists)

        # Check file size for progress logging
        file_size_bytes = 0
        if self.checkpoint_path.exists():
            file_size_bytes = self.checkpoint_path.stat().st_size

        # US-169-007: Use log_progress for large checkpoints
        is_large_checkpoint = file_size_bytes >= self._mmap_threshold_bytes
        if is_large_checkpoint:
            logger.debug(f"Large checkpoint detected: {file_size_bytes} bytes ({file_size_bytes / 1024 / 1024:.1f}MB)")
            log_progress(
                logger,
                "CHECKPOINT_LOAD",
                10,
                1,
                10,
                correlation_id=correlation_id,
                file_size_bytes=file_size_bytes,
                is_large_checkpoint=True,
            )

        if not checkpoint_exists:
            log_stage_skip(logger, "CHECKPOINT_LOAD", reason="no checkpoint file", correlation_id=correlation_id, checkpoint_exists=False)
            return None

        # US-130-004: Try memory-mapped loading for large files first
        main_data = None
        if self._mmap_enabled and self.checkpoint_path.exists():
            file_size = self.checkpoint_path.stat().st_size
            if file_size >= self._mmap_threshold_bytes:
                logger.debug(f"Attempting mmap load for large checkpoint: {file_size} bytes")
                main_data = self._load_mmap(self.checkpoint_path)
                if main_data is not None:
                    self._mmap_stats["mmap_loads"] += 1
                else:
                    self._mmap_stats["mmap_fallbacks"] += 1

        # Fall back to standard loading if mmap disabled, not applicable, or failed
        if main_data is None:
            self._mmap_stats["standard_loads"] += 1
            main_data = self._try_load_file(self.checkpoint_path)

        # Try backup if it exists - also try mmap for large backups
        backup_data = None
        if self.backup_path.exists():
            backup_size = self.backup_path.stat().st_size
            if self._mmap_enabled and backup_size >= self._mmap_threshold_bytes:
                backup_data = self._load_mmap(self.backup_path)
            if backup_data is None:
                backup_data = self._try_load_file(self.backup_path)

        # Decide which checkpoint to use based on validity and timestamps
        data = self._select_checkpoint(main_data, backup_data)

        # US-169-007: Progress logging after checkpoint selection
        if is_large_checkpoint:
            log_progress(
                logger,
                "CHECKPOINT_LOAD",
                50,
                5,
                10,
                correlation_id=correlation_id,
                checkpoint_selected=data is not None,
            )

        # US-51-007: If both main and primary backup failed, try rotated backups
        graceful_recovery = False
        if data is None:
            for i in range(1, self._backup_count):
                rotated_path = self._get_backup_path(i)
                if rotated_path.exists():
                    rotated_data = self._try_load_file(rotated_path)
                    if rotated_data is not None:
                        logger.warning(
                            f"Restored checkpoint from rotated backup: {rotated_path.name}"
                        )
                        # US-138-005: Record graceful recovery
                        self._record_corruption_event(
                            corruption_type="rotated_backup_used",
                            source="rotated_backup",
                            repaired=True,
                        )
                        graceful_recovery = True
                        # Restore to main
                        try:
                            shutil.copy2(rotated_path, self.checkpoint_path)
                        except Exception as e:
                            logger.warning(f"Could not restore rotated backup to main: {e}")
                        data = rotated_data
                        break

        if data is None:
            # US-138-005: Record final failure - no checkpoint could be loaded
            self._record_corruption_event(
                corruption_type="complete_failure",
                source="all",
                repaired=False,
                data_lost=True,
            )
            log_error_with_context(
                logger,
                "PIPE-001",
                "Checkpoint load failed - no valid checkpoint found",
                correlation_id=get_correlation_id(),
                checkpoint_path=str(self.checkpoint_path),
            )
            return None

        # US-138-005: Track if we recovered from corruption via graceful degradation
        if graceful_recovery:
            self._degradation_warnings.append(
                "Checkpoint recovered from rotated backup due to corruption"
            )
            logger.warning(
                "Graceful degradation: Recovered checkpoint from rotated backup"
            )

        # Validate required fields
        if not self._validate_checkpoint_data(data):
            # US-138-005: Track validation failure
            self._degradation_warnings.append(
                "Checkpoint validation failed - continuing with partial data"
            )
            logger.warning("Checkpoint validation failed - data may be incomplete")
            log_error_with_context(
                logger,
                "PIPE-002",
                "Checkpoint validation failed - data may be incomplete",
                correlation_id=get_correlation_id(),
                checkpoint_path=str(self.checkpoint_path),
            )
            # US-138-005: Continue with graceful degradation
            if self._graceful_degradation_enabled:
                self._record_corruption_event(
                    corruption_type="validation_failure",
                    source="primary",
                    repaired=False,
                    data_lost=True,
                )

        # Validate internal consistency (stages before last_completed have data)
        consistency_warnings = data.validate()
        for warning in consistency_warnings:
            # US-138-005: Track consistency warnings as graceful degradation events
            if self._graceful_degradation_enabled and not graceful_recovery:
                self._degradation_warnings.append(f"Consistency: {warning}")
            logger.warning(f"Checkpoint consistency: {warning}")

        # US-159-010: Log validation result summary
        # US-166-008: Validation results at DEBUG level per acceptance criteria
        stages_checked = list(STAGE_ORDER)
        validation_passed = len(consistency_warnings) == 0

        # US-169-007: Progress logging after validation
        if is_large_checkpoint:
            log_progress(
                logger,
                "CHECKPOINT_LOAD",
                80,
                8,
                10,
                correlation_id=correlation_id,
                validation_passed=validation_passed,
                consistency_warnings=len(consistency_warnings),
            )

        if consistency_warnings:
            logger.debug(
                f"Checkpoint validation: invalid, warnings={len(consistency_warnings)}, "
                f"stages_checked={stages_checked}"
            )
        else:
            logger.debug(
                f"Checkpoint validation: valid, stages_checked={stages_checked}"
            )

        self.data = data
        # US-115-005: Mark full checkpoint as loaded
        self._full_checkpoint_loaded = True
        # Also store raw data for reference
        self._raw_checkpoint_data = data.to_dict() if hasattr(data, 'to_dict') else data

        # US-130-012: Normalize paths for cross-platform compatibility
        # This detects if paths were stored on a different platform and converts them
        if self._detect_and_normalize_paths(data):
            logger.info("Checkpoint paths were normalized for current platform")

        # US-115-002: Initialize last saved data for incremental saving
        self._last_saved_data = data.to_dict() if hasattr(data, 'to_dict') else data

        # US-79-012: Log previous transcription metrics for cross-run comparison
        self.log_previous_transcription_metrics()

        # Log load time
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        # US-159-010: Log checkpoint load operation with source info
        checkpoint_source = "main"
        if graceful_recovery:
            checkpoint_source = "rotated_backup"
        elif self.backup_path.exists() and main_data is None and backup_data is not None:
            checkpoint_source = "backup"
        elif self._cloud_backup_manager:
            # Check if cloud restore happened - would be logged elsewhere but we can infer
            pass

        # Get checkpoint file size and timestamp
        file_size_bytes = 0
        checkpoint_timestamp = None
        if self.checkpoint_path.exists():
            try:
                file_size_bytes = self.checkpoint_path.stat().st_size
                checkpoint_timestamp = datetime.fromtimestamp(self.checkpoint_path.stat().st_mtime).isoformat()
            except OSError:
                pass

        # US-164-007: Log checkpoint version and stage progress at INFO level
        checkpoint_version = getattr(self.data, 'version', CURRENT_CHECKPOINT_VERSION)
        # Count total keys in checkpoint data for observability
        data_dict = self.data.to_dict() if hasattr(self.data, 'to_dict') else self.data
        keys_count = len(data_dict) if isinstance(data_dict, dict) else 0

        logger.info(
            f"Checkpoint loaded in {elapsed_ms:.1f}ms: source={checkpoint_source}, "
            f"path={self.checkpoint_path}, version={checkpoint_version}, "
            f"last_completed={self.data.last_completed_stage}, "
            f"file_size={file_size_bytes}, timestamp={checkpoint_timestamp}, "
            f"keys_count={keys_count}"
        )

        log_stage_complete(
            logger,
            "CHECKPOINT_LOAD",
            correlation_id=correlation_id,
            checkpoint_source=checkpoint_source,
            last_stage=self.data.last_completed_stage,
            checkpoint_exists=True,
            file_size_bytes=file_size_bytes,
            checkpoint_timestamp=checkpoint_timestamp,
        )

        # US-167-008: Log loaded entry counts at DEBUG level
        stages_with_data = []
        for stage_name in STAGE_ORDER:
            stage_key = stage_name.lower()
            field_name = STAGE_FIELD_MAP.get(stage_name, stage_key)
            if hasattr(self.data, field_name) and getattr(self.data, field_name):
                stage_data = getattr(self.data, field_name)
                if isinstance(stage_data, dict) and stage_data:
                    stages_with_data.append(stage_name)

        logger.debug(
            f"Checkpoint loaded: {len(stages_with_data)} stages with data: {stages_with_data}"
        )

        return self.data

    # US-115-005: Lazy loading methods for checkpoint stage data

    def load_stage_data(self, stage_name: str) -> Dict[str, Any]:
        """
        US-115-005: Load only the requested stage data (lazy loading).

        Only loads the specific stage from checkpoint, caching it for subsequent accesses.
        This improves startup time for large checkpoints where not all stage data is needed.

        Args:
            stage_name: Stage name (e.g., "MATCH", "VIDEO_SEARCH")

        Returns:
            Dict containing the stage's data, or empty dict if not found/not loaded
        """
        stage_key = stage_name.lower()
        field_name = STAGE_FIELD_MAP.get(stage_name, stage_key)

        # Check if already cached
        if field_name in self._stage_data_cache:
            self._lazy_load_stats["stages_cached"] += 1
            return self._stage_data_cache[field_name]

        # Need to load raw checkpoint data first
        if self._raw_checkpoint_data is None:
            self._load_raw_checkpoint_data()

        if self._raw_checkpoint_data is None:
            return {}

        # Extract just the requested stage data
        stage_data = self._raw_checkpoint_data.get(field_name, {})

        # Validate it's a dict
        if not isinstance(stage_data, dict):
            logger.warning(
                f"Checkpoint stage data for '{field_name}' is {type(stage_data).__name__}, "
                "expected dict — returning empty dict"
            )
            stage_data = {}

        # Cache it
        self._stage_data_cache[field_name] = stage_data
        self._lazy_load_stats["stages_loaded"].append(field_name)
        self._lazy_load_stats["total_load_calls"] += 1

        return stage_data

    def load_all_stage_data(self) -> Optional[CheckpointData]:
        """
        US-115-005: Load all checkpoint data (full load, existing behavior).

        Loads all stage data into a CheckpointData object. This is the existing
        behavior for backward compatibility.

        Returns:
            CheckpointData object if loaded successfully, None otherwise
        """
        if self._full_checkpoint_loaded and self.data is not None:
            return self.data

        # Use the existing load method
        return self.load()

    def _load_raw_checkpoint_data(self) -> None:
        """
        US-115-005: Load raw checkpoint data as dict without converting to CheckpointData.

        This is called internally when lazy loading is needed but raw data isn't loaded yet.
        """
        if not self.exists():
            return

        # Try main checkpoint first
        self._raw_checkpoint_data = self._try_load_checkpoint_data(self.checkpoint_path)

        # Try backup if main doesn't exist or is invalid
        if self._raw_checkpoint_data is None and self.backup_path.exists():
            self._raw_checkpoint_data = self._try_load_checkpoint_data(self.backup_path)

        # Try rotated backups if still None
        if self._raw_checkpoint_data is None:
            for i in range(1, self._backup_count):
                rotated_path = self._get_backup_path(i)
                if rotated_path.exists():
                    self._raw_checkpoint_data = self._try_load_checkpoint_data(rotated_path)
                    if self._raw_checkpoint_data is not None:
                        break

    def get_lazy_load_stats(self) -> Dict[str, Any]:
        """
        US-115-005: Get lazy loading statistics.

        Returns:
            Dict with:
                - stages_loaded: List of stage field names that have been loaded
                - stages_cached: Number of cache hits
                - memory_saved_bytes: Estimated bytes saved by not loading full checkpoint
                - total_load_calls: Total number of load_stage_data calls
        """
        # Estimate memory saved
        if self._raw_checkpoint_data:
            full_size = len(json.dumps(self._raw_checkpoint_data))
            loaded_size = sum(
                len(json.dumps(self._stage_data_cache.get(f, {})))
                for f in self._lazy_load_stats["stages_loaded"]
            )
            self._lazy_load_stats["memory_saved_bytes"] = max(0, full_size - loaded_size)
        else:
            self._lazy_load_stats["memory_saved_bytes"] = 0

        return {
            "stages_loaded": list(self._lazy_load_stats["stages_loaded"]),
            "stages_cached": self._lazy_load_stats["stages_cached"],
            "memory_saved_bytes": self._lazy_load_stats["memory_saved_bytes"],
            "total_load_calls": self._lazy_load_stats["total_load_calls"],
        }

    # US-115-007: Checkpoint history tracking methods

    def _record_history_entry(self, stage: str, is_intermediate: bool = False) -> None:
        """
        Record a checkpoint history entry with metadata indexing.

        Args:
            stage: The stage that was just completed
            is_intermediate: Whether this is an intermediate save (not recorded in history)
        """
        if not self.data:
            return

        # Only record significant (non-intermediate) saves in history
        if is_intermediate:
            return

        # Extract metadata for indexing
        video_count = 0
        match_count = 0

        # Count videos from various stage data
        if hasattr(self.data, 'video_search') and self.data.video_search:
            vs_data = self.data.video_search
            if isinstance(vs_data, dict):
                videos = vs_data.get('videos', [])
                if isinstance(videos, list):
                    video_count = len(videos)

        # Count matches from match stage data
        if hasattr(self.data, 'match') and self.data.match:
            match_data = self.data.match
            if isinstance(match_data, dict):
                segments = match_data.get('segments', [])
                if isinstance(segments, list):
                    match_count = len(segments)

        # Get file size
        size_bytes = 0
        if self.checkpoint_path.exists():
            try:
                size_bytes = self.checkpoint_path.stat().st_size
            except OSError:
                pass

        # Create history entry
        entry = {
            "timestamp": datetime.now().isoformat(),
            "stage": stage,
            "video_count": video_count,
            "match_count": match_count,
            "size_bytes": size_bytes,
        }

        # Load existing history and append
        self._load_history()

        # Keep max 100 entries to prevent unbounded growth
        if len(self._history) >= 100:
            self._history = self._history[-99:]

        self._history.append(entry)

        # Save history to file
        self._save_history()

    def _load_history(self) -> None:
        """Load checkpoint history from file."""
        if self._history_loaded:
            return

        self._history = []
        if not self.history_path.exists():
            self._history_loaded = True
            return

        try:
            # Use non-locking read for compatibility with Ralph
            flags = os.O_RDONLY
            if hasattr(os, 'O_BINARY'):
                flags |= os.O_BINARY
            fd = os.open(str(self.history_path), flags)
            try:
                content = os.read(fd, 1024 * 1024)  # 1MB max
                if content:
                    self._history = json.loads(content.decode('utf-8'))
                    if not isinstance(self._history, list):
                        self._history = []
            finally:
                os.close(fd)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Could not load checkpoint history: {e}")
            self._history = []

        self._history_loaded = True

    def _save_history(self) -> None:
        """Save checkpoint history to file."""
        try:
            with open(self.history_path, 'w', encoding='utf-8') as f:
                json.dump(self._history, f, indent=2)
        except OSError as e:
            logger.warning(f"Could not save checkpoint history: {e}")

    def get_checkpoint_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get checkpoint history entries.

        Args:
            limit: Maximum number of entries to return (default 10)

        Returns:
            List of history entries, each containing:
                - timestamp: ISO timestamp
                - stage: Stage that was completed
                - video_count: Number of videos at that point
                - match_count: Number of matches at that point
                - size_bytes: Checkpoint file size
        """
        self._load_history()

        # Return most recent entries
        return self._history[-limit:] if self._history else []

    def _try_load_checkpoint_data(self, path: Path) -> Optional[Dict[str, Any]]:
        """US-115-003: Try to load checkpoint as dict (for incremental save merge).

        Handles both compressed and uncompressed checkpoint files.
        Returns the raw dict without converting to CheckpointData.
        """
        try:
            # Check for gzip magic number (0x1f 0x8b)
            with open(path, 'rb') as f:
                header = f.read(2)

            if header == b'\x1f\x8b':
                # File is gzip compressed - decompress first
                with gzip.open(path, 'rb') as f:
                    content = f.read().decode('utf-8')
            else:
                # File is plain JSON
                with open(path, 'r', encoding='utf-8') as f:
                    content = f.read()

            # Check for empty or obviously corrupt content
            if not content or len(content) < 10:
                logger.warning(f"Checkpoint file is empty or too small: {path}")
                return None

            data = json.loads(content)

            # Basic structure check
            if not isinstance(data, dict):
                logger.warning(f"Checkpoint is not a dict: {path}")
                return None

            return data

        except Exception as e:
            logger.debug(f"Failed to load checkpoint data from {path}: {e}")
            return None

    def _try_load_file(self, path: Path) -> Optional[CheckpointData]:
        """Try to load and parse a checkpoint file"""
        # Determine source for corruption tracking
        source = "unknown"
        if path.name == self.CHECKPOINT_FILE:
            source = "primary"
        elif path.name == self.CHECKPOINT_BACKUP:
            source = "backup"
        elif "backup" in path.name:
            source = "rotated_backup"

        try:
            # US-115-003: Automatic decompression based on file header detection
            # Check for gzip magic number (0x1f 0x8b)
            with open(path, 'rb') as f:
                header = f.read(2)

            if header == b'\x1f\x8b':
                # File is gzip compressed - decompress first
                with gzip.open(path, 'rb') as f:
                    content = f.read().decode('utf-8')
            else:
                # File is plain JSON
                with open(path, 'r', encoding='utf-8') as f:
                    content = f.read()

            # Check for empty or obviously corrupt content
            if not content or len(content) < 10:
                # US-164-007: Log checkpoint corruption errors with PIPE-xxx codes
                log_error_with_context(
                    logger,
                    "PIPE-003",
                    f"Checkpoint file is empty or too small: {path}",
                    checkpoint_path=str(path),
                    source=source,
                )
                # US-138-005: Record corruption event
                self._record_corruption_event(
                    corruption_type="empty_or_truncated",
                    source=source,
                    repaired=False,
                )
                # US-115-006: Try to repair empty/minimal content
                if self._auto_repair_enabled:
                    repaired = self._auto_repair(content, path)
                    if repaired:
                        data = repaired
                        self._record_corruption_event(
                            corruption_type="empty_or_truncated",
                            source=source,
                            repaired=True,
                        )
                        return self._migrate_checkpoint_if_needed(data)
                return None

            try:
                data = json.loads(content)
            except json.JSONDecodeError as e:
                # US-164-007: Log checkpoint corruption errors with PIPE-xxx codes
                log_error_with_context(
                    logger,
                    "PIPE-004",
                    f"Checkpoint JSON parse error in {path}: {e}",
                    checkpoint_path=str(path),
                    source=source,
                )
                # US-138-005: Record corruption event
                self._record_corruption_event(
                    corruption_type="json_error",
                    source=source,
                    repaired=False,
                )
                # US-115-006: Try to auto-repair corrupted JSON
                if self._auto_repair_enabled:
                    repaired = self._auto_repair(content, path)
                    if repaired:
                        data = repaired
                        self._record_corruption_event(
                            corruption_type="json_error",
                            source=source,
                            repaired=True,
                        )
                        return self._migrate_checkpoint_if_needed(data)
                return None

            # Basic structure check
            if not isinstance(data, dict):
                logger.warning(f"Checkpoint is not a dict: {path}")
                # US-115-006: Try to repair type mismatch
                if self._auto_repair_enabled and isinstance(data, list):
                    # Maybe it's a list with one dict inside
                    if len(data) > 0 and isinstance(data[0], dict):
                        data = data[0]
                    else:
                        repaired = self._auto_repair(content, path)
                        if repaired:
                            data = repaired
                        else:
                            return None
                else:
                    return None

            # US-40-010: Validate checkpoint integrity - check required top-level keys
            # Required keys: last_completed_stage, timestamp (created_at or updated_at)
            # The "state" is distributed across stage fields (analyze, video_search, etc.)
            missing_keys = self._validate_checkpoint_structure(data)
            if missing_keys:
                # US-164-007: Log checkpoint corruption errors with PIPE-xxx codes
                log_error_with_context(
                    logger,
                    "PIPE-005",
                    f"Checkpoint missing required keys: {missing_keys}",
                    checkpoint_path=str(path),
                    source=source,
                    missing_keys=missing_keys,
                )
                # US-138-005: Record corruption event for missing fields
                self._record_corruption_event(
                    corruption_type="missing_field",
                    source=source,
                    repaired=False,
                    data_lost=True,  # Missing fields means data loss
                )
                for key in missing_keys:
                    logger.warning(f"Checkpoint missing required key: {key}")

            # US-130-005: Restore deduplicated data from references
            dedup_store = data.pop("_dedup_store", None)
            if dedup_store:
                data = self._restore_deduplicated_data(data, dedup_store)
                logger.debug(f"Restored deduplicated checkpoint: {len(dedup_store)} unique blocks")

            # US-130-006: Restore differential checkpoint data
            # Check if this is a differential checkpoint (stores only changes)
            if data.get("_diff_marker"):
                base_timestamp = data.pop("_diff_base_timestamp", "")
                diff_fields = data.pop("_diff_fields", {})
                diff_sequence = data.pop("_diff_sequence", 0)

                # Need to load the base checkpoint to apply diff
                # Try to load from backup or find the base checkpoint
                base_data = None
                base_path = self.backup_path if self.backup_path.exists() else self.checkpoint_path

                # Try loading base checkpoint
                if base_path.exists():
                    base_data = self._try_load_checkpoint_data(base_path)
                    # Handle nested diffs - recursively apply if base is also a diff
                    if base_data and base_data.get("_diff_marker"):
                        # Base is also a diff, need to find the root full checkpoint
                        # For simplicity, we'll treat it as a full checkpoint for now
                        # In production, you'd need to chain through the diff history
                        pass

                if base_data:
                    # Apply diff to reconstruct full data
                    # Add diff fields to base data
                    for key, value in diff_fields.items():
                        base_data[key] = value
                    # Copy over critical metadata
                    base_data["updated_at"] = data.get("updated_at", base_data.get("updated_at", ""))
                    base_data["last_completed_stage"] = data.get("last_completed_stage", base_data.get("last_completed_stage", ""))
                    data = base_data
                    logger.debug(f"Restored differential checkpoint (sequence={diff_sequence}, base={base_timestamp})")
                else:
                    # Could not load base, treat as full checkpoint (drop diff fields)
                    diff_fields = data.pop("_diff_fields", {})
                    logger.warning("Could not load base checkpoint for differential restore, data may be incomplete")

            # Migrate checkpoint if needed
            return self._migrate_checkpoint_if_needed(data)

        except json.JSONDecodeError as e:
            logger.warning(f"Checkpoint JSON parse error in {path}: {e}")
            # US-115-006: Try to auto-repair
            if self._auto_repair_enabled:
                # Re-read content for repair attempt
                try:
                    with open(path, 'rb') as f:
                        header = f.read(2)
                    if header == b'\x1f\x8b':
                        with gzip.open(path, 'rb') as f:
                            content = f.read().decode('utf-8')
                    else:
                        with open(path, 'r', encoding='utf-8') as f:
                            content = f.read()
                    repaired = self._auto_repair(content, path)
                    if repaired:
                        return self._migrate_checkpoint_if_needed(repaired)
                except Exception:
                    pass
            return None
        except Exception as e:
            logger.warning(f"Failed to load checkpoint from {path}: {e}")
            return None

    def _load_mmap(self, path: Path) -> Optional[CheckpointData]:
        """
        US-130-004: Load checkpoint using memory-mapped file I/O.

        Memory-mapped loading can improve performance for large checkpoint files (>10MB)
        by reducing memory copy overhead and allowing the OS to manage page caching.
        Falls back to standard loading if mmap fails.

        Args:
            path: Path to the checkpoint file

        Returns:
            CheckpointData if loaded successfully, None otherwise
        """
        try:
            file_size = path.stat().st_size

            # Check file size against threshold
            if file_size < self._mmap_threshold_bytes:
                logger.debug(f"Checkpoint file {path.name} ({file_size} bytes) below mmap threshold, using standard load")
                return None

            # Open file for mmap
            with open(path, 'rb') as f:
                # Check for gzip compression
                header = f.read(2)
                f.seek(0)  # Reset to beginning

                if header == b'\x1f\x8b':
                    # Compressed file - decompress first, then parse
                    # mmap doesn't work well with gzip, fall back to standard
                    logger.debug(f"Checkpoint file is gzip compressed, using standard load")
                    return None

                # Memory-map the file (read-only)
                try:
                    with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                        content = mm[:]  # Read all content into memory
                        content_str = content.decode('utf-8')
                except (ValueError, OSError) as e:
                    # mmap failed (e.g., empty file, permission issues)
                    logger.debug(f"mmap failed for {path.name}: {e}, falling back to standard load")
                    return None

            # Check for empty content
            if not content_str or len(content_str) < 10:
                logger.warning(f"Checkpoint file is empty or too small: {path}")
                self._mmap_stats["mmap_fallbacks"] += 1
                return None

            # Parse JSON
            try:
                data = json.loads(content_str)
            except json.JSONDecodeError as e:
                # US-164-007: Log checkpoint corruption errors with PIPE-xxx codes
                log_error_with_context(
                    logger,
                    "PIPE-004",
                    f"Checkpoint JSON parse error in {path}: {e}",
                    checkpoint_path=str(path),
                    source="mmap",
                )
                self._mmap_stats["mmap_fallbacks"] += 1
                return None

            # Validate structure
            if not isinstance(data, dict):
                # US-164-007: Log checkpoint corruption errors with PIPE-xxx codes
                log_error_with_context(
                    logger,
                    "PIPE-006",
                    f"Checkpoint is not a dict: {path}",
                    checkpoint_path=str(path),
                    source="mmap",
                )
                self._mmap_stats["mmap_fallbacks"] += 1
                return None

            # Validate required keys
            missing_keys = self._validate_checkpoint_structure(data)
            if missing_keys:
                # US-164-007: Log checkpoint corruption errors with PIPE-xxx codes
                log_error_with_context(
                    logger,
                    "PIPE-005",
                    f"Checkpoint missing required keys: {missing_keys}",
                    checkpoint_path=str(path),
                    source="mmap",
                    missing_keys=missing_keys,
                )
                for key in missing_keys:
                    logger.warning(f"Checkpoint missing required key: {key}")

            # Track stats
            self._mmap_stats["mmap_loads"] += 1
            self._mmap_stats["total_mmap_bytes"] += file_size
            logger.info(f"Loaded checkpoint via mmap: {path.name} ({file_size} bytes)")

            # Migrate checkpoint if needed
            return self._migrate_checkpoint_if_needed(data)

        except Exception as e:
            logger.warning(f"Failed to load checkpoint via mmap from {path}: {e}")
            self._mmap_stats["mmap_fallbacks"] += 1
            return None

    def _get_checkpoint_timestamp(self, data: CheckpointData) -> Optional[datetime]:
        """Extract the most recent timestamp from checkpoint data."""
        timestamp_str = data.updated_at or data.created_at
        if not timestamp_str:
            return None
        try:
            return datetime.fromisoformat(timestamp_str)
        except (ValueError, TypeError):
            return None

    def _select_checkpoint(
        self, main_data: Optional[CheckpointData], backup_data: Optional[CheckpointData]
    ) -> Optional[CheckpointData]:
        """
        Select the best checkpoint based on validity and timestamps (US-44-011).

        Logic:
        - If only one is valid, use it (with appropriate logging).
        - If both are valid, compare timestamps and use the newer one.
        - If backup is newer than main, warn about possible mid-save crash.
        """
        if main_data is not None and backup_data is None:
            # Main valid, no backup — use main
            return main_data

        if main_data is None and backup_data is None:
            # Both invalid — try cloud backup restore (US-130-010)
            if self._cloud_backup_manager:
                logger.warning("Main and local backup both corrupt/missing, attempting cloud restore...")
                cloud_report = self._cloud_backup_manager.download_latest(self.checkpoint_path)
                if cloud_report.get("success"):
                    # Reload from the restored checkpoint
                    cloud_data = self._try_load_file(self.checkpoint_path)
                    if cloud_data is not None:
                        logger.info(f"Restored checkpoint from cloud: {cloud_report.get('remote_key')}")
                        return cloud_data
                    else:
                        logger.error("Cloud restore downloaded file is corrupt")
                else:
                    logger.warning(f"Cloud restore failed: {cloud_report.get('errors', ['unknown error'])}")
            return None

        if main_data is None and backup_data is not None:
            # Main corrupt, backup valid — restore from backup
            main_ts = "corrupt"
            backup_ts = self._get_checkpoint_timestamp(backup_data)
            backup_ts_str = backup_ts.isoformat() if backup_ts else "unknown"
            logger.warning(
                f"Main checkpoint corrupted, restoring from backup "
                f"(main: {main_ts}, backup: {backup_ts_str})"
            )
            self._restore_backup_to_main()
            return backup_data

        # Both are valid — compare timestamps to pick the newer one
        main_ts = self._get_checkpoint_timestamp(main_data)
        backup_ts = self._get_checkpoint_timestamp(backup_data)

        main_ts_str = main_ts.isoformat() if main_ts else "unknown"
        backup_ts_str = backup_ts.isoformat() if backup_ts else "unknown"

        if backup_ts and main_ts and backup_ts > main_ts:
            # Backup is newer — indicates a mid-save crash where main was
            # written but backup wasn't yet overwritten, or main got corrupted
            # after the backup was made
            logger.warning(
                f"Backup checkpoint is newer than main (main: {main_ts_str}, "
                f"backup: {backup_ts_str}) — possible mid-save crash. Using backup."
            )
            self._restore_backup_to_main()
            return backup_data

        # Main is valid and newer (or same age) — use main
        logger.debug(
            f"Using main checkpoint (main: {main_ts_str}, backup: {backup_ts_str})"
        )
        return main_data

    def _restore_backup_to_main(self):
        """Copy backup checkpoint over main checkpoint."""
        try:
            shutil.copy2(self.backup_path, self.checkpoint_path)
            logger.info("Restored checkpoint from backup")
        except Exception as e:
            logger.warning(f"Could not restore backup: {e}")

    def _validate_checkpoint_structure(self, data: dict) -> List[str]:
        """
        Validate checkpoint has required top-level structure (US-40-010).

        Checks for:
        - last_completed_stage: To know where to resume from
        - timestamp: created_at or updated_at for staleness checks
        - version: For migration compatibility

        Returns list of missing keys (empty if all present).
        Does NOT fail on missing keys - just logs warnings and uses defaults.
        """
        missing = []

        # Check for last_completed_stage (required to know resume point)
        if 'last_completed_stage' not in data:
            missing.append('last_completed_stage')
            data['last_completed_stage'] = ''  # Initialize to default

        # Check for timestamp (created_at or updated_at)
        has_timestamp = data.get('created_at') or data.get('updated_at')
        if not has_timestamp:
            missing.append('timestamp (created_at/updated_at)')
            # Initialize with current time
            data['created_at'] = datetime.now().isoformat()
            data['updated_at'] = datetime.now().isoformat()

        # Check for version (needed for migration decisions)
        if 'version' not in data:
            missing.append('version')
            data['version'] = '0.9'  # Assume oldest version for migration

        return missing

    def _auto_repair(self, content: str, file_path: Path) -> Optional[Dict[str, Any]]:
        """
        US-115-006: Automatically repair common checkpoint corruption patterns.

        Repairs:
        - Truncated JSON (missing closing braces/brackets)
        - Trailing garbage after valid JSON
        - Missing required fields (with defaults)
        - Type mismatches (e.g., list where dict expected)
        - Invalid stage data structures

        Args:
            content: Raw checkpoint file content
            file_path: Path to checkpoint file (for logging)

        Returns:
            Repaired dict if successful, None if repair failed
        """
        if not self._auto_repair_enabled:
            return None

        # Track repair attempts
        self._repair_stats["total_repair_attempts"] += 1

        repairs_made = []
        repaired_data = None

        for attempt in range(self._auto_repair_max_attempts):
            try:
                # Attempt 1: Try to fix truncated JSON
                if attempt == 0:
                    repaired_data = self._repair_truncated_json(content)
                    if repaired_data:
                        repairs_made.append("truncated_json")

                # Attempt 2: If truncation didn't work, try parsing raw content and fix missing fields
                if attempt == 1 and not repaired_data:
                    try:
                        # Maybe it's partially valid but just missing fields
                        parsed = json.loads(content)
                        if isinstance(parsed, dict):
                            repaired_data = parsed
                            repairs_made.append("json_valid")
                    except json.JSONDecodeError:
                        pass

                # Attempt 2: Try to fix missing required fields (always try this if we have data)
                if (attempt == 1 or attempt == 0) and repaired_data:
                    fixed = self._repair_missing_fields(repaired_data)
                    if fixed:
                        repaired_data = fixed
                        repairs_made.append("missing_fields")

                # Attempt 3: Try to fix type mismatches
                if attempt == 2 and repaired_data:
                    fixed = self._repair_type_mismatches(repaired_data)
                    if fixed:
                        repaired_data = fixed
                        repairs_made.append("type_mismatches")

                # If we have valid data after repairs, validate it
                if repaired_data and self._validate_for_migration(repaired_data):
                    break

            except Exception as e:
                if self._auto_repair_log:
                    logger.warning(f"Repair attempt {attempt + 1} failed: {e}")
                repaired_data = None

        # Log repair actions
        if repairs_made:
            self._repair_stats["successful_repairs"] += 1
            repair_record = {
                "file": str(file_path),
                "repairs_made": repairs_made,
                "timestamp": datetime.now().isoformat(),
            }
            self._repair_stats["repair_history"].append(repair_record)

            if self._auto_repair_log:
                logger.info(f"Checkpoint auto-repair successful for {file_path.name}: {repairs_made}")
        else:
            self._repair_stats["failed_repairs"] += 1
            if self._auto_repair_log:
                logger.warning(f"Checkpoint auto-repair failed for {file_path.name}")

        return repaired_data

    def _repair_truncated_json(self, content: str) -> Optional[Dict[str, Any]]:
        """Attempt to fix truncated JSON by finding valid prefix."""
        import re

        # Remove any leading whitespace/bom
        content = content.lstrip()

        # First, try direct parsing (maybe it's already valid)
        try:
            parsed = json.loads(content)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

        # Try to find a valid JSON object by balancing braces
        # Find the last complete object/array
        depth = 0
        in_string = False
        escape_next = False
        i = 0

        while i < len(content):
            char = content[i]

            if escape_next:
                escape_next = False
                i += 1
                continue

            if char == '\\' and in_string:
                escape_next = True
                i += 1
                continue

            if char == '"' and not escape_next:
                in_string = not in_string
                i += 1
                continue

            if in_string:
                i += 1
                continue

            if char in '{[':
                depth += 1
            elif char in '}]':
                depth -= 1

            # When we find balanced closing, try parsing
            if depth == 0 and char in '}]':
                # Try parsing up to this point
                potential = content[:i + 1]
                try:
                    parsed = json.loads(potential)
                    if isinstance(parsed, dict):
                        return parsed
                except json.JSONDecodeError:
                    pass

            i += 1

        # If that didn't work, try a simpler approach: find last }
        # and try to parse everything before it
        last_brace = max(content.rfind('}'), content.rfind(']'))
        if last_brace > 0:
            try:
                parsed = json.loads(content[:last_brace + 1])
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                pass

        # Try to add closing brace and see if that helps
        # For simple cases like {"last_completed_stage": "MATCH"
        if content.startswith('{'):
            try:
                parsed = json.loads(content + '}')
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                pass

        return None

    def _repair_missing_fields(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Repair missing required fields with defaults."""
        repairs = []

        # Check for last_completed_stage
        if 'last_completed_stage' not in data:
            data['last_completed_stage'] = ''
            repairs.append("last_completed_stage")

        # Check for timestamp
        if not data.get('created_at') and not data.get('updated_at'):
            now = datetime.now().isoformat()
            data['created_at'] = now
            data['updated_at'] = now
            repairs.append("timestamp")

        # Check for version
        if 'version' not in data:
            data['version'] = '0.9'
            repairs.append("version")

        # Ensure all stage fields exist (as dicts)
        required_stages = ['analyze', 'video_search', 'caption', 'match',
                          'iterative_match', 'download_segments', 'output']
        for stage in required_stages:
            if stage not in data:
                data[stage] = {}
                repairs.append(f"stage:{stage}")
            elif data[stage] is None:
                data[stage] = {}
                repairs.append(f"stage:{stage}")

        if repairs and self._auto_repair_log:
            logger.debug(f"Repaired missing fields: {repairs}")

        return data

    def _repair_type_mismatches(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Repair type mismatches in checkpoint data."""
        repairs = []

        # last_completed_stage should be string
        if 'last_completed_stage' in data and not isinstance(data['last_completed_stage'], str):
            data['last_completed_stage'] = str(data['last_completed_stage'])
            repairs.append("last_completed_stage_type")

        # version should be string
        if 'version' in data and not isinstance(data['version'], str):
            data['version'] = str(data['version'])
            repairs.append("version_type")

        # timestamp fields should be strings
        for ts_field in ['created_at', 'updated_at']:
            if ts_field in data and data[ts_field] is not None:
                if not isinstance(data[ts_field], str):
                    data[ts_field] = str(data[ts_field])
                    repairs.append(f"{ts_field}_type")

        # Stage data should be dicts, not lists or other types
        stage_fields = ['analyze', 'video_search', 'caption', 'match',
                       'iterative_match', 'download_segments', 'output']
        for stage in stage_fields:
            if stage in data and data[stage] is not None:
                if not isinstance(data[stage], dict):
                    # Try to convert to dict
                    if isinstance(data[stage], str):
                        try:
                            data[stage] = json.loads(data[stage])
                            repairs.append(f"{stage}_type")
                        except json.JSONDecodeError:
                            data[stage] = {}
                            repairs.append(f"{stage}_type")
                    else:
                        data[stage] = {}
                        repairs.append(f"{stage}_type")

        if repairs and self._auto_repair_log:
            logger.debug(f"Repaired type mismatches: {repairs}")

        return data

    def _validate_for_migration(self, data: Dict[str, Any]) -> bool:
        """Check if data is valid enough to attempt migration."""
        if not isinstance(data, dict):
            return False
        if not data:
            return False
        # Basic sanity check
        return True

    def get_repair_success_rate(self) -> float:
        """
        US-115-006: Get repair success rate for health diagnostics.

        Returns:
            Float between 0.0 and 1.0 representing repair success rate
        """
        total = self._repair_stats["total_repair_attempts"]
        if total == 0:
            return 1.0  # No repairs needed = 100% success
        return self._repair_stats["successful_repairs"] / total

    def get_repair_stats(self) -> Dict[str, Any]:
        """Get repair statistics."""
        return {
            **self._repair_stats,
            "success_rate": self.get_repair_success_rate(),
        }

    def get_corruption_stats(self) -> Dict[str, Any]:
        """
        US-138-005: Get corruption detection and recovery statistics.

        Returns:
            Dict containing:
            - total_corruptions_detected: Total corruption events detected
            - corruption_types: Breakdown by type (json_error, missing_field, etc.)
            - corruption_sources: Breakdown by source (primary, backup, rotated)
            - auto_repairs_attempted: Number of auto-repair attempts
            - auto_repairs_successful: Number of successful repairs
            - graceful_recoveries: Count of graceful recoveries
            - total_data_loss_events: Events where some data was lost
            - success_rate: Ratio of successful recoveries to total detections
        """
        total = self._corruption_stats["total_corruptions_detected"]
        return {
            **self._corruption_stats,
            "success_rate": (
                self._corruption_stats["graceful_recoveries"] / total
                if total > 0 else 1.0
            ),
        }

    def _record_corruption_event(
        self,
        corruption_type: str,
        source: str = "unknown",
        repaired: bool = False,
        data_lost: bool = False,
    ) -> None:
        """
        US-138-005: Record a corruption event for metrics tracking.

        Args:
            corruption_type: Type of corruption (json_error, missing_field, truncated, etc.)
            source: Source of corruption (primary, backup, rotated_backup)
            repaired: Whether auto-repair was attempted/successful
            data_lost: Whether any data was lost due to corruption
        """
        self._corruption_stats["total_corruptions_detected"] += 1

        # Track by type
        if corruption_type not in self._corruption_stats["corruption_types"]:
            self._corruption_stats["corruption_types"][corruption_type] = 0
        self._corruption_stats["corruption_types"][corruption_type] += 1

        # Track by source
        if source not in self._corruption_stats["corruption_sources"]:
            self._corruption_stats["corruption_sources"][source] = 0
        self._corruption_stats["corruption_sources"][source] += 1

        # Track repair outcomes
        if repaired:
            self._corruption_stats["auto_repairs_attempted"] += 1
            self._corruption_stats["auto_repairs_successful"] += 1
            self._corruption_stats["graceful_recoveries"] += 1

        if data_lost:
            self._corruption_stats["total_data_loss_events"] += 1

        # Keep history (last 10 events)
        event = {
            "timestamp": datetime.now().isoformat(),
            "type": corruption_type,
            "source": source,
            "repaired": repaired,
            "data_lost": data_lost,
        }
        self._corruption_stats["corruption_history"].append(event)
        if len(self._corruption_stats["corruption_history"]) > 10:
            self._corruption_stats["corruption_history"] = (
                self._corruption_stats["corruption_history"][-10:]
            )

    def get_degradation_warnings(self) -> List[str]:
        """US-138-005: Get warnings from last load attempt during graceful degradation."""
        return self._degradation_warnings.copy()

    def _validate_stage_metrics_completeness(self, stage: str, metrics: Dict[str, Any]) -> None:
        """
        Validate that stage metrics have all required fields for the stage type.

        US-106-005: Logs warnings for incomplete metrics instead of failing the stage.

        Args:
            stage: Stage name (e.g., 'DOWNLOAD_SEGMENTS')
            metrics: Serialized StageMetrics dict
        """
        # Import here to avoid circular imports at module load time
        from .stages import StageMetrics, StageType

        # Map stage name to StageType using existing mapping
        stage_type_str = _STAGE_NAME_TO_TYPE.get(stage)
        if not stage_type_str:
            logger.debug(f"No stage type mapping for '{stage}', skipping metrics validation")
            return

        try:
            stage_type = StageType(stage_type_str)
        except ValueError:
            logger.warning(f"Unknown stage type '{stage_type_str}' for '{stage}'")
            return

        # Reconstruct StageMetrics from dict to validate
        try:
            stage_metrics = StageMetrics.from_dict(metrics)
        except Exception as e:
            logger.warning(f"Failed to reconstruct StageMetrics for '{stage}': {e}")
            return

        # Validate and log warnings for missing required fields
        warnings = stage_metrics.validate_metrics(stage_type)
        for warning in warnings:
            logger.warning(f"[{stage}] {warning}")

    def _migrate_checkpoint_if_needed(self, data: dict) -> CheckpointData:
        """
        Migrate old checkpoint format to new format.

        Delegates to CheckpointMigrator for versioned upgrades:
        - Version 0.9 -> 1.0: Uppercase stage keys to lowercase
        - Version 1.0 -> 2.0: 13-stage to 7-stage simplified pipeline

        AC5: Creates backup before modifying checkpoint during migration.
        """
        from .checkpoint_migrator import CheckpointMigrator

        migrator = CheckpointMigrator()
        version = data.get('version', '0.9')

        if migrator.needs_migration(data, CURRENT_CHECKPOINT_VERSION):
            logger.warning(f"Checkpoint version mismatch: migrating from v{version} to v{CURRENT_CHECKPOINT_VERSION}")

            # AC5: Create backup before modifying checkpoint during migration
            self._backup_before_modification("migration")

            migrated_data = migrator.migrate(data, version, CURRENT_CHECKPOINT_VERSION)

            # Convert to CheckpointData
            migrated = CheckpointData.from_dict(migrated_data)

            # Save migrated checkpoint
            try:
                self.data = migrated
                self._atomic_save(force_rotate=True)
                logger.info(f"Migrated checkpoint to v{CURRENT_CHECKPOINT_VERSION} saved successfully")
            except Exception as e:
                logger.warning(f"Could not save migrated checkpoint: {e}")

            return migrated

        # Already current version - just convert to CheckpointData
        return CheckpointData.from_dict(data)

    def _backup_before_modification(self, reason: str = "validation"):
        """
        AC5: Create backup before modifying checkpoint during validation/migration.

        This ensures we have a safe copy to restore if the validation or migration
        process corrupts the checkpoint.

        Args:
            reason: Reason for backup (e.g., 'migration', 'validation', 'healing')
        """
        try:
            if self.checkpoint_path.exists():
                # Create a pre-modification backup with timestamp
                import time
                timestamp = int(time.time())
                pre_mod_path = self.project_dir / f"checkpoint.pre_{reason}_{timestamp}.json"

                # Copy current checkpoint to pre-modification backup
                shutil.copy2(self.checkpoint_path, pre_mod_path)
                logger.info(f"Created pre-{reason} backup: {pre_mod_path.name}")

                # Clean up old pre-modification backups (keep last 3)
                self._cleanup_old_backups(prefix=f"checkpoint.pre_{reason}_")
        except Exception as e:
            logger.warning(f"Could not create pre-{reason} backup: {e}")

    def _cleanup_old_backups(self, prefix: str, keep: int = 3):
        """
        Clean up old pre-modification backup files, keeping only the most recent 'keep' files.

        Args:
            prefix: Prefix of backup files to clean up
            keep: Number of recent backups to keep
        """
        try:
            existing_backups = sorted(
                self.project_dir.glob(f"{prefix}*.json"),
                key=lambda p: p.stat().st_mtime,
                reverse=True
            )

            # Remove older backups beyond 'keep' count
            for old_backup in existing_backups[keep:]:
                old_backup.unlink()
                logger.debug(f"Removed old backup: {old_backup.name}")
        except Exception as e:
            logger.warning(f"Could not clean up old backups: {e}")

    def _validate_checkpoint_data(self, data: CheckpointData) -> bool:
        """
        Validate that checkpoint data has required fields.

        US-88-010: Enhanced with deep integrity validation, schema validation,
        data type drift detection, version compatibility, and healing.

        AC5: Creates backup before healing/validation that modifies the checkpoint.

        Returns True if valid, False if there are issues (but data is usable).
        """
        issues = []
        warnings = []

        # AC5: Create backup before healing if we'll be modifying the checkpoint
        healing_actions = self._heal_common_corruption_patterns(data, dry_run=True)
        if healing_actions:
            self._backup_before_modification("validation")

        # US-88-010: Attempt healing for common corruption patterns FIRST
        # This fixes common issues before validation so healed data can pass
        healing_actions = self._heal_common_corruption_patterns(data)
        if healing_actions:
            warnings.append(f"Healing actions applied: {', '.join(healing_actions)}")

        # Check version compatibility
        if data.version and data.version != CURRENT_CHECKPOINT_VERSION:
            issues.append(
                f"Checkpoint version {data.version} differs from current "
                f"{CURRENT_CHECKPOINT_VERSION} — consider re-running with --fresh"
            )

        # Check for required fields
        if not data.created_at:
            issues.append("Missing created_at timestamp")

        if not data.last_completed_stage:
            issues.append("No completed stages recorded")

        # Validate stage is in known list
        if data.last_completed_stage and data.last_completed_stage not in STAGE_ORDER:
            issues.append(f"Unknown stage: {data.last_completed_stage}")

        # US-88-010: Deep integrity validation - validate stage data schemas
        stage_data_issues = self._validate_stage_data_schemas(data)
        issues.extend(stage_data_issues)

        # US-88-010: Check for data type drift in checkpoint fields
        type_drift_issues = self._check_data_type_drift(data)
        warnings.extend(type_drift_issues)

        # US-88-010: Validate checkpoint version compatibility
        version_issues = self._validate_version_compatibility(data)
        issues.extend(version_issues)

        # Log detailed validation report
        self._log_validation_report(issues, warnings)

        if issues:
            for issue in issues:
                logger.debug(f"Checkpoint validation: {issue}")
            return False

        return True

    def _validate_stage_data_schemas(self, data: CheckpointData) -> List[str]:
        """
        Validate that stage data fields match expected schema (US-88-010).

        Checks that each stage data field is a dict with expected structure
        based on what each stage typically produces.
        """
        issues = []
        expected_stage_fields = {
            'analyze': ['keywords', 'segments'],
            'video_search': ['video_ids', 'search_results'],
            'caption': ['captions', 'caption_count'],
            'match': ['matches', 'match_count'],
            'iterative_match': ['gap_analysis', 'additional_matches'],
            'download_segments': ['downloaded_segments', 'download_results'],
        }

        for stage_name, expected_keys in expected_stage_fields.items():
            stage_data = getattr(data, stage_name, None)
            if not stage_data:
                continue

            if not isinstance(stage_data, dict):
                issues.append(
                    f"Stage '{stage_name}' data is {type(stage_data).__name__}, expected dict"
                )
                continue

            # Check for unexpected types in stage data values
            for key, value in stage_data.items():
                if value is None:
                    issues.append(
                        f"Stage '{stage_name}' has null value for key '{key}'"
                    )
                elif isinstance(value, (str, int, float, bool)):
                    # Primitives are acceptable
                    pass
                elif isinstance(value, (list, dict)):
                    # Collections are acceptable
                    pass
                else:
                    issues.append(
                        f"Stage '{stage_name}' has unexpected type {type(value).__name__} "
                        f"for key '{key}'"
                    )

        return issues

    def _check_data_type_drift(self, data: CheckpointData) -> List[str]:
        """
        Check for data type drift in checkpoint fields (US-88-010).

        Detects when field types have changed unexpectedly (e.g., string
        where dict was expected, list where string was expected).
        """
        warnings = []

        # Expected field types based on CheckpointData dataclass
        expected_types = {
            'version': str,
            'created_at': str,
            'updated_at': str,
            'last_completed_stage': str,
            'config_hash': str,
            'voiceover_path': str,
            'voiceover_hash': str,
            'analyze': dict,
            'video_search': dict,
            'caption': dict,
            'match': dict,
            'iterative_match': dict,
            'download_segments': dict,
            'chapter_data': dict,
            'stage_metrics': dict,
            'transcription_metrics': dict,
            'validation_cache': dict,
        }

        for field_name, expected_type in expected_types.items():
            actual_value = getattr(data, field_name, None)
            if actual_value is None:
                continue

            if not isinstance(actual_value, expected_type):
                warnings.append(
                    f"Data type drift in '{field_name}': expected {expected_type.__name__}, "
                    f"got {type(actual_value).__name__}"
                )

        return warnings

    def _validate_version_compatibility(self, data: CheckpointData) -> List[str]:
        """
        Validate checkpoint version compatibility with code version (US-88-010).

        Returns list of compatibility issues.
        """
        issues = []
        checkpoint_version = data.version

        # Version format: "X.Y"
        if checkpoint_version:
            try:
                parts = checkpoint_version.split('.')
                if len(parts) >= 2:
                    major, minor = int(parts[0]), int(parts[1])

                    # Current major version
                    current_parts = CURRENT_CHECKPOINT_VERSION.split('.')
                    current_major = int(current_parts[0])

                    if major < current_major:
                        issues.append(
                            f"Checkpoint version {checkpoint_version} is incompatible "
                            f"with current version {CURRENT_CHECKPOINT_VERSION} "
                            f"(major version mismatch)"
                        )
                    elif major == current_major:
                        # Same major version - minor version differences are OK
                        # but log a warning
                        if minor < int(current_parts[1]):
                            issues.append(
                                f"Checkpoint version {checkpoint_version} is older than "
                                f"current {CURRENT_CHECKPOINT_VERSION}, consider fresh run"
                            )
            except (ValueError, IndexError) as e:
                issues.append(f"Invalid checkpoint version format: {checkpoint_version}")

        return issues

    def _heal_common_corruption_patterns(self, data: CheckpointData, dry_run: bool = False) -> List[str]:
        """
        Attempt healing for common checkpoint corruption patterns (US-88-010).

        Args:
            data: CheckpointData to heal
            dry_run: If True, only return what healing would be done without modifying

        Returns list of healing actions taken (or that would be taken if dry_run=True).
        """
        healing_actions = []

        # Pattern 1: Stage data is a string instead of dict
        for stage_name in ['analyze', 'video_search', 'caption', 'match',
                          'iterative_match', 'download_segments', 'chapter_data']:
            stage_data = getattr(data, stage_name, None)
            if isinstance(stage_data, str):
                # Try to parse as JSON
                import json
                try:
                    parsed = json.loads(stage_data)
                    if isinstance(parsed, dict):
                        if not dry_run:
                            setattr(data, stage_name, parsed)
                        healing_actions.append(f"Parsed {stage_name} from JSON string")
                    else:
                        if not dry_run:
                            setattr(data, stage_name, {})
                        healing_actions.append(f"Reset {stage_name} (parsed JSON was not dict)")
                except json.JSONDecodeError:
                    if not dry_run:
                        setattr(data, stage_name, {})
                    healing_actions.append(f"Reset {stage_name} (string parse failed)")

        # Pattern 2: Stage data is a list instead of dict
        for stage_name in ['analyze', 'video_search', 'caption', 'match',
                          'iterative_match', 'download_segments', 'chapter_data']:
            stage_data = getattr(data, stage_name, None)
            if isinstance(stage_data, list):
                # Convert list to dict if possible (common case: list of items)
                if stage_data and isinstance(stage_data[0], dict):
                    # Convert to indexed dict
                    converted = {f"item_{i}": item for i, item in enumerate(stage_data)}
                    if not dry_run:
                        setattr(data, stage_name, converted)
                    healing_actions.append(f"Converted {stage_name} from list to indexed dict")
                else:
                    if not dry_run:
                        setattr(data, stage_name, {})
                    healing_actions.append(f"Reset {stage_name} (list type not convertible)")

        # Pattern 3: stage_metrics or validation_cache is not a dict
        for field_name in ['stage_metrics', 'validation_cache', 'transcription_metrics']:
            field_data = getattr(data, field_name, None)
            if field_data is not None and not isinstance(field_data, dict):
                if not dry_run:
                    setattr(data, field_name, {})
                healing_actions.append(f"Reset {field_name} (was {type(field_data).__name__})")

        # Pattern 4: Fix empty string timestamps
        for field_name in ['created_at', 'updated_at']:
            field_value = getattr(data, field_name, None)
            if field_value == '':
                from datetime import datetime
                if not dry_run:
                    setattr(data, field_name, datetime.now().isoformat())
                healing_actions.append(f"Fixed empty {field_name}")

        return healing_actions

    def _log_validation_report(self, issues: List[str], warnings: List[str]) -> None:
        """
        Log detailed validation report with warnings (US-88-010).
        """
        if issues or warnings:
            logger.info("=== Checkpoint Validation Report ===")

        if issues:
            logger.warning(f"  Issues found: {len(issues)}")
            for issue in issues:
                logger.warning(f"    - {issue}")

        if warnings:
            logger.info(f"  Warnings: {len(warnings)}")
            for warning in warnings:
                logger.info(f"    - {warning}")

        if not issues and not warnings:
            logger.debug("Checkpoint validation passed - no issues or warnings")

    def restore_state(self, state: 'PipelineState' = None) -> 'PipelineState':
        """
        Restore pipeline state from checkpoint with defensive validation.

        US-40-003: Ensures state attributes are validated after checkpoint
        restoration to handle missing fields from older checkpoints or
        corrupted checkpoint files.

        Args:
            state: Optional PipelineState to restore into. If None, creates new.

        Returns:
            PipelineState with validated attributes. All required fields are
            guaranteed to exist with proper default values.
        """
        from .state import PipelineState

        # Create or use existing state
        if state is None:
            state = PipelineState()

        # Validate and initialize any missing/None attributes
        # This handles incomplete checkpoint data from older versions
        initialized_fields = state.validate_state_attributes()

        # Log any fields that were missing and initialized
        if initialized_fields:
            for field_name in initialized_fields:
                logger.warning(
                    f"Checkpoint missing {field_name}, initialized to default"
                )
            logger.info(
                f"Defensive checkpoint validation initialized {len(initialized_fields)} "
                f"field(s): {', '.join(initialized_fields)}"
            )

        return state

    def _acquire_lock(self, timeout_seconds: Optional[float] = None) -> bool:
        """US-130-003: Acquire exclusive file lock for concurrent process safety.

        Args:
            timeout_seconds: Optional timeout for lock acquisition.
                            Defaults to self._lock_timeout_seconds.

        Returns:
            True if lock acquired successfully, False otherwise.
        """
        if self._lock_acquired:
            logger.debug("Lock already acquired")
            return True

        if timeout_seconds is None:
            timeout_seconds = self._lock_timeout_seconds

        # Create lock file path if not exists
        if self._lock_file_path is None:
            self._lock_file_path = self.project_dir / ".checkpoint.lock"

        try:
            # Open lock file for writing (create if not exists)
            lock_fd = os.open(str(self._lock_file_path), os.O_RDWR | os.O_CREAT)
            self._lock_file_handle = lock_fd

            # Try to acquire exclusive lock with timeout
            start_time = time.time()
            while True:
                try:
                    if sys.platform == 'win32':
                        # Windows: use msvcrt locking
                        msvcrt.locking(lock_fd, msvcrt.LK_NBLCK, 1)
                    else:
                        # Linux/Unix: use fcntl
                        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)

                    # Lock acquired successfully
                    self._lock_acquired = True
                    self._lock_stats["acquired_count"] = self._lock_stats.get("acquired_count", 0) + 1
                    logger.debug(f"Acquired checkpoint lock: {self._lock_file_path}")
                    return True

                except (IOError, OSError) as e:
                    # Lock not available, check timeout
                    if time.time() - start_time >= timeout_seconds:
                        # Timeout exceeded
                        logger.warning(
                            f"Failed to acquire checkpoint lock within {timeout_seconds}s: {e}"
                        )
                        self._lock_stats["failed_count"] = self._lock_stats.get("failed_count", 0) + 1
                        # Close the file handle on failure
                        try:
                            os.close(lock_fd)
                        except:
                            pass
                        self._lock_file_handle = None
                        return False

                    # Wait a bit before retrying
                    time.sleep(0.1)

        except Exception as e:
            logger.warning(f"Error acquiring checkpoint lock: {e}")
            log_error_with_context(
                logger,
                "PIPE-003",
                f"Checkpoint lock acquisition failed: {e}",
                correlation_id=get_correlation_id(),
                lock_file=str(self._lock_file_path),
            )
            self._lock_stats["failed_count"] = self._lock_stats.get("failed_count", 0) + 1
            return False

    def _release_lock(self) -> bool:
        """US-130-003: Release the file lock.

        Returns:
            True if lock released successfully, False otherwise.
        """
        if not self._lock_acquired:
            logger.debug("No lock to release")
            return True

        try:
            if self._lock_file_handle is not None:
                if sys.platform == 'win32':
                    # Windows: unlock the file
                    try:
                        msvcrt.locking(self._lock_file_handle, msvcrt.LK_UNLCK, 1)
                    except:
                        pass
                else:
                    # Linux/Unix: unlock the file
                    try:
                        fcntl.flock(self._lock_file_handle, fcntl.LOCK_UN)
                    except:
                        pass

                # Close the file handle
                try:
                    os.close(self._lock_file_handle)
                except:
                    pass

                self._lock_file_handle = None
                self._lock_acquired = False
                self._lock_stats["release_count"] = self._lock_stats.get("release_count", 0) + 1
                logger.debug(f"Released checkpoint lock: {self._lock_file_path}")
                return True

        except Exception as e:
            logger.warning(f"Error releasing checkpoint lock: {e}")
            return False

    def get_lock_stats(self) -> Dict[str, Any]:
        """US-130-003: Get file locking statistics.

        Returns:
            Dict with lock stats: acquired_count, failed_count, release_count.
        """
        return self._lock_stats.copy()

    def save(self, stage: str, stage_data: Dict[str, Any] = None,
             stage_metrics: Dict[str, Any] = None, force_full: bool = False):
        """Save checkpoint after stage completion.

        Args:
            stage: Stage name (e.g., 'DOWNLOAD_SEGMENTS')
            stage_data: Stage output data dict
            stage_metrics: Optional serialized StageMetrics dict (US-49-012).
                           Persisted under stage_metrics.<STAGE_NAME> in checkpoint.
            force_full: If True, save entire checkpoint (default False for incremental).
                        Use force_full=True for complete snapshot saves.
        """
        correlation_id = get_correlation_id()
        start_time = time.perf_counter()
        log_stage_start(
            logger,
            "CHECKPOINT_SAVE",
            correlation_id=correlation_id,
            total_items=1,
        )

        # US-130-003: Acquire file lock for concurrent process safety
        lock_acquired = self._acquire_lock()
        if not lock_acquired:
            # Graceful failure: log warning and skip save
            logger.warning(
                f"Could not acquire checkpoint lock for stage '{stage}' - "
                f"another process may be writing. Skipping save."
            )
            log_stage_skip(
                logger,
                "CHECKPOINT_SAVE",
                reason="lock acquisition failed",
                correlation_id=correlation_id,
            )
            return

        try:
            if self.data is None:
                self.data = CheckpointData(
                    created_at=datetime.now().isoformat(),
                    config_hash=self.config_hash
                )

            # US-115-002: Track dirty stages for incremental saving
            dirty_stages = self._get_dirty_stages() if not force_full else []

            self.data.updated_at = datetime.now().isoformat()
            self.data.last_completed_stage = stage

            # Store stage-specific data
            if stage_data:
                stage_key = stage.lower()
                if stage in _STAGE_FIELD_MAP_TERMINAL:
                    logger.debug(
                        f"save(): stage '{stage}' is terminal — stage_data not persisted"
                    )
                elif stage_key not in CheckpointData.__dataclass_fields__:
                    logger.warning(
                        f"save(): stage_key '{stage_key}' (from stage '{stage}') "
                        f"does not map to a CheckpointData field — data will not be persisted"
                    )
                elif hasattr(self.data, stage_key):
                    setattr(self.data, stage_key, stage_data)

            # US-71-009: Promote chapter_data from match stage to top-level field
            if stage_data and stage == 'MATCH':
                ch_data = stage_data.get('chapter_data')
                if isinstance(ch_data, dict):
                    self.data.chapter_data = ch_data

            # US-49-012: Persist stage metrics for pipeline observability
            # US-106-005: Add completeness validation for stage metrics
            if stage_metrics:
                self.data.stage_metrics[stage] = stage_metrics
                # Validate metrics completeness and log warnings
                self._validate_stage_metrics_completeness(stage, stage_metrics)

            # US-115-002: Log dirty stage count for observability
            if dirty_stages:
                logger.debug(f"Incremental save: {len(dirty_stages)} dirty stages: {dirty_stages}")

            # Atomic save: write to temp, then rename
            # Stage completions always rotate backups (force_rotate=True)
            self._atomic_save(force_rotate=True, force_full=force_full, dirty_stages=dirty_stages)

            # US-159-010: Log checkpoint save operation with version and size
            checkpoint_version = getattr(self.data, 'version', CURRENT_CHECKPOINT_VERSION)
            # Calculate approximate data size for logging
            data_size = len(json.dumps(self.data.to_dict() if hasattr(self.data, 'to_dict') else self.data).encode('utf-8'))

            # US-166-008: Log checkpoint save with stage name and data size at INFO level
            logger.info(
                f"Checkpoint saved: stage={stage}, version={checkpoint_version}, "
                f"size={data_size} bytes, last_completed={self.data.last_completed_stage}"
            )

            # Log progress and completion
            log_progress(
                logger,
                "CHECKPOINT_SAVE",
                100,
                1,
                1,
                correlation_id=correlation_id,
            )
            log_stage_complete(
                logger,
                "CHECKPOINT_SAVE",
                correlation_id=correlation_id,
                stage=stage,
                checkpoint_saved=True,
            )

            # US-167-008: Log save timing at DEBUG level
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            logger.debug(
                f"Checkpoint save completed in {elapsed_ms:.1f}ms: "
                f"stage={stage}, data_size={data_size} bytes"
            )

            # US-130-007: Hot backup to secondary location after successful save
            # Graceful failure: log warning but don't fail the main save
            self._do_hot_backup()

            # US-130-010: Cloud backup to S3-compatible storage
            # Graceful failure: log warning but don't fail the main save
            self._do_cloud_backup()

            # US-130-009: Update checkpoint metadata index
            # Graceful failure: log warning but don't fail the main save
            self._update_index(stage)

        finally:
            # US-130-003: Always release lock after save
            self._release_lock()

        # US-115-007: Record history entry after save
        self._record_history_entry(stage, is_intermediate=False)

    def save_intermediate(self, stage: str, stage_data: Dict[str, Any] = None):
        """
        Save intermediate checkpoint during a long-running stage.

        Unlike save(), this does NOT update last_completed_stage.
        Used for periodic progress saves during stages like DOWNLOAD_SEGMENTS
        that can take a long time and benefit from being able to resume partway.

        Args:
            stage: The stage currently running (for logging)
            stage_data: Data to save for the stage
        """
        start_time = time.perf_counter()
        # US-130-003: Acquire file lock for concurrent process safety
        lock_acquired = self._acquire_lock()
        if not lock_acquired:
            logger.warning(
                f"Could not acquire checkpoint lock for intermediate save '{stage}' - "
                f"another process may be writing. Skipping save."
            )
            return

        try:
            if self.data is None:
                self.data = CheckpointData(
                    created_at=datetime.now().isoformat(),
                    config_hash=self.config_hash
                )

            self.data.updated_at = datetime.now().isoformat()

            # Store stage-specific data without changing last_completed_stage
            if stage_data:
                stage_key = stage.lower()
                if stage in _STAGE_FIELD_MAP_TERMINAL:
                    logger.debug(
                        f"save_intermediate(): stage '{stage}' is terminal — stage_data not persisted"
                    )
                elif stage_key not in CheckpointData.__dataclass_fields__:
                    logger.warning(
                        f"save_intermediate(): stage_key '{stage_key}' (from stage '{stage}') "
                        f"does not map to a CheckpointData field — data will not be persisted"
                    )
                elif hasattr(self.data, stage_key):
                    setattr(self.data, stage_key, stage_data)

            # US-115-002: Track dirty stages for incremental saving
            dirty_stages = self._get_dirty_stages()

            logger.debug(f"Saving intermediate checkpoint for {stage}")
            # Intermediate saves respect the rotation interval (force_rotate=False)
            self._atomic_save(force_rotate=False, force_full=False, dirty_stages=dirty_stages)

            # US-159-010: Log intermediate checkpoint save operation
            logger.info(
                f"Checkpoint intermediate saved: stage={stage}, path={self.checkpoint_path}"
            )

            # US-167-008: Log save timing at DEBUG level
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            logger.debug(
                f"Checkpoint intermediate save completed in {elapsed_ms:.1f}ms: stage={stage}"
            )

        finally:
            # US-130-003: Always release lock after save
            self._release_lock()

    def save_async(self, stage: str, stage_data: Dict[str, Any] = None):
        """
        Save checkpoint asynchronously in a background thread with coalesced writes.

        Multiple rapid save_async() calls are coalesced into a single write,
        reducing I/O overhead during high-frequency checkpoint updates.

        Args:
            stage: The stage currently running (for logging)
            stage_data: Data to save for the stage
        """
        # Ensure background thread is running
        self._ensure_async_thread()

        # Record queue time for coalescing metrics
        queue_time = time.time()
        request = {
            "stage": stage,
            "stage_data": stage_data,
            "queue_time": queue_time,
        }

        # Try to add to queue (non-blocking)
        try:
            self._async_save_queue.put_nowait(request)
            self._pending_save = True
        except queue.Full:
            # Queue full, do synchronous save as fallback
            logger.warning("Async save queue full, falling back to synchronous save")
            self.save_intermediate(stage, stage_data)

    def _ensure_async_thread(self):
        """Start background thread if not running."""
        if self._async_save_thread is None or not self._async_save_thread.is_alive():
            self._async_save_thread = threading.Thread(
                target=self._async_save_worker,
                daemon=True,
                name="CheckpointAsyncSave"
            )
            self._async_save_thread.start()

    def _async_save_worker(self):
        """Background worker that processes async save requests."""
        while True:
            try:
                # Wait for a request with timeout
                try:
                    request = self._async_save_queue.get(timeout=self._coalesce_window_ms / 1000.0)
                except queue.Empty:
                    # No more requests, exit worker if queue is empty
                    if self._async_save_queue.empty():
                        self._pending_save = False
                        break
                    continue

                # Track coalescing: collect all available requests
                requests = [request]
                while not self._async_save_queue.empty():
                    try:
                        requests.append(self._async_save_queue.get_nowait())
                    except queue.Empty:
                        break

                # Calculate coalescing metrics
                now = time.time()
                if len(requests) > 1:
                    self._async_save_metrics["saves_coalesced"] += len(requests) - 1
                    logger.debug(f"Coalesced {len(requests)} async saves into 1")

                # Track queue delays
                for req in requests:
                    delay_ms = (now - req["queue_time"]) * 1000
                    self._async_save_metrics["queue_delays"].append(delay_ms)

                # Keep only last 100 delays for averaging
                if len(self._async_save_metrics["queue_delays"]) > 100:
                    self._async_save_metrics["queue_delays"] = \
                        self._async_save_metrics["queue_delays"][-100:]

                # Update average
                delays = self._async_save_metrics["queue_delays"]
                if delays:
                    self._async_save_metrics["avg_queue_delay_ms"] = sum(delays) / len(delays)

                # Use the latest request's data (most recent state)
                latest_request = requests[-1]
                stage = latest_request["stage"]
                stage_data = latest_request["stage_data"]

                # Perform the actual save synchronously in the worker thread
                self._do_async_save(stage, stage_data)

                self._async_save_metrics["total_async_saves"] += 1
                self._async_save_queue.task_done()

            except Exception as e:
                logger.error(f"Error in async save worker: {e}")

        # Clear pending flag when exiting
        self._pending_save = False

    def _do_async_save(self, stage: str, stage_data: Dict[str, Any]):
        """Perform the actual checkpoint save for async operations."""
        start_time = time.perf_counter()
        # US-130-003: Acquire file lock for concurrent process safety
        lock_acquired = self._acquire_lock()
        if not lock_acquired:
            logger.warning(
                f"Could not acquire checkpoint lock for async save '{stage}' - "
                f"another process may be writing. Skipping save."
            )
            return

        try:
            if self.data is None:
                self.data = CheckpointData(
                    created_at=datetime.now().isoformat(),
                    config_hash=self.config_hash
                )

            self.data.updated_at = datetime.now().isoformat()

            # Store stage-specific data without changing last_completed_stage
            if stage_data:
                stage_key = stage.lower()
                if stage in _STAGE_FIELD_MAP_TERMINAL:
                    logger.debug(
                        f"save_async(): stage '{stage}' is terminal — stage_data not persisted"
                    )
                elif stage_key not in CheckpointData.__dataclass_fields__:
                    logger.warning(
                        f"save_async(): stage_key '{stage_key}' (from stage '{stage}') "
                        f"does not map to a CheckpointData field — data will not be persisted"
                    )
                elif hasattr(self.data, stage_key):
                    setattr(self.data, stage_key, stage_data)

            # Get dirty stages for incremental saving
            dirty_stages = self._get_dirty_stages()

            logger.debug(f"Async saving checkpoint for {stage}")
            # Async saves don't force rotation (like intermediate saves)
            self._atomic_save(force_rotate=False, force_full=False, dirty_stages=dirty_stages)

            # US-167-008: Log async save timing at DEBUG level
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            logger.debug(
                f"Checkpoint async save completed in {elapsed_ms:.1f}ms: stage={stage}"
            )

        finally:
            # US-130-003: Always release lock after save
            self._release_lock()

    def get_async_save_metrics(self) -> Dict[str, Any]:
        """Get async save metrics for reporting."""
        return {
            "saves_coalesced": self._async_save_metrics["saves_coalesced"],
            "avg_queue_delay_ms": self._async_save_metrics["avg_queue_delay_ms"],
            "total_async_saves": self._async_save_metrics["total_async_saves"],
            "pending_save": self._pending_save,
        }

    def flush_async_saves(self, timeout: float = 5.0) -> bool:
        """
        Wait for all pending async saves to complete.

        Args:
            timeout: Maximum time to wait in seconds

        Returns:
            True if all saves completed, False if timeout
        """
        if not self._pending_save:
            return True

        # Drain the queue first to signal worker to exit
        try:
            while True:
                self._async_save_queue.get_nowait()
        except queue.Empty:
            pass

        # Wait for worker to finish
        if self._async_save_thread and self._async_save_thread.is_alive():
            self._async_save_thread.join(timeout=timeout)
            return not self._async_save_thread.is_alive()

        return True

    def save_stage_timing_summary(
        self,
        stage_timings: Dict[str, float],
        total_duration: float,
        skipped_stages: set,
        validation_cache_stats: Dict[str, Any] = None
    ):
        """
        Persist pipeline timing summary to checkpoint stage_metrics.

        Stores per-stage duration and a '_pipeline' entry with total duration
        so timing data is available for post-run analysis.

        Args:
            stage_timings: Map of stage_name -> elapsed seconds for stages that ran.
            total_duration: Total pipeline wall-clock duration in seconds.
            skipped_stages: Set of stage names that were restored from checkpoint.
            validation_cache_stats: Optional validation cache statistics (US-88-009).
        """
        if self.data is None:
            return

        # Merge timing into existing stage_metrics (don't overwrite per-stage metrics)
        for stage_name, elapsed in stage_timings.items():
            if stage_name not in self.data.stage_metrics:
                self.data.stage_metrics[stage_name] = {}
            self.data.stage_metrics[stage_name]['duration_seconds'] = elapsed

        # Mark skipped stages
        for stage_name in skipped_stages:
            if stage_name not in self.data.stage_metrics:
                self.data.stage_metrics[stage_name] = {}
            self.data.stage_metrics[stage_name]['skipped'] = True

        # Build pipeline-level timing summary
        pipeline_metrics: Dict[str, Any] = {
            'total_duration_seconds': total_duration,
            'stages_run': list(stage_timings.keys()),
            'stages_skipped': list(skipped_stages),
        }

        # US-88-009: Add validation cache statistics
        if validation_cache_stats:
            pipeline_metrics['validation_cache'] = validation_cache_stats
        elif self.data.validation_cache:
            # Use current validation cache stats if not provided
            pipeline_metrics['validation_cache'] = self.get_validation_cache_stats()

        self.data.stage_metrics['_pipeline'] = pipeline_metrics

        self._atomic_save(force_rotate=True)

    def mark_stage_incomplete(self, stage: str):
        """
        Mark a stage as incomplete so it will be re-run.

        Used by --force-rematch to force re-running MATCH stage even when
        checkpoint shows it as complete.

        Args:
            stage: The stage to mark as incomplete
        """
        if self.data is None:
            return

        try:
            stage_idx = STAGE_ORDER.index(stage)
        except ValueError:
            logger.warning(f"Unknown stage: {stage}")
            return

        # Set last_completed_stage to the stage before this one
        if stage_idx > 0:
            self.data.last_completed_stage = STAGE_ORDER[stage_idx - 1]
        else:
            # If it's the first stage, clear last_completed_stage
            self.data.last_completed_stage = None

        # Clear any stage-specific data for this stage
        stage_key = stage.lower()
        if hasattr(self.data, stage_key):
            setattr(self.data, stage_key, {})

        self._atomic_save(force_rotate=True)
        logger.info(f"Marked {stage} as incomplete - will be re-run")

    def _get_backup_path(self, index: int) -> Path:
        """Get the path for a numbered backup file.

        index 0 -> checkpoint.backup.json (primary backup)
        index 1 -> checkpoint.backup.1.json
        index 2 -> checkpoint.backup.2.json
        """
        if index == 0:
            return self.backup_path
        return self.project_dir / f"checkpoint.backup.{index}.json"

    def _get_all_backup_paths(self) -> List[Path]:
        """Return list of all backup paths in order (newest first)."""
        return [self._get_backup_path(i) for i in range(self._backup_count)]

    def _rotate_backups(self):
        """Rotate backup files: N-1 -> deleted, N-2 -> N-1, ..., 0 -> 1, current -> 0.

        US-51-007: Maintains up to _backup_count backup files for resilience.
        US-108-010: Now creates backup BEFORE any write (not just migration).
        """
        try:
            # US-108-010: Always create backup BEFORE rotating/writing
            # This ensures we have a valid copy to restore if write fails
            if self.checkpoint_path.exists():
                # Create timestamped pre-write backup
                timestamp = int(time.time())
                pre_write_backup = self.project_dir / f"checkpoint.pre_write_{timestamp}.json"
                shutil.copy2(self.checkpoint_path, pre_write_backup)
                # US-166-008: Log checkpoint backup creation
                logger.info(f"Checkpoint backup created: {pre_write_backup.name}")
                # Clean up old pre-write backups (keep last 2)
                self._cleanup_old_backups(prefix="checkpoint.pre_write_", keep=2)

            # Delete the oldest backup if it exists
            oldest = self._get_backup_path(self._backup_count - 1)
            if oldest.exists():
                oldest.unlink()

            # Shift each backup up by one slot (work backwards)
            for i in range(self._backup_count - 1, 0, -1):
                src = self._get_backup_path(i - 1)
                dst = self._get_backup_path(i)
                if src.exists():
                    shutil.copy2(src, dst)

            # Copy current checkpoint to primary backup slot
            if self.checkpoint_path.exists():
                shutil.copy2(self.checkpoint_path, self.backup_path)
        except Exception as e:
            logger.warning(f"Failed to rotate backups: {e}")

    def restore_from_backup(self, backup_index: int) -> bool:
        """
        US-108-010: Restore checkpoint from a numbered backup file.

        Args:
            backup_index: Index of backup to restore (0 = checkpoint.backup.json,
                         1 = checkpoint.backup.1.json, etc.)

        Returns:
            True if restore was successful, False otherwise
        """
        if backup_index < 0 or backup_index >= self._backup_count:
            logger.error(f"Invalid backup index {backup_index}. Valid range: 0-{self._backup_count - 1}")
            return False

        backup_path = self._get_backup_path(backup_index)
        if not backup_path.exists():
            logger.error(f"Backup file does not exist: {backup_path}")
            return False

        try:
            # Verify backup is valid before restoring
            backup_data = self._try_load_file(backup_path)
            if backup_data is None:
                logger.error(f"Backup file is corrupt: {backup_path}")
                return False

            # Restore by copying backup to main checkpoint
            shutil.copy2(backup_path, self.checkpoint_path)
            logger.info(f"Restored checkpoint from backup index {backup_index}: {backup_path.name}")

            # Reload the restored data
            self.data = self._try_load_file(self.checkpoint_path)
            return True

        except Exception as e:
            logger.error(f"Failed to restore from backup {backup_index}: {e}")
            return False

    def get_backup_info(self) -> List[Dict[str, Any]]:
        """
        US-108-010: Get information about all available backups.

        Returns list of dicts with backup index, path, size, and modification time.
        """
        backups = []
        for i in range(self._backup_count):
            path = self._get_backup_path(i)
            if path.exists():
                stat = path.stat()
                backups.append({
                    'index': i,
                    'path': str(path.name),
                    'size_bytes': stat.st_size,
                    'modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                })
        return backups

    # US-115-009: Checkpoint export/import for cross-project sharing

    def export_checkpoint(self, output_path: str, include_stages: Optional[List[str]] = None) -> bool:
        """
        US-115-009: Export checkpoint to a file for sharing across projects.

        Args:
            output_path: Path where the exported checkpoint will be saved
            include_stages: Optional list of stage names to export (e.g., ['video_search', 'caption']).
                          If None, exports all stages. Stage names should match STAGE_ORDER entries
                          (e.g., 'VIDEO_SEARCH', 'CAPTION', 'MATCH', etc.)

        Returns:
            True if export was successful, False otherwise
        """
        try:
            # Load current checkpoint data
            data = self.load()
            if data is None:
                logger.error("No checkpoint data to export")
                return False

            # Convert to dict
            export_data = data.to_dict()

            # Filter to specific stages if requested
            if include_stages:
                # Normalize stage names (uppercase for matching, lowercase for field access)
                normalized_stages = []
                for stage in include_stages:
                    if stage.upper() in STAGE_ORDER:
                        normalized_stages.append(stage.upper())
                    elif stage.lower() in STAGE_FIELD_MAP.values():
                        # It's already a field name
                        normalized_stages.append(stage.upper())
                    else:
                        logger.warning(f"Unknown stage '{stage}' in include_stages, skipping")

                # Clear non-included stage data
                for stage in STAGE_ORDER:
                    if stage in _STAGE_FIELD_MAP_TERMINAL:
                        continue
                    field_name = STAGE_FIELD_MAP.get(stage)
                    if field_name and stage not in normalized_stages:
                        export_data[field_name] = {}

                # Also filter stage_metrics if specified
                if 'stage_metrics' in export_data:
                    filtered_metrics = {}
                    for stage in normalized_stages:
                        if stage in export_data.get('stage_metrics', {}):
                            filtered_metrics[stage] = export_data['stage_metrics'][stage]
                    export_data['stage_metrics'] = filtered_metrics

                logger.info(f"Exporting stages: {normalized_stages}")

            # Add export metadata
            export_data['_export_info'] = {
                'exported_at': datetime.now().isoformat(),
                'source_project': str(self.project_dir),
                'version': export_data.get('version', CURRENT_CHECKPOINT_VERSION),
                'included_stages': include_stages or 'all',
            }

            # Ensure output directory exists
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)

            # Write the export file
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2)

            logger.info(f"Checkpoint exported to: {output_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to export checkpoint: {e}")
            return False

    # US-130-011: Checkpoint export with data redaction

    def _get_default_redaction_patterns(self) -> Dict[str, Any]:
        """
        Get default redaction patterns for sensitive data.

        Returns:
            Dict with pattern configurations for different sensitive data types
        """
        return {
            # URL patterns to redact
            'url_patterns': [
                r'https?://[^\s"\'<>]+',  # Generic URLs
                r'youtube\.com/watch\?v=[^\s"\'<>]+',  # YouTube watch URLs
                r'youtu\.be/[^\s"\'<>]+',  # YouTube short URLs
            ],
            # Path patterns to redact
            'path_patterns': [
                r'[A-Za-z]:\\[^:\s"\'<>]+',  # Windows paths
                r'/home/[^/\s"\'<>]+',  # Unix home paths
                r'/Users/[^/\s"\'<>]+',  # macOS paths
            ],
            # Fields to redact entirely
            'redact_fields': [
                'cookie',
                'api_key',
                'apikey',
                'auth_token',
                'password',
                'secret',
                'token',
                'credentials',
            ],
            # Field values to redact (key contains these)
            'redact_field_contains': [
                'path',
                'url',
                'file',
                'dir',
            ],
            # Video ID patterns (redact actual IDs, keep structure)
            'redact_video_ids': True,
            # File path values
            'redact_file_paths': True,
        }

    def _redact_value(self, value: Any, patterns: Dict[str, Any], depth: int = 0) -> Any:
        """
        Recursively redact sensitive data from a value.

        Args:
            value: The value to redact
            patterns: Redaction pattern configuration
            depth: Current recursion depth (to prevent infinite loops)

        Returns:
            Redacted value
        """
        if depth > 20:  # Prevent infinite recursion
            return '[REDACTED:MAX_DEPTH]'

        if value is None:
            return None

        if isinstance(value, str):
            # Check if it's a URL
            if patterns.get('redact_urls', True):
                for url_pattern in patterns.get('url_patterns', []):
                    try:
                        import re
                        if re.search(url_pattern, value, re.IGNORECASE):
                            # For YouTube URLs, extract and hash the ID
                            if 'youtube.com' in value.lower() or 'youtu.be' in value.lower():
                                # Try to extract video ID
                                import re as re_module
                                vid_match = re_module.search(r'(?:v=|/)([a-zA-Z0-9_-]{11})', value)
                                if vid_match:
                                    return f'[REDACTED:youtube_video:{vid_match.group(1)[:3]}...]'
                            return '[REDACTED:URL]'
                    except re.error:
                        pass

            # Check if it's a file path
            if patterns.get('redact_file_paths', True):
                for path_pattern in patterns.get('path_patterns', []):
                    try:
                        import re
                        if re.search(path_pattern, value):
                            return '[REDACTED:PATH]'
                    except re.error:
                        pass

            return value

        if isinstance(value, dict):
            result = {}
            for key, val in value.items():
                key_lower = key.lower()

                # Check if this field should be fully redacted
                should_redact = False
                for redact_field in patterns.get('redact_fields', []):
                    if redact_field in key_lower:
                        should_redact = True
                        break

                if should_redact:
                    result[key] = '[REDACTED]'
                elif patterns.get('redact_field_contains'):
                    # Check if field name suggests sensitive content
                    for field_hint in patterns.get('redact_field_contains', []):
                        if field_hint in key_lower:
                            # Redact path/url values but keep keys
                            result[key] = self._redact_value(val, patterns, depth + 1)
                            break
                    else:
                        result[key] = self._redact_value(val, patterns, depth + 1)
                else:
                    result[key] = self._redact_value(val, patterns, depth + 1)
            return result

        if isinstance(value, list):
            return [self._redact_value(item, patterns, depth + 1) for item in value]

        # For other types, return as-is
        return value

    def export_checkpoint_redacted(
        self,
        output_path: str,
        include_stages: Optional[List[str]] = None,
        redact_patterns: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        US-130-011: Export checkpoint with sensitive data redacted.

        This method exports checkpoint data while redacting sensitive information
        such as URLs, file paths, and specific fields. Useful for debugging
        without exposing private data.

        Args:
            output_path: Path where the exported checkpoint will be saved
            include_stages: Optional list of stage names to export
            redact_patterns: Optional custom redaction patterns. If None, uses defaults.

        Returns:
            True if export was successful, False otherwise
        """
        try:
            # Load current checkpoint data
            data = self.load()
            if data is None:
                logger.error("No checkpoint data to export")
                return False

            # Convert to dict
            export_data = data.to_dict()

            # Get redaction patterns (default or custom)
            patterns = redact_patterns if redact_patterns else self._get_default_redaction_patterns()

            # Filter to specific stages if requested
            if include_stages:
                normalized_stages = []
                for stage in include_stages:
                    if stage.upper() in STAGE_ORDER:
                        normalized_stages.append(stage.upper())
                    elif stage.lower() in STAGE_FIELD_MAP.values():
                        normalized_stages.append(stage.upper())
                    else:
                        logger.warning(f"Unknown stage '{stage}' in include_stages, skipping")

                # Clear non-included stage data
                for stage in STAGE_ORDER:
                    if stage in _STAGE_FIELD_MAP_TERMINAL:
                        continue
                    field_name = STAGE_FIELD_MAP.get(stage)
                    if field_name and stage not in normalized_stages:
                        export_data[field_name] = {}

                # Also filter stage_metrics if specified
                if 'stage_metrics' in export_data:
                    filtered_metrics = {}
                    for stage in normalized_stages:
                        if stage in export_data.get('stage_metrics', {}):
                            filtered_metrics[stage] = export_data['stage_metrics'][stage]
                    export_data['stage_metrics'] = filtered_metrics

                logger.info(f"Exporting stages (redacted): {normalized_stages}")

            # Apply redaction
            export_data = self._redact_value(export_data, patterns)

            # Add export metadata
            export_data['_export_info'] = {
                'exported_at': datetime.now().isoformat(),
                'source_project': str(self.project_dir),
                'version': export_data.get('version', CURRENT_CHECKPOINT_VERSION),
                'included_stages': include_stages or 'all',
                'redacted': True,
            }

            # Ensure output directory exists
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)

            # Write the export file
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2)

            logger.info(f"Checkpoint exported (redacted) to: {output_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to export checkpoint (redacted): {e}")
            return False

    def import_checkpoint(self, input_path: str, target_project_dir: Optional[str] = None) -> bool:
        """
        US-115-009: Import checkpoint from a file for sharing across projects.

        Args:
            input_path: Path to the exported checkpoint file
            target_project_dir: Optional target project directory. If None, imports to
                               the current project directory. Note: This only updates the
                               checkpoint file location - the caller is responsible for
                               ensuring compatibility with the target project.

        Returns:
            True if import was successful, False otherwise
        """
        try:
            import_file = Path(input_path)
            if not import_file.exists():
                logger.error(f"Import file not found: {input_path}")
                return False

            # Load the import file
            with open(import_file, 'r', encoding='utf-8') as f:
                import_data = json.load(f)

            # Extract export info if present
            export_info = import_data.pop('_export_info', {})

            # Validate version compatibility
            import_version = import_data.get('version', 'unknown')
            if import_version != CURRENT_CHECKPOINT_VERSION:
                logger.warning(
                    f"Import checkpoint version '{import_version}' differs from "
                    f"current version '{CURRENT_CHECKPOINT_VERSION}'. Attempting to use anyway."
                )

            # Convert to CheckpointData for validation
            checkpoint_data = CheckpointData.from_dict(import_data)

            # Validate the imported data
            warnings = checkpoint_data.validate()
            if warnings:
                logger.warning(f"Imported checkpoint has warnings: {warnings}")

            # Determine target directory
            if target_project_dir:
                target_dir = Path(target_project_dir)
            else:
                target_dir = self.project_dir

            target_dir.mkdir(parents=True, exist_ok=True)

            # Create the target checkpoint path
            target_checkpoint = target_dir / self.CHECKPOINT_FILE

            # Back up existing checkpoint if it exists
            if target_checkpoint.exists():
                backup_path = target_dir / self.CHECKPOINT_BACKUP
                shutil.copy2(target_checkpoint, backup_path)
                logger.info(f"Backed up existing checkpoint to: {backup_path}")

            # Write the imported checkpoint
            with open(target_checkpoint, 'w', encoding='utf-8') as f:
                json.dump(import_data, f, indent=2)

            # Update internal state if importing to current project
            if not target_project_dir or Path(target_project_dir) == self.project_dir:
                self.data = checkpoint_data
                self._full_checkpoint_loaded = True

            logger.info(f"Checkpoint imported from: {input_path} to {target_checkpoint}")
            return True

        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in import file: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to import checkpoint: {e}")
            return False

    def export_checkpoint_compressed(self, output_path: str, include_stages: Optional[List[str]] = None) -> bool:
        """
        US-115-009: Export checkpoint with gzip compression.

        Args:
            output_path: Path where the exported checkpoint will be saved
            include_stages: Optional list of stage names to export

        Returns:
            True if export was successful, False otherwise
        """
        try:
            # Load current checkpoint data
            data = self.load()
            if data is None:
                logger.error("No checkpoint data to export")
                return False

            # Convert to dict
            export_data = data.to_dict()

            # Filter to specific stages if requested
            if include_stages:
                normalized_stages = [s.upper() for s in include_stages if s.upper() in STAGE_ORDER]
                for stage in STAGE_ORDER:
                    if stage in _STAGE_FIELD_MAP_TERMINAL:
                        continue
                    field_name = STAGE_FIELD_MAP.get(stage)
                    if field_name and stage not in normalized_stages:
                        export_data[field_name] = {}

                if 'stage_metrics' in export_data:
                    filtered_metrics = {
                        s: export_data['stage_metrics'][s]
                        for s in normalized_stages if s in export_data.get('stage_metrics', {})
                    }
                    export_data['stage_metrics'] = filtered_metrics

            # Add export metadata
            export_data['_export_info'] = {
                'exported_at': datetime.now().isoformat(),
                'source_project': str(self.project_dir),
                'version': export_data.get('version', CURRENT_CHECKPOINT_VERSION),
                'included_stages': include_stages or 'all',
                'compressed': True,
            }

            # Ensure output directory exists
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)

            # Write compressed
            with gzip.open(output_file, 'wt', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2)

            logger.info(f"Compressed checkpoint exported to: {output_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to export compressed checkpoint: {e}")
            return False

    def _atomic_save(self, force_rotate: bool = True, force_full: bool = False,
                     dirty_stages: List[str] = None):
        """Atomically save checkpoint (write temp, then rename).

        Args:
            force_rotate: If True, always rotate backups (stage completions).
                If False, only rotate when min_rotation_interval has elapsed
                (intermediate saves during long-running stages).
            force_full: If True, save entire checkpoint. If False, use incremental.
            dirty_stages: List of dirty stage field names for incremental saving.
        """
        start_time = time.perf_counter()
        temp_path = self.checkpoint_path.with_suffix('.tmp')
        try:
            # US-85-003: Gate backup rotation on interval for intermediate saves
            now = time.time()
            elapsed = now - self._last_rotation_time
            if force_rotate or elapsed >= self._min_rotation_interval:
                self._rotate_backups()
                self._last_rotation_time = now
            else:
                logger.debug(
                    f"Skipping backup rotation (elapsed={elapsed:.0f}s < "
                    f"interval={self._min_rotation_interval}s)"
                )

            # US-115-002: Implement incremental saving
            # For incremental saves, we need to merge dirty stages with existing checkpoint
            current_dict = self.data.to_dict()

            # US-130-006: Determine if this should be a full or differential save
            is_diff_save = (
                self._diff_enabled and
                not force_full and
                self._diff_base_data is not None and
                self._diff_save_count > 0 and
                self._diff_save_count < self._diff_full_interval
            )

            if is_diff_save:
                # Compute differential save - only store changes from base
                diff = self._compute_diff(current_dict, self._diff_base_data)
                # Store diff with reference to base checkpoint
                save_data = {
                    "_diff_marker": True,
                    "_diff_base_timestamp": self._diff_base_data.get("updated_at", ""),
                    "_diff_sequence": self._diff_save_count,
                    "_diff_fields": diff.get("_diff_fields", {}),
                    "_diff_version": diff.get("_diff_version", 1),
                }
                # Always include critical metadata in diff saves
                save_data["updated_at"] = current_dict.get("updated_at", "")
                save_data["last_completed_stage"] = current_dict.get("last_completed_stage", "")
                save_data["config_hash"] = current_dict.get("config_hash", "")
                self._diff_stats["diff_saves"] = self._diff_stats.get("diff_saves", 0) + 1

                # Calculate space savings
                full_size = len(json.dumps(current_dict, indent=2, default=str).encode('utf-8'))
                diff_size = len(json.dumps(save_data, indent=2, default=str).encode('utf-8'))
                self._diff_stats["total_original_bytes"] = (
                    self._diff_stats.get("total_original_bytes", 0) + full_size
                )
                self._diff_stats["total_diff_bytes"] = (
                    self._diff_stats.get("total_diff_bytes", 0) + diff_size
                )
                self._diff_stats["space_saved_bytes"] = (
                    self._diff_stats.get("space_saved_bytes", 0) + (full_size - diff_size)
                )
            elif not force_full and dirty_stages and self.checkpoint_path.exists():
                # Load existing checkpoint and merge dirty stages
                try:
                    # Try to load compressed first, then uncompressed
                    existing_data = self._try_load_checkpoint_data(self.checkpoint_path)
                except (json.JSONDecodeError, IOError):
                    logger.warning("Could not load existing checkpoint for incremental merge, doing full save")
                    existing_data = None

                if existing_data:
                    # Update only dirty stages in existing data
                    for stage_field in dirty_stages:
                        existing_data[stage_field] = current_dict.get(stage_field, {})
                    # Also update metadata fields that always change
                    existing_data['updated_at'] = current_dict.get('updated_at', '')
                    existing_data['last_completed_stage'] = current_dict.get('last_completed_stage', '')
                    save_data = existing_data
                else:
                    save_data = current_dict

                # Reset diff tracking after non-differential save
                self._diff_base_data = current_dict
                self._diff_save_count = 0
                self._diff_stats["full_saves"] = self._diff_stats.get("full_saves", 0) + 1
            else:
                save_data = current_dict

                # Set new base for differential saves
                self._diff_base_data = current_dict
                self._diff_save_count = 0
                self._diff_stats["full_saves"] = self._diff_stats.get("full_saves", 0) + 1

            # Increment diff save counter for next time
            if is_diff_save:
                self._diff_save_count += 1

            # US-130-008: Compute cryptographic signature before saving
            if self._signature_enabled and self._signature_key:
                # For diff saves, we need to sign critical metadata
                if is_diff_save:
                    # Add signature to the save_data for diff saves
                    save_data["signature"] = self._compute_signature(save_data)
                else:
                    # For full saves, compute signature on the full data
                    save_data["signature"] = self._compute_signature(save_data)
                logger.debug(f"Checkpoint signature computed for save (diff={is_diff_save})")

            # US-130-005: Apply content-addressable deduplication to stage data
            if self._dedup_enabled:
                deduped_data, dedup_store = self._deduplicate_data(save_data)
                # Include dedup_store in the saved data under special key
                if dedup_store:
                    save_data = deduped_data
                    save_data["_dedup_store"] = dedup_store

            # Serialize to JSON
            json_content = json.dumps(save_data, indent=2, default=str)
            json_bytes = json_content.encode('utf-8')
            original_size = len(json_bytes)

            # US-115-003: Compress checkpoint if enabled
            if self._compression_enabled:
                compressed_bytes = gzip.compress(json_bytes, compresslevel=self._compression_level)
                compressed_size = len(compressed_bytes)
                write_bytes = compressed_bytes
                is_compressed = True
                # Track compression stats
                self._compression_stats = {
                    'original_size': original_size,
                    'compressed_size': compressed_size,
                    'ratio': compressed_size / original_size if original_size > 0 else 1.0,
                }
            else:
                write_bytes = json_bytes
                is_compressed = False
                self._compression_stats = {
                    'original_size': original_size,
                    'compressed_size': original_size,
                    'ratio': 1.0,
                }

            # Write to temp file (compressed or uncompressed based on config)
            # Use .json.gz suffix when compressed to make it identifiable
            if is_compressed:
                temp_path = self.checkpoint_path.with_suffix('.json.gz.tmp')
            else:
                temp_path = self.checkpoint_path.with_suffix('.tmp')

            with open(temp_path, 'wb') as f:
                f.write(write_bytes)

            # US-115-002: Update last saved data for next incremental comparison
            self._last_saved_data = self.data.to_dict()

            # Atomic rename/move
            # On Windows, Path.replace() fails if the file is open.
            # shutil.move() is more robust.
            try:
                if self.checkpoint_path.exists():
                    self.checkpoint_path.unlink()
                shutil.move(str(temp_path), str(self.checkpoint_path))
            except Exception as e:
                logger.debug(f"Path.replace failed, trying os.replace/rename fallback: {e}")
                import os
                if os.path.exists(self.checkpoint_path):
                    try:
                        os.remove(self.checkpoint_path)
                    except:
                        pass
                os.rename(temp_path, self.checkpoint_path)

            # Log save time and I/O stats
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            if dirty_stages:
                logger.debug(f"Checkpoint incrementally saved ({len(dirty_stages)} stages) in {elapsed_ms:.1f}ms")
            else:
                logger.debug(f"Checkpoint saved in {elapsed_ms:.1f}ms")

            # US-159-010: Log checkpoint data size
            actual_size = self.checkpoint_path.stat().st_size if self.checkpoint_path.exists() else 0
            logger.info(
                f"Checkpoint save complete: size={actual_size} bytes, "
                f"compressed={is_compressed}, original_size={original_size} bytes"
            )

            # US-130-002: Update checkpoint statistics
            self._stats["save_count"] = self._stats.get("save_count", 0) + 1
            self._stats["total_save_time_ms"] = self._stats.get("total_save_time_ms", 0.0) + elapsed_ms
            # Keep rolling average of last N save times
            save_times = self._stats.get("save_times_ms", [])
            save_times.append(elapsed_ms)
            max_times = self._stats.get("max_save_times", 50)
            if len(save_times) > max_times:
                save_times = save_times[-max_times:]
            self._stats["save_times_ms"] = save_times
            # Track last save size
            if self._compression_stats:
                self._stats["last_save_size_bytes"] = self._compression_stats.get("compressed_size", 0)
                self._stats["total_size_bytes"] = self._compression_stats.get("compressed_size", 0)
            # Update oldest/newest checkpoint timestamps
            if self.data:
                if self.data.created_at and not self._stats.get("oldest_checkpoint"):
                    self._stats["oldest_checkpoint"] = self.data.created_at
                if self.data.updated_at:
                    self._stats["newest_checkpoint"] = self.data.updated_at

            # Log compression stats if enabled
            if self._compression_enabled and self._compression_stats:
                logger.debug(
                    f"Checkpoint compression: {self._compression_stats['original_size']} -> "
                    f"{self._compression_stats['compressed_size']} bytes "
                    f"({self._compression_stats['ratio']:.1%})"
                )

            # US-115-012: Verify checkpoint integrity after save if enabled
            if self._verify_on_save:
                try:
                    verify_result = self.verify_integrity()
                    self._last_verified_at = datetime.now().isoformat()
                    if not verify_result['is_valid']:
                        logger.warning(
                            f"Post-save integrity check failed: {verify_result.get('issues', [])}"
                        )
                    else:
                        logger.debug(f"Post-save integrity check passed: {verify_result.get('checksum', '')[:8]}...")
                    # Add to history
                    self._integrity_check_history.append({
                        'timestamp': self._last_verified_at,
                        'type': 'on_save',
                        'is_valid': verify_result['is_valid'],
                        'checksum': verify_result.get('checksum', ''),
                    })
                except Exception as verify_err:
                    logger.error(f"Post-save integrity verification failed: {verify_err}")

        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            log_error_with_context(
                logger,
                "PIPE-001",
                f"Checkpoint save failed: {e}",
                correlation_id=get_correlation_id(),
                checkpoint_path=str(self.checkpoint_path),
            )
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except:
                    pass
            raise

    def _get_dirty_stages(self) -> List[str]:
        """US-115-002: Identify which stage data has changed since last save.

        Compares current stage data fields against the last saved state.
        Returns list of field names that have changed.

        Returns:
            List of dirty field names (e.g., ['match', 'caption', 'stage_metrics'])
        """
        if self._last_saved_data is None:
            # First save - everything is dirty
            return self._get_stage_field_names()

        dirty = []
        current_dict = self.data.to_dict()

        for field_name in self._get_stage_field_names():
            current_val = current_dict.get(field_name, {})
            last_val = self._last_saved_data.get(field_name, {})

            # Deep comparison for dicts
            if current_val != last_val:
                dirty.append(field_name)

        return dirty

    def _get_stage_field_names(self) -> List[str]:
        """Get list of stage data field names that can be incrementally saved."""
        return [
            'analyze', 'video_search', 'caption', 'match',
            'iterative_match', 'download_segments', 'chapter_data',
            'stage_metrics', 'transcription_metrics', 'validation_cache',
            'escalation_state', 'circuit_breaker_health', 'circuit_breaker_state'
        ]

    def set_voiceover(self, voiceover_path: str):
        """Set voiceover info for validation on resume.

        US-130-012: Normalizes the path using os.path.realpath for cross-platform
        compatibility before storing in the checkpoint.
        """
        if self.data is None:
            self.data = CheckpointData(
                created_at=datetime.now().isoformat(),
                config_hash=self.config_hash
            )

        # US-130-012: Normalize path for cross-platform compatibility
        normalized_path = self._normalize_path(voiceover_path)
        self.data.voiceover_path = normalized_path
        self.data.voiceover_hash = self._hash_file(normalized_path)
    
    def _normalize_path(self, path: str) -> str:
        """US-130-012: Normalize path for cross-platform compatibility.

        Uses os.path.realpath to resolve the path to its canonical form,
        handling symbolic links and relative paths. Also stores the
        platform separator used for later cross-platform detection.

        Args:
            path: The path to normalize

        Returns:
            Normalized absolute path
        """
        if not path:
            return path

        try:
            # Use realpath to resolve to absolute path with symlinks resolved
            normalized = os.path.realpath(path)
            logger.debug(f"Normalized path: {path} -> {normalized}")
            return normalized
        except (OSError, IOError) as e:
            # If realpath fails (e.g., file doesn't exist), return original
            logger.debug(f"Could not normalize path {path}: {e}")
            return path

    def _convert_path_separators(self, path: str, original_separator: str = None) -> str:
        """US-130-012: Convert path separators between platforms.

        Detects if path was stored on a different platform and converts
        separators accordingly. If original_separator is not provided,
        attempts to detect it from the path.

        Args:
            path: The path with potentially incompatible separators
            original_separator: The separator used when path was stored (e.g., '\\' for Windows)

        Returns:
            Path with separators converted to current platform
        """
        if not path:
            return path

        # Detect original separator if not provided
        if original_separator is None:
            if '\\' in path and os.sep == '/':
                original_separator = '\\'
            elif '/' in path and os.sep == '\\':
                original_separator = '/'
            else:
                # Same platform or no conversion needed
                return path

        if original_separator == os.sep:
            # No conversion needed
            return path

        # Convert separators
        if original_separator == '\\' and os.sep == '/':
            # Windows -> Linux
            converted = path.replace('\\', '/')
            logger.info(f"Converted Windows path to Linux: {path} -> {converted}")
            return converted
        elif original_separator == '/' and os.sep == '\\':
            # Linux -> Windows
            converted = path.replace('/', '\\')
            logger.info(f"Converted Linux path to Windows: {path} -> {converted}")
            return converted

        return path

    def _detect_and_normalize_paths(self, data: CheckpointData) -> bool:
        """US-130-012: Detect and normalize paths on checkpoint load.

        Checks if paths in the checkpoint were stored on a different platform
        and auto-converts if needed. Also normalizes paths using realpath.

        Args:
            data: The CheckpointData to normalize paths in

        Returns:
            True if any path normalization was applied, False otherwise
        """
        normalized = False
        warnings = []

        # Check voiceover_path
        if data.voiceover_path:
            original_path = data.voiceover_path

            # First, convert separators if platform changed
            data.voiceover_path = self._convert_path_separators(data.voiceover_path)

            # Then, normalize using realpath
            normalized_path = self._normalize_path(data.voiceover_path)

            # Check if path actually exists after normalization
            if os.path.exists(normalized_path):
                if normalized_path != original_path:
                    data.voiceover_path = normalized_path
                    normalized = True
                    warnings.append(f"Voiceover path normalized: {original_path} -> {normalized_path}")
            elif os.path.exists(data.voiceover_path):
                # Original path works but might have separator issues
                # Still normalize for consistency
                data.voiceover_path = normalized_path
                normalized = True
                warnings.append(f"Voiceover path separator converted: {original_path} -> {normalized_path}")
            else:
                # Path doesn't exist either way - still try to normalize
                # but warn user
                if normalized_path != original_path:
                    data.voiceover_path = normalized_path
                    normalized = True
                    warnings.append(f"Voiceover path normalized (file may not exist): {original_path} -> {normalized_path}")

        # Log warnings
        for warning in warnings:
            logger.warning(f"Checkpoint path normalization: {warning}")

        return normalized

    def _hash_file(self, filepath: str) -> str:
        """Get hash of file contents"""
        try:
            with open(filepath, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()[:16]
        except (OSError, IOError) as e:
            # File not found, permission denied, or I/O error - return empty hash
            logger.debug(f"Could not hash file {filepath}: {e}")
            return ""
    
    def validate(self, voiceover_path: str = None) -> Dict[str, Any]:
        """
        Validate checkpoint for resume.
        Returns dict with:
          - valid: bool
          - warnings: List[str]
          - errors: List[str]
          - resume_from: str (stage to resume from)
        """
        result = {
            'valid': True,
            'warnings': [],
            'errors': [],
            'resume_from': None,
            'completed_stages': []
        }
        
        if not self.data:
            result['valid'] = False
            result['errors'].append("No checkpoint data loaded")
            return result
        
        # Check config hash
        if self.config_hash and self.data.config_hash:
            if self.config_hash != self.data.config_hash:
                result['warnings'].append(
                    "Configuration has changed since checkpoint was created. "
                    "Some settings may not match."
                )
        
        # Check voiceover
        if voiceover_path:
            current_hash = self._hash_file(voiceover_path)
            if self.data.voiceover_hash and current_hash != self.data.voiceover_hash:
                result['warnings'].append(
                    "Voiceover file has changed since checkpoint was created."
                )
        
        # Determine resume point
        if self.data.last_completed_stage:
            try:
                stage_idx = STAGE_ORDER.index(self.data.last_completed_stage)
                result['completed_stages'] = STAGE_ORDER[:stage_idx + 1]
                
                if stage_idx < len(STAGE_ORDER) - 1:
                    result['resume_from'] = STAGE_ORDER[stage_idx + 1]
                else:
                    result['valid'] = False
                    result['errors'].append("Pipeline already completed")
            except ValueError:
                result['warnings'].append(
                    f"Unknown stage '{self.data.last_completed_stage}' in checkpoint"
                )
        
        # Check video search results
        if self.data.video_search and 'video_ids' in self.data.video_search:
            vid_count = len(self.data.video_search['video_ids'])
            if vid_count == 0:
                result['warnings'].append("No video IDs found in search results")

        return result

    def verify_integrity(self) -> Dict[str, Any]:
        """
        US-108-010: Verify checkpoint integrity with checksum and schema validation.

        Returns dict with:
          - is_valid: bool
          - checksum: str (MD5 hash of checkpoint content)
          - schema_valid: bool
          - issues: List[str]
          - backup_count: int (number of available backups)
        """
        result = {
            'is_valid': False,
            'checksum': '',
            'schema_valid': False,
            'issues': [],
            'backup_count': 0,
            'version': '',
            'last_stage': '',
        }

        if not self.exists():
            result['issues'].append("Checkpoint file does not exist")
            return result

        # Compute checksum
        try:
            with open(self.checkpoint_path, 'rb') as f:
                content = f.read()
                result['checksum'] = hashlib.md5(content).hexdigest()
        except Exception as e:
            result['issues'].append(f"Failed to compute checksum: {e}")
            return result

        # Parse and validate schema (handle compressed files)
        try:
            # Check if file is gzip compressed (magic bytes 0x1f 0x8b)
            if len(content) >= 2 and content[0] == 0x1f and content[1] == 0x8b:
                content = gzip.decompress(content)
            data = json.loads(content.decode('utf-8'))
            result['schema_valid'] = True
            result['version'] = data.get('version', 'unknown')
            result['last_stage'] = data.get('last_completed_stage', '')
        except json.JSONDecodeError as e:
            result['issues'].append(f"Invalid JSON: {e}")
            return result

        # Validate required fields
        required_fields = ['version', 'last_completed_stage', 'created_at']
        for field in required_fields:
            if field not in data:
                result['issues'].append(f"Missing required field: {field}")
                result['schema_valid'] = False

        # Validate version compatibility
        version = data.get('version', '')
        if version and version != CURRENT_CHECKPOINT_VERSION:
            migratable = ['0.9', '1.0', '2.0', '2.1']
            if version not in migratable:
                result['issues'].append(f"Incompatible version: {version}")
                result['schema_valid'] = False

        # US-130-008: Verify cryptographic signature if enabled
        signature_result = self.verify_signature(data)
        result['signature_valid'] = signature_result['is_valid']
        result['signature_present'] = signature_result.get('signature_present', False)
        if not signature_result['is_valid'] and signature_result.get('signature_present'):
            # Only add issue if signature was present but invalid
            result['issues'].append(f"Signature verification failed: {signature_result.get('issue', 'unknown')}")

        # Count available backups
        backup_count = 0
        for i in range(self._backup_count):
            if self._get_backup_path(i).exists():
                backup_count += 1
        result['backup_count'] = backup_count

        # Determine overall validity
        result['is_valid'] = len(result['issues']) == 0 and result['schema_valid']

        return result

    def verify_all_backups(self) -> Dict[str, Any]:
        """US-115-012: Verify integrity of all rotated checkpoint backups.

        Checks each backup file in the rotation chain for:
        - Valid JSON structure
        - Required fields present
        - Version compatibility

        Returns:
            Dict with verification results:
            - all_valid: bool - True if all backups are valid
            - backup_count: int - Number of backups checked
            - valid_count: int - Number of valid backups
            - issues: List[str] - List of issues found per backup
            - backup_results: List[Dict] - Per-backup verification details
        """
        result = {
            'all_valid': True,
            'backup_count': 0,
            'valid_count': 0,
            'issues': [],
            'backup_results': [],
        }

        if not self._verify_backups:
            logger.debug("Backup verification disabled, skipping")
            result['issues'].append("Backup verification is disabled")
            result['all_valid'] = True  # Not an error, just disabled
            return result

        # Check main checkpoint first
        if self.exists():
            result['backup_count'] += 1
            main_result = self.verify_integrity()
            result['backup_results'].append({
                'path': str(self.checkpoint_path),
                'valid': main_result['is_valid'],
                'issues': main_result.get('issues', []),
            })
            if not main_result['is_valid']:
                result['all_valid'] = False
                result['issues'].append(f"Main checkpoint invalid: {main_result.get('issues', [])}")
            else:
                result['valid_count'] += 1

        # Check rotated backups
        for i in range(self._backup_count):
            backup_path = self._get_backup_path(i)
            if not backup_path.exists():
                continue

            result['backup_count'] += 1
            backup_result = {
                'path': str(backup_path),
                'valid': True,
                'issues': [],
            }

            # Verify backup file
            try:
                with open(backup_path, 'rb') as f:
                    content = f.read()

                # Check if compressed (magic bytes 0x1f 0x8b)
                if len(content) >= 2 and content[0] == 0x1f and content[1] == 0x8b:
                    content = gzip.decompress(content)

                data = json.loads(content.decode('utf-8'))

                # Check required fields
                required_fields = ['version', 'last_completed_stage', 'created_at']
                for field in required_fields:
                    if field not in data:
                        backup_result['valid'] = False
                        backup_result['issues'].append(f"Missing required field: {field}")

                # Check version compatibility
                version = data.get('version', '')
                if version and version != CURRENT_CHECKPOINT_VERSION:
                    migratable = ['0.9', '1.0', '2.0', '2.1']
                    if version not in migratable:
                        backup_result['valid'] = False
                        backup_result['issues'].append(f"Incompatible version: {version}")

            except json.JSONDecodeError as e:
                backup_result['valid'] = False
                backup_result['issues'].append(f"Invalid JSON: {e}")
            except Exception as e:
                backup_result['valid'] = False
                backup_result['issues'].append(f"Verification error: {e}")

            result['backup_results'].append(backup_result)

            if not backup_result['valid']:
                result['all_valid'] = False
                result['issues'].append(f"Backup {i} invalid: {backup_result['issues']}")
            else:
                result['valid_count'] += 1

        # Update tracking state
        self._all_backups_valid = result['all_valid']
        self._last_verified_at = datetime.now().isoformat()

        # Add to history
        self._integrity_check_history.append({
            'timestamp': self._last_verified_at,
            'type': 'all_backups',
            'all_valid': result['all_valid'],
            'backup_count': result['backup_count'],
            'valid_count': result['valid_count'],
        })

        logger.info(
            f"Backup verification complete: {result['valid_count']}/{result['backup_count']} valid, "
            f"all_valid={result['all_valid']}"
        )

        return result

    def schedule_verification(self, cron_expression: str = None) -> Dict[str, Any]:
        """US-115-012: Start scheduled periodic integrity verification.

        Starts a background thread that runs integrity checks based on the
        provided cron expression.

        Args:
            cron_expression: Optional cron expression to override config.
                           Format: "minute hour day month weekday"

        Returns:
            Dict with scheduling status:
            - scheduled: bool - True if verification was scheduled
            - cron_expression: str - The cron expression being used
            - message: str - Status message
        """
        if not self._schedule_verification and cron_expression is None:
            return {
                'scheduled': False,
                'cron_expression': self._cron_expression,
                'message': 'Scheduled verification is disabled in config',
            }

        # Use provided cron_expression or fall back to config
        cron_expr = cron_expression or self._cron_expression

        # Stop existing thread if running
        if self._scheduled_verification_thread is not None and self._scheduled_verification_thread.is_alive():
            self.stop_scheduled_verification()
            logger.info("Stopped existing scheduled verification thread")

        # Reset stop event
        self._stop_scheduled_verification.clear()

        # Start new verification thread
        self._scheduled_verification_thread = threading.Thread(
            target=self._scheduled_verification_loop,
            args=(cron_expr,),
            daemon=True,
            name="checkpoint-integrity-scheduler"
        )
        self._scheduled_verification_thread.start()

        logger.info(f"Started scheduled checkpoint verification with cron: {cron_expr}")

        return {
            'scheduled': True,
            'cron_expression': cron_expr,
            'message': 'Scheduled verification started',
        }

    def stop_scheduled_verification(self) -> None:
        """US-115-012: Stop scheduled periodic integrity verification."""
        if self._scheduled_verification_thread is not None and self._scheduled_verification_thread.is_alive():
            self._stop_scheduled_verification.set()
            self._scheduled_verification_thread.join(timeout=5.0)
            logger.info("Stopped scheduled checkpoint verification")

    def _scheduled_verification_loop(self, cron_expression: str) -> None:
        """Background loop for scheduled integrity verification.

        Parses cron expression and runs verification at appropriate intervals.
        Note: Simplified implementation - for production, consider using a
        proper cron library like croniter.
        """
        import re

        # Simple cron parser - extract minute interval from expression
        # Format: "minute hour day month weekday"
        # Default: every hour (at minute 0)
        cron_parts = cron_expression.split()

        if len(cron_parts) < 1:
            logger.warning(f"Invalid cron expression: {cron_expression}, using default (every hour)")
            interval_seconds = 3600
        else:
            minute_part = cron_parts[0]
            # Handle special cases
            if minute_part == '*':
                interval_seconds = 60  # Every minute
            elif ',' in minute_part:
                # Multiple specific minutes - use first interval
                minutes = [int(m) for m in minute_part.split(',') if m.isdigit()]
                interval_seconds = 60 if not minutes else (minutes[1] - minutes[0]) * 60 if len(minutes) > 1 else 3600
            elif '-' in minute_part:
                # Range - use average interval
                parts = minute_part.split('-')
                if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                    interval_seconds = (int(parts[1]) - int(parts[0])) * 60
                else:
                    interval_seconds = 3600
            elif minute_part.isdigit():
                # Specific minute - run every hour
                interval_seconds = 3600
            else:
                interval_seconds = 3600  # Default to hourly

        logger.info(f"Scheduled verification interval: {interval_seconds} seconds")

        while not self._stop_scheduled_verification.wait(timeout=interval_seconds):
            # Run verification
            try:
                result = self.verify_all_backups()
                if not result['all_valid']:
                    logger.warning(
                        f"Scheduled integrity check found issues: {result['issues']}"
                    )
                else:
                    logger.debug(f"Scheduled integrity check passed: {result['valid_count']}/{result['backup_count']} valid")
            except Exception as e:
                logger.error(f"Scheduled integrity check failed: {e}")

    def get_integrity_status(self) -> Dict[str, Any]:
        """US-115-012: Get current integrity check status.

        Returns:
            Dict with:
            - last_verified_at: str or None - ISO timestamp of last verification
            - all_backups_valid: bool - Whether all backups were valid last check
            - verify_on_save_enabled: bool - Whether verify_on_save is enabled
            - schedule_verification_enabled: bool - Whether scheduled verification is running
            - history_count: int - Number of verification events in history
        """
        return {
            'last_verified_at': self._last_verified_at,
            'all_backups_valid': self._all_backups_valid,
            'verify_on_save_enabled': self._verify_on_save,
            'schedule_verification_enabled': self._schedule_verification,
            'schedule_verification_running': (
                self._scheduled_verification_thread is not None and
                self._scheduled_verification_thread.is_alive()
            ),
            'history_count': len(self._integrity_check_history),
        }

    def get_stage_data(self, stage: str) -> Dict[str, Any]:
        """US-115-005: Get saved data for a specific stage with lazy loading.

        Uses lazy loading to load only the requested stage data on-demand,
        improving startup time for large checkpoints.
        """
        # If full checkpoint is already loaded, use existing data
        if self.data is not None:
            stage_key = stage.lower()
            return getattr(self.data, stage_key, {})

        # Use lazy loading - only load requested stage
        return self.load_stage_data(stage)

    def get_chapter_data(self) -> Dict[str, Any]:
        """US-71-009: Get chapter/listicle detection data from checkpoint.

        Returns dict with 'chapters' and 'listicle_groups' lists.
        Returns empty dict if no chapter data is available (backward compat).
        Uses lazy loading if full checkpoint not yet loaded.
        """
        if self.data is not None:
            return self.data.chapter_data or {}

        # Use lazy loading for chapter_data
        if self._raw_checkpoint_data is None:
            self._load_raw_checkpoint_data()

        if self._raw_checkpoint_data is None:
            return {}

        chapter_data = self._raw_checkpoint_data.get("chapter_data", {})
        if not isinstance(chapter_data, dict):
            return {}
        return chapter_data

    def save_transcription_metrics(self, metrics_summary: Dict[str, Any]) -> None:
        """Persist transcription metrics summary to checkpoint (US-79-012).

        Called after transcribe_videos_parallel() completes to enable
        cross-run comparison and debugging slow transcription batches.

        Args:
            metrics_summary: Dict from TranscriptionMetrics.get_summary_dict().
                Must include: total_videos, cached_videos (cached_hits),
                transcribed_videos (transcribed_count), failed_videos (failed_count),
                total_duration_seconds (total_duration_s), average_speed_ratio (avg_speed_ratio).
        """
        if self.data is None:
            self.data = CheckpointData(
                created_at=datetime.now().isoformat(),
                config_hash=self.config_hash
            )

        self.data.transcription_metrics = metrics_summary
        self.data.updated_at = datetime.now().isoformat()
        self._atomic_save(force_rotate=False)
        logger.info(
            f"Saved transcription metrics: {metrics_summary.get('transcribed_count', 0)} transcribed, "
            f"{metrics_summary.get('cached_hits', 0)} cached, "
            f"{metrics_summary.get('failed_count', 0)} failed"
        )

    def get_transcription_metrics(self) -> Dict[str, Any]:
        """Get persisted transcription metrics from checkpoint (US-79-012).

        Returns:
            Dict with transcription metrics summary, or empty dict if not available.
        """
        if not self.data:
            return {}
        return self.data.transcription_metrics or {}

    def log_previous_transcription_metrics(self) -> None:
        """Log previous transcription metrics at INFO level for comparison (US-79-012).

        Called during checkpoint resume to show the user what the previous
        transcription performance was, enabling cross-run comparison.
        """
        prev_metrics = self.get_transcription_metrics()
        if not prev_metrics:
            return

        total = prev_metrics.get('total_videos', 0)
        cached = prev_metrics.get('cached_videos', 0)
        transcribed = prev_metrics.get('transcribed_videos', 0)
        failed = prev_metrics.get('failed_videos', 0)
        speed = prev_metrics.get('average_speed_ratio', 0.0)
        duration = prev_metrics.get('total_duration_seconds', 0.0)

        logger.info(
            f"Previous transcription metrics: {total} videos "
            f"({cached} cached, {transcribed} transcribed, {failed} failed), "
            f"{speed:.1f}x realtime, {duration:.0f}s total audio"
        )

    def get_stage_metrics(self, stage: str) -> Dict[str, Any]:
        """Get persisted stage metrics for a specific stage (US-49-012).

        Args:
            stage: Stage name (e.g., 'DOWNLOAD_SEGMENTS')

        Returns:
            Dict with serialized StageMetrics, or empty dict if not available.
        """
        if not self.data or not self.data.stage_metrics:
            return {}
        return self.data.stage_metrics.get(stage, {})

    # === US-88-009: Stage input validation caching ===

    def save_validation_result(
        self,
        stage: str,
        is_valid: bool,
        error_message: Optional[str] = None
    ) -> None:
        """
        Save stage input validation result to cache.

        Stores the validation result along with the current config hash
        so we can skip redundant validation on resume when config hasn't changed.

        Args:
            stage: Stage name (e.g., 'DOWNLOAD_SEGMENTS')
            is_valid: Whether stage inputs are valid
            error_message: Optional error message if validation failed
        """
        if self.data is None:
            self.data = CheckpointData(
                created_at=datetime.now().isoformat(),
                config_hash=self.config_hash
            )

        self.data.validation_cache[stage] = {
            'config_hash': self.config_hash,
            'validated_at': datetime.now().isoformat(),
            'is_valid': is_valid,
            'error_message': error_message,
        }
        logger.debug(
            f"Validation cache saved for {stage}: valid={is_valid}, "
            f"config_hash={self.config_hash[:8] if self.config_hash else 'none'}"
        )

    def get_cached_validation(self, stage: str) -> Optional[Dict[str, Any]]:
        """
        Get cached validation result for a stage.

        Returns the cached validation if:
        - Cache entry exists for this stage
        - Config hash matches current config

        Args:
            stage: Stage name

        Returns:
            Dict with cached validation result, or None if cache is invalid/missing.
        """
        if not self.data or not self.data.validation_cache:
            return None

        cached = self.data.validation_cache.get(stage)
        if not cached:
            return None

        # Check if config hash matches
        cached_hash = cached.get('config_hash')
        if cached_hash and cached_hash != self.config_hash:
            logger.debug(
                f"Validation cache invalidated for {stage}: "
                f"config changed ({cached_hash[:8]} -> {self.config_hash[:8]})"
            )
            return None

        logger.debug(
            f"Validation cache hit for {stage}: "
            f"is_valid={cached.get('is_valid')}"
        )
        return cached

    def is_validation_cached(self, stage: str) -> bool:
        """Check if valid validation cache exists for a stage."""
        return self.get_cached_validation(stage) is not None

    def clear_validation_cache(self) -> None:
        """Clear all validation cache entries."""
        if self.data:
            self.data.validation_cache = {}
            logger.debug("Validation cache cleared")

    def get_validation_cache_stats(self) -> Dict[str, Any]:
        """
        Get validation cache statistics for reporting.

        Returns:
            Dict with cache statistics including hit/miss counts.
        """
        if not self.data or not self.data.validation_cache:
            return {
                'total_cached': 0,
                'entries': {},
            }

        entries = {}
        for stage, cached in self.data.validation_cache.items():
            entries[stage] = {
                'is_valid': cached.get('is_valid'),
                'validated_at': cached.get('validated_at'),
                'config_hash': cached.get('config_hash', '')[:8] if cached.get('config_hash') else None,
            }

        return {
            'total_cached': len(entries),
            'entries': entries,
        }

    # === US-89-003: Escalation state persistence ===

    def save_escalation_state(self, escalation_manager) -> None:
        """
        Save escalation manager state to checkpoint.

        Persists the escalation tier state per keyword, enabling pipeline
        resume without losing bypass tier progress.

        Args:
            escalation_manager: EscalationManager instance to serialize.
        """
        if self.data is None:
            self.data = CheckpointData(
                created_at=datetime.now().isoformat(),
                config_hash=self.config_hash
            )

        # Delegate to EscalationManager's to_dict() method
        self.data.escalation_state = escalation_manager.to_dict()
        self.data.updated_at = datetime.now().isoformat()
        logger.debug(
            f"Saved escalation state for {len(escalation_manager.keyword_states)} keywords"
        )

    def load_escalation_state(self, escalation_manager, current_keywords: List[str] = None) -> None:
        """
        Load escalation manager state from checkpoint and restore to manager.

        Handles the edge case where keywords in saved state no longer exist
        in the current run - these are filtered out during restoration.

        Args:
            escalation_manager: EscalationManager instance to restore state into.
            current_keywords: Optional list of keywords in current run.
                If provided, keywords in saved state that are not in this list
                will be excluded from restoration.
        """
        if not self.data or not self.data.escalation_state:
            logger.debug("No escalation state in checkpoint to restore")
            return

        # Import here to avoid circular import
        from .downloader.escalation_manager import EscalationManager

        # Filter out keywords that don't exist in current run
        saved_state = self.data.escalation_state
        if current_keywords is not None:
            saved_keywords = set(saved_state.get('keyword_states', {}).keys())
            current_set = set(current_keywords)
            excluded_keywords = saved_keywords - current_set

            if excluded_keywords:
                logger.info(
                    f"Excluding {len(excluded_keywords)} keywords from escalation state "
                    f"that no longer exist in current run: {list(excluded_keywords)[:5]}..."
                )
                # Create filtered state dict
                filtered_state = dict(saved_state)
                filtered_keywords = {
                    k: v for k, v in saved_state.get('keyword_states', {}).items()
                    if k in current_set
                }
                filtered_state['keyword_states'] = filtered_keywords
                saved_state = filtered_state

        # Delegate to EscalationManager's from_dict() method
        # Note: This creates a new EscalationManager - we need to restore its state
        restored = EscalationManager.from_dict(
            data=saved_state,
            impersonation_manager=escalation_manager._impersonation_manager,
            extractor_args_config=escalation_manager._extractor_config,
            budget=escalation_manager._budget,
        )

        # Copy restored keyword states to the target manager
        escalation_manager._keyword_states = restored._keyword_states
        # Also restore metrics
        escalation_manager._metrics = restored._metrics

        restored_count = len(escalation_manager._keyword_states)
        logger.info(f"Restored escalation state for {restored_count} keywords")

    def has_escalation_state(self) -> bool:
        """Check if checkpoint contains saved escalation state."""
        return bool(self.data and self.data.escalation_state)

    # === US-109-010: Circuit breaker state persistence ===

    def save_circuit_breaker_state(self, circuit_breaker, name: str = "search") -> None:
        """
        Save circuit breaker state to checkpoint.

        Persists the circuit breaker state (is_open, consecutive_failures, total_trips)
        to enable faster recovery on pipeline resume.

        Args:
            circuit_breaker: CircuitBreaker instance to serialize
            name: Name of the circuit breaker (e.g., 'search', 'download')
        """
        if self.data is None:
            self.data = CheckpointData(
                created_at=datetime.now().isoformat(),
                config_hash=self.config_hash
            )

        # Get state from the circuit breaker
        state_dict = circuit_breaker.to_checkpoint_dict()
        state_dict['name'] = name
        state_dict['saved_at'] = datetime.now().isoformat()

        self.data.circuit_breaker_state[name] = state_dict
        self.data.updated_at = datetime.now().isoformat()
        logger.debug(
            f"Saved circuit breaker '{name}' state: "
            f"is_open={state_dict.get('is_open')}, "
            f"failures={state_dict.get('consecutive_failures')}, "
            f"trips={state_dict.get('total_trips')}"
        )

    def load_circuit_breaker_state(
        self,
        circuit_breaker,
        name: str = "search"
    ) -> dict:
        """
        Load circuit breaker state from checkpoint and restore to circuit breaker.

        Handles stale checkpoint detection - if checkpoint is >1 hour old,
        the circuit is restored to half-open state instead of fully closed.

        Args:
            circuit_breaker: CircuitBreaker instance to restore state into
            name: Name of the circuit breaker to load

        Returns:
            Dict with restoration info:
            {
                'restored': bool,
                'was_stale': bool,
                'restored_state': str,
                'checkpoint_age_hours': float
            }
        """
        result = {
            'restored': False,
            'was_stale': False,
            'restored_state': 'closed',
            'checkpoint_age_hours': 0.0
        }

        if not self.data or not self.data.circuit_breaker_state:
            logger.debug(f"No circuit breaker state in checkpoint to restore for '{name}'")
            return result

        saved_state = self.data.circuit_breaker_state.get(name)
        if not saved_state:
            logger.debug(f"No saved state for circuit breaker '{name}' in checkpoint")
            return result

        # Calculate checkpoint age
        checkpoint_age_seconds = 0.0
        try:
            saved_at = saved_state.get('saved_at')
            if saved_at:
                saved_time = datetime.fromisoformat(saved_at)
                age = datetime.now() - saved_time
                checkpoint_age_seconds = age.total_seconds()
                result['checkpoint_age_hours'] = checkpoint_age_seconds / 3600
        except Exception:
            pass

        # Restore state using the circuit breaker's from_checkpoint_dict method
        restore_result = circuit_breaker.from_checkpoint_dict(
            saved_state,
            checkpoint_age_seconds=checkpoint_age_seconds
        )

        result.update(restore_result)

        # Log restoration
        if result['restored']:
            logger.info(
                f"Circuit breaker '{name}' state restored: "
                f"state={result['restored_state']}, "
                f"stale={result['was_stale']}, "
                f"age={result['checkpoint_age_hours']:.1f}h"
            )

            # Track metrics for restoration frequency
            self._track_circuit_breaker_restoration(name, result)

        return result

    def _track_circuit_breaker_restoration(self, name: str, result: dict) -> None:
        """Track circuit breaker restoration metrics in health field."""
        if self.data is None:
            return

        # Initialize health tracking if not exists
        if 'restoration_metrics' not in self.data.circuit_breaker_health:
            self.data.circuit_breaker_health['restoration_metrics'] = {}

        metrics = self.data.circuit_breaker_health['restoration_metrics']

        if name not in metrics:
            metrics[name] = {
                'total_restorations': 0,
                'stale_restorations': 0,
                'by_state': {}
            }

        name_metrics = metrics[name]
        name_metrics['total_restorations'] = name_metrics.get('total_restorations', 0) + 1

        if result.get('was_stale'):
            name_metrics['stale_restorations'] = name_metrics.get('stale_restorations', 0) + 1

        state = result.get('restored_state', 'unknown')
        by_state = name_metrics.get('by_state', {})
        by_state[state] = by_state.get(state, 0) + 1
        name_metrics['by_state'] = by_state

        # Also track the most recent restoration
        name_metrics['last_restoration'] = {
            'timestamp': datetime.now().isoformat(),
            'state': state,
            'was_stale': result.get('was_stale', False),
            'checkpoint_age_hours': result.get('checkpoint_age_hours', 0.0)
        }

    def has_circuit_breaker_state(self, name: str = "search") -> bool:
        """Check if checkpoint contains saved circuit breaker state."""
        return bool(
            self.data and
            self.data.circuit_breaker_state and
            name in self.data.circuit_breaker_state
        )

    def get_circuit_breaker_state(self, name: str = "search") -> dict:
        """Get saved circuit breaker state from checkpoint without restoring."""
        if not self.data or not self.data.circuit_breaker_state:
            return {}
        return self.data.circuit_breaker_state.get(name, {})

    # === US-123-007: Cross-session rate limit state persistence ===

    def integrate_with_checkpoint(self, rate_limit_budget, stale_state_threshold: float = 3600.0) -> Dict:
        """
        Integrate rate limit state with checkpoint system.

        This method provides a unified interface for saving and loading rate limit
        state across pipeline sessions. It hooks into the checkpoint system to:
        - Save rate limit state when requested
        - Load and restore state from checkpoint
        - Handle stale state filtering

        Args:
            rate_limit_budget: RateLimitBudget instance to serialize/deserialize
            stale_state_threshold: Seconds after which state is considered stale

        Returns:
            Dict with operation result:
            {
                'operation': 'save' | 'load' | 'none',
                'restored': bool,
                'was_stale': bool,
                'state_age_seconds': float
            }
        """
        # Local import to avoid circular dependency
        from .downloader.rate_limit_budget import RateLimitBudget

        result = {
            'operation': 'none',
            'restored': False,
            'was_stale': False,
            'state_age_seconds': 0.0
        }

        # Check if we have existing state to restore
        if self.data and self.data.rate_limit_state:
            # Try to restore state
            restored_budget = RateLimitBudget.deserialize_rate_limit_state(
                self.data.rate_limit_state,
                stale_state_threshold=stale_state_threshold
            )

            if restored_budget is not None:
                # Copy restored state to the provided budget
                rate_limit_budget.rotations_used = restored_budget.rotations_used
                rate_limit_budget.vpn_switches_used = restored_budget.vpn_switches_used
                rate_limit_budget.backoff_time_spent = restored_budget.backoff_time_spent
                rate_limit_budget.keywords_rate_limited = restored_budget.keywords_rate_limited
                rate_limit_budget.last_escalation_level = restored_budget.last_escalation_level
                rate_limit_budget.successes = restored_budget.successes
                rate_limit_budget.failures = restored_budget.failures

                # Restore tier budgets if enabled
                if restored_budget.tier_isolation_enabled:
                    rate_limit_budget.tier_isolation_enabled = True
                    rate_limit_budget.tier_budget_manager = restored_budget.tier_budget_manager

                result['operation'] = 'load'
                result['restored'] = True

                # Calculate state age
                saved_at = self.data.rate_limit_state.get('saved_at')
                if saved_at:
                    import time
                    result['state_age_seconds'] = time.time() - saved_at
                    result['was_stale'] = result['state_age_seconds'] > stale_state_threshold

                logger.info(
                    f"Restored rate limit state from checkpoint: "
                    f"rotations={rate_limit_budget.rotations_used}, "
                    f"vpn_switches={rate_limit_budget.vpn_switches_used}, "
                    f"age={result['state_age_seconds']:.0f}s"
                )
            else:
                logger.debug("No valid rate limit state in checkpoint (stale or missing)")
        else:
            logger.debug("No rate limit state in checkpoint")

        return result

    def save_rate_limit_state(self, rate_limit_budget) -> None:
        """
        Save rate limit budget state to checkpoint.

        Persists the rate limit budget state (rotations, VPN switches, backoff time)
        to enable learning across pipeline sessions.

        Args:
            rate_limit_budget: RateLimitBudget instance to serialize.
        """
        if self.data is None:
            self.data = CheckpointData(
                created_at=datetime.now().isoformat(),
                config_hash=self.config_hash
            )

        # Delegate to RateLimitBudget's serialize method
        self.data.rate_limit_state = rate_limit_budget.serialize_rate_limit_state()
        self.data.updated_at = datetime.now().isoformat()
        logger.debug(
            f"Saved rate limit state: rotations={rate_limit_budget.rotations_used}, "
            f"vpn_switches={rate_limit_budget.vpn_switches_used}, "
            f"backoff_time={rate_limit_budget.backoff_time_spent:.1f}s"
        )

    def load_rate_limit_state(self, rate_limit_budget, stale_state_threshold: float = 3600.0) -> bool:
        """
        Load rate limit state from checkpoint and restore to budget.

        Handles stale state filtering - if state is older than stale_state_threshold,
        it will be skipped and the budget will not be modified.

        Args:
            rate_limit_budget: RateLimitBudget instance to restore state into.
            stale_state_threshold: Seconds after which state is considered stale.

        Returns:
            True if state was successfully restored, False otherwise.
        """
        if not self.data or not self.data.rate_limit_state:
            logger.debug("No rate limit state in checkpoint to restore")
            return False

        # Use the integrate method for restoration
        result = self.integrate_with_checkpoint(rate_limit_budget, stale_state_threshold)
        return result['restored']

    def has_rate_limit_state(self) -> bool:
        """Check if checkpoint contains saved rate limit state."""
        return bool(self.data and self.data.rate_limit_state)

    def get_checkpoint_stats(self) -> Dict[str, Any]:
        """
        US-130-002: Get comprehensive checkpoint usage statistics.

        Returns:
            Dict with checkpoint statistics:
            - total_size_bytes: Current checkpoint file size in bytes
            - save_count: Total number of save operations performed
            - avg_save_time_ms: Average save time in milliseconds (rolling average)
            - oldest_checkpoint: ISO timestamp of oldest checkpoint creation
            - newest_checkpoint: ISO timestamp of most recent checkpoint update
            - backup_count: Number of backup files currently existing
            - per_stage_sizes: Dict mapping stage field names to their data sizes in bytes
        """
        stats = {
            "total_size_bytes": 0,
            "save_count": self._stats.get("save_count", 0),
            "avg_save_time_ms": 0.0,
            "oldest_checkpoint": None,
            "newest_checkpoint": None,
            "backup_count": 0,
            "per_stage_sizes": {},
        }

        # Get current checkpoint size
        if self.checkpoint_path.exists():
            try:
                stats["total_size_bytes"] = self.checkpoint_path.stat().st_size
            except OSError:
                pass

        # Calculate average save time from tracked save times
        save_times = self._stats.get("save_times_ms", [])
        if save_times:
            stats["avg_save_time_ms"] = sum(save_times) / len(save_times)

        # Get oldest and newest checkpoint from data
        if self.data:
            if self.data.created_at:
                stats["oldest_checkpoint"] = self.data.created_at
            if self.data.updated_at:
                stats["newest_checkpoint"] = self.data.updated_at

        # Count existing backup files
        backup_count = 0
        for i in range(self._backup_count):
            backup_path = self._get_backup_path(i)
            if backup_path.exists():
                backup_count += 1
        stats["backup_count"] = backup_count

        # Calculate per-stage sizes from checkpoint data
        if self.data:
            data_dict = self.data.to_dict()
            for stage_field in STAGE_FIELD_MAP.values():
                if stage_field in data_dict and data_dict[stage_field]:
                    # Calculate approximate size of stage data
                    import json
                    stage_json = json.dumps(data_dict[stage_field], default=str)
                    stats["per_stage_sizes"][stage_field] = len(stage_json.encode('utf-8'))

        # US-130-008: Include signature stats
        stats["signature_stats"] = {
            "enabled": self._signature_enabled,
            "key_configured": bool(self._signature_key),
            "signatures_computed": self._signature_stats.get("signatures_computed", 0),
            "signatures_verified": self._signature_stats.get("signatures_verified", 0),
            "verification_failures": self._signature_stats.get("verification_failures", 0),
        }

        return stats

    def get_rate_limit_state(self) -> dict:
        """Get saved rate limit state from checkpoint without restoring."""
        if not self.data or not self.data.rate_limit_state:
            return {}
        return self.data.rate_limit_state

    def should_skip_stage(self, stage: str) -> bool:
        """Check if a stage should be skipped (already completed)"""
        if not self.data or not self.data.last_completed_stage:
            return False
        
        try:
            completed_idx = STAGE_ORDER.index(self.data.last_completed_stage)
            current_idx = STAGE_ORDER.index(stage)
            return current_idx <= completed_idx
        except ValueError:
            return False
    
    def clear(self):
        """Clear checkpoint (for fresh start)"""
        if self.checkpoint_path.exists():
            self.checkpoint_path.unlink()
        # US-51-007: Clear all rotated backups
        for backup_path in self._get_all_backup_paths():
            if backup_path.exists():
                backup_path.unlink()
        self.data = None
    
    def get_summary(self) -> str:
        """Get human-readable checkpoint summary"""
        if not self.data:
            return "No checkpoint found"
        
        lines = [
            f"Checkpoint from: {self.data.created_at[:19] if self.data.created_at else 'unknown'}",
            f"Last updated: {self.data.updated_at[:19] if self.data.updated_at else 'unknown'}",
            f"Last completed stage: {self.data.last_completed_stage or 'none'}",
        ]

        # US-115-002: Show dirty stage count for incremental saving stats
        if self._last_saved_data:
            dirty_count = len(self._get_dirty_stages())
            lines.append(f"Dirty stages (for incremental save): {dirty_count}")

        # US-115-003: Show compression stats if available
        if self._compression_stats:
            orig = self._compression_stats.get('original_size', 0)
            comp = self._compression_stats.get('compressed_size', 0)
            ratio = self._compression_stats.get('ratio', 1.0)
            lines.append(f"Compression: {orig} -> {comp} bytes ({ratio:.1%})")

        # Add stage summaries
        if self.data.analyze:
            kw_count = len(self.data.analyze.get('keywords', []))
            seg_count = self.data.analyze.get('segment_count', 0)
            lines.append(f"  • ANALYZE: {kw_count} keywords, {seg_count} segments")

        if self.data.video_search:
            vid_count = len(self.data.video_search.get('video_ids', []))
            lines.append(f"  • VIDEO_SEARCH: {vid_count} videos found")

        if self.data.caption:
            caption_count = self.data.caption.get('caption_count', 0)
            lines.append(f"  • CAPTION: {caption_count} captions fetched")

        if self.data.match:
            match_count = self.data.match.get('match_count', 0)
            avg_conf = self.data.match.get('avg_confidence', 0)
            lines.append(f"  • MATCH: {match_count} matches, {avg_conf:.1%} avg confidence")

        # US-49-012: Show stage metrics summary if available
        if self.data.stage_metrics:
            for stage_name, m in self.data.stage_metrics.items():
                processed = m.get('items_processed', 0)
                failed = m.get('items_failed', 0)
                duration = m.get('duration_seconds', 0.0)
                lines.append(
                    f"  • {stage_name} metrics: "
                    f"{processed} processed, {failed} failed, {duration:.1f}s"
                )

        # US-79-012: Show transcription metrics if available
        if self.data.transcription_metrics:
            tm = self.data.transcription_metrics
            lines.append(
                f"  • Transcription: {tm.get('transcribed_videos', 0)} transcribed, "
                f"{tm.get('cached_videos', 0)} cached, "
                f"{tm.get('failed_videos', 0)} failed, "
                f"{tm.get('average_speed_ratio', 0):.1f}x realtime"
            )

        return "\n".join(lines)


def validate_checkpoint(checkpoint_path: str) -> Dict[str, Any]:
    """
    US-120-007: Standalone checkpoint validation function.

    Validates checkpoint integrity without requiring a CheckpointManager instance.
    Checks: valid JSON, required fields, stage order correctness, and age.

    Args:
        checkpoint_path: Path to checkpoint.json file

    Returns:
        Dict with validation results:
        - is_valid: bool - Overall validity
        - json_valid: bool - JSON parsing succeeded
        - schema_valid: bool - Required fields present
        - stage_order_valid: bool - Stage order is correct
        - is_stale: bool - Checkpoint is older than 24 hours
        - age_hours: float - Age of checkpoint in hours
        - issues: List[str] - List of validation issues
        - warnings: List[str] - List of warnings
    """
    import gzip

    result = {
        'is_valid': False,
        'json_valid': False,
        'schema_valid': False,
        'stage_order_valid': False,
        'is_stale': False,
        'age_hours': 0.0,
        'issues': [],
        'warnings': [],
        'version': '',
        'last_stage': '',
        'created_at': '',
    }

    path = Path(checkpoint_path)

    # Check file exists
    if not path.exists():
        result['issues'].append(f"Checkpoint file does not exist: {checkpoint_path}")
        return result

    # Read and parse JSON
    try:
        with open(path, 'rb') as f:
            content = f.read()

        # Handle gzip compression
        if len(content) >= 2 and content[0] == 0x1f and content[1] == 0x8b:
            content = gzip.decompress(content)

        data = json.loads(content.decode('utf-8'))
        result['json_valid'] = True
    except json.JSONDecodeError as e:
        result['issues'].append(f"Invalid JSON: {e}")
        return result
    except Exception as e:
        result['issues'].append(f"Failed to read file: {e}")
        return result

    # Check required fields
    required_fields = ['version', 'last_completed_stage', 'created_at']
    for field in required_fields:
        if field not in data:
            result['issues'].append(f"Missing required field: {field}")

    if result['issues']:
        return result

    result['schema_valid'] = True
    result['version'] = data.get('version', 'unknown')
    result['last_stage'] = data.get('last_completed_stage', '')
    result['created_at'] = data.get('created_at', '')

    # Validate version compatibility
    version = data.get('version', '')
    if version and version != CURRENT_CHECKPOINT_VERSION:
        migratable = ['0.9', '1.0', '2.0', '2.1']
        if version not in migratable:
            result['issues'].append(f"Incompatible version: {version}")
            result['schema_valid'] = False

    # Validate stage order
    last_stage = data.get('last_completed_stage', '')
    if last_stage:
        if last_stage not in STAGE_ORDER:
            result['issues'].append(f"Invalid stage name: {last_stage}")
        else:
            # Check that stages are in order
            last_idx = STAGE_ORDER.index(last_stage)
            completed_stages = data.get('completed_stages', [])

            for stage in completed_stages:
                if stage not in STAGE_ORDER:
                    result['warnings'].append(f"Unknown stage in completed_stages: {stage}")
                    continue

                stage_idx = STAGE_ORDER.index(stage)
                if stage_idx > last_idx:
                    result['issues'].append(
                        f"Stage order violation: {stage} (index {stage_idx}) comes after "
                        f"{last_stage} (index {last_idx})"
                    )

            result['stage_order_valid'] = len([i for i in result['issues'] if 'order violation' in i]) == 0

    # Calculate checkpoint age
    try:
        if result['created_at']:
            created = datetime.fromisoformat(result['created_at'].replace('Z', '+00:00'))
            now = datetime.now(created.tzinfo) if created.tzinfo else datetime.now()
            age = now - created
            result['age_hours'] = age.total_seconds() / 3600
            result['is_stale'] = result['age_hours'] > 24.0

            if result['is_stale']:
                result['warnings'].append(f"Checkpoint is stale: {result['age_hours']:.1f} hours old (> 24h)")
    except Exception as e:
        result['warnings'].append(f"Could not parse created_at timestamp: {e}")

    # Determine overall validity
    result['is_valid'] = (
        result['json_valid'] and
        result['schema_valid'] and
        result['stage_order_valid'] and
        len(result['issues']) == 0
    )

    return result


def format_resume_prompt(checkpoint: CheckpointManager) -> str:
    """Format a user-friendly resume prompt"""
    validation = checkpoint.validate()
    
    lines = [
        "",
        "=" * 60,
        "  CHECKPOINT FOUND",
        "=" * 60,
        "",
        checkpoint.get_summary(),
        ""
    ]
    
    if validation['warnings']:
        lines.append("⚠ Warnings:")
        for w in validation['warnings']:
            lines.append(f"  • {w}")
        lines.append("")
    
    if validation['resume_from']:
        lines.append(f"Resume from: {validation['resume_from']}")
    
    lines.extend([
        "",
        "Options:",
        "  [R] Resume from checkpoint",
        "  [F] Fresh start (delete checkpoint)",
        "  [Q] Quit",
        ""
    ])
    
    return "\n".join(lines)


# KeywordManager and format_keyword_prompt are now in src.keywords.manager
# and re-exported at the top of this file for backward compatibility
