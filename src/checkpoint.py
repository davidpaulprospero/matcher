"""
Checkpoint Manager for Pipeline Resume Functionality

Saves pipeline state after each stage so runs can be resumed if interrupted.
Keyword preset management has been extracted to src/keywords/manager.py.
"""

import json
import hashlib
import shutil
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List, TYPE_CHECKING
from dataclasses import dataclass, field, asdict
import logging

if TYPE_CHECKING:
    from .config import Config

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

# Current checkpoint version — single source of truth for write and validation
CURRENT_CHECKPOINT_VERSION = "2.0"

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


@dataclass
class CheckpointData:
    """Data saved at each checkpoint"""
    version: str = CURRENT_CHECKPOINT_VERSION
    created_at: str = ""
    updated_at: str = ""
    last_completed_stage: str = ""
    config_hash: str = ""
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

    def validate(self) -> List[str]:
        """
        Validate internal consistency of checkpoint data.

        Checks that all stages before last_completed_stage have non-empty data.
        Returns a list of warning messages (empty if fully consistent).
        """
        warnings = []

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
            if not stage_data:
                warnings.append(
                    f"Stage {stage_name} is before last_completed_stage "
                    f"({self.last_completed_stage}) but has no data"
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

    def __init__(self, project_dir: Path, config_hash: str = "", config: 'Config' = None):
        self.project_dir = Path(project_dir)
        self.checkpoint_path = self.project_dir / self.CHECKPOINT_FILE
        self.backup_path = self.project_dir / self.CHECKPOINT_BACKUP
        self.config_hash = config_hash
        self.data: Optional[CheckpointData] = None
        self._config = config  # Store config for stage restore
        # US-51-007: Backup rotation count from config (default 3)
        self._backup_count = getattr(
            getattr(config, 'pipeline', None), 'checkpoint_backup_count', 3
        ) if config else 3
        # US-85-003: Interval-gated backup rotation for intermediate saves
        self._min_rotation_interval = getattr(
            getattr(config, 'pipeline', None), 'min_rotation_interval_seconds', 60
        ) if config else 60
        self._last_rotation_time: float = 0.0  # epoch seconds of last rotation
        
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
        """
        start_time = time.perf_counter()

        if not self.exists():
            return None

        # Try main checkpoint first
        main_data = self._try_load_file(self.checkpoint_path)

        # Try backup if it exists
        backup_data = None
        if self.backup_path.exists():
            backup_data = self._try_load_file(self.backup_path)

        # Decide which checkpoint to use based on validity and timestamps
        data = self._select_checkpoint(main_data, backup_data)

        # US-51-007: If both main and primary backup failed, try rotated backups
        if data is None:
            for i in range(1, self._backup_count):
                rotated_path = self._get_backup_path(i)
                if rotated_path.exists():
                    rotated_data = self._try_load_file(rotated_path)
                    if rotated_data is not None:
                        logger.warning(
                            f"Restored checkpoint from rotated backup: {rotated_path.name}"
                        )
                        # Restore to main
                        try:
                            shutil.copy2(rotated_path, self.checkpoint_path)
                        except Exception as e:
                            logger.warning(f"Could not restore rotated backup to main: {e}")
                        data = rotated_data
                        break

        if data is None:
            return None

        # Validate required fields
        if not self._validate_checkpoint_data(data):
            logger.warning("Checkpoint validation failed - data may be incomplete")
            # Continue with partial data rather than failing completely

        # Validate internal consistency (stages before last_completed have data)
        consistency_warnings = data.validate()
        for warning in consistency_warnings:
            logger.warning(f"Checkpoint consistency: {warning}")

        self.data = data

        # US-79-012: Log previous transcription metrics for cross-run comparison
        self.log_previous_transcription_metrics()

        # Log load time
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        logger.info(f"Checkpoint loaded in {elapsed_ms:.1f}ms")

        return self.data

    def _try_load_file(self, path: Path) -> Optional[CheckpointData]:
        """Try to load and parse a checkpoint file"""
        try:
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

            # US-40-010: Validate checkpoint integrity - check required top-level keys
            # Required keys: last_completed_stage, timestamp (created_at or updated_at)
            # The "state" is distributed across stage fields (analyze, video_search, etc.)
            missing_keys = self._validate_checkpoint_structure(data)
            if missing_keys:
                for key in missing_keys:
                    logger.warning(f"Checkpoint missing required key: {key}")

            # Migrate checkpoint if needed
            return self._migrate_checkpoint_if_needed(data)

        except json.JSONDecodeError as e:
            logger.warning(f"Checkpoint JSON parse error in {path}: {e}")
            return None
        except Exception as e:
            logger.warning(f"Failed to load checkpoint from {path}: {e}")
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
            # Both invalid
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

    def _migrate_checkpoint_if_needed(self, data: dict) -> CheckpointData:
        """
        Migrate old checkpoint format to new format.

        Delegates to CheckpointMigrator for versioned upgrades:
        - Version 0.9 -> 1.0: Uppercase stage keys to lowercase
        - Version 1.0 -> 2.0: 13-stage to 7-stage simplified pipeline
        """
        from .checkpoint_migrator import CheckpointMigrator

        migrator = CheckpointMigrator()
        version = data.get('version', '0.9')

        if migrator.needs_migration(data, CURRENT_CHECKPOINT_VERSION):
            logger.info(f"Migrating checkpoint from v{version} to v{CURRENT_CHECKPOINT_VERSION}")

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

    def _validate_checkpoint_data(self, data: CheckpointData) -> bool:
        """
        Validate that checkpoint data has required fields.

        Returns True if valid, False if there are issues (but data is usable).
        """
        issues = []

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

        if issues:
            for issue in issues:
                logger.debug(f"Checkpoint validation: {issue}")
            return False

        return True

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

    def save(self, stage: str, stage_data: Dict[str, Any] = None,
             stage_metrics: Dict[str, Any] = None):
        """Save checkpoint after stage completion.

        Args:
            stage: Stage name (e.g., 'DOWNLOAD_SEGMENTS')
            stage_data: Stage output data dict
            stage_metrics: Optional serialized StageMetrics dict (US-49-012).
                           Persisted under stage_metrics.<STAGE_NAME> in checkpoint.
        """
        if self.data is None:
            self.data = CheckpointData(
                created_at=datetime.now().isoformat(),
                config_hash=self.config_hash
            )

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
        if stage_metrics:
            self.data.stage_metrics[stage] = stage_metrics

        # Atomic save: write to temp, then rename
        # Stage completions always rotate backups (force_rotate=True)
        self._atomic_save(force_rotate=True)

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

        logger.debug(f"Saving intermediate checkpoint for {stage}")
        # Intermediate saves respect the rotation interval (force_rotate=False)
        self._atomic_save(force_rotate=False)

    def save_stage_timing_summary(
        self,
        stage_timings: Dict[str, float],
        total_duration: float,
        skipped_stages: set
    ):
        """
        Persist pipeline timing summary to checkpoint stage_metrics.

        Stores per-stage duration and a '_pipeline' entry with total duration
        so timing data is available for post-run analysis.

        Args:
            stage_timings: Map of stage_name -> elapsed seconds for stages that ran.
            total_duration: Total pipeline wall-clock duration in seconds.
            skipped_stages: Set of stage names that were restored from checkpoint.
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

        # Store pipeline-level timing summary
        self.data.stage_metrics['_pipeline'] = {
            'total_duration_seconds': total_duration,
            'stages_run': list(stage_timings.keys()),
            'stages_skipped': list(skipped_stages),
        }

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
        """
        try:
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

    def _atomic_save(self, force_rotate: bool = True):
        """Atomically save checkpoint (write temp, then rename).

        Args:
            force_rotate: If True, always rotate backups (stage completions).
                If False, only rotate when min_rotation_interval has elapsed
                (intermediate saves during long-running stages).
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

            # Write to temp file
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(self.data.to_dict(), f, indent=2, default=str)

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

            # Log save time
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            logger.debug(f"Checkpoint saved in {elapsed_ms:.1f}ms")

        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except:
                    pass
            raise
    
    def set_voiceover(self, voiceover_path: str):
        """Set voiceover info for validation on resume"""
        if self.data is None:
            self.data = CheckpointData(
                created_at=datetime.now().isoformat(),
                config_hash=self.config_hash
            )
        
        self.data.voiceover_path = str(voiceover_path)
        self.data.voiceover_hash = self._hash_file(voiceover_path)
    
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
    
    def get_stage_data(self, stage: str) -> Dict[str, Any]:
        """Get saved data for a specific stage"""
        if not self.data:
            return {}
        stage_key = stage.lower()
        return getattr(self.data, stage_key, {})

    def get_chapter_data(self) -> Dict[str, Any]:
        """US-71-009: Get chapter/listicle detection data from checkpoint.

        Returns dict with 'chapters' and 'listicle_groups' lists.
        Returns empty dict if no chapter data is available (backward compat).
        """
        if not self.data:
            return {}
        return self.data.chapter_data or {}

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
