"""
Checkpoint Manager for Pipeline Resume Functionality

Saves pipeline state after each stage so runs can be resumed if interrupted.
Also manages saved keyword presets for reproducible runs.
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

# Legacy stage names for checkpoint migration
LEGACY_STAGES = [
    "DOWNLOAD", "STOCK", "BROLL_DOWNLOAD", "REMIX",
    "TRANSCRIBE", "SCENE_DETECTION", "BROLL_MATCH"
]

# Mapping from stage name to CheckpointData field name
# OUTPUT has no checkpoint field (it's the terminal stage)
STAGE_FIELD_MAP = {
    "ANALYZE": "analyze",
    "VIDEO_SEARCH": "video_search",
    "CAPTION": "caption",
    "MATCH": "match",
    "ITERATIVE_MATCH": "iterative_match",
    "DOWNLOAD_SEGMENTS": "download_segments",
}


@dataclass
class SavedKeywords:
    """Saved keyword preset for reproducible runs"""
    name: str = ""
    created_at: str = ""
    voiceover_hash: str = ""
    keywords: List[str] = field(default_factory=list)
    topic_context: str = ""
    entities: List[Dict[str, Any]] = field(default_factory=list)
    num_keywords: int = 0
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> 'SavedKeywords':
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class CheckpointData:
    """Data saved at each checkpoint"""
    version: str = "2.0"  # Bumped for simplified pipeline
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

    # Stage metrics for pipeline observability (US-49-012)
    # Maps stage name -> serialized StageMetrics dict
    stage_metrics: Dict[str, Any] = field(default_factory=dict)

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

        Handles version upgrades transparently, including:
        - Version 0.9 -> 1.0: Uppercase stage keys to lowercase
        - Version 1.0 -> 2.0: 13-stage to 7-stage simplified pipeline
        """
        version = data.get('version', '0.9')

        # Map old stages to new (for v0.9/v1.0 -> v2.0 migration)
        # DOWNLOAD stage data becomes video_search (search results can be preserved)
        # Other removed stages are dropped
        legacy_stage_mapping = {
            'DOWNLOAD': 'video_search',  # Closest equivalent
            'download': 'video_search',
        }

        # Map for last_completed_stage migration
        stage_remap = {
            'DOWNLOAD': 'VIDEO_SEARCH',
            'STOCK': 'VIDEO_SEARCH',
            'BROLL_DOWNLOAD': 'VIDEO_SEARCH',
            'REMIX': 'CAPTION',
            'TRANSCRIBE': 'CAPTION',
            'SCENE_DETECTION': 'MATCH',
            'BROLL_MATCH': 'MATCH',
        }

        if version in ('0.9', '1.0'):
            logger.info(f"Migrating checkpoint from v{version} to v2.0 (simplified pipeline)")

            # Build new CheckpointData
            migrated_data = {
                'version': '2.0',
                'created_at': data.get('created_at', datetime.now().isoformat()),
                'updated_at': data.get('updated_at', datetime.now().isoformat()),
                'config_hash': data.get('config_hash', ''),
                'voiceover_path': data.get('voiceover_path', ''),
                'voiceover_hash': data.get('voiceover_hash', '')
            }

            # Migrate last_completed_stage
            old_stage = data.get('last_completed_stage', '')
            if old_stage in stage_remap:
                migrated_data['last_completed_stage'] = stage_remap[old_stage]
                logger.info(f"Remapped stage {old_stage} -> {stage_remap[old_stage]}")
            elif old_stage in STAGE_ORDER:
                migrated_data['last_completed_stage'] = old_stage
            else:
                # Unknown stage, reset to beginning
                migrated_data['last_completed_stage'] = ''
                if old_stage:
                    logger.warning(f"Unknown stage '{old_stage}' in old checkpoint, resetting")

            # Copy existing stage data for stages that still exist
            for stage_key in ['analyze', 'caption', 'match', 'iterative_match', 'download_segments']:
                stage_data = data.get(stage_key.upper()) or data.get(stage_key) or {}
                migrated_data[stage_key] = stage_data

            # Migrate DOWNLOAD -> video_search (extract video IDs from downloaded_videos)
            download_data = data.get('DOWNLOAD') or data.get('download') or {}
            if download_data:
                video_ids = []
                # Extract video IDs from old downloaded_videos list
                for vid in download_data.get('downloaded_videos', []):
                    if isinstance(vid, dict):
                        # Try to extract video ID from URL
                        url = vid.get('url', '')
                        if 'youtube.com' in url or 'youtu.be' in url:
                            import re
                            match = re.search(r'(?:v=|/)([a-zA-Z0-9_-]{11})', url)
                            if match:
                                video_ids.append(match.group(1))
                migrated_data['video_search'] = {
                    'video_ids': video_ids,
                    'migrated_from_download': True,
                }

            # Convert to CheckpointData
            migrated = CheckpointData.from_dict(migrated_data)

            # Save migrated checkpoint
            try:
                self.data = migrated
                self._atomic_save()
                logger.info("Migrated checkpoint to v2.0 saved successfully")
            except Exception as e:
                logger.warning(f"Could not save migrated checkpoint: {e}")

            return migrated

        # Already v2.0 - just convert to CheckpointData
        return CheckpointData.from_dict(data)

    def _validate_checkpoint_data(self, data: CheckpointData) -> bool:
        """
        Validate that checkpoint data has required fields.

        Returns True if valid, False if there are issues (but data is usable).
        """
        issues = []

        # Check version compatibility
        if data.version and data.version != "1.0":
            issues.append(f"Checkpoint version {data.version} may not be fully compatible")

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
            if stage_key not in CheckpointData.__dataclass_fields__:
                logger.warning(
                    f"save(): stage_key '{stage_key}' (from stage '{stage}') "
                    f"does not map to a CheckpointData field — data will not be persisted"
                )
            elif hasattr(self.data, stage_key):
                setattr(self.data, stage_key, stage_data)

        # US-49-012: Persist stage metrics for pipeline observability
        if stage_metrics:
            self.data.stage_metrics[stage] = stage_metrics

        # Atomic save: write to temp, then rename
        self._atomic_save()

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
            if stage_key not in CheckpointData.__dataclass_fields__:
                logger.warning(
                    f"save_intermediate(): stage_key '{stage_key}' (from stage '{stage}') "
                    f"does not map to a CheckpointData field — data will not be persisted"
                )
            elif hasattr(self.data, stage_key):
                setattr(self.data, stage_key, stage_data)

        logger.debug(f"Saving intermediate checkpoint for {stage}")
        self._atomic_save()

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

        self._atomic_save()
        logger.info(f"Marked {stage} as incomplete - will be re-run")

    def _atomic_save(self):
        """Atomically save checkpoint (write temp, then rename)"""
        start_time = time.perf_counter()
        temp_path = self.checkpoint_path.with_suffix('.tmp')
        try:
            # Backup existing checkpoint
            if self.checkpoint_path.exists():
                try:
                    shutil.copy2(self.checkpoint_path, self.backup_path)
                except Exception as e:
                    logger.warning(f"Failed to backup checkpoint: {e}")

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
        if self.backup_path.exists():
            self.backup_path.unlink()
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


class KeywordManager:
    """Manages saved keyword presets for reproducible runs"""
    
    KEYWORDS_FILE = "saved_keywords.json"
    
    def __init__(self, project_dir: Path):
        self.project_dir = Path(project_dir)
        self.keywords_path = self.project_dir / self.KEYWORDS_FILE
        self.presets: Dict[str, SavedKeywords] = {}
        self._load()
    
    def _load(self):
        """Load saved keyword presets"""
        if not self.keywords_path.exists():
            return
        
        try:
            with open(self.keywords_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Handle both old format (single preset) and new format (multiple presets)
            if 'presets' in data:
                for name, preset_data in data['presets'].items():
                    self.presets[name] = SavedKeywords.from_dict(preset_data)
            elif 'keywords' in data:
                # Old format - single preset, convert to new format
                preset = SavedKeywords.from_dict(data)
                preset.name = "default"
                self.presets["default"] = preset
        except Exception as e:
            logger.warning(f"Failed to load saved keywords: {e}")
    
    def _save(self):
        """Save keyword presets to disk"""
        try:
            data = {
                'version': '1.0',
                'updated_at': datetime.now().isoformat(),
                'presets': {name: preset.to_dict() for name, preset in self.presets.items()}
            }
            with open(self.keywords_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save keywords: {e}")
    
    def save_keywords(self, keywords: List[str], topic_context: str = "", 
                      entities: List[Dict] = None, name: str = None,
                      voiceover_path: str = None) -> str:
        """
        Save keywords as a preset.
        
        Args:
            keywords: List of keywords
            topic_context: Detected topic
            entities: Extracted entities
            name: Preset name (auto-generated if None)
            voiceover_path: Path to voiceover file (for hash)
            
        Returns:
            Name of saved preset
        """
        if name is None:
            # Auto-generate name from timestamp
            name = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        voiceover_hash = ""
        if voiceover_path:
            try:
                with open(voiceover_path, 'rb') as f:
                    voiceover_hash = hashlib.md5(f.read()).hexdigest()[:16]
            except (OSError, IOError) as e:
                # File not accessible - proceed without hash
                logger.debug(f"Could not hash voiceover file {voiceover_path}: {e}")
        
        preset = SavedKeywords(
            name=name,
            created_at=datetime.now().isoformat(),
            voiceover_hash=voiceover_hash,
            keywords=keywords,
            topic_context=topic_context,
            entities=entities or [],
            num_keywords=len(keywords)
        )
        
        self.presets[name] = preset
        self._save()
        
        return name
    
    def get_preset(self, name: str = None) -> Optional[SavedKeywords]:
        """Get a saved keyword preset by name, or the most recent one"""
        if not self.presets:
            return None
        
        if name:
            # Return specific preset or None if not found
            return self.presets.get(name)
        
        # Return most recent preset
        sorted_presets = sorted(
            self.presets.values(),
            key=lambda p: p.created_at,
            reverse=True
        )
        return sorted_presets[0] if sorted_presets else None
    
    def get_latest(self) -> Optional[SavedKeywords]:
        """Get the most recently saved keyword preset"""
        return self.get_preset()
    
    def list_presets(self) -> List[SavedKeywords]:
        """List all saved presets, newest first"""
        return sorted(
            self.presets.values(),
            key=lambda p: p.created_at,
            reverse=True
        )
    
    def delete_preset(self, name: str) -> bool:
        """Delete a preset by name"""
        if name in self.presets:
            del self.presets[name]
            self._save()
            return True
        return False
    
    def has_presets(self) -> bool:
        """Check if any presets exist"""
        return len(self.presets) > 0
    
    def get_summary(self) -> str:
        """Get human-readable summary of saved presets"""
        if not self.presets:
            return "No saved keyword presets"
        
        lines = [f"Saved keyword presets ({len(self.presets)}):"]
        
        for preset in self.list_presets()[:5]:  # Show up to 5
            created = preset.created_at[:16].replace('T', ' ') if preset.created_at else 'unknown'
            kw_preview = ", ".join(preset.keywords[:3])
            if len(preset.keywords) > 3:
                kw_preview += f"... (+{len(preset.keywords)-3} more)"
            lines.append(f"  • [{preset.name}] {created}")
            lines.append(f"    Keywords: {kw_preview}")
            if preset.topic_context:
                lines.append(f"    Topic: {preset.topic_context[:50]}...")
        
        if len(self.presets) > 5:
            lines.append(f"  ... and {len(self.presets) - 5} more")
        
        return "\n".join(lines)


def format_keyword_prompt(keyword_manager: KeywordManager) -> str:
    """Format a user-friendly keyword selection prompt"""
    lines = [
        "",
        "=" * 60,
        "  SAVED KEYWORDS FOUND",
        "=" * 60,
        "",
        keyword_manager.get_summary(),
        "",
        "Options:",
        "  [U] Use saved keywords (most recent)",
        "  [L] List all saved presets",
        "  [N] Generate new keywords",
        ""
    ]
    
    return "\n".join(lines)
