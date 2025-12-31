"""
Checkpoint Manager for Pipeline Resume Functionality

Saves pipeline state after each stage so runs can be resumed if interrupted.
Also manages saved keyword presets for reproducible runs.
"""

import json
import hashlib
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field, asdict
import logging

logger = logging.getLogger(__name__)


# Stage order for resume logic
STAGE_ORDER = [
    "ANALYZE",
    "ENTITY_IMAGES", 
    "ENTITY_VIDEOS",
    "DOWNLOAD",
    "STOCK",
    "REMIX",
    "TRANSCRIBE",
    "MATCH",
    "OUTPUT"
]


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
    version: str = "1.0"
    created_at: str = ""
    updated_at: str = ""
    last_completed_stage: str = ""
    config_hash: str = ""
    voiceover_path: str = ""
    voiceover_hash: str = ""
    
    # Stage outputs
    analyze: Dict[str, Any] = field(default_factory=dict)
    entity_images: Dict[str, Any] = field(default_factory=dict)
    entity_videos: Dict[str, Any] = field(default_factory=dict)
    download: Dict[str, Any] = field(default_factory=dict)
    remix: Dict[str, Any] = field(default_factory=dict)
    transcribe: Dict[str, Any] = field(default_factory=dict)
    match: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> 'CheckpointData':
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class CheckpointManager:
    """Manages pipeline checkpoints for resume functionality"""
    
    CHECKPOINT_FILE = "checkpoint.json"
    CHECKPOINT_BACKUP = "checkpoint.backup.json"
    
    def __init__(self, project_dir: Path, config_hash: str = ""):
        self.project_dir = Path(project_dir)
        self.checkpoint_path = self.project_dir / self.CHECKPOINT_FILE
        self.backup_path = self.project_dir / self.CHECKPOINT_BACKUP
        self.config_hash = config_hash
        self.data: Optional[CheckpointData] = None
        
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
        """Load existing checkpoint"""
        if not self.exists():
            return None
            
        try:
            with open(self.checkpoint_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.data = CheckpointData.from_dict(data)
            return self.data
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}")
            return None
    
    def save(self, stage: str, stage_data: Dict[str, Any] = None):
        """Save checkpoint after stage completion"""
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
            if hasattr(self.data, stage_key):
                setattr(self.data, stage_key, stage_data)
        
        # Atomic save: write to temp, then rename
        self._atomic_save()
        
    def _atomic_save(self):
        """Atomically save checkpoint (write temp, then rename)"""
        temp_path = self.checkpoint_path.with_suffix('.tmp')
        try:
            # Backup existing checkpoint
            if self.checkpoint_path.exists():
                shutil.copy2(self.checkpoint_path, self.backup_path)
            
            # Write to temp file
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(self.data.to_dict(), f, indent=2, default=str)
            
            # Atomic rename
            temp_path.replace(self.checkpoint_path)
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            if temp_path.exists():
                temp_path.unlink()
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
        except:
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
        
        # Verify downloaded files still exist
        if self.data.download and 'video_paths' in self.data.download:
            missing = []
            for vp in self.data.download['video_paths']:
                if not Path(vp).exists():
                    missing.append(vp)
            if missing:
                result['warnings'].append(
                    f"{len(missing)} downloaded videos are missing from disk"
                )
        
        return result
    
    def get_stage_data(self, stage: str) -> Dict[str, Any]:
        """Get saved data for a specific stage"""
        if not self.data:
            return {}
        stage_key = stage.lower()
        return getattr(self.data, stage_key, {})
    
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
        
        if self.data.download:
            vid_count = len(self.data.download.get('video_paths', []))
            lines.append(f"  • DOWNLOAD: {vid_count} videos")
        
        if self.data.transcribe:
            trans_count = self.data.transcribe.get('transcribed_count', 0)
            embed_count = self.data.transcribe.get('embedding_count', 0)
            lines.append(f"  • TRANSCRIBE: {trans_count} transcribed, {embed_count} embeddings")
        
        if self.data.match:
            match_count = self.data.match.get('match_count', 0)
            avg_conf = self.data.match.get('avg_confidence', 0)
            lines.append(f"  • MATCH: {match_count} matches, {avg_conf:.1%} avg confidence")
        
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
            except:
                pass
        
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
