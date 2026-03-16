"""
Keyword Manager for Saved Keyword Presets

Manages saved keyword presets for reproducible pipeline runs.
Extracted from checkpoint.py to separate keyword management from checkpoint persistence.
"""

import json
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field, asdict
import logging

logger = logging.getLogger(__name__)


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
