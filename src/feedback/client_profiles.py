"""Client profile system for cross-project learning.

Manages client-specific preferences, rejections, and evolved presets.
Stored at ~/.matcher_rejections/client_<name>/ for each client.

Client profiles track:
- Learned content preferences (duration, style)
- Client-specific blacklists (channels, keywords)
- Evolved presets based on project history
- Quality thresholds
"""

from __future__ import annotations

import json
import logging
import statistics
from collections import Counter
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import yaml

logger = logging.getLogger(__name__)

# Global client profiles directory
CLIENT_PROFILES_DIR = Path.home() / ".matcher_rejections"


@dataclass
class ContentPreferences:
    """Learned content preferences for a client."""

    content_style: str = "documentary"  # documentary, raw, stock_footage
    preferred_duration_min: float = 30.0  # seconds
    preferred_duration_max: float = 180.0  # seconds
    avoid_trainers: bool = True
    avoid_movies: bool = True
    avoid_music: bool = True
    prefer_silent_broll: bool = False


@dataclass
class QualityThresholds:
    """Quality thresholds for filtering content."""

    min_channel_score: float = 0.3
    min_llm_relevance: float = 0.6
    min_embedding_score: float = 0.4
    max_videos_per_keyword: int = 12


@dataclass
class EvolvedPreset:
    """Auto-generated preset based on project history.

    Created by analyzing which videos were accepted vs rejected
    across all projects for a client.
    """

    auto_blacklist_channels: List[str] = field(default_factory=list)
    auto_blacklist_keywords: List[str] = field(default_factory=list)
    preferred_channels: List[str] = field(default_factory=list)
    preferred_duration_range: Tuple[float, float] = (30.0, 180.0)
    avg_accepted_duration: float = 60.0
    rejection_patterns: Dict[str, int] = field(default_factory=dict)
    generated_at: str = ""
    projects_analyzed: int = 0

    def __post_init__(self):
        if not self.generated_at:
            self.generated_at = datetime.now().isoformat()


@dataclass
class ClientProfile:
    """Profile for a specific client with learned preferences.

    Stored at ~/.matcher_rejections/client_<name>/profile.yaml
    """

    client_id: str
    display_name: str = ""
    created: str = ""
    updated: str = ""

    # Learned preferences
    preferences: ContentPreferences = field(default_factory=ContentPreferences)
    thresholds: QualityThresholds = field(default_factory=QualityThresholds)

    # Client-specific blacklists (in addition to global)
    blacklist_channels: Set[str] = field(default_factory=set)
    blacklist_keywords: Set[str] = field(default_factory=set)

    # Whitelist (override global rejections)
    whitelist_channels: Set[str] = field(default_factory=set)

    # Evolved preset (auto-generated)
    evolved_preset: Optional[EvolvedPreset] = None

    # Project tracking
    projects: List[str] = field(default_factory=list)  # Project paths
    total_videos_accepted: int = 0
    total_videos_rejected: int = 0

    def __post_init__(self):
        if not self.created:
            self.created = datetime.now().isoformat()
        if not self.display_name:
            self.display_name = self.client_id.title()
        # Convert lists to sets if needed
        if isinstance(self.blacklist_channels, list):
            self.blacklist_channels = set(self.blacklist_channels)
        if isinstance(self.blacklist_keywords, list):
            self.blacklist_keywords = set(self.blacklist_keywords)
        if isinstance(self.whitelist_channels, list):
            self.whitelist_channels = set(self.whitelist_channels)
        # Convert dict to dataclass if needed
        if isinstance(self.preferences, dict):
            self.preferences = ContentPreferences(**self.preferences)
        if isinstance(self.thresholds, dict):
            self.thresholds = QualityThresholds(**self.thresholds)
        if isinstance(self.evolved_preset, dict):
            self.evolved_preset = EvolvedPreset(**self.evolved_preset)

    def get_profile_dir(self) -> Path:
        """Get the directory for this client's profile."""
        return CLIENT_PROFILES_DIR / f"client_{self.client_id}"

    def is_channel_blacklisted(self, channel_name: str) -> bool:
        """Check if a channel is blacklisted for this client."""
        return channel_name.lower() in {c.lower() for c in self.blacklist_channels}

    def is_channel_whitelisted(self, channel_name: str) -> bool:
        """Check if a channel is whitelisted (overrides global rejection)."""
        return channel_name.lower() in {c.lower() for c in self.whitelist_channels}

    def is_keyword_blacklisted(self, title: str) -> bool:
        """Check if a title contains blacklisted keywords."""
        title_lower = title.lower()
        return any(kw.lower() in title_lower for kw in self.blacklist_keywords)

    def add_project(self, project_path: str) -> None:
        """Track a project for this client."""
        if project_path not in self.projects:
            self.projects.append(project_path)
            self.updated = datetime.now().isoformat()

    def add_blacklist_channel(self, channel_name: str) -> None:
        """Add a channel to the client's blacklist."""
        self.blacklist_channels.add(channel_name)
        self.updated = datetime.now().isoformat()

    def add_blacklist_keyword(self, keyword: str) -> None:
        """Add a keyword to the client's blacklist."""
        self.blacklist_keywords.add(keyword)
        self.updated = datetime.now().isoformat()

    def record_acceptance(self, count: int = 1) -> None:
        """Record accepted videos."""
        self.total_videos_accepted += count
        self.updated = datetime.now().isoformat()

    def record_rejection(self, count: int = 1) -> None:
        """Record rejected videos."""
        self.total_videos_rejected += count
        self.updated = datetime.now().isoformat()

    def save(self) -> Path:
        """Save profile to disk."""
        profile_dir = self.get_profile_dir()
        profile_dir.mkdir(parents=True, exist_ok=True)

        profile_path = profile_dir / "profile.yaml"

        # Convert to dict for YAML
        data = {
            "client_id": self.client_id,
            "display_name": self.display_name,
            "created": self.created,
            "updated": datetime.now().isoformat(),
            "preferences": asdict(self.preferences),
            "thresholds": asdict(self.thresholds),
            "blacklist_channels": list(self.blacklist_channels),
            "blacklist_keywords": list(self.blacklist_keywords),
            "whitelist_channels": list(self.whitelist_channels),
            "projects": self.projects,
            "total_videos_accepted": self.total_videos_accepted,
            "total_videos_rejected": self.total_videos_rejected,
        }

        if self.evolved_preset:
            data["evolved_preset"] = asdict(self.evolved_preset)

        with open(profile_path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)

        logger.info(f"Saved client profile: {profile_path}")
        return profile_path

    @classmethod
    def load(cls, client_id: str) -> Optional["ClientProfile"]:
        """Load profile from disk, returns None if not found."""
        profile_dir = CLIENT_PROFILES_DIR / f"client_{client_id}"
        profile_path = profile_dir / "profile.yaml"

        if not profile_path.exists():
            return None

        try:
            with open(profile_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}

            return cls(
                client_id=data.get("client_id", client_id),
                display_name=data.get("display_name", ""),
                created=data.get("created", ""),
                updated=data.get("updated", ""),
                preferences=data.get("preferences", {}),
                thresholds=data.get("thresholds", {}),
                blacklist_channels=set(data.get("blacklist_channels", [])),
                blacklist_keywords=set(data.get("blacklist_keywords", [])),
                whitelist_channels=set(data.get("whitelist_channels", [])),
                evolved_preset=data.get("evolved_preset"),
                projects=data.get("projects", []),
                total_videos_accepted=data.get("total_videos_accepted", 0),
                total_videos_rejected=data.get("total_videos_rejected", 0),
            )
        except Exception as e:
            logger.warning(f"Error loading client profile {client_id}: {e}")
            return None


def get_or_create_client_profile(client_id: str) -> ClientProfile:
    """Get existing client profile or create a new one."""
    profile = ClientProfile.load(client_id)
    if profile is None:
        profile = ClientProfile(client_id=client_id)
        profile.save()
        logger.info(f"Created new client profile: {client_id}")
    return profile


def list_client_profiles() -> List[str]:
    """List all client IDs with profiles."""
    if not CLIENT_PROFILES_DIR.exists():
        return []

    clients = []
    for path in CLIENT_PROFILES_DIR.iterdir():
        if path.is_dir() and path.name.startswith("client_"):
            client_id = path.name[7:]  # Remove "client_" prefix
            clients.append(client_id)

    return sorted(clients)


def get_client_rejections_path(client_id: str) -> Path:
    """Get path to client-specific rejections file."""
    return CLIENT_PROFILES_DIR / f"client_{client_id}" / "rejections.json"


def evolve_preset_from_history(
    client_id: str,
    project_dirs: Optional[List[Path]] = None,
) -> EvolvedPreset:
    """
    Generate an evolved preset based on project history.

    Analyzes:
    - Which videos were used vs skipped
    - Rejection patterns (channels, keywords, duration)
    - LLM relevance scores

    Args:
        client_id: Client to evolve preset for
        project_dirs: Specific projects to analyze (or all from profile)

    Returns:
        EvolvedPreset with learned patterns
    """
    profile = ClientProfile.load(client_id)
    if profile is None:
        logger.warning(f"No profile found for client {client_id}")
        return EvolvedPreset()

    # Get project directories
    if project_dirs is None:
        project_dirs = [Path(p) for p in profile.projects if Path(p).exists()]

    if not project_dirs:
        logger.warning(f"No projects found for client {client_id}")
        return EvolvedPreset()

    # Collect data from all projects
    rejected_channels: Counter = Counter()
    rejected_keywords: Counter = Counter()
    accepted_channels: Counter = Counter()
    accepted_durations: List[float] = []
    rejection_reasons: Counter = Counter()

    for project_dir in project_dirs:
        # Load sources.json to find accepted videos
        sources_file = project_dir / "sources.json"
        if sources_file.exists():
            try:
                with open(sources_file, "r", encoding="utf-8") as f:
                    sources = json.load(f)

                for video_id, info in sources.items():
                    channel = info.get("channel", "")
                    duration = info.get("duration", 0)
                    was_used = info.get("was_used", False)

                    if was_used:
                        accepted_channels[channel] += 1
                        if duration:
                            accepted_durations.append(duration)
                    else:
                        # Not used - might have been filtered
                        pass

            except Exception as e:
                logger.debug(f"Error loading sources from {project_dir}: {e}")

        # Load project rejections
        rejections_file = project_dir / "project_rejections.yaml"
        if rejections_file.exists():
            try:
                with open(rejections_file, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f) or {}

                for video in data.get("rejected_videos", []):
                    channel = video.get("channel", "")
                    reason = video.get("reason", "unknown")
                    title = video.get("title", "")

                    if channel:
                        rejected_channels[channel] += 1
                    rejection_reasons[reason] += 1

                    # Extract keywords from rejected titles
                    for word in _extract_keywords_from_title(title):
                        rejected_keywords[word] += 1

                for channel in data.get("blocked_channels", []):
                    if isinstance(channel, str):
                        rejected_channels[channel] += 1
                    else:
                        rejected_channels[channel.get("name", "")] += 1

            except Exception as e:
                logger.debug(f"Error loading rejections from {project_dir}: {e}")

    # Calculate preferred duration range
    if accepted_durations:
        avg_duration = statistics.mean(accepted_durations)
        duration_min = avg_duration * 0.5
        duration_max = avg_duration * 2.0
    else:
        avg_duration = 60.0
        duration_min = 30.0
        duration_max = 180.0

    # Create evolved preset
    preset = EvolvedPreset(
        auto_blacklist_channels=[ch for ch, _ in rejected_channels.most_common(20)],
        auto_blacklist_keywords=[kw for kw, count in rejected_keywords.most_common(30) if count >= 2],
        preferred_channels=[ch for ch, _ in accepted_channels.most_common(10)],
        preferred_duration_range=(duration_min, duration_max),
        avg_accepted_duration=avg_duration,
        rejection_patterns=dict(rejection_reasons),
        projects_analyzed=len(project_dirs),
    )

    # Update profile with evolved preset
    profile.evolved_preset = preset
    profile.save()

    logger.info(
        f"Evolved preset for {client_id}: "
        f"{len(preset.auto_blacklist_channels)} blocked channels, "
        f"{len(preset.auto_blacklist_keywords)} blocked keywords, "
        f"duration {duration_min:.0f}-{duration_max:.0f}s"
    )

    return preset


def _extract_keywords_from_title(title: str) -> List[str]:
    """Extract meaningful keywords from a video title."""
    # Skip common words
    stopwords = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
        "being", "have", "has", "had", "do", "does", "did", "will", "would",
        "could", "should", "may", "might", "must", "shall", "can", "need",
        "this", "that", "these", "those", "it", "its", "my", "your", "our",
        "their", "his", "her", "video", "clip", "full", "hd", "4k", "official",
    }

    # Clean and split
    import re
    words = re.findall(r'\b[a-zA-Z]{3,}\b', title.lower())

    return [w for w in words if w not in stopwords]


def apply_client_profile_to_config(profile: ClientProfile, config) -> None:
    """
    Apply client profile settings to a config object.

    This modifies the config to use client-specific thresholds and preferences.

    Args:
        profile: ClientProfile with learned settings
        config: Config object to modify
    """
    # Apply quality thresholds
    feedback_config = getattr(config, 'feedback', None)
    if feedback_config:
        channel_scoring = getattr(feedback_config, 'channel_scoring', None)
        if channel_scoring:
            # Use client's min_score if it's stricter
            current_min = getattr(channel_scoring, 'min_score', 0.3)
            if profile.thresholds.min_channel_score > current_min:
                try:
                    channel_scoring.min_score = profile.thresholds.min_channel_score
                except AttributeError:
                    pass  # Config is immutable

    # Apply duration preferences to matching config
    matching_config = getattr(config, 'matching', None)
    if matching_config and profile.evolved_preset:
        # Could set preferred duration range here if supported
        pass

    logger.debug(f"Applied client profile settings for {profile.client_id}")


def merge_client_rejections_to_global(client_id: str) -> int:
    """
    Merge client-specific rejections into the global database.

    Returns count of new rejections added.
    """
    from .rejections import RejectionDatabase, RejectedVideo, GLOBAL_DB_FILE

    client_rejections_path = get_client_rejections_path(client_id)
    if not client_rejections_path.exists():
        return 0

    # Load global database
    global_db = RejectionDatabase.load(GLOBAL_DB_FILE)

    # Load client rejections
    try:
        with open(client_rejections_path, "r", encoding="utf-8") as f:
            client_data = json.load(f)

        count = 0
        for r in client_data.get("rejections", []):
            rejection = RejectedVideo(**r)
            if rejection.video_id not in global_db._rejected_video_ids:
                global_db.add_rejection(rejection)
                count += 1

        if count > 0:
            global_db.save()
            logger.info(f"Merged {count} rejections from client {client_id} to global database")

        return count

    except Exception as e:
        logger.warning(f"Error merging client rejections: {e}")
        return 0
