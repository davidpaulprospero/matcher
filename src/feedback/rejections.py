"""Rejection database for learning from editorial feedback.

Tracks rejected videos and channels to automatically filter them
from future downloads. Supports:
- Manual rejections via project_rejections.yaml
- DaVinci Resolve marker imports
- Cross-project learning via global database
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set

import yaml

logger = logging.getLogger(__name__)

# Global database location
GLOBAL_DB_DIR = Path.home() / ".matcher_rejections"
GLOBAL_DB_FILE = GLOBAL_DB_DIR / "rejections.json"
CHANNEL_SCORES_FILE = GLOBAL_DB_DIR / "channel_scores.json"

# Project-level rejection file
PROJECT_REJECTIONS_FILE = "project_rejections.yaml"


@dataclass
class RejectedVideo:
    """A video that was rejected during editorial review."""

    video_id: str  # YouTube video ID (11 chars)
    channel_id: str = ""  # YouTube channel ID
    channel_name: str = ""  # Human-readable channel name
    title: str = ""  # Video title
    rejection_reason: str = ""  # Why rejected (trainer, movie, low_quality, etc.)
    rejection_source: str = "manual"  # How identified: manual, davinci_marker, auto
    project: str = ""  # Which project rejected it
    timestamp: str = ""  # ISO format datetime

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


@dataclass
class ChannelStats:
    """Statistics for a channel across all projects."""

    channel_id: str
    channel_name: str
    total_videos: int = 0
    accepted: int = 0  # Videos that were used in final output
    rejected: int = 0  # Videos explicitly rejected
    subscriber_count: int = 0  # From YouTube API
    score: float = 0.5  # Computed reputation score (0-1)


@dataclass
class RejectionDatabase:
    """Database of rejected videos and channel scores.

    Stored at ~/.matcher_rejections/ for cross-project learning.
    Also loads project-specific rejections from project_rejections.yaml.
    """

    rejections: List[RejectedVideo] = field(default_factory=list)
    channel_scores: Dict[str, float] = field(default_factory=dict)
    channel_stats: Dict[str, ChannelStats] = field(default_factory=dict)

    # Cached sets for fast lookup
    _rejected_video_ids: Set[str] = field(default_factory=set, repr=False)
    _rejected_channel_ids: Set[str] = field(default_factory=set, repr=False)
    _blocked_channels_by_name: Set[str] = field(default_factory=set, repr=False)

    def __post_init__(self):
        """Build lookup caches."""
        self._rebuild_caches()

    def _rebuild_caches(self):
        """Rebuild fast-lookup caches from rejections list."""
        self._rejected_video_ids = {r.video_id for r in self.rejections}
        self._rejected_channel_ids = {
            r.channel_id for r in self.rejections if r.channel_id
        }
        self._blocked_channels_by_name = {
            r.channel_name.lower() for r in self.rejections if r.channel_name
        }

    def is_video_rejected(self, video_id: str) -> bool:
        """Check if a video ID was previously rejected."""
        return video_id in self._rejected_video_ids

    def is_channel_blocked(
        self, channel_id: str = "", channel_name: str = ""
    ) -> bool:
        """Check if a channel is blocked (by ID or name)."""
        if channel_id and channel_id in self._rejected_channel_ids:
            return True
        if channel_name and channel_name.lower() in self._blocked_channels_by_name:
            return True
        return False

    def get_channel_score(self, channel_id: str, default: float = 0.5) -> float:
        """Get reputation score for a channel (0-1, higher is better)."""
        return self.channel_scores.get(channel_id, default)

    def add_rejection(self, rejection: RejectedVideo) -> None:
        """Add a new rejection and update caches."""
        # Check for duplicates
        if rejection.video_id in self._rejected_video_ids:
            logger.debug(f"Video {rejection.video_id} already rejected, skipping")
            return

        self.rejections.append(rejection)
        self._rejected_video_ids.add(rejection.video_id)

        if rejection.channel_id:
            self._rejected_channel_ids.add(rejection.channel_id)
        if rejection.channel_name:
            self._blocked_channels_by_name.add(rejection.channel_name.lower())

        # Update channel stats
        self._update_channel_stats(rejection)

        logger.info(
            f"Added rejection: {rejection.video_id} "
            f"({rejection.rejection_reason}) from {rejection.channel_name}"
        )

    def add_rejections(self, rejections: List[RejectedVideo]) -> int:
        """Add multiple rejections, returns count of new rejections."""
        count = 0
        for r in rejections:
            if r.video_id not in self._rejected_video_ids:
                self.add_rejection(r)
                count += 1
        return count

    def _update_channel_stats(self, rejection: RejectedVideo) -> None:
        """Update channel statistics after a rejection."""
        if not rejection.channel_id:
            return

        if rejection.channel_id not in self.channel_stats:
            self.channel_stats[rejection.channel_id] = ChannelStats(
                channel_id=rejection.channel_id,
                channel_name=rejection.channel_name,
            )

        stats = self.channel_stats[rejection.channel_id]
        stats.rejected += 1
        stats.total_videos += 1

        # Recalculate score
        self._recalculate_channel_score(rejection.channel_id)

    def _recalculate_channel_score(self, channel_id: str) -> None:
        """Recalculate reputation score for a channel."""
        if channel_id not in self.channel_stats:
            return

        stats = self.channel_stats[channel_id]
        if stats.total_videos == 0:
            score = 0.5
        else:
            # Base score from acceptance rate
            acceptance_rate = stats.accepted / stats.total_videos
            # Rejection penalty (weighted 2x)
            rejection_penalty = (stats.rejected * 2) / max(stats.total_videos, 1)

            score = 0.5 + (acceptance_rate * 0.3) - (rejection_penalty * 0.4)

            # Subscriber bonus
            if stats.subscriber_count > 1_000_000:
                score += 0.1
            elif stats.subscriber_count > 100_000:
                score += 0.05

            score = max(0.0, min(1.0, score))

        stats.score = score
        self.channel_scores[channel_id] = score

    def record_acceptance(self, video_id: str, channel_id: str, channel_name: str = "") -> None:
        """Record that a video was accepted/used in final output."""
        if channel_id not in self.channel_stats:
            self.channel_stats[channel_id] = ChannelStats(
                channel_id=channel_id,
                channel_name=channel_name,
            )

        stats = self.channel_stats[channel_id]
        stats.accepted += 1
        stats.total_videos += 1

        self._recalculate_channel_score(channel_id)

    def get_rejection_count(self) -> int:
        """Get total number of rejections."""
        return len(self.rejections)

    def get_blocked_channel_count(self) -> int:
        """Get number of blocked channels."""
        return len(self._blocked_channels_by_name)

    def get_rejections_by_reason(self) -> Dict[str, int]:
        """Get rejection counts grouped by reason."""
        reasons: Dict[str, int] = {}
        for r in self.rejections:
            reason = r.rejection_reason or "unknown"
            reasons[reason] = reasons.get(reason, 0) + 1
        return reasons

    def save(self, path: Optional[Path] = None) -> None:
        """Save database to JSON file."""
        path = path or GLOBAL_DB_FILE

        # Ensure directory exists
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "version": 1,
            "updated": datetime.now().isoformat(),
            "rejections": [asdict(r) for r in self.rejections],
            "channel_scores": self.channel_scores,
            "channel_stats": {k: asdict(v) for k, v in self.channel_stats.items()},
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved rejection database: {len(self.rejections)} rejections to {path}")

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "RejectionDatabase":
        """Load database from JSON file."""
        path = path or GLOBAL_DB_FILE

        if not path.exists():
            logger.debug(f"No rejection database at {path}, starting fresh")
            return cls()

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            rejections = [RejectedVideo(**r) for r in data.get("rejections", [])]
            channel_scores = data.get("channel_scores", {})
            channel_stats = {
                k: ChannelStats(**v) for k, v in data.get("channel_stats", {}).items()
            }

            db = cls(
                rejections=rejections,
                channel_scores=channel_scores,
                channel_stats=channel_stats,
            )
            logger.info(f"Loaded rejection database: {len(rejections)} rejections from {path}")
            return db

        except Exception as e:
            logger.warning(f"Error loading rejection database: {e}")
            return cls()

    def merge_project_rejections(self, project_dir: Path) -> int:
        """
        Merge rejections from a project's project_rejections.yaml.

        Returns number of new rejections added.
        """
        rejections_file = project_dir / PROJECT_REJECTIONS_FILE
        if not rejections_file.exists():
            return 0

        try:
            with open(rejections_file, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}

            count = 0
            project_name = project_dir.name

            # Process video rejections
            for video in data.get("rejected_videos", []):
                rejection = RejectedVideo(
                    video_id=video.get("id", ""),
                    channel_name=video.get("channel", ""),
                    title=video.get("title", ""),
                    rejection_reason=video.get("reason", "manual"),
                    rejection_source="project_yaml",
                    project=project_name,
                )
                if rejection.video_id and rejection.video_id not in self._rejected_video_ids:
                    self.add_rejection(rejection)
                    count += 1

            # Process channel blocks
            for channel in data.get("blocked_channels", []):
                if isinstance(channel, str):
                    channel_name = channel
                    reason = "blocked"
                else:
                    channel_name = channel.get("name", "")
                    reason = channel.get("reason", "blocked")

                if channel_name.lower() not in self._blocked_channels_by_name:
                    # Add a placeholder rejection for channel-level blocks
                    rejection = RejectedVideo(
                        video_id=f"channel_block_{hash(channel_name) % 10000:04d}",
                        channel_name=channel_name,
                        rejection_reason=reason,
                        rejection_source="project_yaml",
                        project=project_name,
                    )
                    self.add_rejection(rejection)
                    count += 1

            if count > 0:
                logger.info(f"Merged {count} rejections from {rejections_file}")

            return count

        except Exception as e:
            logger.warning(f"Error loading project rejections: {e}")
            return 0


def get_global_rejection_db_path() -> Path:
    """Get path to global rejection database."""
    return GLOBAL_DB_FILE


def load_rejection_database(
    project_dir: Optional[Path] = None,
    include_global: bool = True,
    client_id: Optional[str] = None,
) -> RejectionDatabase:
    """
    Load rejection database, optionally merging project-specific rejections.

    Args:
        project_dir: Project directory to load project_rejections.yaml from
        include_global: Whether to include global rejections
        client_id: Client ID to load client-specific rejections from

    Returns:
        RejectionDatabase with merged rejections
    """
    if include_global:
        db = RejectionDatabase.load(GLOBAL_DB_FILE)
    else:
        db = RejectionDatabase()

    # Load client-specific rejections
    if client_id:
        client_rejections_path = GLOBAL_DB_DIR / f"client_{client_id}" / "rejections.json"
        if client_rejections_path.exists():
            try:
                import json
                with open(client_rejections_path, "r", encoding="utf-8") as f:
                    client_data = json.load(f)

                for r in client_data.get("rejections", []):
                    rejection = RejectedVideo(**r)
                    if rejection.video_id not in db._rejected_video_ids:
                        db.add_rejection(rejection)

                logger.debug(f"Loaded client rejections from {client_rejections_path}")
            except Exception as e:
                logger.warning(f"Error loading client rejections: {e}")

    if project_dir:
        db.merge_project_rejections(Path(project_dir))

    return db


def create_project_rejections_template(project_dir: Path) -> Path:
    """Create a template project_rejections.yaml file."""
    template = """# Project Rejections
# Videos and channels to exclude from this project's downloads.
# These will also be added to the global rejection database.

# Rejected videos (by YouTube video ID)
rejected_videos:
  # - id: "dQw4w9WgXcQ"
  #   title: "Example video title"
  #   channel: "Example Channel"
  #   reason: "trainer content"

# Blocked channels (by name - case insensitive)
blocked_channels:
  # - name: "Cesar Millan"
  #   reason: "professional trainer"
  # - name: "Movieclips"
  #   reason: "copyrighted movie content"

# Blocked keywords in titles (in addition to config.yaml blacklist)
blocked_keywords:
  # - "training tips"
  # - "official trailer"
"""
    path = project_dir / PROJECT_REJECTIONS_FILE
    with open(path, "w", encoding="utf-8") as f:
        f.write(template)

    logger.info(f"Created project rejections template: {path}")
    return path
