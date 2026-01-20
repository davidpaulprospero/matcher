"""Channel reputation scoring system.

Calculates trust scores for YouTube channels based on:
- Historical acceptance/rejection rates
- YouTube metadata (subscriber count, video count)
- Channel category configuration (trusted/blocked/neutral)

Integrates with TitleFilter to prioritize high-quality sources.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Set

if TYPE_CHECKING:
    from .rejections import RejectionDatabase

logger = logging.getLogger(__name__)


@dataclass
class ChannelMetadata:
    """Metadata fetched from YouTube API."""

    channel_id: str
    channel_name: str
    subscriber_count: int = 0
    video_count: int = 0
    view_count: int = 0
    description: str = ""
    country: str = ""
    custom_url: str = ""
    published_at: str = ""  # Channel creation date


@dataclass
class ChannelCategory:
    """Channel categorization for filtering."""

    trusted: Set[str] = field(default_factory=set)  # Always accept (case-insensitive)
    blocked: Set[str] = field(default_factory=set)  # Always reject (case-insensitive)

    def is_trusted(self, channel_name: str) -> bool:
        """Check if channel is in trusted list."""
        return channel_name.lower() in {c.lower() for c in self.trusted}

    def is_blocked(self, channel_name: str) -> bool:
        """Check if channel is in blocked list."""
        return channel_name.lower() in {c.lower() for c in self.blocked}


class ChannelScorer:
    """Calculates and manages channel reputation scores.

    Combines historical data from RejectionDatabase with YouTube API
    metadata to compute trust scores (0.0 - 1.0).

    Higher scores = more trustworthy content sources.
    """

    def __init__(
        self,
        rejection_db: Optional['RejectionDatabase'] = None,
        categories: Optional[ChannelCategory] = None,
        youtube_api_key: Optional[str] = None,
    ):
        """Initialize ChannelScorer.

        Args:
            rejection_db: RejectionDatabase for historical data
            categories: Channel category configuration
            youtube_api_key: YouTube Data API key (optional)
        """
        self.rejection_db = rejection_db
        self.categories = categories or ChannelCategory()
        self.youtube_api_key = youtube_api_key or os.environ.get('YOUTUBE_API_KEY', '')

        # YouTube API client (lazy initialized)
        self._youtube = None

        # Cache for fetched metadata
        self._metadata_cache: Dict[str, ChannelMetadata] = {}

    @property
    def youtube(self):
        """Lazy-initialize YouTube API client."""
        if self._youtube is None and self.youtube_api_key:
            try:
                from googleapiclient.discovery import build
                self._youtube = build('youtube', 'v3', developerKey=self.youtube_api_key)
                logger.debug("YouTube API client initialized for channel scoring")
            except ImportError:
                logger.debug("google-api-python-client not installed")
            except Exception as e:
                logger.warning(f"Failed to initialize YouTube API: {e}")
        return self._youtube

    def fetch_channel_metadata(
        self,
        channel_ids: List[str],
    ) -> Dict[str, ChannelMetadata]:
        """Fetch metadata for multiple channels from YouTube API.

        Args:
            channel_ids: List of YouTube channel IDs

        Returns:
            Dict mapping channel_id to ChannelMetadata
        """
        if not channel_ids:
            return {}

        if not self.youtube:
            logger.debug("YouTube API not available, skipping metadata fetch")
            return {}

        # Filter out already cached channels
        to_fetch = [cid for cid in channel_ids if cid not in self._metadata_cache]
        if not to_fetch:
            return {cid: self._metadata_cache[cid] for cid in channel_ids if cid in self._metadata_cache}

        result = {}

        # Batch in groups of 50 (API limit)
        for i in range(0, len(to_fetch), 50):
            batch = to_fetch[i:i + 50]
            try:
                request = self.youtube.channels().list(
                    part='snippet,statistics',
                    id=','.join(batch)
                )
                response = request.execute()

                for item in response.get('items', []):
                    cid = item['id']
                    snippet = item.get('snippet', {})
                    stats = item.get('statistics', {})

                    metadata = ChannelMetadata(
                        channel_id=cid,
                        channel_name=snippet.get('title', ''),
                        subscriber_count=int(stats.get('subscriberCount', 0)),
                        video_count=int(stats.get('videoCount', 0)),
                        view_count=int(stats.get('viewCount', 0)),
                        description=snippet.get('description', '')[:500],
                        country=snippet.get('country', ''),
                        custom_url=snippet.get('customUrl', ''),
                        published_at=snippet.get('publishedAt', ''),
                    )
                    self._metadata_cache[cid] = metadata
                    result[cid] = metadata

                logger.debug(f"Fetched metadata for {len(response.get('items', []))} channels")

            except Exception as e:
                logger.warning(f"YouTube API error fetching channel metadata: {e}")

        # Return requested channels from cache
        return {cid: self._metadata_cache[cid] for cid in channel_ids if cid in self._metadata_cache}

    def fetch_channel_id_by_name(self, channel_name: str) -> Optional[str]:
        """Look up channel ID by channel name using YouTube search.

        Args:
            channel_name: Channel name to search for

        Returns:
            Channel ID if found, None otherwise
        """
        if not self.youtube:
            return None

        try:
            request = self.youtube.search().list(
                part='snippet',
                q=channel_name,
                type='channel',
                maxResults=1
            )
            response = request.execute()

            items = response.get('items', [])
            if items:
                return items[0]['snippet']['channelId']
        except Exception as e:
            logger.debug(f"Error searching for channel '{channel_name}': {e}")

        return None

    def calculate_score(
        self,
        channel_id: str = "",
        channel_name: str = "",
        metadata: Optional[ChannelMetadata] = None,
    ) -> float:
        """Calculate trust score for a channel.

        Score formula:
        - Base score: 0.5 (neutral)
        - +0.3 max from acceptance rate
        - -0.4 max from rejection penalty (weighted 2x)
        - +0.1 bonus for >1M subscribers
        - +0.05 bonus for >100K subscribers
        - Override: 1.0 for trusted, 0.0 for blocked

        Args:
            channel_id: YouTube channel ID (for historical lookup)
            channel_name: Channel name (for category check)
            metadata: Pre-fetched metadata (optional)

        Returns:
            Trust score from 0.0 to 1.0
        """
        # Check category overrides first
        if channel_name:
            if self.categories.is_trusted(channel_name):
                return 1.0
            if self.categories.is_blocked(channel_name):
                return 0.0

        # Base score for unknown channels
        base_score = 0.5

        # Get historical stats from rejection database
        acceptance_bonus = 0.0
        rejection_penalty = 0.0

        if self.rejection_db and channel_id:
            stats = self.rejection_db.channel_stats.get(channel_id)
            if stats and stats.total_videos > 0:
                acceptance_rate = stats.accepted / stats.total_videos
                rejection_rate = (stats.rejected * 2) / stats.total_videos  # 2x weight

                acceptance_bonus = acceptance_rate * 0.3
                rejection_penalty = rejection_rate * 0.4

        # Subscriber bonus from metadata
        subscriber_bonus = 0.0
        if metadata:
            if metadata.subscriber_count > 1_000_000:
                subscriber_bonus = 0.1
            elif metadata.subscriber_count > 100_000:
                subscriber_bonus = 0.05
            elif metadata.subscriber_count > 10_000:
                subscriber_bonus = 0.02

        # Calculate final score
        score = base_score + acceptance_bonus - rejection_penalty + subscriber_bonus

        # Clamp to valid range
        return max(0.0, min(1.0, score))

    def score_channels(
        self,
        channels: List[Dict],
        fetch_metadata: bool = True,
    ) -> Dict[str, float]:
        """Score multiple channels at once.

        Args:
            channels: List of dicts with 'channel' and optionally 'channel_id' keys
            fetch_metadata: Whether to fetch YouTube metadata

        Returns:
            Dict mapping channel name to score
        """
        scores = {}

        # Extract channel IDs for batch metadata fetch
        channel_ids = [c.get('channel_id') for c in channels if c.get('channel_id')]
        metadata_map = {}

        if fetch_metadata and channel_ids:
            metadata_map = self.fetch_channel_metadata(channel_ids)

        for channel in channels:
            name = channel.get('channel', '')
            cid = channel.get('channel_id', '')

            if not name:
                continue

            metadata = metadata_map.get(cid) if cid else None
            score = self.calculate_score(
                channel_id=cid,
                channel_name=name,
                metadata=metadata,
            )
            scores[name] = score

        return scores

    def get_channel_report(
        self,
        channels: List[Dict],
    ) -> str:
        """Generate a human-readable channel score report.

        Args:
            channels: List of dicts with channel info

        Returns:
            Formatted report string
        """
        scores = self.score_channels(channels, fetch_metadata=True)

        if not scores:
            return "No channels to report."

        lines = ["Channel Reputation Scores:", "=" * 40]

        # Sort by score descending
        sorted_channels = sorted(scores.items(), key=lambda x: -x[1])

        for name, score in sorted_channels:
            # Determine category
            if self.categories.is_trusted(name):
                category = "[TRUSTED]"
            elif self.categories.is_blocked(name):
                category = "[BLOCKED]"
            elif score >= 0.7:
                category = "[HIGH]"
            elif score >= 0.4:
                category = "[MEDIUM]"
            else:
                category = "[LOW]"

            lines.append(f"  {score:.2f} {category:10} {name}")

        return "\n".join(lines)


def load_channel_categories_from_config(config) -> ChannelCategory:
    """Load channel categories from config.

    Args:
        config: Config object with feedback.channel_categories

    Returns:
        ChannelCategory instance
    """
    categories = ChannelCategory()

    # Try to get from config
    feedback_config = getattr(config, 'feedback', None)
    if not feedback_config:
        return categories

    channel_cats = getattr(feedback_config, 'channel_categories', None)
    if not channel_cats:
        return categories

    # Handle both dict and object access
    if isinstance(channel_cats, dict):
        trusted = channel_cats.get('trusted', [])
        blocked = channel_cats.get('blocked', [])
    else:
        trusted = getattr(channel_cats, 'trusted', [])
        blocked = getattr(channel_cats, 'blocked', [])

    categories.trusted = set(trusted) if trusted else set()
    categories.blocked = set(blocked) if blocked else set()

    logger.debug(f"Loaded channel categories: {len(categories.trusted)} trusted, {len(categories.blocked)} blocked")

    return categories


def create_channel_scorer(
    config,
    rejection_db: Optional['RejectionDatabase'] = None,
) -> ChannelScorer:
    """Create a ChannelScorer from config.

    Args:
        config: Config object
        rejection_db: Optional RejectionDatabase instance

    Returns:
        Configured ChannelScorer instance
    """
    categories = load_channel_categories_from_config(config)

    # Get API key from config or environment
    api_key = os.environ.get('YOUTUBE_API_KEY', '')
    if not api_key:
        download_config = getattr(config, 'download', None)
        if download_config:
            api_key = getattr(download_config, 'youtube_api_key', '')

    return ChannelScorer(
        rejection_db=rejection_db,
        categories=categories,
        youtube_api_key=api_key,
    )
