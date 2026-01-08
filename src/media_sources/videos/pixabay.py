"""
Pixabay video search and download client.

Extracted from entity_images.py StockVideoDownloader.search_pixabay_videos()
(lines 1093-1151) and download logic (Jan 7, 2026).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List, Optional, TYPE_CHECKING

from ..base import BaseMediaClient
from ..models import VideoResult

if TYPE_CHECKING:
    from ...config import Config

logger = logging.getLogger(__name__)

PIXABAY_VIDEOS_API = "https://pixabay.com/api/videos/"


class PixabayVideoClient(BaseMediaClient):
    """Client for searching and downloading videos from Pixabay API."""

    def __init__(
        self,
        config: 'Config',
        output_dir: str,
        api_key: Optional[str] = None,
        min_duration: float = 3.0,
        max_duration: float = 30.0,
        prefer_hd: bool = True,
        download_timeout: int = 60
    ):
        """
        Initialize Pixabay video client.

        Args:
            config: Configuration object
            output_dir: Directory for downloaded videos
            api_key: Pixabay API key (falls back to env var)
            min_duration: Minimum video duration in seconds
            max_duration: Maximum video duration in seconds
            prefer_hd: Prefer higher resolution videos
            download_timeout: HTTP timeout for downloads
        """
        super().__init__(
            config=config,
            output_dir=output_dir,
            min_size_mb=0.1,  # Videos are typically larger
            download_timeout=download_timeout,
            rate_limit_delay=0.5,
            user_agent="Voiceover-Matcher/2.3 PixabayVideoClient"
        )

        self.api_key = api_key or os.getenv("PIXABAY_API_KEY")
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.prefer_hd = prefer_hd

    def search(self, query: str, max_results: int = 10) -> List[VideoResult]:
        """
        Search Pixabay for videos.

        Args:
            query: Search query
            max_results: Maximum results to return

        Returns:
            List of VideoResult objects
        """
        if not self.api_key:
            logger.debug("Pixabay API key not available")
            return []

        self._rate_limit()

        params = {
            "key": self.api_key,
            "q": query,
            "per_page": max_results,
            "video_type": "film",  # film, animation, all
            "orientation": "horizontal"
        }

        try:
            response = self.session.get(PIXABAY_VIDEOS_API, params=params)
            response.raise_for_status()
            data = response.json()

            results = []
            for video in data.get("hits", []):
                duration = video.get("duration", 0)

                # Filter by duration
                if duration < self.min_duration or duration > self.max_duration:
                    continue

                # Get video URLs - Pixabay provides multiple sizes
                videos_dict = video.get("videos", {})

                # Prefer large > medium > small
                video_data = None
                for size in ["large", "medium", "small"]:
                    if size in videos_dict and videos_dict[size].get("url"):
                        video_data = videos_dict[size]
                        break

                if not video_data:
                    continue

                results.append(VideoResult(
                    id=str(video.get("id", "")),
                    source="pixabay",
                    url=video.get("pageURL", ""),
                    download_url=video_data.get("url", ""),
                    width=video_data.get("width", 0),
                    height=video_data.get("height", 0),
                    duration=duration,
                    quality=video_data.get("size", "unknown"),
                    file_type="mp4"
                ))

            logger.info(f"Pixabay videos '{query}': {len(results)} results")
            return results

        except Exception as e:
            logger.error(f"Pixabay video search error: {e}")
            return []

    def download_video(self, video: VideoResult) -> Optional[str]:
        """
        Download a single video.

        Args:
            video: VideoResult to download

        Returns:
            Path to downloaded file, or None if failed
        """
        if not video.download_url:
            return None

        self._rate_limit()

        # Create short filename for NLE compatibility
        # Format: {source_letter}{id}.mp4 = ~12 chars
        source_letter = video.source[0].lower()  # p=pixabay
        short_id = str(video.id)[-8:] if len(str(video.id)) > 8 else str(video.id)
        ext = "mp4"
        filename = f"{source_letter}{short_id}.{ext}"
        filepath = self.output_dir / filename

        if filepath.exists():
            logger.debug(f"Video already exists: {filename}")
            return str(filepath)

        try:
            response = self.session.get(video.download_url, stream=True, timeout=self.download_timeout)
            response.raise_for_status()

            # Download with progress
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

            file_size_mb = filepath.stat().st_size / (1024 * 1024)
            logger.info(f"Downloaded video: {filename} ({file_size_mb:.1f}MB, {video.duration}s)")

            return str(filepath)

        except Exception as e:
            logger.error(f"Video download error for {video.id}: {e}")
            if filepath.exists():
                filepath.unlink()
            return None

    def search_and_download(
        self,
        query: str,
        max_videos: int = 3,
        entity_name: str = "",
        entity_type: str = ""
    ) -> List[str]:
        """
        Search and download videos for a query.

        Args:
            query: Search keywords
            max_videos: Maximum videos to download
            entity_name: Entity name for metadata (unused, for API consistency)
            entity_type: Entity type for metadata (unused, for API consistency)

        Returns:
            List of downloaded file paths
        """
        results = self.search(query, max_results=max_videos * 2)

        downloaded = []
        for video in results:
            if len(downloaded) >= max_videos:
                break

            path = self.download_video(video)
            if path:
                downloaded.append(path)

        return downloaded
