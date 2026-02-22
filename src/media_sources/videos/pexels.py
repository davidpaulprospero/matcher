"""
Pexels video search and download client.

Extracted from entity_images.py StockVideoDownloader.search_pexels_videos()
(lines 1036-1091) and download logic (Jan 7, 2026).
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

PEXELS_VIDEOS_API = "https://api.pexels.com/videos/search"


class PexelsVideoClient(BaseMediaClient):
    """Client for searching and downloading videos from Pexels API."""

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
        Initialize Pexels video client.

        Args:
            config: Configuration object
            output_dir: Directory for downloaded videos
            api_key: Pexels API key (falls back to env var)
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
            user_agent="Voiceover-Matcher/2.3 PexelsVideoClient"
        )

        self.api_key = api_key or os.getenv("PEXELS_API_KEY")
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.prefer_hd = prefer_hd

    def search(self, query: str, max_results: int = 10) -> List[VideoResult]:
        """
        Search Pexels for videos.

        Args:
            query: Search query
            max_results: Maximum results to return

        Returns:
            List of VideoResult objects
        """
        if not self.api_key:
            logger.debug("Pexels API key not available")
            return []

        self._rate_limit()

        headers = {"Authorization": self.api_key}
        params = {
            "query": query,
            "per_page": max_results,
            "orientation": "landscape"
        }

        logger.info(f"[PEXELS_VIDEOS] API request started | query={query!r} | max_results={max_results}")

        try:
            response = self.session.get(PEXELS_VIDEOS_API, headers=headers, params=params)

            # Check for authentication errors (401, 403)
            if response.status_code == 401:
                logger.error(f"[PEXELS_VIDEOS] Authentication failed | query={query!r} | status=401 | Check API key validity")
                return []
            if response.status_code == 403:
                logger.error(f"[PEXELS_VIDEOS] Forbidden - API access denied | query={query!r} | status=403 | Check API key permissions")
                return []
            # Check for rate limit (429)
            if response.status_code == 429:
                logger.warning(f"[PEXELS_VIDEOS] Rate limit exceeded | query={query!r} | status=429 | Consider reducing request frequency")
                return []

            response.raise_for_status()
            data = response.json()

            # Get total hits from API response
            total_hits = data.get("total_results", 0)
            logger.info(f"[PEXELS_VIDEOS] API response received | query={query!r} | total_hits={total_hits} | per_page={max_results}")

            results = []
            for video in data.get("videos", []):
                duration = video.get("duration", 0)

                # Filter by duration
                if duration < self.min_duration or duration > self.max_duration:
                    continue

                # Get best quality video file
                video_files = video.get("video_files", [])
                if not video_files:
                    continue

                # Sort by quality (prefer HD)
                if self.prefer_hd:
                    video_files.sort(key=lambda x: x.get("height", 0), reverse=True)

                best_file = video_files[0]

                results.append(VideoResult(
                    id=str(video.get("id", "")),
                    source="pexels",
                    url=video.get("url", ""),
                    download_url=best_file.get("link", ""),
                    width=best_file.get("width", 0),
                    height=best_file.get("height", 0),
                    duration=duration,
                    quality=best_file.get("quality", "unknown"),
                    file_type=best_file.get("file_type", "mp4")
                ))

            logger.info(f"[PEXELS_VIDEOS] Search complete | query={query!r} | results={len(results)}/{total_hits} | duration_filter={self.min_duration}-{self.max_duration}s")
            return results

        except Exception as e:
            logger.error(f"[PEXELS_VIDEOS] API error | query={query!r} | error={e}")
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
        source_letter = video.source[0].lower()  # p=pexels
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
