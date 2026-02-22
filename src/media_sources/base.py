"""
Base class for all media source clients.

Consolidates duplicate initialization, rate limiting, and HTTP session management
from ImageDownloader (lines 578-597) and StockVideoDownloader (lines 1019-1034).

Created Jan 7, 2026 as part of entity_images.py refactoring.
"""

from __future__ import annotations

import logging
import time
import requests
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..config import Config

logger = logging.getLogger(__name__)


class BaseMediaClient(ABC):
    """Abstract base class for all media source clients (images and videos)."""

    def __init__(
        self,
        config: 'Config',
        output_dir: str,
        min_size_mb: float = 1.0,
        download_timeout: int = 30,
        rate_limit_delay: float = 0.3,
        user_agent: str = "Voiceover-Matcher/2.3"
    ):
        """
        Initialize media client with shared configuration.

        Args:
            config: Configuration object
            output_dir: Directory for downloaded files
            min_size_mb: Minimum file size in MB for quality filtering
            download_timeout: Timeout for HTTP downloads (seconds)
            rate_limit_delay: Minimum delay between API requests (seconds)
            user_agent: User-Agent header for HTTP requests
        """
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.min_size = int(min_size_mb * 1024 * 1024)
        self.download_timeout = download_timeout

        # HTTP session with proper headers
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": user_agent})

        # Rate limiting state
        self._last_request_time = 0
        self._min_interval = rate_limit_delay

    def _rate_limit(self):
        """
        Enforce rate limiting between API requests.

        Ensures minimum delay between consecutive requests to avoid
        hitting API rate limits.

        Consolidated from:
        - ImageDownloader._rate_limit() (entity_images.py lines 591-597)
        - StockVideoDownloader._rate_limit() (entity_images.py lines 1028-1034)
        """
        now = time.time()
        elapsed = now - self._last_request_time
        if elapsed < self._min_interval:
            sleep_duration = self._min_interval - elapsed
            logger.debug(f"[RATE-LIMIT] Sleeping {sleep_duration:.2f}s to respect API rate limit (interval={self._min_interval}s)")
            time.sleep(sleep_duration)
        self._last_request_time = time.time()

    def download_with_timeout(
        self,
        url: str,
        output_path: Path,
        timeout: Optional[int] = None
    ) -> bool:
        """
        Download file from URL with timeout and size validation.

        Args:
            url: Download URL
            output_path: Local path to save file
            timeout: Optional timeout override

        Returns:
            True if download succeeded, False otherwise
        """
        timeout = timeout or self.download_timeout

        try:
            response = self.session.get(url, timeout=timeout, stream=True)
            response.raise_for_status()

            # Write to file
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            # Check file size
            file_size = output_path.stat().st_size
            if file_size < self.min_size:
                logger.debug(f"File too small ({file_size} bytes < {self.min_size}): {output_path}")
                output_path.unlink()
                return False

            return True

        except Exception as e:
            logger.debug(f"Download failed: {e}")
            if output_path.exists():
                output_path.unlink()
            return False

    @abstractmethod
    def search(self, query: str, max_results: int) -> List[dict]:
        """
        Search for media matching query.

        Args:
            query: Search query
            max_results: Maximum results to return

        Returns:
            List of result dictionaries (format varies by source)
        """
        pass

    def cleanup(self):
        """Clean up resources (close session, etc.)"""
        if hasattr(self, 'session'):
            self.session.close()
