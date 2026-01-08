"""
Pexels image search and download client.

Extracted from entity_images.py ImageDownloader.search_pexels() (lines 599-644)
and download logic (Jan 7, 2026).
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import List, Optional, TYPE_CHECKING

from ..base import BaseMediaClient
from ..models import ImageResult

if TYPE_CHECKING:
    from ...config import Config

logger = logging.getLogger(__name__)

PEXELS_IMAGES_API = "https://api.pexels.com/v1/search"


class PexelsImageClient(BaseMediaClient):
    """Client for searching and downloading images from Pexels API."""

    def __init__(
        self,
        config: 'Config',
        output_dir: str,
        api_key: Optional[str] = None,
        min_size_mb: float = 1.0,
        download_timeout: int = 60
    ):
        """
        Initialize Pexels image client.

        Args:
            config: Configuration object
            output_dir: Directory for downloaded images
            api_key: Pexels API key (falls back to env var)
            min_size_mb: Minimum file size in MB
            download_timeout: HTTP timeout for downloads
        """
        super().__init__(
            config=config,
            output_dir=output_dir,
            min_size_mb=min_size_mb,
            download_timeout=download_timeout,
            rate_limit_delay=0.3,
            user_agent="Voiceover-Matcher/2.3 PexelsImageClient"
        )

        self.api_key = api_key or os.getenv("PEXELS_API_KEY")

        # Track downloads
        self.downloaded_files: List[dict] = []
        self.failed_downloads: List[str] = []

    def search(self, query: str, max_results: int = 20) -> List[ImageResult]:
        """
        Search Pexels for images.

        Args:
            query: Search query
            max_results: Maximum results to return

        Returns:
            List of ImageResult objects
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

        try:
            response = self.session.get(PEXELS_IMAGES_API, headers=headers, params=params)
            response.raise_for_status()
            data = response.json()

            results = []
            for photo in data.get("photos", []):
                # Get original size URL (largest)
                src = photo.get("src", {})
                download_url = src.get("original", src.get("large2x", src.get("large", "")))

                if not download_url:
                    continue

                results.append(ImageResult(
                    id=f"pexels_{photo['id']}",
                    source="pexels",
                    url=photo.get("url", ""),
                    download_url=download_url,
                    width=photo.get("width", 0),
                    height=photo.get("height", 0),
                    photographer=photo.get("photographer", "Unknown"),
                    description=photo.get("alt", ""),
                    tags=["pexels", "stock_image"]
                ))

            logger.info(f"Pexels images '{query}': {len(results)} results")
            return results

        except Exception as e:
            logger.error(f"Pexels image search error: {e}")
            return []

    def download_image(
        self,
        image: ImageResult,
        check_size: bool = True,
        entity_name: str = "",
        entity_type: str = ""
    ) -> Optional[str]:
        """
        Download a single image.

        Args:
            image: ImageResult to download
            check_size: If True, verify file size meets minimum
            entity_name: Entity name for metadata
            entity_type: Entity type for metadata

        Returns:
            Path to downloaded file, or None if failed/too small
        """
        # Generate short filename for NLE compatibility
        # Format: {id8}{ext} = 12 chars max
        short_id = str(image.id)[-8:] if len(str(image.id)) > 8 else str(image.id)
        ext = Path(image.download_url.split('?')[0]).suffix or '.jpg'
        filename = f"{short_id}{ext}"

        output_path = self.output_dir / filename

        # Skip if already exists
        if output_path.exists():
            file_size = output_path.stat().st_size
            if check_size and file_size < self.min_size:
                logger.debug(f"Existing file too small: {filename} ({file_size/1024/1024:.2f}MB)")
                return None
            return str(output_path)

        self._rate_limit()

        try:
            # First, do a HEAD request to check size if available
            if check_size:
                try:
                    head_response = self.session.head(image.download_url, timeout=10)
                    content_length = int(head_response.headers.get('content-length', 0))
                    if content_length > 0 and content_length < self.min_size:
                        logger.debug(f"Image too small (pre-check): {filename} ({content_length/1024/1024:.2f}MB)")
                        return None
                except:
                    pass  # Continue with download anyway

            # Download
            response = self.session.get(image.download_url, stream=True, timeout=self.download_timeout)
            response.raise_for_status()

            # Check content-length header
            content_length = int(response.headers.get('content-length', 0))
            if check_size and content_length > 0 and content_length < self.min_size:
                logger.debug(f"Image too small: {filename} ({content_length/1024/1024:.2f}MB)")
                return None

            # Download to temp file first
            temp_path = output_path.with_suffix('.tmp')
            downloaded_size = 0

            with open(temp_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    downloaded_size += len(chunk)

            # Check final size
            if check_size and downloaded_size < self.min_size:
                logger.debug(f"Image too small after download: {filename} ({downloaded_size/1024/1024:.2f}MB)")
                temp_path.unlink()
                return None

            # Rename to final
            temp_path.rename(output_path)

            # Save metadata
            metadata = {
                "source": image.source,
                "id": image.id,
                "url": image.url,
                "photographer": image.photographer,
                "description": image.description,
                "resolution": f"{image.width}x{image.height}",
                "tags": image.tags,
                "file_size_mb": round(downloaded_size / 1024 / 1024, 2),
                "is_stock_image": True,
                "downloaded_at": datetime.now().isoformat()
            }

            # Save entity metadata if provided
            if entity_name:
                metadata["entity_name"] = entity_name
                metadata["entity_type"] = entity_type
                meta_path = output_path.with_suffix('.entity.json')
            else:
                meta_path = output_path.with_suffix('.meta.json')

            with open(meta_path, 'w') as f:
                json.dump(metadata, f, indent=2)

            self.downloaded_files.append({
                "path": str(output_path),
                "metadata": metadata
            })

            logger.info(f"Downloaded image: {filename} ({downloaded_size/1024/1024:.2f}MB)")
            return str(output_path)

        except Exception as e:
            logger.error(f"Image download failed for {image.id}: {e}")
            self.failed_downloads.append(image.id)
            return None

    def search_and_download(
        self,
        query: str,
        max_images: int = 5,
        entity_name: str = "",
        entity_type: str = ""
    ) -> List[str]:
        """
        Search and download images for a query.

        Args:
            query: Search keywords
            max_images: Maximum images to download
            entity_name: Entity name for metadata
            entity_type: Entity type for metadata

        Returns:
            List of downloaded file paths
        """
        # Search
        results = self.search(query, max_results=max_images * 2)

        # Download
        downloaded = []
        for image in results:
            if len(downloaded) >= max_images:
                break

            path = self.download_image(
                image,
                check_size=True,
                entity_name=entity_name,
                entity_type=entity_type
            )
            if path:
                downloaded.append(path)

        return downloaded
