"""
Image Downloader for Documentary B-Roll

Downloads high-resolution images from multiple sources.
Images are filtered to be > 1MB for quality assurance.

Supports:
- Pexels Images
- Pixabay Images
- Unsplash (if API key available)
"""

import os
import re
import json
import logging
import requests
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)

# Minimum file size in bytes (1MB)
MIN_IMAGE_SIZE = 1 * 1024 * 1024  # 1MB

# API endpoints
PEXELS_IMAGES_API = "https://api.pexels.com/v1/search"
PIXABAY_IMAGES_API = "https://pixabay.com/api/"
UNSPLASH_API = "https://api.unsplash.com/search/photos"


@dataclass
class ImageResult:
    """Represents a downloadable image"""
    id: str
    source: str  # pexels, pixabay, unsplash
    url: str  # Web page URL
    download_url: str
    width: int
    height: int
    photographer: str
    description: str
    tags: List[str]
    estimated_size: int = 0  # bytes, if known


class ImageDownloader:
    """
    Download high-resolution images from multiple sources.
    
    Features:
    - Multi-source search (Pexels, Pixabay, Unsplash)
    - Size filtering (minimum 1MB)
    - Automatic metadata tagging
    - Duplicate detection
    """
    
    def __init__(
        self,
        output_dir: str = "./downloaded_images",
        pexels_key: str = None,
        pixabay_key: str = None,
        unsplash_key: str = None,
        min_size_mb: float = 1.0
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # API keys
        self.pexels_key = pexels_key or os.getenv("PEXELS_API_KEY")
        self.pixabay_key = pixabay_key or os.getenv("PIXABAY_API_KEY")
        self.unsplash_key = unsplash_key or os.getenv("UNSPLASH_API_KEY")
        
        # Size filter
        self.min_size = int(min_size_mb * 1024 * 1024)
        
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Voiceover-Matcher/2.3 ImageDownloader"
        })
        
        # Rate limiting
        self._last_request_time = 0
        self._min_interval = 0.3
        
        # Tracking
        self.downloaded_files: List[Dict] = []
        self.failed_downloads: List[str] = []
    
    def _rate_limit(self):
        """Enforce rate limiting"""
        now = time.time()
        elapsed = now - self._last_request_time
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_request_time = time.time()
    
    def search_pexels(self, query: str, per_page: int = 20) -> List[ImageResult]:
        """Search Pexels for images"""
        if not self.pexels_key:
            return []
        
        self._rate_limit()
        
        headers = {"Authorization": self.pexels_key}
        params = {
            "query": query,
            "per_page": per_page,
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
    
    def search_pixabay(self, query: str, per_page: int = 20) -> List[ImageResult]:
        """Search Pixabay for images"""
        if not self.pixabay_key:
            return []
        
        self._rate_limit()
        
        params = {
            "key": self.pixabay_key,
            "q": query,
            "per_page": per_page,
            "image_type": "photo",
            "orientation": "horizontal",
            "safesearch": "true",
            "min_width": 1920  # Ensure high resolution
        }
        
        try:
            response = self.session.get(PIXABAY_IMAGES_API, params=params)
            response.raise_for_status()
            data = response.json()
            
            results = []
            for hit in data.get("hits", []):
                # largeImageURL is typically 1280px, but we want bigger
                # fullHDURL (1920px) or imageURL (original) may not be available on free tier
                download_url = hit.get("largeImageURL", "")
                
                if not download_url:
                    continue
                
                # Estimate size from dimensions (rough: 0.5 bytes per pixel for JPEG)
                estimated_size = int(hit.get("imageWidth", 0) * hit.get("imageHeight", 0) * 0.5)
                
                tags = [t.strip() for t in hit.get("tags", "").split(",")]
                tags.extend(["pixabay", "stock_image"])
                
                results.append(ImageResult(
                    id=f"pixabay_{hit['id']}",
                    source="pixabay",
                    url=hit.get("pageURL", ""),
                    download_url=download_url,
                    width=hit.get("imageWidth", 0),
                    height=hit.get("imageHeight", 0),
                    photographer=hit.get("user", "Unknown"),
                    description="",
                    tags=tags,
                    estimated_size=estimated_size
                ))
            
            logger.info(f"Pixabay images '{query}': {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Pixabay image search error: {e}")
            return []
    
    def search_unsplash(self, query: str, per_page: int = 20) -> List[ImageResult]:
        """Search Unsplash for images"""
        if not self.unsplash_key:
            return []
        
        self._rate_limit()
        
        headers = {"Authorization": f"Client-ID {self.unsplash_key}"}
        params = {
            "query": query,
            "per_page": per_page,
            "orientation": "landscape"
        }
        
        try:
            response = self.session.get(UNSPLASH_API, headers=headers, params=params)
            response.raise_for_status()
            data = response.json()
            
            results = []
            for photo in data.get("results", []):
                urls = photo.get("urls", {})
                # Get full resolution
                download_url = urls.get("full", urls.get("regular", ""))
                
                if not download_url:
                    continue
                
                results.append(ImageResult(
                    id=f"unsplash_{photo['id']}",
                    source="unsplash",
                    url=photo.get("links", {}).get("html", ""),
                    download_url=download_url,
                    width=photo.get("width", 0),
                    height=photo.get("height", 0),
                    photographer=photo.get("user", {}).get("name", "Unknown"),
                    description=photo.get("description", "") or photo.get("alt_description", ""),
                    tags=["unsplash", "stock_image"]
                ))
            
            logger.info(f"Unsplash images '{query}': {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Unsplash image search error: {e}")
            return []
    
    def search_all(self, query: str, per_source: int = 10) -> List[ImageResult]:
        """Search all available sources"""
        all_results = []
        
        all_results.extend(self.search_pexels(query, per_source))
        all_results.extend(self.search_pixabay(query, per_source))
        all_results.extend(self.search_unsplash(query, per_source))
        
        # Sort by resolution (width * height) descending
        all_results.sort(key=lambda x: x.width * x.height, reverse=True)
        
        return all_results
    
    def download_image(self, image: ImageResult, check_size: bool = True) -> Optional[str]:
        """
        Download a single image.
        
        Args:
            image: ImageResult to download
            check_size: If True, verify file size meets minimum (1MB)
        
        Returns:
            Path to downloaded file, or None if failed/too small
        """
        # Generate filename
        safe_photographer = re.sub(r'[^\w\-]', '_', image.photographer)[:20]
        ext = Path(image.download_url.split('?')[0]).suffix or '.jpg'
        filename = f"{image.id}_{safe_photographer}{ext}"
        
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
            response = self.session.get(image.download_url, stream=True, timeout=60)
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
        check_size: bool = True
    ) -> List[str]:
        """
        Search and download images for a query.
        
        Args:
            query: Search keywords
            max_images: Maximum images to download
            check_size: Enforce minimum size requirement
        
        Returns:
            List of downloaded file paths
        """
        # Search all sources
        results = self.search_all(query, per_source=max_images * 2)
        
        downloaded = []
        for image in results:
            if len(downloaded) >= max_images:
                break
            
            path = self.download_image(image, check_size=check_size)
            if path:
                downloaded.append(path)
        
        return downloaded


def download_images(
    keywords: List[str],
    output_dir: str,
    per_keyword: int = 3,
    min_size_mb: float = 1.0,
    pexels_key: str = None,
    pixabay_key: str = None,
    unsplash_key: str = None
) -> Tuple[List[str], Dict[str, int]]:
    """
    Convenience function to download images for multiple keywords.
    
    Args:
        keywords: List of search keywords
        output_dir: Directory to save images
        per_keyword: Images to download per keyword
        min_size_mb: Minimum file size in MB (default 1MB)
    
    Returns:
        Tuple of (list of downloaded paths, dict of keyword -> count)
    """
    downloader = ImageDownloader(
        output_dir=output_dir,
        pexels_key=pexels_key,
        pixabay_key=pixabay_key,
        unsplash_key=unsplash_key,
        min_size_mb=min_size_mb
    )
    
    # Check if any API keys available
    if not any([downloader.pexels_key, downloader.pixabay_key, downloader.unsplash_key]):
        logger.warning("No image API keys available (PEXELS_API_KEY, PIXABAY_API_KEY, UNSPLASH_API_KEY)")
        return [], {}
    
    all_downloaded = []
    keyword_counts = {}
    
    for keyword in keywords:
        logger.info(f"Images: Searching '{keyword}'...")
        
        paths = downloader.search_and_download(
            query=keyword,
            max_images=per_keyword,
            check_size=True
        )
        
        keyword_counts[keyword] = len(paths)
        all_downloaded.extend(paths)
        
        time.sleep(0.3)
    
    logger.info(f"Images: Downloaded {len(all_downloaded)} images total (>= {min_size_mb}MB each)")
    return all_downloaded, keyword_counts
