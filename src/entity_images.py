"""
Image Downloader for Documentary B-Roll

Downloads high-resolution images from multiple sources.
Images are filtered to be > 1MB for quality assurance.

Supports:
- Google Images (via imagedl library)
- Bing Images (via imagedl library)
- Pexels Images (API)
- Pixabay Images (API)
- Unsplash (API)
"""

import os
import re
import json
import logging
import requests
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
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
    source: str  # pexels, pixabay, unsplash, google, bing
    url: str  # Web page URL
    download_url: str
    width: int
    height: int
    photographer: str
    description: str
    tags: List[str]
    estimated_size: int = 0  # bytes, if known
    file_path: str = ""  # Path after download
    entity_name: str = ""  # Entity this image represents
    entity_type: str = ""  # PERSON, GPE, ORG, etc.
    query: str = ""  # Search query used


@dataclass
class EntityImageResult:
    """Result of image search for an entity"""
    entity_name: str
    entity_type: str
    context: str
    query: str
    images: List[str] = field(default_factory=list)  # Downloaded file paths
    segment_indices: List[int] = field(default_factory=list)  # Segments mentioning this entity


class GoogleBingImageClient:
    """
    Wrapper for imagedl library to search Google/Bing images.
    Falls back gracefully if imagedl is not installed.
    """
    
    def __init__(
        self,
        output_dir: str = "./downloaded_images",
        source: str = "GoogleImageClient",
        min_size_mb: float = 1.0  # 1MB minimum for quality
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.source = source
        self.min_size = int(min_size_mb * 1024 * 1024)
        self.client = None
        self._init_client()
    
    def _init_client(self):
        """Initialize imagedl client"""
        import sys
        import importlib
        import importlib.util
        
        ImageClient = None
        
        # The pyimagedl package installs as 'imagedl' module
        # Import path: from imagedl.imagedl import ImageClient
        
        # First, find where the real imagedl package is installed
        # (not any local shadowing file)
        try:
            # Find the spec for the installed package
            spec = importlib.util.find_spec('imagedl')
            
            if spec is None:
                logger.info("imagedl package not found in Python path")
            elif spec.origin and 'site-packages' in spec.origin:
                # This is the real installed package
                logger.info(f"Found imagedl at: {spec.origin}")
                
                # Import using the spec
                imagedl_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(imagedl_module)
                
                # Now get the imagedl submodule
                if hasattr(imagedl_module, 'imagedl'):
                    inner_module = imagedl_module.imagedl
                    if hasattr(inner_module, 'ImageClient'):
                        ImageClient = inner_module.ImageClient
                        logger.info("✓ Loaded ImageClient from imagedl.imagedl")
            else:
                # It's a local file shadowing the package
                logger.info(f"Found local imagedl at: {spec.origin} (shadowing installed package)")
                
                # Try to import from site-packages directly
                for path in sys.path:
                    if 'site-packages' in path:
                        imagedl_path = Path(path) / 'imagedl'
                        if imagedl_path.exists() and imagedl_path.is_dir():
                            # Add to front of path temporarily
                            old_path = sys.path.copy()
                            sys.path.insert(0, str(imagedl_path.parent))
                            try:
                                # Force reimport
                                if 'imagedl' in sys.modules:
                                    del sys.modules['imagedl']
                                if 'imagedl.imagedl' in sys.modules:
                                    del sys.modules['imagedl.imagedl']
                                
                                from imagedl.imagedl import ImageClient as IC
                                ImageClient = IC
                                logger.info("✓ Loaded ImageClient from site-packages")
                            except ImportError as e:
                                logger.debug(f"Failed to import from site-packages: {e}")
                            finally:
                                sys.path = old_path
                            break
        except Exception as e:
            logger.debug(f"Error finding imagedl spec: {e}")
        
        # Fallback: try direct import (works if no shadowing)
        if not ImageClient:
            try:
                from imagedl.imagedl import ImageClient as IC
                ImageClient = IC
                logger.info("✓ Loaded ImageClient via direct import")
            except ImportError as e:
                logger.debug(f"Direct import failed: {e}")
        
        if not ImageClient:
            logger.info("Google/Bing image search not available - using Pexels/Pixabay")
            logger.info("To enable: pip install pyimagedl (and ensure no local imagedl.py exists)")
            self.client = None
            return
        
        try:
            self.client = ImageClient(
                image_source=self.source,
                init_image_client_cfg={
                    "work_dir": str(self.output_dir),
                    "disable_print": True,
                    "max_retries": 3,
                    "maintain_session": True
                },
                search_limits=20,
                num_threadings=3
            )
            logger.info(f"✓ Initialized {self.source} for Google/Bing image search")
        except Exception as e:
            logger.warning(f"Failed to initialize {self.source}: {e}")
            self.client = None
    
    def search_and_download(
        self,
        query: str,
        max_images: int = 5,
        entity_name: str = "",
        entity_type: str = ""
    ) -> List[str]:
        """
        Search and download images from Google/Bing.
        Keeps trying until finding enough images >= 1MB or exhausting all results.
        Cleans up search folders if no large images found.
        
        Args:
            query: Search query
            max_images: Target number of images (>= 1MB each)
            entity_name: Name of entity for metadata
            entity_type: Type of entity (PERSON, GPE, etc.)
        
        Returns:
            List of downloaded file paths (all >= 1MB)
        """
        if not self.client:
            return []
        
        # Track folders before search to identify new ones for cleanup
        existing_folders = set()
        source_folder = self.output_dir / self.source
        if source_folder.exists():
            existing_folders = set(f.name for f in source_folder.iterdir() if f.is_dir())
        
        try:
            # Search for LOTS of results - keep trying until we find enough large images
            search_limit = 500  # Get maximum candidates
            
            logger.info(f"Searching {self.source}: '{query}' (looking for {max_images} images >= 1MB)")
            
            # Try to filter for large images (format depends on source)
            # Google: tbs=isz:l (large), isz:m (medium)
            # Bing: imagesize:large
            size_filter = None
            if 'Google' in self.source:
                size_filter = {'size': 'large'}  # Request large images
            elif 'Bing' in self.source:
                size_filter = {'size': 'large'}
            
            image_infos = self.client.search(
                keyword=query,
                search_limits_overrides=search_limit,
                filters=size_filter
            )
            
            if not image_infos:
                logger.info(f"No Google/Bing results for '{query}'")
                return []
            
            logger.info(f"Found {len(image_infos)} image results, searching for >= 1MB files...")
            
            valid_paths = []
            downloaded_count = 0
            skipped_small = 0
            batch_size = 20  # Download in larger batches
            
            # Process ALL results until we have enough large images
            total_batches = (len(image_infos) + batch_size - 1) // batch_size
            
            for batch_num in range(total_batches):
                if len(valid_paths) >= max_images:
                    break
                
                batch_start = batch_num * batch_size
                batch = image_infos[batch_start:batch_start + batch_size]
                
                if not batch:
                    break
                
                logger.info(f"Batch {batch_num + 1}/{total_batches}: Downloading {len(batch)} images...")
                
                # Download this batch
                try:
                    downloaded = self.client.download(
                        batch,
                        num_threadings_overrides=5
                    )
                except Exception as e:
                    logger.warning(f"Batch download error: {e}")
                    continue
                
                if not downloaded:
                    continue
                
                downloaded_count += len(downloaded)
                
                # Check each downloaded image
                for info in downloaded:
                    if len(valid_paths) >= max_images:
                        break
                    
                    file_path = info.get('file_path', '')
                    
                    if not file_path:
                        continue
                        
                    file_path = Path(file_path)
                    
                    if not file_path.exists():
                        # Try common extensions
                        for ext in ['.jpg', '.jpeg', '.png', '.webp', '.gif']:
                            test_path = file_path.with_suffix(ext)
                            if test_path.exists():
                                file_path = test_path
                                break
                    
                    if not file_path.exists():
                        continue
                    
                    size = file_path.stat().st_size
                    size_mb = size / (1024 * 1024)
                    
                    if size >= self.min_size:
                        valid_paths.append(str(file_path))
                        logger.info(f"  ✓ KEPT: {file_path.name} ({size_mb:.2f}MB) [{len(valid_paths)}/{max_images}]")
                        
                        # Save entity metadata
                        if entity_name:
                            meta = {
                                'entity_name': entity_name,
                                'entity_type': entity_type,
                                'query': query,
                                'source': self.source,
                                'file_size_mb': round(size_mb, 2),
                                'original_info': {
                                    'identifier': info.get('identifier', ''),
                                    'url': info.get('candidate_urls', [''])[0] if info.get('candidate_urls') else ''
                                }
                            }
                            meta_path = file_path.with_suffix('.entity.json')
                            with open(meta_path, 'w') as f:
                                json.dump(meta, f, indent=2)
                    else:
                        skipped_small += 1
                        # Delete small files immediately
                        try:
                            file_path.unlink()
                        except:
                            pass
                
                # Progress update (only if still searching)
                remaining = max_images - len(valid_paths)
                if remaining > 0 and batch_num < total_batches - 1:
                    logger.info(f"  Progress: {len(valid_paths)}/{max_images} (need {remaining} more, checked {downloaded_count} images)")
            
            # Find and clean up new folders created by this search
            new_folders = []
            if source_folder.exists():
                current_folders = set(f.name for f in source_folder.iterdir() if f.is_dir())
                new_folder_names = current_folders - existing_folders
                new_folders = [source_folder / name for name in new_folder_names]
            
            # Summary
            if valid_paths:
                logger.info(f"✓ {self.source}: Found {len(valid_paths)} images >= 1MB (checked {downloaded_count} total)")
            else:
                logger.warning(f"✗ {self.source}: No images >= 1MB found after checking {downloaded_count} images")
            
            # Always clean up folders with no valid images
            self._cleanup_empty_folders(new_folders)
            
            return valid_paths
            
        except Exception as e:
            logger.error(f"Google/Bing image search error: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def _cleanup_empty_folders(self, folders: List[Path]):
        """Remove folders that are empty or only contain .pkl files (no images)"""
        import shutil
        
        for folder_path in folders:
            if not folder_path.exists():
                continue
            
            try:
                # Check if folder has any large image files
                files = list(folder_path.iterdir())
                large_images = []
                for f in files:
                    if f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.webp', '.gif'}:
                        if f.stat().st_size >= self.min_size:
                            large_images.append(f)
                
                if not large_images:
                    # No large image files - delete the folder
                    shutil.rmtree(folder_path)
                    logger.info(f"  🗑️ Cleaned up folder (no large images): {folder_path.name}")
            except Exception as e:
                logger.debug(f"Could not clean up folder {folder_path}: {e}")


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


# =============================================================================
# ENTITY IMAGE SEARCH
# =============================================================================

def build_entity_query(entity: Dict, topic: str = "") -> str:
    """
    Build a trimmed search query from entity + context + topic.
    
    Args:
        entity: Dict with 'text', 'type', 'context' keys
        topic: Overall documentary topic
    
    Returns:
        Trimmed search query string
    """
    name = entity.get('text', '')
    context = entity.get('context', '')
    etype = entity.get('type', '')
    
    if not name:
        return ""
    
    # Build query parts
    parts = [name]
    
    # Add context (trimmed)
    if context:
        # Take first 2-3 words of context
        context_words = context.split()[:3]
        parts.extend(context_words)
    
    # Add topic keywords (trimmed)
    if topic:
        # Extract 2-3 key words from topic
        topic_words = topic.replace('documentary', '').replace('footage', '').split()
        topic_words = [w for w in topic_words if len(w) > 3][:2]
        parts.extend(topic_words)
    
    # Join and clean
    query = ' '.join(parts)
    
    # Limit length
    if len(query) > 60:
        query = ' '.join(query.split()[:6])
    
    return query


def download_entity_images(
    entities: List[Dict],
    output_dir: str,
    topic: str = "",
    images_per_entity: int = 3,
    min_size_mb: float = 1.0,
    use_google: bool = True,
    use_stock_apis: bool = True,
    pexels_key: str = None,
    pixabay_key: str = None
) -> Dict[str, EntityImageResult]:
    """
    Download images for entities extracted from voiceover.
    
    Args:
        entities: List of entity dicts with 'text', 'type', 'context' keys
        output_dir: Directory to save images
        topic: Documentary topic for query building
        images_per_entity: Number of images to download per entity (2-5)
        min_size_mb: Minimum file size (default 1MB)
        use_google: Use Google/Bing image search (via imagedl library)
        use_stock_apis: Use Pexels/Pixabay APIs as fallback
    
    Returns:
        Dict mapping entity name to EntityImageResult
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    results: Dict[str, EntityImageResult] = {}
    
    # Initialize clients
    google_client = None
    bing_client = None
    stock_downloader = None
    
    if use_google:
        google_client = GoogleBingImageClient(
            output_dir=str(output_path),
            source="GoogleImageClient",
            min_size_mb=min_size_mb
        )
        # Also initialize Bing as backup
        bing_client = GoogleBingImageClient(
            output_dir=str(output_path),
            source="BingImageClient",
            min_size_mb=min_size_mb
        )
    
    if use_stock_apis:
        stock_downloader = ImageDownloader(
            output_dir=str(output_path),
            pexels_key=pexels_key,
            pixabay_key=pixabay_key,
            min_size_mb=min_size_mb
        )
    
    # Check which sources are available
    has_google = google_client and google_client.client
    has_bing = bing_client and bing_client.client
    has_stock = stock_downloader and any([
        stock_downloader.pexels_key,
        stock_downloader.pixabay_key
    ])
    
    if not has_google and not has_bing and not has_stock:
        logger.warning("No image sources available. Install pyimagedl or set API keys.")
        return results
    
    # Process each entity
    total_downloaded = 0
    
    for entity in entities:
        entity_name = entity.get('text', '')
        entity_type = entity.get('type', '')
        context = entity.get('context', '')
        
        if not entity_name:
            continue
        
        # Skip if already processed (same entity name)
        if entity_name in results:
            continue
        
        # Build search query
        query = build_entity_query(entity, topic)
        
        if not query:
            continue
        
        logger.info(f"Searching images for entity: {entity_name} ({entity_type})")
        logger.info(f"  Query: '{query}'")
        
        downloaded_paths = []
        
        # Try Google first (prioritized)
        if has_google and len(downloaded_paths) < images_per_entity:
            paths = google_client.search_and_download(
                query=query,
                max_images=images_per_entity,
                entity_name=entity_name,
                entity_type=entity_type
            )
            downloaded_paths.extend(paths)
        
        # Try Bing if Google didn't find enough
        if has_bing and len(downloaded_paths) < images_per_entity:
            remaining = images_per_entity - len(downloaded_paths)
            logger.info(f"  Trying Bing for {remaining} more images...")
            paths = bing_client.search_and_download(
                query=query,
                max_images=remaining,
                entity_name=entity_name,
                entity_type=entity_type
            )
            downloaded_paths.extend(paths)
        
        # Fallback to stock APIs if still needed
        if has_stock and len(downloaded_paths) < images_per_entity:
            remaining = images_per_entity - len(downloaded_paths)
            paths = stock_downloader.search_and_download(
                query=query,
                max_images=remaining,
                check_size=True
            )
            
            # Save entity metadata for stock images too
            for path in paths:
                meta = {
                    'entity_name': entity_name,
                    'entity_type': entity_type,
                    'query': query,
                    'source': 'stock_api',
                    'file_size_mb': round(Path(path).stat().st_size / 1024 / 1024, 2)
                }
                meta_path = Path(path).with_suffix('.entity.json')
                with open(meta_path, 'w') as f:
                    json.dump(meta, f, indent=2)
            
            downloaded_paths.extend(paths)
        
        # Store result
        if downloaded_paths:
            results[entity_name] = EntityImageResult(
                entity_name=entity_name,
                entity_type=entity_type,
                context=context,
                query=query,
                images=downloaded_paths
            )
            total_downloaded += len(downloaded_paths)
            logger.info(f"  ✓ Downloaded {len(downloaded_paths)} images for '{entity_name}'")
        else:
            logger.warning(f"  ⚠ No images found for '{entity_name}'")
        
        # Rate limiting
        time.sleep(0.5)
    
    logger.info(f"Entity images: {total_downloaded} total for {len(results)} entities")
    return results


def map_entities_to_segments(
    entities: List[Dict],
    voiceover_segments: List[Dict]
) -> Dict[str, List[int]]:
    """
    Map entity names to segment indices where they appear.
    
    Returns:
        Dict mapping entity name to list of segment indices
    """
    entity_segments: Dict[str, List[int]] = {}
    
    for entity in entities:
        entity_name = entity.get('text', '')
        if not entity_name:
            continue
        
        entity_segments[entity_name] = []
        
        for i, seg in enumerate(voiceover_segments):
            seg_text = seg.get('text', '').lower() if isinstance(seg, dict) else getattr(seg, 'text', '').lower()
            
            if entity_name.lower() in seg_text:
                entity_segments[entity_name].append(i)
    
    return entity_segments