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
        min_size_mb: float = 1.0,  # 1MB minimum for quality
        download_timeout: int = 10,  # Seconds per image
        max_search_time: int = 300,  # Max seconds total
        max_results_to_check: int = 500,  # Max images to check
        search_until_found: bool = True  # Keep searching until target found
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.source = source
        self.min_size = int(min_size_mb * 1024 * 1024)
        self.download_timeout = download_timeout
        self.max_search_time = max_search_time
        self.max_results_to_check = max_results_to_check
        self.search_until_found = search_until_found
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
                    "max_retries": 2,  # Reduce retries for faster timeout
                    "maintain_session": True
                },
                search_limits=20,
                num_threadings=1  # Single thread to avoid alive_progress conflicts
            )
            logger.info(f"✓ Initialized {self.source} for Google/Bing image search")
        except Exception as e:
            logger.warning(f"Failed to initialize {self.source}: {e}")
            self.client = None
    
    def _search_with_timeout(self, keyword: str, search_limit: int, timeout: int = 60) -> List[Dict]:
        """
        Search with timeout to prevent hanging.
        
        The pyimagedl library's search() can hang indefinitely, especially Bing.
        This wraps it in a thread with timeout.
        """
        import threading
        import time as time_module
        
        if not self.client:
            return []
        
        result = []
        error_occurred = [None]  # Use list to allow modification in nested function
        
        def do_search():
            nonlocal result
            try:
                result = self.client.search(
                    keyword=keyword,
                    search_limits_overrides=search_limit
                )
            except Exception as e:
                error_occurred[0] = e
                result = []
        
        # Run search in thread with timeout
        thread = threading.Thread(target=do_search)
        thread.daemon = True  # Daemon thread will be killed when main exits
        thread.start()
        thread.join(timeout=timeout)
        
        if thread.is_alive():
            logger.warning(f"Search timed out after {timeout}s, skipping...")
            # Wait a bit for alive_progress to clean up
            time_module.sleep(2)
            return []
        
        if error_occurred[0]:
            error_msg = str(error_occurred[0])
            if "alive_progress" in error_msg.lower() or "nested" in error_msg.lower():
                logger.warning(f"Search error: {error_msg}")
                # Wait for previous search to clean up
                time_module.sleep(3)
            else:
                logger.warning(f"Search error: {error_msg}")
            return []
        
        return result if result else []
    
    def _reinitialize_client(self):
        """Reinitialize the imagedl client to clear any stuck state."""
        logger.debug(f"Reinitializing {self.source}...")
        self.client = None
        self._init_client()
    
    def search_and_download(
        self,
        query: str,
        max_images: int = 5,
        entity_name: str = "",
        entity_type: str = ""
    ) -> List[str]:
        """
        Search and download images from Google/Bing.
        Downloads images ONE AT A TIME with proper timeout handling.
        
        If search_until_found=True, will try multiple search variations
        until finding the required number of images.
        
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
        # Use short source folder names for NLE compatibility
        short_source = "g" if "google" in self.source.lower() else "b"  # g=Google, b=Bing
        source_folder = self.output_dir / short_source
        if source_folder.exists():
            existing_folders = set(f.name for f in source_folder.iterdir() if f.is_dir())
        
        import time as time_module
        
        valid_paths = []
        total_checked = 0
        total_skipped = 0
        start_time = time_module.time()
        
        # Generate query variations for search_until_found mode
        query_variations = [query]
        if self.search_until_found:
            # Add variations to try if first search doesn't find enough
            base_words = query.split()
            if len(base_words) > 2:
                # Try with fewer words
                query_variations.append(' '.join(base_words[:3]))
                query_variations.append(' '.join(base_words[:2]))
            # Try with "high resolution" or "HD" added
            query_variations.append(f"{query} high resolution")
            query_variations.append(f"{query} HD photo")
            # Try with just entity name if provided
            if entity_name and entity_name not in query_variations:
                query_variations.append(f"{entity_name} photo")
                query_variations.append(f"{entity_name} image HD")
        
        # Create output folder for this query (keep short for NLE compatibility)
        safe_query = "".join(c if c.isalnum() else "" for c in query)[:8]
        query_folder = source_folder / safe_query
        query_folder.mkdir(parents=True, exist_ok=True)
        
        seen_urls = set()  # Track URLs we've already tried
        consecutive_failures = 0  # Track consecutive search failures
        
        for query_idx, current_query in enumerate(query_variations):
            if len(valid_paths) >= max_images:
                break
            
            # Check overall time limit
            elapsed = time_module.time() - start_time
            if elapsed > self.max_search_time:
                logger.warning(f"Search timed out after {elapsed:.0f}s total")
                break
            
            # If we've had too many consecutive failures, reinitialize client
            if consecutive_failures >= 2:
                logger.info(f"  Reinitializing {self.source} after consecutive failures...")
                self._reinitialize_client()
                consecutive_failures = 0
                time_module.sleep(2)  # Give it time to stabilize
                
                if not self.client:
                    logger.warning(f"  Failed to reinitialize {self.source}, stopping")
                    break
            
            try:
                # Search for results with timeout (search can hang too!)
                search_limit = self.max_results_to_check
                
                if query_idx == 0:
                    logger.info(f"Searching {self.source}: '{current_query}' (looking for {max_images} images >= 1MB)")
                else:
                    logger.info(f"  Trying variation {query_idx + 1}: '{current_query}'")
                
                # Wrap search in timeout - pyimagedl search can hang
                image_infos = self._search_with_timeout(
                    keyword=current_query,
                    search_limit=search_limit,
                    timeout=60  # 60 second timeout for search
                )
                
                if not image_infos:
                    logger.info(f"  No results for '{current_query}'")
                    consecutive_failures += 1
                    continue
                
                # Reset failure counter on success
                consecutive_failures = 0
                
                if query_idx == 0:
                    logger.info(f"Found {len(image_infos)} image results, downloading one at a time...")
                else:
                    logger.info(f"  Found {len(image_infos)} more results")
                
                # Download images ONE AT A TIME with timeout
                for idx, info in enumerate(image_infos):
                    if len(valid_paths) >= max_images:
                        break
                    
                    # Check time limit
                    elapsed = time_module.time() - start_time
                    if elapsed > self.max_search_time:
                        logger.warning(f"Search timed out after {elapsed:.0f}s")
                        break
                    
                    # Get image URL
                    urls = info.get('candidate_urls', [])
                    if not urls:
                        continue
                    
                    url = urls[0]
                    
                    # Skip if we've already tried this URL
                    if url in seen_urls:
                        continue
                    seen_urls.add(url)
                    
                    total_checked += 1
                    
                    # Download with timeout using requests directly
                    try:
                        filepath = self._download_single_image(
                            url=url,
                            output_folder=query_folder,
                            index=total_checked,
                            timeout=self.download_timeout
                        )
                        
                        if filepath and filepath.exists():
                            size = filepath.stat().st_size
                            size_mb = size / (1024 * 1024)
                            
                            if size >= self.min_size:
                                valid_paths.append(str(filepath))
                                logger.info(f"  ✓ KEPT: {filepath.name} ({size_mb:.2f}MB) [{len(valid_paths)}/{max_images}]")
                                
                                # Save entity metadata
                                if entity_name:
                                    meta = {
                                        'entity_name': entity_name,
                                        'entity_type': entity_type,
                                        'query': current_query,
                                        'source': self.source,
                                        'file_size_mb': round(size_mb, 2),
                                        'original_url': url
                                    }
                                    meta_path = filepath.with_suffix('.entity.json')
                                    with open(meta_path, 'w') as f:
                                        json.dump(meta, f, indent=2)
                            else:
                                total_skipped += 1
                                # Delete small files
                                try:
                                    filepath.unlink()
                                except:
                                    pass
                    except Exception as e:
                        logger.debug(f"  Skip image {total_checked}: {e}")
                        continue
                    
                    # Progress every 20 images
                    if total_checked % 20 == 0:
                        logger.info(f"  Progress: {len(valid_paths)}/{max_images} (checked {total_checked}, {total_skipped} too small)")
                
                # If search_until_found is False, stop after first query
                if not self.search_until_found:
                    break
                    
            except Exception as e:
                logger.error(f"Search error for '{current_query}': {e}")
                continue
        
        # Clean up if no valid images
        if not valid_paths:
            logger.warning(f"✗ {self.source}: No images >= 1MB found after checking {total_checked}")
            self._cleanup_empty_folders([query_folder])
        else:
            logger.info(f"✓ {self.source}: Found {len(valid_paths)} images >= 1MB (checked {total_checked})")
        
        return valid_paths
    
    def _download_single_image(
        self,
        url: str,
        output_folder: Path,
        index: int,
        timeout: int = 10
    ) -> Optional[Path]:
        """Download a single image with timeout using requests directly."""
        import requests
        
        try:
            # Request with timeout
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'image/webp,image/apng,image/*,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.9',
            }
            
            response = requests.get(url, headers=headers, timeout=timeout, stream=True)
            response.raise_for_status()
            
            # Check content type
            content_type = response.headers.get('content-type', '')
            if 'image' not in content_type.lower() and 'octet-stream' not in content_type.lower():
                return None
            
            # Determine extension
            ext = '.jpg'
            if 'png' in content_type:
                ext = '.png'
            elif 'webp' in content_type:
                ext = '.webp'
            elif 'gif' in content_type:
                ext = '.gif'
            elif url.lower().endswith('.png'):
                ext = '.png'
            elif url.lower().endswith('.webp'):
                ext = '.webp'
            
            # Save file
            filepath = output_folder / f"{index:08d}{ext}"
            
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            return filepath
            
        except requests.Timeout:
            logger.debug(f"  Timeout downloading image {index}")
            return None
        except Exception as e:
            logger.debug(f"  Error downloading image {index}: {e}")
            return None
    
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
        min_size_mb: float = 1.0,
        download_timeout: int = 60  # Timeout for HTTP requests
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # API keys
        self.pexels_key = pexels_key or os.getenv("PEXELS_API_KEY")
        self.pixabay_key = pixabay_key or os.getenv("PIXABAY_API_KEY")
        self.unsplash_key = unsplash_key or os.getenv("UNSPLASH_API_KEY")
        
        # Size filter
        self.min_size = int(min_size_mb * 1024 * 1024)
        
        # Timeout for downloads
        self.download_timeout = download_timeout
        
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
# STOCK VIDEO DOWNLOAD (Pexels/Pixabay)
# =============================================================================

# Pexels Video API
PEXELS_VIDEOS_API = "https://api.pexels.com/videos/search"

# Pixabay Video API  
PIXABAY_VIDEOS_API = "https://pixabay.com/api/videos/"


@dataclass
class VideoResult:
    """Result from video search"""
    id: str
    source: str  # pexels, pixabay
    url: str  # Web page URL
    download_url: str
    width: int
    height: int
    duration: float  # seconds
    quality: str  # hd, sd, etc
    file_type: str  # mp4, etc
    file_path: str = ""  # Path after download


@dataclass
class EntityVideoResult:
    """Result of video search for an entity"""
    entity_name: str
    entity_type: str
    context: str
    query: str
    videos: List[str] = field(default_factory=list)  # Downloaded file paths
    segment_indices: List[int] = field(default_factory=list)


class StockVideoDownloader:
    """
    Download stock videos from Pexels and Pixabay.
    
    Features:
    - Multi-source search (Pexels, Pixabay)
    - Quality preference (HD preferred)
    - Duration filtering
    """
    
    def __init__(
        self,
        output_dir: str = "./downloaded_videos",
        pexels_key: str = None,
        pixabay_key: str = None,
        min_duration: float = 3.0,  # Minimum video duration in seconds
        max_duration: float = 30.0,  # Maximum video duration
        prefer_hd: bool = True,
        download_timeout: int = 60  # Timeout for HTTP requests
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # API keys
        self.pexels_key = pexels_key or os.getenv("PEXELS_API_KEY")
        self.pixabay_key = pixabay_key or os.getenv("PIXABAY_API_KEY")
        
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.prefer_hd = prefer_hd
        self.download_timeout = download_timeout
        
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Voiceover-Matcher/2.3 StockVideoDownloader"
        })
        
        # Rate limiting
        self._last_request_time = 0
        self._min_interval = 0.5
    
    def _rate_limit(self):
        """Enforce rate limiting"""
        now = time.time()
        elapsed = now - self._last_request_time
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_request_time = time.time()
    
    def search_pexels_videos(self, query: str, per_page: int = 10) -> List[VideoResult]:
        """Search Pexels for videos"""
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
            response = self.session.get(PEXELS_VIDEOS_API, headers=headers, params=params)
            response.raise_for_status()
            data = response.json()
            
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
            
            logger.info(f"Pexels videos '{query}': {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Pexels video search error: {e}")
            return []
    
    def search_pixabay_videos(self, query: str, per_page: int = 10) -> List[VideoResult]:
        """Search Pixabay for videos"""
        if not self.pixabay_key:
            return []
        
        self._rate_limit()
        
        params = {
            "key": self.pixabay_key,
            "q": query,
            "per_page": per_page,
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
    
    def search_all(self, query: str, per_source: int = 5) -> List[VideoResult]:
        """Search all video sources"""
        results = []
        
        # Search Pexels
        results.extend(self.search_pexels_videos(query, per_source))
        
        # Search Pixabay
        results.extend(self.search_pixabay_videos(query, per_source))
        
        return results
    
    def download_video(self, video: VideoResult) -> Optional[str]:
        """Download a single video"""
        if not video.download_url:
            return None
        
        self._rate_limit()
        
        # Create short filename for NLE compatibility
        # Format: {source_letter}{id}.mp4 = ~12 chars
        source_letter = video.source[0].lower()  # p=pexels, p=pixabay
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
            total_size = int(response.headers.get('content-length', 0))
            
            with open(filepath, 'wb') as f:
                downloaded = 0
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
            
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
        max_videos: int = 3
    ) -> List[str]:
        """
        Search and download videos for a query.
        
        Args:
            query: Search keywords
            max_videos: Maximum videos to download
        
        Returns:
            List of downloaded file paths
        """
        results = self.search_all(query, per_source=max_videos * 2)
        
        downloaded = []
        for video in results:
            if len(downloaded) >= max_videos:
                break
            
            path = self.download_video(video)
            if path:
                downloaded.append(path)
        
        return downloaded


def download_entity_videos(
    entities: List[Dict],
    output_dir: str,
    topic: str = "",
    videos_per_entity: int = 3,
    min_duration: float = 3.0,
    max_duration: float = 30.0,
    prefer_hd: bool = True,
    download_timeout: int = 60,
    pexels_key: str = None,
    pixabay_key: str = None
) -> Dict[str, EntityVideoResult]:
    """
    Download stock videos for entities extracted from voiceover.
    
    Args:
        entities: List of entity dicts with 'text', 'type', 'context' keys
        output_dir: Directory to save videos
        topic: Documentary topic for query building
        videos_per_entity: Number of videos to download per entity
        min_duration: Minimum video duration in seconds
        max_duration: Maximum video duration in seconds
        prefer_hd: Prefer HD quality videos
        download_timeout: Timeout for HTTP requests
    
    Returns:
        Dict mapping entity name to EntityVideoResult
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    results: Dict[str, EntityVideoResult] = {}
    
    # Initialize downloader with short folder name
    video_downloader = StockVideoDownloader(
        output_dir=str(output_path / "sv"),  # Short for stock_videos
        pexels_key=pexels_key,
        pixabay_key=pixabay_key,
        min_duration=min_duration,
        max_duration=max_duration,
        prefer_hd=prefer_hd,
        download_timeout=download_timeout
    )
    
    # Check if any API keys available
    has_keys = any([video_downloader.pexels_key, video_downloader.pixabay_key])
    
    if not has_keys:
        logger.warning("No video API keys available (PEXELS_API_KEY, PIXABAY_API_KEY)")
        return results
    
    # Process each entity
    for entity in entities:
        entity_name = entity.get('text', '')
        entity_type = entity.get('type', '')
        context = entity.get('context', '')
        
        if not entity_name:
            continue
        
        # Skip if already processed
        if entity_name in results:
            continue
        
        # Build search query
        query = build_entity_query(entity, topic)
        
        if not query:
            continue
        
        logger.info(f"Searching stock videos for: {entity_name} ({entity_type})")
        logger.info(f"  Query: '{query}'")
        
        # Download videos
        video_paths = video_downloader.search_and_download(
            query=query,
            max_videos=videos_per_entity
        )
        
        if video_paths:
            results[entity_name] = EntityVideoResult(
                entity_name=entity_name,
                entity_type=entity_type,
                context=context,
                query=query,
                videos=video_paths
            )
            logger.info(f"  ✓ Downloaded {len(video_paths)} videos for '{entity_name}'")
        else:
            logger.info(f"  ✗ No videos found for '{entity_name}'")
    
    return results

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
    use_bing: bool = False,  # Disabled by default - alive_progress conflicts
    use_stock_apis: bool = True,
    pexels_key: str = None,
    pixabay_key: str = None,
    download_timeout: int = 10,
    max_search_time: int = 300,
    max_results_to_check: int = 500,
    search_until_found: bool = True
) -> Dict[str, EntityImageResult]:
    """
    Download images for entities extracted from voiceover.
    
    Args:
        entities: List of entity dicts with 'text', 'type', 'context' keys
        output_dir: Directory to save images
        topic: Documentary topic for query building
        images_per_entity: Number of images to download per entity (2-5)
        min_size_mb: Minimum file size (default 1MB)
        use_google: Use Google image search (via imagedl library)
        use_bing: Use Bing image search (disabled by default - has conflicts)
        use_stock_apis: Use Pexels/Pixabay APIs as fallback
        download_timeout: Seconds per image download (default 10)
        max_search_time: Max seconds for entire search per entity (default 300)
        max_results_to_check: Max search results to check per entity (default 500)
        search_until_found: If True, keep trying query variations until images found
    
    Returns:
        Dict mapping entity name to EntityImageResult
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    results: Dict[str, EntityImageResult] = {}
    
    # Initialize clients with timeout settings
    google_client = None
    bing_client = None
    stock_downloader = None
    
    if use_google:
        google_client = GoogleBingImageClient(
            output_dir=str(output_path),
            source="GoogleImageClient",
            min_size_mb=min_size_mb,
            download_timeout=download_timeout,
            max_search_time=max_search_time,
            max_results_to_check=max_results_to_check,
            search_until_found=search_until_found
        )
    
    # Only initialize Bing if explicitly enabled (disabled by default)
    if use_bing:
        bing_client = GoogleBingImageClient(
            output_dir=str(output_path),
            source="BingImageClient",
            min_size_mb=min_size_mb,
            download_timeout=download_timeout,
            max_search_time=max_search_time,
            max_results_to_check=max_results_to_check,
            search_until_found=search_until_found
        )
    
    if use_stock_apis:
        stock_downloader = ImageDownloader(
            output_dir=str(output_path),
            pexels_key=pexels_key,
            pixabay_key=pixabay_key,
            min_size_mb=min_size_mb,
            download_timeout=download_timeout * 6  # Longer timeout for stock API downloads
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
            try:
                paths = google_client.search_and_download(
                    query=query,
                    max_images=images_per_entity,
                    entity_name=entity_name,
                    entity_type=entity_type
                )
                downloaded_paths.extend(paths)
            except Exception as e:
                error_msg = str(e)
                if "alive_progress" in error_msg.lower() or "nested" in error_msg.lower():
                    logger.warning(f"  Google search interrupted, reinitializing...")
                    google_client._reinitialize_client()
                    time.sleep(2)
                else:
                    logger.warning(f"  Google search error: {e}")
        
        # Try Bing if Google didn't find enough
        # Note: Skip Bing if Google found partial results (likely timed out) - alive_progress conflict
        # Only use Bing if Google found ZERO images (fresh start)
        use_bing = has_bing and len(downloaded_paths) == 0
        
        if use_bing and len(downloaded_paths) < images_per_entity:
            remaining = images_per_entity - len(downloaded_paths)
            logger.info(f"  Trying Bing for {remaining} more images...")
            
            # Wait for any alive_progress to clean up
            import time as time_module
            time_module.sleep(3)
            
            # Reinitialize Bing client
            bing_client._reinitialize_client()
            
            if bing_client.client:
                try:
                    paths = bing_client.search_and_download(
                        query=query,
                        max_images=remaining,
                        entity_name=entity_name,
                        entity_type=entity_type
                    )
                    downloaded_paths.extend(paths)
                except Exception as e:
                    logger.warning(f"  Bing search failed: {e}")
            else:
                logger.warning("  Bing client unavailable, skipping")
        elif has_bing and len(downloaded_paths) > 0 and len(downloaded_paths) < images_per_entity:
            logger.info(f"  Skipping Bing (Google partial - using stock APIs instead)")
        
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
        
        # Rate limiting - also allows alive_progress to clean up between entities
        time.sleep(2)
    
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