"""
Google and Bing image search client using pyimagedl library.

Extracted from entity_images.py lines 73-542 (Jan 7, 2026).
Kept largely intact due to complex pyimagedl integration and timeout handling.
"""

from __future__ import annotations

import json
import logging
import random
import string
from pathlib import Path
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


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

    def _search_with_timeout(self, keyword: str, search_limit: int, timeout: int = 60) -> List[dict]:
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
                                # Validate path doesn't contain brackets (image sequence notation)
                                path_str = str(filepath)
                                if '[' in path_str or ']' in path_str:
                                    logger.warning(f"  Skipping path with brackets: {path_str}")
                                    continue
                                valid_paths.append(path_str)
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
                                except (OSError, IOError) as e:
                                    # File deletion may fail due to permissions or locks
                                    logger.debug(f"Could not delete small file {filepath}: {e}")
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

            # Save file with completely non-sequential name to prevent image sequence detection
            # DaVinci Resolve detects sequential numbered patterns even with suffixes
            # Use UUID-based name to completely break any pattern: img_a3f2b7x9k2m4.jpg
            random_name = ''.join(random.choices(string.ascii_lowercase + string.digits, k=12))
            filepath = output_folder / f"img_{random_name}{ext}"

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
