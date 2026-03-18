"""
Orchestration logic for entity image downloads.

Extracted from entity_images.py download_entity_images() function (lines 1429-1698)
(Jan 7, 2026).
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

from ..models import EntityImageResult
from ..utils import build_entity_query, check_local_entity_images
from .google_bing import GoogleBingImageClient
from .pexels import PexelsImageClient
from .pixabay import PixabayImageClient
from .unsplash import UnsplashImageClient

logger = logging.getLogger(__name__)


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
    search_until_found: bool = True,
    source_project: str = "",
    skip_local_cache: bool = False,
    config = None  # Config object for stock API clients
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
        source_project: Project name for cache attribution
        skip_local_cache: If True, ignore local cached images and re-download
        config: Config object (required for stock API clients)

    Returns:
        Dict mapping entity name to EntityImageResult
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results: Dict[str, EntityImageResult] = {}

    # Initialize clients with timeout settings
    google_client = None
    bing_client = None
    pexels_client = None
    pixabay_client = None
    unsplash_client = None

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

    # Initialize stock API clients
    if use_stock_apis and config:
        # Create a simple config placeholder if none provided
        if not config:
            class DummyConfig:
                pass
            config = DummyConfig()

        longer_timeout = download_timeout * 6  # Stock APIs need longer timeout

        pexels_client = PexelsImageClient(
            config=config,
            output_dir=str(output_path),
            api_key=pexels_key,
            min_size_mb=min_size_mb,
            download_timeout=longer_timeout
        )

        pixabay_client = PixabayImageClient(
            config=config,
            output_dir=str(output_path),
            api_key=pixabay_key,
            min_size_mb=min_size_mb,
            download_timeout=longer_timeout
        )

        unsplash_client = UnsplashImageClient(
            config=config,
            output_dir=str(output_path),
            api_key=None,  # Can add later
            min_size_mb=min_size_mb,
            download_timeout=longer_timeout
        )

    # Check which sources are available
    has_google = google_client and google_client.client
    has_bing = bing_client and bing_client.client
    has_stock = any([
        pexels_client and pexels_client.api_key,
        pixabay_client and pixabay_client.api_key,
        unsplash_client and unsplash_client.api_key
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

        # Check local images first (unless force refresh)
        if not skip_local_cache:
            existing_images = check_local_entity_images(output_dir, entity_name, entity_type)
            if existing_images and len(existing_images) >= images_per_entity:
                results[entity_name] = EntityImageResult(
                    entity_name=entity_name,
                    entity_type=entity_type,
                    context=context,
                    query="(local cache)",
                    images=existing_images[:images_per_entity]  # Use only needed count
                )
                total_downloaded += len(existing_images[:images_per_entity])
                logger.info(f"  ✓ Local cache: '{entity_name}' ({len(existing_images)} images)")
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
        use_bing_now = has_bing and len(downloaded_paths) == 0

        if use_bing_now and len(downloaded_paths) < images_per_entity:
            remaining = images_per_entity - len(downloaded_paths)
            logger.info(f"  Trying Bing for {remaining} more images...")

            # Wait for any alive_progress to clean up
            time.sleep(3)

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

            # Try Pexels
            if pexels_client and pexels_client.api_key and len(downloaded_paths) < images_per_entity:
                paths = pexels_client.search_and_download(
                    query=query,
                    max_images=remaining,
                    entity_name=entity_name,
                    entity_type=entity_type
                )
                downloaded_paths.extend(paths)

            # Try Pixabay if still need more
            if pixabay_client and pixabay_client.api_key and len(downloaded_paths) < images_per_entity:
                remaining = images_per_entity - len(downloaded_paths)
                paths = pixabay_client.search_and_download(
                    query=query,
                    max_images=remaining,
                    entity_name=entity_name,
                    entity_type=entity_type
                )
                downloaded_paths.extend(paths)

            # Try Unsplash if still need more
            if unsplash_client and unsplash_client.api_key and len(downloaded_paths) < images_per_entity:
                remaining = images_per_entity - len(downloaded_paths)
                paths = unsplash_client.search_and_download(
                    query=query,
                    max_images=remaining,
                    entity_name=entity_name,
                    entity_type=entity_type
                )
                downloaded_paths.extend(paths)

        # Store result - validate paths before storing
        if downloaded_paths:
            # Filter out any paths with brackets (image sequence notation)
            valid_paths = []
            for p in downloaded_paths:
                if '[' in str(p) or ']' in str(p):
                    logger.warning(f"  Skipping invalid path with brackets: {p}")
                elif not Path(p).exists():
                    logger.warning(f"  Skipping non-existent path: {p}")
                else:
                    valid_paths.append(p)

            if valid_paths:
                results[entity_name] = EntityImageResult(
                    entity_name=entity_name,
                    entity_type=entity_type,
                    context=context,
                    query=query,
                    images=valid_paths
                )
                total_downloaded += len(valid_paths)
                logger.info(f"  ✓ Downloaded {len(valid_paths)} images for '{entity_name}'")
        else:
            logger.warning(f"  ⚠ No images found for '{entity_name}'")

        # Rate limiting - also allows alive_progress to clean up between entities
        time.sleep(2)

    logger.info(f"Entity images: {total_downloaded} total for {len(results)} entities")
    return results
