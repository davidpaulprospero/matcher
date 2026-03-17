"""
Orchestration logic for entity video downloads.

Extracted from entity_images.py download_entity_videos() function (lines 1238-1329)
(Jan 7, 2026).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from ..models import EntityVideoResult
from ..utils import build_entity_query
from .pexels import PexelsVideoClient
from .pixabay import PixabayVideoClient

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


def _source_from_path(path: str) -> str:
    """Infer stock source from downloaded filename/path."""
    p = str(path).lower()
    # Current clients use short filenames that begin with 'p' for both providers,
    # so we also inspect the path for provider folder hints when available.
    if "pixabay" in p:
        return "pixabay"
    if "pexels" in p:
        return "pexels"
    return "stock"


def download_entity_videos(
    entities: List[Dict],
    output_dir: str,
    topic: str = "",
    videos_per_entity: int = 2,
    min_duration: float = 3.0,
    max_duration: float = 30.0,
    prefer_hd: bool = True,
    pexels_key: str = None,
    pixabay_key: str = None,
    download_timeout: int = 60,
    config = None  # Config object for video clients
) -> Dict[str, EntityVideoResult]:
    """
    Download stock videos for entities extracted from voiceover.

    Args:
        entities: List of entity dicts with 'text', 'type', 'context' keys
        output_dir: Directory to save videos
        topic: Documentary topic for query building
        videos_per_entity: Number of videos to download per entity (1-3)
        min_duration: Minimum video duration in seconds (default 3.0)
        max_duration: Maximum video duration in seconds (default 30.0)
        prefer_hd: Prefer higher resolution videos
        pexels_key: Pexels API key
        pixabay_key: Pixabay API key
        download_timeout: Seconds per video download
        config: Config object (required for video clients)

    Returns:
        Dict mapping entity name to EntityVideoResult
    """
    output_path = Path(output_dir)

    results: Dict[str, EntityVideoResult] = {}

    # Create a simple config placeholder if none provided
    if not config:
        class DummyConfig:
            pass
        config = DummyConfig()

    # Initialize downloader with short folder name
    pexels_client = PexelsVideoClient(
        config=config,
        output_dir=str(output_path / "sv"),  # Short for stock_videos
        api_key=pexels_key,
        min_duration=min_duration,
        max_duration=max_duration,
        prefer_hd=prefer_hd,
        download_timeout=download_timeout
    )

    pixabay_client = PixabayVideoClient(
        config=config,
        output_dir=str(output_path / "sv"),
        api_key=pixabay_key,
        min_duration=min_duration,
        max_duration=max_duration,
        prefer_hd=prefer_hd,
        download_timeout=download_timeout
    )

    # Check if any API keys available
    has_keys = any([pexels_client.api_key, pixabay_client.api_key])

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

        # Download videos from available sources
        video_paths = []

        # Try Pexels
        if pexels_client.api_key:
            paths = pexels_client.search_and_download(
                query=query,
                max_videos=videos_per_entity
            )
            video_paths.extend(paths)

        # Try Pixabay if needed
        if pixabay_client.api_key and len(video_paths) < videos_per_entity:
            remaining = videos_per_entity - len(video_paths)
            paths = pixabay_client.search_and_download(
                query=query,
                max_videos=remaining
            )
            video_paths.extend(paths)

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


def download_stock_videos(
    segment_queries: Dict[int, str],
    output_dir: str,
    clips_per_segment: int = 1,
    min_duration: float = 3.0,
    max_duration: float = 30.0,
    prefer_hd: bool = True,
    pexels_key: str = None,
    pixabay_key: str = None,
    download_timeout: int = 60,
    config=None,
) -> Dict[int, Dict[str, Any]]:
    """
    Download generic (non-entity) stock videos per selected segment query.

    Args:
        segment_queries: Mapping of segment_index -> search query
        output_dir: Directory to save videos
        clips_per_segment: Maximum clips to download per selected segment
        min_duration: Minimum clip duration in seconds
        max_duration: Maximum clip duration in seconds
        prefer_hd: Prefer higher resolution clips when available
        pexels_key: Pexels API key
        pixabay_key: Pixabay API key
        download_timeout: Seconds per video download
        config: Config object for underlying clients

    Returns:
        Dict mapping segment_index -> {
            "segment_index": int,
            "query": str,
            "videos": [paths...],
            "sources": [source labels...],
        }
    """
    output_path = Path(output_dir)
    results: Dict[int, Dict[str, Any]] = {}

    if not config:
        class DummyConfig:
            pass
        config = DummyConfig()

    pexels_client = PexelsVideoClient(
        config=config,
        output_dir=str(output_path / "sv"),
        api_key=pexels_key,
        min_duration=min_duration,
        max_duration=max_duration,
        prefer_hd=prefer_hd,
        download_timeout=download_timeout,
    )

    pixabay_client = PixabayVideoClient(
        config=config,
        output_dir=str(output_path / "sv"),
        api_key=pixabay_key,
        min_duration=min_duration,
        max_duration=max_duration,
        prefer_hd=prefer_hd,
        download_timeout=download_timeout,
    )

    if not any([pexels_client.api_key, pixabay_client.api_key]):
        logger.warning("No video API keys available for stock footage (PEXELS_API_KEY, PIXABAY_API_KEY)")
        return results

    for segment_index, query in sorted(segment_queries.items(), key=lambda x: x[0]):
        if not query:
            continue

        logger.info(f"Searching generic stock videos for segment {segment_index}: '{query}'")
        video_paths: List[str] = []
        sources: List[str] = []

        if pexels_client.api_key:
            paths = pexels_client.search_and_download(query=query, max_videos=clips_per_segment)
            for path in paths:
                if path not in video_paths and len(video_paths) < clips_per_segment:
                    video_paths.append(path)
                    sources.append("pexels")

        if pixabay_client.api_key and len(video_paths) < clips_per_segment:
            remaining = clips_per_segment - len(video_paths)
            paths = pixabay_client.search_and_download(query=query, max_videos=remaining)
            for path in paths:
                if path not in video_paths and len(video_paths) < clips_per_segment:
                    video_paths.append(path)
                    # Use provider hint, fallback to inferred source.
                    sources.append("pixabay" if "pixabay" in str(path).lower() else _source_from_path(path))

        if video_paths:
            results[segment_index] = {
                "segment_index": segment_index,
                "query": query,
                "videos": video_paths,
                "sources": sources or [_source_from_path(p) for p in video_paths],
            }
            logger.info(f"  Downloaded {len(video_paths)} stock clips for segment {segment_index}")
        else:
            logger.info(f"  No stock clips found for segment {segment_index}")

    return results
