"""
Media sources package for entity images and videos.

Provides modular clients for downloading images and videos from multiple sources:
- Google Images, Bing Images (via pyimagedl)
- Pexels Images & Videos
- Pixabay Images & Videos
- Unsplash Images

Created Jan 7, 2026 by refactoring entity_images.py (1,818 lines) into 14 focused modules.
"""

# Data models
from .models import (
    ImageResult,
    EntityImageResult,
    VideoResult,
    EntityVideoResult,
)

# Base class
from .base import BaseMediaClient

# Utilities
from .utils import (
    build_entity_query,
    check_local_entity_images,
    map_entities_to_segments,
    restore_entity_images_from_disk,
)

# Image sources
from .images import (
    GoogleBingImageClient,
    PexelsImageClient,
    PixabayImageClient,
    UnsplashImageClient,
    download_entity_images,
)

# Video sources
from .videos import (
    PexelsVideoClient,
    PixabayVideoClient,
    download_entity_videos,
)

__all__ = [
    # Models
    'ImageResult',
    'EntityImageResult',
    'VideoResult',
    'EntityVideoResult',
    # Base
    'BaseMediaClient',
    # Utils
    'build_entity_query',
    'check_local_entity_images',
    'map_entities_to_segments',
    'restore_entity_images_from_disk',
    # Image sources
    'GoogleBingImageClient',
    'PexelsImageClient',
    'PixabayImageClient',
    'UnsplashImageClient',
    'download_entity_images',
    # Video sources
    'PexelsVideoClient',
    'PixabayVideoClient',
    'download_entity_videos',
]
