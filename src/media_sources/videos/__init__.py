"""
Video source clients and orchestration.

Provides clients for Pexels and Pixabay stock video search,
plus orchestration logic for entity video downloads.
"""

from .pexels import PexelsVideoClient
from .pixabay import PixabayVideoClient
from .orchestrator import download_entity_videos

__all__ = [
    'PexelsVideoClient',
    'PixabayVideoClient',
    'download_entity_videos',
]
