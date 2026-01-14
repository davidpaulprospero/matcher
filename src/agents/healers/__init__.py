"""
Individual healers for specific error categories.
"""

from .otio import OTIOHealer
from .api import APIHealer
from .checkpoint import CheckpointHealer
from .download import DownloadHealer
from .disk import DiskHealer
from .path import PathHealer

__all__ = [
    'OTIOHealer',
    'APIHealer',
    'CheckpointHealer',
    'DownloadHealer',
    'DiskHealer',
    'PathHealer',
]

# Registry of all available healers (order matters - first match wins)
HEALER_REGISTRY = [
    CheckpointHealer,  # Try checkpoint recovery first
    APIHealer,         # API rate limits
    DownloadHealer,    # Download failures
    DiskHealer,        # Disk space issues
    PathHealer,        # Path length issues
    OTIOHealer,        # Timeline generation
]
