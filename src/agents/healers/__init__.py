"""
Individual healers for specific error categories.
"""

from .otio import OTIOHealer
from .api import APIHealer
from .checkpoint import CheckpointHealer
from .download import DownloadHealer
from .disk import DiskHealer
from .path import PathHealer
from .llm_healer import LLMHealer
from .caption import CaptionHealer, NoCaptionsStatus, SubtitleInfo

__all__ = [
    'OTIOHealer',
    'APIHealer',
    'CheckpointHealer',
    'DownloadHealer',
    'DiskHealer',
    'PathHealer',
    'LLMHealer',
    'CaptionHealer',
    'NoCaptionsStatus',
    'SubtitleInfo',
]

# Registry of all available healers (order matters - first match wins)
# Note: LLMHealer is NOT in the registry - it's invoked by the orchestrator
# only when standard healers fail and watcher recommends escalation
HEALER_REGISTRY = [
    CheckpointHealer,  # Try checkpoint recovery first
    APIHealer,         # API rate limits
    CaptionHealer,     # Caption format discovery and fallback
    DownloadHealer,    # Download failures
    DiskHealer,        # Disk space issues
    PathHealer,        # Path length issues
    OTIOHealer,        # Timeline generation
]
