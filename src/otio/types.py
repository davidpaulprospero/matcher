"""
Type definitions and constants for OTIO generation.

Extracted from otio_builder.py to centralize timeline configuration.
"""

from typing import TypeAlias, Literal

# Track names for V1-V10 (video tracks)
TRACK_NAMES = [
    "Primary Video",
    "Alternative Video 1",
    "Alternative Video 2",
    "Secondary Diversity 1",
    "Secondary Diversity 2",
    "Secondary Diversity 3",
    "Embedding-Diversity Strategy",
    "B-roll Only",
    "Entity Images (Google)",
    "Stock Videos (Pexels/Pixabay)",
]

# Track strategies
TrackStrategy: TypeAlias = Literal[
    "primary",
    "alternative",
    "diversity",
    "embedding_diversity",
    "broll_only",
    "entity_images",
    "entity_videos"
]

# Default frame rate
DEFAULT_FRAME_RATE = 30.0

# Default timeline start timecode
DEFAULT_TIMELINE_START = "00:00:00:00"
