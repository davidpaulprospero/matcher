"""
OTIO timeline generation and export package.

This package provides modular components for building OTIO timelines
from video matches, with support for multiple track strategies and
various export formats.

Main modules:
- types: Constants and type definitions
- utils: Utility functions (path handling, type conversion, etc.)
- entities: Entity track building (V9 images, V10 videos)
- tracks: Track building strategies (V1-V10)
- timeline: Timeline orchestration
- export: Format exporters (OTIO, EDL)
- reporting: Statistics and segment mapping

Public API (100% backward compatible with otio_builder.py):
- create_timeline()
- save_timeline()
- save_timeline_split()
- save_timeline_as_edl()
- generate_segment_map()
"""

# Version info
__version__ = "2.0.0"
__author__ = "Claude Code Refactoring - Jan 2026"

# Import public API functions for backward compatibility
from .timeline import create_timeline
from .export import (
    save_timeline,
    save_timeline_split,
    save_timeline_split_with_config,
    save_timeline_as_edl,
)
from .reporting import generate_segment_map
from .xml_export import generate_resolve_xml_with_bins
from .utils import count_timeline_items

__all__ = [
    'create_timeline',
    'save_timeline',
    'save_timeline_split',
    'save_timeline_split_with_config',
    'save_timeline_as_edl',
    'generate_segment_map',
    'generate_resolve_xml_with_bins',
    'count_timeline_items',
]
