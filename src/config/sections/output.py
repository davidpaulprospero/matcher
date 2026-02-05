"""Output configuration: Timeline generation, track layout, deduplication.

Extracted from monolithic config.py during refactoring (Jan 7, 2026).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

__all__ = [
    'DeduplicationConfig',
    'VarietyConfig',
    'OutputConfig',
    'MultiStyleConfig',
]


@dataclass
class DeduplicationConfig:
    """Video deduplication settings"""
    enabled: bool = True
    hash_threshold: int = 10  # Hamming distance (0-64, lower = stricter)
    use_first_frame: bool = True
    auto_delete: bool = True
    generate_report: bool = True
    frame_timeout: int = 30  # Seconds for FFmpeg frame extraction

    def __post_init__(self):
        if not (0 <= self.hash_threshold <= 64):
            raise ValueError(
                f"DeduplicationConfig.hash_threshold must be in range 0-64 "
                f"(hamming distance for 64-bit hash), got {self.hash_threshold}"
            )
        if self.frame_timeout <= 0:
            raise ValueError(
                f"DeduplicationConfig.frame_timeout must be positive "
                f"(seconds for FFmpeg extraction), got {self.frame_timeout}"
            )


@dataclass
class VarietyConfig:
    """Settings to ensure variety across tracks (for strategy matching)"""
    require_different_source: bool = True  # Each track must use different source video
    exclude_same_clip: bool = True  # Never use exact same clip on multiple tracks
    min_time_distance: float = 10.0  # Clips must be N seconds apart (same source)
    min_embedding_distance: float = 0.3  # V4+ must have distance > this from V1-V3

    # Timeline variety enforcement - prevents same source video from dominating
    # Within timeline_variety_window seconds, same source can only appear max_source_repeats times
    enforce_timeline_variety: bool = True  # Enable timeline-based variety enforcement
    timeline_variety_window: float = 600.0  # 10 minutes - no same source within this window
    max_source_repeats_in_window: int = 1  # Max times same source can appear in window


@dataclass
class OutputConfig:
    """Output generation settings

    Chain-of-thought: Multiple tracks give editors options
    Reasoning: V1-V3 alternatives, V4-V8 strategies
    Decision: Enable all by default, let editors disable unused
    """
    output_dir: str = "output"

    # File formats
    generate_otio: bool = True
    split_otio: bool = True  # Split OTIO into multiple files by clip batches
    otio_clips_per_file: int = 10  # Max clips per OTIO file
    generate_edl: bool = True
    generate_xml: bool = True  # DaVinci Resolve XML (fallback if OTIO fails)
    xml_parts: int = 2  # Split XML into multiple files (helps with large projects)
    generate_report: bool = True
    quality_report_enabled: bool = True  # Generate quality_report.json with match metrics

    # Timeline settings
    frame_rate: float = 30.0
    timeline_start_tc: str = "01:00:00:00"  # Standard broadcast start

    # Voiceover alignment offset (seconds)
    # Use this to fix alignment when SRT timestamps don't match the actual audio
    # Positive = shift clips later (audio is ahead of SRT)
    # Negative = shift clips earlier (audio is behind SRT)
    voiceover_offset: float = 0.0

    # Gap threshold (seconds) - collapse gaps smaller than this value
    # Whisper often inserts small gaps (~0.5-0.8s) between segments
    # Setting this to 1.0 will place clips back-to-back for gaps under 1 second
    # Set to 0.0 to preserve all gaps from SRT (default behavior)
    min_gap_threshold: float = 0.0

    # Time scale factor for SRT timestamps
    # Use when actual audio duration differs from SRT end time
    # Values: 0.0 = auto-calculate, 1.0 = no scaling (default), other = manual scale
    # Example: 1596.9 / 1499.6 = 1.065 stretches timeline by 6.5%
    time_scale_factor: float = 1.0

    # Gap distribution mode - how to handle gaps between voiceover segments
    # Controls how silence/pauses are placed in the timeline
    # Options:
    #   - "scale": (default) Scale SRT gaps by time_scale_factor
    #   - "proportional": Recalculate gaps to distribute content evenly across audio
    #   - "none": No gaps between clips, only leading/trailing gaps
    #   - "extend": Extend previous clip to fill gaps (max 2x original clip duration)
    # Use "proportional" when Whisper's gap timing is inaccurate but segment durations are good
    # Use "extend" to minimize gaps by stretching footage, creating a more continuous flow
    gap_mode: str = "scale"

    # Track structure
    # V1: Primary, V2-V3: Alternatives
    # V4-V6: Secondary (different video files from V1-V3)
    # V7: Embedding-Diversity strategy
    num_alternatives: int = 2  # V2-V3
    include_alternatives: bool = True
    include_strategy_tracks: bool = True
    strategy_tracks: List[str] = field(default_factory=lambda: [
        "embedding_diversity",  # V7 - maximally different clips from V1-V3
        "broll_only"            # V8 - B-roll only matching (silent footage)
    ])

    # Variety enforcement for strategy tracks
    variety: VarietyConfig = field(default_factory=VarietyConfig)

    # Markers
    include_speed_markers: bool = True
    max_recommended_speed: float = 1.5
    markers_for_low_confidence: bool = True
    markers_for_gaps: bool = True
    markers_for_speed: bool = True

    def __post_init__(self):
        """Convert variety dict to VarietyConfig if needed"""
        if isinstance(self.variety, dict):
            self.variety = VarietyConfig(**self.variety)


@dataclass
class MultiStyleConfig:
    """Multi-style OTIO generation"""
    enabled: bool = False
    styles: List[str] = field(default_factory=lambda: ["default", "strict"])
