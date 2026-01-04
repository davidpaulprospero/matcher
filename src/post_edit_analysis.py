"""
Post-Edit Analysis Tool

Analyzes final OTIO exported from DaVinci Resolve to understand which clips
were selected from the original match recommendations.

Two analysis modes:
1. FILENAME-BASED (default): Tracks which recommended files appear in final edit
   - Works with restructured timelines
   - Answers: "How much of the AI's work did I keep?"

2. POSITION-BASED: Matches clips by timeline frame position
   - For strict 1:1 QA testing
   - Requires timeline structure to be preserved

Usage:
    # Filename-based (default)
    python -m src.post_edit_analysis --edited export.otio --original timeline_FULL.otio

    # Position-based
    python -m src.post_edit_analysis --edited export.otio --segments timeline_segments.json --mode position
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _format_timecode(seconds: float) -> str:
    """Format seconds as MM:SS or HH:MM:SS."""
    if seconds < 0:
        return "-" + _format_timecode(abs(seconds))

    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    if hours > 0:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def _render_coverage_bar(coverage_pct: float, width: int = 10) -> str:
    """Render coverage as ASCII bar: [████░░░░░░]"""
    filled = int(round(coverage_pct / 100 * width))
    empty = width - filled
    return f"[{'#' * filled}{'.' * empty}]"


def _is_matcher_track(track_name: str) -> bool:
    """Check if track name matches matcher-generated patterns."""
    track_upper = track_name.upper()
    # V1-V10 with word boundary (V1 but not V10, V10 but not V100)
    if re.search(r'\bV([1-9]|10)\b', track_upper):
        return True
    # Keywords that indicate matcher tracks
    keywords = ['PRIMARY', 'ALTERNATIVE', 'SECONDARY', 'EMBEDDING',
                'DIVERSITY', 'STRATEGY', 'ENTITY', 'STOCK']
    return any(kw in track_upper for kw in keywords)


def _get_track_category(track_name: str) -> str:
    """
    Get category for a matcher track.

    Returns:
        'v1', 'v2_v3', 'v4_v6', 'v7_plus', or None if not a matcher track
    """
    track_upper = track_name.upper()

    # Check V-number patterns FIRST (more specific than keywords)
    # V1 only
    if re.search(r'\bV1\b', track_upper):
        return 'v1'

    # V2-V3
    if re.search(r'\bV[23]\b', track_upper):
        return 'v2_v3'

    # V4-V6
    if re.search(r'\bV[456]\b', track_upper):
        return 'v4_v6'

    # V7-V10
    if re.search(r'\bV([789]|10)\b', track_upper):
        return 'v7_plus'

    # Then check keywords (less specific, could appear in any track name)
    # Primary keyword (without V-number) = v1
    if 'PRIMARY' in track_upper:
        return 'v1'

    # Alternative keyword = v2_v3
    if 'ALTERNATIVE' in track_upper:
        return 'v2_v3'

    # Secondary keyword = v4_v6
    if 'SECONDARY' in track_upper:
        return 'v4_v6'

    # Strategy/experimental keywords = v7_plus
    if any(kw in track_upper for kw in ['EMBEDDING', 'DIVERSITY', 'STRATEGY', 'ENTITY', 'STOCK']):
        return 'v7_plus'

    return None  # Unknown/user track


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class FilenameAnalysisResult:
    """Results from filename-based analysis."""
    mode: str = "filename"
    analyzed_at: str = ""
    original_file: str = ""
    edited_file: str = ""

    # Counts (unique files)
    total_segments: int = 0  # Total V1 clip instances (voiceover segments)
    v1_kept: int = 0         # Unique V1 files used in edit
    v1_kept_pct: float = 0.0
    v2_v3_used: int = 0
    v4_v6_used: int = 0
    v7_plus_used: int = 0
    external_added: int = 0
    segments_dropped: int = 0  # Unique V1 files not used

    # Segment-level coverage (for coverage map)
    segments_covered: int = 0   # V1 segments whose file was used
    segments_not_covered: int = 0

    # Track breakdown
    track_breakdown: Dict[str, int] = field(default_factory=dict)
    user_added_tracks: Dict[str, int] = field(default_factory=dict)  # Tracks not in original

    # Detailed lists
    v1_clips_used: List[str] = field(default_factory=list)
    alternative_clips_used: List[str] = field(default_factory=list)
    secondary_clips_used: List[str] = field(default_factory=list)
    strategy_clips_used: List[str] = field(default_factory=list)
    external_clips: List[str] = field(default_factory=list)
    dropped_v1_clips: List[str] = field(default_factory=list)
    user_added_clips: List[str] = field(default_factory=list)  # Clips from user-added tracks

    # Duration comparison (optional)
    original_duration_sec: float = 0.0
    edited_duration_sec: float = 0.0
    coverage_ratio: float = 0.0
    duration_enabled: bool = False

    # Confidence correlation (optional)
    kept_avg_confidence: float = 0.0
    dropped_avg_confidence: float = 0.0
    confidence_correlation: str = ""  # "positive", "negative", "none"
    confidence_enabled: bool = False
    kept_confidences: List[float] = field(default_factory=list)
    dropped_confidences: List[float] = field(default_factory=list)

    # Coverage map (optional)
    coverage_map: List[Dict[str, Any]] = field(default_factory=list)
    coverage_map_enabled: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        result = {
            "mode": self.mode,
            "analyzed_at": self.analyzed_at,
            "original_file": self.original_file,
            "edited_file": self.edited_file,
            "summary": {
                "total_segments": self.total_segments,
                "v1_kept": self.v1_kept,
                "v1_kept_pct": round(self.v1_kept_pct, 1),
                "v2_v3_used": self.v2_v3_used,
                "v4_v6_used": self.v4_v6_used,
                "v7_plus_used": self.v7_plus_used,
                "external_added": self.external_added,
                "segments_dropped": self.segments_dropped,
                "segments_covered": self.segments_covered,
                "segments_not_covered": self.segments_not_covered
            },
            "track_breakdown": self.track_breakdown,
            "user_added_tracks": self.user_added_tracks,
            "external_clips": self.external_clips[:50],  # Limit for readability
            "user_added_clips": self.user_added_clips[:50],
            "dropped_v1_clips": self.dropped_v1_clips[:50]
        }

        # Add duration comparison if enabled
        if self.duration_enabled:
            result["duration"] = {
                "original_duration_sec": round(self.original_duration_sec, 2),
                "edited_duration_sec": round(self.edited_duration_sec, 2),
                "original_duration_formatted": _format_timecode(self.original_duration_sec),
                "edited_duration_formatted": _format_timecode(self.edited_duration_sec),
                "coverage_ratio": round(self.coverage_ratio, 3),
                "time_cut_sec": round(self.original_duration_sec - self.edited_duration_sec, 2)
            }

        # Add confidence correlation if enabled
        if self.confidence_enabled:
            result["confidence"] = {
                "kept_avg_confidence": round(self.kept_avg_confidence, 3),
                "dropped_avg_confidence": round(self.dropped_avg_confidence, 3),
                "correlation": self.confidence_correlation,
                "kept_count": len(self.kept_confidences),
                "dropped_count": len(self.dropped_confidences)
            }

        # Add coverage map if enabled
        if self.coverage_map_enabled:
            # Exclude covered_segments list for JSON brevity
            result["coverage_map"] = [
                {
                    "start_segment": b["start_segment"],
                    "end_segment": b["end_segment"],
                    "total": b["total"],
                    "covered": b["covered"],
                    "coverage_pct": b["coverage_pct"]
                }
                for b in self.coverage_map
            ]

        return result

    def summary(self) -> str:
        """Generate human-readable summary."""
        total_used = self.v1_kept + self.v2_v3_used + self.v4_v6_used + self.v7_plus_used + self.external_added

        # Calculate segment coverage percentage
        seg_covered_pct = (self.segments_covered / self.total_segments * 100) if self.total_segments > 0 else 0

        lines = [
            "=" * 65,
            "                    POST-EDIT ANALYSIS REPORT",
            "=" * 65,
            "",
            f"Original: {Path(self.original_file).name}",
            f"Edited:   {Path(self.edited_file).name}",
            f"Analyzed: {self.analyzed_at}",
            "",
            f"VOICEOVER SEGMENTS: {self.total_segments} total",
            f"  Covered by V1 files: {self.segments_covered:>4} ({seg_covered_pct:>5.1f}%)",
            f"  Not covered:         {self.segments_not_covered:>4} ({100-seg_covered_pct:>5.1f}%)",
            "",
            f"UNIQUE FILES USED (from recommendations):",
            f"  V1 CLIPS USED:           {self.v1_kept:>4}  [OK] System picks kept",
            f"  V2-V3 ALTERNATIVES:      {self.v2_v3_used:>4}  [^] Editor preferred alt",
            f"  V4-V6 SECONDARY:         {self.v4_v6_used:>4}  [^] Editor preferred secondary",
            f"  V7+ STRATEGY:            {self.v7_plus_used:>4}  [^] Diversity/experimental",
            f"  EXTERNAL CLIPS:          {self.external_added:>4}  [*] Editor's own footage",
            f"  V1 CLIPS DROPPED:        {self.segments_dropped:>4}  [X] Not used in final",
        ]

        # Duration comparison section
        if self.duration_enabled:
            time_cut = self.original_duration_sec - self.edited_duration_sec
            lines.extend([
                "",
                "-" * 65,
                "DURATION COMPARISON",
                "-" * 65,
                f"  Original voiceover:  {_format_timecode(self.original_duration_sec):>8} ({self.total_segments} segments)",
                f"  Final edit:          {_format_timecode(self.edited_duration_sec):>8} ({total_used} clips used)",
                f"  Coverage ratio:      {self.coverage_ratio * 100:>7.1f}%",
                f"  Time cut:            {_format_timecode(time_cut):>8}",
            ])

        # Confidence correlation section
        if self.confidence_enabled:
            lines.extend([
                "",
                "-" * 65,
                "CONFIDENCE ANALYSIS",
                "-" * 65,
                f"  Avg confidence (kept clips):    {self.kept_avg_confidence:.3f}",
                f"  Avg confidence (dropped clips): {self.dropped_avg_confidence:.3f}",
            ])
            if self.confidence_correlation == "positive":
                lines.append("  Correlation: Positive (higher confidence clips more likely kept)")
            elif self.confidence_correlation == "negative":
                lines.append("  Correlation: Negative (lower confidence clips more likely kept)")
            elif self.confidence_correlation == "weak_positive":
                lines.append("  Correlation: Weak positive (slightly higher conf clips more likely kept)")
            elif self.confidence_correlation == "weak_negative":
                lines.append("  Correlation: Weak negative (slightly lower conf clips more likely kept)")
            else:
                lines.append("  Correlation: None (confidence did not predict editor choices)")

        # Coverage map section
        if self.coverage_map_enabled and self.coverage_map:
            lines.extend([
                "",
                "-" * 65,
                "SEGMENT COVERAGE MAP",
                "-" * 65,
            ])
            for bucket in self.coverage_map:
                bar = _render_coverage_bar(bucket['coverage_pct'])
                lines.append(
                    f"  {bucket['start_segment']}-{bucket['end_segment']}:  "
                    f"{bar} {bucket['coverage_pct']:5.0f}%   "
                    f"{bucket['covered']}/{bucket['total']} segments"
                )
            lines.append("")
            lines.append("  Legend: # = covered, . = dropped/cut")

        lines.extend([
            "",
            "-" * 65,
            "TRACK PREFERENCE BREAKDOWN",
            "-" * 65,
        ])

        for track, count in sorted(self.track_breakdown.items()):
            lines.append(f"  {track}: {count} clips")

        # Show user-added tracks (graphics, lower thirds, etc.)
        if self.user_added_tracks:
            lines.extend([
                "",
                "-" * 65,
                "USER-ADDED TRACKS (not from matcher)",
                "-" * 65,
            ])
            for track, count in sorted(self.user_added_tracks.items()):
                lines.append(f"  {track}: {count} clips")

        if self.external_clips:
            lines.append("")
            lines.append("-" * 65)
            lines.append("EXTERNAL CLIPS ADDED")
            lines.append("-" * 65)
            for clip in self.external_clips[:10]:
                lines.append(f"  - {clip}")
            if len(self.external_clips) > 10:
                lines.append(f"  ... and {len(self.external_clips) - 10} more")

        if self.dropped_v1_clips:
            lines.append("")
            lines.append("-" * 65)
            lines.append(f"DROPPED V1 CLIPS ({len(self.dropped_v1_clips)} total)")
            lines.append("-" * 65)
            for clip in self.dropped_v1_clips[:10]:
                lines.append(f"  - {clip}")
            if len(self.dropped_v1_clips) > 10:
                lines.append(f"  ... and {len(self.dropped_v1_clips) - 10} more")

        lines.append("")
        lines.append("=" * 65)

        return "\n".join(lines)

    def _pct(self, value: int) -> float:
        """Calculate percentage of total segments."""
        return (value / self.total_segments * 100) if self.total_segments > 0 else 0.0


@dataclass
class PositionAnalysisResult:
    """Results from position-based analysis."""
    mode: str = "position"
    analyzed_at: str = ""
    edited_file: str = ""
    segment_map_file: str = ""
    frame_rate: float = 30.0

    # Summary stats
    total_segments: int = 0
    v1_kept: int = 0
    v1_kept_pct: float = 0.0
    replaced_with_alt: int = 0
    replaced_with_secondary: int = 0
    replaced_with_external: int = 0
    all_disabled: int = 0
    missing: int = 0

    # Track usage
    track_usage: Dict[str, int] = field(default_factory=dict)

    # Detailed
    replacements: List[Dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "mode": self.mode,
            "analyzed_at": self.analyzed_at,
            "edited_file": self.edited_file,
            "segment_map_file": self.segment_map_file,
            "summary": {
                "total_segments": self.total_segments,
                "v1_kept": self.v1_kept,
                "v1_kept_pct": round(self.v1_kept_pct, 1),
                "replaced": self.total_segments - self.v1_kept - self.all_disabled - self.missing,
                "all_disabled": self.all_disabled,
                "missing": self.missing
            },
            "track_usage": self.track_usage,
            "replacements": self.replacements[:50]
        }

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            "=" * 50,
            "POST-EDIT ANALYSIS (POSITION-BASED)",
            "=" * 50,
            f"Edited file: {Path(self.edited_file).name}",
            f"Segment map: {Path(self.segment_map_file).name}",
            f"Analyzed at: {self.analyzed_at}",
            "",
            f"Total segments:     {self.total_segments:>4}",
            f"V1 clips kept:      {self.v1_kept:>4} ({self.v1_kept_pct:.1f}%)",
            f"Replaced with alt:  {self.replaced_with_alt:>4}",
            f"Replaced with sec:  {self.replaced_with_secondary:>4}",
            f"External clips:     {self.replaced_with_external:>4}",
            f"All disabled:       {self.all_disabled:>4}",
            f"Missing:            {self.missing:>4}",
            "",
            "Track Usage:",
        ]

        for track, count in sorted(self.track_usage.items()):
            lines.append(f"  {track}: {count}")

        if self.replacements:
            lines.append("")
            lines.append(f"Replacements ({len(self.replacements)}):")
            for r in self.replacements[:10]:
                lines.append(f"  {r['segment']}: {r['original'][:25]} → {r['replaced_with'][:25]} ({r['new_track']})")
            if len(self.replacements) > 10:
                lines.append(f"  ... and {len(self.replacements) - 10} more")

        lines.append("=" * 50)
        return "\n".join(lines)


# =============================================================================
# FILENAME-BASED ANALYZER
# =============================================================================

class FilenameAnalyzer:
    """
    Analyzes edited timeline by tracking which recommended files appear in final edit.

    This approach works regardless of timeline restructuring because it only
    cares about which source files were used, not their positions.
    """

    def __init__(self, original_otio_path: str):
        """
        Initialize with original OTIO file.

        Args:
            original_otio_path: Path to the original timeline_FULL.otio
        """
        self.original_path = original_otio_path
        self.recommendations: Dict[str, Set[str]] = {
            'v1': set(),
            'v2_v3': set(),
            'v4_v6': set(),
            'v7_plus': set()
        }
        self.track_to_category: Dict[str, str] = {}

        # For confidence correlation
        self.v1_confidences: Dict[str, float] = {}  # filename -> confidence

        # For duration comparison
        self.original_duration_sec: float = 0.0
        self.frame_rate: float = 30.0

        # For coverage map - track segment order and their V1 filenames
        self.segment_order: List[str] = []  # ['S000', 'S001', 'S002', ...]
        self.segment_to_filename: Dict[str, str] = {}  # 'S000' -> 'abc.mp4'

        self._load_recommendations()

    def _load_recommendations(self) -> None:
        """Load all recommended clips from original OTIO, organized by track."""
        try:
            import opentimelineio as otio
        except ImportError:
            raise ImportError("opentimelineio is required")

        timeline = otio.adapters.read_from_file(self.original_path)

        # Get frame rate for duration calculation
        if timeline.global_start_time:
            self.frame_rate = timeline.global_start_time.rate
        else:
            self.frame_rate = 30.0

        v1_total_frames = 0
        v1_segment_idx = 0  # Counter for V1 segments (across all V1 tracks)
        found_v1_track = False  # Only process the FIRST V1 track for segment mapping

        for track in timeline.tracks:
            # Check if video track (handle both enum and string)
            track_kind = str(track.kind) if hasattr(track.kind, 'name') else track.kind
            if track_kind not in ('Video', 'TrackKind.Video'):
                continue

            track_name = track.name or ""

            # Categorize track using helper function (handles V1/V10 correctly)
            category = _get_track_category(track_name)
            if category is None:
                # Skip unknown/user tracks
                continue

            self.track_to_category[track_name] = category

            # Only map segments from the FIRST V1 track we encounter
            is_primary_v1 = (category == 'v1' and not found_v1_track)
            if is_primary_v1:
                found_v1_track = True

            for item in track:
                if not isinstance(item, otio.schema.Clip):
                    continue

                filename = self._extract_filename(item)
                if filename:
                    self.recommendations[category].add(filename)

                    # For primary V1 track, also track confidence, duration, and segment mapping
                    if is_primary_v1:
                        # Get metadata
                        metadata = getattr(item, 'metadata', {}) or {}

                        # Get or generate segment ID
                        seg_id = metadata.get('segment_id')
                        if not seg_id:
                            seg_id = f"S{v1_segment_idx:03d}"
                        self.segment_order.append(seg_id)
                        self.segment_to_filename[seg_id] = filename
                        v1_segment_idx += 1

                        # Get confidence
                        confidence = metadata.get('confidence', 0.0)
                        if isinstance(confidence, (int, float)):
                            self.v1_confidences[filename] = float(confidence)

                        # Calculate duration
                        if item.source_range:
                            duration_frames = int(item.source_range.duration.value)
                            v1_total_frames += duration_frames

        # Calculate original voiceover duration from V1 track
        self.original_duration_sec = v1_total_frames / self.frame_rate if self.frame_rate > 0 else 0

        total = sum(len(s) for s in self.recommendations.values())
        logger.info(f"Loaded {total} recommended clips from original OTIO")
        logger.info(f"  V1: {len(self.recommendations['v1'])}, "
                   f"V2-V3: {len(self.recommendations['v2_v3'])}, "
                   f"V4-V6: {len(self.recommendations['v4_v6'])}, "
                   f"V7+: {len(self.recommendations['v7_plus'])}")
        logger.info(f"  Original duration: {_format_timecode(self.original_duration_sec)}")

    def analyze(
        self,
        edited_otio_path: str,
        include_duration: bool = False,
        include_confidence: bool = False,
        include_coverage_map: bool = False,
        bucket_size: int = 25
    ) -> FilenameAnalysisResult:
        """
        Analyze edited OTIO by filename matching.

        Args:
            edited_otio_path: Path to the edited OTIO export
            include_duration: Include duration comparison analysis
            include_confidence: Include confidence correlation analysis
            include_coverage_map: Include segment coverage map visualization
            bucket_size: Number of segments per coverage map bucket

        Returns:
            FilenameAnalysisResult with detailed analysis
        """
        try:
            import opentimelineio as otio
        except ImportError:
            raise ImportError("opentimelineio is required")

        # Use segment_order length if available (actual V1 clip count),
        # otherwise fall back to unique V1 filenames
        total_segs = len(self.segment_order) if self.segment_order else len(self.recommendations['v1'])

        result = FilenameAnalysisResult(
            analyzed_at=datetime.now().isoformat(),
            original_file=self.original_path,
            edited_file=edited_otio_path,
            total_segments=total_segs
        )

        timeline = otio.adapters.read_from_file(edited_otio_path)

        # Get frame rate from edited timeline
        edited_frame_rate = self.frame_rate
        if timeline.global_start_time:
            edited_frame_rate = timeline.global_start_time.rate

        # Collect all used files with their track info
        used_files: Dict[str, str] = {}  # filename -> track_name
        edited_total_frames = 0
        user_track_clips: Dict[str, List[str]] = {}  # track_name -> [filenames]

        # Build set of all recommendations for clip-based detection
        all_recommendations = (
            self.recommendations['v1'] |
            self.recommendations['v2_v3'] |
            self.recommendations['v4_v6'] |
            self.recommendations['v7_plus']
        )

        for track in timeline.tracks:
            # Check if video track (handle both enum and string)
            track_kind = str(track.kind) if hasattr(track.kind, 'name') else track.kind
            if track_kind not in ('Video', 'TrackKind.Video'):
                continue

            track_name = track.name or "Unknown"

            # Check if this is a matcher-generated track or user-added (by name)
            # Uses regex to properly distinguish V1 from V10, etc.
            is_matcher_track_by_name = _is_matcher_track(track_name)

            for item in track:
                if not isinstance(item, otio.schema.Clip):
                    continue

                # Check enabled state
                is_enabled = getattr(item, 'enabled', True)
                if hasattr(item, 'metadata') and 'enabled' in item.metadata:
                    is_enabled = item.metadata.get('enabled', True)

                if not is_enabled:
                    continue

                filename = self._extract_filename(item)
                if filename:
                    # Check if this clip is from recommendations (handles renamed tracks)
                    is_recommendation_clip = filename in all_recommendations

                    # User-added track with non-recommendation clip = truly user content
                    if not is_matcher_track_by_name and not is_recommendation_clip:
                        if track_name not in user_track_clips:
                            user_track_clips[track_name] = []
                        user_track_clips[track_name].append(filename)
                    # Matcher track OR recommendation clip = analyze for matching
                    elif filename not in used_files:
                        used_files[filename] = track_name

                        # Count duration only for first occurrence (avoid double counting)
                        if include_duration and item.source_range:
                            edited_total_frames += int(item.source_range.duration.value)

        # Categorize used files
        for filename, track_name in used_files.items():
            # Update track breakdown
            result.track_breakdown[track_name] = result.track_breakdown.get(track_name, 0) + 1

            # Categorize
            if filename in self.recommendations['v1']:
                result.v1_kept += 1
                result.v1_clips_used.append(filename)
            elif filename in self.recommendations['v2_v3']:
                result.v2_v3_used += 1
                result.alternative_clips_used.append(filename)
            elif filename in self.recommendations['v4_v6']:
                result.v4_v6_used += 1
                result.secondary_clips_used.append(filename)
            elif filename in self.recommendations['v7_plus']:
                result.v7_plus_used += 1
                result.strategy_clips_used.append(filename)
            else:
                result.external_added += 1
                result.external_clips.append(filename)

        # Find dropped V1 clips
        used_file_set = set(used_files.keys())
        result.dropped_v1_clips = list(self.recommendations['v1'] - used_file_set)
        result.segments_dropped = len(result.dropped_v1_clips)

        # Add user-added track info
        for track_name, clips in user_track_clips.items():
            result.user_added_tracks[track_name] = len(clips)
            result.user_added_clips.extend(clips)

        # Calculate percentage
        if result.total_segments > 0:
            result.v1_kept_pct = (result.v1_kept / result.total_segments) * 100

        # Duration comparison
        if include_duration:
            result.duration_enabled = True
            result.original_duration_sec = self.original_duration_sec
            result.edited_duration_sec = edited_total_frames / edited_frame_rate if edited_frame_rate > 0 else 0
            if result.original_duration_sec > 0:
                result.coverage_ratio = result.edited_duration_sec / result.original_duration_sec
            else:
                result.coverage_ratio = 0.0

        # Confidence correlation
        if include_confidence:
            result.confidence_enabled = True

            # Collect confidence values for kept and dropped V1 clips
            for filename in result.v1_clips_used:
                if filename in self.v1_confidences:
                    result.kept_confidences.append(self.v1_confidences[filename])

            for filename in result.dropped_v1_clips:
                if filename in self.v1_confidences:
                    result.dropped_confidences.append(self.v1_confidences[filename])

            # Calculate averages
            if result.kept_confidences:
                result.kept_avg_confidence = sum(result.kept_confidences) / len(result.kept_confidences)
            if result.dropped_confidences:
                result.dropped_avg_confidence = sum(result.dropped_confidences) / len(result.dropped_confidences)

            # Determine correlation
            if result.kept_confidences and result.dropped_confidences:
                diff = result.kept_avg_confidence - result.dropped_avg_confidence
                if diff > 0.05:
                    result.confidence_correlation = "positive"
                elif diff > 0.02:
                    result.confidence_correlation = "weak_positive"
                elif diff < -0.05:
                    result.confidence_correlation = "negative"
                elif diff < -0.02:
                    result.confidence_correlation = "weak_negative"
                else:
                    result.confidence_correlation = "none"
            else:
                result.confidence_correlation = "insufficient_data"

        # Calculate segment-level coverage (how many V1 segments have their file used)
        if self.segment_order:
            for seg_id in self.segment_order:
                filename = self.segment_to_filename.get(seg_id)
                if filename and filename in used_file_set:
                    result.segments_covered += 1
                else:
                    result.segments_not_covered += 1

        # Coverage map
        if include_coverage_map and self.segment_order:
            result.coverage_map_enabled = True
            result.coverage_map = self._build_coverage_map(
                used_filenames=used_file_set,
                bucket_size=bucket_size
            )

        return result

    def _extract_filename(self, clip) -> str:
        """Extract normalized filename from OTIO clip."""
        try:
            # Try media_references (may be a method in newer OTIO versions)
            if hasattr(clip, 'media_references'):
                refs = clip.media_references
                # Handle both method (newer OTIO) and property (older OTIO)
                if callable(refs):
                    refs = refs()
                if isinstance(refs, dict):
                    ref = refs.get('DEFAULT_MEDIA')
                    if ref and hasattr(ref, 'target_url') and ref.target_url:
                        return self._normalize(os.path.basename(ref.target_url))

            # Try direct media_reference
            if hasattr(clip, 'media_reference') and clip.media_reference:
                ref = clip.media_reference
                if hasattr(ref, 'target_url') and ref.target_url:
                    return self._normalize(os.path.basename(ref.target_url))

            # Fall back to clip name
            if clip.name:
                return self._normalize(clip.name)

            return ""
        except Exception:
            return ""

    def _normalize(self, filename: str) -> str:
        """Normalize filename for comparison."""
        if not filename:
            return ""

        # Get just the filename
        name = os.path.basename(filename)

        # Case-insensitive extension removal
        name_lower = name.lower()
        extensions = ['.mp4', '.mov', '.avi', '.mkv', '.webm', '.m4v', '.flv', '.wmv', '.mpg', '.mpeg']
        for ext in extensions:
            if name_lower.endswith(ext):
                name = name[:-len(ext)]
                break

        # Strip audio-first segment suffix (e.g., abc123_0045 -> abc123)
        # This allows segment files to match their source video recommendations
        segment_match = re.match(r'^(.+)_\d{4}$', name)
        if segment_match:
            name = segment_match.group(1)

        return name

    def _build_coverage_map(
        self,
        used_filenames: Set[str],
        bucket_size: int = 25
    ) -> List[Dict[str, Any]]:
        """
        Group segments into buckets and calculate coverage per bucket.

        Args:
            used_filenames: Set of filenames that were used in the edited timeline
            bucket_size: Number of segments per bucket

        Returns:
            List of bucket dictionaries with coverage info
        """
        coverage_map = []

        for i in range(0, len(self.segment_order), bucket_size):
            bucket_segments = self.segment_order[i:i + bucket_size]

            # Count how many segments in this bucket have their V1 clip used
            covered = 0
            covered_segments = []
            for seg_id in bucket_segments:
                filename = self.segment_to_filename.get(seg_id)
                if filename and filename in used_filenames:
                    covered += 1
                    covered_segments.append(seg_id)

            total = len(bucket_segments)
            coverage_pct = (covered / total * 100) if total > 0 else 0

            coverage_map.append({
                'start_segment': bucket_segments[0],
                'end_segment': bucket_segments[-1],
                'total': total,
                'covered': covered,
                'coverage_pct': round(coverage_pct, 1),
                'covered_segments': covered_segments
            })

        return coverage_map


# =============================================================================
# POSITION-BASED ANALYZER (kept for QA/testing)
# =============================================================================

class PositionAnalyzer:
    """
    Analyzes edited timeline by matching clips to segments via timeline position.

    This approach requires the timeline structure to be mostly preserved.
    Use for QA testing or when clips haven't been moved.
    """

    def __init__(self, segment_map_path: str):
        """
        Initialize with segment map.

        Args:
            segment_map_path: Path to timeline_segments.json
        """
        self.segment_map_path = segment_map_path
        self.segments: List[Dict[str, Any]] = []
        self.frame_rate: float = 30.0

        self._load_segment_map()

    def _load_segment_map(self) -> None:
        """Load segment map from JSON file."""
        with open(self.segment_map_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.frame_rate = data.get('frame_rate', 30.0)
        self.segments = data.get('segments', [])

        logger.info(f"Loaded segment map with {len(self.segments)} segments")

    def analyze(self, edited_otio_path: str) -> PositionAnalysisResult:
        """
        Analyze edited OTIO by position matching.

        Args:
            edited_otio_path: Path to the edited OTIO export

        Returns:
            PositionAnalysisResult with analysis
        """
        try:
            import opentimelineio as otio
        except ImportError:
            raise ImportError("opentimelineio is required")

        result = PositionAnalysisResult(
            analyzed_at=datetime.now().isoformat(),
            edited_file=edited_otio_path,
            segment_map_file=self.segment_map_path,
            frame_rate=self.frame_rate,
            total_segments=len(self.segments)
        )

        timeline = otio.adapters.read_from_file(edited_otio_path)

        # Build lookup of all known clips
        original_clips: Dict[str, Tuple[str, str]] = {}
        for seg in self.segments:
            v1_file = seg.get('v1_clip', {}).get('file', '')
            if v1_file:
                original_clips[self._normalize(v1_file)] = (seg['id'], "v1")
            for alt in seg.get('alternatives', []):
                original_clips[self._normalize(alt['file'])] = (seg['id'], "alternative")
            for sec in seg.get('secondary', []):
                original_clips[self._normalize(sec['file'])] = (seg['id'], "secondary")

        # Extract clips from edited timeline
        clips_by_track: Dict[str, List[Dict]] = {}

        for track in timeline.tracks:
            # Check if video track (handle both enum and string)
            track_kind = str(track.kind) if hasattr(track.kind, 'name') else track.kind
            if track_kind not in ('Video', 'TrackKind.Video'):
                continue

            track_name = track.name or "V1"
            track_number = 1
            num_match = re.search(r'V(\d+)', track_name)
            if num_match:
                track_number = int(num_match.group(1))

            clips_by_track[track_name] = []
            position_frames = 0

            for item in track:
                if isinstance(item, otio.schema.Gap):
                    duration = item.source_range.duration if item.source_range else item.duration()
                    position_frames += int(duration.value)
                    continue

                if not isinstance(item, otio.schema.Clip):
                    continue

                if item.source_range:
                    duration_frames = int(item.source_range.duration.value)
                else:
                    duration_frames = int(item.duration().value) if item.duration() else 0

                is_enabled = getattr(item, 'enabled', True)

                filename = self._extract_filename(item)

                clips_by_track[track_name].append({
                    'filename': filename,
                    'track_name': track_name,
                    'track_number': track_number,
                    'start_frame': position_frames,
                    'end_frame': position_frames + duration_frames,
                    'is_enabled': is_enabled
                })

                position_frames += duration_frames

        # Match segments to clips by position
        for seg in self.segments:
            seg_start = seg['start_frame']
            seg_end = seg['end_frame']
            seg_id = seg['id']
            v1_file = seg.get('v1_clip', {}).get('file', '')

            # Find overlapping enabled clips
            overlapping = []
            for track_name, clips in clips_by_track.items():
                for clip in clips:
                    if not clip['is_enabled']:
                        continue

                    overlap_start = max(clip['start_frame'], seg_start)
                    overlap_end = min(clip['end_frame'], seg_end)

                    if overlap_end > overlap_start:
                        seg_len = seg_end - seg_start
                        overlap_ratio = (overlap_end - overlap_start) / seg_len if seg_len > 0 else 0
                        if overlap_ratio > 0.5:
                            overlapping.append(clip)

            if not overlapping:
                result.missing += 1
                continue

            # Pick highest track
            overlapping.sort(key=lambda c: c['track_number'], reverse=True)
            selected = overlapping[0]

            result.track_usage[selected['track_name']] = result.track_usage.get(selected['track_name'], 0) + 1

            # Compare to original
            if self._normalize(selected['filename']) == self._normalize(v1_file):
                result.v1_kept += 1
            else:
                norm_selected = self._normalize(selected['filename'])
                if norm_selected in original_clips:
                    _, source_type = original_clips[norm_selected]
                    if source_type == "alternative":
                        result.replaced_with_alt += 1
                    elif source_type == "secondary":
                        result.replaced_with_secondary += 1
                else:
                    result.replaced_with_external += 1

                result.replacements.append({
                    "segment": seg_id,
                    "original": v1_file,
                    "replaced_with": selected['filename'],
                    "new_track": selected['track_name']
                })

        if result.total_segments > 0:
            result.v1_kept_pct = (result.v1_kept / result.total_segments) * 100

        return result

    def _extract_filename(self, clip) -> str:
        """Extract filename from OTIO clip."""
        try:
            # Try media_references (may be a method in newer OTIO versions)
            if hasattr(clip, 'media_references'):
                refs = clip.media_references
                # Handle both method (newer OTIO) and property (older OTIO)
                if callable(refs):
                    refs = refs()
                if isinstance(refs, dict):
                    ref = refs.get('DEFAULT_MEDIA')
                    if ref and hasattr(ref, 'target_url') and ref.target_url:
                        return os.path.basename(ref.target_url)

            if hasattr(clip, 'media_reference') and clip.media_reference:
                ref = clip.media_reference
                if hasattr(ref, 'target_url') and ref.target_url:
                    return os.path.basename(ref.target_url)

            return clip.name or ""
        except Exception:
            return ""

    def _normalize(self, filename: str) -> str:
        """Normalize filename for comparison."""
        if not filename:
            return ""
        name = os.path.basename(filename)
        name = os.path.splitext(name)[0]
        return name.lower()


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def analyze_filename_based(
    edited_path: str,
    original_path: str,
    output_path: Optional[str] = None,
    verbose: bool = True,
    include_duration: bool = False,
    include_confidence: bool = False,
    include_coverage_map: bool = False,
    bucket_size: int = 25
) -> FilenameAnalysisResult:
    """
    Analyze edited timeline using filename-based matching.

    Args:
        edited_path: Path to edited OTIO export
        original_path: Path to original timeline_FULL.otio
        output_path: Optional path to save JSON report
        verbose: Whether to print summary
        include_duration: Include duration comparison analysis
        include_confidence: Include confidence correlation analysis
        include_coverage_map: Include segment coverage map visualization
        bucket_size: Number of segments per coverage map bucket

    Returns:
        FilenameAnalysisResult with analysis
    """
    analyzer = FilenameAnalyzer(original_path)
    result = analyzer.analyze(
        edited_path,
        include_duration=include_duration,
        include_confidence=include_confidence,
        include_coverage_map=include_coverage_map,
        bucket_size=bucket_size
    )

    if verbose:
        print(result.summary())

    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result.to_dict(), f, indent=2)
        print(f"\nReport saved to: {output_path}")

    return result


def analyze_position_based(
    edited_path: str,
    segment_map_path: str,
    output_path: Optional[str] = None,
    verbose: bool = True
) -> PositionAnalysisResult:
    """
    Analyze edited timeline using position-based matching.

    Args:
        edited_path: Path to edited OTIO export
        segment_map_path: Path to segment map JSON
        output_path: Optional path to save JSON report
        verbose: Whether to print summary

    Returns:
        PositionAnalysisResult with analysis
    """
    analyzer = PositionAnalyzer(segment_map_path)
    result = analyzer.analyze(edited_path)

    if verbose:
        print(result.summary())

    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result.to_dict(), f, indent=2)
        print(f"\nReport saved to: {output_path}")

    return result


def extract_segment_map_from_otio(otio_path: str) -> Optional[str]:
    """
    Extract segment information from an original OTIO file.

    Args:
        otio_path: Path to original OTIO file

    Returns:
        Path to generated segment map JSON, or None if failed
    """
    try:
        import opentimelineio as otio
    except ImportError:
        logger.error("OpenTimelineIO not installed")
        return None

    timeline = otio.adapters.read_from_file(otio_path)

    frame_rate = 30.0
    if timeline.global_start_time:
        frame_rate = timeline.global_start_time.rate

    segments = []
    position_frames = 0

    # Find V1 track
    v1_track = None
    for track in timeline.tracks:
        track_kind = str(track.kind) if hasattr(track.kind, 'name') else track.kind
        if track_kind not in ('Video', 'TrackKind.Video'):
            continue
        if 'V1' in (track.name or '') or 'Primary' in (track.name or '').lower():
            v1_track = track
            break

    if not v1_track:
        for track in timeline.tracks:
            track_kind = str(track.kind) if hasattr(track.kind, 'name') else track.kind
            if track_kind in ('Video', 'TrackKind.Video'):
                v1_track = track
                break

    if not v1_track:
        logger.error("No video track found")
        return None

    seg_idx = 0
    for item in v1_track:
        if isinstance(item, otio.schema.Gap):
            duration = item.source_range.duration if item.source_range else item.duration()
            position_frames += int(duration.value)
            continue

        if not isinstance(item, otio.schema.Clip):
            continue

        duration_frames = int(item.source_range.duration.value) if item.source_range else 0
        end_frame = position_frames + duration_frames

        # Get filename
        filename = ""
        if hasattr(item, 'media_reference') and item.media_reference:
            ref = item.media_reference
            if hasattr(ref, 'target_url') and ref.target_url:
                filename = os.path.basename(ref.target_url)

        metadata = item.metadata or {}

        segments.append({
            "id": metadata.get('segment_id', f"S{seg_idx:03d}"),
            "start_frame": position_frames,
            "end_frame": end_frame,
            "start_tc": "",
            "end_tc": "",
            "voiceover_text": metadata.get('voiceover_text', ''),
            "duration_sec": duration_frames / frame_rate,
            "v1_clip": {
                "file": filename,
                "confidence": metadata.get('confidence', 0.0)
            }
        })

        position_frames = end_frame
        seg_idx += 1

    if not segments:
        logger.error("No segments found")
        return None

    segment_map = {
        "generated_at": datetime.now().isoformat(),
        "source_srt": "",
        "frame_rate": frame_rate,
        "total_segments": len(segments),
        "segments": segments
    }

    output_path = str(Path(otio_path).with_suffix('')) + '_extracted_segments.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(segment_map, f, indent=2)

    logger.info(f"Extracted segment map to {output_path}")
    return output_path


# =============================================================================
# CLI
# =============================================================================

def main():
    """Command-line interface for post-edit analysis."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Analyze edited timeline to understand clip selections',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Filename-based analysis (default, recommended)
  python -m src.post_edit_analysis --edited export.otio --original timeline_FULL.otio

  # Include duration comparison
  python -m src.post_edit_analysis --edited export.otio --original timeline_FULL.otio --duration

  # Include confidence correlation
  python -m src.post_edit_analysis --edited export.otio --original timeline_FULL.otio --confidence

  # Include segment coverage map
  python -m src.post_edit_analysis --edited export.otio --original timeline_FULL.otio --coverage-map

  # Coverage map with custom bucket size
  python -m src.post_edit_analysis --edited export.otio --original timeline_FULL.otio --coverage-map -b 10

  # Full analysis with all metrics
  python -m src.post_edit_analysis --edited export.otio --original timeline_FULL.otio --verbose

  # Position-based analysis (for QA testing)
  python -m src.post_edit_analysis --edited export.otio --segments timeline_segments.json --mode position

  # Save report to JSON
  python -m src.post_edit_analysis --edited export.otio --original timeline_FULL.otio --output report.json
        """
    )

    parser.add_argument(
        '--edited', '-e',
        required=True,
        help='Path to edited OTIO export from DaVinci Resolve'
    )

    parser.add_argument(
        '--original', '-o',
        help='Path to original timeline_FULL.otio (for filename-based analysis)'
    )

    parser.add_argument(
        '--segments', '-s',
        help='Path to segment map JSON (for position-based analysis)'
    )

    parser.add_argument(
        '--mode', '-m',
        choices=['filename', 'position'],
        default='filename',
        help='Analysis mode: filename (default) or position'
    )

    parser.add_argument(
        '--output', '-O',
        help='Path to save JSON analysis report'
    )

    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress summary output'
    )

    parser.add_argument(
        '--duration', '-d',
        action='store_true',
        help='Include duration comparison (original vs edited timeline)'
    )

    parser.add_argument(
        '--confidence', '-c',
        action='store_true',
        help='Include confidence correlation analysis (kept vs dropped clips)'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Include all available metrics (duration + confidence + coverage map)'
    )

    parser.add_argument(
        '--coverage-map',
        action='store_true',
        help='Include segment coverage map visualization'
    )

    parser.add_argument(
        '--bucket-size', '-b',
        type=int,
        default=25,
        help='Number of segments per coverage map bucket (default: 25)'
    )

    args = parser.parse_args()

    # Validate inputs
    if not Path(args.edited).exists():
        print(f"ERROR: Edited file not found: {args.edited}")
        return 1

    if args.mode == 'filename':
        if not args.original:
            print("ERROR: --original is required for filename-based analysis")
            return 1
        if not Path(args.original).exists():
            print(f"ERROR: Original file not found: {args.original}")
            return 1

        # Determine which optional analyses to include
        include_duration = args.duration or args.verbose
        include_confidence = args.confidence or args.verbose
        include_coverage_map = getattr(args, 'coverage_map', False) or args.verbose

        try:
            analyze_filename_based(
                edited_path=args.edited,
                original_path=args.original,
                output_path=args.output,
                verbose=not args.quiet,
                include_duration=include_duration,
                include_confidence=include_confidence,
                include_coverage_map=include_coverage_map,
                bucket_size=args.bucket_size
            )
            return 0
        except Exception as e:
            print(f"ERROR: {e}")
            logger.exception("Analysis failed")
            return 1

    elif args.mode == 'position':
        segment_map_path = args.segments

        if not segment_map_path and args.original:
            print(f"Extracting segment map from {args.original}...")
            segment_map_path = extract_segment_map_from_otio(args.original)
            if not segment_map_path:
                print("ERROR: Could not extract segment map")
                return 1

        if not segment_map_path:
            print("ERROR: --segments or --original required for position-based analysis")
            return 1

        if not Path(segment_map_path).exists():
            print(f"ERROR: Segment map not found: {segment_map_path}")
            return 1

        try:
            analyze_position_based(
                edited_path=args.edited,
                segment_map_path=segment_map_path,
                output_path=args.output,
                verbose=not args.quiet
            )
            return 0
        except Exception as e:
            print(f"ERROR: {e}")
            logger.exception("Analysis failed")
            return 1


if __name__ == '__main__':
    import sys
    sys.exit(main())
