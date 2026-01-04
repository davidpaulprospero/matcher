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
# DATA CLASSES
# =============================================================================

@dataclass
class FilenameAnalysisResult:
    """Results from filename-based analysis."""
    mode: str = "filename"
    analyzed_at: str = ""
    original_file: str = ""
    edited_file: str = ""

    # Counts
    total_segments: int = 0
    v1_kept: int = 0
    v1_kept_pct: float = 0.0
    v2_v3_used: int = 0
    v4_v6_used: int = 0
    v7_plus_used: int = 0
    external_added: int = 0
    segments_dropped: int = 0

    # Track breakdown
    track_breakdown: Dict[str, int] = field(default_factory=dict)

    # Detailed lists
    v1_clips_used: List[str] = field(default_factory=list)
    alternative_clips_used: List[str] = field(default_factory=list)
    secondary_clips_used: List[str] = field(default_factory=list)
    strategy_clips_used: List[str] = field(default_factory=list)
    external_clips: List[str] = field(default_factory=list)
    dropped_v1_clips: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
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
                "segments_dropped": self.segments_dropped
            },
            "track_breakdown": self.track_breakdown,
            "external_clips": self.external_clips[:50],  # Limit for readability
            "dropped_v1_clips": self.dropped_v1_clips[:50]
        }

    def summary(self) -> str:
        """Generate human-readable summary."""
        total_used = self.v1_kept + self.v2_v3_used + self.v4_v6_used + self.v7_plus_used + self.external_added

        lines = [
            "=" * 65,
            "                    POST-EDIT ANALYSIS REPORT",
            "=" * 65,
            "",
            f"Original: {Path(self.original_file).name}",
            f"Edited:   {Path(self.edited_file).name}",
            f"Analyzed: {self.analyzed_at}",
            "",
            f"ORIGINAL RECOMMENDATIONS: {self.total_segments} segments",
            "",
            f"V1 CLIPS USED:           {self.v1_kept:>4} ({self.v1_kept_pct:>5.1f}%)  [OK] System picks kept",
            f"V2-V3 ALTERNATIVES:      {self.v2_v3_used:>4} ({self._pct(self.v2_v3_used):>5.1f}%)  [^] Editor preferred alt",
            f"V4-V6 SECONDARY:         {self.v4_v6_used:>4} ({self._pct(self.v4_v6_used):>5.1f}%)  [^] Editor preferred secondary",
            f"V7+ STRATEGY:            {self.v7_plus_used:>4} ({self._pct(self.v7_plus_used):>5.1f}%)  [^] Diversity/experimental",
            f"EXTERNAL CLIPS:          {self.external_added:>4} ({self._pct(self.external_added):>5.1f}%)  [*] Editor's own footage",
            f"SEGMENTS DROPPED:        {self.segments_dropped:>4} ({self._pct(self.segments_dropped):>5.1f}%)  [X] Deleted from final",
            "",
            "-" * 65,
            "TRACK PREFERENCE BREAKDOWN",
            "-" * 65,
        ]

        for track, count in sorted(self.track_breakdown.items()):
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

        self._load_recommendations()

    def _load_recommendations(self) -> None:
        """Load all recommended clips from original OTIO, organized by track."""
        try:
            import opentimelineio as otio
        except ImportError:
            raise ImportError("opentimelineio is required")

        timeline = otio.adapters.read_from_file(self.original_path)

        for track in timeline.tracks:
            # Check if video track (handle both enum and string)
            track_kind = str(track.kind) if hasattr(track.kind, 'name') else track.kind
            if track_kind not in ('Video', 'TrackKind.Video'):
                continue

            track_name = track.name or ""

            # Categorize track
            if 'V1' in track_name or 'Primary' in track_name:
                category = 'v1'
            elif 'V2' in track_name or 'V3' in track_name or 'Alternative' in track_name:
                category = 'v2_v3'
            elif 'V4' in track_name or 'V5' in track_name or 'V6' in track_name or 'Secondary' in track_name:
                category = 'v4_v6'
            elif 'V7' in track_name or 'V8' in track_name or 'V9' in track_name or 'V10' in track_name:
                category = 'v7_plus'
            elif 'Embedding' in track_name or 'Diversity' in track_name or 'Strategy' in track_name:
                category = 'v7_plus'
            elif 'Entity' in track_name or 'Stock' in track_name:
                category = 'v7_plus'
            else:
                # Default to v1 for unknown tracks
                category = 'v1'

            self.track_to_category[track_name] = category

            for item in track:
                if not isinstance(item, otio.schema.Clip):
                    continue

                filename = self._extract_filename(item)
                if filename:
                    self.recommendations[category].add(filename)

        total = sum(len(s) for s in self.recommendations.values())
        logger.info(f"Loaded {total} recommended clips from original OTIO")
        logger.info(f"  V1: {len(self.recommendations['v1'])}, "
                   f"V2-V3: {len(self.recommendations['v2_v3'])}, "
                   f"V4-V6: {len(self.recommendations['v4_v6'])}, "
                   f"V7+: {len(self.recommendations['v7_plus'])}")

    def analyze(self, edited_otio_path: str) -> FilenameAnalysisResult:
        """
        Analyze edited OTIO by filename matching.

        Args:
            edited_otio_path: Path to the edited OTIO export

        Returns:
            FilenameAnalysisResult with detailed analysis
        """
        try:
            import opentimelineio as otio
        except ImportError:
            raise ImportError("opentimelineio is required")

        result = FilenameAnalysisResult(
            analyzed_at=datetime.now().isoformat(),
            original_file=self.original_path,
            edited_file=edited_otio_path,
            total_segments=len(self.recommendations['v1'])
        )

        timeline = otio.adapters.read_from_file(edited_otio_path)

        # Collect all used files with their track info
        used_files: Dict[str, str] = {}  # filename -> track_name

        for track in timeline.tracks:
            # Check if video track (handle both enum and string)
            track_kind = str(track.kind) if hasattr(track.kind, 'name') else track.kind
            if track_kind not in ('Video', 'TrackKind.Video'):
                continue

            track_name = track.name or "Unknown"

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
                    # Track the first (or highest) track where this file appears
                    if filename not in used_files:
                        used_files[filename] = track_name

        # Build set of all recommendations
        all_recommendations = (
            self.recommendations['v1'] |
            self.recommendations['v2_v3'] |
            self.recommendations['v4_v6'] |
            self.recommendations['v7_plus']
        )

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

        # Calculate percentage
        if result.total_segments > 0:
            result.v1_kept_pct = (result.v1_kept / result.total_segments) * 100

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

        # Remove common video extensions for comparison
        for ext in ['.mp4', '.mov', '.avi', '.mkv', '.webm', '.MP4', '.MOV']:
            if name.endswith(ext):
                name = name[:-len(ext)]
                break

        return name


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
    verbose: bool = True
) -> FilenameAnalysisResult:
    """
    Analyze edited timeline using filename-based matching.

    Args:
        edited_path: Path to edited OTIO export
        original_path: Path to original timeline_FULL.otio
        output_path: Optional path to save JSON report
        verbose: Whether to print summary

    Returns:
        FilenameAnalysisResult with analysis
    """
    analyzer = FilenameAnalyzer(original_path)
    result = analyzer.analyze(edited_path)

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

        try:
            analyze_filename_based(
                edited_path=args.edited,
                original_path=args.original,
                output_path=args.output,
                verbose=not args.quiet
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
