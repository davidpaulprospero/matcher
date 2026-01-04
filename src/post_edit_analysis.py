"""
Post-Edit Analysis Tool

Analyzes final OTIO/XML exported from DaVinci Resolve to understand which clips
were selected from the original match recommendations.

Since DaVinci Resolve strips custom metadata and segment IDs from clip names,
this tool uses TIMECODE POSITION MATCHING against a segment map JSON file
generated during the original timeline creation.

This tool helps:
1. Track editor preferences for future matching improvements
2. Feed successful selections back to global cache
3. Understand which types of matches get kept vs replaced

Usage:
    python -m src.post_edit_analysis --edited export.otio --segments timeline_segments.json
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class SegmentInfo:
    """Segment information from the segment map."""
    id: str  # e.g., "S000"
    index: int  # 0, 1, 2...
    start_frame: int
    end_frame: int
    start_tc: str
    end_tc: str
    voiceover_text: str
    duration_sec: float
    v1_clip: str  # Original V1 clip filename
    v1_confidence: float
    alternatives: List[Dict[str, Any]] = field(default_factory=list)
    secondary: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class EditedClip:
    """A clip found in the edited timeline."""
    filename: str
    track_name: str
    track_number: int
    start_frame: int
    end_frame: int
    duration_frames: int
    is_enabled: bool = True
    source_start: float = 0.0
    source_end: float = 0.0


@dataclass
class SegmentSelection:
    """Which clip was selected for a segment after editing."""
    segment_id: str
    segment_index: int
    voiceover_text: str

    # Selection info
    selected_clip: Optional[str] = None  # Filename of selected clip
    selected_track: Optional[str] = None  # Track name (e.g., "V1 - Primary")
    selected_track_number: int = 0

    # Original recommendation
    original_v1_clip: str = ""
    original_confidence: float = 0.0

    # Analysis flags
    was_v1_kept: bool = False  # True if V1 clip was kept
    was_replaced: bool = False  # True if a different clip was selected
    replacement_source: str = ""  # "alternative", "secondary", "external"
    is_disabled: bool = False  # True if all clips for this segment are disabled
    is_missing: bool = False  # True if no clip found for this segment


@dataclass
class AnalysisResult:
    """Complete analysis results."""
    analyzed_at: str
    edited_file: str
    segment_map_file: str
    frame_rate: float

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

    # Detailed selections
    selections: List[SegmentSelection] = field(default_factory=list)
    replacements: List[Dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "analyzed_at": self.analyzed_at,
            "edited_file": self.edited_file,
            "segment_map_file": self.segment_map_file,
            "summary": {
                "total_segments": self.total_segments,
                "v1_kept": self.v1_kept,
                "v1_kept_pct": round(self.v1_kept_pct, 1),
                "replaced": self.total_segments - self.v1_kept - self.all_disabled - self.missing,
                "replaced_pct": round(100 - self.v1_kept_pct - (self.all_disabled + self.missing) / max(1, self.total_segments) * 100, 1),
                "all_disabled": self.all_disabled,
                "missing": self.missing
            },
            "track_usage": self.track_usage,
            "selections": [
                {
                    "segment": s.segment_id,
                    "track": s.selected_track,
                    "file": s.selected_clip,
                    "was_v1": s.was_v1_kept
                }
                for s in self.selections
            ],
            "replacements": self.replacements
        }

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            "=" * 50,
            "POST-EDIT ANALYSIS SUMMARY",
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
# TIMECODE ANALYZER
# =============================================================================

class TimecodeAnalyzer:
    """
    Analyzes edited timeline by matching clips to segments via timeline position.

    This approach works even when DaVinci Resolve strips metadata because:
    1. We know the exact frame range for each segment from the segment map
    2. Clips in the edited timeline have start/end positions
    3. We match clips to segments by position overlap
    """

    def __init__(self, segment_map_path: str):
        """
        Initialize with segment map.

        Args:
            segment_map_path: Path to timeline_segments.json
        """
        self.segment_map_path = segment_map_path
        self.segments: List[SegmentInfo] = []
        self.frame_rate: float = 30.0
        self.timeline_start_tc: str = "01:00:00:00"

        self._load_segment_map()

    def _load_segment_map(self) -> None:
        """Load segment map from JSON file."""
        with open(self.segment_map_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.frame_rate = data.get('frame_rate', 30.0)
        self.timeline_start_tc = data.get('timeline_start_tc', '01:00:00:00')

        for idx, seg in enumerate(data.get('segments', [])):
            info = SegmentInfo(
                id=seg['id'],
                index=idx,
                start_frame=seg['start_frame'],
                end_frame=seg['end_frame'],
                start_tc=seg['start_tc'],
                end_tc=seg['end_tc'],
                voiceover_text=seg.get('voiceover_text', ''),
                duration_sec=seg.get('duration_sec', 0.0),
                v1_clip=seg.get('v1_clip', {}).get('file', ''),
                v1_confidence=seg.get('v1_clip', {}).get('confidence', 0.0),
                alternatives=seg.get('alternatives', []),
                secondary=seg.get('secondary', [])
            )
            self.segments.append(info)

        logger.info(f"Loaded segment map with {len(self.segments)} segments")

    def analyze_otio(self, otio_path: str) -> AnalysisResult:
        """
        Analyze an edited OTIO file.

        Args:
            otio_path: Path to the edited OTIO export

        Returns:
            AnalysisResult with detailed analysis
        """
        try:
            import opentimelineio as otio
        except ImportError:
            logger.error("OpenTimelineIO not installed")
            raise ImportError("opentimelineio is required for OTIO analysis")

        timeline = otio.adapters.read_from_file(otio_path)

        # Extract clips from all video tracks
        clips_by_track: Dict[str, List[EditedClip]] = {}

        for track in timeline.tracks:
            if track.kind != otio.schema.TrackKind.Video:
                continue

            track_name = track.name or "V1"

            # Extract track number
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

                # Get duration
                if item.source_range:
                    duration_frames = int(item.source_range.duration.value)
                else:
                    duration_frames = int(item.duration().value) if item.duration() else 0

                # Get enabled state
                is_enabled = getattr(item, 'enabled', True)
                if 'enabled' in item.metadata:
                    is_enabled = item.metadata.get('enabled', True)

                # Extract filename from media reference
                filename = self._extract_filename(item)

                clip = EditedClip(
                    filename=filename,
                    track_name=track_name,
                    track_number=track_number,
                    start_frame=position_frames,
                    end_frame=position_frames + duration_frames,
                    duration_frames=duration_frames,
                    is_enabled=is_enabled
                )

                clips_by_track[track_name].append(clip)
                position_frames += duration_frames

        return self._build_analysis(otio_path, clips_by_track)

    def analyze_xml(self, xml_path: str) -> AnalysisResult:
        """
        Analyze an edited XML file.

        Args:
            xml_path: Path to the edited XML export

        Returns:
            AnalysisResult with detailed analysis
        """
        import xml.etree.ElementTree as ET

        tree = ET.parse(xml_path)
        root = tree.getroot()

        clips_by_track: Dict[str, List[EditedClip]] = {}

        # Find video tracks
        for track_idx, track in enumerate(root.findall('.//video/track')):
            track_name = f"V{track_idx + 1}"
            track_number = track_idx + 1

            clips_by_track[track_name] = []

            for clipitem in track.findall('clipitem'):
                # Get timing
                start_elem = clipitem.find('start')
                end_elem = clipitem.find('end')

                if start_elem is None or end_elem is None:
                    continue

                start_frame = int(start_elem.text or 0)
                end_frame = int(end_elem.text or 0)

                # Get enabled state
                is_enabled = True
                enabled_elem = clipitem.find('enabled')
                if enabled_elem is not None and enabled_elem.text:
                    is_enabled = enabled_elem.text.lower() in ('true', '1', 'yes')

                # Get filename
                filename = ""
                name_elem = clipitem.find('name')
                if name_elem is not None and name_elem.text:
                    filename = name_elem.text

                # Also try pathurl
                pathurl = clipitem.find('.//pathurl')
                if pathurl is not None and pathurl.text:
                    filename = os.path.basename(pathurl.text)

                clip = EditedClip(
                    filename=filename,
                    track_name=track_name,
                    track_number=track_number,
                    start_frame=start_frame,
                    end_frame=end_frame,
                    duration_frames=end_frame - start_frame,
                    is_enabled=is_enabled
                )

                clips_by_track[track_name].append(clip)

        return self._build_analysis(xml_path, clips_by_track)

    def _extract_filename(self, clip) -> str:
        """Extract filename from OTIO clip's media reference."""
        try:
            # Try DEFAULT_MEDIA reference
            if hasattr(clip, 'media_references') and clip.media_references:
                ref = clip.media_references.get('DEFAULT_MEDIA')
                if ref and hasattr(ref, 'target_url') and ref.target_url:
                    return os.path.basename(ref.target_url)

            # Try direct media_reference
            if hasattr(clip, 'media_reference'):
                ref = clip.media_reference
                if ref and hasattr(ref, 'target_url') and ref.target_url:
                    return os.path.basename(ref.target_url)

            # Fall back to clip name
            return clip.name or ""

        except Exception:
            return clip.name or ""

    def _build_analysis(
        self,
        edited_path: str,
        clips_by_track: Dict[str, List[EditedClip]]
    ) -> AnalysisResult:
        """
        Build analysis result by matching clips to segments.

        Args:
            edited_path: Path to the edited file
            clips_by_track: Clips organized by track name

        Returns:
            AnalysisResult with complete analysis
        """
        result = AnalysisResult(
            analyzed_at=datetime.now().isoformat(),
            edited_file=edited_path,
            segment_map_file=self.segment_map_path,
            frame_rate=self.frame_rate,
            total_segments=len(self.segments)
        )

        # Build lookup of all known clips (original + alternatives)
        original_clips: Dict[str, Tuple[str, str]] = {}  # filename -> (segment_id, source_type)
        for seg in self.segments:
            original_clips[seg.v1_clip] = (seg.id, "v1")
            for alt in seg.alternatives:
                original_clips[alt['file']] = (seg.id, "alternative")
            for sec in seg.secondary:
                original_clips[sec['file']] = (seg.id, "secondary")

        # Match each segment to clips by position
        for seg in self.segments:
            selection = SegmentSelection(
                segment_id=seg.id,
                segment_index=seg.index,
                voiceover_text=seg.voiceover_text[:50] if seg.voiceover_text else "",
                original_v1_clip=seg.v1_clip,
                original_confidence=seg.v1_confidence
            )

            # Find clips that overlap with this segment's frame range
            overlapping_clips: List[EditedClip] = []

            for track_name, clips in clips_by_track.items():
                for clip in clips:
                    # Check if clip overlaps with segment
                    overlap_start = max(clip.start_frame, seg.start_frame)
                    overlap_end = min(clip.end_frame, seg.end_frame)

                    if overlap_end > overlap_start:
                        # Calculate overlap ratio
                        overlap_frames = overlap_end - overlap_start
                        segment_frames = seg.end_frame - seg.start_frame
                        overlap_ratio = overlap_frames / segment_frames if segment_frames > 0 else 0

                        # Consider it a match if >50% overlap
                        if overlap_ratio > 0.5:
                            overlapping_clips.append(clip)

            if not overlapping_clips:
                selection.is_missing = True
                result.missing += 1
                result.selections.append(selection)
                continue

            # Filter to enabled clips
            enabled_clips = [c for c in overlapping_clips if c.is_enabled]

            if not enabled_clips:
                selection.is_disabled = True
                result.all_disabled += 1
                result.selections.append(selection)
                continue

            # Sort by track number (lowest = V1 takes priority for visual)
            # But for analysis, we want the HIGHEST enabled track (editor's choice)
            # DaVinci stacking: V2 renders on top of V1
            enabled_clips.sort(key=lambda c: c.track_number, reverse=True)

            # The highest enabled track is the selection
            selected = enabled_clips[0]
            selection.selected_clip = selected.filename
            selection.selected_track = selected.track_name
            selection.selected_track_number = selected.track_number

            # Track usage
            result.track_usage[selected.track_name] = result.track_usage.get(selected.track_name, 0) + 1

            # Determine if V1 was kept or replaced
            # Compare filenames (may be different path but same file)
            selected_base = self._normalize_filename(selected.filename)
            v1_base = self._normalize_filename(seg.v1_clip)

            if selected_base == v1_base:
                selection.was_v1_kept = True
                result.v1_kept += 1
            else:
                selection.was_replaced = True

                # Determine replacement source
                if selected.filename in original_clips:
                    _, source_type = original_clips[selected.filename]
                    selection.replacement_source = source_type

                    if source_type == "alternative":
                        result.replaced_with_alt += 1
                    elif source_type == "secondary":
                        result.replaced_with_secondary += 1
                else:
                    selection.replacement_source = "external"
                    result.replaced_with_external += 1

                # Record replacement
                result.replacements.append({
                    "segment": seg.id,
                    "original": seg.v1_clip,
                    "replaced_with": selected.filename,
                    "new_track": selected.track_name
                })

            result.selections.append(selection)

        # Calculate percentages
        if result.total_segments > 0:
            result.v1_kept_pct = (result.v1_kept / result.total_segments) * 100

        return result

    def _normalize_filename(self, filename: str) -> str:
        """Normalize filename for comparison."""
        if not filename:
            return ""

        # Get just the filename without path
        name = os.path.basename(filename)

        # Remove extension
        name = os.path.splitext(name)[0]

        # Lowercase for comparison
        return name.lower()


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def analyze_edited_timeline(
    edited_path: str,
    segment_map_path: str,
    output_path: Optional[str] = None,
    verbose: bool = True
) -> AnalysisResult:
    """
    Analyze an edited timeline against the segment map.

    Args:
        edited_path: Path to edited OTIO or XML file
        segment_map_path: Path to segment map JSON
        output_path: Optional path to save JSON report
        verbose: Whether to print summary

    Returns:
        AnalysisResult with analysis
    """
    analyzer = TimecodeAnalyzer(segment_map_path)

    ext = Path(edited_path).suffix.lower()
    if ext == '.otio':
        result = analyzer.analyze_otio(edited_path)
    elif ext in ('.xml', '.fcpxml'):
        result = analyzer.analyze_xml(edited_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    if verbose:
        print(result.summary())

    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result.to_dict(), f, indent=2)
        logger.info(f"Saved analysis report to {output_path}")

    return result


def extract_segment_map_from_otio(otio_path: str) -> Optional[str]:
    """
    Extract segment information from an original (non-edited) OTIO file.

    This is a fallback when segment map JSON is not available.
    Works only if the OTIO was generated by this tool (has segment metadata).

    Args:
        otio_path: Path to original OTIO file

    Returns:
        Path to generated segment map JSON, or None if extraction failed
    """
    try:
        import opentimelineio as otio
    except ImportError:
        logger.error("OpenTimelineIO not installed")
        return None

    timeline = otio.adapters.read_from_file(otio_path)

    # Get frame rate
    frame_rate = 30.0
    if timeline.global_start_time:
        frame_rate = timeline.global_start_time.rate

    segments = []
    position_frames = 0

    # Find V1 track
    v1_track = None
    for track in timeline.tracks:
        if track.kind != otio.schema.TrackKind.Video:
            continue
        if 'V1' in track.name or 'Primary' in track.name.lower():
            v1_track = track
            break

    if not v1_track:
        # Use first video track
        for track in timeline.tracks:
            if track.kind == otio.schema.TrackKind.Video:
                v1_track = track
                break

    if not v1_track:
        logger.error("No video track found in OTIO")
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

        # Extract metadata
        metadata = item.metadata or {}
        segment_id = metadata.get('segment_id', f"S{seg_idx:03d}")
        voiceover_text = metadata.get('voiceover_text', '')
        confidence = metadata.get('confidence', 0.0)

        # Get filename
        filename = ""
        if hasattr(item, 'media_reference') and item.media_reference:
            ref = item.media_reference
            if hasattr(ref, 'target_url') and ref.target_url:
                filename = os.path.basename(ref.target_url)

        segments.append({
            "id": segment_id,
            "start_frame": position_frames,
            "end_frame": end_frame,
            "start_tc": "",  # Would need frame-to-tc conversion
            "end_tc": "",
            "voiceover_text": voiceover_text,
            "duration_sec": duration_frames / frame_rate,
            "v1_clip": {
                "file": filename,
                "confidence": confidence
            }
        })

        position_frames = end_frame
        seg_idx += 1

    if not segments:
        logger.error("No segments found in OTIO")
        return None

    # Write segment map
    segment_map = {
        "generated_at": datetime.now().isoformat(),
        "source_srt": "",
        "frame_rate": frame_rate,
        "total_segments": len(segments),
        "segments": segments
    }

    output_path = Path(otio_path).with_suffix('').with_suffix('_extracted_segments.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(segment_map, f, indent=2)

    logger.info(f"Extracted segment map to {output_path}")
    return str(output_path)


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
  # Analyze with segment map
  python -m src.post_edit_analysis --edited export.otio --segments timeline_segments.json

  # Analyze with output report
  python -m src.post_edit_analysis --edited export.otio --segments timeline_segments.json --output report.json

  # Extract segment map from original OTIO (fallback)
  python -m src.post_edit_analysis --edited export.otio --original timeline_FULL.otio
        """
    )

    parser.add_argument(
        '--edited', '-e',
        required=True,
        help='Path to edited OTIO/XML export from DaVinci Resolve'
    )

    parser.add_argument(
        '--segments', '-s',
        help='Path to segment map JSON (timeline_segments.json)'
    )

    parser.add_argument(
        '--original', '-o',
        help='Path to original OTIO (extracts segment map if --segments not provided)'
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

    # Determine segment map path
    segment_map_path = args.segments

    if not segment_map_path and args.original:
        # Extract from original OTIO
        print(f"Extracting segment map from {args.original}...")
        segment_map_path = extract_segment_map_from_otio(args.original)
        if not segment_map_path:
            print("ERROR: Could not extract segment map from original OTIO")
            return 1

    if not segment_map_path:
        print("ERROR: Must provide either --segments or --original")
        return 1

    if not Path(segment_map_path).exists():
        print(f"ERROR: Segment map not found: {segment_map_path}")
        return 1

    if not Path(args.edited).exists():
        print(f"ERROR: Edited file not found: {args.edited}")
        return 1

    # Run analysis
    try:
        result = analyze_edited_timeline(
            edited_path=args.edited,
            segment_map_path=segment_map_path,
            output_path=args.output,
            verbose=not args.quiet
        )

        if args.output:
            print(f"\nReport saved to: {args.output}")

        return 0

    except Exception as e:
        print(f"ERROR: {e}")
        logger.exception("Analysis failed")
        return 1


if __name__ == '__main__':
    import sys
    sys.exit(main())
