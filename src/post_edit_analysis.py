"""
Post-Edit Analysis Tool

Analyzes final XML exported from DaVinci Resolve to understand which clips
were selected from the original match recommendations. Uses segment IDs
embedded in clip names ([S001], [S002], etc.) for reliable tracing.

This tool helps:
1. Track editor preferences for future matching improvements
2. Feed successful selections back to global cache
3. Understand which types of matches get kept vs replaced
"""

import logging
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class ClipSelection:
    """Represents a clip that was selected in the final edit."""
    segment_id: str  # e.g., "S001"
    segment_index: int  # e.g., 1
    clip_name: str  # Full clip name from timeline
    source_file: Optional[str] = None  # Extracted source file name
    track: str = "V1"  # Which track it came from
    track_number: int = 1  # Numeric track number (V2 = 2) for priority comparison
    was_alternative: bool = False  # True if ALT clip was used
    alternative_index: Optional[int] = None  # Which ALT (1, 2, etc.)
    was_secondary: bool = False  # True if secondary source was used
    is_enabled: bool = True  # False if clip is disabled in timeline
    timeline_start: float = 0.0  # Timeline position in seconds
    timeline_end: float = 0.0  # End position in seconds
    duration: float = 0.0  # Duration in seconds


@dataclass
class SegmentSelection:
    """
    Represents all clips for a single segment, with priority logic applied.

    In DaVinci Resolve workflow:
    - Editor disables unwanted clips rather than deleting
    - Higher tracks (V2 > V1) take visual priority
    - Multiple clips can be active for layered compositions
    """
    segment_index: int
    segment_id: str

    # Primary selection: highest enabled track
    primary_clip: Optional[ClipSelection] = None

    # All active (enabled) clips for this segment (for layered compositions)
    active_clips: List[ClipSelection] = field(default_factory=list)

    # All disabled clips (kept for reference)
    disabled_clips: List[ClipSelection] = field(default_factory=list)

    # Analysis flags
    is_layered: bool = False  # True if multiple active clips
    all_disabled: bool = False  # True if all clips for this segment are disabled

    def get_winning_track(self) -> Optional[str]:
        """Get the track name of the primary (highest) active clip."""
        return self.primary_clip.track if self.primary_clip else None


@dataclass
class EditAnalysisResult:
    """Results from analyzing a final edit."""
    total_segments: int = 0
    clips_kept: int = 0  # Primary clips kept as-is (V1, enabled)
    clips_replaced_with_alt: int = 0  # Replaced with alternative (ALT on higher track)
    clips_replaced_with_secondary: int = 0  # Replaced with secondary source
    clips_replaced_with_external: int = 0  # Replaced with footage not in original
    clips_removed: int = 0  # Segments with no enabled clips
    clips_all_disabled: int = 0  # Segments where all clips were disabled

    # Detailed selection data
    selections: List[ClipSelection] = field(default_factory=list)  # Primary selections only
    segment_selections: List[SegmentSelection] = field(default_factory=list)  # Full segment data
    missing_segments: List[int] = field(default_factory=list)  # Segment IDs not found

    # Layered composition tracking
    segments_with_layers: int = 0  # Segments with multiple active clips
    layered_segments: List[int] = field(default_factory=list)  # Which segments are layered

    # Track usage statistics
    track_usage: Dict[str, int] = field(default_factory=dict)  # Count per track

    # Statistics
    avg_kept_confidence: float = 0.0
    avg_replaced_confidence: float = 0.0

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            "=== Post-Edit Analysis Summary ===",
            f"Total segments analyzed: {self.total_segments}",
            f"Primary clips kept (V1): {self.clips_kept} ({self.clips_kept/max(1,self.total_segments)*100:.1f}%)",
            f"Replaced with alternatives: {self.clips_replaced_with_alt}",
            f"Replaced with secondary: {self.clips_replaced_with_secondary}",
            f"Replaced with external: {self.clips_replaced_with_external}",
            f"All disabled (removed): {self.clips_all_disabled}",
            f"Missing segments: {self.clips_removed}",
        ]

        if self.segments_with_layers > 0:
            lines.append(f"Layered compositions: {self.segments_with_layers} segments")

        if self.track_usage:
            usage_str = ", ".join(f"{k}:{v}" for k, v in sorted(self.track_usage.items()))
            lines.append(f"Track usage: {usage_str}")

        if self.missing_segments:
            lines.append(f"Missing segment IDs: {self.missing_segments[:10]}{'...' if len(self.missing_segments) > 10 else ''}")

        return "\n".join(lines)


class PostEditAnalyzer:
    """Analyzes edited XML to understand clip selections."""

    # Regex to extract segment ID from clip name: [S001], [S123], etc.
    SEGMENT_ID_PATTERN = re.compile(r'\[S(\d{3})\]')

    # Regex to detect alternative clips: ALT1:, ALT2:, etc.
    ALT_PATTERN = re.compile(r'ALT(\d+):')

    # Regex to extract track number from track name: V1, V2, A1, etc.
    TRACK_NUMBER_PATTERN = re.compile(r'[VA](\d+)')

    # Regex to detect secondary clips (common labels)
    SECONDARY_PATTERNS = [
        re.compile(r'^(Different Source|Alternative File|Secondary):'),
        re.compile(r'^SEC\d*:'),
    ]

    def __init__(self, original_srt_path: Optional[str] = None):
        """
        Initialize analyzer.

        Args:
            original_srt_path: Path to original SRT file for cross-reference
        """
        self.original_srt_path = original_srt_path
        self.original_segments: Dict[int, str] = {}

        if original_srt_path and Path(original_srt_path).exists():
            self._load_original_srt(original_srt_path)

    def _load_original_srt(self, srt_path: str) -> None:
        """Load original SRT to get segment texts."""
        try:
            with open(srt_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Simple SRT parsing
            segments = content.strip().split('\n\n')
            for i, segment in enumerate(segments):
                lines = segment.strip().split('\n')
                if len(lines) >= 3:
                    # Lines: index, timestamp, text...
                    text = ' '.join(lines[2:])
                    self.original_segments[i] = text

            logger.info(f"Loaded {len(self.original_segments)} segments from original SRT")
        except Exception as e:
            logger.warning(f"Could not load original SRT: {e}")

    def analyze_xml(self, xml_path: str) -> EditAnalysisResult:
        """
        Analyze an XML file exported from DaVinci Resolve.

        Implements the following selection logic:
        1. Disabled clips are ignored (editor disables unwanted clips)
        2. For each segment, the highest enabled track wins (V2 > V1)
        3. Multiple active clips = layered composition (all tracked)

        Args:
            xml_path: Path to the exported XML file

        Returns:
            EditAnalysisResult with analysis details
        """
        result = EditAnalysisResult()

        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
        except ET.ParseError as e:
            logger.error(f"Failed to parse XML: {e}")
            return result

        # Find all clip items in the timeline with track info
        clips_by_segment: Dict[int, List[ClipSelection]] = {}

        # Get clips with track information
        clip_elements = self._find_clip_elements(root)

        for clip_elem, track_name, track_number in clip_elements:
            selection = self._parse_clip_element(clip_elem, track_name, track_number)
            if selection:
                seg_idx = selection.segment_index
                if seg_idx not in clips_by_segment:
                    clips_by_segment[seg_idx] = []
                clips_by_segment[seg_idx].append(selection)

        # Analyze which clips were selected using track priority
        if clips_by_segment:
            max_segment = max(clips_by_segment.keys())
            result.total_segments = max_segment + 1

            for seg_idx in range(result.total_segments):
                if seg_idx not in clips_by_segment:
                    result.missing_segments.append(seg_idx)
                    result.clips_removed += 1
                    continue

                clips = clips_by_segment[seg_idx]

                # Separate enabled and disabled clips
                enabled_clips = [c for c in clips if c.is_enabled]
                disabled_clips = [c for c in clips if not c.is_enabled]

                # Create segment selection
                segment_sel = SegmentSelection(
                    segment_index=seg_idx,
                    segment_id=f"S{seg_idx:03d}",
                    active_clips=enabled_clips,
                    disabled_clips=disabled_clips
                )

                # Check if all clips are disabled
                if not enabled_clips:
                    segment_sel.all_disabled = True
                    result.clips_all_disabled += 1
                    result.segment_selections.append(segment_sel)
                    continue

                # Sort enabled clips by track number (highest first)
                enabled_clips_sorted = sorted(enabled_clips, key=lambda c: c.track_number, reverse=True)

                # Primary selection = highest track number
                primary_clip = enabled_clips_sorted[0]
                segment_sel.primary_clip = primary_clip

                # Check for layered composition (multiple active clips)
                if len(enabled_clips) > 1:
                    segment_sel.is_layered = True
                    result.segments_with_layers += 1
                    result.layered_segments.append(seg_idx)

                result.segment_selections.append(segment_sel)
                result.selections.append(primary_clip)

                # Track usage statistics
                track = primary_clip.track
                result.track_usage[track] = result.track_usage.get(track, 0) + 1

                # Categorize the selection
                if primary_clip.was_alternative:
                    result.clips_replaced_with_alt += 1
                elif primary_clip.was_secondary:
                    result.clips_replaced_with_secondary += 1
                elif primary_clip.track_number == 1:
                    # V1 is the primary/original recommendation
                    result.clips_kept += 1
                else:
                    # Higher track but not marked as ALT - could be external or manual
                    result.clips_replaced_with_external += 1

        logger.info(f"Analyzed {len(clip_elements)} clips from {xml_path}")
        logger.info(f"Segments: {result.total_segments}, Kept: {result.clips_kept}, "
                   f"Alt: {result.clips_replaced_with_alt}, Layered: {result.segments_with_layers}")
        return result

    def _find_clip_elements(self, root: ET.Element) -> List[Tuple[ET.Element, str, int]]:
        """
        Find all clip/clipitem elements in the XML tree with track info.

        Returns:
            List of tuples: (clip_element, track_name, track_number)
        """
        clips_with_tracks = []

        # Find video tracks and their clips
        # FCPXML format: sequence/media/video/track
        for track_idx, track in enumerate(root.findall('.//video/track')):
            track_name = f"V{track_idx + 1}"
            track_number = track_idx + 1

            # Check for track name attribute
            track_name_attr = track.get('name')
            if track_name_attr:
                track_name = track_name_attr
                # Extract number from name if possible
                num_match = self.TRACK_NUMBER_PATTERN.search(track_name_attr)
                if num_match:
                    track_number = int(num_match.group(1))

            for clip in track.findall('clipitem'):
                clips_with_tracks.append((clip, track_name, track_number))

        # DaVinci FCPXML format: spine with clips
        for track_idx, spine in enumerate(root.findall('.//spine')):
            track_name = f"V{track_idx + 1}"
            track_number = track_idx + 1

            for clip in spine.findall('.//clip'):
                clips_with_tracks.append((clip, track_name, track_number))
            for clip in spine.findall('.//asset-clip'):
                clips_with_tracks.append((clip, track_name, track_number))

        # Fallback: find orphan clips and assign to V1
        if not clips_with_tracks:
            for clip in root.findall('.//clipitem'):
                if clip.find('start') is not None:
                    clips_with_tracks.append((clip, "V1", 1))
            for clip in root.findall('.//clip'):
                if clip.get('offset') is not None:
                    clips_with_tracks.append((clip, "V1", 1))

        return clips_with_tracks

    def _parse_clip_element(
        self,
        clip_elem: ET.Element,
        track_name: str = "V1",
        track_number: int = 1
    ) -> Optional[ClipSelection]:
        """
        Parse a clip element into a ClipSelection.

        Args:
            clip_elem: The XML clip element
            track_name: Name of the track (e.g., "V1", "V2 - Alternatives")
            track_number: Numeric track number for priority comparison
        """
        # Get clip name from various possible locations
        name = None

        # Try <name> element
        name_elem = clip_elem.find('name')
        if name_elem is not None and name_elem.text:
            name = name_elem.text

        # Try <n> element (DaVinci format)
        if not name:
            n_elem = clip_elem.find('n')
            if n_elem is not None and n_elem.text:
                name = n_elem.text

        # Try name attribute
        if not name:
            name = clip_elem.get('name')

        if not name:
            return None

        # Extract segment ID from name
        match = self.SEGMENT_ID_PATTERN.search(name)
        if not match:
            # No segment ID - this is external footage
            logger.debug(f"Clip without segment ID: {name}")
            return None

        segment_index = int(match.group(1))
        segment_id = f"S{segment_index:03d}"

        # Check if it's an alternative
        was_alt = False
        alt_index = None
        alt_match = self.ALT_PATTERN.search(name)
        if alt_match:
            was_alt = True
            alt_index = int(alt_match.group(1))

        # Check if it's a secondary source
        was_secondary = False
        for pattern in self.SECONDARY_PATTERNS:
            if pattern.search(name):
                was_secondary = True
                break

        # Check if clip is enabled/disabled
        # DaVinci/FCPXML uses <enabled> element or enabled attribute
        is_enabled = True  # Default to enabled

        # Method 1: <enabled> element (FCPXML)
        enabled_elem = clip_elem.find('enabled')
        if enabled_elem is not None and enabled_elem.text:
            is_enabled = enabled_elem.text.lower() in ('true', '1', 'yes')

        # Method 2: enabled attribute
        enabled_attr = clip_elem.get('enabled')
        if enabled_attr is not None:
            is_enabled = enabled_attr.lower() in ('true', '1', 'yes')

        # Method 3: DaVinci uses <disable> flag (inverted logic)
        disable_elem = clip_elem.find('disable')
        if disable_elem is not None:
            is_enabled = False

        # Method 4: Check for "disabled" in metadata
        metadata = clip_elem.find('.//metadata')
        if metadata is not None:
            for meta in metadata.findall('meta'):
                key = meta.get('key', '') or meta.find('key')
                if key and 'disable' in str(key).lower():
                    is_enabled = False

        # Extract timing info
        timeline_start = 0.0
        timeline_end = 0.0

        start_elem = clip_elem.find('start')
        end_elem = clip_elem.find('end')

        # Try to get frame rate for conversion
        rate_elem = clip_elem.find('.//rate/timebase')
        fps = 24.0  # Default
        if rate_elem is not None and rate_elem.text:
            try:
                fps = float(rate_elem.text)
            except ValueError:
                pass

        if start_elem is not None and start_elem.text:
            try:
                timeline_start = float(start_elem.text) / fps
            except ValueError:
                pass

        if end_elem is not None and end_elem.text:
            try:
                timeline_end = float(end_elem.text) / fps
            except ValueError:
                pass

        # Extract source file name from the clip name
        # Format: [S001] folder_filename [123.4s]
        source_file = None
        name_without_segment = self.SEGMENT_ID_PATTERN.sub('', name).strip()
        name_without_alt = self.ALT_PATTERN.sub('', name_without_segment).strip()
        # Remove trailing timestamp like [123.4s]
        name_cleaned = re.sub(r'\s*\[\d+\.?\d*s\]\s*$', '', name_without_alt).strip()
        if name_cleaned:
            source_file = name_cleaned

        return ClipSelection(
            segment_id=segment_id,
            segment_index=segment_index,
            clip_name=name,
            source_file=source_file,
            track=track_name,
            track_number=track_number,
            was_alternative=was_alt,
            alternative_index=alt_index,
            was_secondary=was_secondary,
            is_enabled=is_enabled,
            timeline_start=timeline_start,
            timeline_end=timeline_end,
            duration=timeline_end - timeline_start if timeline_end > timeline_start else 0.0
        )

    def analyze_otio(self, otio_path: str) -> EditAnalysisResult:
        """
        Analyze an OTIO file.

        Implements the same selection logic as analyze_xml:
        1. Disabled clips are ignored
        2. Highest enabled track wins (V2 > V1)
        3. Multiple active clips = layered composition

        Args:
            otio_path: Path to the OTIO file

        Returns:
            EditAnalysisResult with analysis details
        """
        try:
            import opentimelineio as otio_lib
        except ImportError:
            logger.error("OpenTimelineIO not installed, cannot analyze OTIO files")
            return EditAnalysisResult()

        result = EditAnalysisResult()

        try:
            timeline = otio_lib.adapters.read_from_file(otio_path)
        except Exception as e:
            logger.error(f"Failed to read OTIO file: {e}")
            return result

        clips_by_segment: Dict[int, List[ClipSelection]] = {}

        # Get video tracks and their index (for track number)
        video_tracks = [t for t in timeline.tracks if t.kind == otio_lib.schema.TrackKind.Video]

        # Iterate through all video tracks with index
        for track_idx, track in enumerate(video_tracks):
            track_name = track.name or f"V{track_idx + 1}"
            track_number = track_idx + 1

            # Extract track number from name if possible
            num_match = self.TRACK_NUMBER_PATTERN.search(track_name)
            if num_match:
                track_number = int(num_match.group(1))

            for item in track:
                if not isinstance(item, otio_lib.schema.Clip):
                    continue

                name = item.name
                if not name:
                    continue

                # Extract segment ID
                match = self.SEGMENT_ID_PATTERN.search(name)
                if not match:
                    continue

                segment_index = int(match.group(1))
                segment_id = f"S{segment_index:03d}"

                # Check if alternative
                was_alt = False
                alt_index = None
                alt_match = self.ALT_PATTERN.search(name)
                if alt_match:
                    was_alt = True
                    alt_index = int(alt_match.group(1))

                # Check enabled state from metadata
                is_enabled = True
                if hasattr(item, 'enabled'):
                    is_enabled = item.enabled
                elif 'enabled' in item.metadata:
                    is_enabled = item.metadata.get('enabled', True)
                elif 'disabled' in item.metadata:
                    is_enabled = not item.metadata.get('disabled', False)

                # Get timing
                timeline_start = 0.0
                if item.range_in_parent():
                    timeline_start = item.range_in_parent().start_time.to_seconds()

                duration = 0.0
                if item.duration():
                    duration = item.duration().to_seconds()

                selection = ClipSelection(
                    segment_id=segment_id,
                    segment_index=segment_index,
                    clip_name=name,
                    track=track_name,
                    track_number=track_number,
                    was_alternative=was_alt,
                    alternative_index=alt_index,
                    is_enabled=is_enabled,
                    timeline_start=timeline_start,
                    timeline_end=timeline_start + duration,
                    duration=duration
                )

                if segment_index not in clips_by_segment:
                    clips_by_segment[segment_index] = []
                clips_by_segment[segment_index].append(selection)

        # Analyze results using same priority logic as analyze_xml
        if clips_by_segment:
            max_segment = max(clips_by_segment.keys())
            result.total_segments = max_segment + 1

            for seg_idx in range(result.total_segments):
                if seg_idx not in clips_by_segment:
                    result.missing_segments.append(seg_idx)
                    result.clips_removed += 1
                    continue

                clips = clips_by_segment[seg_idx]

                # Separate enabled and disabled clips
                enabled_clips = [c for c in clips if c.is_enabled]
                disabled_clips = [c for c in clips if not c.is_enabled]

                # Create segment selection
                segment_sel = SegmentSelection(
                    segment_index=seg_idx,
                    segment_id=f"S{seg_idx:03d}",
                    active_clips=enabled_clips,
                    disabled_clips=disabled_clips
                )

                if not enabled_clips:
                    segment_sel.all_disabled = True
                    result.clips_all_disabled += 1
                    result.segment_selections.append(segment_sel)
                    continue

                # Sort by track number (highest first)
                enabled_clips_sorted = sorted(enabled_clips, key=lambda c: c.track_number, reverse=True)
                primary_clip = enabled_clips_sorted[0]
                segment_sel.primary_clip = primary_clip

                if len(enabled_clips) > 1:
                    segment_sel.is_layered = True
                    result.segments_with_layers += 1
                    result.layered_segments.append(seg_idx)

                result.segment_selections.append(segment_sel)
                result.selections.append(primary_clip)

                # Track usage
                track = primary_clip.track
                result.track_usage[track] = result.track_usage.get(track, 0) + 1

                # Categorize
                if primary_clip.was_alternative:
                    result.clips_replaced_with_alt += 1
                elif primary_clip.track_number == 1:
                    result.clips_kept += 1
                else:
                    result.clips_replaced_with_external += 1

        return result

    def compare_before_after(
        self,
        original_matches_path: str,
        final_edit_path: str
    ) -> Dict[str, any]:
        """
        Compare original match results with final edit to understand preferences.

        Args:
            original_matches_path: Path to original OTIO/XML with match results
            final_edit_path: Path to final exported XML after editing

        Returns:
            Dictionary with comparison statistics
        """
        # Determine file type and analyze
        original_ext = Path(original_matches_path).suffix.lower()
        final_ext = Path(final_edit_path).suffix.lower()

        if original_ext == '.otio':
            original_result = self.analyze_otio(original_matches_path)
        else:
            original_result = self.analyze_xml(original_matches_path)

        if final_ext == '.otio':
            final_result = self.analyze_otio(final_edit_path)
        else:
            final_result = self.analyze_xml(final_edit_path)

        # Build comparison
        comparison = {
            'original_segments': original_result.total_segments,
            'final_segments': final_result.total_segments,
            'clips_kept': 0,
            'clips_changed': 0,
            'clips_removed': 0,
            'changes': []
        }

        # Map original selections by segment
        original_by_seg = {s.segment_index: s for s in original_result.selections}
        final_by_seg = {s.segment_index: s for s in final_result.selections}

        all_segments = set(original_by_seg.keys()) | set(final_by_seg.keys())

        for seg_idx in sorted(all_segments):
            orig = original_by_seg.get(seg_idx)
            final = final_by_seg.get(seg_idx)

            if orig and final:
                # Both exist - check if same clip
                if orig.source_file == final.source_file:
                    comparison['clips_kept'] += 1
                else:
                    comparison['clips_changed'] += 1
                    comparison['changes'].append({
                        'segment': seg_idx,
                        'original': orig.clip_name,
                        'final': final.clip_name,
                        'used_alternative': final.was_alternative
                    })
            elif orig and not final:
                comparison['clips_removed'] += 1
            elif final and not orig:
                # New clip added (unusual)
                comparison['clips_changed'] += 1

        return comparison

    def generate_feedback_for_cache(
        self,
        analysis: EditAnalysisResult,
        original_srt_path: Optional[str] = None
    ) -> List[Dict[str, any]]:
        """
        Generate feedback data for global cache learning.

        Args:
            analysis: EditAnalysisResult from analyzing final edit
            original_srt_path: Path to original SRT for segment text

        Returns:
            List of feedback entries for cache learning
        """
        feedback = []

        # Load SRT if provided
        segment_texts = {}
        if original_srt_path:
            self._load_original_srt(original_srt_path)
            segment_texts = self.original_segments

        for selection in analysis.selections:
            entry = {
                'segment_index': selection.segment_index,
                'segment_text': segment_texts.get(selection.segment_index, ''),
                'selected_clip': selection.source_file,
                'was_original_choice': not selection.was_alternative,
                'was_alternative': selection.was_alternative,
                'alternative_index': selection.alternative_index,
                'duration': selection.duration,
                # Score boost/penalty for future matching
                'preference_signal': 1.0 if not selection.was_alternative else -0.5
            }
            feedback.append(entry)

        return feedback


def analyze_final_edit(
    xml_path: str,
    original_srt_path: Optional[str] = None,
    verbose: bool = True
) -> EditAnalysisResult:
    """
    Convenience function to analyze a final edit XML.

    Args:
        xml_path: Path to exported XML from DaVinci Resolve
        original_srt_path: Optional path to original SRT file
        verbose: Whether to print summary

    Returns:
        EditAnalysisResult with analysis
    """
    analyzer = PostEditAnalyzer(original_srt_path)

    ext = Path(xml_path).suffix.lower()
    if ext == '.otio':
        result = analyzer.analyze_otio(xml_path)
    else:
        result = analyzer.analyze_xml(xml_path)

    if verbose:
        print(result.summary())

    return result


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Analyze final edit to understand clip selections'
    )
    parser.add_argument('xml_path', help='Path to exported XML/OTIO file')
    parser.add_argument('--srt', help='Path to original SRT file')
    parser.add_argument('--compare', help='Path to original OTIO for comparison')
    parser.add_argument('--output', help='Output JSON file for feedback data')

    args = parser.parse_args()

    analyzer = PostEditAnalyzer(args.srt)

    if args.compare:
        comparison = analyzer.compare_before_after(args.compare, args.xml_path)
        print("\n=== Before/After Comparison ===")
        print(f"Clips kept: {comparison['clips_kept']}")
        print(f"Clips changed: {comparison['clips_changed']}")
        print(f"Clips removed: {comparison['clips_removed']}")
        if comparison['changes']:
            print("\nChanges made:")
            for change in comparison['changes'][:10]:
                print(f"  S{change['segment']:03d}: {change['original'][:30]} -> {change['final'][:30]}")
    else:
        result = analyze_final_edit(args.xml_path, args.srt)

        if args.output:
            import json
            feedback = analyzer.generate_feedback_for_cache(result, args.srt)
            with open(args.output, 'w') as f:
                json.dump(feedback, f, indent=2)
            print(f"\nFeedback data saved to {args.output}")
