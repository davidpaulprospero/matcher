"""
Caption timestamp normalization.

This module handles normalizing YouTube caption timestamps for consistency,
handling overlaps, gaps, and invalid timestamps.
"""

from __future__ import annotations

import logging
import re
from typing import List, Optional, TYPE_CHECKING

from .models import CaptionSegment, NormalizationConfig
from .exceptions import CaptionNormalizationError, CaptionParseWarning

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class CaptionNormalizer:
    """Normalizes YouTube caption timestamps for consistency.

    YouTube captions can have irregular timing:
    - Overlapping segments (two segments with overlapping time ranges)
    - Gaps between segments
    - Inconsistent timestamp formats (VTT, SRT, JSON3)
    - Invalid timestamps (negative, end < start)

    This class normalizes segments to a consistent format expected by
    the matching stage.

    Edge Case Behaviors (US-009 Sprint 6):
    -------------------------------------

    **Negative Start Times:**
        - start_time < 0 is clamped to 0
        - Example: start=-5, end=2 → start=0, end=2
        - If clamping creates duration < min_segment_duration, end is extended

    **Negative Duration (start > end):**
        - When start > end (inverted timestamps), end is extended
        - New end = start + min_segment_duration
        - Example: start=8, end=3 with min=0.5 → start=8, end=8.5
        - Original start time is preserved; only end is adjusted

    **Overlapping Segments (overlap_strategy):**
        - "truncate" (default): First segment's end truncated to second's start
        - "merge": Segments merged, text concatenated with space
          Example: (0-5 "First"), (3-8 "Second") → (0-8 "First Second")
        - "split": Split at midpoint of overlap region

    **Duplicate Segment Indices:**
        - All segments are re-indexed sequentially (0, 1, 2, ...)
        - Original indices are ignored
        - Example: indices [5, 5, 5] → [0, 1, 2]

    **Gap Handling (gap_strategy):**
        - "ignore" (default): Gaps left as-is
        - "extend": Small gaps (≤ max_gap_to_extend) filled by extending
          previous segment's end_time. Large gaps create placeholder segments.
        - "placeholder": Always insert empty placeholder segment in gaps
        - Placeholder segments have text="" and preserve gap timing
          Example: (0-5), (15-20) with max_gap=2 → (0-5), (5-15 placeholder), (15-20)

    **Empty/Whitespace Text:**
        - Segments with empty or whitespace-only text are filtered out
        - This is intentional filtering, not counted as parse errors

    **Partial Recovery (US-005):**
        - Malformed segments that throw exceptions are skipped
        - Normalization succeeds if > min_success_ratio segments parse
        - Default threshold: 50% must succeed
        - Filtered segments (empty text) don't count against threshold

    Usage:
        normalizer = CaptionNormalizer()
        normalized = normalizer.normalize(caption_result.segments, video_id="abc123")

        # With custom config:
        config = NormalizationConfig(overlap_strategy="merge", gap_strategy="extend")
        normalizer = CaptionNormalizer(config)
        normalized = normalizer.normalize(segments)

    Example:
        >>> segments = [
        ...     CaptionSegment(0, 0.0, 5.0, "First", "vid"),
        ...     CaptionSegment(1, 4.0, 8.0, "Second", "vid"),  # Overlaps with first
        ...     CaptionSegment(2, 10.0, 12.0, "Third", "vid"),  # Gap before this
        ... ]
        >>> normalizer = CaptionNormalizer()
        >>> normalized = normalizer.normalize(segments)
    """

    def __init__(self, config: Optional[NormalizationConfig] = None):
        """Initialize the normalizer.

        Args:
            config: Optional normalization configuration.
        """
        self.config = config or NormalizationConfig()

    def normalize(
        self,
        segments: List[CaptionSegment],
        video_id: str = "",
        min_success_ratio: float = 0.5
    ) -> tuple[List[CaptionSegment], int]:
        """Normalize a list of caption segments with partial recovery (US-005).

        Applies the following normalization steps:
        1. Validate and fix individual segment timestamps (with per-segment error handling)
        2. Sort segments by start time
        3. Handle overlapping segments (based on overlap_strategy)
        4. Handle gaps between segments (based on gap_strategy)
        5. Re-index segments sequentially

        Partial recovery (US-005): If a segment fails to parse due to unexpected
        errors, it is skipped rather than failing the entire normalization. The
        number of skipped segments is returned, and partial results are accepted
        if more than min_success_ratio (default 50%) of segments parse successfully.

        Args:
            segments: List of CaptionSegment to normalize.
            video_id: Video ID for source_file field in new segments.
            min_success_ratio: Minimum ratio of segments that must parse successfully
                for partial results to be accepted (default: 0.5 = 50%).

        Returns:
            Tuple of (normalized_segments, skipped_count). The skipped_count
            indicates how many segments were skipped due to parse errors.

        Raises:
            CaptionNormalizationError: If normalization fails catastrophically
                (less than min_success_ratio of segments parsed successfully).
        """
        if not segments:
            return ([], 0)

        original_count = len(segments)
        skipped_count = 0  # Segments skipped due to parse errors (exceptions)
        filtered_count = 0  # Segments intentionally filtered (invalid but not errors)
        logger.debug(f"Normalizing {original_count} caption segments for video {video_id}")

        # Step 1: Validate and fix individual timestamps with per-segment error handling
        validated_segments = []
        if self.config.validate_timestamps:
            for i, seg in enumerate(segments):
                try:
                    validated = self._validate_segment(seg, video_id)
                    if validated is not None:
                        validated_segments.append(validated)
                    else:
                        # Segment was intentionally filtered (e.g., empty text, invalid timing)
                        # This is expected behavior, not an error
                        filtered_count += 1
                except Exception as e:
                    # Unexpected error during validation - log and skip segment (US-005)
                    # This counts toward the skip threshold
                    skipped_count += 1
                    logger.warning(
                        f"Skipped malformed segment {i} for video {video_id}: {e}"
                    )
                    # Emit a CaptionParseWarning for tracking (non-fatal)
                    try:
                        warning = CaptionParseWarning(video_id, i, str(e))
                        logger.debug(f"CaptionParseWarning: {warning}")
                    except Exception:
                        pass  # Don't let warning emission cause failures
            segments = validated_segments
        else:
            segments = list(segments)

        # Check if too many segments had parse ERRORS (US-005)
        # Only count unexpected exceptions against the threshold, not intentional filters
        if skipped_count > 0 and original_count > 0:
            # Calculate what fraction of segments had actual errors
            error_ratio = skipped_count / original_count
            if error_ratio > (1.0 - min_success_ratio):
                logger.warning(
                    f"Too many parse errors: {skipped_count}/{original_count} segments "
                    f"({error_ratio:.1%}) had errors, threshold is {1.0 - min_success_ratio:.0%}"
                )
                raise CaptionNormalizationError(
                    f"Partial recovery failed: {error_ratio:.1%} of segments had parse errors "
                    f"(threshold: {1.0 - min_success_ratio:.0%})"
                )

        if not segments:
            logger.warning("All segments were invalid after validation")
            return ([], skipped_count)

        # Log skipped/filtered segments summary (US-005)
        if skipped_count > 0 or filtered_count > 0:
            logger.info(
                f"Partial recovery: {skipped_count} segments had errors, "
                f"{filtered_count} filtered, {len(segments)} usable for video {video_id}"
            )

        # Step 2: Sort by start time
        segments = sorted(segments, key=lambda s: s.start_time)

        # Step 3: Handle overlapping segments
        segments = self._handle_overlaps(segments, video_id)

        # Step 4: Handle gaps
        segments = self._handle_gaps(segments, video_id)

        # Step 5: Re-index
        segments = self._reindex(segments)

        logger.debug(f"Normalization complete: {len(segments)} segments, {skipped_count} skipped")
        return (segments, skipped_count)

    def _validate_segment(
        self,
        segment: CaptionSegment,
        video_id: str
    ) -> Optional[CaptionSegment]:
        """Validate and fix a single segment's timestamps.

        Validation rules:
        - start_time must be >= 0
        - end_time must be > start_time
        - Duration must be >= min_segment_duration

        Args:
            segment: Segment to validate.
            video_id: Video ID for logging.

        Returns:
            Fixed segment, or None if segment is invalid and unfixable.
        """
        start = segment.start_time
        end = segment.end_time

        # Fix negative start time
        if start < 0:
            logger.debug(f"Fixing negative start_time ({start}) in segment {segment.index}")
            start = 0.0

        # Fix negative end time
        if end < 0:
            logger.debug(f"Invalid negative end_time ({end}) in segment {segment.index}")
            return None

        # Fix end <= start
        if end <= start:
            # Try extending end by minimum duration
            new_end = start + self.config.min_segment_duration
            logger.debug(
                f"Fixing end <= start ({end} <= {start}) in segment {segment.index}, "
                f"extending to {new_end}"
            )
            end = new_end

        # Check minimum duration
        duration = end - start
        if duration < self.config.min_segment_duration:
            logger.debug(
                f"Segment {segment.index} duration ({duration}) below minimum "
                f"({self.config.min_segment_duration}), extending"
            )
            end = start + self.config.min_segment_duration

        # Skip empty text
        if not segment.text.strip():
            logger.debug(f"Skipping empty segment {segment.index}")
            return None

        return CaptionSegment(
            index=segment.index,
            start_time=start,
            end_time=end,
            text=segment.text,
            source_file=segment.source_file or video_id
        )

    def _handle_overlaps(
        self,
        segments: List[CaptionSegment],
        video_id: str
    ) -> List[CaptionSegment]:
        """Handle overlapping segments based on strategy.

        Args:
            segments: Sorted list of segments.
            video_id: Video ID for new segments.

        Returns:
            List of segments with overlaps resolved.
        """
        if len(segments) < 2:
            return segments

        strategy = self.config.overlap_strategy
        result = []

        i = 0
        while i < len(segments):
            current = segments[i]

            # Check for overlap with next segment
            if i + 1 < len(segments):
                next_seg = segments[i + 1]

                if current.end_time > next_seg.start_time:
                    # Overlap detected
                    logger.debug(
                        f"Overlap detected: segment {current.index} "
                        f"({current.start_time:.2f}-{current.end_time:.2f}) overlaps with "
                        f"segment {next_seg.index} ({next_seg.start_time:.2f}-{next_seg.end_time:.2f})"
                    )

                    if strategy == "merge":
                        # Merge overlapping segments
                        merged = self._merge_segments(current, next_seg, video_id)
                        # Replace next segment with merged for further processing
                        segments[i + 1] = merged
                        i += 1
                        continue

                    elif strategy == "split":
                        # Split at midpoint
                        midpoint = (current.end_time + next_seg.start_time) / 2
                        current = CaptionSegment(
                            index=current.index,
                            start_time=current.start_time,
                            end_time=midpoint,
                            text=current.text,
                            source_file=current.source_file or video_id
                        )
                        segments[i + 1] = CaptionSegment(
                            index=next_seg.index,
                            start_time=midpoint,
                            end_time=next_seg.end_time,
                            text=next_seg.text,
                            source_file=next_seg.source_file or video_id
                        )

                    elif strategy == "truncate":
                        # Truncate current's end to next's start
                        current = CaptionSegment(
                            index=current.index,
                            start_time=current.start_time,
                            end_time=next_seg.start_time,
                            text=current.text,
                            source_file=current.source_file or video_id
                        )

            result.append(current)
            i += 1

        return result

    def _merge_segments(
        self,
        seg1: CaptionSegment,
        seg2: CaptionSegment,
        video_id: str
    ) -> CaptionSegment:
        """Merge two overlapping segments into one.

        Args:
            seg1: First segment (earlier start time).
            seg2: Second segment (overlapping).
            video_id: Video ID for merged segment.

        Returns:
            Merged CaptionSegment.
        """
        return CaptionSegment(
            index=seg1.index,
            start_time=min(seg1.start_time, seg2.start_time),
            end_time=max(seg1.end_time, seg2.end_time),
            text=f"{seg1.text} {seg2.text}".strip(),
            source_file=seg1.source_file or seg2.source_file or video_id
        )

    def _handle_gaps(
        self,
        segments: List[CaptionSegment],
        video_id: str
    ) -> List[CaptionSegment]:
        """Handle gaps between segments based on strategy.

        Args:
            segments: List of segments with overlaps resolved.
            video_id: Video ID for new segments.

        Returns:
            List of segments with gaps handled.
        """
        if len(segments) < 2 or self.config.gap_strategy == "ignore":
            return segments

        strategy = self.config.gap_strategy
        result = []

        for i, segment in enumerate(segments):
            result.append(segment)

            # Check for gap before next segment
            if i + 1 < len(segments):
                next_seg = segments[i + 1]
                gap = next_seg.start_time - segment.end_time

                if gap > 0:
                    logger.debug(
                        f"Gap detected: {gap:.2f}s between segment {segment.index} "
                        f"and {next_seg.index}"
                    )

                    if strategy == "extend":
                        # Extend if gap is small enough
                        if gap <= self.config.max_gap_to_extend:
                            # Extend current segment's end
                            result[-1] = CaptionSegment(
                                index=segment.index,
                                start_time=segment.start_time,
                                end_time=next_seg.start_time,
                                text=segment.text,
                                source_file=segment.source_file or video_id
                            )
                        else:
                            # Insert placeholder for large gaps
                            placeholder = CaptionSegment(
                                index=-1,  # Will be re-indexed
                                start_time=segment.end_time,
                                end_time=next_seg.start_time,
                                text="",  # Empty placeholder
                                source_file=video_id
                            )
                            result.append(placeholder)

                    elif strategy == "placeholder":
                        # Always insert placeholder
                        placeholder = CaptionSegment(
                            index=-1,
                            start_time=segment.end_time,
                            end_time=next_seg.start_time,
                            text="",
                            source_file=video_id
                        )
                        result.append(placeholder)

        return result

    def _reindex(self, segments: List[CaptionSegment]) -> List[CaptionSegment]:
        """Re-index segments sequentially.

        Args:
            segments: List of segments to re-index.

        Returns:
            List of segments with sequential indexes starting from 0.
        """
        return [
            CaptionSegment(
                index=i,
                start_time=seg.start_time,
                end_time=seg.end_time,
                text=seg.text,
                source_file=seg.source_file
            )
            for i, seg in enumerate(segments)
        ]

    def validate_continuity(
        self,
        segments: List[CaptionSegment]
    ) -> List[str]:
        """Validate segment timing continuity.

        Checks for common issues:
        - Negative durations
        - end_time < start_time
        - Overlaps
        - Gaps

        Args:
            segments: List of segments to validate.

        Returns:
            List of warning messages (empty if all valid).
        """
        warnings = []

        for i, seg in enumerate(segments):
            # Check individual segment
            if seg.end_time < seg.start_time:
                warnings.append(
                    f"Segment {i}: end_time ({seg.end_time}) < start_time ({seg.start_time})"
                )

            duration = seg.end_time - seg.start_time
            if duration <= 0:
                warnings.append(f"Segment {i}: non-positive duration ({duration})")

            # Check relationship with next segment
            if i + 1 < len(segments):
                next_seg = segments[i + 1]

                if seg.end_time > next_seg.start_time:
                    overlap = seg.end_time - next_seg.start_time
                    warnings.append(
                        f"Segments {i}-{i+1}: overlap of {overlap:.2f}s"
                    )

        return warnings

    @staticmethod
    def convert_timestamp_to_seconds(timestamp: str) -> Optional[float]:
        """Convert a timestamp string to seconds.

        Supports formats:
        - HH:MM:SS,mmm (SRT)
        - HH:MM:SS.mmm (VTT)
        - MM:SS.mmm (VTT short)
        - Seconds as float string

        Args:
            timestamp: Timestamp string to convert.

        Returns:
            Time in seconds, or None if parsing fails.
        """
        if not timestamp:
            return None

        timestamp = timestamp.strip().replace(',', '.')

        # Try float directly (e.g., "1.5")
        try:
            return float(timestamp)
        except ValueError:
            pass

        # Pattern for HH:MM:SS.mmm or MM:SS.mmm
        match = re.match(r'^(?:(\d+):)?(\d+):(\d+)(?:\.(\d+))?$', timestamp)
        if match:
            hours = int(match.group(1)) if match.group(1) else 0
            minutes = int(match.group(2))
            seconds = int(match.group(3))
            millis = int(match.group(4).ljust(3, '0')[:3]) if match.group(4) else 0

            return hours * 3600 + minutes * 60 + seconds + millis / 1000.0

        logger.warning(f"Could not parse timestamp: {timestamp}")
        return None

    @staticmethod
    def seconds_to_vtt_timestamp(seconds: float) -> str:
        """Convert seconds to VTT timestamp format.

        Args:
            seconds: Time in seconds.

        Returns:
            VTT timestamp string (HH:MM:SS.mmm).
        """
        if seconds < 0:
            seconds = 0

        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds * 1000) % 1000)

        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"

    @staticmethod
    def seconds_to_srt_timestamp(seconds: float) -> str:
        """Convert seconds to SRT timestamp format.

        Args:
            seconds: Time in seconds.

        Returns:
            SRT timestamp string (HH:MM:SS,mmm).
        """
        vtt_ts = CaptionNormalizer.seconds_to_vtt_timestamp(seconds)
        return vtt_ts.replace('.', ',')
