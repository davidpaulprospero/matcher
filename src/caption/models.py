"""
Caption fetcher data models.

Contains core dataclasses for caption segments, results, quality metrics,
and related data structures.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from .enums import CaptionStatus, StreamState

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    pass


@dataclass
class CaptionSegment:
    """A single caption segment with timing information.

    Compatible with TranscriptSegment for downstream matching.
    """
    index: int
    start_time: float
    end_time: float
    text: str
    source_file: str = ""  # Video ID or path

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            'index': self.index,
            'start': self.start_time,
            'end': self.end_time,
            'text': self.text,
            'source_file': self.source_file,
        }


@dataclass
class SegmentQualityMetrics:
    """Quality metrics for caption segments (US-006 Sprint 8).

    Provides sophisticated quality scoring based on:
    - density_score: Segments per minute (normalized 0-1)
    - timing_precision: Percentage with exact millisecond timestamps
    - text_completeness: Average chars per segment vs expected (50-200)

    The combined quality_score weights these: 0.4*density + 0.3*precision + 0.3*completeness.
    Higher scores indicate better quality captions for matching purposes.
    """
    density_score: float
    timing_precision: float
    text_completeness: float
    quality_score: float
    segment_count: int
    total_duration: float

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            'density_score': self.density_score,
            'timing_precision': self.timing_precision,
            'text_completeness': self.text_completeness,
            'quality_score': self.quality_score,
            'segment_count': self.segment_count,
            'total_duration': self.total_duration,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'SegmentQualityMetrics':
        """Create from dictionary."""
        return cls(
            density_score=data.get('density_score', 0.0),
            timing_precision=data.get('timing_precision', 0.0),
            text_completeness=data.get('text_completeness', 0.0),
            quality_score=data.get('quality_score', 0.0),
            segment_count=data.get('segment_count', 0),
            total_duration=data.get('total_duration', 0.0),
        )


@dataclass
class TimingValidationResult:
    """Result of caption timing validation against video duration (US-007).

    Attributes:
        is_valid: True if timing is within acceptable bounds.
        caption_end_time: End time of the last caption segment.
        video_duration: Video duration used for comparison.
        exceeds_duration: True if captions extend beyond video duration + tolerance.
        below_coverage: True if caption coverage is below minimum threshold.
        message: Human-readable description of validation result.
        timing_epsilon_applied: Epsilon tolerance in milliseconds that was applied.
        exceeds_ratio: How much captions exceed video duration (0.0 = at/below).
        coverage_ratio: Caption coverage as ratio of video duration (1.0 = 100%).
    """
    is_valid: bool
    caption_end_time: float
    video_duration: float
    exceeds_duration: bool = False
    below_coverage: bool = False
    message: str = ""
    timing_epsilon_applied: float = 0.0
    exceeds_ratio: float = 0.0
    coverage_ratio: float = 1.0

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            'is_valid': self.is_valid,
            'caption_end_time': self.caption_end_time,
            'video_duration': self.video_duration,
            'exceeds_duration': self.exceeds_duration,
            'below_coverage': self.below_coverage,
            'message': self.message,
            'timing_epsilon_applied': self.timing_epsilon_applied,
            'exceeds_ratio': self.exceeds_ratio,
            'coverage_ratio': self.coverage_ratio,
        }


@dataclass
class StreamStateResult:
    """Result of stream state classification (US-007 Sprint 8).

    Contains the classified state plus metadata for logging and decision-making.
    """
    state: StreamState
    video_id: str
    is_live: bool = False
    was_live: bool = False
    live_status: Optional[str] = None
    scheduled_start: Optional[str] = None
    duration: Optional[float] = None

    def __str__(self) -> str:
        """Format for logging with scheduled time if applicable."""
        base = f"{self.video_id}: {self.state.name}"
        if self.scheduled_start and self.state in (StreamState.UPCOMING, StreamState.PREMIERE):
            base += f" (scheduled {self.scheduled_start})"
        if self.live_status:
            base += f" [live_status={self.live_status}]"
        return base


@dataclass
class ParseResult:
    """Result of parsing caption content with error recovery (US-001 Sprint 7).

    This dataclass holds both successfully parsed segments and information about
    segments that were skipped due to parse errors.
    """
    segments: List['CaptionSegment'] = field(default_factory=list)
    skipped_segments: List[tuple] = field(default_factory=list)  # (index, reason)
    total_attempted: int = 0

    @property
    def success_rate(self) -> float:
        """Calculate success rate of parsing."""
        if self.total_attempted == 0:
            return 1.0
        return len(self.segments) / self.total_attempted

    @property
    def has_skipped(self) -> bool:
        """Check if any segments were skipped."""
        return len(self.skipped_segments) > 0


@dataclass
class ErrorPatternResult:
    """Result of error pattern detection (US-007 Sprint 7).

    When batch fetching encounters repeated errors of the same type,
    this dataclass captures the detected pattern for logging and decision-making.
    """
    detected: bool = False
    error_signature: str = ""
    affected_video_ids: List[str] = field(default_factory=list)
    sample_size: int = 0
    ratio: float = 0.0
    likely_cause: str = ""

    def __str__(self) -> str:
        """Format as log-friendly string."""
        if not self.detected:
            return "No error pattern detected"
        count = len(self.affected_video_ids)
        pct = self.ratio * 100
        return (
            f"Pattern detected: {self.error_signature} "
            f"({count}/{self.sample_size} videos, {pct:.1f}%) - {self.likely_cause}"
        )


@dataclass
class AvailableLanguage:
    """Represents an available caption language for a video."""
    code: str  # ISO 639-1 code (e.g., 'en', 'es', 'fr')
    name: str  # Human-readable name (e.g., 'English', 'Spanish')
    is_auto_generated: bool  # True if auto-generated captions


@dataclass
class NormalizationConfig:
    """Configuration for caption timestamp normalization.

    Attributes:
        overlap_strategy: How to handle overlapping segments.
            - 'merge': Merge overlapping segments into one.
            - 'split': Split at the midpoint of overlap.
            - 'truncate': Truncate earlier segment's end to later's start.
        gap_strategy: How to handle gaps between segments.
            - 'extend': Extend previous segment's end to next segment's start.
            - 'placeholder': Insert empty placeholder segments.
            - 'ignore': Leave gaps as-is.
        max_gap_to_extend: Maximum gap size (seconds) to extend.
        min_segment_duration: Minimum valid segment duration (seconds).
        validate_timestamps: Whether to validate and fix timestamps.
    """
    overlap_strategy: str = "truncate"
    gap_strategy: str = "ignore"
    max_gap_to_extend: float = 1.0
    min_segment_duration: float = 0.1
    validate_timestamps: bool = True


@dataclass
class CaptionConfigValidationResult:
    """Result of full caption configuration validation (US-005 Sprint 7).

    Attributes:
        is_valid: True if all validation checks passed.
        errors: List of error messages (validation failures).
        warnings: List of warning messages (non-fatal issues).
        checks_performed: Dict mapping check name -> status (passed/failed/warning).
    """
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    checks_performed: Dict[str, str] = field(default_factory=dict)

    def __str__(self) -> str:
        """Human-readable summary."""
        status = "VALID" if self.is_valid else "INVALID"
        lines = [f"Caption Config Validation: {status}"]
        if self.errors:
            lines.append("Errors:")
            for e in self.errors:
                lines.append(f"  - {e}")
        if self.warnings:
            lines.append("Warnings:")
            for w in self.warnings:
                lines.append(f"  - {w}")
        return "\n".join(lines)


@dataclass
class TestFetchResult:
    """Result of caption test fetch operation (US-005 Sprint 7).

    Attributes:
        video_id: YouTube video ID tested.
        success: True if fetch succeeded.
        format_used: Caption format that succeeded (e.g., 'json3', 'vtt').
        elapsed_seconds: Time taken for fetch.
        error: Error message if fetch failed.
        segment_count: Number of segments fetched (0 if failed).
    """
    video_id: str
    success: bool
    format_used: str = ""
    elapsed_seconds: float = 0.0
    error: str = ""
    segment_count: int = 0


@dataclass
class TestFetchSummary:
    """Summary of multiple test fetch operations (US-005 Sprint 7).

    Attributes:
        total: Total number of test fetches attempted.
        successes: Number of successful fetches.
        failures: Number of failed fetches.
        results: List of individual TestFetchResult objects.
        avg_time: Average fetch time in seconds.
        dominant_format: Most common successful format.
    """
    total: int = 0
    successes: int = 0
    failures: int = 0
    results: List[TestFetchResult] = field(default_factory=list)
    avg_time: float = 0.0
    dominant_format: str = ""

    def __str__(self) -> str:
        """Human-readable summary for CLI output."""
        if self.total == 0:
            return "Test fetch: No videos tested"

        success_rate = f"{self.successes}/{self.total}"
        avg_time_str = f"{self.avg_time:.1f}s" if self.avg_time > 0 else "N/A"
        format_str = self.dominant_format or "N/A"

        return f"Test fetch: {success_rate} success, avg {avg_time_str}, {format_str} format"


def _token_overlap(text_a: str, text_b: str) -> float:
    """Calculate token-level overlap ratio between two strings.

    Returns the ratio of shared tokens to total unique tokens (Jaccard similarity).
    """
    tokens_a = set(text_a.lower().split())
    tokens_b = set(text_b.lower().split())
    if not tokens_a and not tokens_b:
        return 1.0
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b
    return len(intersection) / len(union)


def _temporal_overlap_ratio(seg_a: 'CaptionSegment', seg_b: 'CaptionSegment') -> float:
    """Calculate temporal overlap ratio between two segments.

    Returns the ratio of overlap duration to the shorter segment's duration.
    """
    overlap_start = max(seg_a.start_time, seg_b.start_time)
    overlap_end = min(seg_a.end_time, seg_b.end_time)
    overlap_duration = max(0.0, overlap_end - overlap_start)

    dur_a = max(0.0, seg_a.end_time - seg_a.start_time)
    dur_b = max(0.0, seg_b.end_time - seg_b.start_time)
    shorter_duration = min(dur_a, dur_b)

    if shorter_duration <= 0:
        return 0.0
    return overlap_duration / shorter_duration


def _text_quality_score(text: str) -> float:
    """Score text quality: longer text with fewer repeated chars is better."""
    if not text:
        return 0.0
    length = len(text)
    unique_chars = len(set(text))
    # Ratio of unique chars penalizes repeated content
    uniqueness = unique_chars / length if length > 0 else 0.0
    return length * uniqueness


def deduplicate_caption_segments(
    segments: List['CaptionSegment'],
    temporal_threshold: float = 0.8,
    text_threshold: float = 0.7,
) -> List['CaptionSegment']:
    """Remove overlapping near-duplicate caption segments.

    Segments with >80% temporal overlap AND >70% text similarity are merged,
    keeping the one with higher text quality (longer text, fewer repeated chars).

    Args:
        segments: List of CaptionSegment to deduplicate.
        temporal_threshold: Minimum temporal overlap ratio to consider duplicate (default 0.8).
        text_threshold: Minimum text similarity (token overlap) to consider duplicate (default 0.7).

    Returns:
        Deduplicated list of CaptionSegment with updated indices.
    """
    if len(segments) <= 1:
        return segments

    # Sort by start_time for efficient pairwise comparison
    sorted_segs = sorted(segments, key=lambda s: s.start_time)
    keep = [True] * len(sorted_segs)

    for i in range(len(sorted_segs)):
        if not keep[i]:
            continue
        for j in range(i + 1, len(sorted_segs)):
            if not keep[j]:
                continue
            # Early exit: if next segment starts well after current ends, no overlap possible
            if sorted_segs[j].start_time >= sorted_segs[i].end_time + 1.0:
                break

            temporal = _temporal_overlap_ratio(sorted_segs[i], sorted_segs[j])
            if temporal < temporal_threshold:
                continue

            text_sim = _token_overlap(sorted_segs[i].text, sorted_segs[j].text)
            if text_sim < text_threshold:
                continue

            # Both thresholds met — mark the lower-quality one for removal
            quality_i = _text_quality_score(sorted_segs[i].text)
            quality_j = _text_quality_score(sorted_segs[j].text)

            if quality_j > quality_i:
                # j is better, remove i
                logger.debug(
                    "Dedup: removing segment %d (%.1fs-%.1fs) in favor of %d (%.1fs-%.1fs) "
                    "[temporal=%.2f, text=%.2f]",
                    sorted_segs[i].index, sorted_segs[i].start_time, sorted_segs[i].end_time,
                    sorted_segs[j].index, sorted_segs[j].start_time, sorted_segs[j].end_time,
                    temporal, text_sim,
                )
                keep[i] = False
                break  # i is removed, no need to compare further
            else:
                # i is better or equal, remove j
                logger.debug(
                    "Dedup: removing segment %d (%.1fs-%.1fs) in favor of %d (%.1fs-%.1fs) "
                    "[temporal=%.2f, text=%.2f]",
                    sorted_segs[j].index, sorted_segs[j].start_time, sorted_segs[j].end_time,
                    sorted_segs[i].index, sorted_segs[i].start_time, sorted_segs[i].end_time,
                    temporal, text_sim,
                )
                keep[j] = False

    result = [seg for seg, k in zip(sorted_segs, keep) if k]

    removed_count = len(segments) - len(result)
    if removed_count > 0:
        logger.debug("Dedup: removed %d duplicate segments from %d total", removed_count, len(segments))
        # Re-index
        for idx, seg in enumerate(result):
            seg.index = idx

    return result


@dataclass
class CoverageAnalysis:
    """Result of caption coverage analysis (US-73-010).

    Identifies timing gaps and coverage holes in captions to help
    downstream matching understand uncaptioned periods.
    """
    total_video_duration: float
    total_captioned_duration: float
    coverage_ratio: float
    gap_count: int
    largest_gap_seconds: float

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            'total_video_duration': self.total_video_duration,
            'total_captioned_duration': self.total_captioned_duration,
            'coverage_ratio': self.coverage_ratio,
            'gap_count': self.gap_count,
            'largest_gap_seconds': self.largest_gap_seconds,
        }


def analyze_caption_coverage(
    segments: List['CaptionSegment'],
    video_duration: Optional[float],
) -> Optional['CoverageAnalysis']:
    """Compute caption coverage analysis including gap detection (US-73-010).

    Analyzes gaps between consecutive caption segments to identify uncaptioned
    periods (music, silence, non-speech audio). Gaps are periods >2 seconds
    between consecutive segment end_time and next start_time.

    Args:
        segments: List of CaptionSegment sorted by start_time.
        video_duration: Total video duration in seconds. Required for analysis.

    Returns:
        CoverageAnalysis with coverage metrics, or None if video_duration unavailable.
    """
    if not video_duration or video_duration <= 0:
        return None

    if not segments:
        return CoverageAnalysis(
            total_video_duration=video_duration,
            total_captioned_duration=0.0,
            coverage_ratio=0.0,
            gap_count=1,  # Entire video is one gap
            largest_gap_seconds=video_duration,
        )

    sorted_segs = sorted(segments, key=lambda s: s.start_time)

    # Sum actual captioned duration
    total_captioned = sum(
        max(0.0, seg.end_time - seg.start_time)
        for seg in sorted_segs
    )

    # Detect gaps >2 seconds between consecutive segments
    gap_threshold = 2.0
    gap_count = 0
    largest_gap = 0.0

    # Gap before first segment
    if sorted_segs[0].start_time > gap_threshold:
        gap_count += 1
        largest_gap = max(largest_gap, sorted_segs[0].start_time)

    # Gaps between consecutive segments
    for i in range(len(sorted_segs) - 1):
        gap = sorted_segs[i + 1].start_time - sorted_segs[i].end_time
        if gap > gap_threshold:
            gap_count += 1
            largest_gap = max(largest_gap, gap)

    # Gap after last segment
    trailing_gap = video_duration - sorted_segs[-1].end_time
    if trailing_gap > gap_threshold:
        gap_count += 1
        largest_gap = max(largest_gap, trailing_gap)

    coverage_ratio = min(1.0, total_captioned / video_duration)

    return CoverageAnalysis(
        total_video_duration=video_duration,
        total_captioned_duration=total_captioned,
        coverage_ratio=coverage_ratio,
        gap_count=gap_count,
        largest_gap_seconds=largest_gap,
    )


@dataclass
class CaptionResult:
    """Result of a caption fetch operation.

    Attributes:
        video_id: YouTube video ID.
        segments: List of caption segments with timing.
        language: ISO 639-1 language code (e.g., 'en').
        is_auto_generated: True if auto-generated captions.
        format_source: Caption format ('vtt', 'srv3', 'json3', etc.).
        video_duration: Optional video duration for coverage calculation (US-004).
        skipped_segments: List of (index, reason) tuples for skipped segments (US-001 Sprint 7).
        partial_recovery: True when segments were skipped but result is still usable (US-001 Sprint 7).
        timing_validated: Result of timing validation, or None if not validated (US-007).
        status: Structured status of the fetch operation (US-63-006).
            Values: 'success', 'no_captions', 'error', 'cached_unavailable'
        no_captions_available: True when video has no captions (not an error, triggers transcription fallback) (US-62-007).
            Deprecated: Use status == CaptionStatus.NO_CAPTIONS instead.
        fetch_error: Error message when fetch failed due to error (distinct from no_captions_available) (US-62-007).
    """
    video_id: str
    segments: List[CaptionSegment] = field(default_factory=list)
    language: str = ""  # ISO 639-1 code (e.g., 'en')
    is_auto_generated: bool = False
    format_source: str = ""  # 'vtt', 'srv3', 'json3', etc.
    video_duration: Optional[float] = None  # US-004: For coverage calculation
    skipped_segments: List[tuple] = field(default_factory=list)  # US-001: (index, reason) tuples
    partial_recovery: bool = False  # US-001: True when segments skipped but result usable
    timing_validated: Optional[TimingValidationResult] = None  # US-007: Timing validation result
    status: CaptionStatus = CaptionStatus.SUCCESS  # US-63-006: Structured status
    no_captions_available: bool = False  # US-62-007: True when video has no captions (not error)
    fetch_error: Optional[str] = None  # US-62-007: Error message when fetch failed
    # US-70-002: Video metadata for context-enriched matching
    video_description: str = ""  # Full video description text
    video_chapters: List[dict] = field(default_factory=list)  # Parsed chapter markers [{title, start_time, end_time}]
    video_tags: List[str] = field(default_factory=list)  # Video tags/keywords
    # US-73-012: Language confidence and fallback tracking
    language_confidence: float = 1.0  # 0.0-1.0: manual=1.0, auto target=0.8, auto translated=0.5
    fallback_language: str = ""  # Language actually used when different from requested

    def __post_init__(self):
        """Ensure list fields are never None (dict-vs-object safety, Rule 2/6).
        Also deduplicate overlapping caption segments after parsing.
        """
        if self.video_chapters is None:
            self.video_chapters = []
        if self.video_tags is None:
            self.video_tags = []
        if self.video_description is None:
            self.video_description = ""
        # US-73-007: Deduplicate overlapping caption segments
        if self.segments and len(self.segments) > 1:
            self.segments = deduplicate_caption_segments(self.segments)

    @property
    def coverage_analysis(self) -> Optional['CoverageAnalysis']:
        """Lazily compute and cache caption coverage analysis (US-73-010).

        Returns CoverageAnalysis with gap detection, or None if video_duration unknown.
        Emits WARNING when coverage_ratio < 0.7.
        """
        if not hasattr(self, '_coverage_analysis_cache'):
            analysis = analyze_caption_coverage(self.segments, self.video_duration)
            if analysis and analysis.coverage_ratio < 0.7:
                logger.warning(
                    "Low caption coverage for %s: %.1f%% covered, %d gaps, largest gap %.1fs",
                    self.video_id,
                    analysis.coverage_ratio * 100,
                    analysis.gap_count,
                    analysis.largest_gap_seconds,
                )
            object.__setattr__(self, '_coverage_analysis_cache', analysis)
        return self._coverage_analysis_cache

    @property
    def skipped_segments_count(self) -> int:
        """Backwards-compatible property for number of skipped segments."""
        return len(self.skipped_segments)

    @property
    def text(self) -> str:
        """Get full caption text concatenated."""
        return " ".join(seg.text for seg in self.segments)

    @property
    def duration(self) -> float:
        """Get total duration covered by captions."""
        if not self.segments:
            return 0.0
        return self.segments[-1].end_time - self.segments[0].start_time

    @property
    def caption_quality(self) -> str:
        """Determine caption quality based on source and completeness.

        Quality levels:
        - 'high': Human-uploaded captions (is_auto_generated=False) with good completeness
        - 'medium': Auto-generated captions with reasonable completeness
        - 'low': Missing, sparse, or fallback captions

        Quality is assessed based on:
        1. is_auto_generated flag (human > auto)
        2. Segment count (completeness indicator)
        3. Average segment duration (too long = sparse captions)

        Returns:
            'high', 'medium', or 'low'
        """
        # Import at runtime to avoid circular import
        from .quality import determine_caption_quality
        return determine_caption_quality(
            is_auto_generated=self.is_auto_generated,
            segment_count=len(self.segments),
            total_duration=self.duration
        )

    def calculate_coverage(self, video_duration: Optional[float] = None) -> float:
        """Calculate what percentage of video duration is covered by captions (US-004).

        Coverage is calculated by summing actual caption segment durations
        (not just start-to-end span) and dividing by video duration.

        Args:
            video_duration: Video duration in seconds. If not provided, uses
                self.video_duration if set, otherwise returns 0.0.

        Returns:
            Coverage ratio from 0.0 to 1.0. Returns 0.0 if video_duration is
            unknown or zero.

        Example:
            >>> result = CaptionResult(video_id="abc", segments=[...])
            >>> coverage = result.calculate_coverage(video_duration=300.0)
            >>> print(f"{coverage:.1%}")  # "85.3%"
        """
        duration = video_duration or self.video_duration
        if not duration or duration <= 0:
            return 0.0

        if not self.segments:
            return 0.0

        # Sum actual segment durations (not just start-to-end span)
        # This handles gaps between segments correctly
        total_caption_duration = sum(
            max(0.0, seg.end_time - seg.start_time)
            for seg in self.segments
        )

        # Clamp to 1.0 in case captions overlap or extend past video
        return min(1.0, total_caption_duration / duration)

    @property
    def coverage_ratio(self) -> float:
        """Get coverage ratio using stored video_duration (US-004).

        Returns:
            Coverage ratio from 0.0 to 1.0, or 0.0 if video_duration not set.
        """
        return self.calculate_coverage()

    @property
    def timing_penalty_factor(self) -> float:
        """Calculate timing penalty factor based on validation results (US-008 Sprint 7).

        The penalty is applied multiplicatively to match confidence to account for
        caption timing issues that may affect match quality.

        Formula:
            penalty = 1.0 - (exceeds_ratio * 0.3) - ((1 - coverage_ratio) * 0.2)

        Example calculations:
            - Perfect timing (100% coverage, no exceeds): 1.0 (no penalty)
            - 50% coverage, 20% exceeds: 1.0 - (0.2 * 0.3) - (0.5 * 0.2) = 0.84 (~16% penalty)
            - 80% coverage, no exceeds: 1.0 - 0 - (0.2 * 0.2) = 0.96 (~4% penalty)
            - 100% coverage, 10% exceeds: 1.0 - (0.1 * 0.3) - 0 = 0.97 (~3% penalty)

        Returns:
            Float between 0.0 and 1.0. Returns 1.0 (no penalty) if timing not validated
            or video duration unknown.
        """
        if self.timing_validated is None:
            return 1.0

        # Get ratios from timing validation result
        exceeds_ratio = getattr(self.timing_validated, 'exceeds_ratio', 0.0)
        coverage_ratio = getattr(self.timing_validated, 'coverage_ratio', 1.0)

        # Apply penalty formula: 1.0 - (exceeds_ratio * 0.3) - ((1 - coverage_ratio) * 0.2)
        # Exceeds penalty: penalize up to 30% of confidence for captions extending past video
        # Coverage penalty: penalize up to 20% of confidence for low caption coverage
        exceeds_penalty = exceeds_ratio * 0.3
        coverage_penalty = (1.0 - coverage_ratio) * 0.2

        penalty_factor = 1.0 - exceeds_penalty - coverage_penalty

        # Clamp to valid range [0.0, 1.0]
        return max(0.0, min(1.0, penalty_factor))

    def validate_timing(
        self,
        video_duration: Optional[float] = None,
        max_exceed_ratio: float = 1.1,
        min_coverage_ratio: float = 0.5,
        timing_epsilon_ms: float = 100.0,
    ) -> TimingValidationResult:
        """Validate caption timestamps against video duration (US-007).

        Checks two conditions:
        1. Caption end time should not exceed video duration by more than 10% (default)
        2. Caption coverage should not be below 50% of video duration (default)

        Args:
            video_duration: Video duration in seconds. If not provided, uses
                self.video_duration if set.
            max_exceed_ratio: Maximum allowed ratio of caption_end/video_duration.
                Default 1.1 means captions can extend up to 10% beyond video.
            min_coverage_ratio: Minimum required coverage ratio.
                Default 0.5 means captions must cover at least 50% of video.
            timing_epsilon_ms: Tolerance in milliseconds for floating-point precision
                at video duration boundary.

        Returns:
            TimingValidationResult with validation details. Also stores
            result in self.timing_validated.
        """
        duration = video_duration or self.video_duration

        # Cannot validate without video duration
        if not duration or duration <= 0:
            result = TimingValidationResult(
                is_valid=True,  # Can't fail validation without duration
                caption_end_time=0.0,
                video_duration=0.0,
                message="Cannot validate timing: video duration unknown",
                timing_epsilon_applied=timing_epsilon_ms,
            )
            self.timing_validated = result
            return result

        # Get caption end time from last segment
        caption_end_time = self.segments[-1].end_time if self.segments else 0.0

        # Convert epsilon from milliseconds to seconds
        epsilon_seconds = timing_epsilon_ms / 1000.0

        # Apply epsilon tolerance: if caption ends within epsilon of video duration,
        # treat it as ending exactly at duration for the exceeds check
        effective_caption_end = caption_end_time
        if abs(caption_end_time - duration) <= epsilon_seconds:
            # Caption is "close enough" to video duration - snap to duration
            effective_caption_end = duration

        # Check if captions exceed video duration (using effective end time)
        exceeds_duration = effective_caption_end > (duration * max_exceed_ratio)

        # Check coverage (using last segment end time, not summed duration)
        caption_coverage = caption_end_time / duration if duration > 0 else 0.0
        below_coverage = caption_coverage < min_coverage_ratio

        # Determine overall validity
        is_valid = not exceeds_duration and not below_coverage

        # Build message
        messages = []
        if exceeds_duration:
            exceed_pct = (caption_end_time / duration - 1.0) * 100
            messages.append(
                f"Caption end ({caption_end_time:.1f}s) exceeds video duration "
                f"({duration:.1f}s) by {exceed_pct:.1f}%"
            )
        if below_coverage:
            coverage_pct = caption_coverage * 100
            messages.append(
                f"Caption coverage ({coverage_pct:.1f}%) is below minimum "
                f"({min_coverage_ratio * 100:.0f}%)"
            )
        if is_valid:
            messages.append(f"Timing valid: captions end at {caption_end_time:.1f}s, "
                          f"video is {duration:.1f}s")

        # US-008: Calculate ratios for timing penalty
        exceeds_ratio = max(0.0, (caption_end_time / duration) - 1.0) if duration > 0 else 0.0
        coverage_ratio_val = min(1.0, caption_coverage)

        result = TimingValidationResult(
            is_valid=is_valid,
            caption_end_time=caption_end_time,
            video_duration=duration,
            exceeds_duration=exceeds_duration,
            below_coverage=below_coverage,
            message="; ".join(messages),
            timing_epsilon_applied=timing_epsilon_ms,
            exceeds_ratio=exceeds_ratio,
            coverage_ratio=coverage_ratio_val,
        )

        self.timing_validated = result
        return result

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            'video_id': self.video_id,
            'segments': [seg.to_dict() for seg in self.segments],
            'language': self.language,
            'is_auto_generated': self.is_auto_generated,
            'format_source': self.format_source,
            'caption_quality': self.caption_quality,
            'video_duration': self.video_duration,  # US-004
            'coverage_ratio': self.coverage_ratio,  # US-004
            'skipped_segments_count': self.skipped_segments_count,  # US-005
            'timing_validated': self.timing_validated.to_dict() if self.timing_validated else None,  # US-007
            'video_description': self.video_description,  # US-70-002
            'video_chapters': self.video_chapters,  # US-70-002
            'video_tags': self.video_tags,  # US-70-002
            'language_confidence': self.language_confidence,  # US-73-012
            'fallback_language': self.fallback_language,  # US-73-012
        }
