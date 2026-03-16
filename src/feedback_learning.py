"""
Feedback Learning Module

Analyzes post-edit feedback to learn editor preferences and adjust confidence
weights for future matching. This enables the pipeline to improve match quality
over time based on actual editor decisions.

Usage:
    from src.feedback_learning import FeedbackLearner, FeedbackBatch, FeedbackEntry

    # Create feedback from post-edit analysis
    feedback = FeedbackBatch(entries=[
        FeedbackEntry(segment_id="S001", original_confidence=0.7, was_kept=True,
                      matched_keywords=["travel"], video_source="youtube"),
    ])

    # Learn and persist
    learner = FeedbackLearner(cache_dir=project_dir / ".cache" / "feedback")
    learner.load()  # Load previous learnings
    learner.learn(feedback)
    learner.save()

    # Apply to new matches
    adjusted_conf = learner.apply_adjustment(confidence, keywords, source)
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class FeedbackValidationError(ValueError):
    """Raised when feedback entry fails validation."""
    pass


@dataclass
class FeedbackEntry:
    """A single feedback entry from post-edit analysis."""
    segment_id: str
    original_confidence: float
    was_kept: bool
    matched_keywords: Optional[List[str]] = None
    video_source: Optional[str] = None
    track_used: Optional[str] = None  # Which track the clip came from (V1, V2, etc.)
    replacement_source: Optional[str] = None  # If replaced, what was used instead


@dataclass
class FeedbackBatch:
    """A batch of feedback entries from a single editing session."""
    entries: List[FeedbackEntry] = field(default_factory=list)
    project_name: str = ""
    created_at: str = ""

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()


@dataclass
class FeedbackAnalysis:
    """Results from analyzing feedback batch."""
    total_entries: int = 0
    kept_count: int = 0
    dropped_count: int = 0

    # Confidence statistics
    avg_kept_confidence: float = 0.0
    avg_dropped_confidence: float = 0.0
    confidence_adjustment: float = 0.0

    # Keyword analysis
    keyword_boosts: Dict[str, float] = field(default_factory=dict)
    keyword_penalties: Dict[str, float] = field(default_factory=dict)

    # Source preferences
    source_preferences: Dict[str, float] = field(default_factory=dict)

    # Inconsistency tracking
    inconsistent_entries: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "total_entries": self.total_entries,
            "kept_count": self.kept_count,
            "dropped_count": self.dropped_count,
            "avg_kept_confidence": round(self.avg_kept_confidence, 4),
            "avg_dropped_confidence": round(self.avg_dropped_confidence, 4),
            "confidence_adjustment": round(self.confidence_adjustment, 4),
            "keyword_boosts": {k: round(v, 4) for k, v in self.keyword_boosts.items()},
            "keyword_penalties": {k: round(v, 4) for k, v in self.keyword_penalties.items()},
            "source_preferences": {k: round(v, 4) for k, v in self.source_preferences.items()},
            "inconsistent_entries": self.inconsistent_entries,
        }


def validate_feedback_entry(entry: FeedbackEntry) -> None:
    """
    Validate a feedback entry.

    Args:
        entry: The feedback entry to validate

    Raises:
        FeedbackValidationError: If entry is invalid
    """
    # Validate segment_id
    if entry.segment_id is None or entry.segment_id == "":
        raise FeedbackValidationError("segment_id cannot be None or empty")

    # Validate confidence range
    if entry.original_confidence < 0.0:
        raise FeedbackValidationError(f"confidence must be >= 0.0, got {entry.original_confidence}")
    if entry.original_confidence > 1.0:
        raise FeedbackValidationError(f"confidence must be <= 1.0, got {entry.original_confidence}")


def analyze_feedback(feedback: FeedbackBatch) -> FeedbackAnalysis:
    """
    Analyze a batch of feedback entries to extract learning signals.

    Args:
        feedback: Batch of feedback entries from post-edit analysis

    Returns:
        FeedbackAnalysis with aggregated statistics and adjustments
    """
    analysis = FeedbackAnalysis()

    if not feedback.entries:
        return analysis

    # Track kept/dropped for each segment (for inconsistency detection)
    segment_decisions: Dict[str, Set[bool]] = defaultdict(set)

    # Track keyword occurrences and outcomes
    keyword_kept_count: Dict[str, int] = defaultdict(int)
    keyword_dropped_count: Dict[str, int] = defaultdict(int)

    # Track source outcomes
    source_kept: Dict[str, int] = defaultdict(int)
    source_total: Dict[str, int] = defaultdict(int)

    # Confidence lists
    kept_confidences: List[float] = []
    dropped_confidences: List[float] = []

    for entry in feedback.entries:
        analysis.total_entries += 1
        segment_decisions[entry.segment_id].add(entry.was_kept)

        if entry.was_kept:
            analysis.kept_count += 1
            kept_confidences.append(entry.original_confidence)
        else:
            analysis.dropped_count += 1
            dropped_confidences.append(entry.original_confidence)

        # Track keywords
        keywords = entry.matched_keywords or []
        for kw in keywords:
            if entry.was_kept:
                keyword_kept_count[kw] += 1
            else:
                keyword_dropped_count[kw] += 1

        # Track source
        source = entry.video_source or "unknown"
        source_total[source] += 1
        if entry.was_kept:
            source_kept[source] += 1

    # Detect inconsistencies (same segment with different decisions)
    for segment_id, decisions in segment_decisions.items():
        if len(decisions) > 1:  # Both True and False for same segment
            analysis.inconsistent_entries += 1

    # Calculate average confidences
    if kept_confidences:
        analysis.avg_kept_confidence = sum(kept_confidences) / len(kept_confidences)
    if dropped_confidences:
        analysis.avg_dropped_confidence = sum(dropped_confidences) / len(dropped_confidences)

    # Calculate confidence adjustment
    # Positive adjustment means we should boost similar matches
    # Negative means we should penalize
    if kept_confidences and dropped_confidences:
        # If kept clips had higher avg confidence than dropped, our confidence is calibrated
        # If kept clips had lower avg confidence, we're overconfident (need negative adjustment)
        analysis.confidence_adjustment = analysis.avg_kept_confidence - analysis.avg_dropped_confidence
    elif kept_confidences:
        # Only kept clips - small positive adjustment based on how confident they were
        analysis.confidence_adjustment = max(0, analysis.avg_kept_confidence - 0.5) * 0.1
    elif dropped_confidences:
        # Only dropped clips - negative adjustment, bigger for higher confidence drops
        analysis.confidence_adjustment = -analysis.avg_dropped_confidence * 0.1

    # Calculate keyword boosts/penalties
    all_keywords = set(keyword_kept_count.keys()) | set(keyword_dropped_count.keys())
    for kw in all_keywords:
        kept = keyword_kept_count.get(kw, 0)
        dropped = keyword_dropped_count.get(kw, 0)
        total = kept + dropped

        if total == 0:
            continue

        # Calculate preference ratio
        keep_ratio = kept / total

        if keep_ratio > 0.6:
            # Keyword associated with kept clips - boost it
            boost = (keep_ratio - 0.5) * 0.2  # Max boost of 0.1
            analysis.keyword_boosts[kw] = boost
        elif keep_ratio < 0.4:
            # Keyword associated with dropped clips - penalize it
            penalty = (0.5 - keep_ratio) * 0.2  # Max penalty of 0.1
            analysis.keyword_penalties[kw] = penalty

    # Calculate source preferences
    for source, total in source_total.items():
        if total > 0:
            kept = source_kept.get(source, 0)
            analysis.source_preferences[source] = kept / total

    return analysis


class FeedbackLearner:
    """
    Learns from editor feedback to improve future matching confidence.

    Persists learned adjustments to disk for use across pipeline runs.
    """

    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize the feedback learner.

        Args:
            cache_dir: Directory to store learned adjustments.
                      Defaults to ~/.matcher_feedback_cache/
        """
        if cache_dir is None:
            cache_dir = Path.home() / ".matcher_feedback_cache"

        self.cache_dir = Path(cache_dir)
        self.cache_file = self.cache_dir / "learned_adjustments.json"

        # Learned parameters
        self.keyword_boosts: Dict[str, float] = {}
        self.keyword_penalties: Dict[str, float] = {}
        self.source_preferences: Dict[str, float] = {}
        self.confidence_adjustment: float = 0.0
        self.total_feedback_entries: int = 0

    def load(self) -> bool:
        """
        Load previously learned adjustments from cache.

        Returns:
            True if loaded successfully, False if no cache or error
        """
        if not self.cache_file.exists():
            logger.debug(f"No feedback cache found at {self.cache_file}")
            return False

        try:
            with open(self.cache_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            self.keyword_boosts = data.get("keyword_boosts", {})
            self.keyword_penalties = data.get("keyword_penalties", {})
            self.source_preferences = data.get("source_preferences", {})
            self.confidence_adjustment = data.get("confidence_adjustment", 0.0)
            self.total_feedback_entries = data.get("total_feedback_entries", 0)

            logger.info(f"Loaded feedback learning from {self.cache_file} "
                       f"({self.total_feedback_entries} entries)")
            return True

        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to load feedback cache: {e}. Starting fresh.")
            self._reset()
            return False
        except Exception as e:
            logger.warning(f"Unexpected error loading feedback cache: {e}")
            self._reset()
            return False

    def _reset(self) -> None:
        """Reset learned parameters to defaults."""
        self.keyword_boosts = {}
        self.keyword_penalties = {}
        self.source_preferences = {}
        self.confidence_adjustment = 0.0
        self.total_feedback_entries = 0

    def save(self) -> bool:
        """
        Save learned adjustments to cache.

        Returns:
            True if saved successfully
        """
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

            data = {
                "keyword_boosts": self.keyword_boosts,
                "keyword_penalties": self.keyword_penalties,
                "source_preferences": self.source_preferences,
                "confidence_adjustment": self.confidence_adjustment,
                "total_feedback_entries": self.total_feedback_entries,
                "updated_at": datetime.now().isoformat(),
            }

            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)

            logger.info(f"Saved feedback learning to {self.cache_file}")
            return True

        except Exception as e:
            logger.error(f"Failed to save feedback cache: {e}")
            return False

    def learn(self, feedback: FeedbackBatch) -> FeedbackAnalysis:
        """
        Learn from a batch of feedback.

        Args:
            feedback: Batch of feedback entries

        Returns:
            Analysis of the feedback
        """
        analysis = analyze_feedback(feedback)

        # Update keyword boosts (using exponential moving average)
        alpha = 0.3  # Learning rate
        for kw, boost in analysis.keyword_boosts.items():
            current = self.keyword_boosts.get(kw, 0.0)
            self.keyword_boosts[kw] = current + alpha * (boost - current)

        # Update keyword penalties
        for kw, penalty in analysis.keyword_penalties.items():
            current = self.keyword_penalties.get(kw, 0.0)
            self.keyword_penalties[kw] = current + alpha * (penalty - current)

        # Update source preferences
        for source, pref in analysis.source_preferences.items():
            current = self.source_preferences.get(source, 0.5)  # Default to neutral
            self.source_preferences[source] = current + alpha * (pref - current)

        # Update confidence adjustment
        self.confidence_adjustment = (
            self.confidence_adjustment + alpha * (analysis.confidence_adjustment - self.confidence_adjustment)
        )

        # Track total entries
        self.total_feedback_entries += analysis.total_entries

        logger.info(f"Learned from {analysis.total_entries} feedback entries "
                   f"(kept: {analysis.kept_count}, dropped: {analysis.dropped_count})")

        return analysis

    def apply_adjustment(
        self,
        confidence: float,
        keywords: Optional[List[str]] = None,
        source: Optional[str] = None
    ) -> float:
        """
        Apply learned adjustments to a confidence score.

        Args:
            confidence: Original confidence score (0.0-1.0)
            keywords: List of matched keywords
            source: Video source (youtube, pexels, etc.)

        Returns:
            Adjusted confidence score (clamped to 0.0-1.0)
        """
        adjusted = confidence

        # Apply global confidence adjustment
        adjusted += self.confidence_adjustment * 0.5  # Dampen global adjustment

        # Apply keyword boosts/penalties
        keywords = keywords or []
        for kw in keywords:
            if kw in self.keyword_boosts:
                adjusted += self.keyword_boosts[kw]
            if kw in self.keyword_penalties:
                adjusted -= self.keyword_penalties[kw]

        # Apply source preference
        if source and source in self.source_preferences:
            # Adjust based on how much this source differs from neutral (0.5)
            source_pref = self.source_preferences[source]
            adjusted += (source_pref - 0.5) * 0.1

        # Clamp to valid range
        return max(0.0, min(1.0, adjusted))

    def get_summary(self) -> str:
        """Get human-readable summary of learned preferences."""
        lines = [
            "=" * 50,
            "FEEDBACK LEARNING SUMMARY",
            "=" * 50,
            f"Total feedback entries learned: {self.total_feedback_entries}",
            f"Global confidence adjustment: {self.confidence_adjustment:+.3f}",
            "",
        ]

        if self.keyword_boosts:
            lines.append("Top keyword boosts:")
            for kw, boost in sorted(self.keyword_boosts.items(), key=lambda x: -x[1])[:5]:
                lines.append(f"  {kw}: +{boost:.3f}")

        if self.keyword_penalties:
            lines.append("Top keyword penalties:")
            for kw, penalty in sorted(self.keyword_penalties.items(), key=lambda x: -x[1])[:5]:
                lines.append(f"  {kw}: -{penalty:.3f}")

        if self.source_preferences:
            lines.append("Source preferences:")
            for source, pref in sorted(self.source_preferences.items(), key=lambda x: -x[1]):
                status = "preferred" if pref > 0.6 else "avoided" if pref < 0.4 else "neutral"
                lines.append(f"  {source}: {pref:.1%} ({status})")

        lines.append("=" * 50)
        return "\n".join(lines)
