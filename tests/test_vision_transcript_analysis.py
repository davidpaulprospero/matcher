"""
Tests for vision module TranscriptAnalyzer — US-008 (Sprint 15).

Covers:
- TranscriptAnalyzer.analyze_video_transcript() with empty scenes
- Sparse scene identification by word count
- needs_vision=False when all scenes have sufficient transcript
- SceneAnalysis dataclass serialization
- VideoVisionDecision.transcript_coverage calculation (0.0, 1.0, proportional)
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock
from dataclasses import asdict

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.vision import (
    TranscriptAnalyzer,
    VideoVisionDecision,
    SceneAnalysis,
)


# ============================================================================
# Helpers
# ============================================================================

def _make_config(min_words=5, coverage_threshold=0.3, max_scenes=50):
    """Build a mock config with vision settings."""
    config = Mock()
    config.vision = Mock()
    config.vision.min_words_per_scene = min_words
    config.vision.coverage_threshold = coverage_threshold
    config.vision.max_scenes_per_video = max_scenes
    return config


def _scene(start, end):
    """Shorthand scene dict."""
    return {"start_time": start, "end_time": end}


def _seg(start, end, text):
    """Shorthand transcript segment dict."""
    return {"start_time": start, "end_time": end, "text": text}


# ============================================================================
# AC-1: analyze_video_transcript() with empty scenes list
# ============================================================================

class TestEmptyScenes:
    """AC-1: Empty scenes → needs_vision=True, reason mentions 'no scenes'."""

    @pytest.mark.fast
    def test_empty_scenes_needs_vision_true(self):
        """Empty scenes list → needs_vision=True."""
        analyzer = TranscriptAnalyzer(_make_config())
        decision = analyzer.analyze_video_transcript("/v.mp4", [], [])
        assert decision.needs_vision is True

    @pytest.mark.fast
    def test_empty_scenes_reason_mentions_no_scenes(self):
        """Reason should reference 'no scenes' (case-insensitive)."""
        analyzer = TranscriptAnalyzer(_make_config())
        decision = analyzer.analyze_video_transcript("/v.mp4", [], [])
        assert "no scenes" in decision.reason.lower()

    @pytest.mark.fast
    def test_empty_scenes_returns_video_vision_decision(self):
        """Return type is VideoVisionDecision."""
        analyzer = TranscriptAnalyzer(_make_config())
        decision = analyzer.analyze_video_transcript("/v.mp4", [], [])
        assert isinstance(decision, VideoVisionDecision)

    @pytest.mark.fast
    def test_empty_scenes_total_scenes_zero(self):
        """total_scenes should be 0."""
        analyzer = TranscriptAnalyzer(_make_config())
        decision = analyzer.analyze_video_transcript("/v.mp4", [], [])
        assert decision.total_scenes == 0

    @pytest.mark.fast
    def test_empty_scenes_sparse_list_empty(self):
        """sparse_scenes should be empty (no scenes to mark sparse)."""
        analyzer = TranscriptAnalyzer(_make_config())
        decision = analyzer.analyze_video_transcript("/v.mp4", [], [])
        assert decision.sparse_scenes == []

    @pytest.mark.fast
    def test_empty_scenes_coverage_zero(self):
        """transcript_coverage should be 0.0."""
        analyzer = TranscriptAnalyzer(_make_config())
        decision = analyzer.analyze_video_transcript("/v.mp4", [], [])
        assert decision.transcript_coverage == 0.0

    @pytest.mark.fast
    def test_empty_scenes_with_transcript_segments(self):
        """Empty scenes but non-empty transcript still yields needs_vision=True."""
        analyzer = TranscriptAnalyzer(_make_config())
        decision = analyzer.analyze_video_transcript(
            "/v.mp4", [], [_seg(0, 5, "some words here")]
        )
        assert decision.needs_vision is True
        assert decision.total_scenes == 0


# ============================================================================
# AC-2: Sparse scene identification
# ============================================================================

class TestSparseSceneIdentification:
    """AC-2: Scenes with word_count < min_words_per_scene flagged in sparse_scenes."""

    @pytest.mark.fast
    def test_single_sparse_scene_flagged(self):
        """Scene with 2 words (< min_words=5) is in sparse_scenes."""
        analyzer = TranscriptAnalyzer(_make_config(min_words=5))
        scenes = [_scene(0, 10)]
        transcript = [_seg(0, 10, "only two")]
        decision = analyzer.analyze_video_transcript("/v.mp4", scenes, transcript)
        assert 0 in decision.sparse_scenes

    @pytest.mark.fast
    def test_all_sparse_when_no_transcript(self):
        """All scenes sparse when transcript is empty."""
        analyzer = TranscriptAnalyzer(_make_config(min_words=5))
        scenes = [_scene(0, 5), _scene(5, 10), _scene(10, 15)]
        decision = analyzer.analyze_video_transcript("/v.mp4", scenes, [])
        assert decision.sparse_scenes == [0, 1, 2]

    @pytest.mark.fast
    def test_non_sparse_scene_excluded(self):
        """Scene with enough words NOT in sparse_scenes."""
        analyzer = TranscriptAnalyzer(_make_config(min_words=5))
        scenes = [_scene(0, 10)]
        transcript = [_seg(0, 10, "one two three four five six")]  # 6 words ≥ 5
        decision = analyzer.analyze_video_transcript("/v.mp4", scenes, transcript)
        assert 0 not in decision.sparse_scenes

    @pytest.mark.fast
    def test_mixed_sparse_and_sufficient(self):
        """Mix of sparse and sufficient scenes, only sparse flagged."""
        analyzer = TranscriptAnalyzer(_make_config(min_words=5))
        scenes = [_scene(0, 5), _scene(5, 10), _scene(10, 15)]
        transcript = [
            _seg(0, 5, "a b c d e f"),  # 6 words → sufficient (scene 0)
            _seg(5, 10, "hi"),            # 1 word → sparse (scene 1)
            _seg(10, 15, "one two three four five"),  # 5 words → sufficient (scene 2)
        ]
        decision = analyzer.analyze_video_transcript("/v.mp4", scenes, transcript)
        assert 1 in decision.sparse_scenes
        assert 0 not in decision.sparse_scenes
        assert 2 not in decision.sparse_scenes

    @pytest.mark.fast
    def test_sparse_with_srt_segment_objects(self):
        """Sparse detection works with SRTSegment-like objects, not just dicts."""
        analyzer = TranscriptAnalyzer(_make_config(min_words=5))
        scenes = [_scene(0, 10)]
        seg = Mock()
        seg.start_time = 0.0
        seg.end_time = 10.0
        seg.text = "short"  # 1 word
        decision = analyzer.analyze_video_transcript("/v.mp4", scenes, [seg])
        assert 0 in decision.sparse_scenes

    @pytest.mark.fast
    def test_exact_threshold_not_sparse(self):
        """Exactly min_words_per_scene words → NOT sparse."""
        analyzer = TranscriptAnalyzer(_make_config(min_words=5))
        scenes = [_scene(0, 10)]
        transcript = [_seg(0, 10, "one two three four five")]  # exactly 5
        decision = analyzer.analyze_video_transcript("/v.mp4", scenes, transcript)
        assert 0 not in decision.sparse_scenes

    @pytest.mark.fast
    def test_below_threshold_sparse(self):
        """min_words - 1 words → sparse."""
        analyzer = TranscriptAnalyzer(_make_config(min_words=5))
        scenes = [_scene(0, 10)]
        transcript = [_seg(0, 10, "one two three four")]  # 4 words < 5
        decision = analyzer.analyze_video_transcript("/v.mp4", scenes, transcript)
        assert 0 in decision.sparse_scenes


# ============================================================================
# AC-3: needs_vision=False when all scenes have sufficient transcript
# ============================================================================

class TestSufficientTranscript:
    """AC-3: All scenes sufficient → needs_vision=False, sparse_scenes empty."""

    @pytest.mark.fast
    def test_all_scenes_sufficient_needs_vision_false(self):
        """All scenes with ≥ min_words → needs_vision=False."""
        analyzer = TranscriptAnalyzer(_make_config(min_words=5, coverage_threshold=0.3))
        scenes = [_scene(0, 5), _scene(5, 10)]
        transcript = [
            _seg(0, 5, "one two three four five six"),
            _seg(5, 10, "seven eight nine ten eleven twelve"),
        ]
        decision = analyzer.analyze_video_transcript("/v.mp4", scenes, transcript)
        assert decision.needs_vision is False

    @pytest.mark.fast
    def test_all_scenes_sufficient_sparse_empty(self):
        """All scenes sufficient → sparse_scenes is empty list."""
        analyzer = TranscriptAnalyzer(_make_config(min_words=5, coverage_threshold=0.3))
        scenes = [_scene(0, 5), _scene(5, 10)]
        transcript = [
            _seg(0, 5, "a b c d e f"),
            _seg(5, 10, "g h i j k l"),
        ]
        decision = analyzer.analyze_video_transcript("/v.mp4", scenes, transcript)
        assert decision.sparse_scenes == []

    @pytest.mark.fast
    def test_coverage_threshold_met(self):
        """Coverage passes threshold → checked correctly."""
        analyzer = TranscriptAnalyzer(_make_config(min_words=5, coverage_threshold=0.5))
        scenes = [_scene(0, 5), _scene(5, 10)]
        transcript = [
            _seg(0, 5, "a b c d e f"),
            _seg(5, 10, "g h i j k l"),
        ]
        decision = analyzer.analyze_video_transcript("/v.mp4", scenes, transcript)
        assert decision.transcript_coverage >= 0.5
        assert decision.needs_vision is False

    @pytest.mark.fast
    def test_single_scene_sufficient(self):
        """Single scene with enough words → no vision needed."""
        analyzer = TranscriptAnalyzer(_make_config(min_words=3, coverage_threshold=0.3))
        scenes = [_scene(0, 10)]
        transcript = [_seg(0, 10, "plenty of words here")]
        decision = analyzer.analyze_video_transcript("/v.mp4", scenes, transcript)
        assert decision.needs_vision is False
        assert decision.sparse_scenes == []

    @pytest.mark.fast
    def test_reason_mentions_good_coverage(self):
        """Reason text reflects good coverage."""
        analyzer = TranscriptAnalyzer(_make_config(min_words=3, coverage_threshold=0.3))
        scenes = [_scene(0, 10)]
        transcript = [_seg(0, 10, "one two three four")]
        decision = analyzer.analyze_video_transcript("/v.mp4", scenes, transcript)
        assert "good" in decision.reason.lower() or "coverage" in decision.reason.lower()


# ============================================================================
# AC-4: SceneAnalysis dataclass serialization
# ============================================================================

class TestSceneAnalysisSerialization:
    """AC-4: SceneAnalysis stores and serializes correct fields."""

    @pytest.mark.fast
    def test_required_fields_stored(self):
        """All required fields are accessible."""
        sa = SceneAnalysis(
            scene_index=3,
            start_time=10.0,
            end_time=15.0,
            transcript_text="hello world",
            transcript_word_count=2,
            needs_vision=True,
            reason="Low text",
        )
        assert sa.scene_index == 3
        assert sa.start_time == 10.0
        assert sa.end_time == 15.0
        assert sa.transcript_text == "hello world"
        assert sa.transcript_word_count == 2
        assert sa.needs_vision is True
        assert sa.reason == "Low text"

    @pytest.mark.fast
    def test_optional_vision_description_default_none(self):
        """vision_description defaults to None."""
        sa = SceneAnalysis(
            scene_index=0, start_time=0, end_time=5,
            transcript_text="", transcript_word_count=0,
            needs_vision=True, reason="test",
        )
        assert sa.vision_description is None

    @pytest.mark.fast
    def test_optional_combined_description_default_none(self):
        """combined_description defaults to None."""
        sa = SceneAnalysis(
            scene_index=0, start_time=0, end_time=5,
            transcript_text="", transcript_word_count=0,
            needs_vision=True, reason="test",
        )
        assert sa.combined_description is None

    @pytest.mark.fast
    def test_vision_description_stored(self):
        """Explicit vision_description is stored."""
        sa = SceneAnalysis(
            scene_index=1, start_time=0, end_time=5,
            transcript_text="words", transcript_word_count=1,
            needs_vision=True, reason="sparse",
            vision_description="A mountain view",
        )
        assert sa.vision_description == "A mountain view"

    @pytest.mark.fast
    def test_asdict_round_trip(self):
        """asdict() produces a dict containing all fields."""
        sa = SceneAnalysis(
            scene_index=2,
            start_time=5.0,
            end_time=10.0,
            transcript_text="some text",
            transcript_word_count=2,
            needs_vision=False,
            reason="sufficient",
            vision_description="trees and sky",
            combined_description="some text [Visual: trees and sky]",
        )
        d = asdict(sa)
        assert d["scene_index"] == 2
        assert d["start_time"] == 5.0
        assert d["end_time"] == 10.0
        assert d["transcript_text"] == "some text"
        assert d["transcript_word_count"] == 2
        assert d["needs_vision"] is False
        assert d["reason"] == "sufficient"
        assert d["vision_description"] == "trees and sky"
        assert d["combined_description"] == "some text [Visual: trees and sky]"

    @pytest.mark.fast
    def test_asdict_none_optionals(self):
        """asdict() preserves None for optional fields."""
        sa = SceneAnalysis(
            scene_index=0, start_time=0, end_time=5,
            transcript_text="", transcript_word_count=0,
            needs_vision=True, reason="empty",
        )
        d = asdict(sa)
        assert d["vision_description"] is None
        assert d["combined_description"] is None

    @pytest.mark.fast
    def test_asdict_is_json_serializable(self):
        """asdict() output can be serialized to JSON without error."""
        import json
        sa = SceneAnalysis(
            scene_index=0, start_time=1.5, end_time=3.5,
            transcript_text="word", transcript_word_count=1,
            needs_vision=True, reason="test",
            vision_description="desc",
            combined_description="word [Visual: desc]",
        )
        serialized = json.dumps(asdict(sa))
        assert isinstance(serialized, str)
        loaded = json.loads(serialized)
        assert loaded["scene_index"] == 0
        assert loaded["vision_description"] == "desc"


# ============================================================================
# AC-5: VideoVisionDecision.transcript_coverage calculation
# ============================================================================

class TestTranscriptCoverage:
    """AC-5: Coverage is 0.0 for no-transcript, 1.0 for full, proportional for partial."""

    @pytest.mark.fast
    def test_coverage_zero_no_transcript(self):
        """No transcript segments → coverage 0.0."""
        analyzer = TranscriptAnalyzer(_make_config(min_words=5))
        scenes = [_scene(0, 5), _scene(5, 10)]
        decision = analyzer.analyze_video_transcript("/v.mp4", scenes, [])
        assert decision.transcript_coverage == 0.0

    @pytest.mark.fast
    def test_coverage_one_all_scenes_covered(self):
        """All scenes covered → coverage 1.0."""
        analyzer = TranscriptAnalyzer(_make_config(min_words=3))
        scenes = [_scene(0, 5), _scene(5, 10)]
        transcript = [
            _seg(0, 5, "one two three four"),
            _seg(5, 10, "five six seven eight"),
        ]
        decision = analyzer.analyze_video_transcript("/v.mp4", scenes, transcript)
        assert decision.transcript_coverage == 1.0

    @pytest.mark.fast
    def test_coverage_proportional_half(self):
        """Half scenes covered → coverage 0.5."""
        analyzer = TranscriptAnalyzer(_make_config(min_words=5))
        scenes = [_scene(0, 5), _scene(5, 10)]
        transcript = [
            _seg(0, 5, "one two three four five six"),  # sufficient for scene 0
            # scene 1 has no transcript
        ]
        decision = analyzer.analyze_video_transcript("/v.mp4", scenes, transcript)
        assert decision.transcript_coverage == 0.5

    @pytest.mark.fast
    def test_coverage_proportional_two_thirds(self):
        """2 of 3 scenes covered → coverage ≈ 0.667."""
        analyzer = TranscriptAnalyzer(_make_config(min_words=3))
        scenes = [_scene(0, 5), _scene(5, 10), _scene(10, 15)]
        transcript = [
            _seg(0, 5, "one two three four"),      # sufficient
            _seg(5, 10, "five six seven eight"),    # sufficient
            # scene 2 has no transcript
        ]
        decision = analyzer.analyze_video_transcript("/v.mp4", scenes, transcript)
        assert abs(decision.transcript_coverage - 2 / 3) < 0.01

    @pytest.mark.fast
    def test_coverage_proportional_one_third(self):
        """1 of 3 scenes covered → coverage ≈ 0.333."""
        analyzer = TranscriptAnalyzer(_make_config(min_words=3))
        scenes = [_scene(0, 5), _scene(5, 10), _scene(10, 15)]
        transcript = [
            _seg(0, 5, "one two three four"),
        ]
        decision = analyzer.analyze_video_transcript("/v.mp4", scenes, transcript)
        assert abs(decision.transcript_coverage - 1 / 3) < 0.01

    @pytest.mark.fast
    def test_coverage_type_is_float(self):
        """transcript_coverage is always a float."""
        analyzer = TranscriptAnalyzer(_make_config(min_words=3))
        scenes = [_scene(0, 5)]
        transcript = [_seg(0, 5, "one two three four")]
        decision = analyzer.analyze_video_transcript("/v.mp4", scenes, transcript)
        assert isinstance(decision.transcript_coverage, float)

    @pytest.mark.fast
    def test_coverage_zero_with_empty_text_segments(self):
        """Segments with empty text → coverage 0.0."""
        analyzer = TranscriptAnalyzer(_make_config(min_words=5))
        scenes = [_scene(0, 5), _scene(5, 10)]
        transcript = [
            _seg(0, 5, ""),
            _seg(5, 10, "   "),
        ]
        decision = analyzer.analyze_video_transcript("/v.mp4", scenes, transcript)
        assert decision.transcript_coverage == 0.0

    @pytest.mark.fast
    def test_video_vision_decision_fields(self):
        """VideoVisionDecision stores all fields correctly."""
        decision = VideoVisionDecision(
            video_path="/test.mp4",
            needs_vision=True,
            reason="No scenes detected",
            transcript_coverage=0.0,
            sparse_scenes=[],
            total_scenes=0,
        )
        assert decision.video_path == "/test.mp4"
        assert decision.needs_vision is True
        assert decision.reason == "No scenes detected"
        assert decision.transcript_coverage == 0.0
        assert decision.sparse_scenes == []
        assert decision.total_scenes == 0

    @pytest.mark.fast
    def test_video_vision_decision_asdict(self):
        """VideoVisionDecision serializes via asdict()."""
        decision = VideoVisionDecision(
            video_path="/v.mp4",
            needs_vision=False,
            reason="Good coverage",
            transcript_coverage=0.85,
            sparse_scenes=[2, 4],
            total_scenes=5,
        )
        d = asdict(decision)
        assert d["video_path"] == "/v.mp4"
        assert d["transcript_coverage"] == 0.85
        assert d["sparse_scenes"] == [2, 4]
        assert d["total_scenes"] == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
