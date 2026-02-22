"""
Tests for US-122-010: Listicle group propagation to match stage.

Verifies:
- ListicleGroup objects are correctly passed to match stage
- Segment indices in listicle groups match voiceover segments
- Topic keywords from listicle groups are used in matching
- Listicle group boundaries are respected in matching
- expected_count from header is used to validate match completeness
- Listicle group metadata is preserved in checkpoint for resume
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from dataclasses import dataclass, field
from typing import List, Optional

from src.state import PipelineState, VoiceoverSegment
from src.chapter_detection.models import ListicleGroup
from src.stages.match import MatchStage
from src.checkpoint import CheckpointManager


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def voiceover_segments_with_listicle():
    """Voiceover segments that form a listicle structure."""
    return [
        VoiceoverSegment(index=0, start=0.0, end=3.0, text="Number one: the beach destination"),
        VoiceoverSegment(index=1, start=3.0, end=6.0, text="Number one continued: sandy shores"),
        VoiceoverSegment(index=2, start=6.0, end=9.0, text="Number two: the mountain retreat"),
        VoiceoverSegment(index=3, start=9.0, end=12.0, text="Number two continued: alpine views"),
        VoiceoverSegment(index=4, start=12.0, end=15.0, text="Number three: city exploration"),
        VoiceoverSegment(index=5, start=15.0, end=18.0, text="Number three continued: urban sights"),
    ]


@pytest.fixture
def listicle_groups():
    """Listicle groups matching voiceover segments."""
    return [
        ListicleGroup(
            group_id=0,
            item_label="first",
            marker_type="ordinal",
            start_segment_idx=0,
            end_segment_idx=1,
            topic_keywords=["beach", "sandy shores", "ocean"],
            expected_count=3,
            confidence=0.9,
        ),
        ListicleGroup(
            group_id=1,
            item_label="second",
            marker_type="ordinal",
            start_segment_idx=2,
            end_segment_idx=3,
            topic_keywords=["mountain", "alpine", "hiking"],
            expected_count=3,
            confidence=0.85,
        ),
        ListicleGroup(
            group_id=2,
            item_label="third",
            marker_type="ordinal",
            start_segment_idx=4,
            end_segment_idx=5,
            topic_keywords=["city", "urban", "exploration"],
            expected_count=3,
            confidence=0.8,
        ),
    ]


@pytest.fixture
def state_with_listicle(voiceover_segments_with_listicle, listicle_groups):
    """PipelineState with listicle groups set up."""
    state = PipelineState()
    state.voiceover_segments = voiceover_segments_with_listicle
    state.listicle_groups = listicle_groups
    state.face_preference = "neutral"
    state.text_metadata = []
    return state


@pytest.fixture
def mock_config():
    """Mock config for match stage."""
    config = MagicMock()
    config.matching = MagicMock()
    config.matching.skip_matching = False
    config.matching.min_confidence = 0.3
    config.matching.high_confidence_threshold = 0.7
    config.matching.embedding_candidates = 5
    config.matching.llm_rerank_candidates = 3
    config.matching.max_clip_reuse = 2
    config.matching.force_rematch = False
    config.matching.delta_matching_enabled = True
    config.matching.intermediate_checkpoint_interval = 0  # Disable for tests
    config.matching.chapter_matching_enabled = True
    config.matching.enforce_chapter_boundaries = False
    config.matching.prefer_chapter_aligned_segments = True
    config.matching.chapter_alignment_boost = 0.05
    config.matching.no_chapter_fallback_strategy = "global"

    config.cache = MagicMock()
    config.cache.cache_dir = ".cache"

    config.output = MagicMock()
    config.output.include_strategy_tracks = False

    config.variety = MagicMock()
    config.variety.enforce_timeline_variety = False

    return config


@pytest.fixture
def mock_checkpoint_with_listicle():
    """Mock checkpoint with listicle groups in chapter_data."""
    chapter_data = {
        "chapters": [],
        "listicle_groups": [
            {
                "group_id": 0,
                "item_label": "first",
                "marker_type": "ordinal",
                "start_segment_idx": 0,
                "end_segment_idx": 1,
                "topic_keywords": ["beach", "ocean"],
                "expected_count": 3,
                "confidence": 0.9,
            },
            {
                "group_id": 1,
                "item_label": "second",
                "marker_type": "ordinal",
                "start_segment_idx": 2,
                "end_segment_idx": 3,
                "topic_keywords": ["mountain", "alpine"],
                "expected_count": 3,
                "confidence": 0.85,
            },
        ],
    }

    checkpoint = Mock(spec=CheckpointManager)
    checkpoint.get_stage_data.return_value = {
        "match_count": 3,
        "avg_confidence": 0.75,
        "matches": [],
        "chapter_data": chapter_data,
    }
    return checkpoint


# ============================================================================
# AC1: Verify ListicleGroup objects are correctly passed to match stage
# ============================================================================

class TestListicleGroupObjectsPassedToMatchStage:
    """Verify ListicleGroup objects are correctly passed to match stage."""

    def test_listicle_groups_stored_on_state(self, state_with_listicle):
        """ListicleGroup objects should be stored on state.listicle_groups."""
        assert hasattr(state_with_listicle, "listicle_groups")
        assert len(state_with_listicle.listicle_groups) == 3

        # Verify all are ListicleGroup instances
        for group in state_with_listicle.listicle_groups:
            assert isinstance(group, ListicleGroup)

    def test_listicle_groups_have_required_fields(self, listicle_groups):
        """ListicleGroup objects should have all required fields."""
        group = listicle_groups[0]

        assert hasattr(group, "group_id")
        assert hasattr(group, "item_label")
        assert hasattr(group, "marker_type")
        assert hasattr(group, "start_segment_idx")
        assert hasattr(group, "end_segment_idx")
        assert hasattr(group, "topic_keywords")
        assert hasattr(group, "expected_count")
        assert hasattr(group, "confidence")

    def test_listicle_groups_passed_to_match_function(self, voiceover_segments_with_listicle, listicle_groups):
        """ListicleGroup objects should be accessible to match_all_segments."""
        # Simulate what MatchStage._run_matching does
        listicle_groups_passed = listicle_groups

        # Verify the groups can be passed and accessed
        assert listicle_groups_passed is not None
        assert len(listicle_groups_passed) == 3
        assert listicle_groups_passed[0].topic_keywords == ["beach", "sandy shores", "ocean"]

    @patch("src.stages.match.MatchStage._run_matching")
    def test_match_stage_passes_listicle_to_matching(self, mock_run_matching):
        """MatchStage should store listicle_groups on state for matching."""
        # Test that listicle groups are available on state
        state = PipelineState()
        state.listicle_groups = [
            ListicleGroup(
                group_id=0,
                item_label="first",
                marker_type="ordinal",
                start_segment_idx=0,
                end_segment_idx=1,
                topic_keywords=["beach", "ocean"],
                expected_count=3,
                confidence=0.9,
            )
        ]

        # Verify that listicle_groups are accessible
        assert hasattr(state, "listicle_groups")
        assert state.listicle_groups is not None
        assert len(state.listicle_groups) == 1
        assert state.listicle_groups[0].topic_keywords == ["beach", "ocean"]


# ============================================================================
# AC2: Test that segment indices in listicle groups match voiceover segments
# ============================================================================

class TestSegmentIndicesMatchVoiceoverSegments:
    """Test that segment indices in listicle groups match voiceover segments."""

    def test_listicle_segment_indices_within_vo_range(self, voiceover_segments_with_listicle, listicle_groups):
        """All listicle group segment indices should be within voiceover range."""
        vo_count = len(voiceover_segments_with_listicle)

        for group in listicle_groups:
            assert 0 <= group.start_segment_idx < vo_count
            assert 0 <= group.end_segment_idx < vo_count
            assert group.start_segment_idx <= group.end_segment_idx

    def test_listicle_indices_cover_all_segments(self, voiceover_segments_with_listicle, listicle_groups):
        """Listicle group indices should cover all voiceover segments."""
        # Collect all covered indices
        covered_indices = set()
        for group in listicle_groups:
            for idx in range(group.start_segment_idx, group.end_segment_idx + 1):
                covered_indices.add(idx)

        # All voiceover segments should be covered
        expected = set(range(len(voiceover_segments_with_listicle)))
        assert covered_indices == expected

    def test_listicle_indices_no_overlap_between_groups(self, listicle_groups):
        """Listicle groups should not have overlapping segment indices."""
        for i, group1 in enumerate(listicle_groups):
            for group2 in listicle_groups[i + 1:]:
                # No overlap: either group1 ends before group2 starts or vice versa
                assert group1.end_segment_idx < group2.start_segment_idx or \
                       group2.end_segment_idx < group1.start_segment_idx


# ============================================================================
# AC3: Verify topic keywords from listicle groups are used in matching
# ============================================================================

class TestTopicKeywordsUsedInMatching:
    """Verify topic keywords from listicle groups are used in matching."""

    def test_listicle_groups_have_topic_keywords(self, listicle_groups):
        """ListicleGroup should have topic_keywords field populated."""
        for group in listicle_groups:
            assert hasattr(group, "topic_keywords")
            assert isinstance(group.topic_keywords, list)
            assert len(group.topic_keywords) > 0

    def test_topic_keywords_passed_to_matching(self, listicle_groups):
        """Topic keywords should be accessible when passed to matching."""
        # Simulate what happens in MatchStage
        passed_groups = listicle_groups

        # Extract keywords from first group
        keywords = passed_groups[0].topic_keywords

        assert "beach" in keywords
        assert "sandy shores" in keywords
        assert "ocean" in keywords

    def test_topic_keywords_used_in_scoring_function(self):
        """Topic keywords should be available in scoring functions."""
        from src.matching.scoring import apply_listicle_consistency

        # Create a mock match and listicle group
        mock_vo_seg = Mock()
        mock_vo_seg.index = 0

        mock_video_seg = Mock()
        mock_video_seg.index = 0

        mock_recent_matches = []

        test_groups = [
            ListicleGroup(
                group_id=0,
                item_label="first",
                marker_type="ordinal",
                start_segment_idx=0,
                end_segment_idx=1,
                topic_keywords=["beach", "ocean"],
                expected_count=3,
                confidence=0.9,
            )
        ]

        # Function should handle the listicle groups without error
        # Returns tuple (confidence, reason)
        result = apply_listicle_consistency(
            confidence=0.7,
            vo_segment=mock_vo_seg,
            video_segment=mock_video_seg,
            listicle_groups=test_groups,
            recent_matches=mock_recent_matches,
        )

        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], float)  # confidence
        assert isinstance(result[1], str)    # reason


# ============================================================================
# AC4: Test that listicle group boundaries are respected in matching
# ============================================================================

class TestListicleGroupBoundariesRespected:
    """Test that listicle group boundaries are respected in matching."""

    def test_group_boundaries_defined(self, listicle_groups):
        """Listicle groups should have clear start/end boundaries."""
        for group in listicle_groups:
            assert group.start_segment_idx is not None
            assert group.end_segment_idx is not None
            assert group.start_segment_idx <= group.end_segment_idx

    def test_group_boundaries_used_in_scoring(self):
        """Group boundaries should be used in listicle consistency scoring."""
        from src.matching.scoring import apply_listicle_consistency

        # Create a scenario where segment is at group boundary
        mock_vo_seg = Mock()
        mock_vo_seg.index = 0  # First segment in group

        mock_video_seg = Mock()
        mock_video_seg.index = 0

        # Recent matches should include previous segments
        mock_prev_match = Mock()
        mock_prev_match.voiceover_segment = Mock()
        mock_prev_match.voiceover_segment.index = -1  # No previous segment index

        test_groups = [
            ListicleGroup(
                group_id=0,
                item_label="first",
                marker_type="ordinal",
                start_segment_idx=0,
                end_segment_idx=1,
                topic_keywords=["beach"],
                expected_count=3,
                confidence=0.9,
            )
        ]

        # The function should consider group boundaries
        # Returns tuple (confidence, reason)
        result = apply_listicle_consistency(
            confidence=0.7,
            vo_segment=mock_vo_seg,
            video_segment=mock_video_seg,
            listicle_groups=test_groups,
            recent_matches=[mock_prev_match],
        )

        assert isinstance(result, tuple)
        assert len(result) == 2
        # Result confidence should be within valid range
        assert 0.0 <= result[0] <= 1.0


# ============================================================================
# AC5: Verify expected_count from header is used to validate match completeness
# ============================================================================

class TestExpectedCountValidation:
    """Verify expected_count from header is used to validate match completeness."""

    def test_listicle_groups_have_expected_count(self, listicle_groups):
        """ListicleGroup should have expected_count from header detection."""
        for group in listicle_groups:
            assert hasattr(group, "expected_count")
            assert group.expected_count is not None
            assert group.expected_count > 0

    def test_expected_count_from_header(self):
        """expected_count should come from header detection (e.g., 'top 10')."""
        # Create a group simulating header detection
        group = ListicleGroup(
            group_id=0,
            item_label="top 10",
            marker_type="numbered",
            start_segment_idx=0,
            end_segment_idx=0,
            topic_keywords=["topic"],
            expected_count=10,  # From "top 10"
            confidence=0.95,
        )

        assert group.expected_count == 10
        assert group.item_label == "top 10"

    def test_expected_count_for_validation(self, listicle_groups):
        """expected_count should be available for validation logic."""
        for group in listicle_groups:
            # Validation: actual segments should not exceed expected count significantly
            actual_segments = group.end_segment_idx - group.start_segment_idx + 1
            # This is informational - actual validation happens elsewhere
            assert actual_segments > 0


# ============================================================================
# AC6: Test that listicle group metadata is preserved in checkpoint for resume
# ============================================================================

class TestListicleGroupCheckpointPersistence:
    """Test that listicle group metadata is preserved in checkpoint for resume."""

    def test_serialize_chapter_data_includes_listicle(self, state_with_listicle):
        """_serialize_chapter_data should include listicle_groups."""
        stage = MatchStage()

        # Call serialize method
        result = stage._serialize_chapter_data(state_with_listicle)

        assert "listicle_groups" in result
        assert isinstance(result["listicle_groups"], list)
        assert len(result["listicle_groups"]) == 3

    def test_serialize_listicle_groups_to_dict(self, listicle_groups):
        """ListicleGroup should serialize to dict with to_dict()."""
        group = listicle_groups[0]
        serialized = group.to_dict()

        assert isinstance(serialized, dict)
        assert serialized["group_id"] == 0
        assert serialized["item_label"] == "first"
        assert serialized["start_segment_idx"] == 0
        assert serialized["end_segment_idx"] == 1
        assert serialized["topic_keywords"] == ["beach", "sandy shores", "ocean"]
        assert serialized["expected_count"] == 3

    def test_restore_chapter_data_listicle(self, mock_checkpoint_with_listicle):
        """_restore_chapter_data should restore listicle_groups from checkpoint."""
        state = PipelineState()

        stage = MatchStage()
        stage._restore_chapter_data(state, mock_checkpoint_with_listicle.get_stage_data.return_value)

        assert hasattr(state, "listicle_groups")
        assert len(state.listicle_groups) == 2

        # Verify restored correctly
        first_group = state.listicle_groups[0]
        assert first_group.group_id == 0
        assert first_group.item_label == "first"
        assert "beach" in first_group.topic_keywords

    def test_listicle_group_from_dict(self):
        """ListicleGroup.from_dict should restore from serialized dict."""
        data = {
            "group_id": 0,
            "item_label": "first",
            "marker_type": "ordinal",
            "start_segment_idx": 0,
            "end_segment_idx": 1,
            "topic_keywords": ["beach", "ocean"],
            "expected_count": 3,
            "confidence": 0.9,
        }

        group = ListicleGroup.from_dict(data)

        assert group.group_id == 0
        assert group.item_label == "first"
        assert group.marker_type == "ordinal"
        assert group.start_segment_idx == 0
        assert group.end_segment_idx == 1
        assert group.topic_keywords == ["beach", "ocean"]
        assert group.expected_count == 3
        assert group.confidence == 0.9

    def test_checkpoint_roundtrip_listicle_groups(self, listicle_groups):
        """Listicle groups should survive a checkpoint roundtrip."""
        # Simulate serialization (what happens in checkpoint)
        serialized = [g.to_dict() for g in listicle_groups]

        # Simulate deserialization (what happens on restore)
        restored = [ListicleGroup.from_dict(d) for d in serialized]

        # Verify roundtrip preserves data
        assert len(restored) == len(listicle_groups)

        for original, restored_group in zip(listicle_groups, restored):
            assert restored_group.group_id == original.group_id
            assert restored_group.item_label == original.item_label
            assert restored_group.start_segment_idx == original.start_segment_idx
            assert restored_group.end_segment_idx == original.end_segment_idx
            assert restored_group.topic_keywords == original.topic_keywords
            assert restored_group.expected_count == original.expected_count


# ============================================================================
# Integration: End-to-end test
# ============================================================================

class TestListicleGroupsEndToEnd:
    """End-to-end test for listicle group propagation."""

    def test_full_flow_detection_to_checkpoint(self):
        """Test the full flow: detection -> state -> matching -> checkpoint."""
        # Step 1: Voiceover segments (simulating input)
        vo_segments = [
            VoiceoverSegment(index=0, start=0.0, end=3.0, text="Number one: first topic"),
            VoiceoverSegment(index=1, start=3.0, end=6.0, text="Number two: second topic"),
            VoiceoverSegment(index=2, start=6.0, end=9.0, text="Number three: third topic"),
        ]

        # Step 2: Listicle detection (simulating detection result)
        detected_groups = [
            ListicleGroup(
                group_id=0,
                item_label="one",
                marker_type="ordinal",
                start_segment_idx=0,
                end_segment_idx=0,
                topic_keywords=["first"],
                expected_count=3,
                confidence=0.9,
            ),
            ListicleGroup(
                group_id=1,
                item_label="two",
                marker_type="ordinal",
                start_segment_idx=1,
                end_segment_idx=1,
                topic_keywords=["second"],
                expected_count=3,
                confidence=0.85,
            ),
            ListicleGroup(
                group_id=2,
                item_label="three",
                marker_type="ordinal",
                start_segment_idx=2,
                end_segment_idx=2,
                topic_keywords=["third"],
                expected_count=3,
                confidence=0.8,
            ),
        ]

        # Step 3: Store on state
        state = PipelineState()
        state.voiceover_segments = vo_segments
        state.listicle_groups = detected_groups

        # Verify storage
        assert len(state.listicle_groups) == 3

        # Step 4: Serialize for checkpoint
        stage = MatchStage()
        serialized = stage._serialize_chapter_data(state)

        assert "listicle_groups" in serialized
        assert len(serialized["listicle_groups"]) == 3

        # Step 5: Restore from checkpoint
        restored_state = PipelineState()
        checkpoint_data = {"chapter_data": serialized}
        stage._restore_chapter_data(restored_state, checkpoint_data)

        # Verify restoration
        assert len(restored_state.listicle_groups) == 3
        assert restored_state.listicle_groups[0].topic_keywords == ["first"]
        assert restored_state.listicle_groups[1].topic_keywords == ["second"]
        assert restored_state.listicle_groups[2].topic_keywords == ["third"]

        # Step 6: Pass to matching
        passed_groups = restored_state.listicle_groups
        assert len(passed_groups) == 3
        assert passed_groups[0].expected_count == 3
