"""
US-004: BrollMatchStage silent scene detection and scoring tests

Dedicated tests for:
- AC1: _detect_silent_scenes() word count threshold detection
- AC2: _detect_silent_scenes() face-detection B-roll (is_broll=True)
- AC3: _validate_and_normalize_weights() normalization and logging
- AC4: _calculate_embedding_score() cosine similarity and edge cases
- AC5: _calculate_keyword_score() overlap ratio for identical/disjoint/empty
- AC6: _calculate_entity_score() partial/substring matches and empty entities
"""

import pytest
import sys
import logging
import numpy as np
from unittest.mock import Mock, MagicMock, patch
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.sections.broll import BrollConfig, BrollSourceBoostConfig
from src.stages.broll_match import BrollMatchStage, BrollScene, BrollMatch
from src.state import PipelineState, VoiceoverSegment, DownloadedVideo


# ============================================================================
# AC1: _detect_silent_scenes() word count threshold
# ============================================================================

class TestDetectSilentScenesWordCountUS004:
    """AC1: _detect_silent_scenes() identifies scenes with word_count below threshold."""

    @pytest.fixture
    def stage(self):
        return BrollMatchStage()

    @pytest.fixture
    def state_5_entries(self):
        """State with 5 text_metadata entries: 3 silent (<10 words), 2 speech (>=10 words)."""
        state = PipelineState()
        state.text_metadata = [
            # Silent scenes (< 10 words)
            {
                "source_file": "/videos/silent1.mp4",
                "start_time": 0.0,
                "end_time": 5.0,
                "text": "short clip",
                "is_broll": False,
            },
            {
                "source_file": "/videos/silent2.mp4",
                "start_time": 0.0,
                "end_time": 8.0,
                "text": "[Music]",
                "is_broll": False,
            },
            {
                "source_file": "/videos/silent3.mp4",
                "start_time": 0.0,
                "end_time": 6.0,
                "text": "",
                "is_broll": False,
            },
            # Speech scenes (>= 10 words)
            {
                "source_file": "/videos/speech1.mp4",
                "start_time": 0.0,
                "end_time": 30.0,
                "text": "The earthquake struck at dawn and many people were caught off guard by the intensity",
                "is_broll": False,
            },
            {
                "source_file": "/videos/speech2.mp4",
                "start_time": 0.0,
                "end_time": 25.0,
                "text": "Rescue teams arrived quickly and began searching through the rubble for survivors trapped underneath",
                "is_broll": False,
            },
        ]
        state.transcripts = {}
        state.downloaded_videos = []
        return state

    @pytest.mark.fast
    def test_detects_3_silent_scenes_from_5(self, stage, state_5_entries):
        """3 entries have word_count < 10, 2 have >= 10 words."""
        broll_config = BrollConfig(min_words_threshold=10)
        scenes = stage._detect_silent_scenes(state_5_entries, broll_config)

        assert len(scenes) == 3
        silent_files = {s.source_file for s in scenes}
        assert "/videos/silent1.mp4" in silent_files
        assert "/videos/silent2.mp4" in silent_files
        assert "/videos/silent3.mp4" in silent_files

    @pytest.mark.fast
    def test_speech_scenes_excluded(self, stage, state_5_entries):
        """Speech entries (>= 10 words) should NOT appear in results."""
        broll_config = BrollConfig(min_words_threshold=10)
        scenes = stage._detect_silent_scenes(state_5_entries, broll_config)

        scene_files = {s.source_file for s in scenes}
        assert "/videos/speech1.mp4" not in scene_files
        assert "/videos/speech2.mp4" not in scene_files

    @pytest.mark.fast
    def test_returns_broll_scene_objects(self, stage, state_5_entries):
        """Each result is a BrollScene with correct fields."""
        broll_config = BrollConfig(min_words_threshold=10)
        scenes = stage._detect_silent_scenes(state_5_entries, broll_config)

        for scene in scenes:
            assert isinstance(scene, BrollScene)
            assert isinstance(scene.source_file, str)
            assert isinstance(scene.start_time, float)
            assert isinstance(scene.end_time, float)
            assert isinstance(scene.word_count, int)

    @pytest.mark.fast
    def test_ignore_markers_stripped_before_count(self, stage):
        """[Music] markers are stripped before word counting."""
        state = PipelineState()
        state.text_metadata = [
            {
                "source_file": "/videos/music_only.mp4",
                "start_time": 0.0,
                "end_time": 10.0,
                "text": "[Music] [Applause] [Silence]",
                "is_broll": False,
            },
        ]
        state.transcripts = {}
        state.downloaded_videos = []

        broll_config = BrollConfig(min_words_threshold=10)
        scenes = stage._detect_silent_scenes(state, broll_config)

        # After stripping markers, text is mostly empty -> silent
        assert len(scenes) == 1

    @pytest.mark.fast
    def test_custom_threshold_changes_detection(self, stage):
        """Changing threshold to 5 should exclude entries with 5+ words."""
        state = PipelineState()
        state.text_metadata = [
            {
                "source_file": "/videos/four_words.mp4",
                "start_time": 0.0,
                "end_time": 5.0,
                "text": "just four words here",
                "is_broll": False,
            },
            {
                "source_file": "/videos/six_words.mp4",
                "start_time": 0.0,
                "end_time": 5.0,
                "text": "this has exactly six words total",
                "is_broll": False,
            },
        ]
        state.transcripts = {}
        state.downloaded_videos = []

        # threshold=5 -> 4-word entry is silent, 6-word is not
        broll_config = BrollConfig(min_words_threshold=5)
        scenes = stage._detect_silent_scenes(state, broll_config)

        assert len(scenes) == 1
        assert scenes[0].source_file == "/videos/four_words.mp4"


# ============================================================================
# AC2: _detect_silent_scenes() face-detection B-roll (is_broll=True)
# ============================================================================

class TestDetectSilentScenesFaceDetectionUS004:
    """AC2: _detect_silent_scenes() also detects is_broll=True from SceneDetection."""

    @pytest.fixture
    def stage(self):
        return BrollMatchStage()

    @pytest.mark.fast
    def test_is_broll_true_included(self, stage):
        """Entries with is_broll=True are included regardless of word count."""
        state = PipelineState()
        state.text_metadata = [
            {
                "source_file": "/videos/face_broll.mp4",
                "start_time": 0.0,
                "end_time": 10.0,
                "text": "This scene has many words but is still marked as broll by face detection system",
                "is_broll": True,  # face detection flagged
            },
        ]
        state.transcripts = {}
        state.downloaded_videos = []

        broll_config = BrollConfig(min_words_threshold=10)
        scenes = stage._detect_silent_scenes(state, broll_config)

        assert len(scenes) == 1
        assert scenes[0].source_file == "/videos/face_broll.mp4"

    @pytest.mark.fast
    def test_both_detection_methods_contribute(self, stage):
        """Both word-count silent and is_broll=True scenes appear in results."""
        state = PipelineState()
        state.text_metadata = [
            # Face-detection B-roll (has speech but is_broll=True)
            {
                "source_file": "/videos/face_detected.mp4",
                "start_time": 0.0,
                "end_time": 10.0,
                "text": "This has enough words to be speech but face detection flagged it",
                "is_broll": True,
            },
            # Word-count silent (< threshold)
            {
                "source_file": "/videos/word_count_silent.mp4",
                "start_time": 0.0,
                "end_time": 5.0,
                "text": "short",
                "is_broll": False,
            },
            # Not B-roll (speech + not flagged)
            {
                "source_file": "/videos/not_broll.mp4",
                "start_time": 0.0,
                "end_time": 20.0,
                "text": "This is a full speech segment with many words about the topic of earthquakes",
                "is_broll": False,
            },
        ]
        state.transcripts = {}
        state.downloaded_videos = []

        broll_config = BrollConfig(min_words_threshold=10)
        scenes = stage._detect_silent_scenes(state, broll_config)

        assert len(scenes) == 2
        scene_files = {s.source_file for s in scenes}
        assert "/videos/face_detected.mp4" in scene_files
        assert "/videos/word_count_silent.mp4" in scene_files
        assert "/videos/not_broll.mp4" not in scene_files

    @pytest.mark.fast
    def test_is_broll_scene_has_word_count_zero(self, stage):
        """Scenes detected via is_broll=True have word_count=0."""
        state = PipelineState()
        state.text_metadata = [
            {
                "source_file": "/videos/broll_flagged.mp4",
                "start_time": 5.0,
                "end_time": 15.0,
                "text": "Some text here",
                "is_broll": True,
            },
        ]
        state.transcripts = {}
        state.downloaded_videos = []

        broll_config = BrollConfig(min_words_threshold=10)
        scenes = stage._detect_silent_scenes(state, broll_config)

        assert len(scenes) == 1
        assert scenes[0].word_count == 0

    @pytest.mark.fast
    def test_empty_transcripts_with_downloaded_video(self, stage):
        """Videos with empty transcript list in state.transcripts are detected."""
        state = PipelineState()
        state.text_metadata = []
        state.transcripts = {"fully_silent": []}
        state.downloaded_videos = [
            DownloadedVideo(file="/videos/fully_silent.mp4", source="youtube"),
        ]

        broll_config = BrollConfig(min_words_threshold=10)
        scenes = stage._detect_silent_scenes(state, broll_config)

        assert len(scenes) == 1
        assert scenes[0].source_file == "/videos/fully_silent.mp4"


# ============================================================================
# AC3: _validate_and_normalize_weights()
# ============================================================================

class TestValidateAndNormalizeWeightsUS004:
    """AC3: _validate_and_normalize_weights() normalizes to sum=1.0 and warns."""

    @pytest.fixture
    def stage(self):
        return BrollMatchStage()

    @pytest.mark.fast
    def test_default_weights_no_warning(self, stage):
        """Default weights (0.4 + 0.35 + 0.25 = 1.0) return None (no warning)."""
        broll_config = BrollConfig()
        result = stage._validate_and_normalize_weights(broll_config)
        assert result is None

    @pytest.mark.fast
    def test_exact_sum_no_normalization(self, stage):
        """Weights that sum exactly to 1.0 return None."""
        broll_config = BrollConfig(
            embedding_weight=0.5,
            keyword_weight=0.3,
            entity_weight=0.2,
        )
        result = stage._validate_and_normalize_weights(broll_config)
        assert result is None

    @pytest.mark.fast
    def test_within_tolerance_normalizes(self, stage):
        """Weights within 0.01 tolerance are auto-normalized."""
        broll_config = BrollConfig(
            embedding_weight=0.40,
            keyword_weight=0.355,
            entity_weight=0.25,  # Sum = 1.005 (within 0.01)
        )
        result = stage._validate_and_normalize_weights(broll_config)

        # Should return a normalization message
        assert result is not None
        assert "normalized" in result.lower()

        # Weights should now sum to ~1.0
        new_sum = broll_config.embedding_weight + broll_config.keyword_weight + broll_config.entity_weight
        assert abs(new_sum - 1.0) < 1e-6

    @pytest.mark.fast
    def test_outside_tolerance_returns_warning(self, stage):
        """Weights outside 0.01 tolerance return warning without normalizing."""
        broll_config = BrollConfig(
            embedding_weight=0.5,
            keyword_weight=0.5,
            entity_weight=0.5,  # Sum = 1.5
        )
        result = stage._validate_and_normalize_weights(broll_config)

        # Should return warning about non-standard weights
        assert result is not None
        assert "1.5" in result or "not auto-normalized" in result.lower()

        # Weights should NOT be modified
        assert broll_config.embedding_weight == 0.5
        assert broll_config.keyword_weight == 0.5
        assert broll_config.entity_weight == 0.5

    @pytest.mark.fast
    def test_logs_warning_when_sum_differs(self, stage, caplog):
        """Logger.warning called when weights don't sum to 1.0."""
        broll_config = BrollConfig(
            embedding_weight=0.3,
            keyword_weight=0.3,
            entity_weight=0.3,  # Sum = 0.9
        )
        with caplog.at_level(logging.WARNING, logger="src.stages.broll_match"):
            stage._validate_and_normalize_weights(broll_config)

        assert any("weights sum to" in record.message.lower() for record in caplog.records)


# ============================================================================
# AC4: _calculate_embedding_score()
# ============================================================================

class TestCalculateEmbeddingScoreUS004:
    """AC4: _calculate_embedding_score() returns cosine similarity in [0.0, 1.0]."""

    @pytest.fixture
    def stage(self):
        return BrollMatchStage()

    @pytest.mark.fast
    def test_identical_vectors_score_one(self, stage):
        """Same vector should return ~1.0."""
        vec = np.array([1.0, 2.0, 3.0])
        score = stage._calculate_embedding_score(vec, vec)
        assert abs(score - 1.0) < 0.01

    @pytest.mark.fast
    def test_orthogonal_vectors_score_zero(self, stage):
        """Orthogonal vectors should return ~0.0."""
        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([0.0, 1.0, 0.0])
        score = stage._calculate_embedding_score(v1, v2)
        assert abs(score) < 0.01

    @pytest.mark.fast
    def test_returns_float_in_range(self, stage):
        """Score should be a float in [0.0, 1.0]."""
        v1 = np.array([0.5, 0.3, 0.8])
        v2 = np.array([0.2, 0.9, 0.1])
        score = stage._calculate_embedding_score(v1, v2)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    @pytest.mark.fast
    def test_none_voiceover_embedding_returns_zero(self, stage):
        """None voiceover embedding returns 0.0."""
        score = stage._calculate_embedding_score(None, np.array([1.0, 2.0]))
        assert score == 0.0

    @pytest.mark.fast
    def test_none_scene_embedding_returns_zero(self, stage):
        """None scene embedding returns 0.0."""
        score = stage._calculate_embedding_score(np.array([1.0, 2.0]), None)
        assert score == 0.0

    @pytest.mark.fast
    def test_both_none_returns_zero(self, stage):
        """Both None returns 0.0."""
        score = stage._calculate_embedding_score(None, None)
        assert score == 0.0

    @pytest.mark.fast
    def test_zero_vector_no_crash(self, stage):
        """Zero-vector input should not crash (returns 0.0)."""
        zero_vec = np.array([0.0, 0.0, 0.0])
        normal_vec = np.array([1.0, 2.0, 3.0])

        score = stage._calculate_embedding_score(zero_vec, normal_vec)
        assert isinstance(score, float)
        # Division by zero in norm produces nan/inf, caught by try/except -> 0.0
        assert score == 0.0

    @pytest.mark.fast
    def test_negative_cosine_clamped_to_zero(self, stage):
        """Opposite vectors have negative cosine; result clamped to 0.0."""
        v1 = np.array([1.0, 0.0])
        v2 = np.array([-1.0, 0.0])
        score = stage._calculate_embedding_score(v1, v2)
        assert score == 0.0


# ============================================================================
# AC5: _calculate_keyword_score()
# ============================================================================

class TestCalculateKeywordScoreUS004:
    """AC5: _calculate_keyword_score() returns overlap ratio."""

    @pytest.fixture
    def stage(self):
        return BrollMatchStage()

    @pytest.mark.fast
    def test_identical_keyword_sets_score_one(self, stage):
        """Identical keyword overlap should return 1.0."""
        keywords = {"earthquake", "damage"}
        score = stage._calculate_keyword_score(
            "earthquake damage report",
            "earthquake damage footage",
            keywords
        )
        assert score == 1.0

    @pytest.mark.fast
    def test_disjoint_sets_score_zero(self, stage):
        """No keyword overlap returns 0.0."""
        keywords = {"earthquake", "tsunami"}
        score = stage._calculate_keyword_score(
            "earthquake tsunami",
            "sunset beach ocean",
            keywords
        )
        assert score == 0.0

    @pytest.mark.fast
    def test_empty_keyword_list_fallback(self, stage):
        """Empty keyword set falls back to direct word overlap."""
        keywords = set()
        score = stage._calculate_keyword_score(
            "sunset beach ocean",
            "sunset beach waves",
            keywords
        )
        # "sunset" and "beach" overlap out of 3 vo words -> 2/3
        assert abs(score - 2.0 / 3.0) < 0.01

    @pytest.mark.fast
    def test_empty_voiceover_text(self, stage):
        """Empty voiceover text with keywords returns 0.0."""
        keywords = {"earthquake"}
        score = stage._calculate_keyword_score(
            "",
            "earthquake footage",
            keywords
        )
        # Empty vo_words -> vo_keywords is empty -> falls through to fallback
        # Empty vo_words -> len(vo_words)==0 -> return 0.0
        assert score == 0.0

    @pytest.mark.fast
    def test_empty_scene_description_returns_zero(self, stage):
        """Empty scene description returns 0.0."""
        keywords = {"earthquake"}
        score = stage._calculate_keyword_score(
            "earthquake occurred",
            "",
            keywords
        )
        assert score == 0.0

    @pytest.mark.fast
    def test_partial_keyword_overlap(self, stage):
        """Partial overlap returns fractional score."""
        keywords = {"earthquake", "tsunami", "flood"}
        score = stage._calculate_keyword_score(
            "earthquake tsunami flood",
            "earthquake footage",  # Only "earthquake" overlaps
            keywords
        )
        # vo_keywords = {earthquake, tsunami, flood}, scene_keywords = {earthquake}
        # overlap = {earthquake}, score = 1/3
        assert abs(score - 1.0 / 3.0) < 0.01


# ============================================================================
# AC6: _calculate_entity_score()
# ============================================================================

class TestCalculateEntityScoreUS004:
    """AC6: _calculate_entity_score() scores entity overlap with substring support."""

    @pytest.fixture
    def stage(self):
        return BrollMatchStage()

    @pytest.mark.fast
    def test_full_entity_overlap(self, stage):
        """All entities in both texts -> score 1.0."""
        entities = {"california", "san francisco"}
        score = stage._calculate_entity_score(
            "california san francisco earthquake",
            "california san francisco footage",
            entities
        )
        assert score == 1.0

    @pytest.mark.fast
    def test_partial_entity_match(self, stage):
        """Only some entities match -> fractional score."""
        entities = {"california", "new york"}
        score = stage._calculate_entity_score(
            "california new york",
            "california footage only",
            entities
        )
        # vo has both, scene has only "california" -> 1/2
        assert abs(score - 0.5) < 0.01

    @pytest.mark.fast
    def test_substring_entity_match(self, stage):
        """Substring matching: 'san' inside 'san francisco' counts."""
        entities = {"san francisco"}
        score = stage._calculate_entity_score(
            "visit san francisco today",
            "beautiful san francisco skyline",
            entities
        )
        # "san francisco" is in both -> 1/1
        assert score == 1.0

    @pytest.mark.fast
    def test_empty_entities_returns_zero(self, stage):
        """Empty entity set returns 0.0."""
        score = stage._calculate_entity_score(
            "california earthquake",
            "california footage",
            set()
        )
        assert score == 0.0

    @pytest.mark.fast
    def test_no_entities_in_voiceover(self, stage):
        """Entities not in voiceover text -> 0.0."""
        entities = {"california", "new york"}
        score = stage._calculate_entity_score(
            "beautiful weather today",
            "california new york footage",
            entities
        )
        assert score == 0.0

    @pytest.mark.fast
    def test_empty_scene_description(self, stage):
        """Empty scene description returns 0.0."""
        entities = {"california"}
        score = stage._calculate_entity_score(
            "california earthquake",
            "",
            entities
        )
        assert score == 0.0

    @pytest.mark.fast
    def test_case_insensitive_matching(self, stage):
        """Entity matching is case-insensitive."""
        entities = {"california"}
        score = stage._calculate_entity_score(
            "CALIFORNIA earthquake",
            "California footage",
            entities
        )
        # Both lowered -> "california" found in both
        assert score == 1.0
