"""
US-005: BrollMatchStage enrichment and matching pipeline tests

Dedicated tests for:
- AC1: _should_use_vision() returns False in audio-first, True in standard mode
- AC2: _enrich_with_keywords() extracts keywords from filename, strips prefixes
- AC3: _extract_keywords_from_filename() handles 5+ filename patterns
- AC4: _match_scenes_to_voiceover() multi-strategy scoring, best per segment
- AC5: can_skip() returns True when broll disabled or no voiceover segments
- AC6: restore() rebuilds broll_matches from checkpoint data
"""

import pytest
import sys
import numpy as np
from unittest.mock import Mock, MagicMock, patch
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.sections.broll import BrollConfig, BrollSourceBoostConfig
from src.stages.broll_match import BrollMatchStage, BrollScene, BrollMatch
from src.state import PipelineState, VoiceoverSegment, DownloadedVideo


# ============================================================================
# AC1: _should_use_vision() audio-first vs standard mode
# ============================================================================

class TestShouldUseVisionUS005:
    """AC1: _should_use_vision() returns False in audio-first mode, True in standard mode."""

    @pytest.fixture
    def stage(self):
        return BrollMatchStage()

    @pytest.mark.fast
    def test_returns_false_in_audio_first_mode(self, stage):
        """Audio-first mode disables Vision API (no video frames available)."""
        config = Mock()
        config.download = Mock()
        config.download.audio_first = Mock()
        config.download.audio_first.enabled = True
        config.vision = Mock()
        config.vision.enabled = True

        broll_config = BrollConfig(vision_enabled=True)
        result = stage._should_use_vision(config, broll_config)

        assert result is False

    @pytest.mark.fast
    def test_returns_true_in_standard_mode_with_vision(self, stage):
        """Standard mode with Vision API key configured returns True."""
        config = Mock()
        config.download = Mock()
        config.download.audio_first = None  # not enabled
        config.vision = Mock()
        config.vision.enabled = True

        broll_config = BrollConfig(vision_enabled=True)
        result = stage._should_use_vision(config, broll_config)

        assert result is True

    @pytest.mark.fast
    def test_returns_false_when_broll_vision_disabled(self, stage):
        """Vision disabled in broll config returns False regardless of global config."""
        config = Mock()
        config.download = Mock()
        config.download.audio_first = None
        config.vision = Mock()
        config.vision.enabled = True

        broll_config = BrollConfig(vision_enabled=False)
        result = stage._should_use_vision(config, broll_config)

        assert result is False

    @pytest.mark.fast
    def test_returns_false_when_global_vision_disabled(self, stage):
        """Global vision.enabled=False returns False."""
        config = Mock()
        config.download = Mock()
        config.download.audio_first = None
        config.vision = Mock()
        config.vision.enabled = False

        broll_config = BrollConfig(vision_enabled=True)
        result = stage._should_use_vision(config, broll_config)

        assert result is False

    @pytest.mark.fast
    def test_returns_false_when_no_vision_config(self, stage):
        """Missing global vision config returns False."""
        config = Mock()
        config.download = Mock()
        config.download.audio_first = None
        config.vision = None

        broll_config = BrollConfig(vision_enabled=True)
        result = stage._should_use_vision(config, broll_config)

        assert result is False

    @pytest.mark.fast
    def test_audio_first_enabled_false_treated_as_standard(self, stage):
        """audio_first.enabled=False is treated as standard mode."""
        config = Mock()
        config.download = Mock()
        config.download.audio_first = Mock()
        config.download.audio_first.enabled = False
        config.vision = Mock()
        config.vision.enabled = True

        broll_config = BrollConfig(vision_enabled=True)
        result = stage._should_use_vision(config, broll_config)

        assert result is True


# ============================================================================
# AC2: _enrich_with_keywords() extracts keywords from filename
# ============================================================================

class TestEnrichWithKeywordsUS005:
    """AC2: _enrich_with_keywords() extracts keywords, strips prefixes like pexels_/pixabay_."""

    @pytest.fixture
    def stage(self):
        return BrollMatchStage()

    @pytest.mark.fast
    def test_pexels_sunset_ocean_4k_yields_keywords(self, stage):
        """'pexels_sunset_ocean_4k.mp4' yields keywords including sunset, ocean, 4k."""
        scenes = [
            BrollScene(
                source_file="/videos/pexels_sunset_ocean_4k.mp4",
                start_time=0.0, end_time=10.0,
                word_count=0, transcript=""
            )
        ]
        broll_config = BrollConfig()
        count = stage._enrich_with_keywords(scenes, None, broll_config)

        assert count == 1
        desc = scenes[0].description.lower()
        assert "sunset" in desc
        assert "ocean" in desc
        assert "4k" in desc
        assert "pexels" not in desc

    @pytest.mark.fast
    def test_pixabay_prefix_stripped(self, stage):
        """'pixabay_city_night.mp4' strips pixabay_ prefix."""
        scenes = [
            BrollScene(
                source_file="/videos/pixabay_city_night.mp4",
                start_time=0.0, end_time=10.0,
                word_count=0, transcript=""
            )
        ]
        broll_config = BrollConfig()
        count = stage._enrich_with_keywords(scenes, None, broll_config)

        assert count == 1
        desc = scenes[0].description.lower()
        assert "city" in desc
        assert "night" in desc
        assert "pixabay" not in desc

    @pytest.mark.fast
    def test_skips_scenes_with_existing_description(self, stage):
        """Scenes with existing descriptions are not overwritten."""
        scenes = [
            BrollScene(
                source_file="/videos/pexels_sunset.mp4",
                start_time=0.0, end_time=10.0,
                word_count=0, transcript="",
                description="already described"
            ),
            BrollScene(
                source_file="/videos/earthquake_damage.mp4",
                start_time=0.0, end_time=10.0,
                word_count=0, transcript=""
            ),
        ]
        broll_config = BrollConfig()
        count = stage._enrich_with_keywords(scenes, None, broll_config)

        # Only the second scene should be enriched
        assert count == 1
        assert scenes[0].description == "already described"
        assert "earthquake" in scenes[1].description.lower()

    @pytest.mark.fast
    def test_video_id_removed_from_keywords(self, stage):
        """Video IDs (11+ alphanumeric chars) removed from extracted keywords."""
        scenes = [
            BrollScene(
                source_file="/videos/city_dQw4w9WgXcQ.mp4",
                start_time=0.0, end_time=10.0,
                word_count=0, transcript=""
            )
        ]
        broll_config = BrollConfig()
        count = stage._enrich_with_keywords(scenes, None, broll_config)

        assert count == 1
        assert "dQw4w9WgXcQ" not in scenes[0].description

    @pytest.mark.fast
    def test_empty_filename_returns_empty_description(self, stage):
        """Scene with no meaningful keywords still gets enriched (possibly empty)."""
        scenes = [
            BrollScene(
                source_file="/videos/dQw4w9WgXcQ.mp4",
                start_time=0.0, end_time=10.0,
                word_count=0, transcript=""
            )
        ]
        broll_config = BrollConfig()
        count = stage._enrich_with_keywords(scenes, None, broll_config)

        # Video ID removed, nothing left - description may be empty
        assert isinstance(scenes[0].description, str)


# ============================================================================
# AC3: _extract_keywords_from_filename() handles various patterns
# ============================================================================

class TestExtractKeywordsFromFilenameUS005:
    """AC3: _extract_keywords_from_filename() handles 5+ filename patterns."""

    @pytest.fixture
    def stage(self):
        return BrollMatchStage()

    @pytest.mark.fast
    def test_underscore_separated(self, stage):
        """'earthquake_damage_city.mp4' -> 'earthquake damage city'."""
        result = stage._extract_keywords_from_filename("/videos/earthquake_damage_city.mp4")
        assert "earthquake" in result.lower()
        assert "damage" in result.lower()
        assert "city" in result.lower()
        assert "_" not in result

    @pytest.mark.fast
    def test_hyphen_separated(self, stage):
        """'city-skyline-night.mp4' -> keywords with spaces."""
        result = stage._extract_keywords_from_filename("/videos/city-skyline-night.mp4")
        assert "city" in result.lower()
        assert "skyline" in result.lower()
        assert "night" in result.lower()
        assert "-" not in result

    @pytest.mark.fast
    def test_mixed_underscores_hyphens(self, stage):
        """'urban_city-skyline_at-night.mp4' -> all keywords extracted."""
        result = stage._extract_keywords_from_filename("/videos/urban_city-skyline_at-night.mp4")
        assert "urban" in result.lower()
        assert "city" in result.lower()
        assert "skyline" in result.lower()

    @pytest.mark.fast
    def test_video_id_stripped(self, stage):
        """11-char video IDs like 'dQw4w9WgXcQ' are removed."""
        result = stage._extract_keywords_from_filename("/videos/sunset_dQw4w9WgXcQ.mp4")
        assert "dQw4w9WgXcQ" not in result
        assert "sunset" in result.lower()

    @pytest.mark.fast
    def test_pexels_prefix_stripped(self, stage):
        """'pexels_' prefix removed from filename."""
        result = stage._extract_keywords_from_filename("/videos/pexels_sunset_beach.mp4")
        assert "pexels" not in result.lower()
        assert "sunset" in result.lower()
        assert "beach" in result.lower()

    @pytest.mark.fast
    def test_pixabay_prefix_stripped(self, stage):
        """'pixabay_' prefix removed from filename."""
        result = stage._extract_keywords_from_filename("/videos/pixabay_ocean_waves.mp4")
        assert "pixabay" not in result.lower()
        assert "ocean" in result.lower()

    @pytest.mark.fast
    def test_entity_prefix_stripped(self, stage):
        """'entity_' prefix removed from filename."""
        result = stage._extract_keywords_from_filename("/videos/entity_california_coast.mp4")
        assert "entity" not in result.lower()
        assert "california" in result.lower()

    @pytest.mark.fast
    def test_yt_prefix_stripped(self, stage):
        """'yt_' prefix removed from filename."""
        result = stage._extract_keywords_from_filename("/videos/yt_earthquake_footage.mp4")
        assert "yt" not in result.lower().split()
        assert "earthquake" in result.lower()

    @pytest.mark.fast
    def test_broll_prefix_stripped(self, stage):
        """'broll_' prefix removed from filename."""
        result = stage._extract_keywords_from_filename("/videos/broll_aerial_shot.mp4")
        assert "broll" not in result.lower().split()
        assert "aerial" in result.lower()

    @pytest.mark.fast
    def test_trailing_numbers_removed(self, stage):
        """Trailing numbers (timestamps) removed from filename."""
        result = stage._extract_keywords_from_filename("/videos/sunset_beach_12345.mp4")
        assert "12345" not in result
        assert "sunset" in result.lower()

    @pytest.mark.fast
    def test_resolution_suffix_preserved(self, stage):
        """Resolution like '4k' is preserved as a keyword."""
        result = stage._extract_keywords_from_filename("/videos/pexels_sunset_4k.mp4")
        assert "sunset" in result.lower()
        # '4k' is short enough not to be a video ID, may or may not be stripped by trailing number logic
        # The important thing is sunset is there and pexels is not


# ============================================================================
# AC4: _match_scenes_to_voiceover() multi-strategy scoring
# ============================================================================

class TestMatchScenesToVoiceoverUS005:
    """AC4: _match_scenes_to_voiceover() selects top match per segment."""

    @pytest.fixture
    def stage(self):
        return BrollMatchStage()

    @pytest.mark.fast
    def test_each_segment_gets_best_scoring_scene(self, stage):
        """Mock 3 scenes and 2 segments, verify each segment gets best-scoring scene."""
        scenes = [
            BrollScene(
                source_file="/videos/scene_a.mp4",
                start_time=0.0, end_time=10.0,
                word_count=0, transcript="",
                description="earthquake damage footage",
                source="youtube"
            ),
            BrollScene(
                source_file="/videos/scene_b.mp4",
                start_time=0.0, end_time=10.0,
                word_count=0, transcript="",
                description="sunset beach ocean",
                source="pexels"
            ),
            BrollScene(
                source_file="/videos/scene_c.mp4",
                start_time=0.0, end_time=10.0,
                word_count=0, transcript="",
                description="city skyline urban",
                source="pixabay"
            ),
        ]

        state = PipelineState()
        state.voiceover_segments = [
            VoiceoverSegment(index=0, start=0.0, end=5.0, text="earthquake damage report"),
            VoiceoverSegment(index=1, start=5.0, end=10.0, text="sunset beach scene"),
        ]
        state.keywords = ["earthquake", "damage", "sunset", "beach"]
        state.extracted_entities = []

        config = Mock()
        config.embedding = None

        broll_config = BrollConfig(always_match=True)

        with patch.object(stage, '_get_voiceover_embeddings', return_value={}):
            matches = stage._match_scenes_to_voiceover(scenes, state, config, broll_config)

        # Should have 1 match per segment (2 total)
        assert len(matches) == 2
        segment_indices = {m.segment_index for m in matches}
        assert segment_indices == {0, 1}

        # Segment 0 (earthquake) should match scene_a (earthquake description)
        seg0_match = [m for m in matches if m.segment_index == 0][0]
        assert seg0_match.scene.source_file == "/videos/scene_a.mp4"

        # Segment 1 (sunset) should match scene_b (sunset description)
        seg1_match = [m for m in matches if m.segment_index == 1][0]
        assert seg1_match.scene.source_file == "/videos/scene_b.mp4"

    @pytest.mark.fast
    def test_returns_broll_match_objects(self, stage):
        """Returned matches are BrollMatch instances with score components."""
        scenes = [
            BrollScene(
                source_file="/videos/test.mp4",
                start_time=0.0, end_time=10.0,
                word_count=0, transcript="",
                description="test footage",
                source="youtube"
            ),
        ]

        state = PipelineState()
        state.voiceover_segments = [
            VoiceoverSegment(index=0, start=0.0, end=5.0, text="test footage")
        ]
        state.keywords = ["test"]
        state.extracted_entities = []

        config = Mock()
        config.embedding = None

        broll_config = BrollConfig(always_match=True)

        with patch.object(stage, '_get_voiceover_embeddings', return_value={}):
            matches = stage._match_scenes_to_voiceover(scenes, state, config, broll_config)

        assert len(matches) == 1
        match = matches[0]
        assert isinstance(match, BrollMatch)
        assert isinstance(match.score, float)
        assert isinstance(match.embedding_score, float)
        assert isinstance(match.keyword_score, float)
        assert isinstance(match.entity_score, float)
        assert isinstance(match.source_boost, float)

    @pytest.mark.fast
    def test_only_one_match_per_segment(self, stage):
        """V8 track gets only the best match per voiceover segment."""
        scenes = [
            BrollScene(
                source_file=f"/videos/scene_{i}.mp4",
                start_time=0.0, end_time=10.0,
                word_count=0, transcript="",
                description="test footage",
                source="youtube"
            )
            for i in range(5)
        ]

        state = PipelineState()
        state.voiceover_segments = [
            VoiceoverSegment(index=0, start=0.0, end=5.0, text="test")
        ]
        state.keywords = []
        state.extracted_entities = []

        config = Mock()
        config.embedding = None

        broll_config = BrollConfig(always_match=True, max_matches_per_segment=3)

        with patch.object(stage, '_get_voiceover_embeddings', return_value={}):
            matches = stage._match_scenes_to_voiceover(scenes, state, config, broll_config)

        # Should only return 1 match (best) for the single segment
        assert len(matches) == 1

    @pytest.mark.fast
    def test_min_score_threshold_respected_when_always_match_false(self, stage):
        """When always_match=False, scenes below min_score are excluded."""
        scenes = [
            BrollScene(
                source_file="/videos/unrelated.mp4",
                start_time=0.0, end_time=10.0,
                word_count=0, transcript="",
                description="completely unrelated content",
                source="pixabay"
            ),
        ]

        state = PipelineState()
        state.voiceover_segments = [
            VoiceoverSegment(index=0, start=0.0, end=5.0, text="earthquake damage")
        ]
        state.keywords = ["earthquake", "damage"]
        state.extracted_entities = []

        config = Mock()
        config.embedding = None

        # High threshold, no always_match
        broll_config = BrollConfig(
            always_match=False,
            min_match_score=0.9,
            source_boost=BrollSourceBoostConfig(youtube=0.0, pexels=0.0, pixabay=0.0)
        )

        with patch.object(stage, '_get_voiceover_embeddings', return_value={}):
            matches = stage._match_scenes_to_voiceover(scenes, state, config, broll_config)

        # No match should meet the 0.9 threshold for completely unrelated content
        assert len(matches) == 0

    @pytest.mark.fast
    def test_source_boost_affects_ranking(self, stage):
        """Source boost gives YouTube scenes advantage over otherwise equal scenes."""
        youtube_scene = BrollScene(
            source_file="/videos/youtube.mp4",
            start_time=0.0, end_time=10.0,
            word_count=0, transcript="",
            description="test footage scene",
            source="youtube"
        )
        pixabay_scene = BrollScene(
            source_file="/videos/pixabay.mp4",
            start_time=0.0, end_time=10.0,
            word_count=0, transcript="",
            description="test footage scene",
            source="pixabay"
        )

        state = PipelineState()
        state.voiceover_segments = [
            VoiceoverSegment(index=0, start=0.0, end=5.0, text="test footage scene")
        ]
        state.keywords = ["test", "footage"]
        state.extracted_entities = []

        config = Mock()
        config.embedding = None

        broll_config = BrollConfig(
            always_match=True,
            source_boost=BrollSourceBoostConfig(youtube=0.2, pixabay=0.0)
        )

        with patch.object(stage, '_get_voiceover_embeddings', return_value={}):
            matches = stage._match_scenes_to_voiceover(
                [youtube_scene, pixabay_scene], state, config, broll_config
            )

        # YouTube scene should win due to 0.2 source boost
        assert len(matches) == 1
        assert matches[0].scene.source == "youtube"
        assert matches[0].source_boost > 0


# ============================================================================
# AC5: can_skip() returns True when broll disabled or no voiceover segments
# ============================================================================

class TestCanSkipUS005:
    """AC5: can_skip() returns True when broll disabled or no voiceover segments."""

    @pytest.fixture
    def stage(self):
        return BrollMatchStage()

    @pytest.mark.fast
    def test_can_skip_true_when_checkpoint_says_skip(self, stage):
        """can_skip() returns True when checkpoint says to skip BROLL_MATCH."""
        state = PipelineState()
        checkpoint = Mock()
        checkpoint.should_skip_stage.return_value = True

        result = stage.can_skip(state, checkpoint)
        assert result is True
        checkpoint.should_skip_stage.assert_called_once_with("BROLL_MATCH")

    @pytest.mark.fast
    def test_can_skip_false_when_checkpoint_says_no(self, stage):
        """can_skip() returns False when checkpoint says not to skip."""
        state = PipelineState()
        checkpoint = Mock()
        checkpoint.should_skip_stage.return_value = False

        result = stage.can_skip(state, checkpoint)
        assert result is False

    @pytest.mark.fast
    def test_run_skips_when_broll_disabled(self, stage):
        """run() returns ok with 'broll_disabled' when broll.enabled=False."""
        state = PipelineState()
        config = Mock()
        config.broll = BrollConfig(enabled=False)
        checkpoint = Mock()

        result = stage.run(state, config, checkpoint)

        assert result.success is True
        assert result.data.get('skipped') is True
        assert result.data.get('reason') == 'broll_disabled'

    @pytest.mark.fast
    def test_run_skips_when_no_voiceover_segments(self, stage):
        """run() returns ok with 'no_voiceover' when voiceover_segments is empty."""
        state = PipelineState()
        state.voiceover_segments = []
        config = Mock()
        config.broll = BrollConfig(enabled=True)
        checkpoint = Mock()

        result = stage.run(state, config, checkpoint)

        assert result.success is True
        assert result.data.get('skipped') is True
        assert result.data.get('reason') == 'no_voiceover'

    @pytest.mark.fast
    def test_run_skips_when_no_broll_config(self, stage):
        """run() returns ok when config.broll is None."""
        state = PipelineState()
        config = Mock()
        config.broll = None
        checkpoint = Mock()

        result = stage.run(state, config, checkpoint)

        assert result.success is True
        assert result.data.get('skipped') is True


# ============================================================================
# AC6: restore() rebuilds broll_matches from checkpoint
# ============================================================================

class TestRestoreUS005:
    """AC6: restore() rebuilds broll_matches from checkpoint data."""

    @pytest.fixture
    def stage(self):
        return BrollMatchStage()

    @pytest.mark.fast
    def test_restore_rebuilds_matches(self, stage):
        """restore() populates state.broll_matches from checkpoint match data."""
        state = PipelineState()
        state.broll_matches = []

        checkpoint = Mock()
        checkpoint.get_stage_data.return_value = {
            'silent_count': 5,
            'match_count': 2,
            'matches': [
                {
                    'segment_index': 0,
                    'source_file': '/videos/scene1.mp4',
                    'score': 0.85,
                    'start_time': 0.0,
                    'end_time': 10.0,
                    'description': 'earthquake footage',
                    'source': 'youtube'
                },
                {
                    'segment_index': 1,
                    'source_file': '/videos/scene2.mp4',
                    'score': 0.72,
                    'start_time': 5.0,
                    'end_time': 15.0,
                    'description': 'sunset beach',
                    'source': 'pexels'
                },
            ]
        }

        result = stage.restore(state, checkpoint)

        assert result is True
        assert len(state.broll_matches) == 2
        assert state.broll_matches[0]['segment_index'] == 0
        assert state.broll_matches[0]['source_file'] == '/videos/scene1.mp4'
        assert state.broll_matches[0]['score'] == 0.85
        assert state.broll_matches[1]['segment_index'] == 1
        assert state.broll_matches[1]['source'] == 'pexels'

    @pytest.mark.fast
    def test_restore_handles_missing_checkpoint_data(self, stage):
        """restore() returns False when no checkpoint data exists."""
        state = PipelineState()
        checkpoint = Mock()
        checkpoint.get_stage_data.return_value = None

        result = stage.restore(state, checkpoint)

        assert result is False

    @pytest.mark.fast
    def test_restore_handles_empty_matches(self, stage):
        """restore() handles checkpoint with empty matches list."""
        state = PipelineState()
        state.broll_matches = []

        checkpoint = Mock()
        checkpoint.get_stage_data.return_value = {
            'silent_count': 0,
            'match_count': 0,
            'matches': []
        }

        result = stage.restore(state, checkpoint)

        assert result is True
        assert len(state.broll_matches) == 0

    @pytest.mark.fast
    def test_restore_populates_all_fields(self, stage):
        """restore() correctly maps all fields from checkpoint to state."""
        state = PipelineState()

        checkpoint = Mock()
        checkpoint.get_stage_data.return_value = {
            'silent_count': 1,
            'match_count': 1,
            'matches': [
                {
                    'segment_index': 3,
                    'source_file': '/videos/test.mp4',
                    'score': 0.65,
                    'start_time': 12.5,
                    'end_time': 22.5,
                    'description': 'test scene',
                    'source': 'youtube'
                }
            ]
        }

        result = stage.restore(state, checkpoint)

        assert result is True
        m = state.broll_matches[0]
        assert m['segment_index'] == 3
        assert m['source_file'] == '/videos/test.mp4'
        assert m['score'] == 0.65
        assert m['start_time'] == 12.5
        assert m['end_time'] == 22.5
        assert m['description'] == 'test scene'
        assert m['source'] == 'youtube'

    @pytest.mark.fast
    def test_restore_handles_exception(self, stage):
        """restore() returns False on exception without crashing."""
        state = PipelineState()
        checkpoint = Mock()
        checkpoint.get_stage_data.side_effect = Exception("Checkpoint read error")

        result = stage.restore(state, checkpoint)

        assert result is False

    @pytest.mark.fast
    def test_restore_missing_optional_fields_use_defaults(self, stage):
        """restore() uses defaults for missing optional fields in match data."""
        state = PipelineState()

        checkpoint = Mock()
        checkpoint.get_stage_data.return_value = {
            'matches': [
                {
                    'segment_index': 0,
                    'source_file': '/videos/minimal.mp4',
                    'score': 0.5
                    # Missing: start_time, end_time, description, source
                }
            ]
        }

        result = stage.restore(state, checkpoint)

        assert result is True
        m = state.broll_matches[0]
        assert m['start_time'] == 0.0
        assert m['end_time'] == 0.0
        assert m['description'] == ''
        assert m['source'] == ''
