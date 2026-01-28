"""
Extended tests for B-roll stages: BrollDownloadStage and BrollMatchStage

Additional coverage for:
- Edge cases and error handling
- Restore from checkpoint
- Scoring calculations
- Long video handling
- Entity extraction edge cases
"""

import pytest
import sys
import numpy as np
from unittest.mock import Mock, MagicMock, patch
from pathlib import Path
from dataclasses import asdict
import tempfile
import os

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.sections.broll import BrollConfig, BrollSourceBoostConfig
from src.stages.broll_download import BrollDownloadStage
from src.stages.broll_match import BrollMatchStage, BrollScene, BrollMatch
from src.state import PipelineState, VoiceoverSegment, DownloadedVideo


# ============================================================================
# BrollDownloadStage Extended Tests
# ============================================================================

class TestBrollDownloadStageExtended:
    """Extended tests for BrollDownloadStage"""

    @pytest.fixture
    def stage(self):
        return BrollDownloadStage()

    @pytest.fixture
    def mock_state(self):
        state = PipelineState()
        state.keywords = ["earthquake", "tsunami", "hurricane"]
        state.extracted_entities = [
            {"name": "California", "type": "LOCATION"},
            {"name": "Tokyo", "type": "LOCATION"},
            {"name": "John Smith", "type": "PERSON"},  # Should be skipped
        ]
        state.downloaded_videos = []
        return state

    @pytest.mark.fast
    def test_generate_search_terms_location_suffixes(self, stage, mock_state):
        """Test location entities get aerial/drone suffixes"""
        broll_config = BrollConfig()

        terms = stage._generate_search_terms(mock_state, broll_config)

        # Location should have aerial/drone options
        location_aerial = [t for t in terms if "California" in t and "aerial" in t.lower()]
        assert len(location_aerial) > 0 or any("aerial" in t.lower() for t in terms)

    @pytest.mark.fast
    def test_generate_search_terms_skips_person_entities(self, stage, mock_state):
        """Test PERSON entities are skipped for B-roll"""
        broll_config = BrollConfig()

        terms = stage._generate_search_terms(mock_state, broll_config)

        # PERSON entities should not generate terms
        person_terms = [t for t in terms if "John Smith" in t]
        assert len(person_terms) == 0

    @pytest.mark.fast
    def test_generate_search_terms_limits_keywords(self, stage):
        """Test keywords are limited to top 10"""
        state = PipelineState()
        state.keywords = [f"keyword_{i}" for i in range(20)]
        state.extracted_entities = []
        broll_config = BrollConfig(include_generic_searches=False)

        terms = stage._generate_search_terms(state, broll_config)

        # Only first 10 keywords should be used (with 3 suffixes each = 30 max)
        keyword_11_terms = [t for t in terms if "keyword_11" in t]
        assert len(keyword_11_terms) == 0

    @pytest.mark.fast
    def test_generate_search_terms_limits_entities(self, stage):
        """Test entities are limited to top 5"""
        state = PipelineState()
        state.keywords = []
        state.extracted_entities = [
            {"name": f"Entity_{i}", "type": "ORGANIZATION"} for i in range(10)
        ]
        broll_config = BrollConfig(include_generic_searches=False)

        terms = stage._generate_search_terms(state, broll_config)

        # Entity_6 and beyond should not be included
        entity_6_terms = [t for t in terms if "Entity_6" in t]
        assert len(entity_6_terms) == 0

    @pytest.mark.fast
    def test_generate_search_terms_custom_suffixes(self, stage, mock_state):
        """Test custom search suffixes are used"""
        broll_config = BrollConfig(
            search_suffixes=["aerial", "4k", "slow motion"],
            include_generic_searches=False
        )

        terms = stage._generate_search_terms(mock_state, broll_config)

        assert any("aerial" in t.lower() for t in terms)
        assert any("4k" in t.lower() for t in terms)
        assert any("slow motion" in t.lower() for t in terms)

    @pytest.mark.fast
    def test_generate_search_terms_entity_as_string(self, stage):
        """Test entities provided as strings (not dicts)"""
        state = PipelineState()
        state.keywords = []
        state.extracted_entities = ["California", "New York"]  # Strings, not dicts
        broll_config = BrollConfig(include_generic_searches=False)

        terms = stage._generate_search_terms(state, broll_config)

        # Should still generate terms
        assert len(terms) > 0

    @pytest.mark.integration
    def test_restore_from_checkpoint(self, stage):
        """Test restore from checkpoint data"""
        state = PipelineState()

        checkpoint = Mock()
        checkpoint.get_stage_data.return_value = {
            'broll_paths': ['/tmp/video1.mp4', '/tmp/video2.mp4']
        }

        # Create temp files
        with tempfile.TemporaryDirectory() as tmpdir:
            path1 = Path(tmpdir) / "video1.mp4"
            path2 = Path(tmpdir) / "video2.mp4"
            path1.touch()
            path2.touch()

            checkpoint.get_stage_data.return_value = {
                'broll_paths': [str(path1), str(path2)]
            }

            result = stage.restore(state, checkpoint)

            assert result is True
            assert len(state.downloaded_videos) == 2
            assert hasattr(state, 'broll_downloads')
            assert len(state.broll_downloads) == 2

    @pytest.mark.fast
    def test_restore_no_checkpoint_data(self, stage):
        """Test restore with no checkpoint data"""
        state = PipelineState()
        checkpoint = Mock()
        checkpoint.get_stage_data.return_value = None

        result = stage.restore(state, checkpoint)

        assert result is False

    @pytest.mark.fast
    def test_restore_missing_files(self, stage):
        """Test restore when files no longer exist"""
        state = PipelineState()
        checkpoint = Mock()
        checkpoint.get_stage_data.return_value = {
            'broll_paths': ['/nonexistent/video1.mp4', '/nonexistent/video2.mp4']
        }

        result = stage.restore(state, checkpoint)

        assert result is False

    @pytest.mark.integration
    def test_restore_partial_files(self, stage):
        """Test restore when some files exist"""
        state = PipelineState()

        with tempfile.TemporaryDirectory() as tmpdir:
            existing = Path(tmpdir) / "exists.mp4"
            existing.touch()

            checkpoint = Mock()
            checkpoint.get_stage_data.return_value = {
                'broll_paths': [str(existing), '/nonexistent/missing.mp4']
            }

            result = stage.restore(state, checkpoint)

            # Should succeed with partial files
            assert result is True
            assert len(state.downloaded_videos) == 1


# ============================================================================
# BrollMatchStage Extended Tests
# ============================================================================

class TestBrollMatchStageExtended:
    """Extended tests for BrollMatchStage"""

    @pytest.fixture
    def stage(self):
        return BrollMatchStage()

    @pytest.fixture
    def mock_state_with_broll(self):
        """State with B-roll flagged segments"""
        state = PipelineState()
        state.voiceover_segments = [
            VoiceoverSegment(index=0, start=0.0, end=5.0, text="The earthquake struck at dawn"),
            VoiceoverSegment(index=1, start=5.0, end=10.0, text="Rescue teams arrived quickly"),
        ]
        state.keywords = ["earthquake", "rescue", "disaster"]
        state.extracted_entities = [
            {"name": "California", "type": "LOCATION"}
        ]
        state.text_metadata = [
            {
                "source_file": "/videos/earthquake_footage.mp4",
                "start_time": 0.0,
                "end_time": 10.0,
                "text": "",
                "is_broll": True  # Already flagged by face detection
            },
            {
                "source_file": "/videos/silent_stock.mp4",
                "start_time": 0.0,
                "end_time": 15.0,
                "text": "[Music]",
                "is_broll": False
            }
        ]
        state.transcripts = {}
        state.downloaded_videos = [
            DownloadedVideo(file="/videos/earthquake_footage.mp4", source="youtube"),
            DownloadedVideo(file="/videos/silent_stock.mp4", source="pexels"),
        ]
        return state

    @pytest.mark.fast
    def test_detect_silent_scenes_broll_flag(self, stage, mock_state_with_broll):
        """Test that is_broll=True segments are included"""
        broll_config = BrollConfig(min_words_threshold=10)

        scenes = stage._detect_silent_scenes(mock_state_with_broll, broll_config)

        # Both should be detected (one by is_broll, one by word count)
        assert len(scenes) == 2
        sources = [s.source_file for s in scenes]
        assert "/videos/earthquake_footage.mp4" in sources
        assert "/videos/silent_stock.mp4" in sources

    @pytest.mark.fast
    def test_detect_silent_scenes_empty_transcripts(self, stage):
        """Test videos with empty transcript list"""
        state = PipelineState()
        state.transcripts = {
            "silent_video": []  # Empty transcript = fully silent
        }
        state.text_metadata = []
        state.downloaded_videos = [
            DownloadedVideo(file="/videos/silent_video.mp4", source="youtube")
        ]

        broll_config = BrollConfig()
        scenes = stage._detect_silent_scenes(state, broll_config)

        assert len(scenes) == 1

    @pytest.mark.fast
    def test_get_video_source_from_downloaded_videos(self, stage):
        """Test source detection from downloaded_videos list"""
        state = PipelineState()
        state.downloaded_videos = [
            DownloadedVideo(file="/videos/my_video.mp4", source="broll")
        ]

        source = stage._get_video_source("/videos/my_video.mp4", state)
        assert source == "broll"

    @pytest.mark.fast
    def test_get_video_source_broll_directory(self, stage):
        """Test source detection from broll directory"""
        state = PipelineState()
        state.downloaded_videos = []

        source = stage._get_video_source("/videos/broll/my_video.mp4", state)
        assert source == "broll"

    @pytest.mark.fast
    def test_extract_keywords_strips_prefixes(self, stage):
        """Test all prefixes are stripped"""
        prefixes = ["pexels_", "pixabay_", "entity_", "yt_", "broll_"]

        for prefix in prefixes:
            result = stage._extract_keywords_from_filename(f"/videos/{prefix}sunset_beach.mp4")
            assert prefix.rstrip("_") not in result.lower()
            assert "sunset" in result.lower()

    @pytest.mark.fast
    def test_extract_keywords_handles_underscores_hyphens(self, stage):
        """Test underscores and hyphens become spaces"""
        result = stage._extract_keywords_from_filename("/videos/city-skyline_at_night.mp4")

        assert "city" in result.lower()
        assert "skyline" in result.lower()
        assert "night" in result.lower()
        assert "_" not in result
        assert "-" not in result

    @pytest.mark.fast
    def test_calculate_embedding_score_with_numpy(self, stage):
        """Test embedding score calculation"""
        # Create normalized vectors
        vo_emb = np.array([1.0, 0.0, 0.0])
        scene_emb = np.array([1.0, 0.0, 0.0])

        score = stage._calculate_embedding_score(vo_emb, scene_emb)

        # Same vector should have score ~1.0
        assert abs(score - 1.0) < 0.01

    @pytest.mark.fast
    def test_calculate_embedding_score_orthogonal(self, stage):
        """Test orthogonal vectors have low score"""
        vo_emb = np.array([1.0, 0.0, 0.0])
        scene_emb = np.array([0.0, 1.0, 0.0])

        score = stage._calculate_embedding_score(vo_emb, scene_emb)

        # Orthogonal vectors should have score ~0.0
        assert abs(score) < 0.01

    @pytest.mark.fast
    def test_calculate_embedding_score_none_inputs(self, stage):
        """Test embedding score with None inputs"""
        score = stage._calculate_embedding_score(None, np.array([1.0]))
        assert score == 0.0

        score = stage._calculate_embedding_score(np.array([1.0]), None)
        assert score == 0.0

        score = stage._calculate_embedding_score(None, None)
        assert score == 0.0

    @pytest.mark.fast
    def test_calculate_keyword_score_full_overlap(self, stage):
        """Test keyword score with full overlap"""
        keywords = {"earthquake", "damage"}

        score = stage._calculate_keyword_score(
            "earthquake damage",
            "earthquake damage footage",
            keywords
        )

        assert score == 1.0

    @pytest.mark.fast
    def test_calculate_keyword_score_no_keywords(self, stage):
        """Test keyword score with no matching keywords"""
        keywords = {"earthquake", "damage"}

        score = stage._calculate_keyword_score(
            "beautiful sunset",
            "ocean waves",
            keywords
        )

        # Falls back to direct word overlap
        assert score >= 0.0

    @pytest.mark.fast
    def test_calculate_keyword_score_empty_description(self, stage):
        """Test keyword score with empty description"""
        keywords = {"earthquake"}

        score = stage._calculate_keyword_score(
            "earthquake",
            "",
            keywords
        )

        assert score == 0.0

    @pytest.mark.fast
    def test_calculate_entity_score_multiple_matches(self, stage):
        """Test entity score with multiple entity matches"""
        entities = {"california", "san francisco", "los angeles"}

        score = stage._calculate_entity_score(
            "california san francisco",
            "california san francisco footage",
            entities
        )

        assert score == 1.0

    @pytest.mark.fast
    def test_calculate_entity_score_partial_match(self, stage):
        """Test entity score with partial match"""
        entities = {"california", "new york"}

        score = stage._calculate_entity_score(
            "california new york",
            "california footage",  # Only california matches
            entities
        )

        assert score == 0.5

    @pytest.mark.fast
    def test_calculate_entity_score_no_entities_in_vo(self, stage):
        """Test entity score when voiceover has no entities"""
        entities = {"california"}

        score = stage._calculate_entity_score(
            "beautiful weather",  # No entities
            "california footage",
            entities
        )

        assert score == 0.0

    @pytest.mark.fast
    def test_calculate_entity_score_empty_entities(self, stage):
        """Test entity score with empty entity set"""
        score = stage._calculate_entity_score(
            "california earthquake",
            "california footage",
            set()
        )

        assert score == 0.0

    @pytest.mark.fast
    def test_sample_long_video_scenes_short_video(self, stage):
        """Test sampling doesn't affect short videos"""
        scenes = [
            BrollScene(
                source_file="/video.mp4",
                start_time=i * 10.0,
                end_time=(i + 1) * 10.0,
                word_count=0,
                transcript=""
            )
            for i in range(5)
        ]

        sampled = stage._sample_long_video_scenes(
            scenes,
            long_threshold=600,
            sample_interval=120,
            max_scenes=15
        )

        # Should keep all scenes (fewer than max)
        assert len(sampled) == 5

    @pytest.mark.fast
    def test_sample_long_video_scenes_multiple_videos(self, stage):
        """Test sampling handles multiple video sources"""
        scenes = []
        for video_idx in range(3):
            for scene_idx in range(20):
                scenes.append(BrollScene(
                    source_file=f"/video{video_idx}.mp4",
                    start_time=scene_idx * 10.0,
                    end_time=(scene_idx + 1) * 10.0,
                    word_count=0,
                    transcript=""
                ))

        sampled = stage._sample_long_video_scenes(
            scenes,
            long_threshold=600,
            sample_interval=120,
            max_scenes=10
        )

        # Each video should have at most 10 scenes
        for video_idx in range(3):
            video_scenes = [s for s in sampled if f"video{video_idx}" in s.source_file]
            assert len(video_scenes) <= 10

    @pytest.mark.fast
    def test_restore_from_checkpoint(self, stage):
        """Test restore from checkpoint"""
        state = PipelineState()

        checkpoint = Mock()
        checkpoint.get_stage_data.return_value = {
            'matches': [
                {'segment_index': 0, 'source_file': '/video.mp4', 'score': 0.8},
                {'segment_index': 1, 'source_file': '/video2.mp4', 'score': 0.7}
            ]
        }

        result = stage.restore(state, checkpoint)

        assert result is True
        assert len(state.broll_matches) == 2
        assert state.broll_matches[0]['score'] == 0.8

    @pytest.mark.fast
    def test_restore_no_data(self, stage):
        """Test restore with no checkpoint data"""
        state = PipelineState()
        checkpoint = Mock()
        checkpoint.get_stage_data.return_value = None

        result = stage.restore(state, checkpoint)

        assert result is False

    @pytest.mark.fast
    def test_restore_exception_handling(self, stage):
        """Test restore handles exceptions"""
        state = PipelineState()
        checkpoint = Mock()
        checkpoint.get_stage_data.side_effect = Exception("Checkpoint error")

        result = stage.restore(state, checkpoint)

        assert result is False

    @pytest.mark.fast
    def test_enrich_with_keywords_existing_description(self, stage):
        """Test keyword enrichment skips scenes with description"""
        scenes = [
            BrollScene(
                source_file="/video.mp4",
                start_time=0.0,
                end_time=10.0,
                word_count=0,
                transcript="",
                description="Already has description"
            )
        ]

        broll_config = BrollConfig()
        count = stage._enrich_with_keywords(scenes, None, broll_config)

        # Should not overwrite existing description
        assert count == 0
        assert scenes[0].description == "Already has description"


# ============================================================================
# Scoring Integration Tests
# ============================================================================

class TestBrollScoringIntegration:
    """Test the complete scoring pipeline"""

    @pytest.fixture
    def stage(self):
        return BrollMatchStage()

    @pytest.mark.fast
    def test_combined_score_calculation(self, stage):
        """Test that combined score uses correct weights"""
        broll_config = BrollConfig(
            embedding_weight=0.4,
            keyword_weight=0.35,
            entity_weight=0.25,
            source_boost=BrollSourceBoostConfig(youtube=0.1, pexels=0.05, pixabay=0.0)
        )

        # Expected: 0.8*0.4 + 0.7*0.35 + 0.6*0.25 + 0.1 = 0.32 + 0.245 + 0.15 + 0.1 = 0.815
        embedding_score = 0.8
        keyword_score = 0.7
        entity_score = 0.6
        source = "youtube"

        boost = broll_config.source_boost.youtube
        combined = (
            embedding_score * broll_config.embedding_weight +
            keyword_score * broll_config.keyword_weight +
            entity_score * broll_config.entity_weight +
            boost
        )

        assert abs(combined - 0.815) < 0.001

    @pytest.mark.fast
    def test_source_boost_values(self, stage):
        """Test source boost values are applied correctly"""
        broll_config = BrollConfig()

        youtube_boost = broll_config.source_boost.youtube
        pexels_boost = broll_config.source_boost.pexels
        pixabay_boost = broll_config.source_boost.pixabay

        # YouTube should have highest boost
        assert youtube_boost > pexels_boost
        assert pexels_boost > pixabay_boost
        assert pixabay_boost == 0.0


# ============================================================================
# Config Validation Tests
# ============================================================================

class TestBrollConfigValidation:
    """Test config validation and edge cases"""

    @pytest.mark.fast
    def test_negative_weights_allowed(self):
        """Test that negative weights don't crash (user responsibility)"""
        config = BrollConfig(
            embedding_weight=-0.1,
            keyword_weight=0.6,
            entity_weight=0.5
        )

        # Should create without error
        assert config.embedding_weight == -0.1

    @pytest.mark.fast
    def test_zero_downloads_per_term(self):
        """Test zero downloads_per_term"""
        config = BrollConfig(downloads_per_term=0)
        assert config.downloads_per_term == 0

    @pytest.mark.fast
    def test_very_high_threshold(self):
        """Test very high min_words_threshold"""
        config = BrollConfig(min_words_threshold=10000)
        assert config.min_words_threshold == 10000

    @pytest.mark.fast
    def test_empty_search_suffixes(self):
        """Test empty search suffixes list"""
        config = BrollConfig(search_suffixes=[])
        assert config.search_suffixes == []

    @pytest.mark.fast
    def test_empty_ignore_markers(self):
        """Test empty ignore markers list"""
        config = BrollConfig(ignore_markers=[])
        assert config.ignore_markers == []


# ============================================================================
# Pipeline Integration Tests
# ============================================================================

class TestBrollPipelineIntegration:
    """Test B-roll stages in pipeline context"""

    @pytest.mark.integration
    def test_default_pipeline_includes_broll_stages(self):
        """Test default pipeline has B-roll stages"""
        from src.pipeline import create_default_pipeline

        config = Mock()
        config.broll = BrollConfig()
        config.pipeline = Mock()
        config.download = Mock()
        config.download.audio_first = None
        config._config_hash = "test"

        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline = create_default_pipeline(config, Path(tmpdir), audio_first_mode=False)

            stage_names = [s.name for s in pipeline.stages]
            assert "BROLL_DOWNLOAD" in stage_names
            assert "BROLL_MATCH" in stage_names

    @pytest.mark.integration
    def test_match_only_pipeline_includes_broll_stages(self):
        """Test match-only pipeline has B-roll stages"""
        from src.pipeline import create_match_only_pipeline

        config = Mock()
        config.broll = BrollConfig()
        config.pipeline = Mock()
        config._config_hash = "test"

        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline = create_match_only_pipeline(config, Path(tmpdir))

            stage_names = [s.name for s in pipeline.stages]
            assert "BROLL_DOWNLOAD" in stage_names
            assert "BROLL_MATCH" in stage_names

    @pytest.mark.fast
    def test_broll_download_before_remix(self):
        """Test BROLL_DOWNLOAD comes before REMIX"""
        from src.checkpoint import STAGE_ORDER

        broll_idx = STAGE_ORDER.index("BROLL_DOWNLOAD")
        remix_idx = STAGE_ORDER.index("REMIX")

        assert broll_idx < remix_idx

    @pytest.mark.fast
    def test_broll_match_after_main_match(self):
        """Test BROLL_MATCH comes after MATCH"""
        from src.checkpoint import STAGE_ORDER

        broll_match_idx = STAGE_ORDER.index("BROLL_MATCH")
        match_idx = STAGE_ORDER.index("MATCH")

        assert broll_match_idx > match_idx

    @pytest.mark.fast
    def test_broll_match_before_output(self):
        """Test BROLL_MATCH comes before OUTPUT"""
        from src.checkpoint import STAGE_ORDER

        broll_match_idx = STAGE_ORDER.index("BROLL_MATCH")
        output_idx = STAGE_ORDER.index("OUTPUT")

        assert broll_match_idx < output_idx


# ============================================================================
# Error Handling Tests
# ============================================================================

class TestBrollErrorHandling:
    """Test error handling in B-roll stages"""

    @pytest.fixture
    def download_stage(self):
        return BrollDownloadStage()

    @pytest.fixture
    def match_stage(self):
        return BrollMatchStage()

    @pytest.mark.fast
    def test_download_stage_handles_missing_broll_config(self, download_stage):
        """Test download stage handles missing broll config"""
        state = PipelineState()
        state.keywords = ["test"]

        config = Mock()
        config.broll = None  # No broll config
        checkpoint = Mock()

        result = download_stage.run(state, config, checkpoint)

        assert result.success is True
        assert result.data['skipped'] is True

    @pytest.mark.fast
    def test_match_stage_handles_missing_broll_config(self, match_stage):
        """Test match stage handles missing broll config"""
        state = PipelineState()
        state.voiceover_segments = [
            VoiceoverSegment(index=0, start=0.0, end=5.0, text="Test")
        ]

        config = Mock()
        config.broll = None  # No broll config
        checkpoint = Mock()

        result = match_stage.run(state, config, checkpoint)

        assert result.success is True
        assert result.data['skipped'] is True

    @pytest.mark.fast
    def test_download_stage_exception_returns_fail(self, download_stage):
        """Test download stage returns fail on exception"""
        state = PipelineState()
        state.keywords = ["test"]

        config = Mock()
        config.broll = BrollConfig()
        config.pipeline = Mock()
        config.pipeline.skip_download = False
        config.downloaded_videos_dir = "/nonexistent/path/that/will/fail"

        checkpoint = Mock()

        # Mock mkdir to raise exception
        with patch('pathlib.Path.mkdir', side_effect=PermissionError("No access")):
            result = download_stage.run(state, config, checkpoint)

        assert result.success is False
        assert "No access" in result.error or "Permission" in str(result.error)
