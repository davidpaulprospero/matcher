"""
Tests for B-roll stages: BrollDownloadStage and BrollMatchStage

Tests:
- BrollConfig and BrollSourceBoostConfig dataclasses
- BrollDownloadStage search term generation and download logic
- BrollMatchStage silent detection and matching
"""

import pytest
import sys
from unittest.mock import Mock, MagicMock, patch
from pathlib import Path
from dataclasses import asdict

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.sections.broll import BrollConfig, BrollSourceBoostConfig
from src.stages.broll_download import BrollDownloadStage
from src.stages.broll_match import BrollMatchStage, BrollScene, BrollMatch
from src.state import PipelineState, VoiceoverSegment


# ============================================================================
# BrollConfig Tests
# ============================================================================

class TestBrollSourceBoostConfig:
    """Test BrollSourceBoostConfig dataclass"""

    def test_default_initialization(self):
        """Test default source boost values"""
        config = BrollSourceBoostConfig()

        assert config.youtube == 0.1
        assert config.pexels == 0.05
        assert config.pixabay == 0.0

    def test_custom_values(self):
        """Test with custom boost values"""
        config = BrollSourceBoostConfig(
            youtube=0.2,
            pexels=0.1,
            pixabay=0.05
        )

        assert config.youtube == 0.2
        assert config.pexels == 0.1
        assert config.pixabay == 0.05

    def test_zero_boosts(self):
        """Test all zero boosts"""
        config = BrollSourceBoostConfig(
            youtube=0.0,
            pexels=0.0,
            pixabay=0.0
        )

        assert config.youtube == 0.0
        assert config.pexels == 0.0
        assert config.pixabay == 0.0


class TestBrollConfig:
    """Test BrollConfig dataclass"""

    def test_default_initialization(self):
        """Test default B-roll config values"""
        config = BrollConfig()

        # Master enable
        assert config.enabled is True

        # Download settings
        assert config.download_enabled is True
        assert config.downloads_per_term == 3
        assert config.max_total_downloads == 30
        assert config.duration_tier == "short"
        assert config.include_generic_searches is True
        assert config.search_suffixes == ["b-roll", "footage", "cinematic"]

        # Detection settings
        assert config.min_words_threshold == 10
        assert config.ignore_markers == ["[Music]", "[Applause]", "[Silence]"]

        # Vision settings
        assert config.vision_enabled is True
        assert config.vision_prompt == "detailed"
        assert config.frames_per_scene == 3
        assert config.cache_descriptions is True
        assert config.vision_fallback_to_keywords is True

        # Scoring weights
        assert config.embedding_weight == 0.4
        assert config.keyword_weight == 0.35
        assert config.entity_weight == 0.25

        # Matching behavior
        assert config.min_match_score == 0.3
        assert config.always_match is True
        assert config.max_matches_per_segment == 3

        # Long video handling
        assert config.long_video_threshold == 600
        assert config.sample_interval_seconds == 120
        assert config.max_scenes_per_video == 15

    def test_scoring_weights_sum(self):
        """Test that default scoring weights sum to 1.0"""
        config = BrollConfig()
        total = config.embedding_weight + config.keyword_weight + config.entity_weight
        assert abs(total - 1.0) < 0.001

    def test_custom_values(self):
        """Test with custom config values"""
        config = BrollConfig(
            enabled=False,
            downloads_per_term=5,
            min_words_threshold=5,
            embedding_weight=0.5,
            keyword_weight=0.3,
            entity_weight=0.2
        )

        assert config.enabled is False
        assert config.downloads_per_term == 5
        assert config.min_words_threshold == 5
        assert config.embedding_weight == 0.5

    def test_post_init_dict_conversion(self):
        """Test __post_init__ converts dict to BrollSourceBoostConfig"""
        config = BrollConfig(
            source_boost={'youtube': 0.15, 'pexels': 0.08, 'pixabay': 0.02}
        )

        assert isinstance(config.source_boost, BrollSourceBoostConfig)
        assert config.source_boost.youtube == 0.15
        assert config.source_boost.pexels == 0.08
        assert config.source_boost.pixabay == 0.02

    def test_source_boost_dataclass(self):
        """Test source_boost as BrollSourceBoostConfig"""
        boost = BrollSourceBoostConfig(youtube=0.2, pexels=0.1, pixabay=0.05)
        config = BrollConfig(source_boost=boost)

        assert config.source_boost.youtube == 0.2
        assert config.source_boost.pexels == 0.1

    def test_custom_search_suffixes(self):
        """Test custom search suffixes"""
        config = BrollConfig(
            search_suffixes=["aerial", "drone", "timelapse"]
        )

        assert "aerial" in config.search_suffixes
        assert "b-roll" not in config.search_suffixes

    def test_custom_ignore_markers(self):
        """Test custom ignore markers"""
        config = BrollConfig(
            ignore_markers=["[Music]", "[Laughter]", "[Background noise]"]
        )

        assert "[Laughter]" in config.ignore_markers
        assert "[Silence]" not in config.ignore_markers


# ============================================================================
# BrollDownloadStage Tests
# ============================================================================

class TestBrollDownloadStage:
    """Test BrollDownloadStage"""

    @pytest.fixture
    def stage(self):
        """Create BrollDownloadStage instance"""
        return BrollDownloadStage()

    @pytest.fixture
    def mock_state(self):
        """Create mock pipeline state"""
        state = PipelineState()
        state.keywords = ["earthquake", "tsunami", "disaster relief"]
        state.extracted_entities = [
            {"name": "California", "type": "LOCATION"},
            {"name": "FEMA", "type": "ORGANIZATION"}
        ]
        return state

    @pytest.fixture
    def mock_config(self):
        """Create mock config"""
        config = Mock()
        config.broll = BrollConfig()
        config.pipeline = Mock()
        config.pipeline.skip_download = False
        config.downloaded_videos_dir = "/tmp/videos"
        return config

    @pytest.fixture
    def mock_checkpoint(self):
        """Create mock checkpoint manager"""
        checkpoint = Mock()
        checkpoint.should_skip_stage.return_value = False
        checkpoint.get_stage_data.return_value = None
        return checkpoint

    def test_stage_attributes(self, stage):
        """Test stage name and description"""
        assert stage.name == "BROLL_DOWNLOAD"
        assert "B-roll" in stage.description

    def test_generate_search_terms_keywords(self, stage, mock_state):
        """Test search term generation from keywords"""
        broll_config = BrollConfig()

        terms = stage._generate_search_terms(mock_state, broll_config)

        # Should have keyword + suffix combinations
        assert "earthquake b-roll" in terms
        assert "earthquake footage" in terms
        assert "tsunami cinematic" in terms

    def test_generate_search_terms_entities(self, stage, mock_state):
        """Test search term generation from entities"""
        broll_config = BrollConfig()

        terms = stage._generate_search_terms(mock_state, broll_config)

        # Should have location entity terms
        assert any("California" in t for t in terms)

    def test_generate_search_terms_generic(self, stage, mock_state):
        """Test generic search terms are included"""
        broll_config = BrollConfig(include_generic_searches=True)

        terms = stage._generate_search_terms(mock_state, broll_config)

        # Should include generic terms
        assert "stock footage compilation" in terms
        assert "cinematic footage 4k" in terms

    def test_generate_search_terms_no_generic(self, stage, mock_state):
        """Test generic searches can be disabled"""
        broll_config = BrollConfig(include_generic_searches=False)

        terms = stage._generate_search_terms(mock_state, broll_config)

        # Should NOT include generic terms
        assert "stock footage compilation" not in terms

    def test_generate_search_terms_deduplication(self, stage, mock_state):
        """Test duplicate terms are removed"""
        broll_config = BrollConfig()

        terms = stage._generate_search_terms(mock_state, broll_config)

        # No duplicates
        assert len(terms) == len(set(t.lower() for t in terms))

    def test_generate_search_terms_empty_keywords(self, stage):
        """Test with no keywords"""
        state = PipelineState()
        state.keywords = []
        state.extracted_entities = []
        broll_config = BrollConfig(include_generic_searches=False)

        terms = stage._generate_search_terms(state, broll_config)

        assert terms == []

    def test_run_broll_disabled(self, stage, mock_state, mock_checkpoint):
        """Test stage skips when broll disabled"""
        config = Mock()
        config.broll = BrollConfig(enabled=False)

        result = stage.run(mock_state, config, mock_checkpoint)

        assert result.success is True
        assert result.data['skipped'] is True
        assert result.data['reason'] == 'broll_disabled'

    def test_run_download_disabled(self, stage, mock_state, mock_checkpoint):
        """Test stage skips when download disabled"""
        config = Mock()
        config.broll = BrollConfig(enabled=True, download_enabled=False)

        result = stage.run(mock_state, config, mock_checkpoint)

        assert result.success is True
        assert result.data['skipped'] is True
        assert result.data['reason'] == 'download_disabled'

    def test_run_skip_download_enabled(self, stage, mock_state, mock_checkpoint):
        """Test stage skips when pipeline.skip_download is true"""
        config = Mock()
        config.broll = BrollConfig()
        config.pipeline = Mock()
        config.pipeline.skip_download = True

        result = stage.run(mock_state, config, mock_checkpoint)

        assert result.success is True
        assert result.data['skipped'] is True
        assert result.data['reason'] == 'skip_download_enabled'

    def test_run_no_keywords(self, stage, mock_checkpoint):
        """Test stage skips when no keywords"""
        state = PipelineState()
        state.keywords = []

        config = Mock()
        config.broll = BrollConfig()
        config.pipeline = Mock()
        config.pipeline.skip_download = False

        result = stage.run(state, config, mock_checkpoint)

        assert result.success is True
        assert result.data['skipped'] is True
        assert result.data['reason'] == 'no_keywords'

    def test_can_skip(self, stage, mock_state, mock_checkpoint):
        """Test can_skip checks checkpoint"""
        mock_checkpoint.should_skip_stage.return_value = True

        assert stage.can_skip(mock_state, mock_checkpoint) is True
        mock_checkpoint.should_skip_stage.assert_called_with("BROLL_DOWNLOAD")

    def test_validate_inputs(self, stage, mock_state, mock_config):
        """Test validate_inputs is optional stage"""
        result = stage.validate_inputs(mock_state, mock_config)
        assert result is None


# ============================================================================
# BrollMatchStage Tests
# ============================================================================

class TestBrollScene:
    """Test BrollScene dataclass"""

    def test_broll_scene_creation(self):
        """Test BrollScene creation"""
        scene = BrollScene(
            source_file="/video.mp4",
            start_time=10.0,
            end_time=20.0,
            word_count=5,
            transcript="[Music]"
        )

        assert scene.source_file == "/video.mp4"
        assert scene.start_time == 10.0
        assert scene.end_time == 20.0
        assert scene.word_count == 5
        assert scene.description == ""
        assert scene.embedding is None
        assert scene.source == ""

    def test_broll_scene_with_description(self):
        """Test BrollScene with description"""
        scene = BrollScene(
            source_file="/video.mp4",
            start_time=0.0,
            end_time=10.0,
            word_count=0,
            transcript="",
            description="Aerial view of city skyline",
            source="youtube"
        )

        assert scene.description == "Aerial view of city skyline"
        assert scene.source == "youtube"


class TestBrollMatch:
    """Test BrollMatch dataclass"""

    def test_broll_match_creation(self):
        """Test BrollMatch creation"""
        scene = BrollScene(
            source_file="/video.mp4",
            start_time=0.0,
            end_time=10.0,
            word_count=0,
            transcript=""
        )

        match = BrollMatch(
            segment_index=0,
            scene=scene,
            score=0.75,
            embedding_score=0.8,
            keyword_score=0.7,
            entity_score=0.6,
            source_boost=0.1
        )

        assert match.segment_index == 0
        assert match.score == 0.75
        assert match.embedding_score == 0.8
        assert match.source_boost == 0.1


class TestBrollMatchStage:
    """Test BrollMatchStage"""

    @pytest.fixture
    def stage(self):
        """Create BrollMatchStage instance"""
        return BrollMatchStage()

    @pytest.fixture
    def mock_state(self):
        """Create mock pipeline state with voiceover and transcripts"""
        state = PipelineState()
        state.voiceover_segments = [
            VoiceoverSegment(index=0, start=0.0, end=5.0, text="The earthquake caused massive destruction"),
            VoiceoverSegment(index=1, start=5.0, end=10.0, text="Relief workers arrived on scene"),
        ]
        state.keywords = ["earthquake", "destruction", "relief"]
        state.extracted_entities = [
            {"name": "California", "type": "LOCATION"}
        ]
        state.transcripts = {
            "video1": [
                {"start_time": 0.0, "end_time": 10.0, "text": ""}
            ]
        }
        state.text_metadata = [
            {
                "source_file": "/video1.mp4",
                "start_time": 0.0,
                "end_time": 10.0,
                "text": "[Music]",
                "is_broll": False
            },
            {
                "source_file": "/video2.mp4",
                "start_time": 0.0,
                "end_time": 15.0,
                "text": "This has more than ten words so it should not be considered silent footage at all",
                "is_broll": False
            }
        ]
        state.downloaded_videos = []
        return state

    @pytest.fixture
    def mock_config(self):
        """Create mock config"""
        config = Mock()
        config.broll = BrollConfig()
        config.download = Mock()
        config.download.audio_first = None
        config.vision = None
        config.embedding = Mock()
        config.embedding.model = "all-MiniLM-L6-v2"
        return config

    @pytest.fixture
    def mock_checkpoint(self):
        """Create mock checkpoint manager"""
        checkpoint = Mock()
        checkpoint.should_skip_stage.return_value = False
        checkpoint.get_stage_data.return_value = None
        return checkpoint

    def test_stage_attributes(self, stage):
        """Test stage name and description"""
        assert stage.name == "BROLL_MATCH"
        assert "V8" in stage.description or "silent" in stage.description.lower()

    def test_detect_silent_scenes_by_word_count(self, stage, mock_state):
        """Test silent detection by word count"""
        broll_config = BrollConfig(min_words_threshold=10)

        scenes = stage._detect_silent_scenes(mock_state, broll_config)

        # Should detect video1 as silent (only has [Music])
        assert len(scenes) >= 1
        silent_files = [s.source_file for s in scenes]
        assert "/video1.mp4" in silent_files

    def test_detect_silent_scenes_respects_threshold(self, stage, mock_state):
        """Test threshold is respected"""
        # With very high threshold, more scenes are silent
        broll_config = BrollConfig(min_words_threshold=100)
        scenes = stage._detect_silent_scenes(mock_state, broll_config)
        high_count = len(scenes)

        # With low threshold, fewer scenes are silent
        broll_config = BrollConfig(min_words_threshold=1)
        scenes = stage._detect_silent_scenes(mock_state, broll_config)
        low_count = len(scenes)

        assert high_count >= low_count

    def test_detect_silent_scenes_ignores_markers(self, stage):
        """Test ignore markers are stripped"""
        state = PipelineState()
        state.text_metadata = [
            {
                "source_file": "/video.mp4",
                "start_time": 0.0,
                "end_time": 10.0,
                "text": "[Music] [Applause] [Silence]",
                "is_broll": False
            }
        ]
        state.transcripts = {}

        broll_config = BrollConfig(
            min_words_threshold=10,
            ignore_markers=["[Music]", "[Applause]", "[Silence]"]
        )

        scenes = stage._detect_silent_scenes(state, broll_config)

        # Should be detected as silent (markers don't count as words)
        assert len(scenes) == 1

    def test_detect_silent_scenes_includes_is_broll(self, stage):
        """Test scenes with is_broll=True are included"""
        state = PipelineState()
        state.text_metadata = [
            {
                "source_file": "/video.mp4",
                "start_time": 0.0,
                "end_time": 10.0,
                "text": "This has plenty of words",
                "is_broll": True  # Marked as B-roll by face detection
            }
        ]
        state.transcripts = {}

        broll_config = BrollConfig(min_words_threshold=10)

        scenes = stage._detect_silent_scenes(state, broll_config)

        # Should be included because is_broll=True
        assert len(scenes) == 1

    def test_get_video_source_youtube(self, stage, mock_state):
        """Test source detection for YouTube"""
        source = stage._get_video_source("/videos/some_video.mp4", mock_state)
        assert source == "youtube"

    def test_get_video_source_pexels(self, stage, mock_state):
        """Test source detection for Pexels"""
        source = stage._get_video_source("/videos/pexels_12345.mp4", mock_state)
        assert source == "pexels"

    def test_get_video_source_pixabay(self, stage, mock_state):
        """Test source detection for Pixabay"""
        source = stage._get_video_source("/videos/pixabay_67890.mp4", mock_state)
        assert source == "pixabay"

    def test_extract_keywords_from_filename(self, stage):
        """Test keyword extraction from filename"""
        # Normal filename
        keywords = stage._extract_keywords_from_filename("/videos/earthquake_damage_city.mp4")
        assert "earthquake" in keywords.lower()
        assert "damage" in keywords.lower()

        # Pexels prefix stripped
        keywords = stage._extract_keywords_from_filename("/videos/pexels_sunset_beach.mp4")
        assert "sunset" in keywords.lower()
        assert "pexels" not in keywords.lower()

    def test_extract_keywords_removes_video_ids(self, stage):
        """Test video IDs are removed from keywords"""
        keywords = stage._extract_keywords_from_filename("/videos/city_dQw4w9WgXcQ.mp4")
        assert "dQw4w9WgXcQ" not in keywords

    def test_should_use_vision_default(self, stage, mock_config):
        """Test Vision API is used by default when available"""
        mock_config.vision = Mock()
        mock_config.vision.enabled = True
        broll_config = BrollConfig(vision_enabled=True)

        result = stage._should_use_vision(mock_config, broll_config)

        assert result is True

    def test_should_use_vision_disabled_in_broll(self, stage, mock_config):
        """Test Vision API disabled in broll config"""
        mock_config.vision = Mock()
        mock_config.vision.enabled = True
        broll_config = BrollConfig(vision_enabled=False)

        result = stage._should_use_vision(mock_config, broll_config)

        assert result is False

    def test_should_use_vision_audio_first_mode(self, stage, mock_config):
        """Test Vision API disabled in audio-first mode"""
        mock_config.vision = Mock()
        mock_config.vision.enabled = True
        mock_config.download.audio_first = Mock()
        mock_config.download.audio_first.enabled = True
        broll_config = BrollConfig(vision_enabled=True)

        result = stage._should_use_vision(mock_config, broll_config)

        assert result is False

    def test_calculate_keyword_score(self, stage):
        """Test keyword overlap scoring"""
        keywords = {"earthquake", "disaster", "damage"}

        # High overlap
        score = stage._calculate_keyword_score(
            "earthquake caused damage",
            "earthquake damage footage",
            keywords
        )
        assert score > 0.5

        # No overlap
        score = stage._calculate_keyword_score(
            "beautiful sunset",
            "ocean waves",
            keywords
        )
        assert score < 0.3

    def test_calculate_entity_score(self, stage):
        """Test entity matching score"""
        entities = {"california", "san francisco"}

        # Both match
        score = stage._calculate_entity_score(
            "california earthquake",
            "california footage",
            entities
        )
        assert score > 0

        # No match
        score = stage._calculate_entity_score(
            "california earthquake",
            "tokyo footage",
            entities
        )
        assert score == 0

    def test_run_broll_disabled(self, stage, mock_state, mock_checkpoint):
        """Test stage skips when broll disabled"""
        config = Mock()
        config.broll = BrollConfig(enabled=False)

        result = stage.run(mock_state, config, mock_checkpoint)

        assert result.success is True
        assert result.data['skipped'] is True
        assert result.data['reason'] == 'broll_disabled'

    def test_run_no_voiceover(self, stage, mock_checkpoint):
        """Test stage skips with no voiceover"""
        state = PipelineState()
        state.voiceover_segments = []

        config = Mock()
        config.broll = BrollConfig()

        result = stage.run(state, config, mock_checkpoint)

        assert result.success is True
        assert result.data['skipped'] is True
        assert result.data['reason'] == 'no_voiceover'

    def test_run_no_transcripts(self, stage, mock_checkpoint):
        """Test stage skips with no transcripts"""
        state = PipelineState()
        state.voiceover_segments = [
            VoiceoverSegment(index=0, start=0.0, end=5.0, text="Test")
        ]
        state.transcripts = {}
        state.text_metadata = []

        config = Mock()
        config.broll = BrollConfig()

        result = stage.run(state, config, mock_checkpoint)

        assert result.success is True
        assert result.data['skipped'] is True
        assert result.data['reason'] == 'no_transcripts'

    def test_can_skip(self, stage, mock_state, mock_checkpoint):
        """Test can_skip checks checkpoint"""
        mock_checkpoint.should_skip_stage.return_value = True

        assert stage.can_skip(mock_state, mock_checkpoint) is True
        mock_checkpoint.should_skip_stage.assert_called_with("BROLL_MATCH")

    def test_validate_inputs(self, stage, mock_state, mock_config):
        """Test validate_inputs returns None (optional stage)"""
        result = stage.validate_inputs(mock_state, mock_config)
        assert result is None

    def test_sample_long_video_scenes(self, stage):
        """Test long video scene sampling"""
        # Create many scenes from same file
        scenes = [
            BrollScene(
                source_file="/video.mp4",
                start_time=i * 10.0,
                end_time=(i + 1) * 10.0,
                word_count=0,
                transcript=""
            )
            for i in range(50)
        ]

        # Sample with max 15 scenes per video
        sampled = stage._sample_long_video_scenes(
            scenes,
            long_threshold=600,
            sample_interval=120,
            max_scenes=15
        )

        assert len(sampled) <= 15


# ============================================================================
# Integration Tests
# ============================================================================

class TestBrollStagesIntegration:
    """Integration tests for B-roll stages"""

    def test_checkpoint_stage_order(self):
        """Test stages are in checkpoint STAGE_ORDER"""
        from src.checkpoint import STAGE_ORDER

        assert "BROLL_DOWNLOAD" in STAGE_ORDER
        assert "BROLL_MATCH" in STAGE_ORDER

        # BROLL_DOWNLOAD should come after STOCK
        download_idx = STAGE_ORDER.index("BROLL_DOWNLOAD")
        stock_idx = STAGE_ORDER.index("STOCK")
        assert download_idx > stock_idx

        # BROLL_MATCH should come after MATCH
        match_idx = STAGE_ORDER.index("BROLL_MATCH")
        main_match_idx = STAGE_ORDER.index("MATCH")
        assert match_idx > main_match_idx

    def test_pipeline_state_has_broll_fields(self):
        """Test PipelineState has B-roll fields"""
        state = PipelineState()

        assert hasattr(state, 'broll_downloads')
        assert hasattr(state, 'broll_matches')
        assert isinstance(state.broll_downloads, list)
        assert isinstance(state.broll_matches, list)

    def test_config_has_broll_section(self):
        """Test Config has broll section"""
        from src.config.base import Config

        config = Config()
        assert hasattr(config, 'broll')
        assert isinstance(config.broll, BrollConfig)

    def test_stages_registered(self):
        """Test stages are registered in stage registry"""
        from src.stages import get_stage

        broll_download = get_stage("BROLL_DOWNLOAD")
        broll_match = get_stage("BROLL_MATCH")

        assert broll_download is not None
        assert broll_match is not None
