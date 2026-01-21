"""
Tests for Premise-Based Scoring

Tests the premise extraction stage and premise-based matching scoring.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile
import json


class TestPremiseScoringFunction:
    """Tests for apply_premise_scoring() in scoring.py"""

    def test_no_premise_config_returns_original(self):
        """When premise config is None, return original score"""
        from src.matching.scoring import apply_premise_scoring
        from src.utils import SRTSegment

        vo_seg = SRTSegment(index=0, start_time=0, end_time=5, text='test')
        video_seg = SRTSegment(index=0, start_time=0, end_time=5, text='test', source_file='test.mp4')

        score, premise_score, reason = apply_premise_scoring(0.8, vo_seg, video_seg, {}, None)

        assert score == 0.8
        assert premise_score == 0.0
        assert reason == ""

    def test_premise_config_disabled_returns_original(self):
        """When premise scoring is disabled, return original score"""
        from src.matching.scoring import apply_premise_scoring
        from src.utils import SRTSegment

        class DisabledConfig:
            enabled = False

        vo_seg = SRTSegment(index=0, start_time=0, end_time=5, text='test')
        video_seg = SRTSegment(index=0, start_time=0, end_time=5, text='test', source_file='test.mp4')

        score, premise_score, reason = apply_premise_scoring(
            0.8, vo_seg, video_seg, {'test': 'some premise'}, DisabledConfig()
        )

        assert score == 0.8
        assert premise_score == 0.0

    def test_no_matching_premise_returns_original(self):
        """When video has no premise in dict, return original score"""
        from src.matching.scoring import apply_premise_scoring
        from src.utils import SRTSegment

        class EnabledConfig:
            enabled = True
            premise_weight = 0.50
            embedding_weight = 0.15
            keyword_weight = 0.25
            transcript_weight = 0.10

        vo_seg = SRTSegment(index=0, start_time=0, end_time=5, text='restaurant food')
        video_seg = SRTSegment(index=0, start_time=0, end_time=5, text='test', source_file='unknown_video.mp4')

        score, premise_score, reason = apply_premise_scoring(
            0.8, vo_seg, video_seg, {'other_video': 'some premise'}, EnabledConfig()
        )

        assert score == 0.8
        assert premise_score == 0.0
        assert reason == "no premise"

    def test_music_video_penalty_applied(self):
        """Music videos should get heavy penalty for non-music content"""
        from src.matching.scoring import apply_premise_scoring
        from src.utils import SRTSegment

        class EnabledConfig:
            enabled = True
            premise_weight = 0.50
            embedding_weight = 0.15
            keyword_weight = 0.25
            transcript_weight = 0.10

        vo_seg = SRTSegment(index=0, start_time=0, end_time=5, text='restaurant closures are affecting communities')
        video_seg = SRTSegment(index=0, start_time=0, end_time=5, text='lyrics', source_file='test_abc123def45.mp4')

        premises = {'abc123def45': 'Music lyric video for a love song'}
        score, premise_score, reason = apply_premise_scoring(
            0.8, vo_seg, video_seg, premises, EnabledConfig()
        )

        # Music penalty should significantly reduce score
        assert score < 0.5
        assert "music penalty" in reason

    def test_music_video_no_penalty_for_music_content(self):
        """Music videos should NOT be penalized for music-related voiceover"""
        from src.matching.scoring import apply_premise_scoring
        from src.utils import SRTSegment

        class EnabledConfig:
            enabled = True
            premise_weight = 0.50
            embedding_weight = 0.15
            keyword_weight = 0.25
            transcript_weight = 0.10

        vo_seg = SRTSegment(index=0, start_time=0, end_time=5, text='this song has beautiful lyrics')
        video_seg = SRTSegment(index=0, start_time=0, end_time=5, text='lyrics', source_file='test_abc123def45.mp4')

        premises = {'abc123def45': 'Music lyric video for a love song'}
        score, premise_score, reason = apply_premise_scoring(
            0.8, vo_seg, video_seg, premises, EnabledConfig()
        )

        # No music penalty for music content
        assert "music penalty" not in reason

    def test_matching_topic_scores_higher(self):
        """Videos with matching topic should score well"""
        from src.matching.scoring import apply_premise_scoring
        from src.utils import SRTSegment

        class EnabledConfig:
            enabled = True
            premise_weight = 0.50
            embedding_weight = 0.15
            keyword_weight = 0.25
            transcript_weight = 0.10

        vo_seg = SRTSegment(index=0, start_time=0, end_time=5, text='restaurant closures are affecting many people')
        video_seg = SRTSegment(index=0, start_time=0, end_time=5, text='closures', source_file='test_abc123def45.mp4')

        premises = {'abc123def45': 'Documentary about restaurant closures in America'}
        score, premise_score, reason = apply_premise_scoring(
            0.8, vo_seg, video_seg, premises, EnabledConfig()
        )

        # Matching topic should have positive premise score
        assert premise_score > 0
        assert "premise:" in reason

    def test_video_id_extraction_youtube_format(self):
        """Test video ID extraction from YouTube filename format"""
        from src.matching.scoring import _extract_video_id_for_premise

        # Standard YouTube format: title_videoId
        assert _extract_video_id_for_premise('My Video Title_abc123def45.mp4') == 'abc123def45'

        # Audio-first segment format: videoId_0001
        assert _extract_video_id_for_premise('abc123def45_0001.mp4') == 'abc123def45'

    def test_video_id_extraction_stock_footage(self):
        """Test video ID extraction for stock footage"""
        from src.matching.scoring import _extract_video_id_for_premise

        assert _extract_video_id_for_premise('pexels_12345.mp4') == 'pexels_12345'
        assert _extract_video_id_for_premise('pixabay_67890.mp4') == 'pixabay_67890'

    def test_video_id_extraction_fallback(self):
        """Test video ID extraction fallback to filename"""
        from src.matching.scoring import _extract_video_id_for_premise

        assert _extract_video_id_for_premise('random_video_name.mp4') == 'random_video_name'
        assert _extract_video_id_for_premise('') == ''


class TestPremiseStage:
    """Tests for PremiseStage"""

    @pytest.fixture
    def mock_state(self):
        """Create mock pipeline state"""
        from src.state import PipelineState, DownloadedVideo

        state = PipelineState()
        state.project_dir = tempfile.mkdtemp()
        state.downloaded_videos = [
            DownloadedVideo(file='/path/to/video_abc123def45.mp4', title='Test Video'),
        ]
        state.transcripts = {
            '/path/to/video_abc123def45.mp4': [
                {'text': 'This is about restaurants closing down.'}
            ]
        }
        state.video_premises = {}
        return state

    @pytest.fixture
    def mock_config(self):
        """Create mock config"""
        config = Mock()
        config.matching = Mock()
        config.matching.premise_scoring = Mock()
        config.matching.premise_scoring.enabled = True
        config.matching.premise_scoring.provider = 'gemini'
        config.matching.premise_scoring.model = 'gemini-2.0-flash'
        return config

    @pytest.fixture
    def mock_checkpoint(self):
        """Create mock checkpoint manager"""
        checkpoint = Mock()
        checkpoint.should_skip_stage.return_value = False
        return checkpoint

    def test_stage_name(self):
        """Test stage has correct name"""
        from src.stages.premise import PremiseStage

        stage = PremiseStage()
        assert stage.name == "PREMISE"

    def test_stage_description(self):
        """Test stage has description"""
        from src.stages.premise import PremiseStage

        stage = PremiseStage()
        assert stage.description == "Extract video topic/theme premises"

    def test_stage_registration(self):
        """Test stage is registered"""
        from src.stages import get_stage

        stage_class = get_stage("PREMISE")
        assert stage_class is not None
        assert stage_class.name == "PREMISE"

    def test_run_disabled_skips(self, mock_state, mock_checkpoint):
        """Test stage skips when premise scoring disabled"""
        from src.stages.premise import PremiseStage

        config = Mock()
        config.matching = Mock()
        config.matching.premise_scoring = Mock()
        config.matching.premise_scoring.enabled = False

        stage = PremiseStage()
        result = stage.run(mock_state, config, mock_checkpoint)

        assert result.success
        assert result.data.get('skipped') is True

    def test_can_skip_checks_checkpoint(self, mock_state, mock_checkpoint):
        """Test can_skip delegates to checkpoint"""
        from src.stages.premise import PremiseStage

        stage = PremiseStage()

        mock_checkpoint.should_skip_stage.return_value = True
        assert stage.can_skip(mock_state, mock_checkpoint) is True

        mock_checkpoint.should_skip_stage.return_value = False
        assert stage.can_skip(mock_state, mock_checkpoint) is False

    def test_restore_loads_from_cache(self, mock_state, mock_checkpoint, mock_config):
        """Test restore loads premises from cache"""
        from src.stages.premise import PremiseStage
        import os

        # Create cache directory with a premise file
        cache_dir = Path(mock_state.project_dir) / ".cache" / "premises"
        cache_dir.mkdir(parents=True, exist_ok=True)

        premise_data = {
            'video_id': 'test123',
            'premise': 'Documentary about food'
        }
        with open(cache_dir / 'test123.json', 'w') as f:
            json.dump(premise_data, f)

        stage = PremiseStage()
        result = stage.restore(mock_state, mock_checkpoint, mock_config)

        assert result is True
        assert 'test123' in mock_state.video_premises
        assert mock_state.video_premises['test123'] == 'Documentary about food'

    def test_restore_no_cache_returns_false(self, mock_state, mock_checkpoint, mock_config):
        """Test restore returns False when no cache exists"""
        from src.stages.premise import PremiseStage

        # Ensure cache directory doesn't exist
        cache_dir = Path(mock_state.project_dir) / ".cache" / "premises"
        if cache_dir.exists():
            import shutil
            shutil.rmtree(cache_dir)

        stage = PremiseStage()
        result = stage.restore(mock_state, mock_checkpoint, mock_config)

        assert result is False

    def test_extract_video_id_youtube_format(self):
        """Test _extract_video_id with YouTube filename"""
        from src.stages.premise import PremiseStage

        stage = PremiseStage()
        video_id = stage._extract_video_id('Some Title_abc123def45.mp4')
        assert video_id == 'abc123def45'

    def test_extract_video_id_segment_format(self):
        """Test _extract_video_id with audio-first segment filename"""
        from src.stages.premise import PremiseStage

        stage = PremiseStage()
        video_id = stage._extract_video_id('abc123def45_0001.mp4')
        assert video_id == 'abc123def45'

    def test_extract_video_id_stock_footage(self):
        """Test _extract_video_id with stock footage"""
        from src.stages.premise import PremiseStage

        stage = PremiseStage()
        video_id = stage._extract_video_id('pexels_12345.mp4')
        assert video_id == 'pexels_12345'

    def test_segments_to_text(self):
        """Test _segments_to_text converts segments correctly"""
        from src.stages.premise import PremiseStage

        stage = PremiseStage()

        # Dict segments
        segments = [
            {'text': 'Hello'},
            {'text': 'World'}
        ]
        result = stage._segments_to_text(segments)
        assert result == 'Hello World'

    def test_generate_fallback_premise(self):
        """Test fallback premise generation"""
        from src.stages.premise import PremiseStage

        stage = PremiseStage()

        # With title
        info = {'title': 'Test Video Title', 'video_id': 'test123'}
        premise = stage._generate_fallback_premise(info)
        assert 'Test Video Title' in premise

        # Pexels video
        info = {'title': '', 'video_id': 'pexels_12345'}
        premise = stage._generate_fallback_premise(info)
        assert 'Pexels' in premise

        # Pixabay video
        info = {'title': '', 'video_id': 'pixabay_67890'}
        premise = stage._generate_fallback_premise(info)
        assert 'Pixabay' in premise

    def test_collect_videos_deduplicates(self, mock_state):
        """Test _collect_videos deduplicates by video_id"""
        from src.stages.premise import PremiseStage
        from src.state import DownloadedVideo

        # Add duplicate video with same ID
        mock_state.downloaded_videos.append(
            DownloadedVideo(file='/other/path/video_abc123def45.mp4', title='Duplicate')
        )

        stage = PremiseStage()
        videos = stage._collect_videos(mock_state)

        # Should only have one entry for abc123def45
        video_ids = [v['video_id'] for v in videos]
        assert video_ids.count('abc123def45') == 1


class TestTieredMatcherPremiseIntegration:
    """Tests for premise scoring integration in TieredMatcher"""

    def test_set_video_premises(self):
        """Test setting video premises"""
        from src.matching.tiered_matcher import TieredMatcher

        matcher = TieredMatcher()
        premises = {'video1': 'Documentary about food', 'video2': 'Music video'}

        matcher.set_video_premises(premises)

        assert matcher.video_premises == premises

    def test_init_with_video_premises(self):
        """Test initializing TieredMatcher with video_premises"""
        from src.matching.tiered_matcher import TieredMatcher

        premises = {'video1': 'Test premise'}
        matcher = TieredMatcher(video_premises=premises)

        assert matcher.video_premises == premises

    def test_init_without_video_premises(self):
        """Test TieredMatcher defaults to empty premises"""
        from src.matching.tiered_matcher import TieredMatcher

        matcher = TieredMatcher()

        assert matcher.video_premises == {}


class TestMatchAllSegmentsPremiseIntegration:
    """Tests for premise scoring in match_all_segments"""

    def test_match_all_segments_accepts_video_premises(self):
        """Test match_all_segments accepts video_premises parameter"""
        from src.matching.main import match_all_segments
        import inspect

        sig = inspect.signature(match_all_segments)
        params = list(sig.parameters.keys())

        assert 'video_premises' in params


class TestPremiseScoringConfig:
    """Tests for PremiseScoringConfig dataclass"""

    def test_default_values(self):
        """Test PremiseScoringConfig has correct defaults"""
        from src.config.sections.matching import PremiseScoringConfig

        config = PremiseScoringConfig()

        assert config.enabled is True
        assert config.premise_weight == 0.50
        assert config.keyword_weight == 0.25
        assert config.embedding_weight == 0.15
        assert config.transcript_weight == 0.10

    def test_weights_sum_to_one(self):
        """Test default weights sum to 1.0"""
        from src.config.sections.matching import PremiseScoringConfig

        config = PremiseScoringConfig()
        total = (
            config.premise_weight +
            config.keyword_weight +
            config.embedding_weight +
            config.transcript_weight
        )

        assert abs(total - 1.0) < 0.01

    def test_custom_weights(self):
        """Test custom weight configuration"""
        from src.config.sections.matching import PremiseScoringConfig

        config = PremiseScoringConfig(
            premise_weight=0.60,
            keyword_weight=0.20,
            embedding_weight=0.10,
            transcript_weight=0.10
        )

        assert config.premise_weight == 0.60
        assert config.keyword_weight == 0.20

    def test_llm_settings(self):
        """Test LLM configuration settings"""
        from src.config.sections.matching import PremiseScoringConfig

        config = PremiseScoringConfig()

        assert config.provider == "gemini"
        assert config.model == "gemini-2.0-flash"
        assert config.llm_rerank is True
        assert config.llm_rerank_top_n == 10


class TestCheckpointPremiseStage:
    """Tests for PREMISE stage in checkpoint"""

    def test_premise_in_stage_order(self):
        """Test PREMISE is in STAGE_ORDER"""
        from src.checkpoint import STAGE_ORDER

        assert "PREMISE" in STAGE_ORDER

    def test_premise_after_transcribe(self):
        """Test PREMISE comes after TRANSCRIBE"""
        from src.checkpoint import STAGE_ORDER

        transcribe_idx = STAGE_ORDER.index("TRANSCRIBE")
        premise_idx = STAGE_ORDER.index("PREMISE")

        assert premise_idx > transcribe_idx

    def test_premise_before_scene_detection(self):
        """Test PREMISE comes before SCENE_DETECTION"""
        from src.checkpoint import STAGE_ORDER

        premise_idx = STAGE_ORDER.index("PREMISE")
        scene_idx = STAGE_ORDER.index("SCENE_DETECTION")

        assert premise_idx < scene_idx

    def test_checkpoint_data_has_premise_field(self):
        """Test CheckpointData has premise field"""
        from src.checkpoint import CheckpointData

        data = CheckpointData()
        assert hasattr(data, 'premise')
        assert data.premise == {}


class TestPipelinePremiseStageRegistration:
    """Tests for PREMISE stage registration in pipeline"""

    def test_create_pipeline_includes_premise(self):
        """Test create_default_pipeline includes PremiseStage"""
        from src.pipeline import create_default_pipeline
        from unittest.mock import Mock

        config = Mock()
        config.download = Mock()
        config.download.caption_first = Mock()
        config.download.caption_first.enabled = False

        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline = create_default_pipeline(config, tmpdir)

            stage_names = [s.name for s in pipeline.stages]
            assert "PREMISE" in stage_names

    def test_create_match_only_pipeline_includes_premise(self):
        """Test create_match_only_pipeline includes PremiseStage"""
        from src.pipeline import create_match_only_pipeline
        from unittest.mock import Mock

        config = Mock()
        config.download = Mock()
        config.download.caption_first = Mock()
        config.download.caption_first.enabled = False

        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline = create_match_only_pipeline(config, tmpdir)

            stage_names = [s.name for s in pipeline.stages]
            assert "PREMISE" in stage_names


class TestPremiseCaching:
    """Tests for premise caching functionality"""

    def test_save_and_load_premise_cache(self):
        """Test saving and loading premise from cache"""
        from src.stages.premise import PremiseStage
        import tempfile
        import shutil

        stage = PremiseStage()

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)

            # Save premise
            video_info = {'title': 'Test', 'video_id': 'test123'}
            stage._save_to_cache(cache_dir, 'test123', 'Documentary about testing', video_info)

            # Load premise
            loaded = stage._load_from_cache(cache_dir, 'test123')

            assert loaded == 'Documentary about testing'

    def test_load_nonexistent_cache_returns_none(self):
        """Test loading from nonexistent cache returns None"""
        from src.stages.premise import PremiseStage
        import tempfile

        stage = PremiseStage()

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)

            result = stage._load_from_cache(cache_dir, 'nonexistent')

            assert result is None

    def test_cache_file_format(self):
        """Test cache file has correct JSON structure"""
        from src.stages.premise import PremiseStage
        import tempfile

        stage = PremiseStage()

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)

            video_info = {'title': 'Test Video', 'has_transcript': True}
            stage._save_to_cache(cache_dir, 'test123', 'Test premise', video_info)

            # Read raw JSON
            cache_file = cache_dir / 'test123.json'
            with open(cache_file) as f:
                data = json.load(f)

            assert data['video_id'] == 'test123'
            assert data['premise'] == 'Test premise'
            assert 'extracted_at' in data
            assert 'source' in data
