"""Tests for US-125-009: Stage input preview dry-run showing estimated API calls"""

import pytest
from unittest.mock import MagicMock, patch
from src.stages.video_search import VideoSearchStage
from src.stages.caption_stage import CaptionStage
from src.stages.match import MatchStage
from src.stages.iterative_match import IterativeMatchStage


class MockState:
    """Mock PipelineState for testing"""
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


class MockConfig:
    """Mock Config for testing"""
    def __init__(self, **kwargs):
        self._attrs = kwargs

    def __getattr__(self, name):
        if name in self._attrs:
            return self._attrs[name]
        # Return nested mock objects for config sections
        if name == 'video_search':
            mock_section = MagicMock()
            mock_section.max_results = 50
            return mock_section
        if name == 'matching':
            mock_section = MagicMock()
            mock_section.use_llm_reranker = False
            mock_section.embeddings = {}
            return mock_section
        if name == 'download':
            mock_section = MagicMock()
            mock_section.caption_first = {'retry_budget': {'max_attempts': 100}}
            return mock_section
        raise AttributeError(f"MockConfig has no attribute {name}")


class TestVideoSearchApiEstimates:
    """Test API estimates for VideoSearchStage"""

    def test_get_api_estimates_with_keywords(self):
        """Test that video search returns correct API estimates"""
        stage = VideoSearchStage()

        # Create mock state with keywords
        state = MockState(
            keywords=['python tutorial', 'coding basics', 'programming'],
            video_candidates={}
        )
        config = MockConfig()

        estimates = stage.get_api_estimates(state, config)

        # 3 keywords = 3 YouTube API calls
        assert estimates['youtube_api_calls'] == 3
        # Cost: 3 * 100 quota * $0.002/1000 = $0.0006
        assert estimates['estimated_cost_usd'] == 0.0006
        # Duration: 3 * 0.5s = 1.5s
        assert estimates['estimated_duration_seconds'] == 1.5

    def test_get_api_estimates_no_keywords(self):
        """Test estimates with no keywords"""
        stage = VideoSearchStage()

        state = MockState(keywords=[], video_candidates={})
        config = MockConfig()

        estimates = stage.get_api_estimates(state, config)

        assert estimates['youtube_api_calls'] == 0
        assert estimates['estimated_cost_usd'] == 0.0
        assert estimates['estimated_duration_seconds'] == 0.0


class TestCaptionStageApiEstimates:
    """Test API estimates for CaptionStage"""

    def test_get_api_estimates_with_videos(self):
        """Test caption fetch estimates"""
        stage = CaptionStage()

        # Create mock state with video IDs
        state = MockState(
            video_ids=['vid1', 'vid2', 'vid3', 'vid4', 'vid5'],
            captions={}
        )
        config = MockConfig()

        estimates = stage.get_api_estimates(state, config)

        # 5 videos with retry budget
        assert estimates['caption_fetch_attempts'] > 0
        assert estimates['caption_fetch_attempts'] <= 7  # 5 + ~20% retry
        assert estimates['estimated_cost_usd'] == 0.0  # Caption API is free
        assert estimates['estimated_duration_seconds'] > 0

    def test_get_api_estimates_no_videos(self):
        """Test estimates with no videos"""
        stage = CaptionStage()

        state = MockState(video_ids=[], captions={})
        config = MockConfig()

        estimates = stage.get_api_estimates(state, config)

        assert estimates['caption_fetch_attempts'] == 0
        assert estimates['estimated_cost_usd'] == 0.0
        assert estimates['estimated_duration_seconds'] == 0.0


class TestMatchStageApiEstimates:
    """Test API estimates for MatchStage"""

    def test_get_api_estimates_with_segments_and_videos(self):
        """Test matching estimates with segments and videos"""
        stage = MatchStage()

        # Create mock state
        state = MockState(
            voiceover_segments=[MagicMock(), MagicMock(), MagicMock()],  # 3 segments
            caption_results={'vid1': {}, 'vid2': {}},  # 2 videos
            matches=[]
        )
        config = MockConfig()

        estimates = stage.get_api_estimates(state, config)

        # 3 segments + 2 videos = 5 embedding calls
        assert estimates['embedding_calls'] == 5
        assert estimates['estimated_cost_usd'] > 0
        assert estimates['estimated_duration_seconds'] > 0

    def test_get_api_estimates_with_llm_reranker(self):
        """Test matching estimates with LLM reranker enabled"""
        stage = MatchStage()

        state = MockState(
            voiceover_segments=[MagicMock()] * 25,  # 25 segments -> ~2-3 LLM calls
            caption_results={'vid1': {}},
            matches=[]
        )

        # Create config with LLM reranker enabled
        config = MockConfig()
        config._attrs['matching'] = MagicMock()
        config._attrs['matching'].use_llm_reranker = True
        config._attrs['matching'].embeddings = {}

        estimates = stage.get_api_estimates(state, config)

        assert estimates['embedding_calls'] == 26  # 25 segments + 1 video
        assert 'llm_calls' in estimates
        assert estimates['llm_calls'] >= 2  # 25 // 10 = 2

    def test_get_api_estimates_no_inputs(self):
        """Test estimates with no inputs"""
        stage = MatchStage()

        state = MockState(
            voiceover_segments=[],
            caption_results={},
            matches=[]
        )
        config = MockConfig()

        estimates = stage.get_api_estimates(state, config)

        assert estimates['embedding_calls'] == 0
        assert estimates['estimated_cost_usd'] == 0.0
        assert estimates['estimated_duration_seconds'] == 0.0


class TestIterativeMatchApiEstimates:
    """Test API estimates for IterativeMatchStage"""

    def test_get_api_estimates_with_matches(self):
        """Test iterative matching estimates"""
        stage = IterativeMatchStage()

        # Create mock state with matches and segments
        state = MockState(
            matches=[MagicMock(), MagicMock(), MagicMock()],  # 3 matches
            voiceover_segments=[MagicMock()] * 20  # 20 segments
        )
        config = MockConfig()

        estimates = stage.get_api_estimates(state, config)

        # 20 segments // 5 = 4 LLM calls
        assert estimates['llm_calls'] == 4
        # 20 segments // 10 = 2 video search calls
        assert estimates['video_search_calls'] == 2
        # LLM cost: 4 * $0.01 = $0.04
        # Search cost: 2 * $0.002 = $0.004
        assert estimates['estimated_cost_usd'] == pytest.approx(0.044, rel=0.01)
        # Duration: 4 * 2.0 + 2 * 0.5 = 9s
        assert estimates['estimated_duration_seconds'] == 9.0

    def test_get_api_estimates_no_segments(self):
        """Test estimates with no segments"""
        stage = IterativeMatchStage()

        state = MockState(
            matches=[],
            voiceover_segments=[]
        )
        config = MockConfig()

        estimates = stage.get_api_estimates(state, config)

        # Should have at least 1 LLM call and 1 search call for any run
        assert estimates['llm_calls'] == 1
        assert estimates['video_search_calls'] == 1


class TestDryRunApiEstimatesIntegration:
    """Integration tests for dry-run API estimates"""

    def test_all_stages_have_get_api_estimates_method(self):
        """Verify all major stages have get_api_estimates method"""
        stages = [
            VideoSearchStage(),
            CaptionStage(),
            MatchStage(),
            IterativeMatchStage(),
        ]

        for stage in stages:
            assert hasattr(stage, 'get_api_estimates'), f"{stage.__class__.__name__} missing get_api_estimates"
            assert callable(getattr(stage, 'get_api_estimates'))

    def test_estimates_format_consistency(self):
        """Verify all estimates return consistent format"""
        stages = [
            VideoSearchStage(),
            CaptionStage(),
            MatchStage(),
            IterativeMatchStage(),
        ]

        state = MockState(
            keywords=['test'],
            video_ids=['vid1'],
            voiceover_segments=[MagicMock()],
            caption_results={'vid1': {}},
            matches=[]
        )
        config = MockConfig()

        for stage in stages:
            estimates = stage.get_api_estimates(state, config)
            assert 'estimated_cost_usd' in estimates
            assert 'estimated_duration_seconds' in estimates
            assert isinstance(estimates['estimated_cost_usd'], (int, float))
            assert isinstance(estimates['estimated_duration_seconds'], (int, float))
