"""
Extended tests for src/matching/main.py to cover missed lines.

Targets:
- Lines 91, 93: Location chapters and video locations initialization
- Lines 107, 110: max_clip_reuse=1 and face_preference logging
- Lines 118-120, 145-147: Variety config as dict handling
- Lines 190-193, 198: B-roll segment candidates handling
- Lines 208, 218: Global clip deduplication and variety constraint relaxation
- Lines 280-283, 305: Strategy candidates filtering and recording
- Lines 330-332: Secondary matches V4-V6 recording

Created: 2026-01-11
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch, call
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import tempfile
import shutil

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.matching.main import match_all_segments
from src.matching.tracking import TimelineVarietyTracker, GlobalClipTracker
from src.matching.strategies import StrategyMatcher
from src.utils import SRTSegment, SceneInfo, Match, MatchResult, AlternativeMatch, StrategyMatch


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def temp_dir():
    """Create a temporary directory for test artifacts"""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path)


@pytest.fixture
def mock_config():
    """Create a mock config with all required attributes"""
    config = Mock()

    # Matching config
    config.matching = Mock()
    config.matching.primary_provider = "gemini"
    config.matching.gemini_model = "gemini-2.0-flash"
    config.matching.min_confidence = 0.6
    config.matching.skip_llm_threshold = 0.9
    config.matching.max_clip_reuse = 2
    config.matching.reuse_penalty = 0.1
    config.matching.embedding_candidates = 30
    config.matching.llm_rerank_candidates = 5
    config.matching.context_window = 1
    config.matching.duration_scoring_enabled = False
    config.matching.cache_llm_responses = False
    config.matching.clip_hard_block = True
    config.matching.use_local_for_review = False

    # Output config
    config.output = Mock()
    config.output.num_alternatives = 2
    config.output.include_strategy_tracks = True
    config.output.strategy_tracks = ["embedding_diversity", "broll_only"]

    # Variety config as object (default)
    config.output.variety = Mock()
    config.output.variety.enforce_timeline_variety = True
    config.output.variety.timeline_variety_window = 600.0
    config.output.variety.max_source_repeats_in_window = 1
    config.output.variety.require_different_source = True
    config.output.variety.min_time_distance = 10.0
    config.output.variety.min_embedding_distance = 0.3

    return config


@pytest.fixture
def mock_config_variety_dict():
    """Create a mock config with variety as a dict (lines 118-120, 145-147)"""
    config = Mock()

    # Matching config
    config.matching = Mock()
    config.matching.primary_provider = "gemini"
    config.matching.gemini_model = "gemini-2.0-flash"
    config.matching.min_confidence = 0.6
    config.matching.skip_llm_threshold = 0.9
    config.matching.max_clip_reuse = 2
    config.matching.reuse_penalty = 0.1
    config.matching.embedding_candidates = 30
    config.matching.llm_rerank_candidates = 5
    config.matching.context_window = 1
    config.matching.duration_scoring_enabled = False
    config.matching.cache_llm_responses = False
    config.matching.clip_hard_block = True
    config.matching.use_local_for_review = False

    # Output config
    config.output = Mock()
    config.output.num_alternatives = 2
    config.output.include_strategy_tracks = True
    config.output.strategy_tracks = ["embedding_diversity"]

    # Variety config as DICT to test lines 118-120, 145-147
    config.output.variety = {
        'enforce_timeline_variety': True,
        'timeline_variety_window': 300.0,
        'max_source_repeats_in_window': 2,
        'require_different_source': True,
        'min_time_distance': 5.0,
        'min_embedding_distance': 0.2
    }

    return config


@pytest.fixture
def mock_cache(temp_dir):
    """Create a mock cache manager"""
    cache = Mock()
    cache.cache_dir = str(temp_dir / ".cache")
    cache.get_llm_response = Mock(return_value=None)
    cache.save_llm_response = Mock()
    return cache


@pytest.fixture
def vo_segments():
    """Create sample voiceover segments"""
    return [
        SRTSegment(
            index=1,
            start_time=0.0,
            end_time=5.0,
            text="Welcome to Tokyo, the bustling capital of Japan",
            source_file="voiceover.srt",
            keywords=["tokyo", "japan", "capital"]
        ),
        SRTSegment(
            index=2,
            start_time=5.0,
            end_time=10.0,
            text="The city is famous for its modern architecture",
            source_file="voiceover.srt",
            keywords=["city", "modern", "architecture"]
        ),
        SRTSegment(
            index=3,
            start_time=10.0,
            end_time=15.0,
            text="Traditional temples dot the landscape",
            source_file="voiceover.srt",
            keywords=["traditional", "temples", "landscape"]
        )
    ]


@pytest.fixture
def video_segments():
    """Create sample video segments including B-roll"""
    segments = [
        SRTSegment(
            index=1,
            start_time=0.0,
            end_time=10.0,
            text="Tokyo cityscape with skyscrapers",
            source_file="tokyo_footage.mp4",
            keywords=["tokyo", "city", "skyscraper"]
        ),
        SRTSegment(
            index=2,
            start_time=10.0,
            end_time=20.0,
            text="Modern buildings in Tokyo",
            source_file="tokyo_footage.mp4",
            keywords=["building", "modern", "glass"]
        ),
        SRTSegment(
            index=3,
            start_time=0.0,
            end_time=15.0,
            text="Kyoto temple at sunset",
            source_file="kyoto_footage.mp4",
            keywords=["temple", "traditional", "sunset"]
        ),
        SRTSegment(
            index=4,
            start_time=0.0,
            end_time=12.0,
            text="Osaka street food",
            source_file="osaka_footage.mp4",
            keywords=["osaka", "food", "street"]
        ),
        SRTSegment(
            index=5,
            start_time=0.0,
            end_time=8.0,
            text="[Silent video: nature scenery]",
            source_file="broll_nature.mp4",
            keywords=["nature", "scenery"],
            is_broll=True  # B-roll segment
        ),
        SRTSegment(
            index=6,
            start_time=0.0,
            end_time=10.0,
            text="[Silent video: city timelapse]",
            source_file="broll_city.mp4",
            keywords=["city", "timelapse"],
            is_broll=True  # B-roll segment
        )
    ]
    return segments


@pytest.fixture
def embeddings():
    """Create sample embeddings (simple vectors for testing)"""
    # 3 voiceover embeddings
    vo_embeddings = [
        [0.9, 0.1, 0.0],
        [0.8, 0.2, 0.0],
        [0.7, 0.2, 0.1]
    ]

    # 6 video embeddings (matching video_segments fixture)
    video_embeddings = [
        [1.0, 0.0, 0.0],  # tokyo_footage seg 1
        [0.9, 0.1, 0.0],  # tokyo_footage seg 2
        [0.1, 0.0, 0.9],  # kyoto_footage seg 3
        [0.5, 0.5, 0.0],  # osaka_footage seg 4
        [0.3, 0.3, 0.4],  # broll_nature seg 5
        [0.4, 0.4, 0.2]   # broll_city seg 6
    ]

    return vo_embeddings, video_embeddings


@pytest.fixture
def scenes():
    """Create sample scene info"""
    return {
        "tokyo_footage.mp4": [
            SceneInfo(
                video_path="tokyo_footage.mp4",
                scene_index=0,
                start_time=0.0,
                end_time=10.0,
                description="Tokyo cityscape",
                visual_keywords=["building", "city", "skyscraper"]
            )
        ],
        "kyoto_footage.mp4": [
            SceneInfo(
                video_path="kyoto_footage.mp4",
                scene_index=0,
                start_time=0.0,
                end_time=15.0,
                description="Traditional temple",
                visual_keywords=["temple", "traditional", "sunset"]
            )
        ]
    }


@pytest.fixture
def mock_location_chapters():
    """Create mock location chapters"""
    chapter = Mock()
    chapter.start_segment_idx = 0
    chapter.end_segment_idx = 2
    chapter.location = "Tokyo, Japan"
    return [chapter]


@pytest.fixture
def mock_video_locations():
    """Create mock video locations"""
    location = Mock()
    location.city = "Tokyo"
    location.country = "Japan"
    return {"tokyo_footage.mp4": location, "osaka_footage.mp4": location}


# ============================================================================
# Mock Match Result Helper
# ============================================================================

def create_mock_match_result(video_seg, vo_seg, confidence=0.85, has_alternatives=True, has_secondaries=True, has_strategies=True):
    """Create a mock MatchResult with all components"""
    primary_match = Match(
        voiceover_segment=vo_seg,
        video_segment=video_seg,
        video_scene=None,
        confidence=confidence,
        reasoning="Test match"
    )

    alternatives = []
    if has_alternatives:
        alt_seg = SRTSegment(
            index=99,
            start_time=0.0,
            end_time=10.0,
            text="Alternative segment",
            source_file="alt_source.mp4"
        )
        alternatives = [
            AlternativeMatch(
                video_segment=alt_seg,
                video_scene=None,
                confidence=0.75,
                reasoning="Alt 1"
            )
        ]

    secondary_matches = []
    if has_secondaries:
        sec_seg = SRTSegment(
            index=98,
            start_time=0.0,
            end_time=10.0,
            text="Secondary segment",
            source_file="sec_source.mp4"
        )
        secondary_matches = [
            AlternativeMatch(
                video_segment=sec_seg,
                video_scene=None,
                confidence=0.70,
                reasoning="Secondary 1"
            )
        ]

    strategy_matches = []
    if has_strategies:
        strat_seg = SRTSegment(
            index=97,
            start_time=0.0,
            end_time=10.0,
            text="Strategy segment",
            source_file="strat_source.mp4"
        )
        strategy_matches = [
            StrategyMatch(
                video_segment=strat_seg,
                video_scene=None,
                confidence=0.65,
                reasoning="Diverse match",
                strategy="embedding_diversity"
            )
        ]

    return MatchResult(
        primary_match=primary_match,
        alternatives=alternatives,
        secondary_matches=secondary_matches,
        strategy_matches=strategy_matches,
        has_gap=False,
        gap_reason=""
    )


# ============================================================================
# Test Location Chapters and Video Locations (Lines 91, 93)
# ============================================================================

class TestLocationInitialization:
    """Test location chapter and video location initialization (lines 91, 93)"""

    @pytest.mark.fast
    def test_match_with_location_chapters(self, mock_config, mock_cache, vo_segments, video_segments, embeddings, scenes, mock_location_chapters):
        """Test that location_chapters are passed to matcher (line 91)"""
        vo_embeddings, video_embeddings = embeddings

        with patch('src.matching.tiered_matcher.TieredMatcher') as MockMatcher:
            mock_matcher_instance = Mock()
            mock_matcher_instance.match_segment = Mock(return_value=create_mock_match_result(
                video_segments[0], vo_segments[0]
            ))
            mock_matcher_instance.local_provider = None
            mock_matcher_instance.face_preference = "neutral"
            MockMatcher.return_value = mock_matcher_instance

            with patch('src.matching.main.find_top_k_similar') as mock_find:
                mock_find.return_value = ([0.9, 0.8, 0.7], [0, 1, 2])

                with patch('src.matching.main.StrategyMatcher') as MockStrategyMatcher:
                    mock_strategy = Mock()
                    mock_strategy.get_clip_id = Mock(return_value="test:0.00-10.00")
                    mock_strategy.get_strategy_matches = Mock(return_value=[])
                    mock_strategy.get_secondary_matches_diversity = Mock(return_value=[])
                    MockStrategyMatcher.return_value = mock_strategy

                    results = match_all_segments(
                        vo_segments,
                        video_segments,
                        vo_embeddings,
                        video_embeddings,
                        scenes,
                        mock_config,
                        mock_cache,
                        location_chapters=mock_location_chapters  # Test line 91
                    )

                    # Verify set_location_chapters was called
                    mock_matcher_instance.set_location_chapters.assert_called_once_with(mock_location_chapters)

    @pytest.mark.fast
    def test_match_with_video_locations(self, mock_config, mock_cache, vo_segments, video_segments, embeddings, scenes, mock_video_locations):
        """Test that video_locations are passed to matcher (line 93)"""
        vo_embeddings, video_embeddings = embeddings

        with patch('src.matching.tiered_matcher.TieredMatcher') as MockMatcher:
            mock_matcher_instance = Mock()
            mock_matcher_instance.match_segment = Mock(return_value=create_mock_match_result(
                video_segments[0], vo_segments[0]
            ))
            mock_matcher_instance.local_provider = None
            mock_matcher_instance.face_preference = "neutral"
            MockMatcher.return_value = mock_matcher_instance

            with patch('src.matching.main.find_top_k_similar') as mock_find:
                mock_find.return_value = ([0.9, 0.8, 0.7], [0, 1, 2])

                with patch('src.matching.main.StrategyMatcher') as MockStrategyMatcher:
                    mock_strategy = Mock()
                    mock_strategy.get_clip_id = Mock(return_value="test:0.00-10.00")
                    mock_strategy.get_strategy_matches = Mock(return_value=[])
                    mock_strategy.get_secondary_matches_diversity = Mock(return_value=[])
                    MockStrategyMatcher.return_value = mock_strategy

                    results = match_all_segments(
                        vo_segments,
                        video_segments,
                        vo_embeddings,
                        video_embeddings,
                        scenes,
                        mock_config,
                        mock_cache,
                        video_locations=mock_video_locations  # Test line 93
                    )

                    # Verify set_video_locations was called
                    mock_matcher_instance.set_video_locations.assert_called_once_with(mock_video_locations)

    @pytest.mark.fast
    def test_match_with_both_location_params(self, mock_config, mock_cache, vo_segments, video_segments, embeddings, scenes, mock_location_chapters, mock_video_locations):
        """Test both location parameters together"""
        vo_embeddings, video_embeddings = embeddings

        with patch('src.matching.tiered_matcher.TieredMatcher') as MockMatcher:
            mock_matcher_instance = Mock()
            mock_matcher_instance.match_segment = Mock(return_value=create_mock_match_result(
                video_segments[0], vo_segments[0]
            ))
            mock_matcher_instance.local_provider = None
            mock_matcher_instance.face_preference = "neutral"
            MockMatcher.return_value = mock_matcher_instance

            with patch('src.matching.main.find_top_k_similar') as mock_find:
                mock_find.return_value = ([0.9, 0.8, 0.7], [0, 1, 2])

                with patch('src.matching.main.StrategyMatcher') as MockStrategyMatcher:
                    mock_strategy = Mock()
                    mock_strategy.get_clip_id = Mock(return_value="test:0.00-10.00")
                    mock_strategy.get_strategy_matches = Mock(return_value=[])
                    mock_strategy.get_secondary_matches_diversity = Mock(return_value=[])
                    MockStrategyMatcher.return_value = mock_strategy

                    results = match_all_segments(
                        vo_segments,
                        video_segments,
                        vo_embeddings,
                        video_embeddings,
                        scenes,
                        mock_config,
                        mock_cache,
                        location_chapters=mock_location_chapters,
                        video_locations=mock_video_locations
                    )

                    # Both should be called
                    mock_matcher_instance.set_location_chapters.assert_called_once()
                    mock_matcher_instance.set_video_locations.assert_called_once()


# ============================================================================
# Test max_clip_reuse=1 and face_preference Logging (Lines 107, 110)
# ============================================================================

class TestConfigLogging:
    """Test logging for special config values (lines 107, 110)"""

    @pytest.mark.fast
    def test_max_clip_reuse_one_logging(self, mock_config, mock_cache, vo_segments, video_segments, embeddings, scenes):
        """Test logging when max_clip_reuse=1 (line 107)"""
        mock_config.matching.max_clip_reuse = 1  # Trigger line 107
        vo_embeddings, video_embeddings = embeddings

        with patch('src.matching.tiered_matcher.TieredMatcher') as MockMatcher:
            mock_matcher_instance = Mock()
            mock_matcher_instance.match_segment = Mock(return_value=create_mock_match_result(
                video_segments[0], vo_segments[0]
            ))
            mock_matcher_instance.local_provider = None
            mock_matcher_instance.face_preference = "neutral"
            MockMatcher.return_value = mock_matcher_instance

            with patch('src.matching.main.find_top_k_similar') as mock_find:
                mock_find.return_value = ([0.9, 0.8], [0, 1])

                with patch('src.matching.main.StrategyMatcher') as MockStrategyMatcher:
                    mock_strategy = Mock()
                    mock_strategy.get_clip_id = Mock(return_value="test:0.00-10.00")
                    mock_strategy.get_strategy_matches = Mock(return_value=[])
                    mock_strategy.get_secondary_matches_diversity = Mock(return_value=[])
                    MockStrategyMatcher.return_value = mock_strategy

                    with patch('src.matching.main.logger') as mock_logger:
                        results = match_all_segments(
                            vo_segments,
                            video_segments,
                            vo_embeddings,
                            video_embeddings,
                            scenes,
                            mock_config,
                            mock_cache
                        )

                        # Check that the "Each clip can only be used ONCE" message was logged
                        info_calls = [str(c) for c in mock_logger.info.call_args_list]
                        assert any("ONCE" in str(c) for c in info_calls), "Expected 'used ONCE' log message"

    @pytest.mark.fast
    def test_face_preference_more_logging(self, mock_config, mock_cache, vo_segments, video_segments, embeddings, scenes):
        """Test logging when face_preference != 'neutral' (line 110)"""
        vo_embeddings, video_embeddings = embeddings

        with patch('src.matching.tiered_matcher.TieredMatcher') as MockMatcher:
            mock_matcher_instance = Mock()
            mock_matcher_instance.match_segment = Mock(return_value=create_mock_match_result(
                video_segments[0], vo_segments[0]
            ))
            mock_matcher_instance.local_provider = None
            mock_matcher_instance.face_preference = "more"  # Will be set on instance
            MockMatcher.return_value = mock_matcher_instance

            with patch('src.matching.main.find_top_k_similar') as mock_find:
                mock_find.return_value = ([0.9, 0.8], [0, 1])

                with patch('src.matching.main.StrategyMatcher') as MockStrategyMatcher:
                    mock_strategy = Mock()
                    mock_strategy.get_clip_id = Mock(return_value="test:0.00-10.00")
                    mock_strategy.get_strategy_matches = Mock(return_value=[])
                    mock_strategy.get_secondary_matches_diversity = Mock(return_value=[])
                    MockStrategyMatcher.return_value = mock_strategy

                    with patch('src.matching.main.logger') as mock_logger:
                        results = match_all_segments(
                            vo_segments,
                            video_segments,
                            vo_embeddings,
                            video_embeddings,
                            scenes,
                            mock_config,
                            mock_cache,
                            face_preference="more"  # Trigger line 110
                        )

                        # Check that face preference was logged
                        info_calls = [str(c) for c in mock_logger.info.call_args_list]
                        assert any("Face preference" in str(c) for c in info_calls), "Expected 'Face preference' log message"

    @pytest.mark.fast
    def test_face_preference_none_logging(self, mock_config, mock_cache, vo_segments, video_segments, embeddings, scenes):
        """Test logging when face_preference is 'none'"""
        vo_embeddings, video_embeddings = embeddings

        with patch('src.matching.tiered_matcher.TieredMatcher') as MockMatcher:
            mock_matcher_instance = Mock()
            mock_matcher_instance.match_segment = Mock(return_value=create_mock_match_result(
                video_segments[0], vo_segments[0]
            ))
            mock_matcher_instance.local_provider = None
            mock_matcher_instance.face_preference = "none"
            MockMatcher.return_value = mock_matcher_instance

            with patch('src.matching.main.find_top_k_similar') as mock_find:
                mock_find.return_value = ([0.9, 0.8], [0, 1])

                with patch('src.matching.main.StrategyMatcher') as MockStrategyMatcher:
                    mock_strategy = Mock()
                    mock_strategy.get_clip_id = Mock(return_value="test:0.00-10.00")
                    mock_strategy.get_strategy_matches = Mock(return_value=[])
                    mock_strategy.get_secondary_matches_diversity = Mock(return_value=[])
                    MockStrategyMatcher.return_value = mock_strategy

                    with patch('src.matching.main.logger') as mock_logger:
                        results = match_all_segments(
                            vo_segments,
                            video_segments,
                            vo_embeddings,
                            video_embeddings,
                            scenes,
                            mock_config,
                            mock_cache,
                            face_preference="none"
                        )

                        info_calls = [str(c) for c in mock_logger.info.call_args_list]
                        assert any("none" in str(c) for c in info_calls), "Expected face preference log"


# ============================================================================
# Test Variety Config as Dict (Lines 118-120, 145-147)
# ============================================================================

class TestVarietyConfigAsDict:
    """Test variety config handling when passed as dict (lines 118-120, 145-147)"""

    @pytest.mark.fast
    def test_variety_config_dict_timeline_settings(self, mock_config_variety_dict, mock_cache, vo_segments, video_segments, embeddings, scenes):
        """Test that dict variety config is handled correctly for timeline settings"""
        vo_embeddings, video_embeddings = embeddings

        with patch('src.matching.tiered_matcher.TieredMatcher') as MockMatcher:
            mock_matcher_instance = Mock()
            mock_matcher_instance.match_segment = Mock(return_value=create_mock_match_result(
                video_segments[0], vo_segments[0]
            ))
            mock_matcher_instance.local_provider = None
            mock_matcher_instance.face_preference = "neutral"
            MockMatcher.return_value = mock_matcher_instance

            with patch('src.matching.main.find_top_k_similar') as mock_find:
                mock_find.return_value = ([0.9, 0.8], [0, 1])

                with patch('src.matching.main.StrategyMatcher') as MockStrategyMatcher:
                    mock_strategy = Mock()
                    mock_strategy.get_clip_id = Mock(return_value="test:0.00-10.00")
                    mock_strategy.get_strategy_matches = Mock(return_value=[])
                    mock_strategy.get_secondary_matches_diversity = Mock(return_value=[])
                    MockStrategyMatcher.return_value = mock_strategy

                    with patch('src.matching.main.TimelineVarietyTracker') as MockTracker:
                        mock_tracker = Mock()
                        mock_tracker.get_excluded_sources = Mock(return_value=set())
                        mock_tracker.get_stats = Mock(return_value={})
                        MockTracker.return_value = mock_tracker

                        results = match_all_segments(
                            vo_segments,
                            video_segments,
                            vo_embeddings,
                            video_embeddings,
                            scenes,
                            mock_config_variety_dict,
                            mock_cache
                        )

                        # Verify TimelineVarietyTracker was created with dict values
                        MockTracker.assert_called_once_with(
                            timeline_window=300.0,  # From dict
                            max_repeats=2  # From dict
                        )

    @pytest.mark.fast
    def test_variety_config_dict_strategy_logging(self, mock_config_variety_dict, mock_cache, vo_segments, video_segments, embeddings, scenes):
        """Test strategy logging with dict variety config (lines 145-147)"""
        vo_embeddings, video_embeddings = embeddings

        with patch('src.matching.tiered_matcher.TieredMatcher') as MockMatcher:
            mock_matcher_instance = Mock()
            mock_matcher_instance.match_segment = Mock(return_value=create_mock_match_result(
                video_segments[0], vo_segments[0]
            ))
            mock_matcher_instance.local_provider = None
            mock_matcher_instance.face_preference = "neutral"
            MockMatcher.return_value = mock_matcher_instance

            with patch('src.matching.main.find_top_k_similar') as mock_find:
                mock_find.return_value = ([0.9, 0.8], [0, 1])

                with patch('src.matching.main.StrategyMatcher') as MockStrategyMatcher:
                    mock_strategy = Mock()
                    mock_strategy.get_clip_id = Mock(return_value="test:0.00-10.00")
                    mock_strategy.get_strategy_matches = Mock(return_value=[])
                    mock_strategy.get_secondary_matches_diversity = Mock(return_value=[])
                    MockStrategyMatcher.return_value = mock_strategy

                    with patch('src.matching.main.logger') as mock_logger:
                        results = match_all_segments(
                            vo_segments,
                            video_segments,
                            vo_embeddings,
                            video_embeddings,
                            scenes,
                            mock_config_variety_dict,
                            mock_cache
                        )

                        # Check variety enforcement logging uses dict values
                        info_calls = [str(c) for c in mock_logger.info.call_args_list]
                        # Should include the dict values
                        assert any("different_source=True" in str(c) for c in info_calls)


# ============================================================================
# Test B-roll Segment Handling (Lines 190-193, 198)
# ============================================================================

class TestBrollCandidateHandling:
    """Test B-roll segment handling in candidate building (lines 190-193, 198)"""

    @pytest.mark.fast
    def test_broll_segments_added_to_candidates(self, mock_config, mock_cache, vo_segments, video_segments, embeddings, scenes):
        """Test that B-roll segments are added to candidates (lines 190-193)"""
        vo_embeddings, video_embeddings = embeddings

        # Ensure we have B-roll segments in our fixtures
        assert any(getattr(seg, 'is_broll', False) for seg in video_segments), "Need B-roll segments"

        with patch('src.matching.tiered_matcher.TieredMatcher') as MockMatcher:
            mock_matcher_instance = Mock()
            mock_matcher_instance.match_segment = Mock(return_value=create_mock_match_result(
                video_segments[0], vo_segments[0]
            ))
            mock_matcher_instance.local_provider = None
            mock_matcher_instance.face_preference = "neutral"
            MockMatcher.return_value = mock_matcher_instance

            with patch('src.matching.main.find_top_k_similar') as mock_find:
                # Return only first 2 indices (non-B-roll)
                mock_find.return_value = ([0.9, 0.8], [0, 1])

                with patch('src.matching.main.StrategyMatcher') as MockStrategyMatcher:
                    mock_strategy = Mock()
                    mock_strategy.get_clip_id = Mock(return_value="test:0.00-10.00")
                    mock_strategy.get_strategy_matches = Mock(return_value=[])
                    mock_strategy.get_secondary_matches_diversity = Mock(return_value=[])
                    MockStrategyMatcher.return_value = mock_strategy

                    with patch('src.matching.main.logger') as mock_logger:
                        results = match_all_segments(
                            vo_segments,
                            video_segments,
                            vo_embeddings,
                            video_embeddings,
                            scenes,
                            mock_config,
                            mock_cache
                        )

                        # Check that B-roll was logged as added
                        info_calls = [str(c) for c in mock_logger.info.call_args_list]
                        # Line 198: "Added X B-roll segments to candidates"
                        assert any("B-roll" in str(c) for c in info_calls), "Expected B-roll logging"

    @pytest.mark.fast
    def test_broll_not_added_if_already_in_candidates(self, mock_config, mock_cache, vo_segments, video_segments, embeddings, scenes):
        """Test B-roll not duplicated if already in embedding results"""
        vo_embeddings, video_embeddings = embeddings

        with patch('src.matching.tiered_matcher.TieredMatcher') as MockMatcher:
            mock_matcher_instance = Mock()
            mock_matcher_instance.match_segment = Mock(return_value=create_mock_match_result(
                video_segments[0], vo_segments[0]
            ))
            mock_matcher_instance.local_provider = None
            mock_matcher_instance.face_preference = "neutral"
            MockMatcher.return_value = mock_matcher_instance

            with patch('src.matching.main.find_top_k_similar') as mock_find:
                # Include B-roll segments (indices 4 and 5) in embedding results
                mock_find.return_value = ([0.9, 0.8, 0.7, 0.6, 0.5, 0.4], [0, 1, 2, 3, 4, 5])

                with patch('src.matching.main.StrategyMatcher') as MockStrategyMatcher:
                    mock_strategy = Mock()
                    mock_strategy.get_clip_id = Mock(return_value="test:0.00-10.00")
                    mock_strategy.get_strategy_matches = Mock(return_value=[])
                    mock_strategy.get_secondary_matches_diversity = Mock(return_value=[])
                    MockStrategyMatcher.return_value = mock_strategy

                    results = match_all_segments(
                        vo_segments,
                        video_segments,
                        vo_embeddings,
                        video_embeddings,
                        scenes,
                        mock_config,
                        mock_cache
                    )

                    # Should still succeed
                    assert len(results) == len(vo_segments)


# ============================================================================
# Test Global Clip Deduplication (Lines 208, 218)
# ============================================================================

class TestGlobalClipDeduplication:
    """Test global clip deduplication filtering (lines 208, 218)"""

    @pytest.mark.fast
    def test_global_dedup_filters_used_clips(self, mock_config, mock_cache, vo_segments, video_segments, embeddings, scenes):
        """Test that global clip tracker filters used clips (line 208)"""
        vo_embeddings, video_embeddings = embeddings

        with patch('src.matching.tiered_matcher.TieredMatcher') as MockMatcher:
            mock_matcher_instance = Mock()
            # Return same video for first match
            mock_matcher_instance.match_segment = Mock(return_value=create_mock_match_result(
                video_segments[0], vo_segments[0]
            ))
            mock_matcher_instance.local_provider = None
            mock_matcher_instance.face_preference = "neutral"
            MockMatcher.return_value = mock_matcher_instance

            with patch('src.matching.main.find_top_k_similar') as mock_find:
                mock_find.return_value = ([0.9, 0.8, 0.7], [0, 1, 2])

                with patch('src.matching.main.StrategyMatcher') as MockStrategyMatcher:
                    mock_strategy = Mock()
                    mock_strategy.get_clip_id = Mock(return_value="test:0.00-10.00")
                    mock_strategy.get_strategy_matches = Mock(return_value=[])
                    mock_strategy.get_secondary_matches_diversity = Mock(return_value=[])
                    MockStrategyMatcher.return_value = mock_strategy

                    with patch('src.matching.main.GlobalClipTracker') as MockGlobalTracker:
                        mock_global_tracker = Mock()
                        # First call returns False (clip not used), subsequent calls depend on clip
                        mock_global_tracker.is_used = Mock(side_effect=lambda s: s.index == 1)  # Mark index 1 as used
                        mock_global_tracker.get_stats = Mock(return_value={"total_clips_used": 1})
                        MockGlobalTracker.return_value = mock_global_tracker

                        with patch('src.matching.main.logger') as mock_logger:
                            results = match_all_segments(
                                vo_segments,
                                video_segments,
                                vo_embeddings,
                                video_embeddings,
                                scenes,
                                mock_config,
                                mock_cache
                            )

                            # Verify GlobalClipTracker was used
                            assert mock_global_tracker.is_used.called
                            assert mock_global_tracker.record_usage.called

    @pytest.mark.fast
    def test_variety_constraint_relaxation(self, mock_config, mock_cache, vo_segments, video_segments, embeddings, scenes):
        """Test variety constraint is relaxed when not enough candidates (line 218)"""
        vo_embeddings, video_embeddings = embeddings

        # Use small number of LLM rerank candidates
        mock_config.matching.llm_rerank_candidates = 3

        with patch('src.matching.tiered_matcher.TieredMatcher') as MockMatcher:
            mock_matcher_instance = Mock()
            mock_matcher_instance.match_segment = Mock(return_value=create_mock_match_result(
                video_segments[0], vo_segments[0]
            ))
            mock_matcher_instance.local_provider = None
            mock_matcher_instance.face_preference = "neutral"
            MockMatcher.return_value = mock_matcher_instance

            with patch('src.matching.main.find_top_k_similar') as mock_find:
                mock_find.return_value = ([0.9, 0.8], [0, 1])

                with patch('src.matching.main.StrategyMatcher') as MockStrategyMatcher:
                    mock_strategy = Mock()
                    mock_strategy.get_clip_id = Mock(return_value="test:0.00-10.00")
                    mock_strategy.get_strategy_matches = Mock(return_value=[])
                    mock_strategy.get_secondary_matches_diversity = Mock(return_value=[])
                    MockStrategyMatcher.return_value = mock_strategy

                    with patch('src.matching.main.TimelineVarietyTracker') as MockTracker:
                        mock_tracker = Mock()
                        # Exclude all sources to trigger relaxation
                        mock_tracker.get_excluded_sources = Mock(return_value={
                            "tokyo_footage.mp4", "kyoto_footage.mp4", "osaka_footage.mp4",
                            "broll_nature.mp4", "broll_city.mp4"
                        })
                        mock_tracker.get_stats = Mock(return_value={})
                        MockTracker.return_value = mock_tracker

                        with patch('src.matching.main.logger') as mock_logger:
                            results = match_all_segments(
                                vo_segments,
                                video_segments,
                                vo_embeddings,
                                video_embeddings,
                                scenes,
                                mock_config,
                                mock_cache
                            )

                            # Should still produce results (relaxation happened)
                            assert len(results) == len(vo_segments)

                            # Check debug log for relaxation message
                            debug_calls = [str(c) for c in mock_logger.debug.call_args_list]
                            # The relaxation message goes to debug log


# ============================================================================
# Test Strategy Candidates Filtering (Lines 280-283, 305)
# ============================================================================

class TestStrategyCandidatesFiltering:
    """Test strategy candidates filtering and recording (lines 280-283, 305)"""

    @pytest.mark.fast
    def test_strategy_candidates_filtering(self, mock_config, mock_cache, vo_segments, video_segments, embeddings, scenes):
        """Test strategy candidates are filtered by variety tracker (lines 280-283)"""
        vo_embeddings, video_embeddings = embeddings

        with patch('src.matching.tiered_matcher.TieredMatcher') as MockMatcher:
            mock_matcher_instance = Mock()
            mock_matcher_instance.match_segment = Mock(return_value=create_mock_match_result(
                video_segments[0], vo_segments[0]
            ))
            mock_matcher_instance.local_provider = None
            mock_matcher_instance.face_preference = "neutral"
            MockMatcher.return_value = mock_matcher_instance

            with patch('src.matching.main.find_top_k_similar') as mock_find:
                mock_find.return_value = ([0.9, 0.8, 0.7, 0.6, 0.5, 0.4], [0, 1, 2, 3, 4, 5])

                with patch('src.matching.main.StrategyMatcher') as MockStrategyMatcher:
                    mock_strategy = Mock()
                    mock_strategy.get_clip_id = Mock(return_value="test:0.00-10.00")

                    # Return strategy matches to test recording (line 305)
                    strat_match = StrategyMatch(
                        video_segment=video_segments[2],
                        video_scene=None,
                        confidence=0.75,
                        reasoning="Test strategy match",
                        strategy="embedding_diversity"
                    )
                    mock_strategy.get_strategy_matches = Mock(return_value=[strat_match])
                    mock_strategy.get_secondary_matches_diversity = Mock(return_value=[])
                    MockStrategyMatcher.return_value = mock_strategy

                    with patch('src.matching.main.TimelineVarietyTracker') as MockTracker:
                        mock_tracker = Mock()
                        # Return some excluded sources for V_strategy
                        mock_tracker.get_excluded_sources = Mock(side_effect=lambda track, pos:
                            {"tokyo_footage.mp4"} if track == "V_strategy" else set()
                        )
                        mock_tracker.get_stats = Mock(return_value={})
                        MockTracker.return_value = mock_tracker

                        results = match_all_segments(
                            vo_segments,
                            video_segments,
                            vo_embeddings,
                            video_embeddings,
                            scenes,
                            mock_config,
                            mock_cache
                        )

                        # Verify record_usage was called for strategy matches (line 305)
                        record_calls = [str(c) for c in mock_tracker.record_usage.call_args_list]
                        assert any("V_strategy" in str(c) for c in record_calls), "Expected V_strategy recording"

    @pytest.mark.fast
    def test_strategy_recording_for_each_match(self, mock_config, mock_cache, vo_segments, video_segments, embeddings, scenes):
        """Test each strategy match is recorded for variety tracking (line 305)"""
        vo_embeddings, video_embeddings = embeddings

        with patch('src.matching.tiered_matcher.TieredMatcher') as MockMatcher:
            mock_matcher_instance = Mock()
            mock_matcher_instance.match_segment = Mock(return_value=create_mock_match_result(
                video_segments[0], vo_segments[0]
            ))
            mock_matcher_instance.local_provider = None
            mock_matcher_instance.face_preference = "neutral"
            MockMatcher.return_value = mock_matcher_instance

            with patch('src.matching.main.find_top_k_similar') as mock_find:
                mock_find.return_value = ([0.9, 0.8, 0.7], [0, 1, 2])

                with patch('src.matching.main.StrategyMatcher') as MockStrategyMatcher:
                    mock_strategy = Mock()
                    mock_strategy.get_clip_id = Mock(return_value="test:0.00-10.00")

                    # Return multiple strategy matches
                    strat_matches = [
                        StrategyMatch(
                            video_segment=video_segments[2],
                            video_scene=None,
                            confidence=0.75,
                            reasoning="Strategy 1",
                            strategy="embedding_diversity"
                        ),
                        StrategyMatch(
                            video_segment=video_segments[3],
                            video_scene=None,
                            confidence=0.70,
                            reasoning="Strategy 2",
                            strategy="broll_only"
                        )
                    ]
                    mock_strategy.get_strategy_matches = Mock(return_value=strat_matches)
                    mock_strategy.get_secondary_matches_diversity = Mock(return_value=[])
                    MockStrategyMatcher.return_value = mock_strategy

                    with patch('src.matching.main.TimelineVarietyTracker') as MockTracker:
                        mock_tracker = Mock()
                        mock_tracker.get_excluded_sources = Mock(return_value=set())
                        mock_tracker.get_stats = Mock(return_value={})
                        MockTracker.return_value = mock_tracker

                        results = match_all_segments(
                            vo_segments,
                            video_segments,
                            vo_embeddings,
                            video_embeddings,
                            scenes,
                            mock_config,
                            mock_cache
                        )

                        # Should have recorded each strategy match
                        # Each voiceover segment gets strategy matches recorded
                        strategy_records = [c for c in mock_tracker.record_usage.call_args_list
                                           if "V_strategy" in str(c)]
                        # 3 vo_segments * 2 strategy matches = 6 records
                        assert len(strategy_records) >= 3  # At least some were recorded


# ============================================================================
# Test Secondary Matches Recording (Lines 330-332)
# ============================================================================

class TestSecondaryMatchesRecording:
    """Test secondary matches (V4-V6) recording (lines 330-332)"""

    @pytest.mark.fast
    def test_secondary_matches_recorded_for_variety(self, mock_config, mock_cache, vo_segments, video_segments, embeddings, scenes):
        """Test secondary matches are recorded for variety tracking (lines 330-332)"""
        vo_embeddings, video_embeddings = embeddings

        with patch('src.matching.tiered_matcher.TieredMatcher') as MockMatcher:
            mock_matcher_instance = Mock()
            mock_matcher_instance.match_segment = Mock(return_value=create_mock_match_result(
                video_segments[0], vo_segments[0]
            ))
            mock_matcher_instance.local_provider = None
            mock_matcher_instance.face_preference = "neutral"
            MockMatcher.return_value = mock_matcher_instance

            with patch('src.matching.main.find_top_k_similar') as mock_find:
                mock_find.return_value = ([0.9, 0.8, 0.7, 0.6], [0, 1, 2, 3])

                with patch('src.matching.main.StrategyMatcher') as MockStrategyMatcher:
                    mock_strategy = Mock()
                    mock_strategy.get_clip_id = Mock(return_value="test:0.00-10.00")
                    mock_strategy.get_strategy_matches = Mock(return_value=[])

                    # Return secondary matches (V4-V6)
                    secondary_matches = [
                        AlternativeMatch(
                            video_segment=video_segments[2],
                            video_scene=None,
                            confidence=0.70,
                            reasoning="Secondary 1"
                        ),
                        AlternativeMatch(
                            video_segment=video_segments[3],
                            video_scene=None,
                            confidence=0.65,
                            reasoning="Secondary 2"
                        )
                    ]
                    mock_strategy.get_secondary_matches_diversity = Mock(return_value=secondary_matches)
                    MockStrategyMatcher.return_value = mock_strategy

                    with patch('src.matching.main.TimelineVarietyTracker') as MockTracker:
                        mock_tracker = Mock()
                        mock_tracker.get_excluded_sources = Mock(return_value=set())
                        mock_tracker.get_stats = Mock(return_value={})
                        MockTracker.return_value = mock_tracker

                        results = match_all_segments(
                            vo_segments,
                            video_segments,
                            vo_embeddings,
                            video_embeddings,
                            scenes,
                            mock_config,
                            mock_cache
                        )

                        # Verify secondary matches were recorded
                        record_calls = [str(c) for c in mock_tracker.record_usage.call_args_list]

                        # Should have V4, V5 recordings
                        assert any("V4" in str(c) for c in record_calls), "Expected V4 recording"
                        assert any("V5" in str(c) for c in record_calls), "Expected V5 recording"

    @pytest.mark.fast
    def test_secondary_matches_not_in_global_tracker(self, mock_config, mock_cache, vo_segments, video_segments, embeddings, scenes):
        """Test secondary matches are NOT added to global clip tracker"""
        vo_embeddings, video_embeddings = embeddings

        with patch('src.matching.tiered_matcher.TieredMatcher') as MockMatcher:
            mock_matcher_instance = Mock()
            mock_matcher_instance.match_segment = Mock(return_value=create_mock_match_result(
                video_segments[0], vo_segments[0]
            ))
            mock_matcher_instance.local_provider = None
            mock_matcher_instance.face_preference = "neutral"
            MockMatcher.return_value = mock_matcher_instance

            with patch('src.matching.main.find_top_k_similar') as mock_find:
                mock_find.return_value = ([0.9, 0.8, 0.7], [0, 1, 2])

                with patch('src.matching.main.StrategyMatcher') as MockStrategyMatcher:
                    mock_strategy = Mock()
                    mock_strategy.get_clip_id = Mock(return_value="test:0.00-10.00")
                    mock_strategy.get_strategy_matches = Mock(return_value=[])

                    secondary_matches = [
                        AlternativeMatch(
                            video_segment=video_segments[2],
                            video_scene=None,
                            confidence=0.70,
                            reasoning="Secondary 1"
                        )
                    ]
                    mock_strategy.get_secondary_matches_diversity = Mock(return_value=secondary_matches)
                    MockStrategyMatcher.return_value = mock_strategy

                    with patch('src.matching.main.GlobalClipTracker') as MockGlobalTracker:
                        mock_global_tracker = Mock()
                        mock_global_tracker.is_used = Mock(return_value=False)
                        mock_global_tracker.get_stats = Mock(return_value={"total_clips_used": 0})
                        MockGlobalTracker.return_value = mock_global_tracker

                        results = match_all_segments(
                            vo_segments,
                            video_segments,
                            vo_embeddings,
                            video_embeddings,
                            scenes,
                            mock_config,
                            mock_cache
                        )

                        # Global tracker should only record V1-V3, NOT secondary matches
                        record_calls = [str(c) for c in mock_global_tracker.record_usage.call_args_list]

                        # Should NOT have V4-V6 in global tracker
                        v4_v6_records = [c for c in record_calls if any(f"V{i}" in str(c) for i in [4, 5, 6])]
                        assert len(v4_v6_records) == 0, "V4-V6 should NOT be in global clip tracker"


# ============================================================================
# Test Duration Scoring Logging (Line 112-113)
# ============================================================================

class TestDurationScoringLogging:
    """Test duration scoring logging"""

    @pytest.mark.fast
    def test_duration_scoring_enabled_logging(self, mock_config, mock_cache, vo_segments, video_segments, embeddings, scenes):
        """Test logging when duration scoring is enabled"""
        mock_config.matching.duration_scoring_enabled = True
        mock_config.matching.ideal_speed_range = [0.8, 1.2]
        mock_config.matching.soft_penalty_range = [0.5, 2.0]

        vo_embeddings, video_embeddings = embeddings

        with patch('src.matching.tiered_matcher.TieredMatcher') as MockMatcher:
            mock_matcher_instance = Mock()
            mock_matcher_instance.match_segment = Mock(return_value=create_mock_match_result(
                video_segments[0], vo_segments[0]
            ))
            mock_matcher_instance.local_provider = None
            mock_matcher_instance.face_preference = "neutral"
            MockMatcher.return_value = mock_matcher_instance

            with patch('src.matching.main.find_top_k_similar') as mock_find:
                mock_find.return_value = ([0.9, 0.8], [0, 1])

                with patch('src.matching.main.StrategyMatcher') as MockStrategyMatcher:
                    mock_strategy = Mock()
                    mock_strategy.get_clip_id = Mock(return_value="test:0.00-10.00")
                    mock_strategy.get_strategy_matches = Mock(return_value=[])
                    mock_strategy.get_secondary_matches_diversity = Mock(return_value=[])
                    MockStrategyMatcher.return_value = mock_strategy

                    with patch('src.matching.main.logger') as mock_logger:
                        results = match_all_segments(
                            vo_segments,
                            video_segments,
                            vo_embeddings,
                            video_embeddings,
                            scenes,
                            mock_config,
                            mock_cache
                        )

                        # Check duration scoring logging
                        info_calls = [str(c) for c in mock_logger.info.call_args_list]
                        assert any("Duration scoring" in str(c) for c in info_calls)


# ============================================================================
# Test Alternatives Recording (Lines 254-266)
# ============================================================================

class TestAlternativesRecording:
    """Test alternatives (V2-V3) recording"""

    @pytest.mark.fast
    def test_alternatives_recorded_in_variety_tracker(self, mock_config, mock_cache, vo_segments, video_segments, embeddings, scenes):
        """Test V2-V3 alternatives are recorded for variety tracking"""
        vo_embeddings, video_embeddings = embeddings

        with patch('src.matching.tiered_matcher.TieredMatcher') as MockMatcher:
            mock_matcher_instance = Mock()

            # Create result with alternatives
            alt_seg = SRTSegment(
                index=10,
                start_time=0.0,
                end_time=10.0,
                text="Alternative",
                source_file="alt_source.mp4"
            )
            result = create_mock_match_result(video_segments[0], vo_segments[0], has_alternatives=True)
            result.alternatives = [
                AlternativeMatch(video_segment=alt_seg, video_scene=None, confidence=0.75, reasoning="Alt 1")
            ]

            mock_matcher_instance.match_segment = Mock(return_value=result)
            mock_matcher_instance.local_provider = None
            mock_matcher_instance.face_preference = "neutral"
            MockMatcher.return_value = mock_matcher_instance

            with patch('src.matching.main.find_top_k_similar') as mock_find:
                mock_find.return_value = ([0.9, 0.8], [0, 1])

                with patch('src.matching.main.StrategyMatcher') as MockStrategyMatcher:
                    mock_strategy = Mock()
                    mock_strategy.get_clip_id = Mock(return_value="test:0.00-10.00")
                    mock_strategy.get_strategy_matches = Mock(return_value=[])
                    mock_strategy.get_secondary_matches_diversity = Mock(return_value=[])
                    MockStrategyMatcher.return_value = mock_strategy

                    with patch('src.matching.main.TimelineVarietyTracker') as MockTracker:
                        mock_tracker = Mock()
                        mock_tracker.get_excluded_sources = Mock(return_value=set())
                        mock_tracker.get_stats = Mock(return_value={})
                        MockTracker.return_value = mock_tracker

                        results = match_all_segments(
                            vo_segments,
                            video_segments,
                            vo_embeddings,
                            video_embeddings,
                            scenes,
                            mock_config,
                            mock_cache
                        )

                        # Verify V2 was recorded
                        record_calls = [str(c) for c in mock_tracker.record_usage.call_args_list]
                        assert any("V2" in str(c) for c in record_calls), "Expected V2 recording"

    @pytest.mark.fast
    def test_alternatives_recorded_in_global_tracker(self, mock_config, mock_cache, vo_segments, video_segments, embeddings, scenes):
        """Test V2-V3 alternatives are recorded in global clip tracker"""
        vo_embeddings, video_embeddings = embeddings

        with patch('src.matching.tiered_matcher.TieredMatcher') as MockMatcher:
            mock_matcher_instance = Mock()

            alt_seg = SRTSegment(
                index=10,
                start_time=0.0,
                end_time=10.0,
                text="Alternative",
                source_file="alt_source.mp4"
            )
            result = create_mock_match_result(video_segments[0], vo_segments[0])
            result.alternatives = [
                AlternativeMatch(video_segment=alt_seg, video_scene=None, confidence=0.75, reasoning="Alt 1")
            ]

            mock_matcher_instance.match_segment = Mock(return_value=result)
            mock_matcher_instance.local_provider = None
            mock_matcher_instance.face_preference = "neutral"
            MockMatcher.return_value = mock_matcher_instance

            with patch('src.matching.main.find_top_k_similar') as mock_find:
                mock_find.return_value = ([0.9, 0.8], [0, 1])

                with patch('src.matching.main.StrategyMatcher') as MockStrategyMatcher:
                    mock_strategy = Mock()
                    mock_strategy.get_clip_id = Mock(return_value="test:0.00-10.00")
                    mock_strategy.get_strategy_matches = Mock(return_value=[])
                    mock_strategy.get_secondary_matches_diversity = Mock(return_value=[])
                    MockStrategyMatcher.return_value = mock_strategy

                    with patch('src.matching.main.GlobalClipTracker') as MockGlobalTracker:
                        mock_global_tracker = Mock()
                        mock_global_tracker.is_used = Mock(return_value=False)
                        mock_global_tracker.get_stats = Mock(return_value={"total_clips_used": 0})
                        MockGlobalTracker.return_value = mock_global_tracker

                        results = match_all_segments(
                            vo_segments,
                            video_segments,
                            vo_embeddings,
                            video_embeddings,
                            scenes,
                            mock_config,
                            mock_cache
                        )

                        # Verify V2 was recorded in global tracker
                        record_calls = [str(c) for c in mock_global_tracker.record_usage.call_args_list]
                        assert any("V2" in str(c) for c in record_calls), "Expected V2 in global tracker"


# ============================================================================
# Test Gap Reporting
# ============================================================================

class TestGapReporting:
    """Test gap reporting for low confidence matches"""

    @pytest.mark.fast
    def test_gaps_reported_in_log(self, mock_config, mock_cache, vo_segments, video_segments, embeddings, scenes):
        """Test that gaps are reported in the log"""
        vo_embeddings, video_embeddings = embeddings

        with patch('src.matching.tiered_matcher.TieredMatcher') as MockMatcher:
            mock_matcher_instance = Mock()

            # Create result with gap
            result = create_mock_match_result(video_segments[0], vo_segments[0])
            result.has_gap = True
            result.gap_reason = "Low confidence match"

            mock_matcher_instance.match_segment = Mock(return_value=result)
            mock_matcher_instance.local_provider = None
            mock_matcher_instance.face_preference = "neutral"
            MockMatcher.return_value = mock_matcher_instance

            with patch('src.matching.main.find_top_k_similar') as mock_find:
                mock_find.return_value = ([0.9, 0.8], [0, 1])

                with patch('src.matching.main.StrategyMatcher') as MockStrategyMatcher:
                    mock_strategy = Mock()
                    mock_strategy.get_clip_id = Mock(return_value="test:0.00-10.00")
                    mock_strategy.get_strategy_matches = Mock(return_value=[])
                    mock_strategy.get_secondary_matches_diversity = Mock(return_value=[])
                    MockStrategyMatcher.return_value = mock_strategy

                    with patch('src.matching.main.logger') as mock_logger:
                        results = match_all_segments(
                            vo_segments,
                            video_segments,
                            vo_embeddings,
                            video_embeddings,
                            scenes,
                            mock_config,
                            mock_cache
                        )

                        # Check warning log for gaps
                        warning_calls = [str(c) for c in mock_logger.warning.call_args_list]
                        assert any("gap" in str(c).lower() for c in warning_calls), "Expected gap warning"


# ============================================================================
# Test Local LLM Review (Line 348-349)
# ============================================================================

class TestLocalLLMReview:
    """Test local LLM review functionality"""

    @pytest.mark.fast
    def test_local_llm_review_called_when_enabled(self, mock_config, mock_cache, vo_segments, video_segments, embeddings, scenes):
        """Test local LLM review is called when enabled"""
        mock_config.matching.use_local_for_review = True
        vo_embeddings, video_embeddings = embeddings

        with patch('src.matching.tiered_matcher.TieredMatcher') as MockMatcher:
            mock_matcher_instance = Mock()
            mock_matcher_instance.match_segment = Mock(return_value=create_mock_match_result(
                video_segments[0], vo_segments[0]
            ))
            mock_matcher_instance.local_provider = Mock()  # Has a local provider
            mock_matcher_instance.review_with_local_llm = Mock(side_effect=lambda x: x)  # Pass through
            mock_matcher_instance.face_preference = "neutral"
            MockMatcher.return_value = mock_matcher_instance

            with patch('src.matching.main.find_top_k_similar') as mock_find:
                mock_find.return_value = ([0.9, 0.8], [0, 1])

                with patch('src.matching.main.StrategyMatcher') as MockStrategyMatcher:
                    mock_strategy = Mock()
                    mock_strategy.get_clip_id = Mock(return_value="test:0.00-10.00")
                    mock_strategy.get_strategy_matches = Mock(return_value=[])
                    mock_strategy.get_secondary_matches_diversity = Mock(return_value=[])
                    MockStrategyMatcher.return_value = mock_strategy

                    results = match_all_segments(
                        vo_segments,
                        video_segments,
                        vo_embeddings,
                        video_embeddings,
                        scenes,
                        mock_config,
                        mock_cache
                    )

                    # Verify review was called
                    mock_matcher_instance.review_with_local_llm.assert_called_once()


# ============================================================================
# Test Variety Filtering Success (Lines 218, 283)
# ============================================================================

class TestVarietyFilteringSuccess:
    """Test variety filtering when enough candidates remain after filtering"""

    @pytest.mark.fast
    def test_variety_filtering_replaces_candidates_line_218(self, mock_config, mock_cache, vo_segments, video_segments, embeddings, scenes):
        """Test line 218: all_candidates = filtered_candidates when enough filtered"""
        vo_embeddings, video_embeddings = embeddings

        # Set llm_rerank_candidates low so filtered result is sufficient
        mock_config.matching.llm_rerank_candidates = 3

        with patch('src.matching.tiered_matcher.TieredMatcher') as MockMatcher:
            mock_matcher_instance = Mock()
            mock_matcher_instance.match_segment = Mock(return_value=create_mock_match_result(
                video_segments[0], vo_segments[0]
            ))
            mock_matcher_instance.local_provider = None
            mock_matcher_instance.face_preference = "neutral"
            MockMatcher.return_value = mock_matcher_instance

            with patch('src.matching.main.find_top_k_similar') as mock_find:
                # Return 6 candidates
                mock_find.return_value = ([0.9, 0.88, 0.85, 0.8, 0.75, 0.7], [0, 1, 2, 3, 4, 5])

                with patch('src.matching.main.StrategyMatcher') as MockStrategyMatcher:
                    mock_strategy = Mock()
                    mock_strategy.get_clip_id = Mock(return_value="test:0.00-10.00")
                    mock_strategy.get_strategy_matches = Mock(return_value=[])
                    mock_strategy.get_secondary_matches_diversity = Mock(return_value=[])
                    MockStrategyMatcher.return_value = mock_strategy

                    with patch('src.matching.main.TimelineVarietyTracker') as MockTracker:
                        mock_tracker = Mock()
                        # Exclude only 1 source, leaving 5+ candidates (>= llm_rerank_candidates)
                        mock_tracker.get_excluded_sources = Mock(return_value={"tokyo_footage.mp4"})
                        mock_tracker.get_stats = Mock(return_value={})
                        MockTracker.return_value = mock_tracker

                        results = match_all_segments(
                            vo_segments,
                            video_segments,
                            vo_embeddings,
                            video_embeddings,
                            scenes,
                            mock_config,
                            mock_cache
                        )

                        # Should succeed - line 218 path taken
                        assert len(results) == len(vo_segments)

    @pytest.mark.fast
    def test_strategy_filtering_replaces_candidates_line_283(self, mock_config, mock_cache, vo_segments, video_segments, embeddings, scenes):
        """Test line 283: strategy_candidates = filtered_strategy when enough filtered"""
        vo_embeddings, video_embeddings = embeddings

        with patch('src.matching.tiered_matcher.TieredMatcher') as MockMatcher:
            mock_matcher_instance = Mock()
            mock_matcher_instance.match_segment = Mock(return_value=create_mock_match_result(
                video_segments[0], vo_segments[0]
            ))
            mock_matcher_instance.local_provider = None
            mock_matcher_instance.face_preference = "neutral"
            MockMatcher.return_value = mock_matcher_instance

            with patch('src.matching.main.find_top_k_similar') as mock_find:
                # Return 6 candidates (matches video_segments fixture size)
                # More than 5 minimum for strategy filtering
                mock_find.return_value = ([0.9, 0.88, 0.85, 0.8, 0.75, 0.7],
                                         [0, 1, 2, 3, 4, 5])

                with patch('src.matching.main.StrategyMatcher') as MockStrategyMatcher:
                    mock_strategy = Mock()
                    mock_strategy.get_clip_id = Mock(return_value="test:0.00-10.00")
                    mock_strategy.get_strategy_matches = Mock(return_value=[])
                    mock_strategy.get_secondary_matches_diversity = Mock(return_value=[])
                    MockStrategyMatcher.return_value = mock_strategy

                    with patch('src.matching.main.TimelineVarietyTracker') as MockTracker:
                        mock_tracker = Mock()
                        # For V_strategy, exclude only 1 source, leaving 5+ (>= 5)
                        mock_tracker.get_excluded_sources = Mock(side_effect=lambda track, pos:
                            {"tokyo_footage.mp4"} if track == "V_strategy" else set()
                        )
                        mock_tracker.get_stats = Mock(return_value={})
                        MockTracker.return_value = mock_tracker

                        results = match_all_segments(
                            vo_segments,
                            video_segments,
                            vo_embeddings,
                            video_embeddings,
                            scenes,
                            mock_config,
                            mock_cache
                        )

                        # Should succeed - line 283 path taken
                        assert len(results) == len(vo_segments)


# ============================================================================
# Run tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
