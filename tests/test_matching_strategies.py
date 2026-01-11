"""
Comprehensive tests for matching/strategies.py

Tests all 6 matching strategies and variety enforcement logic.
"""

import pytest
import sys
from unittest.mock import Mock, MagicMock
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.matching.strategies import StrategyMatcher
from src.utils import SRTSegment, SceneInfo, StrategyMatch, AlternativeMatch


@pytest.fixture
def mock_config():
    """Create mock config with variety settings"""
    config = Mock()
    config.output = Mock()
    config.output.variety = Mock()
    config.output.variety.exclude_same_clip = True
    config.output.variety.require_different_source = True
    config.output.variety.min_time_distance = 10.0
    config.output.variety.min_embedding_distance = 0.3
    config.output.variety.enforce_timeline_variety = True
    config.output.variety.timeline_variety_window = 600.0
    config.output.variety.max_source_repeats_in_window = 1
    config.output.include_strategy_tracks = True
    config.output.strategy_tracks = ["visual_first", "different_source", "keyword_only",
                                    "embedding_diversity", "broll_only", "source_rotation"]
    return config


@pytest.fixture
def mock_scenes():
    """Create mock scene data"""
    return {
        "/video1.mp4": [
            SceneInfo(
                video_path="/video1.mp4",
                scene_index=0,
                start_time=0.0,
                end_time=10.0,
                description="earthquake damage in city",
                visual_keywords=["earthquake", "damage", "city", "destruction"]
            ),
            SceneInfo(
                video_path="/video1.mp4",
                scene_index=1,
                start_time=10.0,
                end_time=20.0,
                description="rescue workers helping people",
                visual_keywords=["rescue", "workers", "people", "help"]
            )
        ],
        "/video2.mp4": [
            SceneInfo(
                video_path="/video2.mp4",
                scene_index=0,
                start_time=0.0,
                end_time=15.0,
                description="tsunami waves hitting coast",
                visual_keywords=["tsunami", "waves", "water", "coast"]
            )
        ]
    }


@pytest.fixture
def sample_segment():
    """Create sample SRTSegment"""
    seg = SRTSegment(
        index=0,
        start_time=0.0,
        end_time=5.0,
        text="This is a test segment about earthquakes",
        source_file="/video1.mp4"
    )
    seg.keywords = ["earthquake", "disaster", "damage"]
    seg.entities = ["California", "San Francisco"]
    return seg


@pytest.fixture
def sample_vo_segment():
    """Create sample voiceover segment"""
    vo = SRTSegment(
        index=0,
        start_time=0.0,
        end_time=5.0,
        text="The earthquake caused massive destruction in the city",
        source_file=""
    )
    vo.keywords = ["earthquake", "destruction", "city"]
    vo.entities = [{"text": "California", "type": "LOCATION"}]
    return vo


@pytest.fixture
def sample_candidates(sample_segment):
    """Create sample candidate list"""
    seg1 = SRTSegment(0, 0.0, 5.0, "earthquake damage in urban area", "/video1.mp4")
    seg1.keywords = ["earthquake", "damage", "urban"]
    seg1.entities = []

    seg2 = SRTSegment(0, 10.0, 15.0, "tsunami waves crashing", "/video2.mp4")
    seg2.keywords = ["tsunami", "waves", "water"]
    seg2.entities = []

    seg3 = SRTSegment(1, 5.0, 10.0, "people evacuating buildings", "/video1.mp4")
    seg3.keywords = ["people", "evacuation", "buildings"]
    seg3.entities = []

    # Return list of (segment, similarity_score) tuples
    return [
        (seg1, 0.85),
        (seg2, 0.72),
        (seg3, 0.68)
    ]


class TestStrategyMatcherInit:
    """Test StrategyMatcher initialization"""

    def test_init_with_scenes(self, mock_config, mock_scenes):
        """Test initialization with scene data"""
        matcher = StrategyMatcher(mock_config, mock_scenes)

        assert matcher.config == mock_config
        assert matcher.scenes == mock_scenes
        assert len(matcher.scenes) == 2

    def test_init_without_scenes(self, mock_config):
        """Test initialization without scene data"""
        matcher = StrategyMatcher(mock_config, None)

        assert matcher.config == mock_config
        assert matcher.scenes == {}

    def test_init_with_dict_variety_config(self):
        """Test initialization when variety config is dict"""
        config = Mock()
        config.output = Mock()
        config.output.variety = {
            'exclude_same_clip': False,
            'require_different_source': False,
            'min_time_distance': 5.0,
            'min_embedding_distance': 0.2
        }

        matcher = StrategyMatcher(config, None)

        # Should convert dict to wrapper object
        assert hasattr(matcher.variety_config, 'exclude_same_clip')
        assert matcher.variety_config.exclude_same_clip == False
        assert matcher.variety_config.require_different_source == False
        assert matcher.variety_config.min_time_distance == 5.0


class TestClipIdentification:
    """Test clip ID generation and exclusion logic"""

    def test_get_clip_id(self, mock_config):
        """Test unique clip ID generation"""
        matcher = StrategyMatcher(mock_config, None)

        seg = SRTSegment(0, 10.5, 15.3, "test", "/video1.mp4")
        clip_id = matcher.get_clip_id(seg)

        assert clip_id == "/video1.mp4:10.50-15.30"

    def test_is_clip_excluded_same_clip(self, mock_config):
        """Test exclusion of same clip"""
        matcher = StrategyMatcher(mock_config, None)

        seg1 = SRTSegment(0, 10.0, 15.0, "test", "/video1.mp4")
        seg2 = SRTSegment(0, 10.0, 15.0, "test", "/video1.mp4")  # Same clip

        is_excluded, reason = matcher.is_clip_excluded(seg2, [seg1])

        assert is_excluded == True
        assert "same clip" in reason.lower()

    def test_is_clip_excluded_same_source_forced(self, mock_config):
        """Test forced different source exclusion"""
        matcher = StrategyMatcher(mock_config, None)

        seg1 = SRTSegment(0, 10.0, 15.0, "test", "/video1.mp4")
        seg2 = SRTSegment(1, 20.0, 25.0, "different", "/video1.mp4")  # Same source

        is_excluded, reason = matcher.is_clip_excluded(
            seg2, [seg1], force_different_source=True
        )

        assert is_excluded == True
        assert "source" in reason.lower()

    def test_is_clip_excluded_time_distance(self, mock_config):
        """Test time distance exclusion"""
        matcher = StrategyMatcher(mock_config, None)

        seg1 = SRTSegment(0, 10.0, 15.0, "test", "/video1.mp4")
        seg2 = SRTSegment(1, 12.0, 17.0, "close", "/video1.mp4")  # 2s apart

        is_excluded, reason = matcher.is_clip_excluded(seg2, [seg1])

        assert is_excluded == True
        assert "time" in reason.lower()

    def test_is_clip_excluded_embedding_distance(self, mock_config):
        """Test embedding similarity exclusion"""
        matcher = StrategyMatcher(mock_config, None)

        seg1 = SRTSegment(0, 10.0, 15.0, "test", "/video1.mp4")
        seg2 = SRTSegment(1, 30.0, 35.0, "far", "/video2.mp4")  # Different source

        # Very similar embeddings (distance = 0.1)
        emb1 = [1.0, 0.5, 0.3]
        emb2 = [0.95, 0.48, 0.32]

        is_excluded, reason = matcher.is_clip_excluded(
            seg2, [seg1],
            existing_embeddings=[emb1],
            candidate_embedding=emb2
        )

        assert is_excluded == True
        assert "similar" in reason.lower()

    def test_is_clip_excluded_passes_all_checks(self, mock_config):
        """Test candidate that passes all exclusion checks"""
        matcher = StrategyMatcher(mock_config, None)

        seg1 = SRTSegment(0, 10.0, 15.0, "test", "/video1.mp4")
        seg2 = SRTSegment(1, 30.0, 35.0, "different", "/video2.mp4")  # Different source, far apart

        is_excluded, reason = matcher.is_clip_excluded(seg2, [seg1])

        assert is_excluded == False
        assert reason == ""


class TestVisualFirstStrategy:
    """Test visual_first matching strategy"""

    def test_visual_first_with_scene_description(self, mock_config, mock_scenes, sample_vo_segment, sample_candidates):
        """Test visual matching using scene descriptions"""
        matcher = StrategyMatcher(mock_config, mock_scenes)

        # Voiceover mentions "earthquake" and "city"
        sample_vo_segment.text = "The earthquake destroyed buildings in the city"
        sample_vo_segment.keywords = ["earthquake", "city", "buildings"]

        result = matcher.match_visual_first(
            sample_vo_segment,
            sample_candidates,
            [],  # No existing matches
            None,  # No embeddings
            {}  # No candidate embeddings
        )

        assert result is not None
        assert result.strategy == "visual_first"
        assert result.confidence > 0
        assert "visual" in result.reasoning.lower()

    def test_visual_first_fallback_to_filename(self, mock_config, sample_vo_segment, sample_candidates):
        """Test visual matching falls back to filename when no scenes"""
        matcher = StrategyMatcher(mock_config, None)  # No scenes

        sample_vo_segment.text = "The earthquake caused damage"
        sample_vo_segment.keywords = ["earthquake", "damage"]

        result = matcher.match_visual_first(
            sample_vo_segment,
            sample_candidates,
            [],
            None,
            {}
        )

        # Should still return a match using text/filename matching
        assert result is not None or len(sample_candidates) == 0

    def test_visual_first_respects_variety_rules(self, mock_config, mock_scenes, sample_vo_segment, sample_candidates):
        """Test visual_first enforces different source"""
        matcher = StrategyMatcher(mock_config, mock_scenes)

        # Mark first candidate as already used
        used_seg = sample_candidates[0][0]

        result = matcher.match_visual_first(
            sample_vo_segment,
            sample_candidates,
            [used_seg],  # Already used
            None,
            {}
        )

        # Should skip the used segment and pick a different one
        if result:
            assert result.video_segment.source_file != used_seg.source_file


class TestDifferentSourceStrategy:
    """Test different_source matching strategy"""

    def test_different_source_finds_new_video(self, mock_config, sample_vo_segment, sample_candidates):
        """Test different_source picks from unused video"""
        matcher = StrategyMatcher(mock_config, None)

        # Mark video1 as already used
        used_seg = SRTSegment(0, 0.0, 5.0, "used", "/video1.mp4")

        result = matcher.match_different_source(
            sample_vo_segment,
            sample_candidates,
            [used_seg],
            None,
            {}
        )

        assert result is not None
        assert result.strategy == "different_source"
        assert result.video_segment.source_file != "/video1.mp4"
        assert "different source" in result.reasoning.lower()

    def test_different_source_fallback(self, mock_config, sample_vo_segment):
        """Test fallback when no different source available"""
        matcher = StrategyMatcher(mock_config, None)

        # All candidates from same source
        candidates = [
            (SRTSegment(0, 0.0, 5.0, "test1", "/video1.mp4"), 0.8),
            (SRTSegment(1, 10.0, 15.0, "test2", "/video1.mp4"), 0.7),
        ]

        used_seg = SRTSegment(0, 20.0, 25.0, "used", "/video1.mp4")

        result = matcher.match_different_source(
            sample_vo_segment,
            candidates,
            [used_seg],
            None,
            {}
        )

        # Should fallback with penalty
        if result:
            assert result.confidence < 0.8  # Penalty applied
            assert "fallback" in result.reasoning.lower()


class TestKeywordOnlyStrategy:
    """Test keyword_only matching strategy"""

    def test_keyword_only_with_keywords(self, mock_config, sample_vo_segment, sample_candidates):
        """Test keyword matching with populated keywords"""
        matcher = StrategyMatcher(mock_config, None)

        sample_vo_segment.keywords = ["earthquake", "damage", "disaster"]
        sample_vo_segment.entities = [{"text": "California", "type": "LOCATION"}]

        # Set keywords on candidates
        sample_candidates[0][0].keywords = ["earthquake", "damage", "buildings"]
        sample_candidates[1][0].keywords = ["tsunami", "waves"]
        sample_candidates[2][0].keywords = ["evacuation", "people"]

        result = matcher.match_keyword_only(
            sample_vo_segment,
            sample_candidates,
            [],
            None,
            {}
        )

        assert result is not None
        assert result.strategy == "keyword_only"
        assert "keyword" in result.reasoning.lower()
        # Should match first candidate (earthquake + damage overlap)
        assert result.video_segment.keywords[0] in sample_vo_segment.keywords

    def test_keyword_only_fallback_to_words(self, mock_config, sample_candidates):
        """Test keyword matching falls back to word extraction"""
        matcher = StrategyMatcher(mock_config, None)

        # Voiceover with no keywords populated
        vo = SRTSegment(0, 0.0, 5.0, "earthquake destruction aftermath damage severe", "")
        vo.keywords = []
        vo.entities = []

        # Candidates also without keywords
        for seg, _ in sample_candidates:
            seg.keywords = []
            seg.entities = []

        result = matcher.match_keyword_only(
            vo,
            sample_candidates,
            [],
            None,
            {}
        )

        # Should extract words and still find matches
        assert result is not None or len(sample_candidates) == 0

    def test_keyword_only_respects_variety(self, mock_config, sample_vo_segment, sample_candidates):
        """Test keyword_only enforces different source"""
        matcher = StrategyMatcher(mock_config, None)

        sample_vo_segment.keywords = ["earthquake", "damage"]

        # Mark first candidate (highest keyword overlap) as used
        used_seg = sample_candidates[0][0]

        result = matcher.match_keyword_only(
            sample_vo_segment,
            sample_candidates,
            [used_seg],
            None,
            {}
        )

        # Should skip used segment
        if result:
            assert result.video_segment != used_seg


class TestEmbeddingDiversityStrategy:
    """Test embedding_diversity matching strategy"""

    def test_embedding_diversity_finds_diverse_match(self, mock_config, sample_vo_segment, sample_candidates):
        """Test embedding diversity picks maximally different clip"""
        matcher = StrategyMatcher(mock_config, None)

        vo_emb = [1.0, 0.0, 0.0]  # Voiceover embedding

        # Existing matches with similar embeddings
        existing_embs = [
            [0.9, 0.1, 0.0],  # Very similar to vo
            [0.8, 0.2, 0.0]   # Similar to vo
        ]

        # Candidate embeddings
        candidate_embs = {
            matcher.get_clip_id(sample_candidates[0][0]): [0.85, 0.15, 0.0],  # Similar
            matcher.get_clip_id(sample_candidates[1][0]): [0.2, 0.3, 0.5],   # More diverse
            matcher.get_clip_id(sample_candidates[2][0]): [0.1, 0.1, 0.8]    # Most diverse
        }

        existing_segs = [SRTSegment(0, 0.0, 5.0, "test", "/video1.mp4")]

        result = matcher.match_embedding_diversity(
            sample_vo_segment,
            sample_candidates,
            existing_segs,
            existing_embs,
            candidate_embs,
            vo_emb
        )

        assert result is not None
        assert result.strategy == "embedding_diversity"
        assert "diversity" in result.reasoning.lower()

    def test_embedding_diversity_requires_embeddings(self, mock_config, sample_vo_segment, sample_candidates):
        """Test embedding diversity returns None without embeddings"""
        matcher = StrategyMatcher(mock_config, None)

        result = matcher.match_embedding_diversity(
            sample_vo_segment,
            sample_candidates,
            [],
            [],  # No existing embeddings
            {},  # No candidate embeddings
            []   # No vo embedding
        )

        assert result is None

    def test_embedding_diversity_enforces_different_source(self, mock_config, sample_vo_segment, sample_candidates):
        """Test embedding diversity requires different source from V1-V3"""
        matcher = StrategyMatcher(mock_config, None)

        vo_emb = [1.0, 0.0, 0.0]
        existing_embs = [[0.9, 0.1, 0.0]]

        # Mark video1 as used
        existing_segs = [SRTSegment(0, 0.0, 5.0, "test", "/video1.mp4")]

        candidate_embs = {
            matcher.get_clip_id(sample_candidates[0][0]): [0.5, 0.5, 0.0],  # video1 - excluded
            matcher.get_clip_id(sample_candidates[1][0]): [0.4, 0.4, 0.2],  # video2 - allowed
        }

        result = matcher.match_embedding_diversity(
            sample_vo_segment,
            sample_candidates,
            existing_segs,
            existing_embs,
            candidate_embs,
            vo_emb
        )

        # Should skip video1 candidates
        if result:
            assert result.video_segment.source_file != "/video1.mp4"


class TestBRollOnlyStrategy:
    """Test broll_only matching strategy"""

    def test_broll_only_finds_silent_footage(self, mock_config, sample_vo_segment):
        """Test broll_only picks segments with is_broll=True"""
        matcher = StrategyMatcher(mock_config, None)

        # Create candidates with is_broll flag
        broll_seg = SRTSegment(0, 0.0, 5.0, "[Silent video]", "/video1.mp4")
        broll_seg.is_broll = True

        speech_seg = SRTSegment(1, 10.0, 15.0, "person talking", "/video2.mp4")
        speech_seg.is_broll = False

        candidates = [
            (speech_seg, 0.8),
            (broll_seg, 0.7)
        ]

        vo_emb = [1.0, 0.0, 0.0]
        candidate_embs = {
            matcher.get_clip_id(broll_seg): [0.8, 0.2, 0.0],
            matcher.get_clip_id(speech_seg): [0.9, 0.1, 0.0]
        }

        result = matcher.match_broll_only(
            sample_vo_segment,
            candidates,
            [],
            [],
            candidate_embs,
            vo_emb
        )

        assert result is not None
        assert result.strategy == "broll_only"
        assert result.video_segment == broll_seg
        assert "b-roll" in result.reasoning.lower()

    def test_broll_only_returns_none_without_broll(self, mock_config, sample_vo_segment, sample_candidates):
        """Test broll_only returns None when no B-roll available"""
        matcher = StrategyMatcher(mock_config, None)

        # Mark all as non-broll
        for seg, _ in sample_candidates:
            seg.is_broll = False

        result = matcher.match_broll_only(
            sample_vo_segment,
            sample_candidates,
            [],
            [],
            {},
            []
        )

        assert result is None

    def test_broll_only_enforces_variety(self, mock_config, sample_vo_segment):
        """Test broll_only enforces different source"""
        matcher = StrategyMatcher(mock_config, None)

        broll1 = SRTSegment(0, 0.0, 5.0, "[Silent]", "/video1.mp4")
        broll1.is_broll = True

        broll2 = SRTSegment(1, 10.0, 15.0, "[Silent]", "/video2.mp4")
        broll2.is_broll = True

        candidates = [(broll1, 0.8), (broll2, 0.7)]

        # Mark video1 as used
        existing_segs = [SRTSegment(0, 20.0, 25.0, "used", "/video1.mp4")]

        vo_emb = [1.0, 0.0, 0.0]
        candidate_embs = {
            matcher.get_clip_id(broll1): [0.8, 0.2, 0.0],
            matcher.get_clip_id(broll2): [0.7, 0.3, 0.0]
        }

        result = matcher.match_broll_only(
            sample_vo_segment,
            candidates,
            existing_segs,
            [],
            candidate_embs,
            vo_emb
        )

        # Should skip video1 and pick video2
        if result:
            assert result.video_segment.source_file == "/video2.mp4"


class TestSourceRotationStrategy:
    """Test source_rotation matching strategy"""

    def test_source_rotation_cycles_sources(self, mock_config, sample_vo_segment, sample_candidates):
        """Test source rotation picks assigned source for segment"""
        matcher = StrategyMatcher(mock_config, None)

        # For segment_index=0, should pick from first source (alphabetically)
        result = matcher.match_source_rotation(
            sample_vo_segment,
            sample_candidates,
            [],
            None,
            {},
            segment_index=0
        )

        assert result is not None
        assert result.strategy == "source_rotation"
        assert "rotation" in result.reasoning.lower()

    def test_source_rotation_fallback_to_next_source(self, mock_config, sample_vo_segment):
        """Test source rotation falls back if assigned source unavailable"""
        matcher = StrategyMatcher(mock_config, None)

        # Only candidates from video2
        candidates = [
            (SRTSegment(0, 0.0, 5.0, "test", "/video2.mp4"), 0.8),
            (SRTSegment(1, 10.0, 15.0, "test2", "/video2.mp4"), 0.7),
        ]

        # segment_index=0 would normally assign video1, but it's not available
        result = matcher.match_source_rotation(
            sample_vo_segment,
            candidates,
            [],
            None,
            {},
            segment_index=0
        )

        # Should fallback to video2
        assert result is not None
        assert result.video_segment.source_file == "/video2.mp4"

    def test_source_rotation_respects_variety_rules(self, mock_config, sample_vo_segment, sample_candidates):
        """Test source rotation still enforces variety within source"""
        matcher = StrategyMatcher(mock_config, None)

        # Mark a clip as already used
        used_seg = sample_candidates[0][0]

        result = matcher.match_source_rotation(
            sample_vo_segment,
            sample_candidates,
            [used_seg],
            None,
            {},
            segment_index=0
        )

        # Should skip the exact same clip
        if result:
            assert matcher.get_clip_id(result.video_segment) != matcher.get_clip_id(used_seg)


class TestSecondaryMatchesDiversity:
    """Test V4-V6 secondary matches with diversity scoring"""

    def test_secondary_matches_enforces_different_sources(self, mock_config, sample_vo_segment, sample_candidates):
        """Test secondary matches require different source per track"""
        matcher = StrategyMatcher(mock_config, None)

        primary = sample_candidates[0][0]  # video1
        alternatives = [sample_candidates[2][0]]  # video1

        vo_emb = [1.0, 0.0, 0.0]
        candidate_embs = {
            matcher.get_clip_id(seg): [0.8 - i*0.1, 0.2 + i*0.1, 0.0]
            for i, (seg, _) in enumerate(sample_candidates)
        }

        result = matcher.get_secondary_matches_diversity(
            sample_vo_segment,
            sample_candidates,
            primary,
            alternatives,
            vo_emb,
            candidate_embs
        )

        # Should return up to 3 secondary matches
        assert len(result) <= 3

        # Each should be from different source than V1-V3
        used_sources = {primary.source_file}
        for alt in alternatives:
            if alt:
                used_sources.add(alt.source_file)

        for match in result:
            assert match.video_segment.source_file not in used_sources
            used_sources.add(match.video_segment.source_file)

    def test_secondary_matches_returns_empty_without_embeddings(self, mock_config, sample_vo_segment, sample_candidates):
        """Test secondary matches requires embeddings"""
        matcher = StrategyMatcher(mock_config, None)

        primary = sample_candidates[0][0]

        result = matcher.get_secondary_matches_diversity(
            sample_vo_segment,
            sample_candidates,
            primary,
            [],
            [],  # No vo embedding
            {}   # No candidate embeddings
        )

        assert result == []

    def test_secondary_matches_respects_global_used_clips(self, mock_config, sample_vo_segment, sample_candidates):
        """Test secondary matches skips globally used clips"""
        matcher = StrategyMatcher(mock_config, None)

        primary = sample_candidates[0][0]

        vo_emb = [1.0, 0.0, 0.0]
        candidate_embs = {
            matcher.get_clip_id(seg): [0.7, 0.3, 0.0]
            for seg, _ in sample_candidates
        }

        # Mark one candidate as globally used
        global_used = {matcher.get_clip_id(sample_candidates[1][0])}

        result = matcher.get_secondary_matches_diversity(
            sample_vo_segment,
            sample_candidates,
            primary,
            [],
            vo_emb,
            candidate_embs,
            global_used_clips=global_used
        )

        # Should not include the globally used clip
        for match in result:
            assert matcher.get_clip_id(match.video_segment) not in global_used


class TestGetStrategyMatches:
    """Test get_strategy_matches orchestration"""

    def test_get_strategy_matches_returns_all_enabled(self, mock_config, sample_vo_segment, sample_candidates):
        """Test get_strategy_matches runs all enabled strategies"""
        mock_config.output.include_strategy_tracks = True
        mock_config.output.strategy_tracks = ["visual_first", "different_source"]

        matcher = StrategyMatcher(mock_config, None)

        primary = sample_candidates[0][0]
        vo_emb = [1.0, 0.0, 0.0]
        candidate_embs = {
            matcher.get_clip_id(seg): [0.8, 0.2, 0.0]
            for seg, _ in sample_candidates
        }

        result = matcher.get_strategy_matches(
            sample_vo_segment,
            sample_candidates,
            primary,
            [],
            vo_emb,
            candidate_embs,
            segment_index=0
        )

        # Should return matches (up to 2 strategies enabled)
        assert len(result) <= 2

        # Each should be from a different strategy
        strategies = [m.strategy for m in result]
        assert len(strategies) == len(set(strategies))  # All unique

    def test_get_strategy_matches_returns_empty_when_disabled(self, mock_config, sample_vo_segment, sample_candidates):
        """Test get_strategy_matches returns empty when disabled"""
        mock_config.output.include_strategy_tracks = False

        matcher = StrategyMatcher(mock_config, None)

        primary = sample_candidates[0][0]

        result = matcher.get_strategy_matches(
            sample_vo_segment,
            sample_candidates,
            primary,
            [],
            [],
            {},
            segment_index=0
        )

        assert result == []

    def test_get_strategy_matches_filters_global_used_clips(self, mock_config, sample_vo_segment, sample_candidates):
        """Test get_strategy_matches pre-filters by global used clips"""
        mock_config.output.strategy_tracks = ["visual_first"]

        matcher = StrategyMatcher(mock_config, None)

        primary = sample_candidates[0][0]

        # Mark all candidates as globally used
        global_used = {
            matcher.get_clip_id(seg) for seg, _ in sample_candidates
        }

        result = matcher.get_strategy_matches(
            sample_vo_segment,
            sample_candidates,
            primary,
            [],
            [],
            {},
            segment_index=0,
            global_used_clips=global_used
        )

        # Should return empty since all candidates filtered out
        assert result == []


class TestSceneLookup:
    """Test _get_scene_for_segment helper"""

    def test_get_scene_for_segment_finds_correct_scene(self, mock_config, mock_scenes):
        """Test scene lookup by time range"""
        matcher = StrategyMatcher(mock_config, mock_scenes)

        seg = SRTSegment(0, 5.0, 8.0, "test", "/video1.mp4")

        scene = matcher._get_scene_for_segment(seg)

        assert scene is not None
        assert scene.start_time == 0.0
        assert scene.end_time == 10.0
        assert "earthquake" in scene.description

    def test_get_scene_for_segment_returns_none_when_not_found(self, mock_config, mock_scenes):
        """Test scene lookup returns None when no match"""
        matcher = StrategyMatcher(mock_config, mock_scenes)

        # Segment outside any scene range
        seg = SRTSegment(0, 100.0, 105.0, "test", "/video1.mp4")

        scene = matcher._get_scene_for_segment(seg)

        assert scene is None

    def test_get_scene_for_segment_returns_none_for_unknown_video(self, mock_config, mock_scenes):
        """Test scene lookup returns None for unknown video"""
        matcher = StrategyMatcher(mock_config, mock_scenes)

        seg = SRTSegment(0, 5.0, 8.0, "test", "/video_unknown.mp4")

        scene = matcher._get_scene_for_segment(seg)

        assert scene is None
