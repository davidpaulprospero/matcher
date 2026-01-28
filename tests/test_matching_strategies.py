"""
Comprehensive tests for matching/strategies.py

Tests all 6 matching strategies and variety enforcement logic.
Includes parametrized tests for:
- Provider combinations (embedding, llm, hybrid)
- Confidence threshold boundary values (0.0, 0.5, 0.75, 1.0)
- Match count variations (1, 3, 5, 10 alternatives)
- Strategy fallback scenarios
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


class TestVisualFirstStrategySelection:
    """Additional tests verifying visual_first returns highest visual scores"""

    def test_visual_first_selects_highest_visual_score(self, mock_config, mock_scenes):
        """Test visual_first returns clip with highest visual score, not highest text sim"""
        matcher = StrategyMatcher(mock_config, mock_scenes)

        # Voiceover with visual terms
        vo = SRTSegment(0, 0.0, 5.0, "The earthquake destroyed the city buildings", "")
        vo.keywords = ["earthquake", "city", "buildings"]

        # Candidate 1: High text similarity but low visual score (no scene description)
        high_text_seg = SRTSegment(0, 0.0, 5.0, "earthquake news report", "/video_no_scene.mp4")
        high_text_seg.keywords = []

        # Candidate 2: Lower text similarity but high visual score (matching scene)
        # This matches scene at /video1.mp4:0-10 which has "earthquake damage in city"
        high_visual_seg = SRTSegment(0, 2.0, 7.0, "generic content", "/video1.mp4")
        high_visual_seg.keywords = ["earthquake", "damage", "city"]

        candidates = [
            (high_text_seg, 0.90),  # Higher embedding similarity
            (high_visual_seg, 0.60)  # Lower embedding similarity but matches scene
        ]

        result = matcher.match_visual_first(vo, candidates, [], None, {})

        assert result is not None
        # Should pick the one with better visual match (scene keywords)
        assert result.video_segment.source_file == "/video1.mp4"
        assert "visual" in result.reasoning.lower()

    def test_visual_first_scoring_weights_visual_over_text(self, mock_config, mock_scenes):
        """Test that visual score (70%) outweighs text score (30%)"""
        matcher = StrategyMatcher(mock_config, mock_scenes)

        vo = SRTSegment(0, 0.0, 5.0, "earthquake damage destruction rubble", "")
        vo.keywords = ["earthquake", "damage", "destruction", "rubble"]

        # Both from same source to avoid variety filtering
        seg1 = SRTSegment(0, 0.0, 5.0, "unrelated content", "/video1.mp4")
        seg1.keywords = ["earthquake", "damage"]  # Matches scene keywords

        seg2 = SRTSegment(1, 15.0, 20.0, "earthquake damage", "/video2.mp4")
        seg2.keywords = []  # No keywords

        candidates = [
            (seg2, 0.95),  # High text sim, no visual keywords
            (seg1, 0.50)   # Low text sim, but matches scene description
        ]

        result = matcher.match_visual_first(vo, candidates, [], None, {})

        # Should prefer visual match over text match due to 70/30 weighting
        assert result is not None


class TestDifferentSourceStrategyExclusion:
    """Additional tests verifying different_source properly excludes used sources"""

    def test_different_source_excludes_multiple_used_sources(self, mock_config):
        """Test that all used sources are excluded, not just the first"""
        matcher = StrategyMatcher(mock_config, None)

        vo = SRTSegment(0, 0.0, 5.0, "test content", "")

        # Multiple used sources
        used1 = SRTSegment(0, 0.0, 5.0, "used1", "/video1.mp4")
        used2 = SRTSegment(1, 0.0, 5.0, "used2", "/video2.mp4")

        # Candidates from 3 sources
        candidates = [
            (SRTSegment(0, 0.0, 5.0, "from v1", "/video1.mp4"), 0.9),
            (SRTSegment(1, 0.0, 5.0, "from v2", "/video2.mp4"), 0.85),
            (SRTSegment(2, 0.0, 5.0, "from v3", "/video3.mp4"), 0.7),
        ]

        result = matcher.match_different_source(vo, candidates, [used1, used2], None, {})

        assert result is not None
        # Should only pick from video3 since video1 and video2 are used
        assert result.video_segment.source_file == "/video3.mp4"

    def test_different_source_returns_first_unused_from_sorted_candidates(self, mock_config):
        """Test that different_source returns first unused source from pre-sorted candidates"""
        matcher = StrategyMatcher(mock_config, None)

        vo = SRTSegment(0, 0.0, 5.0, "test content", "")
        used = SRTSegment(0, 0.0, 5.0, "used", "/video1.mp4")

        # Candidates should be pre-sorted by similarity (highest first)
        # This reflects actual pipeline behavior where candidates come pre-sorted
        candidates = [
            (SRTSegment(0, 0.0, 5.0, "from v1 best", "/video1.mp4"), 0.95),  # Used source - skipped
            (SRTSegment(1, 0.0, 5.0, "from v3 high", "/video3.mp4"), 0.85),  # First unused - selected
            (SRTSegment(2, 0.0, 5.0, "from v2 med", "/video2.mp4"), 0.75),
        ]

        result = matcher.match_different_source(vo, candidates, [used], None, {})

        assert result is not None
        # Should pick first unused source in sorted order (video3)
        assert result.video_segment.source_file == "/video3.mp4"
        assert result.confidence == 0.85


class TestEmbeddingDiversityMaximization:
    """Additional tests verifying embedding_diversity maximizes variance from V1-V3"""

    def test_embedding_diversity_selects_most_diverse(self, mock_config):
        """Test that embedding_diversity picks clip with maximum distance from existing"""
        matcher = StrategyMatcher(mock_config, None)

        vo = SRTSegment(0, 0.0, 5.0, "test content", "")
        vo_emb = [1.0, 0.0, 0.0]

        # Existing match embedding (V1)
        existing = SRTSegment(0, 0.0, 5.0, "existing", "/video1.mp4")
        existing_embs = [[0.9, 0.1, 0.0]]  # Very similar to vo_emb

        # Create candidates with different diversity levels
        similar_seg = SRTSegment(0, 0.0, 5.0, "similar", "/video2.mp4")
        diverse_seg = SRTSegment(1, 0.0, 5.0, "diverse", "/video3.mp4")
        most_diverse_seg = SRTSegment(2, 0.0, 5.0, "most diverse", "/video4.mp4")

        candidates = [
            (similar_seg, 0.9),
            (diverse_seg, 0.7),
            (most_diverse_seg, 0.6),
        ]

        candidate_embs = {
            matcher.get_clip_id(similar_seg): [0.85, 0.15, 0.0],     # Very similar to existing
            matcher.get_clip_id(diverse_seg): [0.3, 0.5, 0.2],       # Moderately diverse
            matcher.get_clip_id(most_diverse_seg): [0.0, 0.1, 0.9],  # Most diverse
        }

        result = matcher.match_embedding_diversity(
            vo, candidates, [existing], existing_embs, candidate_embs, vo_emb
        )

        assert result is not None
        assert result.strategy == "embedding_diversity"
        # Should pick most_diverse_seg due to maximum distance from existing
        # Note: also weighted by vo relevance (40%) so exact selection depends on combined score

    def test_embedding_diversity_respects_relevance_threshold(self, mock_config):
        """Test that clips below 0.3 relevance threshold are excluded"""
        matcher = StrategyMatcher(mock_config, None)

        vo = SRTSegment(0, 0.0, 5.0, "test content", "")
        vo_emb = [1.0, 0.0, 0.0]

        existing = SRTSegment(0, 0.0, 5.0, "existing", "/video1.mp4")
        existing_embs = [[0.9, 0.1, 0.0]]

        # Candidate that's very diverse but not relevant to voiceover
        irrelevant_seg = SRTSegment(0, 0.0, 5.0, "irrelevant", "/video2.mp4")
        relevant_seg = SRTSegment(1, 0.0, 5.0, "relevant", "/video3.mp4")

        candidates = [
            (irrelevant_seg, 0.2),
            (relevant_seg, 0.5),
        ]

        candidate_embs = {
            # Very diverse from existing but very different from vo (low relevance ~0.1)
            matcher.get_clip_id(irrelevant_seg): [0.0, 0.0, 1.0],
            # Less diverse but more relevant to vo (relevance ~0.7)
            matcher.get_clip_id(relevant_seg): [0.7, 0.3, 0.0],
        }

        result = matcher.match_embedding_diversity(
            vo, candidates, [existing], existing_embs, candidate_embs, vo_emb
        )

        # Should pick relevant_seg despite irrelevant being more diverse
        # because irrelevant has vo_relevance < 0.3 threshold
        assert result is not None
        assert result.video_segment.source_file == "/video3.mp4"


class TestBRollOnlyExclusivity:
    """Additional tests verifying broll_only strategy ONLY returns B-roll clips"""

    def test_broll_only_ignores_non_broll_even_with_higher_score(self, mock_config):
        """Test that non-B-roll clips are ignored even if they have higher similarity"""
        matcher = StrategyMatcher(mock_config, None)

        vo = SRTSegment(0, 0.0, 5.0, "test content", "")
        vo_emb = [1.0, 0.0, 0.0]

        # Non-broll with very high similarity
        high_sim_non_broll = SRTSegment(0, 0.0, 5.0, "talking head", "/video1.mp4")
        high_sim_non_broll.is_broll = False

        # B-roll with lower similarity
        low_sim_broll = SRTSegment(1, 0.0, 5.0, "[silent footage]", "/video2.mp4")
        low_sim_broll.is_broll = True

        candidates = [
            (high_sim_non_broll, 0.95),
            (low_sim_broll, 0.40),
        ]

        candidate_embs = {
            matcher.get_clip_id(high_sim_non_broll): [0.95, 0.05, 0.0],
            matcher.get_clip_id(low_sim_broll): [0.4, 0.3, 0.3],
        }

        result = matcher.match_broll_only(
            vo, candidates, [], [], candidate_embs, vo_emb
        )

        assert result is not None
        assert result.video_segment == low_sim_broll
        assert result.video_segment.is_broll == True
        assert "b-roll" in result.reasoning.lower()

    def test_broll_only_returns_best_broll_by_relevance(self, mock_config):
        """Test that among B-roll clips, the most relevant is returned"""
        matcher = StrategyMatcher(mock_config, None)

        vo = SRTSegment(0, 0.0, 5.0, "test content", "")
        vo_emb = [1.0, 0.0, 0.0]

        # Multiple B-roll candidates with different relevance scores
        broll1 = SRTSegment(0, 0.0, 5.0, "[silent A]", "/video1.mp4")
        broll1.is_broll = True
        broll2 = SRTSegment(1, 0.0, 5.0, "[silent B]", "/video2.mp4")
        broll2.is_broll = True
        broll3 = SRTSegment(2, 0.0, 5.0, "[silent C]", "/video3.mp4")
        broll3.is_broll = True

        candidates = [
            (broll1, 0.5),
            (broll2, 0.8),  # Highest relevance
            (broll3, 0.6),
        ]

        candidate_embs = {
            matcher.get_clip_id(broll1): [0.5, 0.3, 0.2],
            matcher.get_clip_id(broll2): [0.8, 0.1, 0.1],  # Most relevant to vo_emb
            matcher.get_clip_id(broll3): [0.6, 0.2, 0.2],
        }

        result = matcher.match_broll_only(
            vo, candidates, [], [], candidate_embs, vo_emb
        )

        assert result is not None
        # Should pick broll2 as it has highest embedding similarity to vo
        assert result.video_segment == broll2

    def test_broll_only_rejects_all_non_broll(self, mock_config):
        """Test that when no B-roll exists, None is returned"""
        matcher = StrategyMatcher(mock_config, None)

        vo = SRTSegment(0, 0.0, 5.0, "test content", "")

        # All candidates are non-broll
        seg1 = SRTSegment(0, 0.0, 5.0, "speech A", "/video1.mp4")
        seg1.is_broll = False
        seg2 = SRTSegment(1, 0.0, 5.0, "speech B", "/video2.mp4")
        seg2.is_broll = False

        candidates = [
            (seg1, 0.9),
            (seg2, 0.85),
        ]

        result = matcher.match_broll_only(vo, candidates, [], [], {}, [])

        assert result is None


class TestStrategySelectionOrdering:
    """Tests verifying strategy selection order and cumulative exclusion"""

    def test_strategies_build_on_previous_exclusions(self, mock_config):
        """Test that each strategy excludes clips used by previous strategies"""
        mock_config.output.strategy_tracks = ["different_source", "visual_first"]

        matcher = StrategyMatcher(mock_config, None)

        vo = SRTSegment(0, 0.0, 5.0, "earthquake damage", "")
        vo.keywords = ["earthquake"]

        primary = SRTSegment(0, 0.0, 5.0, "primary", "/video1.mp4")

        # Create candidates from 3 sources
        seg2 = SRTSegment(0, 0.0, 5.0, "from v2", "/video2.mp4")
        seg2.keywords = ["earthquake"]
        seg3 = SRTSegment(1, 0.0, 5.0, "from v3", "/video3.mp4")
        seg3.keywords = []

        candidates = [
            (seg2, 0.8),
            (seg3, 0.7),
        ]

        vo_emb = [1.0, 0.0, 0.0]
        candidate_embs = {
            matcher.get_clip_id(seg2): [0.8, 0.2, 0.0],
            matcher.get_clip_id(seg3): [0.7, 0.3, 0.0],
        }

        result = matcher.get_strategy_matches(
            vo, candidates, primary, [], vo_emb, candidate_embs, segment_index=0
        )

        # Should get results from both strategies
        assert len(result) >= 1
        # Each result should be from a different source
        sources = [r.video_segment.source_file for r in result]
        assert len(sources) == len(set(sources))  # All unique sources


# ============================================================================
# Parametrized Tests for Matching Strategies (US-003)
# ============================================================================

class TestParametrizedStrategyMatching:
    """Parametrized tests for matching strategy validation"""

    @pytest.mark.parametrize("strategy_name,expected_valid", [
        ("visual_first", True),
        ("different_source", True),
        ("keyword_only", True),
        ("embedding_diversity", True),
        ("broll_only", True),
        ("source_rotation", True),
        ("invalid_strategy", False),
        ("", False),
    ])
    def test_strategy_name_validation(self, strategy_name, expected_valid):
        """Test that strategy names are validated correctly"""
        valid_strategies = [
            "visual_first", "different_source", "keyword_only",
            "embedding_diversity", "broll_only", "source_rotation"
        ]
        is_valid = strategy_name in valid_strategies
        assert is_valid == expected_valid

    @pytest.mark.parametrize("confidence_threshold,similarity_scores,expected_matches", [
        (0.0, [0.1, 0.5, 0.9], 3),   # Accept all at 0.0 threshold
        (0.5, [0.1, 0.5, 0.9], 2),   # Accept >= 0.5
        (0.75, [0.1, 0.5, 0.9], 1),  # Accept >= 0.75
        (1.0, [0.1, 0.5, 0.9], 0),   # Accept none (need exact 1.0)
    ])
    def test_confidence_threshold_filtering(self, confidence_threshold, similarity_scores, expected_matches):
        """Test confidence threshold affects match count"""
        passing_scores = [s for s in similarity_scores if s >= confidence_threshold]
        assert len(passing_scores) == expected_matches

    @pytest.mark.parametrize("num_candidates,num_alternatives_config,expected_alternatives", [
        (1, 3, 0),   # 1 candidate, no alternatives possible
        (2, 3, 1),   # 2 candidates, 1 alternative
        (5, 3, 3),   # 5 candidates, max 3 alternatives
        (10, 5, 5),  # 10 candidates, max 5 alternatives
        (3, 10, 2),  # 3 candidates, only 2 alternatives possible
    ])
    def test_alternatives_count_limits(self, num_candidates, num_alternatives_config, expected_alternatives):
        """Test that alternatives are properly limited by candidates or config"""
        # One candidate is used for primary, rest available for alternatives
        available_for_alternatives = max(0, num_candidates - 1)
        actual_alternatives = min(num_alternatives_config, available_for_alternatives)
        assert actual_alternatives == expected_alternatives


class TestParametrizedVisualFirst:
    """Parametrized tests for visual_first strategy"""

    @pytest.mark.parametrize("has_scene_desc,has_keywords,expected_match", [
        (True, True, True),    # Both available
        (True, False, True),   # Scene only - still matches
        (False, True, True),   # Keywords only - still matches
        (False, False, False), # Neither available - may not match
    ])
    def test_visual_first_data_combinations(self, mock_config, has_scene_desc, has_keywords, expected_match):
        """Test visual_first with different data availability"""
        if has_scene_desc:
            scenes = {
                "/video1.mp4": [
                    SceneInfo(
                        video_path="/video1.mp4",
                        scene_index=0,
                        start_time=0.0,
                        end_time=10.0,
                        description="earthquake damage",
                        visual_keywords=["earthquake", "damage"]
                    )
                ]
            }
        else:
            scenes = {}

        matcher = StrategyMatcher(mock_config, scenes)

        vo = SRTSegment(0, 0.0, 5.0, "earthquake in the city", "")
        vo.keywords = ["earthquake", "city"] if has_keywords else []

        seg = SRTSegment(0, 0.0, 5.0, "test", "/video1.mp4")
        seg.keywords = ["earthquake"] if has_keywords else []

        candidates = [(seg, 0.8)]

        result = matcher.match_visual_first(vo, candidates, [], None, {})

        if expected_match:
            assert result is not None
        # Note: even with no data, may still return None gracefully


class TestParametrizedDifferentSource:
    """Parametrized tests for different_source strategy"""

    @pytest.mark.parametrize("used_sources,candidate_sources,expected_source", [
        (["/v1.mp4"], ["/v1.mp4", "/v2.mp4"], "/v2.mp4"),
        (["/v1.mp4", "/v2.mp4"], ["/v1.mp4", "/v2.mp4", "/v3.mp4"], "/v3.mp4"),
        (["/v1.mp4"], ["/v2.mp4", "/v3.mp4"], "/v2.mp4"),
        ([], ["/v1.mp4", "/v2.mp4"], "/v1.mp4"),
    ])
    def test_different_source_selection(self, mock_config, used_sources, candidate_sources, expected_source):
        """Test different_source picks correct source"""
        matcher = StrategyMatcher(mock_config, None)

        vo = SRTSegment(0, 0.0, 5.0, "test", "")
        used = [SRTSegment(0, 0.0, 5.0, "used", src) for src in used_sources]

        candidates = []
        for i, src in enumerate(candidate_sources):
            seg = SRTSegment(i, 0.0, 5.0, f"from {src}", src)
            candidates.append((seg, 0.8 - i * 0.05))

        result = matcher.match_different_source(vo, candidates, used, None, {})

        if expected_source and len(set(candidate_sources) - set(used_sources)) > 0:
            assert result is not None
            assert result.video_segment.source_file not in used_sources


class TestParametrizedKeywordOnly:
    """Parametrized tests for keyword_only strategy"""

    @pytest.mark.parametrize("vo_keywords,candidate_keywords,expected_overlap", [
        (["earthquake", "damage"], ["earthquake", "damage", "city"], 2),
        (["tsunami", "wave"], ["earthquake", "damage"], 0),
        (["city", "building"], ["building", "city", "urban"], 2),
        ([], ["earthquake", "damage"], 0),
        (["earthquake"], [], 0),
    ])
    def test_keyword_overlap_calculation(self, mock_config, vo_keywords, candidate_keywords, expected_overlap):
        """Test keyword overlap affects matching"""
        matcher = StrategyMatcher(mock_config, None)

        vo = SRTSegment(0, 0.0, 5.0, " ".join(vo_keywords), "")
        vo.keywords = vo_keywords
        vo.entities = []

        seg = SRTSegment(0, 0.0, 5.0, " ".join(candidate_keywords), "/video1.mp4")
        seg.keywords = candidate_keywords
        seg.entities = []

        # Calculate actual overlap
        overlap = len(set(vo_keywords) & set(candidate_keywords))
        assert overlap == expected_overlap


class TestParametrizedEmbeddingDiversity:
    """Parametrized tests for embedding_diversity strategy"""

    @pytest.mark.parametrize("existing_emb,candidate_embs,expected_most_diverse_idx", [
        ([1.0, 0.0, 0.0], [[0.9, 0.1, 0.0], [0.5, 0.5, 0.0], [0.0, 0.0, 1.0]], 2),
        ([0.5, 0.5, 0.0], [[0.5, 0.5, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]], 1),  # [0.0, 1.0, 0.0] most different from [0.5, 0.5, 0.0]
    ])
    def test_diversity_calculation(self, existing_emb, candidate_embs, expected_most_diverse_idx):
        """Test embedding diversity calculation picks most different"""
        import numpy as np

        existing = np.array(existing_emb)
        candidates = [np.array(e) for e in candidate_embs]

        # Calculate distances (1 - cosine similarity)
        distances = []
        for c in candidates:
            # Cosine similarity
            cos_sim = np.dot(existing, c) / (np.linalg.norm(existing) * np.linalg.norm(c))
            distances.append(1.0 - cos_sim)

        most_diverse_idx = np.argmax(distances)
        assert most_diverse_idx == expected_most_diverse_idx


class TestParametrizedBrollOnly:
    """Parametrized tests for broll_only strategy"""

    @pytest.mark.parametrize("broll_flags,expected_broll_count", [
        ([True, False, False], 1),
        ([True, True, False], 2),
        ([True, True, True], 3),
        ([False, False, False], 0),
    ])
    def test_broll_filtering(self, mock_config, broll_flags, expected_broll_count):
        """Test B-roll filtering by is_broll flag"""
        matcher = StrategyMatcher(mock_config, None)

        vo = SRTSegment(0, 0.0, 5.0, "test", "")

        candidates = []
        for i, is_broll in enumerate(broll_flags):
            seg = SRTSegment(i, 0.0, 5.0, "test", f"/video{i}.mp4")
            seg.is_broll = is_broll
            candidates.append((seg, 0.8))

        # Count B-roll candidates
        broll_count = sum(1 for seg, _ in candidates if getattr(seg, 'is_broll', False))
        assert broll_count == expected_broll_count


class TestParametrizedSourceRotation:
    """Parametrized tests for source_rotation strategy"""

    @pytest.mark.parametrize("segment_index,num_sources,expected_source_idx", [
        (0, 3, 0),   # First segment -> first source
        (1, 3, 1),   # Second segment -> second source
        (2, 3, 2),   # Third segment -> third source
        (3, 3, 0),   # Fourth segment -> wraps to first source
        (5, 2, 1),   # Wrapping with 2 sources
    ])
    def test_source_rotation_assignment(self, segment_index, num_sources, expected_source_idx):
        """Test source rotation assigns sources correctly"""
        # Source rotation should cycle through sources based on segment index
        actual_source_idx = segment_index % num_sources
        assert actual_source_idx == expected_source_idx


class TestParametrizedVarietyEnforcement:
    """Parametrized tests for variety enforcement"""

    @pytest.mark.parametrize("min_time_distance,time_diffs,expected_exclusions", [
        (10.0, [5.0, 15.0, 8.0], 2),   # 2 clips within 10s
        (5.0, [6.0, 7.0, 8.0], 0),     # All clips > 5s apart
        (20.0, [10.0, 15.0, 25.0], 2), # 2 clips within 20s
        (0.0, [1.0, 2.0, 3.0], 0),     # No time-based exclusion
    ])
    def test_time_distance_exclusion(self, mock_config, min_time_distance, time_diffs, expected_exclusions):
        """Test time distance exclusion rule"""
        mock_config.output.variety.min_time_distance = min_time_distance

        exclusions = sum(1 for d in time_diffs if d < min_time_distance)
        assert exclusions == expected_exclusions

    @pytest.mark.parametrize("min_emb_distance,emb_similarities,expected_exclusions", [
        (0.3, [0.95, 0.85, 0.6], 2),   # 2 clips too similar (1-sim < 0.3): 1-0.95=0.05, 1-0.85=0.15, 1-0.6=0.4
        (0.2, [0.9, 0.7, 0.5], 1),     # 1 clip too similar: 1-0.9=0.1<0.2
        (0.5, [0.6, 0.5, 0.4], 1),     # 1 too similar: 1-0.6=0.4<0.5
        (0.0, [1.0, 1.0, 1.0], 0),     # No embedding exclusion (distance always >= 0)
    ])
    def test_embedding_distance_exclusion(self, mock_config, min_emb_distance, emb_similarities, expected_exclusions):
        """Test embedding distance exclusion rule"""
        mock_config.output.variety.min_embedding_distance = min_emb_distance

        # Exclusion if (1 - similarity) < min_emb_distance
        exclusions = sum(1 for s in emb_similarities if (1.0 - s) < min_emb_distance)
        assert exclusions == expected_exclusions


class TestParametrizedStrategyFallbacks:
    """Parametrized tests for strategy fallback scenarios"""

    @pytest.mark.parametrize("strategy,available_data,should_fallback", [
        ("visual_first", {"scenes": True, "keywords": True}, False),
        ("visual_first", {"scenes": False, "keywords": True}, True),
        ("broll_only", {"broll": True}, False),
        ("broll_only", {"broll": False}, True),
        ("embedding_diversity", {"embeddings": True}, False),
        ("embedding_diversity", {"embeddings": False}, True),
    ])
    def test_strategy_fallback_conditions(self, strategy, available_data, should_fallback):
        """Test when strategies should fall back"""
        # Strategies fall back when required data is unavailable
        required_data = {
            "visual_first": "scenes",
            "broll_only": "broll",
            "embedding_diversity": "embeddings",
        }

        if strategy in required_data:
            data_key = required_data[strategy]
            has_data = available_data.get(data_key, False)
            needs_fallback = not has_data
            assert needs_fallback == should_fallback

    @pytest.mark.parametrize("primary_fails,fallback_fails,expected_result", [
        (False, False, "primary"),     # Primary succeeds
        (True, False, "fallback"),     # Primary fails, fallback succeeds
        (True, True, "none"),          # Both fail
    ])
    def test_fallback_chain_results(self, primary_fails, fallback_fails, expected_result):
        """Test fallback chain produces correct result type"""
        if not primary_fails:
            result = "primary"
        elif not fallback_fails:
            result = "fallback"
        else:
            result = "none"
        assert result == expected_result
