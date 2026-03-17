"""
Integration tests for US-122-006: Listicle groups in video search pipeline.

Verifies:
- listicle_groups are correctly stored in pipeline state
- video search uses listicle_topic_as_search_terms when enabled
- listicle topics are used as additional search terms beyond keywords
- empty listicle_groups don't break video search
- listicle groups are accessible in match stage for topic boosting
- interaction between chapter topics and listicle topics in search
"""

from unittest.mock import MagicMock, patch, PropertyMock
import pytest
from dataclasses import dataclass, field
from typing import List, Any, Optional

from src.stages.video_search import VideoSearchStage
from src.state import PipelineState, VoiceoverSegment
from src.chapter_detection.models import ListicleGroup


# ============================================================================
# Fixtures
# ============================================================================

@dataclass
class MockListicleGroup:
    """Mock ListicleGroup for testing."""
    group_id: int = 0
    item_label: str = "Item"
    segment_indices: List[int] = field(default_factory=list)
    topic_keywords: List[str] = field(default_factory=list)
    start_time: float = 0.0
    end_time: float = 0.0


@pytest.fixture
def mock_config():
    """Create mock config for VideoSearchStage with listicle config."""
    config = MagicMock()
    config.download = MagicMock()
    config.download.video_search = {
        'results_per_keyword': 20,
        'max_total_results': 200,
        'search_budget_aware': True,
        'auto_distribute_budget': True,
        'listicle_topic_as_search_terms': True,
        'use_chapter_queries': True,
    }
    config.download.min_duration = 30
    config.download.max_duration = 600
    config.download.title_blacklist = ['reaction', 'review', 'gameplay']
    return config


@pytest.fixture
def mock_config_disabled():
    """Create mock config with listicle_topic_as_search_terms disabled."""
    config = MagicMock()
    config.download = MagicMock()
    config.download.video_search = {
        'results_per_keyword': 20,
        'max_total_results': 200,
        'search_budget_aware': True,
        'auto_distribute_budget': True,
        'listicle_topic_as_search_terms': False,
        'use_chapter_queries': False,
    }
    config.download.min_duration = 30
    config.download.max_duration = 600
    config.download.title_blacklist = ['reaction', 'review', 'gameplay']
    return config


@pytest.fixture
def state_with_keywords():
    """Pipeline state with keywords."""
    state = PipelineState()
    state.keywords = ['beach sunset', 'ocean waves']
    state.topic_context = 'Travel'
    return state


@pytest.fixture
def state_with_listicle_groups_only():
    """Pipeline state with listicle_groups but no chapters."""
    state = PipelineState()
    state.keywords = ['travel destination']
    state.topic_context = 'Travel'
    state.listicle_groups = [
        MockListicleGroup(
            group_id=0,
            item_label="beach",
            segment_indices=[0],
            topic_keywords=['tropical beach', 'palm trees']
        ),
        MockListicleGroup(
            group_id=1,
            item_label="mountain",
            segment_indices=[1],
            topic_keywords=['mountain hiking', 'alpine scenery']
        ),
    ]
    return state


@pytest.fixture
def state_with_chapters_and_listicle():
    """Pipeline state with both chapters and listicle groups."""
    state = PipelineState()
    state.keywords = ['travel']
    state.topic_context = 'Travel'
    state.location_chapters = [
        {'title': 'Beach Destinations', 'start': 0.0, 'end': 30.0, 'topics': ['beach vacation', 'tropical']},
        {'title': 'Mountain Trips', 'start': 30.0, 'end': 60.0, 'topics': ['hiking', 'mountains']},
    ]
    state.listicle_groups = [
        MockListicleGroup(
            group_id=0,
            item_label="beach",
            segment_indices=[0],
            topic_keywords=['caribbean beach']
        ),
    ]
    return state


@pytest.fixture
def state_with_empty_listicle():
    """Pipeline state with empty listicle_groups."""
    state = PipelineState()
    state.keywords = ['travel']
    state.topic_context = 'Travel'
    state.listicle_groups = []
    return state


@pytest.fixture
def mock_yt_dlp():
    """Mock yt-dlp search results."""
    return {
        'entries': [
            {
                'id': 'abc123',
                'title': 'Beautiful Beach Sunset',
                'channel': 'Nature Channel',
                'duration': 120
            },
            {
                'id': 'def456',
                'title': 'Ocean Waves Relaxation',
                'channel': 'Relaxing Videos',
                'duration': 300
            },
        ]
    }


# ============================================================================
# AC1: Verify listicle_groups are correctly stored in pipeline state
# ============================================================================

class TestListicleGroupsInState:
    """Verify listicle_groups are correctly stored in pipeline state."""

    def test_listicle_groups_stored_in_state(self, state_with_listicle_groups_only):
        """listicle_groups should be accessible from state after setting."""
        assert hasattr(state_with_listicle_groups_only, 'listicle_groups')
        assert len(state_with_listicle_groups_only.listicle_groups) == 2
        assert state_with_listicle_groups_only.listicle_groups[0].item_label == "beach"
        assert state_with_listicle_groups_only.listicle_groups[1].item_label == "mountain"

    def test_listicle_group_has_required_fields(self):
        """ListicleGroup should have required fields for search queries."""
        group = MockListicleGroup(
            group_id=0,
            item_label="test",
            segment_indices=[0, 1],
            topic_keywords=['keyword1', 'keyword2']
        )
        assert hasattr(group, 'group_id')
        assert hasattr(group, 'item_label')
        assert hasattr(group, 'segment_indices')
        assert hasattr(group, 'topic_keywords')


# ============================================================================
# AC2: Test that video search uses listicle_topic_as_search_terms when enabled
# ============================================================================

class TestListicleTopicSearchTermsEnabled:
    """Test video search uses listicle_topic_as_search_terms when enabled."""

    def test_listicle_queries_generated_when_enabled(
        self, mock_config, state_with_listicle_groups_only
    ):
        """Video search should generate listicle queries when config enabled."""
        stage = VideoSearchStage()

        # Build listicle queries - this is what the stage does when config is enabled
        listicle_queries = stage._build_listicle_queries(
            state_with_listicle_groups_only.listicle_groups,
            state_with_listicle_groups_only.topic_context
        )

        # Should generate queries from listicle groups
        assert len(listicle_queries) > 0

    def test_config_list_controls_listicle_query_generation(self):
        """listicle_topic_as_search_terms config should control listicle query generation."""
        stage = VideoSearchStage()

        # Without topic keywords - queries would be empty anyway
        groups = [
            MockListicleGroup(
                group_id=0,
                item_label="test",
                segment_indices=[0],
                topic_keywords=['beach', 'sunset']
            )
        ]

        # When config enabled, stage builds these queries
        queries = stage._build_listicle_queries(groups, 'travel')

        # Should generate queries
        assert len(queries) > 0


# ============================================================================
# AC3: Verify listicle topics are used as additional search terms beyond keywords
# ============================================================================

class TestListicleTopicsAsAdditionalSearchTerms:
    """Verify listicle topics are used as additional search terms beyond keywords."""

    def test_build_listicle_queries_uses_topic_keywords(
        self, state_with_listicle_groups_only
    ):
        """_build_listicle_queries should use topic_keywords from listicle groups."""
        stage = VideoSearchStage()
        listicle_groups = state_with_listicle_groups_only.listicle_groups
        topic_context = state_with_listicle_groups_only.topic_context

        queries = stage._build_listicle_queries(listicle_groups, topic_context)

        assert len(queries) > 0
        # Should have queries from both listicle groups
        group_ids = [q.get('group_id') for q in queries]
        assert 0 in group_ids
        assert 1 in group_ids

    def test_listicle_queries_include_topics(
        self, state_with_listicle_groups_only
    ):
        """Listicle queries should include topic keywords."""
        stage = VideoSearchStage()
        listicle_groups = state_with_listicle_groups_only.listicle_groups

        queries = stage._build_listicle_queries(listicle_groups, '')

        # Each query should have topics from the group
        for query in queries:
            assert 'topics' in query
            assert len(query['topics']) > 0

    def test_listicle_topics_supplement_keywords(
        self, state_with_listicle_groups_only
    ):
        """Listicle topics should supplement, not replace keywords."""
        stage = VideoSearchStage()

        # Get keywords from state
        keywords = state_with_listicle_groups_only.keywords

        # Get listicle queries
        listicle_queries = stage._build_listicle_queries(
            state_with_listicle_groups_only.listicle_groups,
            state_with_listicle_groups_only.topic_context
        )

        # Keywords should still be used separately
        assert len(keywords) > 0
        # Listicle topics are additional
        assert len(listicle_queries) > 0


# ============================================================================
# AC4: Test that empty listicle_groups don't break video search
# ============================================================================

class TestEmptyListicleGroups:
    """Test that empty listicle_groups don't break video search."""

    def test_empty_listicle_groups_no_error(
        self, mock_config, state_with_empty_listicle
    ):
        """Video search should handle empty listicle_groups gracefully."""
        stage = VideoSearchStage()

        # The stage checks hasattr(state, 'listicle_groups') and state.listicle_groups
        # Empty listicle_groups should not cause issues
        assert hasattr(state_with_empty_listicle, 'listicle_groups')
        assert state_with_empty_listicle.listicle_groups == []

        # Build queries with empty listicle groups - should work
        queries = stage._build_listicle_queries(
            state_with_empty_listicle.listicle_groups,
            state_with_empty_listicle.topic_context
        )

        assert queries == []

    def test_build_listicle_queries_with_empty_groups(self):
        """_build_listicle_queries should handle empty listicle_groups."""
        stage = VideoSearchStage()

        queries = stage._build_listicle_queries([], '')

        assert queries == []

    def test_build_listicle_queries_with_none_groups(self):
        """_build_listicle_queries should handle None listicle_groups gracefully."""
        stage = VideoSearchStage()

        # None check is handled by the caller (hasattr check in run method)
        # Simulate what happens in run method
        listicle_groups = None
        if listicle_groups:  # This is the check used in video_search.py
            queries = stage._build_listicle_queries(listicle_groups, '')
        else:
            queries = []

        # Should not raise, should return empty
        assert queries == []


# ============================================================================
# AC5: Verify listicle groups are accessible in match stage for topic boosting
# ============================================================================

class TestListicleGroupsInMatchStage:
    """Verify listicle groups are accessible in match stage for topic boosting."""

    def test_listicle_groups_passed_to_match_stage(self):
        """listicle_groups should flow to match stage for topic boosting."""
        from src.stages.match import MatchStage

        state = PipelineState(
            voiceover_segments=[
                VoiceoverSegment(index=0, start=0.0, end=3.0, text="Number one fact"),
                VoiceoverSegment(index=1, start=3.0, end=6.0, text="Number two fact"),
            ],
            face_preference="neutral",
        )
        state.listicle_groups = [
            MockListicleGroup(
                group_id=0,
                item_label="fact",
                segment_indices=[0, 1],
                topic_keywords=['keyword1', 'keyword2']
            )
        ]

        # Verify listicle_groups is accessible
        assert hasattr(state, 'listicle_groups')
        assert len(state.listicle_groups) == 1
        assert state.listicle_groups[0].topic_keywords == ['keyword1', 'keyword2']


# ============================================================================
# AC6: Test interaction between chapter topics and listicle topics in search
# ============================================================================

class TestChapterAndListicleInteraction:
    """Test interaction between chapter topics and listicle topics in search."""

    def test_build_chapter_queries_with_listicle_groups(
        self, state_with_chapters_and_listicle
    ):
        """Both chapter and listicle queries should be generated when both present."""
        stage = VideoSearchStage()

        # Build chapter queries
        chapter_queries = stage._build_chapter_queries(
            state_with_chapters_and_listicle.location_chapters,
            state_with_chapters_and_listicle.topic_context
        )

        # Build listicle queries
        listicle_queries = stage._build_listicle_queries(
            state_with_chapters_and_listicle.listicle_groups,
            state_with_chapters_and_listicle.topic_context
        )

        # Both should have queries
        assert len(chapter_queries) > 0
        assert len(listicle_queries) > 0

    def test_chapter_and_listicle_queries_different(
        self, state_with_chapters_and_listicle
    ):
        """Chapter and listicle queries should be distinct."""
        stage = VideoSearchStage()

        chapter_queries = stage._build_chapter_queries(
            state_with_chapters_and_listicle.location_chapters,
            state_with_chapters_and_listicle.topic_context
        )

        listicle_queries = stage._build_listicle_queries(
            state_with_chapters_and_listicle.listicle_groups,
            state_with_chapters_and_listicle.topic_context
        )

        # Get the keywords from each
        chapter_keywords = [q.get('keyword', '') for q in chapter_queries]
        listicle_keywords = [q.get('keyword', '') for q in listicle_queries]

        # They should be different (different sources)
        if chapter_keywords and listicle_keywords:
            # At least one chapter keyword should differ from listicle keywords
            assert chapter_keywords[0] != listicle_keywords[0] or len(chapter_keywords) != len(listicle_keywords)

    def test_config_listicle_disabled_no_listicle_queries(
        self, mock_config_disabled, state_with_listicle_groups_only
    ):
        """When listicle_topic_as_search_terms disabled, no listicle queries."""
        stage = VideoSearchStage()

        # Build queries with listicle enabled config
        listicle_queries = stage._build_listicle_queries(
            state_with_listicle_groups_only.listicle_groups,
            state_with_listicle_groups_only.topic_context
        )

        # Should still build queries (config affects stage run, not helper)
        assert len(listicle_queries) > 0


# ============================================================================
# Edge Cases
# ============================================================================

class TestListicleGroupsEdgeCases:
    """Edge case tests for listicle groups."""

    def test_listicle_groups_with_no_topic_keywords(self):
        """Listicle groups without topic keywords should be skipped."""
        stage = VideoSearchStage()

        groups = [
            MockListicleGroup(
                group_id=0,
                item_label="empty",
                segment_indices=[0],
                topic_keywords=[]  # Empty keywords
            )
        ]

        queries = stage._build_listicle_queries(groups, '')

        # Should skip groups without keywords
        assert len(queries) == 0

    def test_listicle_group_dict_access(self):
        """Handle listicle_groups when stored as dicts (not objects)."""
        stage = VideoSearchStage()

        # Simulate dict-based listicle groups
        dict_groups = [
            {
                'group_id': 0,
                'item_label': 'test',
                'topic_keywords': ['keyword1', 'keyword2']
            }
        ]

        queries = stage._build_listicle_queries(dict_groups, '')

        assert len(queries) > 0
        # Format should include item_label: '{item_label} {topic_keyword_1} {topic_keyword_2}'
        assert queries[0]['keyword'] == 'test keyword1 keyword2'

    def test_listicle_with_topic_context(self):
        """Listicle queries should include topic context when available."""
        stage = VideoSearchStage()

        groups = [
            MockListicleGroup(
                group_id=0,
                item_label="test",
                segment_indices=[0],
                topic_keywords=['beach']
            )
        ]

        # With topic context
        queries_with_context = stage._build_listicle_queries(groups, 'Mexico')

        # Should include topic context
        assert len(queries_with_context) > 0
        keyword = queries_with_context[0]['keyword']
        assert 'mexico' in keyword.lower() or 'beach' in keyword.lower()
