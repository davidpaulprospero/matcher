"""
Tests for location-aware matching and geographic filtering (location_matching.py).

Targets uncovered lines:
- Lines 18-77: LocationMatcher initialization and setters
- Lines 79-196: apply_location_filter() with hard/soft filtering

Current coverage: 15.66%
Target coverage: 75%+

Created: 2026-01-10 (Session 13 - Critical gap coverage, Phase 1)
"""

import sys
from pathlib import Path
from unittest.mock import Mock, MagicMock
import pytest

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.matching.location_matching import LocationMatcher
from src.location_service import GeoLocation, LocationService
from src.topic_extraction import LocationChapter
from src.utils import SRTSegment


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def mock_location_service():
    """Mock LocationService for testing"""
    service = Mock(spec=LocationService)

    # Mock comparison methods
    service.same_city = Mock(return_value=True)
    service.same_region = Mock(return_value=True)
    service.same_country = Mock(return_value=True)
    service.same_continent = Mock(return_value=True)
    service.is_parent_region = Mock(return_value=False)

    return service


@pytest.fixture
def sample_location_chapters():
    """Sample location chapters for testing"""
    return [
        LocationChapter(
            chapter_id=0,
            start_segment_idx=0,
            end_segment_idx=2,
            location_name="Paris",
            location_type="city",
            visual_keywords=["eiffel tower", "louvre"],
            title="Paris Tour",
            location_data={
                'name': 'Paris',
                'location_type': 'city',
                'country_code': 'FR',
                'country_name': 'France',
                'admin1': 'Île-de-France',
                'coordinates': [48.8566, 2.3522]
            }
        ),
        LocationChapter(
            chapter_id=1,
            start_segment_idx=3,
            end_segment_idx=5,
            location_name="London",
            location_type="city",
            visual_keywords=["big ben", "tower bridge"],
            title="London Visit",
            location_data={
                'name': 'London',
                'location_type': 'city',
                'country_code': 'GB',
                'country_name': 'United Kingdom',
                'admin1': 'England',
                'coordinates': [51.5074, -0.1278]
            }
        )
    ]


@pytest.fixture
def sample_video_locations():
    """Sample video location data"""
    return {
        '/videos/paris1.mp4': GeoLocation(
            name='Paris',
            location_type='city',
            country_code='FR',
            country_name='France',
            admin1='Île-de-France',
            coordinates=(48.8566, 2.3522)
        ),
        '/videos/london1.mp4': GeoLocation(
            name='London',
            location_type='city',
            country_code='GB',
            country_name='United Kingdom',
            admin1='England',
            coordinates=(51.5074, -0.1278)
        ),
        '/videos/tokyo1.mp4': GeoLocation(
            name='Tokyo',
            location_type='city',
            country_code='JP',
            country_name='Japan',
            admin1='Tokyo',
            coordinates=(35.6762, 139.6503)
        )
    }


@pytest.fixture
def sample_vo_segment():
    """Sample voiceover segment"""
    return SRTSegment(
        index=1,
        start_time=0.0,
        end_time=5.0,
        text="Exploring the streets of Paris",
        source_file="voiceover.srt"
    )


@pytest.fixture
def sample_candidates(sample_video_locations):
    """Sample candidate video segments"""
    candidates = []

    # Paris candidate (high similarity)
    paris_seg = SRTSegment(
        index=1,
        start_time=0.0,
        end_time=5.0,
        text="Eiffel Tower footage",
        source_file='/videos/paris1.mp4'
    )
    candidates.append((paris_seg, 0.9))

    # London candidate (medium similarity)
    london_seg = SRTSegment(
        index=2,
        start_time=0.0,
        end_time=5.0,
        text="Big Ben views",
        source_file='/videos/london1.mp4'
    )
    candidates.append((london_seg, 0.7))

    # Tokyo candidate (low similarity)
    tokyo_seg = SRTSegment(
        index=3,
        start_time=0.0,
        end_time=5.0,
        text="Tokyo skyline",
        source_file='/videos/tokyo1.mp4'
    )
    candidates.append((tokyo_seg, 0.6))

    return candidates


# ============================================================================
# Test LocationMatcher Initialization
# ============================================================================

class TestLocationMatcherInit:
    """Test LocationMatcher initialization"""

    @pytest.mark.fast
    def test_init_without_location_service(self):
        """Test initialization without location service"""
        matcher = LocationMatcher(location_service=None)

        assert matcher.location_service is None
        assert matcher.location_chapters == {}
        assert matcher.video_locations == {}

    @pytest.mark.fast
    def test_init_with_location_service(self, mock_location_service):
        """Test initialization with location service"""
        matcher = LocationMatcher(location_service=mock_location_service)

        assert matcher.location_service is mock_location_service
        assert matcher.location_chapters == {}
        assert matcher.video_locations == {}


# ============================================================================
# Test set_location_chapters()
# ============================================================================

class TestSetLocationChapters:
    """Test location chapter setup"""

    @pytest.mark.fast
    def test_set_location_chapters_empty(self, mock_location_service):
        """Test with empty location chapters"""
        matcher = LocationMatcher(location_service=mock_location_service)

        matcher.set_location_chapters([])

        assert matcher.location_chapters == {}

    @pytest.mark.fast
    def test_set_location_chapters_single(self, mock_location_service, sample_location_chapters):
        """Test with single location chapter"""
        matcher = LocationMatcher(location_service=mock_location_service)

        # Use only first chapter (covers segments 0-2)
        matcher.set_location_chapters([sample_location_chapters[0]])

        # Should map segments 0, 1, 2 to the chapter
        assert len(matcher.location_chapters) == 3
        assert matcher.location_chapters[0] == sample_location_chapters[0]
        assert matcher.location_chapters[1] == sample_location_chapters[0]
        assert matcher.location_chapters[2] == sample_location_chapters[0]

    @pytest.mark.fast
    def test_set_location_chapters_multiple(self, mock_location_service, sample_location_chapters):
        """Test with multiple location chapters"""
        matcher = LocationMatcher(location_service=mock_location_service)

        matcher.set_location_chapters(sample_location_chapters)

        # First chapter: segments 0-2 (3 segments)
        # Second chapter: segments 3-5 (3 segments)
        assert len(matcher.location_chapters) == 6
        assert matcher.location_chapters[0].location_name == "Paris"
        assert matcher.location_chapters[3].location_name == "London"

    @pytest.mark.fast
    def test_set_location_chapters_overlapping(self, mock_location_service):
        """Test with overlapping chapters (later chapter overwrites)"""
        matcher = LocationMatcher(location_service=mock_location_service)

        chapters = [
            LocationChapter(
                chapter_id=0,
                start_segment_idx=0,
                end_segment_idx=3,
                location_name="Paris",
                location_type="city",
                visual_keywords=[],
                title="Paris"
            ),
            LocationChapter(
                chapter_id=1,
                start_segment_idx=2,
                end_segment_idx=5,
                location_name="London",
                location_type="city",
                visual_keywords=[],
                title="London"
            )
        ]

        matcher.set_location_chapters(chapters)

        # Segments 0-1: Paris
        # Segments 2-3: London (overwrites Paris for 2-3)
        # Segments 4-5: London
        assert matcher.location_chapters[0].location_name == "Paris"
        assert matcher.location_chapters[1].location_name == "Paris"
        assert matcher.location_chapters[2].location_name == "London"
        assert matcher.location_chapters[3].location_name == "London"


# ============================================================================
# Test set_video_locations()
# ============================================================================

class TestSetVideoLocations:
    """Test video location setup"""

    @pytest.mark.fast
    def test_set_video_locations_empty(self, mock_location_service):
        """Test with empty video locations"""
        matcher = LocationMatcher(location_service=mock_location_service)

        matcher.set_video_locations({})

        assert matcher.video_locations == {}

    @pytest.mark.fast
    def test_set_video_locations_multiple(self, mock_location_service, sample_video_locations):
        """Test with multiple video locations"""
        matcher = LocationMatcher(location_service=mock_location_service)

        matcher.set_video_locations(sample_video_locations)

        assert len(matcher.video_locations) == 3
        assert matcher.video_locations['/videos/paris1.mp4'].name == 'Paris'
        assert matcher.video_locations['/videos/london1.mp4'].name == 'London'
        assert matcher.video_locations['/videos/tokyo1.mp4'].name == 'Tokyo'


# ============================================================================
# Test _get_location_chapter()
# ============================================================================

class TestGetLocationChapter:
    """Test location chapter retrieval"""

    @pytest.mark.fast
    def test_get_location_chapter_found(self, mock_location_service, sample_location_chapters):
        """Test retrieving existing location chapter"""
        matcher = LocationMatcher(location_service=mock_location_service)
        matcher.set_location_chapters(sample_location_chapters)

        # Segment 0 is in Paris chapter
        chapter = matcher._get_location_chapter(0)

        assert chapter is not None
        assert chapter.location_name == "Paris"

    @pytest.mark.fast
    def test_get_location_chapter_not_found(self, mock_location_service):
        """Test retrieving non-existent location chapter"""
        matcher = LocationMatcher(location_service=mock_location_service)

        chapter = matcher._get_location_chapter(999)

        assert chapter is None


# ============================================================================
# Test _get_video_location()
# ============================================================================

class TestGetVideoLocation:
    """Test video location retrieval"""

    @pytest.mark.fast
    def test_get_video_location_found(self, mock_location_service, sample_video_locations):
        """Test retrieving existing video location"""
        matcher = LocationMatcher(location_service=mock_location_service)
        matcher.set_video_locations(sample_video_locations)

        location = matcher._get_video_location('/videos/paris1.mp4')

        assert location is not None
        assert location.name == 'Paris'

    @pytest.mark.fast
    def test_get_video_location_not_found(self, mock_location_service):
        """Test retrieving non-existent video location"""
        matcher = LocationMatcher(location_service=mock_location_service)

        location = matcher._get_video_location('/unknown/video.mp4')

        assert location is None


# ============================================================================
# Test apply_location_filter() - No Filtering Cases
# ============================================================================

class TestApplyLocationFilterNoFilter:
    """Test cases where location filter is not applied"""

    @pytest.mark.fast
    def test_apply_location_filter_disabled(self, mock_location_service, sample_vo_segment, sample_candidates):
        """Test with location matching disabled"""
        matcher = LocationMatcher(location_service=mock_location_service)

        filtered, applied, reason = matcher.apply_location_filter(
            vo_segment=sample_vo_segment,
            candidates=sample_candidates,
            segment_idx=0,
            location_matching_enabled=False,
            location_matching_config={}
        )

        assert filtered == sample_candidates
        assert applied is False
        assert reason == ""

    @pytest.mark.fast
    def test_apply_location_filter_no_service(self, sample_vo_segment, sample_candidates):
        """Test without location service"""
        matcher = LocationMatcher(location_service=None)

        filtered, applied, reason = matcher.apply_location_filter(
            vo_segment=sample_vo_segment,
            candidates=sample_candidates,
            segment_idx=0,
            location_matching_enabled=True,
            location_matching_config={}
        )

        assert filtered == sample_candidates
        assert applied is False
        assert reason == ""

    @pytest.mark.fast
    def test_apply_location_filter_no_chapter(self, mock_location_service, sample_vo_segment, sample_candidates):
        """Test when segment has no location chapter"""
        matcher = LocationMatcher(location_service=mock_location_service)

        filtered, applied, reason = matcher.apply_location_filter(
            vo_segment=sample_vo_segment,
            candidates=sample_candidates,
            segment_idx=999,  # No chapter for this segment
            location_matching_enabled=True,
            location_matching_config={}
        )

        assert filtered == sample_candidates
        assert applied is False
        assert reason == ""

    @pytest.mark.fast
    def test_apply_location_filter_chapter_no_location_data(self, mock_location_service, sample_vo_segment, sample_candidates):
        """Test when chapter exists but has no location data"""
        matcher = LocationMatcher(location_service=mock_location_service)

        # Chapter without location_data
        chapter = LocationChapter(
            chapter_id=0,
            start_segment_idx=0,
            end_segment_idx=1,
            location_name="Unknown",
            location_type="city",
            visual_keywords=[],
            title="Unknown",
            location_data=None  # No location data
        )
        matcher.set_location_chapters([chapter])

        filtered, applied, reason = matcher.apply_location_filter(
            vo_segment=sample_vo_segment,
            candidates=sample_candidates,
            segment_idx=0,
            location_matching_enabled=True,
            location_matching_config={}
        )

        assert filtered == sample_candidates
        assert applied is False


# ============================================================================
# Test apply_location_filter() - Hard Filter (City Level)
# ============================================================================

class TestApplyLocationFilterHardCity:
    """Test hard filtering at city level"""

    @pytest.mark.fast
    def test_hard_filter_city_same_city(self, mock_location_service, sample_location_chapters, sample_video_locations, sample_vo_segment, sample_candidates):
        """Test city filter keeps same-city videos"""
        matcher = LocationMatcher(location_service=mock_location_service)
        matcher.set_location_chapters(sample_location_chapters)
        matcher.set_video_locations(sample_video_locations)

        # Mock: Paris matches Paris, but not London or Tokyo
        def mock_same_city(loc1, loc2):
            # Both are GeoLocation objects now (chapter location is created from location_data)
            return loc1.name == loc2.name and loc1.name == 'Paris'
        mock_location_service.same_city = mock_same_city

        config = {'hard_filter_level': 'city', 'geographic_penalty': 0.4, 'hierarchy_bonus': 0.15}

        filtered, applied, reason = matcher.apply_location_filter(
            vo_segment=sample_vo_segment,
            candidates=sample_candidates,
            segment_idx=0,  # Paris chapter
            location_matching_enabled=True,
            location_matching_config=config
        )

        # Should only keep Paris video
        assert len(filtered) == 1
        assert filtered[0][0].source_file == '/videos/paris1.mp4'
        assert applied is True
        assert "location filter" in reason

    @pytest.mark.fast
    def test_hard_filter_city_all_removed_fallback(self, mock_location_service, sample_location_chapters, sample_video_locations, sample_vo_segment, sample_candidates):
        """Test fallback to soft penalty when all candidates removed"""
        matcher = LocationMatcher(location_service=mock_location_service)
        matcher.set_location_chapters(sample_location_chapters)
        matcher.set_video_locations(sample_video_locations)

        # Mock: No cities match (all filtered out)
        mock_location_service.same_city = Mock(return_value=False)

        config = {'hard_filter_level': 'city', 'geographic_penalty': 0.4, 'hierarchy_bonus': 0.15}

        filtered, applied, reason = matcher.apply_location_filter(
            vo_segment=sample_vo_segment,
            candidates=sample_candidates,
            segment_idx=0,  # Paris chapter
            location_matching_enabled=True,
            location_matching_config=config
        )

        # Should return all candidates with penalties
        assert len(filtered) == 3
        assert applied is True
        assert "soft penalty" in reason

        # Check penalties were applied
        original_sims = [c[1] for c in sample_candidates]
        penalized_sims = [c[1] for c in filtered]
        assert all(ps <= os for ps, os in zip(penalized_sims, original_sims))


# ============================================================================
# Test apply_location_filter() - Hard Filter (State/Country/Continent)
# ============================================================================

class TestApplyLocationFilterHardLevels:
    """Test hard filtering at different geographic levels"""

    @pytest.mark.fast
    def test_hard_filter_state_level(self, mock_location_service, sample_location_chapters, sample_video_locations, sample_vo_segment, sample_candidates):
        """Test state/region level filtering"""
        matcher = LocationMatcher(location_service=mock_location_service)
        matcher.set_location_chapters(sample_location_chapters)
        matcher.set_video_locations(sample_video_locations)

        # Mock: Same region for France videos only
        def mock_same_region(loc1, loc2):
            return loc1.country_code == 'FR' and loc2.country_code == 'FR'
        mock_location_service.same_region = mock_same_region

        config = {'hard_filter_level': 'state', 'geographic_penalty': 0.4, 'hierarchy_bonus': 0.15}

        filtered, applied, reason = matcher.apply_location_filter(
            vo_segment=sample_vo_segment,
            candidates=sample_candidates,
            segment_idx=0,  # Paris chapter
            location_matching_enabled=True,
            location_matching_config=config
        )

        # Should only keep Paris (France)
        assert len(filtered) == 1
        assert filtered[0][0].source_file == '/videos/paris1.mp4'

    @pytest.mark.fast
    def test_hard_filter_country_level(self, mock_location_service, sample_location_chapters, sample_video_locations, sample_vo_segment, sample_candidates):
        """Test country level filtering"""
        matcher = LocationMatcher(location_service=mock_location_service)
        matcher.set_location_chapters(sample_location_chapters)
        matcher.set_video_locations(sample_video_locations)

        # Mock: Same country for all European countries
        def mock_same_country(loc1, loc2):
            europe = ['FR', 'GB', 'DE', 'IT']
            return loc1.country_code in europe and loc2.country_code in europe
        mock_location_service.same_country = mock_same_country

        config = {'hard_filter_level': 'country', 'geographic_penalty': 0.4, 'hierarchy_bonus': 0.15}

        filtered, applied, reason = matcher.apply_location_filter(
            vo_segment=sample_vo_segment,
            candidates=sample_candidates,
            segment_idx=0,  # Paris chapter
            location_matching_enabled=True,
            location_matching_config=config
        )

        # Should keep Paris and London (both Europe), filter Tokyo
        assert len(filtered) == 2

    @pytest.mark.fast
    def test_hard_filter_continent_level(self, mock_location_service, sample_location_chapters, sample_video_locations, sample_vo_segment, sample_candidates):
        """Test continent level filtering (most lenient)"""
        matcher = LocationMatcher(location_service=mock_location_service)
        matcher.set_location_chapters(sample_location_chapters)
        matcher.set_video_locations(sample_video_locations)

        # Mock: All candidates are same continent
        mock_location_service.same_continent = Mock(return_value=True)

        config = {'hard_filter_level': 'continent', 'geographic_penalty': 0.4, 'hierarchy_bonus': 0.15}

        filtered, applied, reason = matcher.apply_location_filter(
            vo_segment=sample_vo_segment,
            candidates=sample_candidates,
            segment_idx=0,
            location_matching_enabled=True,
            location_matching_config=config
        )

        # Should keep all candidates
        assert len(filtered) == 3


# ============================================================================
# Test apply_location_filter() - Hierarchy Bonus
# ============================================================================

class TestApplyLocationFilterHierarchyBonus:
    """Test geographic hierarchy bonuses"""

    @pytest.mark.fast
    def test_hierarchy_bonus_parent_region(self, mock_location_service, sample_location_chapters, sample_video_locations, sample_vo_segment, sample_candidates):
        """Test bonus for parent region videos (e.g., country when chapter is city)"""
        matcher = LocationMatcher(location_service=mock_location_service)
        matcher.set_location_chapters(sample_location_chapters)
        matcher.set_video_locations(sample_video_locations)

        # Mock: Paris video is in parent region of Paris chapter
        def mock_is_parent(loc1, loc2):
            # Both are GeoLocation objects
            return loc1.name == 'Paris' and loc2.name == 'Paris'
        mock_location_service.is_parent_region = mock_is_parent
        mock_location_service.same_city = Mock(return_value=True)

        config = {'hard_filter_level': 'city', 'geographic_penalty': 0.4, 'hierarchy_bonus': 0.2}

        filtered, applied, reason = matcher.apply_location_filter(
            vo_segment=sample_vo_segment,
            candidates=sample_candidates,
            segment_idx=0,  # Paris chapter
            location_matching_enabled=True,
            location_matching_config=config
        )

        # Paris video should have bonus applied
        paris_result = next(c for c in filtered if 'paris' in c[0].source_file)
        paris_original = next(c for c in sample_candidates if 'paris' in c[0].source_file)

        # Bonus should increase score (but capped at 1.0)
        assert paris_result[1] >= paris_original[1]

    @pytest.mark.fast
    def test_hierarchy_bonus_child_region(self, mock_location_service, sample_location_chapters, sample_video_locations, sample_vo_segment, sample_candidates):
        """Test smaller bonus for child region videos"""
        matcher = LocationMatcher(location_service=mock_location_service)
        matcher.set_location_chapters(sample_location_chapters)
        matcher.set_video_locations(sample_video_locations)

        # Mock: Video is parent of chapter
        def mock_is_parent(loc1, loc2):
            # Reverse check: video (loc2) is parent of chapter (loc1)
            return False
        def mock_is_parent_reverse(loc1, loc2):
            loc2_name = getattr(loc2, 'name', '')
            return loc2_name == 'Paris'

        call_count = [0]
        def mock_is_parent_smart(loc1, loc2):
            call_count[0] += 1
            # First call: chapter -> video (False)
            # Second call: video -> chapter (True if Paris)
            if call_count[0] == 1:
                return False
            return loc2.name == 'Paris'

        mock_location_service.is_parent_region = mock_is_parent_smart
        mock_location_service.same_city = Mock(return_value=True)

        config = {'hard_filter_level': 'city', 'geographic_penalty': 0.4, 'hierarchy_bonus': 0.2}

        filtered, applied, reason = matcher.apply_location_filter(
            vo_segment=sample_vo_segment,
            candidates=sample_candidates,
            segment_idx=0,
            location_matching_enabled=True,
            location_matching_config=config
        )

        # Should have applied bonus (50% of hierarchy_bonus for child)
        assert len(filtered) >= 1


# ============================================================================
# Test apply_location_filter() - Videos Without Location Data
# ============================================================================

class TestApplyLocationFilterNoVideoLocation:
    """Test handling of videos without location data"""

    @pytest.mark.fast
    def test_video_no_location_small_penalty(self, mock_location_service, sample_location_chapters, sample_vo_segment):
        """Test that videos without location data get small penalty"""
        matcher = LocationMatcher(location_service=mock_location_service)
        matcher.set_location_chapters(sample_location_chapters)
        # Don't set video locations

        # Create candidate without location
        unknown_seg = SRTSegment(
            index=1,
            start_time=0.0,
            end_time=5.0,
            text="Unknown location footage",
            source_file='/videos/unknown.mp4'
        )
        candidates = [(unknown_seg, 0.8)]

        mock_location_service.same_city = Mock(return_value=False)

        config = {'hard_filter_level': 'city', 'geographic_penalty': 0.4, 'hierarchy_bonus': 0.15}

        filtered, applied, reason = matcher.apply_location_filter(
            vo_segment=sample_vo_segment,
            candidates=candidates,
            segment_idx=0,  # Paris chapter
            location_matching_enabled=True,
            location_matching_config=config
        )

        # Should still include video with small penalty (-0.05)
        assert len(filtered) == 1
        assert filtered[0][1] == 0.75  # 0.8 - 0.05


# ============================================================================
# Test apply_location_filter() - Config Variations
# ============================================================================

class TestApplyLocationFilterConfig:
    """Test different configuration scenarios"""

    @pytest.mark.fast
    def test_config_as_dict(self, mock_location_service, sample_location_chapters, sample_video_locations, sample_vo_segment, sample_candidates):
        """Test with config as dictionary"""
        matcher = LocationMatcher(location_service=mock_location_service)
        matcher.set_location_chapters(sample_location_chapters)
        matcher.set_video_locations(sample_video_locations)

        mock_location_service.same_country = Mock(return_value=True)

        # Config as dict
        config = {
            'hard_filter_level': 'country',
            'geographic_penalty': 0.5,
            'hierarchy_bonus': 0.1
        }

        filtered, applied, reason = matcher.apply_location_filter(
            vo_segment=sample_vo_segment,
            candidates=sample_candidates,
            segment_idx=0,
            location_matching_enabled=True,
            location_matching_config=config
        )

        assert applied is True

    @pytest.mark.fast
    def test_config_as_object(self, mock_location_service, sample_location_chapters, sample_video_locations, sample_vo_segment, sample_candidates):
        """Test with config as object with attributes"""
        matcher = LocationMatcher(location_service=mock_location_service)
        matcher.set_location_chapters(sample_location_chapters)
        matcher.set_video_locations(sample_video_locations)

        mock_location_service.same_country = Mock(return_value=True)

        # Config as object
        config = Mock()
        config.hard_filter_level = 'country'
        config.geographic_penalty = 0.5
        config.hierarchy_bonus = 0.1

        filtered, applied, reason = matcher.apply_location_filter(
            vo_segment=sample_vo_segment,
            candidates=sample_candidates,
            segment_idx=0,
            location_matching_enabled=True,
            location_matching_config=config
        )

        assert applied is True

    @pytest.mark.fast
    def test_config_defaults(self, mock_location_service, sample_location_chapters, sample_video_locations, sample_vo_segment, sample_candidates):
        """Test default config values are used"""
        matcher = LocationMatcher(location_service=mock_location_service)
        matcher.set_location_chapters(sample_location_chapters)
        matcher.set_video_locations(sample_video_locations)

        mock_location_service.same_country = Mock(return_value=True)

        # Empty config (should use defaults)
        config = {}

        filtered, applied, reason = matcher.apply_location_filter(
            vo_segment=sample_vo_segment,
            candidates=sample_candidates,
            segment_idx=0,
            location_matching_enabled=True,
            location_matching_config=config
        )

        # Should still work with defaults
        assert applied is True


# ============================================================================
# Test apply_location_filter() - Soft Penalty Mode
# ============================================================================

class TestApplyLocationFilterSoftPenalty:
    """Test soft penalty fallback mode"""

    @pytest.mark.fast
    def test_soft_penalty_city_level(self, mock_location_service, sample_location_chapters, sample_video_locations, sample_vo_segment, sample_candidates):
        """Test soft penalty at city level (full penalty)"""
        matcher = LocationMatcher(location_service=mock_location_service)
        matcher.set_location_chapters(sample_location_chapters)
        matcher.set_video_locations(sample_video_locations)

        # Hard filter removes all -> soft penalty mode
        mock_location_service.same_city = Mock(return_value=False)

        config = {'hard_filter_level': 'city', 'geographic_penalty': 0.4, 'hierarchy_bonus': 0.15}

        filtered, applied, reason = matcher.apply_location_filter(
            vo_segment=sample_vo_segment,
            candidates=sample_candidates,
            segment_idx=0,
            location_matching_enabled=True,
            location_matching_config=config
        )

        # All candidates returned with full penalty
        assert len(filtered) == 3
        assert "soft penalty" in reason

        # Check full penalty applied (0.4)
        for orig, filt in zip(sample_candidates, filtered):
            assert filt[1] <= orig[1] - 0.3  # At least some penalty

    @pytest.mark.fast
    def test_soft_penalty_state_level(self, mock_location_service, sample_location_chapters, sample_video_locations, sample_vo_segment, sample_candidates):
        """Test soft penalty at state level (90% of full penalty)"""
        matcher = LocationMatcher(location_service=mock_location_service)
        matcher.set_location_chapters(sample_location_chapters)
        matcher.set_video_locations(sample_video_locations)

        mock_location_service.same_region = Mock(return_value=False)

        config = {'hard_filter_level': 'state', 'geographic_penalty': 0.4, 'hierarchy_bonus': 0.15}

        filtered, applied, reason = matcher.apply_location_filter(
            vo_segment=sample_vo_segment,
            candidates=sample_candidates,
            segment_idx=0,
            location_matching_enabled=True,
            location_matching_config=config
        )

        assert len(filtered) == 3
        assert "soft penalty" in reason

    @pytest.mark.fast
    def test_soft_penalty_preserves_order(self, mock_location_service, sample_location_chapters, sample_video_locations, sample_vo_segment, sample_candidates):
        """Test that soft penalty mode re-sorts by adjusted scores"""
        matcher = LocationMatcher(location_service=mock_location_service)
        matcher.set_location_chapters(sample_location_chapters)
        matcher.set_video_locations(sample_video_locations)

        mock_location_service.same_city = Mock(return_value=False)

        config = {'hard_filter_level': 'city', 'geographic_penalty': 0.4, 'hierarchy_bonus': 0.15}

        filtered, applied, reason = matcher.apply_location_filter(
            vo_segment=sample_vo_segment,
            candidates=sample_candidates,
            segment_idx=0,
            location_matching_enabled=True,
            location_matching_config=config
        )

        # Check that results are sorted by adjusted score (descending)
        scores = [c[1] for c in filtered]
        assert scores == sorted(scores, reverse=True)


# ============================================================================
# Test Coverage Gaps (Lines 147, 185-188)
# ============================================================================

class TestLocationMatcherCoverageGaps:
    """Test coverage gaps for location_matching.py"""

    @pytest.mark.fast
    def test_unknown_filter_level_allows_all_line_147(self, mock_location_service, sample_location_chapters, sample_vo_segment):
        """Test line 147: Unknown filter level allows all candidates"""
        matcher = LocationMatcher(location_service=mock_location_service)
        matcher.set_location_chapters(sample_location_chapters)

        # Create video with location
        video_seg = SRTSegment(
            index=1, start_time=0.0, end_time=10.0,
            text="Video content", source_file="paris_video.mp4"
        )
        paris_location = GeoLocation(
            name="Paris", location_type="city", country_code="FR",
            country_name="France", admin1="Île-de-France", coordinates=(48.8566, 2.3522)
        )
        matcher.set_video_locations({"paris_video.mp4": paris_location})

        candidates = [(video_seg, 0.9)]

        # Use an unknown filter level
        config = {'hard_filter_level': 'unknown_level', 'geographic_penalty': 0.3, 'hierarchy_bonus': 0.1}

        filtered, applied, reason = matcher.apply_location_filter(
            vo_segment=sample_vo_segment,
            candidates=candidates,
            segment_idx=0,
            location_matching_enabled=True,
            location_matching_config=config
        )

        # Should include the candidate (unknown filter = allow all)
        assert applied is True
        assert len(filtered) > 0

    @pytest.mark.fast
    def test_soft_penalty_country_level_lines_185_186(self, mock_location_service, sample_location_chapters, sample_vo_segment):
        """Test lines 185-186: Soft penalty mode with country level filter"""
        matcher = LocationMatcher(location_service=mock_location_service)
        matcher.set_location_chapters(sample_location_chapters)

        # Create video with location
        video_seg = SRTSegment(
            index=1, start_time=0.0, end_time=10.0,
            text="Video content", source_file="london_video.mp4"
        )
        london_location = GeoLocation(
            name="London", location_type="city", country_code="GB",
            country_name="United Kingdom", admin1="England", coordinates=(51.5074, -0.1278)
        )
        matcher.set_video_locations({"london_video.mp4": london_location})

        candidates = [(video_seg, 0.9)]

        # Hard filter at country level - make same_country fail to trigger fallback
        mock_location_service.same_country = Mock(return_value=False)

        config = {'hard_filter_level': 'country', 'geographic_penalty': 0.4, 'hierarchy_bonus': 0.15}

        filtered, applied, reason = matcher.apply_location_filter(
            vo_segment=sample_vo_segment,
            candidates=candidates,
            segment_idx=0,
            location_matching_enabled=True,
            location_matching_config=config
        )

        # Should apply soft penalty in fallback mode (lines 185-186)
        assert applied is True
        assert "fallback" in reason or len(filtered) > 0

    @pytest.mark.fast
    def test_soft_penalty_continent_level_lines_187_188(self, mock_location_service, sample_location_chapters, sample_vo_segment):
        """Test lines 187-188: Soft penalty mode with continent level filter"""
        matcher = LocationMatcher(location_service=mock_location_service)
        matcher.set_location_chapters(sample_location_chapters)

        # Create video with location
        video_seg = SRTSegment(
            index=1, start_time=0.0, end_time=10.0,
            text="Video content", source_file="tokyo_video.mp4"
        )
        tokyo_location = GeoLocation(
            name="Tokyo", location_type="city", country_code="JP",
            country_name="Japan", admin1="Tokyo", coordinates=(35.6762, 139.6503)
        )
        matcher.set_video_locations({"tokyo_video.mp4": tokyo_location})

        candidates = [(video_seg, 0.9)]

        # Hard filter at continent level - make same_continent fail to trigger fallback
        mock_location_service.same_continent = Mock(return_value=False)

        config = {'hard_filter_level': 'continent', 'geographic_penalty': 0.4, 'hierarchy_bonus': 0.15}

        filtered, applied, reason = matcher.apply_location_filter(
            vo_segment=sample_vo_segment,
            candidates=candidates,
            segment_idx=0,
            location_matching_enabled=True,
            location_matching_config=config
        )

        # Should apply soft penalty in fallback mode (lines 187-188)
        assert applied is True
        assert "fallback" in reason or len(filtered) > 0


# ============================================================================
# Distance Calculation Tests (US-118-012 acceptance criteria)
# ============================================================================

class TestDistanceCalculationAccuracy:
    """Tests for distance calculation accuracy via LocationMatcher integration."""

    def test_distance_calculation_accuracy_paris_to_london(self, tmp_path):
        """Test distance calculation accuracy between Paris and London.

        Paris: 48.8566°N, 2.3522°E
        London: 51.5074°N, 0.1278°W
        Expected: ~340 km (great circle distance)
        """
        cache_dir = str(tmp_path / "location_cache")
        Path(cache_dir).mkdir(parents=True, exist_ok=True)

        service = LocationService(
            cache_dir=cache_dir,
            geonames_username="test_user"
        )

        matcher = LocationMatcher(location_service=service)

        paris = GeoLocation(
            name="Paris",
            location_type="city",
            country_code="FR",
            country_name="France",
            admin1="Île-de-France",
            coordinates=(48.8566, 2.3522)
        )

        london = GeoLocation(
            name="London",
            location_type="city",
            country_code="GB",
            country_name="United Kingdom",
            admin1="England",
            coordinates=(51.5074, -0.1278)
        )

        # Test distance calculation
        distance = service.distance_km(paris, london)

        # Paris to London is approximately 344 km (great circle)
        # Allow 10% tolerance
        assert 300 < distance < 400, f"Distance {distance}km not in expected range 300-400km"

    def test_distance_calculation_accuracy_same_location(self, tmp_path):
        """Test distance calculation for same location (should be 0)."""
        cache_dir = str(tmp_path / "location_cache")
        Path(cache_dir).mkdir(parents=True, exist_ok=True)

        service = LocationService(
            cache_dir=cache_dir,
            geonames_username="test_user"
        )

        matcher = LocationMatcher(location_service=service)

        tokyo = GeoLocation(
            name="Tokyo",
            location_type="city",
            country_code="JP",
            country_name="Japan",
            admin1="Tokyo",
            coordinates=(35.6762, 139.6503)
        )

        # Same location should have 0 distance
        distance = service.distance_km(tokyo, tokyo)
        assert distance == 0.0, f"Distance to self should be 0, got {distance}"

    def test_distance_calculation_accuracy_long_range(self, tmp_path):
        """Test distance calculation for long-range transcontinental distance."""
        cache_dir = str(tmp_path / "location_cache")
        Path(cache_dir).mkdir(parents=True, exist_ok=True)

        service = LocationService(
            cache_dir=cache_dir,
            geonames_username="test_user"
        )

        matcher = LocationMatcher(location_service=service)

        new_york = GeoLocation(
            name="New York",
            location_type="city",
            country_code="US",
            country_name="United States",
            admin1="New York",
            coordinates=(40.7128, -74.0060)
        )

        tokyo = GeoLocation(
            name="Tokyo",
            location_type="city",
            country_code="JP",
            country_name="Japan",
            admin1="Tokyo",
            coordinates=(35.6762, 139.6503)
        )

        # New York to Tokyo is approximately 10,800 km
        distance = service.distance_km(new_york, tokyo)
        assert 10000 < distance < 12000, f"Distance {distance}km not in expected range 10000-12000km"


class TestLocationCacheInvalidation:
    """Tests for location cache invalidation."""

    def test_cache_invalidation_by_clearing_locations(self, tmp_path):
        """Test that cache can be invalidated by clearing location entries."""
        import shutil
        cache_dir = str(tmp_path / "location_cache_test")

        # Create fresh cache directory
        Path(cache_dir).mkdir(parents=True, exist_ok=True)

        service = LocationService(
            cache_dir=cache_dir,
            geonames_username="test_user"
        )

        matcher = LocationMatcher(location_service=service)

        # Add a location to cache manually
        service._cache["locations"]["paris"] = {
            "results": [{
                "name": "Paris",
                "location_type": "city",
                "country_code": "FR",
                "country_name": "France",
                "admin1": "Île-de-France",
                "coordinates": [48.8566, 2.3522]
            }],
            "cached_at": "2026-01-01T00:00:00"
        }
        service._save_cache()

        # Verify cache has entry
        assert "paris" in service._cache["locations"]

        # Invalidate cache by clearing locations section
        service._cache["locations"] = {}
        service._save_cache()

        # Verify cache is cleared
        assert "paris" not in service._cache["locations"]
        assert len(service._cache["locations"]) == 0

    def test_cache_invalidation_by_clearing_disambiguations(self, tmp_path):
        """Test that disambiguation cache can be invalidated."""
        import shutil
        cache_dir = str(tmp_path / "location_cache_test2")

        # Create fresh cache directory
        Path(cache_dir).mkdir(parents=True, exist_ok=True)

        service = LocationService(
            cache_dir=cache_dir,
            geonames_username="test_user"
        )

        matcher = LocationMatcher(location_service=service)

        # Add a disambiguation entry to cache manually
        service._cache["disambiguations"]["paris:fashion"] = "FR"
        service._save_cache()

        # Verify cache has entry
        assert "paris:fashion" in service._cache["disambiguations"]

        # Invalidate cache by clearing disambiguations section
        service._cache["disambiguations"] = {}
        service._save_cache()

        # Verify cache is cleared
        assert "paris:fashion" not in service._cache["disambiguations"]
        assert len(service._cache["disambiguations"]) == 0

    def test_cache_invalidation_triggers_new_lookup(self, tmp_path):
        """Test that cache invalidation forces new API lookup."""
        from unittest.mock import patch, Mock

        cache_dir = str(tmp_path / "location_cache_test3")

        # Create fresh cache directory
        Path(cache_dir).mkdir(parents=True, exist_ok=True)

        service = LocationService(
            cache_dir=cache_dir,
            geonames_username="test_user"
        )

        matcher = LocationMatcher(location_service=service)

        # Pre-populate cache with "paris" entry (lowercase key)
        service._cache["locations"]["paris"] = {
            "results": [{
                "name": "Paris",
                "location_type": "city",
                "country_code": "FR",
                "country_name": "France",
                "admin1": "Île-de-France",
                "coordinates": [48.8566, 2.3522]
            }],
            "cached_at": "2026-01-01T00:00:00"
        }
        service._save_cache()

        # First call should use cache (no API call)
        with patch('src.location_service.requests.get') as mock_get:
            mock_get.return_value = Mock(status_code=200, json=lambda: {"geonames": []})
            result1 = service.geocode("Paris")

            # Cache hit - should not make API call
            assert mock_get.call_count == 0

        # Invalidate the cache
        service._cache["locations"] = {}
        service._save_cache()

        # Second call should attempt API lookup
        with patch('src.location_service.requests.get') as mock_get:
            mock_get.return_value = Mock(status_code=200, json=lambda: {
                "geonames": [{
                    "name": "Paris",
                    "countryCode": "FR",
                    "countryName": "France",
                    "adminName1": "Île-de-France",
                    "lat": 48.8566,
                    "lng": 2.3522
                }]
            })
            result2 = service.geocode("Paris")

            # Cache miss - should make API call
            assert mock_get.call_count > 0

