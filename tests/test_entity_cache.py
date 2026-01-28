"""
Comprehensive tests for entity cache module.

Covers:
- EntityCache initialization
- CachedEntity dataclass operations
- Entity registration and storage
- Fuzzy entity name matching
- Age-based expiration
- Image copying/symlinking strategies
- Cache cleanup
- Migration from old format
- Cache statistics

Created: 2026-01-09 (Phase 4.2)
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch, mock_open
import tempfile
import shutil
import json
from datetime import datetime, timedelta

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.entity_cache import EntityCache, CachedEntity


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def temp_dir():
    """Create a temporary directory"""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path)


@pytest.fixture
def mock_config():
    """Create mock entity cache config"""
    config = Mock()
    config.enabled = True
    config.cache_dir = None  # Will be set per test
    config.fuzzy_threshold = 0.85
    config.max_age_days = 30
    config.cache_strategy = 'copy'
    return config


@pytest.fixture
def entity_cache(temp_dir, mock_config):
    """Create EntityCache with temp directory"""
    mock_config.cache_dir = str(temp_dir / "entity_cache")
    return EntityCache(mock_config)


@pytest.fixture
def sample_entity():
    """Create sample CachedEntity"""
    return CachedEntity(
        entity_name="Eiffel Tower",
        entity_type="GPE",
        images=["/cache/images/eiffel_tower/img1.jpg", "/cache/images/eiffel_tower/img2.jpg"],
        source_project="paris_travel",
        cached_at=datetime.now().isoformat(),
        query="eiffel tower landmark"
    )


@pytest.fixture
def sample_image_files(temp_dir):
    """Create sample image files"""
    images_dir = temp_dir / "images"
    images_dir.mkdir()

    img1 = images_dir / "test1.jpg"
    img2 = images_dir / "test2.jpg"
    img1.write_bytes(b"fake image 1")
    img2.write_bytes(b"fake image 2")

    return [str(img1), str(img2)]


# ============================================================================
# Test CachedEntity Dataclass
# ============================================================================

class TestCachedEntity:
    """Test CachedEntity dataclass"""

    @pytest.mark.fast
    def test_cached_entity_creation(self):
        """Test creating CachedEntity"""
        entity = CachedEntity(
            entity_name="Paris",
            entity_type="GPE",
            images=["/path/to/img.jpg"],
            source_project="travel_doc",
            cached_at="2026-01-09T10:00:00"
        )

        assert entity.entity_name == "Paris"
        assert entity.entity_type == "GPE"
        assert len(entity.images) == 1

    @pytest.mark.fast
    def test_cached_entity_to_dict(self, sample_entity):
        """Test CachedEntity serialization"""
        data = sample_entity.to_dict()

        assert data['entity_name'] == "Eiffel Tower"
        assert data['entity_type'] == "GPE"
        assert len(data['images']) == 2

    @pytest.mark.fast
    def test_cached_entity_from_dict(self, sample_entity):
        """Test CachedEntity deserialization"""
        data = sample_entity.to_dict()
        restored = CachedEntity.from_dict(data)

        assert restored.entity_name == sample_entity.entity_name
        assert restored.entity_type == sample_entity.entity_type
        assert len(restored.images) == len(sample_entity.images)


# ============================================================================
# Test EntityCache Initialization
# ============================================================================

class TestEntityCacheInit:
    """Test EntityCache initialization"""

    @pytest.mark.fast
    def test_init_creates_directories(self, entity_cache):
        """Test initialization creates cache directories"""
        assert entity_cache.cache_dir.exists()
        assert (entity_cache.cache_dir / EntityCache.IMAGES_SUBDIR).exists()

    @pytest.mark.fast
    def test_init_with_disabled_cache(self, temp_dir):
        """Test initialization with disabled cache"""
        config = Mock()
        config.enabled = False
        config.cache_dir = str(temp_dir / "cache")
        config.fuzzy_threshold = 0.85
        config.max_age_days = 30
        config.cache_strategy = 'copy'

        cache = EntityCache(config)

        assert cache.enabled is False

    @pytest.mark.fast
    def test_init_sets_config_values(self, temp_dir, mock_config):
        """Test initialization sets all config values"""
        mock_config.cache_dir = str(temp_dir / "cache")
        mock_config.fuzzy_threshold = 0.9
        mock_config.max_age_days = 60
        mock_config.cache_strategy = 'symlink'

        cache = EntityCache(mock_config)

        assert cache.fuzzy_threshold == 0.9
        assert cache.max_age_days == 60
        assert cache.cache_strategy == 'symlink'


# ============================================================================
# Test Entity Registration
# ============================================================================

class TestEntityRegistration:
    """Test adding entities to cache"""

    @pytest.mark.fast
    def test_add_entity_basic(self, entity_cache, sample_image_files):
        """Test adding entity to cache"""
        entity_cache.add_entity(
            entity_name="Paris",
            entity_type="GPE",
            image_paths=sample_image_files,
            source_project="test_proj"
        )

        # Should be in index
        result = entity_cache.find_entity("Paris")
        assert result is not None
        assert result.entity_name == "Paris"

    @pytest.mark.fast
    def test_add_entity_copies_images(self, entity_cache, sample_image_files):
        """Test adding entity copies images to cache"""
        entity_cache.add_entity(
            entity_name="London",
            entity_type="GPE",
            image_paths=sample_image_files,
            source_project="test_proj"
        )

        # Images should be copied to cache
        safe_name = entity_cache._safe_name("London")
        cache_entity_dir = entity_cache.cache_dir / EntityCache.IMAGES_SUBDIR / safe_name

        assert cache_entity_dir.exists()
        assert len(list(cache_entity_dir.glob("*.jpg"))) == 2

    @pytest.mark.fast
    def test_add_entity_with_empty_images(self, entity_cache):
        """Test adding entity with no images does nothing"""
        entity_cache.add_entity(
            entity_name="Empty",
            entity_type="PERSON",
            image_paths=[],
            source_project="test_proj"
        )

        # Should not be added
        result = entity_cache.find_entity("Empty")
        assert result is None

    @pytest.mark.fast
    def test_add_entity_when_disabled(self, temp_dir):
        """Test adding entity when cache is disabled"""
        config = Mock()
        config.enabled = False
        config.cache_dir = str(temp_dir / "cache")
        config.fuzzy_threshold = 0.85
        config.max_age_days = 30
        config.cache_strategy = 'copy'

        cache = EntityCache(config)

        # Should not raise, just return
        cache.add_entity("Test", "PERSON", ["/fake/path.jpg"], "proj")


# ============================================================================
# Test Entity Finding
# ============================================================================

class TestEntityFinding:
    """Test finding entities with fuzzy matching"""

    @pytest.mark.fast
    def test_find_entity_exact_match(self, entity_cache, sample_image_files):
        """Test finding entity with exact name match"""
        entity_cache.add_entity("Paris", "GPE", sample_image_files, "proj")

        result = entity_cache.find_entity("Paris")

        assert result is not None
        assert result.entity_name == "Paris"

    @pytest.mark.fast
    def test_find_entity_case_insensitive(self, entity_cache, sample_image_files):
        """Test finding is case-insensitive"""
        entity_cache.add_entity("Paris", "GPE", sample_image_files, "proj")

        result = entity_cache.find_entity("PARIS")

        assert result is not None
        assert result.entity_name == "Paris"

    @pytest.mark.fast
    def test_find_entity_fuzzy_match(self, entity_cache, sample_image_files):
        """Test finding entity with fuzzy matching"""
        entity_cache.add_entity("Eiffel Tower", "GPE", sample_image_files, "proj")
        entity_cache.fuzzy_threshold = 0.8

        result = entity_cache.find_entity("Eifel Tower")  # Typo

        assert result is not None
        assert result.entity_name == "Eiffel Tower"

    @pytest.mark.fast
    def test_find_entity_with_type_filter(self, entity_cache, sample_image_files):
        """Test finding entity filtered by type"""
        entity_cache.add_entity("Paris", "GPE", sample_image_files, "proj")

        # Should find with correct type
        result = entity_cache.find_entity("Paris", entity_type="GPE")
        assert result is not None

        # Should not find with wrong type
        result = entity_cache.find_entity("Paris", entity_type="PERSON")
        assert result is None

    @pytest.mark.fast
    def test_find_entity_not_found(self, entity_cache):
        """Test finding non-existent entity"""
        result = entity_cache.find_entity("NonExistent")

        assert result is None

    @pytest.mark.fast
    def test_find_entity_when_disabled(self, temp_dir):
        """Test finding entity when cache is disabled"""
        config = Mock()
        config.enabled = False
        config.cache_dir = str(temp_dir / "cache")
        config.fuzzy_threshold = 0.85
        config.max_age_days = 30
        config.cache_strategy = 'copy'

        cache = EntityCache(config)
        result = cache.find_entity("Test")

        assert result is None


# ============================================================================
# Test Fuzzy Matching
# ============================================================================

class TestFuzzyMatching:
    """Test fuzzy string matching"""

    @pytest.mark.fast
    def test_fuzzy_similarity_identical(self, entity_cache):
        """Test similarity of identical strings"""
        score = entity_cache._fuzzy_similarity("Paris", "Paris")

        assert score == 1.0

    @pytest.mark.fast
    def test_fuzzy_similarity_different_case(self, entity_cache):
        """Test similarity ignores case"""
        score = entity_cache._fuzzy_similarity("Paris", "PARIS")

        assert score == 1.0

    @pytest.mark.fast
    def test_fuzzy_similarity_similar(self, entity_cache):
        """Test similarity of similar strings"""
        score = entity_cache._fuzzy_similarity("Eiffel Tower", "Eifel Tower")

        assert score > 0.9  # Very similar

    @pytest.mark.fast
    def test_fuzzy_similarity_different(self, entity_cache):
        """Test similarity of different strings"""
        score = entity_cache._fuzzy_similarity("Paris", "London")

        assert score < 0.5

    @pytest.mark.fast
    def test_fuzzy_similarity_empty_string(self, entity_cache):
        """Test similarity with empty string"""
        score = entity_cache._fuzzy_similarity("Paris", "")

        assert score == 0.0


# ============================================================================
# Test Cache Validation
# ============================================================================

class TestCacheValidation:
    """Test cache entry validation"""

    @pytest.mark.fast
    def test_is_valid_recent_entry(self, entity_cache, sample_image_files):
        """Test valid recent entry"""
        entity_cache.add_entity("Paris", "GPE", sample_image_files, "proj")
        cached = entity_cache.find_entity("Paris")

        is_valid = entity_cache._is_valid(cached)

        assert is_valid is True

    @pytest.mark.fast
    def test_is_valid_expired_entry(self, entity_cache, temp_dir):
        """Test expired entry is invalid"""
        entity_cache.max_age_days = 1

        # Create entity with old timestamp
        old_entity = CachedEntity(
            entity_name="Old",
            entity_type="PERSON",
            images=[str(temp_dir / "img.jpg")],
            source_project="proj",
            cached_at=(datetime.now() - timedelta(days=2)).isoformat()
        )

        # Create the image file
        (temp_dir / "img.jpg").write_bytes(b"image")
        old_entity.images = [str(temp_dir / "img.jpg")]

        is_valid = entity_cache._is_valid(old_entity)

        assert is_valid is False

    @pytest.mark.fast
    def test_is_valid_missing_files(self, entity_cache):
        """Test entry with missing files is invalid"""
        entity_with_missing = CachedEntity(
            entity_name="Missing",
            entity_type="PERSON",
            images=["/nonexistent/img1.jpg", "/nonexistent/img2.jpg"],
            source_project="proj",
            cached_at=datetime.now().isoformat()
        )

        is_valid = entity_cache._is_valid(entity_with_missing)

        assert is_valid is False


# ============================================================================
# Test Image Retrieval Strategies
# ============================================================================

class TestImageRetrievalStrategies:
    """Test different image retrieval strategies"""

    @pytest.mark.fast
    def test_get_images_reference_strategy(self, entity_cache, sample_image_files, temp_dir):
        """Test reference strategy returns absolute paths"""
        entity_cache.cache_strategy = 'reference'
        entity_cache.add_entity("Test", "PERSON", sample_image_files, "proj")

        cached = entity_cache.find_entity("Test")
        project_dir = temp_dir / "project_images"

        paths = entity_cache.get_images_for_project(cached, str(project_dir))

        assert len(paths) == 2
        assert all(Path(p).is_absolute() for p in paths)

    @pytest.mark.fast
    def test_get_images_copy_strategy(self, entity_cache, sample_image_files, temp_dir):
        """Test copy strategy copies images to project"""
        entity_cache.cache_strategy = 'copy'
        entity_cache.add_entity("Test", "PERSON", sample_image_files, "proj")

        cached = entity_cache.find_entity("Test")
        project_dir = temp_dir / "project_images"

        paths = entity_cache.get_images_for_project(cached, str(project_dir))

        assert len(paths) == 2
        assert all(Path(p).exists() for p in paths)
        # Should be in project directory
        assert all(str(project_dir) in p for p in paths)

    @pytest.mark.fast
    def test_get_images_symlink_strategy(self, entity_cache, sample_image_files, temp_dir):
        """Test symlink strategy creates symlinks"""
        entity_cache.cache_strategy = 'symlink'
        entity_cache.add_entity("Test", "PERSON", sample_image_files, "proj")

        cached = entity_cache.find_entity("Test")
        project_dir = temp_dir / "project_images"

        paths = entity_cache.get_images_for_project(cached, str(project_dir))

        assert len(paths) == 2


# ============================================================================
# Test Cache Cleanup
# ============================================================================

class TestCacheCleanup:
    """Test cache cleanup"""

    @pytest.mark.fast
    def test_cleanup_removes_expired(self, entity_cache, sample_image_files):
        """Test cleanup removes expired entries"""
        entity_cache.max_age_days = 1
        entity_cache.add_entity("Recent", "PERSON", sample_image_files, "proj")

        # Manually create expired entry
        old_entity = CachedEntity(
            entity_name="Old",
            entity_type="PERSON",
            images=sample_image_files,
            source_project="proj",
            cached_at=(datetime.now() - timedelta(days=2)).isoformat()
        )
        entity_cache.set("Old", old_entity.to_dict())

        removed_count = entity_cache.cleanup()

        # Should remove old but keep recent
        assert removed_count == 1
        assert entity_cache.find_entity("Recent") is not None
        assert entity_cache.find_entity("Old") is None

    @pytest.mark.fast
    def test_cleanup_removes_missing_files(self, entity_cache):
        """Test cleanup removes entries with missing files"""
        missing_entity = CachedEntity(
            entity_name="Missing",
            entity_type="PERSON",
            images=["/nonexistent/img.jpg"],
            source_project="proj",
            cached_at=datetime.now().isoformat()
        )
        entity_cache.set("Missing", missing_entity.to_dict())

        removed_count = entity_cache.cleanup()

        assert removed_count == 1

    @pytest.mark.fast
    def test_cleanup_when_disabled(self, temp_dir):
        """Test cleanup when cache is disabled"""
        config = Mock()
        config.enabled = False
        config.cache_dir = str(temp_dir / "cache")
        config.fuzzy_threshold = 0.85
        config.max_age_days = 30
        config.cache_strategy = 'copy'

        cache = EntityCache(config)
        removed = cache.cleanup()

        assert removed == 0


# ============================================================================
# Test Cache Statistics
# ============================================================================

class TestCacheStatistics:
    """Test cache statistics"""

    @pytest.mark.fast
    def test_get_stats_empty_cache(self, entity_cache):
        """Test stats on empty cache"""
        stats = entity_cache.get_stats()

        assert stats['enabled'] is True
        assert stats['total_entities'] == 0
        assert stats['valid_entities'] == 0

    @pytest.mark.fast
    def test_get_stats_with_entities(self, entity_cache, sample_image_files):
        """Test stats with entities"""
        entity_cache.add_entity("Entity1", "PERSON", sample_image_files, "proj")
        entity_cache.add_entity("Entity2", "GPE", sample_image_files, "proj")

        stats = entity_cache.get_stats()

        assert stats['total_entities'] == 2
        assert stats['total_images'] == 4  # 2 images each
        assert stats['fuzzy_threshold'] == 0.85

    @pytest.mark.fast
    def test_get_stats_when_disabled(self, temp_dir):
        """Test stats when cache is disabled"""
        config = Mock()
        config.enabled = False
        config.cache_dir = str(temp_dir / "cache")
        config.fuzzy_threshold = 0.85
        config.max_age_days = 30
        config.cache_strategy = 'copy'

        cache = EntityCache(config)
        stats = cache.get_stats()

        assert stats['enabled'] is False


# ============================================================================
# Test Utility Functions
# ============================================================================

class TestUtilityFunctions:
    """Test utility functions"""

    @pytest.mark.fast
    def test_safe_name_basic(self, entity_cache):
        """Test safe name conversion"""
        safe = entity_cache._safe_name("Eiffel Tower")

        assert " " not in safe
        assert safe == "Eiffel_Tower"

    @pytest.mark.fast
    def test_safe_name_special_chars(self, entity_cache):
        """Test safe name removes special characters"""
        safe = entity_cache._safe_name("Entity/Name:Test")

        assert "/" not in safe
        assert ":" not in safe
        assert safe == "Entity_Name_Test"

    @pytest.mark.fast
    def test_safe_name_length_limit(self, entity_cache):
        """Test safe name limits length"""
        long_name = "A" * 100
        safe = entity_cache._safe_name(long_name)

        assert len(safe) <= 50


# ============================================================================
# Test Edge Cases
# ============================================================================

class TestEdgeCases:
    """Test edge cases"""

    @pytest.mark.fast
    def test_find_entity_best_fuzzy_match(self, entity_cache, sample_image_files):
        """Test finding returns best fuzzy match"""
        entity_cache.add_entity("Eiffel Tower", "GPE", sample_image_files, "proj1")
        entity_cache.add_entity("Tower Bridge", "GPE", sample_image_files, "proj2")
        entity_cache.fuzzy_threshold = 0.5

        # Should match Eiffel Tower better than Tower Bridge
        result = entity_cache.find_entity("Eiffel Towe")

        assert result.entity_name == "Eiffel Tower"

    @pytest.mark.fast
    def test_add_entity_skips_nonexistent_files(self, entity_cache, temp_dir):
        """Test adding entity skips files that don't exist"""
        fake_paths = [
            str(temp_dir / "nonexistent1.jpg"),
            str(temp_dir / "nonexistent2.jpg")
        ]

        entity_cache.add_entity("Test", "PERSON", fake_paths, "proj")

        # Should not crash, just skip those files
        result = entity_cache.find_entity("Test")
        assert result is None  # No valid images, so not added

    @pytest.mark.fast
    def test_get_images_creates_project_dir(self, entity_cache, sample_image_files, temp_dir):
        """Test get_images creates project directory if missing"""
        entity_cache.add_entity("Test", "PERSON", sample_image_files, "proj")
        cached = entity_cache.find_entity("Test")

        project_dir = temp_dir / "new_project" / "images"

        paths = entity_cache.get_images_for_project(cached, str(project_dir))

        assert project_dir.exists()
        assert len(paths) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
