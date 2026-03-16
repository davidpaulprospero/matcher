"""
Tests for src/media_sources/utils.py

Tests utility functions for entity image handling:
- build_entity_query() - trimmed search query generation
- check_local_entity_images() - local image file discovery
- map_entities_to_segments() - entity-to-segment mapping
- restore_entity_images_from_disk() - metadata reconstruction
"""

import pytest
import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.media_sources.utils import (
    build_entity_query,
    check_local_entity_images,
    map_entities_to_segments,
    restore_entity_images_from_disk
)


class TestBuildEntityQuery:
    """Test build_entity_query() function"""

    @pytest.mark.fast
    def test_build_entity_query_name_only(self):
        """Test query with entity name only"""
        entity = {'text': 'Eiffel Tower', 'type': 'LOCATION', 'context': ''}

        query = build_entity_query(entity)

        assert query == 'Eiffel Tower'

    @pytest.mark.fast
    def test_build_entity_query_with_context(self):
        """Test query with entity name and context"""
        entity = {
            'text': 'Einstein',
            'type': 'PERSON',
            'context': 'physicist famous scientist'
        }

        query = build_entity_query(entity)

        assert 'Einstein' in query
        assert 'physicist' in query

    @pytest.mark.fast
    def test_build_entity_query_with_topic(self):
        """Test query with entity name and topic"""
        entity = {'text': 'Napoleon', 'type': 'PERSON', 'context': ''}
        topic = 'French history documentary'

        query = build_entity_query(entity, topic=topic)

        assert 'Napoleon' in query
        # Should include topic words (excluding 'documentary')
        assert 'French' in query or 'history' in query

    @pytest.mark.fast
    def test_build_entity_query_full(self):
        """Test query with all fields"""
        entity = {
            'text': 'Mount Everest',
            'type': 'LOCATION',
            'context': 'highest mountain peak'
        }
        topic = 'Himalayan expedition documentary footage'

        query = build_entity_query(entity, topic=topic)

        assert 'Mount Everest' in query
        assert 'highest' in query or 'mountain' in query

    @pytest.mark.fast
    def test_build_entity_query_long_truncation(self):
        """Test query truncation for very long inputs"""
        entity = {
            'text': 'VeryLongEntityName',
            'type': 'ORGANIZATION',
            'context': 'word1 word2 word3 word4 word5 word6 word7 word8'
        }
        topic = 'topic1 topic2 topic3 topic4 topic5'

        query = build_entity_query(entity, topic=topic)

        # Should be limited to ~6 words max
        assert len(query.split()) <= 7

    @pytest.mark.fast
    def test_build_entity_query_empty_entity(self):
        """Test with empty entity name"""
        entity = {'text': '', 'type': 'PERSON', 'context': 'some context'}

        query = build_entity_query(entity)

        assert query == ""

    @pytest.mark.fast
    def test_build_entity_query_filters_common_words(self):
        """Test that common words are filtered from topic"""
        entity = {'text': 'Paris', 'type': 'LOCATION', 'context': ''}
        topic = 'documentary footage about Paris history'

        query = build_entity_query(entity, topic=topic)

        # 'documentary' and 'footage' should be filtered out
        assert 'documentary' not in query.lower()
        assert 'footage' not in query.lower()


class TestCheckLocalEntityImages:
    """Test check_local_entity_images() function"""

    @pytest.mark.fast
    def test_check_local_entity_images_found(self, tmp_path):
        """Test finding existing entity images"""
        # Create image and metadata
        img_path = tmp_path / "12345.jpg"
        img_path.write_text("fake image")

        meta_path = tmp_path / "12345.entity.json"
        meta_path.write_text(json.dumps({
            'entity_name': 'Einstein',
            'entity_type': 'PERSON',
            'query': 'Einstein physicist'
        }))

        images = check_local_entity_images(str(tmp_path), 'Einstein')

        assert len(images) == 1
        assert '12345.jpg' in images[0]

    @pytest.mark.fast
    def test_check_local_entity_images_case_insensitive(self, tmp_path):
        """Test case-insensitive entity matching"""
        img_path = tmp_path / "67890.jpg"
        img_path.write_text("fake image")

        meta_path = tmp_path / "67890.entity.json"
        meta_path.write_text(json.dumps({
            'entity_name': 'Einstein',
            'entity_type': 'PERSON'
        }))

        # Search with different case
        images = check_local_entity_images(str(tmp_path), 'einstein')

        assert len(images) == 1

    @pytest.mark.fast
    def test_check_local_entity_images_multiple_formats(self, tmp_path):
        """Test finding images in different formats"""
        # Create PNG image
        img_path = tmp_path / "11111.png"
        img_path.write_text("fake png")

        meta_path = tmp_path / "11111.entity.json"
        meta_path.write_text(json.dumps({
            'entity_name': 'Tesla',
            'entity_type': 'PERSON'
        }))

        images = check_local_entity_images(str(tmp_path), 'Tesla')

        assert len(images) == 1
        assert '11111.png' in images[0]

    @pytest.mark.fast
    def test_check_local_entity_images_not_found(self, tmp_path):
        """Test when entity images don't exist"""
        images = check_local_entity_images(str(tmp_path), 'NonExistent')

        assert images == []

    @pytest.mark.fast
    def test_check_local_entity_images_directory_not_found(self):
        """Test with nonexistent directory"""
        images = check_local_entity_images('/nonexistent/path', 'Einstein')

        assert images == []

    @pytest.mark.fast
    def test_check_local_entity_images_corrupted_metadata(self, tmp_path):
        """Test handling of corrupted JSON metadata"""
        img_path = tmp_path / "22222.jpg"
        img_path.write_text("fake image")

        meta_path = tmp_path / "22222.entity.json"
        meta_path.write_text("invalid json {{{")

        images = check_local_entity_images(str(tmp_path), 'Einstein')

        # Should handle gracefully
        assert images == []

    @pytest.mark.fast
    def test_check_local_entity_images_multiple_entities(self, tmp_path):
        """Test finding multiple images for same entity"""
        # Create two images for Einstein
        for i, filename in enumerate(['11111', '22222']):
            img_path = tmp_path / f"{filename}.jpg"
            img_path.write_text(f"fake image {i}")

            meta_path = tmp_path / f"{filename}.entity.json"
            meta_path.write_text(json.dumps({
                'entity_name': 'Einstein',
                'entity_type': 'PERSON'
            }))

        images = check_local_entity_images(str(tmp_path), 'Einstein')

        assert len(images) == 2


class TestMapEntitiesToSegments:
    """Test map_entities_to_segments() function"""

    @pytest.mark.fast
    def test_map_entities_to_segments_basic(self):
        """Test basic entity-to-segment mapping"""
        entities = [
            {'text': 'Einstein', 'type': 'PERSON'},
            {'text': 'Paris', 'type': 'LOCATION'}
        ]
        segments = [
            {'text': 'Einstein was born in Germany'},
            {'text': 'He later moved to Paris'},
            {'text': 'Einstein developed the theory of relativity'}
        ]

        mapping = map_entities_to_segments(entities, segments)

        assert 'Einstein' in mapping
        assert 'Paris' in mapping
        assert 0 in mapping['Einstein']
        assert 2 in mapping['Einstein']
        assert 1 in mapping['Paris']

    @pytest.mark.fast
    def test_map_entities_to_segments_case_insensitive(self):
        """Test case-insensitive matching"""
        entities = [{'text': 'Einstein', 'type': 'PERSON'}]
        segments = [
            {'text': 'EINSTEIN was a physicist'},
            {'text': 'einstein made discoveries'}
        ]

        mapping = map_entities_to_segments(entities, segments)

        assert len(mapping['Einstein']) == 2

    @pytest.mark.fast
    def test_map_entities_to_segments_no_matches(self):
        """Test when entity doesn't appear in any segment"""
        entities = [{'text': 'Tesla', 'type': 'PERSON'}]
        segments = [
            {'text': 'Einstein was a physicist'},
            {'text': 'He lived in Germany'}
        ]

        mapping = map_entities_to_segments(entities, segments)

        assert 'Tesla' in mapping
        assert mapping['Tesla'] == []

    @pytest.mark.fast
    def test_map_entities_to_segments_empty_entities(self):
        """Test with empty entity list"""
        entities = []
        segments = [{'text': 'Some text'}]

        mapping = map_entities_to_segments(entities, segments)

        assert mapping == {}

    @pytest.mark.fast
    def test_map_entities_to_segments_empty_entity_name(self):
        """Test handling of entity with empty name"""
        entities = [
            {'text': '', 'type': 'PERSON'},
            {'text': 'Paris', 'type': 'LOCATION'}
        ]
        segments = [{'text': 'Paris is beautiful'}]

        mapping = map_entities_to_segments(entities, segments)

        # Empty entity should be skipped
        assert '' not in mapping
        assert 'Paris' in mapping


class TestRestoreEntityImagesFromDisk:
    """Test restore_entity_images_from_disk() function"""

    @pytest.mark.fast
    def test_restore_entity_images_basic(self, tmp_path):
        """Test basic restoration of entity images"""
        # Create entity image with metadata
        img_path = tmp_path / "12345.jpg"
        img_path.write_text("fake image")

        meta_path = tmp_path / "12345.entity.json"
        meta_path.write_text(json.dumps({
            'entity_name': 'Einstein',
            'entity_type': 'PERSON',
            'context': 'physicist',
            'query': 'Einstein physicist'
        }))

        results = restore_entity_images_from_disk(str(tmp_path))

        assert 'Einstein' in results
        assert results['Einstein'].entity_name == 'Einstein'
        assert results['Einstein'].entity_type == 'PERSON'
        assert len(results['Einstein'].images) == 1
        assert '12345.jpg' in results['Einstein'].images[0]

    @pytest.mark.fast
    def test_restore_entity_images_multiple_images(self, tmp_path):
        """Test restoration of multiple images for same entity"""
        # Create two images for Einstein
        for i, filename in enumerate(['11111', '22222']):
            img_path = tmp_path / f"{filename}.jpg"
            img_path.write_text(f"fake image {i}")

            meta_path = tmp_path / f"{filename}.entity.json"
            meta_path.write_text(json.dumps({
                'entity_name': 'Einstein',
                'entity_type': 'PERSON',
                'context': 'physicist',
                'query': 'Einstein physicist'
            }))

        results = restore_entity_images_from_disk(str(tmp_path))

        assert 'Einstein' in results
        assert len(results['Einstein'].images) == 2

    @pytest.mark.fast
    def test_restore_entity_images_with_segments(self, tmp_path):
        """Test restoration with voiceover segment mapping"""
        img_path = tmp_path / "12345.jpg"
        img_path.write_text("fake image")

        meta_path = tmp_path / "12345.entity.json"
        meta_path.write_text(json.dumps({
            'entity_name': 'Einstein',
            'entity_type': 'PERSON',
            'context': 'physicist',
            'query': 'Einstein physicist'
        }))

        segments = [
            {'text': 'Einstein was born in Germany'},
            {'text': 'Einstein developed relativity theory'}  # Changed "He" to "Einstein"
        ]

        results = restore_entity_images_from_disk(str(tmp_path), segments)

        assert 'Einstein' in results
        # Should have segment indices mapped
        assert len(results['Einstein'].segment_indices) == 2

    @pytest.mark.fast
    def test_restore_entity_images_directory_not_found(self):
        """Test with nonexistent directory"""
        results = restore_entity_images_from_disk('/nonexistent/path')

        assert results == {}

    @pytest.mark.fast
    def test_restore_entity_images_no_metadata_files(self, tmp_path):
        """Test with directory containing no metadata files"""
        results = restore_entity_images_from_disk(str(tmp_path))

        assert results == {}

    @pytest.mark.fast
    def test_restore_entity_images_missing_image_file(self, tmp_path):
        """Test when metadata exists but image file is missing"""
        meta_path = tmp_path / "12345.entity.json"
        meta_path.write_text(json.dumps({
            'entity_name': 'Einstein',
            'entity_type': 'PERSON'
        }))
        # Don't create corresponding image file

        results = restore_entity_images_from_disk(str(tmp_path))

        # Should skip entities with missing images
        assert results == {}

    @pytest.mark.fast
    def test_restore_entity_images_corrupted_metadata(self, tmp_path):
        """Test handling of corrupted JSON metadata"""
        img_path = tmp_path / "12345.jpg"
        img_path.write_text("fake image")

        meta_path = tmp_path / "12345.entity.json"
        meta_path.write_text("invalid json {{{")

        results = restore_entity_images_from_disk(str(tmp_path))

        # Should handle gracefully
        assert results == {}

    @pytest.mark.fast
    def test_restore_entity_images_multiple_entities(self, tmp_path):
        """Test restoration of multiple different entities"""
        # Create images for Einstein and Tesla
        for entity_name, filename in [('Einstein', '11111'), ('Tesla', '22222')]:
            img_path = tmp_path / f"{filename}.jpg"
            img_path.write_text(f"fake image {entity_name}")

            meta_path = tmp_path / f"{filename}.entity.json"
            meta_path.write_text(json.dumps({
                'entity_name': entity_name,
                'entity_type': 'PERSON',
                'query': f'{entity_name} physicist'
            }))

        results = restore_entity_images_from_disk(str(tmp_path))

        assert len(results) == 2
        assert 'Einstein' in results
        assert 'Tesla' in results


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
