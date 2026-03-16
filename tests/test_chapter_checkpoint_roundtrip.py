"""
Tests for US-71-009: Persist chapter-segment mapping in checkpoint.

Verifies:
- AC1: Chapter detection results serialized to checkpoint after match stage
- AC2: Chapter data survives pipeline resume (round-trip)
- AC3: Output stage can read chapter/listicle data from checkpoint
- AC4: Backward compat: missing chapter_data produces empty lists
- AC5: Round-trip serialization of VideoChapter and ListicleGroup
"""

import json
import pytest
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.checkpoint import CheckpointManager, CheckpointData, STAGE_ORDER
from src.chapter_detection.models import ChapterCandidate, ListicleGroup


# Sample chapter and listicle data for tests
SAMPLE_CHAPTERS = [
    ChapterCandidate(
        chapter_id=0,
        start_segment_idx=0,
        end_segment_idx=4,
        title="Introduction to Wildlife",
        topics=["wildlife", "nature"],
        confidence=0.92,
        detection_strategy="topic",
    ),
    ChapterCandidate(
        chapter_id=1,
        start_segment_idx=5,
        end_segment_idx=12,
        title="African Safari",
        topics=["safari", "africa"],
        location_name="Serengeti",
        location_type="landmark",
        confidence=0.88,
        detection_strategy="location",
    ),
]

SAMPLE_LISTICLE_GROUPS = [
    ListicleGroup(
        group_id=0,
        item_label="first",
        start_segment_idx=0,
        end_segment_idx=3,
        topic_keywords=["introduction", "overview"],
        expected_count=5,
    ),
    ListicleGroup(
        group_id=1,
        item_label="second",
        start_segment_idx=4,
        end_segment_idx=7,
        topic_keywords=["wildlife", "species"],
        expected_count=5,
    ),
]


class TestChapterDataRoundTrip:
    """AC1 + AC2 + AC5: Chapter/listicle data round-trips through checkpoint."""

    @pytest.mark.fast
    def test_chapter_data_saved_to_checkpoint(self, tmp_path):
        """AC1: Chapter detection results serialized to checkpoint.json after match stage."""
        manager = CheckpointManager(tmp_path, config_hash="test")

        chapter_data = {
            'chapters': [ch.to_dict() for ch in SAMPLE_CHAPTERS],
            'listicle_groups': [lg.to_dict() for lg in SAMPLE_LISTICLE_GROUPS],
        }

        match_stage_data = {
            'match_count': 10,
            'avg_confidence': 0.85,
            'matches': [],
            'chapter_data': chapter_data,
        }

        manager.save('MATCH', match_stage_data)

        # Verify chapter_data is in the checkpoint file
        with open(tmp_path / "checkpoint.json", 'r') as f:
            raw = json.load(f)

        # Top-level chapter_data field should be populated
        assert 'chapter_data' in raw
        assert len(raw['chapter_data']['chapters']) == 2
        assert len(raw['chapter_data']['listicle_groups']) == 2

        # Also stored inside match stage data
        assert raw['match']['chapter_data']['chapters'][0]['title'] == "Introduction to Wildlife"

    @pytest.mark.fast
    def test_chapter_data_survives_resume(self, tmp_path):
        """AC2: Loading checkpoint restores chapter/listicle structures."""
        manager = CheckpointManager(tmp_path, config_hash="test")

        chapter_data = {
            'chapters': [ch.to_dict() for ch in SAMPLE_CHAPTERS],
            'listicle_groups': [lg.to_dict() for lg in SAMPLE_LISTICLE_GROUPS],
        }

        manager.save('MATCH', {
            'match_count': 10,
            'avg_confidence': 0.85,
            'matches': [],
            'chapter_data': chapter_data,
        })

        # Reload from disk
        manager2 = CheckpointManager(tmp_path, config_hash="test")
        loaded = manager2.load()

        assert loaded is not None
        assert loaded.chapter_data is not None
        assert len(loaded.chapter_data['chapters']) == 2
        assert len(loaded.chapter_data['listicle_groups']) == 2

        # Verify chapter fields survive
        ch0 = loaded.chapter_data['chapters'][0]
        assert ch0['title'] == "Introduction to Wildlife"
        assert ch0['topics'] == ["wildlife", "nature"]
        assert ch0['confidence'] == 0.92

        # Verify listicle group fields survive
        lg0 = loaded.chapter_data['listicle_groups'][0]
        assert lg0['item_label'] == "first"
        assert lg0['expected_count'] == 5

    @pytest.mark.fast
    def test_roundtrip_deserialization_to_dataclasses(self, tmp_path):
        """AC5: Round-trip serialization produces valid ChapterCandidate/ListicleGroup."""
        manager = CheckpointManager(tmp_path)

        chapter_data = {
            'chapters': [ch.to_dict() for ch in SAMPLE_CHAPTERS],
            'listicle_groups': [lg.to_dict() for lg in SAMPLE_LISTICLE_GROUPS],
        }

        manager.save('MATCH', {
            'match_count': 5,
            'avg_confidence': 0.9,
            'matches': [],
            'chapter_data': chapter_data,
        })

        # Reload
        manager2 = CheckpointManager(tmp_path)
        loaded = manager2.load()
        restored_cd = loaded.chapter_data

        # Deserialize chapters back to dataclass
        restored_chapters = [
            ChapterCandidate.from_dict(d) for d in restored_cd['chapters']
        ]
        assert len(restored_chapters) == 2
        assert restored_chapters[0].title == "Introduction to Wildlife"
        assert restored_chapters[0].segment_count == 5  # 0-4 inclusive
        assert restored_chapters[1].location_name == "Serengeti"

        # Deserialize listicle groups back to dataclass
        restored_groups = [
            ListicleGroup.from_dict(d) for d in restored_cd['listicle_groups']
        ]
        assert len(restored_groups) == 2
        assert restored_groups[0].item_label == "first"
        assert restored_groups[0].segment_count == 4  # 0-3 inclusive
        assert restored_groups[1].topic_keywords == ["wildlife", "species"]


class TestOutputStageAccess:
    """AC3: Output stage can read chapter/listicle data from checkpoint."""

    @pytest.mark.fast
    def test_get_chapter_data_returns_chapters(self, tmp_path):
        """Output stage uses get_chapter_data() to access chapter info."""
        manager = CheckpointManager(tmp_path)

        chapter_data = {
            'chapters': [ch.to_dict() for ch in SAMPLE_CHAPTERS],
            'listicle_groups': [lg.to_dict() for lg in SAMPLE_LISTICLE_GROUPS],
        }

        manager.save('MATCH', {
            'match_count': 5,
            'avg_confidence': 0.9,
            'matches': [],
            'chapter_data': chapter_data,
        })

        # Simulate output stage reading chapter data
        manager2 = CheckpointManager(tmp_path)
        manager2.load()

        cd = manager2.get_chapter_data()
        assert len(cd['chapters']) == 2
        assert len(cd['listicle_groups']) == 2

    @pytest.mark.fast
    def test_get_chapter_data_also_in_match_stage_data(self, tmp_path):
        """Chapter data accessible via get_stage_data('MATCH') as well."""
        manager = CheckpointManager(tmp_path)

        chapter_data = {
            'chapters': [SAMPLE_CHAPTERS[0].to_dict()],
            'listicle_groups': [],
        }

        manager.save('MATCH', {
            'match_count': 1,
            'avg_confidence': 0.9,
            'matches': [],
            'chapter_data': chapter_data,
        })

        manager2 = CheckpointManager(tmp_path)
        manager2.load()

        match_data = manager2.get_stage_data('MATCH')
        assert 'chapter_data' in match_data
        assert len(match_data['chapter_data']['chapters']) == 1


class TestBackwardCompatibility:
    """AC4: Loading checkpoint without chapter_data produces empty lists."""

    @pytest.mark.fast
    def test_old_checkpoint_no_chapter_data(self, tmp_path):
        """Loading a pre-US-71-009 checkpoint returns empty chapter_data."""
        manager = CheckpointManager(tmp_path)

        # Save checkpoint without chapter_data (simulating old format)
        manager.save('MATCH', {
            'match_count': 10,
            'avg_confidence': 0.85,
            'matches': [],
        })

        manager2 = CheckpointManager(tmp_path)
        loaded = manager2.load()

        # Top-level chapter_data should be empty dict (default)
        assert loaded.chapter_data == {}

        # get_chapter_data() should return empty dict
        cd = manager2.get_chapter_data()
        assert cd == {}

    @pytest.mark.fast
    def test_old_checkpoint_no_chapter_data_field_at_all(self, tmp_path):
        """Manually crafted old checkpoint without chapter_data field."""
        old_checkpoint = {
            'version': '2.0',
            'created_at': '2026-01-01T00:00:00',
            'updated_at': '2026-01-01T00:00:00',
            'last_completed_stage': 'MATCH',
            'config_hash': '',
            'voiceover_path': '',
            'voiceover_hash': '',
            'analyze': {},
            'video_search': {},
            'caption': {},
            'match': {'match_count': 5, 'matches': []},
            'iterative_match': {},
            'download_segments': {},
            'stage_metrics': {},
            # Note: no chapter_data field
        }

        (tmp_path / "checkpoint.json").write_text(json.dumps(old_checkpoint))

        manager = CheckpointManager(tmp_path)
        loaded = manager.load()

        assert loaded is not None
        assert loaded.chapter_data == {}
        assert manager.get_chapter_data() == {}

    @pytest.mark.fast
    def test_empty_chapter_data_produces_empty_lists(self, tmp_path):
        """chapter_data with empty chapters/groups returns empty lists."""
        manager = CheckpointManager(tmp_path)

        manager.save('MATCH', {
            'match_count': 5,
            'avg_confidence': 0.9,
            'matches': [],
            'chapter_data': {'chapters': [], 'listicle_groups': []},
        })

        manager2 = CheckpointManager(tmp_path)
        loaded = manager2.load()

        cd = manager2.get_chapter_data()
        assert cd['chapters'] == []
        assert cd['listicle_groups'] == []
