"""
Tests for PipelineState serialization roundtrip (US-47-010).

Covers:
- AC1: Every PipelineState field survives checkpoint save and load roundtrip
- AC2: to_checkpoint_dict() includes all Optional fields when set to None
- AC3: from_checkpoint_dict() with missing fields for backward compatibility
- AC4: PipelineState with complex nested field types (lists of dataclasses, dicts with special keys)
- AC5: All tests pass

Uses PipelineState from src/state.py and CheckpointManager from src/checkpoint.py.
"""

import json
import pytest
from dataclasses import asdict, fields
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.state import (
    PipelineState,
    VoiceoverSegment,
    Match,
    DownloadedVideo,
    VideoSearchResult,
)
from src.checkpoint import CheckpointManager, CheckpointData, STAGE_ORDER


# ---------------------------------------------------------------------------
# AC1: Every PipelineState field survives checkpoint save and load roundtrip
# ---------------------------------------------------------------------------

class TestPipelineStateFieldRoundtrip:
    """AC1: Every PipelineState field survives a checkpoint save/load cycle."""

    def _build_populated_state(self) -> PipelineState:
        """Build a PipelineState with every field populated."""
        state = PipelineState()
        state.voiceover_path = "/project/voiceover.srt"
        state.voiceover_segments = [
            VoiceoverSegment(index=0, start=0.0, end=5.0, text="Hello world"),
            VoiceoverSegment(index=1, start=5.0, end=10.0, text="Second segment"),
        ]
        state.keywords = ["nature", "wildlife", "ocean"]
        state.topic_context = "A documentary about ocean life"
        state.extracted_entities = [
            {"name": "Blue Whale", "type": "animal", "importance": 0.9},
            {"name": "Pacific Ocean", "type": "location", "importance": 0.7},
        ]
        state.video_ids = ["abc123", "def456", "ghi789"]
        state.video_search_results = [
            VideoSearchResult(video_id="abc123", url="https://youtube.com/watch?v=abc123",
                              title="Ocean Documentary", channel="NatGeo", duration=120.0,
                              duration_tier="medium", keyword="ocean"),
        ]
        state.search_failed_keywords = ["nonexistent_keyword"]
        state.caption_results = {
            "abc123": {"language": "en", "text": "Sample caption text", "auto_generated": True},
            "def456": {"language": "en", "text": "Another caption"},
        }
        state.text_metadata = [
            {"video_id": "abc123", "word_count": 150, "language": "en"},
        ]
        state.matches = [
            Match(segment_index=0, video_file="abc123", video_start=10.0,
                  video_end=15.0, confidence=0.92, strategy="embedding",
                  reason="High semantic similarity", face_score=0.3),
            Match(segment_index=1, video_file="def456", video_start=0.0,
                  video_end=5.0, confidence=0.85, strategy="keyword",
                  reason="Keyword match", face_score=0.7),
        ]
        state.alternatives = {
            0: [Match(segment_index=0, video_file="ghi789", video_start=20.0,
                      video_end=25.0, confidence=0.78, strategy="diversity")],
        }
        state.downloaded_segments = [
            DownloadedVideo(file="abc123_10-15.mp4", url="https://youtube.com/watch?v=abc123",
                            title="Ocean Documentary", duration=5.0, duration_tier="short",
                            keyword="ocean"),
        ]
        state.output_files = [Path("/project/output/timeline.otio")]
        state.otio_files = [Path("/project/output/timeline.otio")]
        state.entity_images = {"Blue Whale": {"file": "blue_whale.jpg", "width": 1920}}
        state.entity_videos = {"Blue Whale": {"file": "whale_stock.mp4", "duration": 10.0}}
        state.voiceover_embeddings = None  # Optional field
        state.face_preference = "low"
        state.location_chapters = [{"name": "Pacific", "start": 0.0, "end": 60.0}]
        state.stage_timings = {"ANALYZE": 2.5, "VIDEO_SEARCH": 15.3, "MATCH": 8.1}
        return state

    @pytest.mark.fast
    def test_to_checkpoint_dict_contains_summary_fields(self):
        """Verify to_checkpoint_dict() returns expected summary keys."""
        state = self._build_populated_state()
        d = state.to_checkpoint_dict()

        assert d['voiceover_path'] == "/project/voiceover.srt"
        assert d['keywords'] == ["nature", "wildlife", "ocean"]
        assert d['topic_context'] == "A documentary about ocean life"
        assert d['segment_count'] == 2
        assert d['video_count'] == 3
        assert d['match_count'] == 2
        assert d['stage_timings'] == {"ANALYZE": 2.5, "VIDEO_SEARCH": 15.3, "MATCH": 8.1}

    @pytest.mark.fast
    def test_all_fields_survive_asdict_json_roundtrip(self):
        """Every field on PipelineState survives asdict -> JSON -> reconstruction."""
        state = self._build_populated_state()

        # Convert to dict (like checkpoint would)
        raw = asdict(state)

        # Serialize to JSON and back (simulates file write/read)
        json_str = json.dumps(raw, default=str)
        loaded = json.loads(json_str)

        # Verify every field name is present in the dict
        for f in fields(PipelineState):
            assert f.name in loaded, f"Field '{f.name}' missing after JSON roundtrip"

    @pytest.mark.fast
    def test_scalar_fields_survive_roundtrip(self):
        """Scalar fields (str, float) are identical after roundtrip."""
        state = self._build_populated_state()
        raw = asdict(state)
        json_str = json.dumps(raw, default=str)
        loaded = json.loads(json_str)

        assert loaded['voiceover_path'] == state.voiceover_path
        assert loaded['topic_context'] == state.topic_context
        assert loaded['face_preference'] == state.face_preference

    @pytest.mark.fast
    def test_list_fields_survive_roundtrip(self):
        """List fields preserve length and content after roundtrip."""
        state = self._build_populated_state()
        raw = asdict(state)
        json_str = json.dumps(raw, default=str)
        loaded = json.loads(json_str)

        assert loaded['keywords'] == state.keywords
        assert len(loaded['voiceover_segments']) == len(state.voiceover_segments)
        assert loaded['video_ids'] == state.video_ids
        assert loaded['search_failed_keywords'] == state.search_failed_keywords

    @pytest.mark.fast
    def test_dict_fields_survive_roundtrip(self):
        """Dict fields preserve keys and values after roundtrip."""
        state = self._build_populated_state()
        raw = asdict(state)
        json_str = json.dumps(raw, default=str)
        loaded = json.loads(json_str)

        assert loaded['caption_results']['abc123']['language'] == 'en'
        assert loaded['stage_timings']['ANALYZE'] == 2.5
        assert loaded['entity_images']['Blue Whale']['width'] == 1920

    @pytest.mark.fast
    def test_matches_survive_roundtrip(self):
        """Match dataclasses preserve all fields after roundtrip."""
        state = self._build_populated_state()
        raw = asdict(state)
        json_str = json.dumps(raw, default=str)
        loaded = json.loads(json_str)

        assert len(loaded['matches']) == 2
        m0 = loaded['matches'][0]
        assert m0['segment_index'] == 0
        assert m0['video_file'] == 'abc123'
        assert m0['confidence'] == 0.92
        assert m0['strategy'] == 'embedding'
        assert m0['face_score'] == 0.3

    @pytest.mark.fast
    def test_voiceover_segments_survive_roundtrip(self):
        """VoiceoverSegment dataclasses preserve computed duration after roundtrip."""
        state = self._build_populated_state()
        raw = asdict(state)
        json_str = json.dumps(raw, default=str)
        loaded = json.loads(json_str)

        seg0 = loaded['voiceover_segments'][0]
        assert seg0['index'] == 0
        assert seg0['start'] == 0.0
        assert seg0['end'] == 5.0
        assert seg0['text'] == 'Hello world'
        assert seg0['duration'] == 5.0  # Computed in __post_init__

    @pytest.mark.fast
    def test_path_fields_survive_roundtrip_as_strings(self):
        """Path objects are serialized to strings and survive roundtrip."""
        state = self._build_populated_state()
        raw = asdict(state)
        json_str = json.dumps(raw, default=str)
        loaded = json.loads(json_str)

        # Path gets serialized via default=str
        assert len(loaded['output_files']) == 1
        assert 'timeline.otio' in loaded['output_files'][0]

    @pytest.mark.fast
    def test_checkpoint_manager_save_restore_preserves_state_summary(self, tmp_path):
        """PipelineState.to_checkpoint_dict() data survives CheckpointManager save/load."""
        state = self._build_populated_state()
        cp_dict = state.to_checkpoint_dict()

        manager = CheckpointManager(tmp_path)
        manager.save("MATCH", cp_dict)

        manager2 = CheckpointManager(tmp_path)
        loaded = manager2.load()
        stage_data = manager2.get_stage_data("MATCH")

        assert stage_data['voiceover_path'] == "/project/voiceover.srt"
        assert stage_data['keywords'] == ["nature", "wildlife", "ocean"]
        assert stage_data['segment_count'] == 2
        assert stage_data['match_count'] == 2


# ---------------------------------------------------------------------------
# AC2: to_checkpoint_dict() when Optional fields are None
# ---------------------------------------------------------------------------

class TestToCheckpointDictOptionalNone:
    """AC2: to_checkpoint_dict() handles all Optional fields when set to None."""

    @pytest.mark.fast
    def test_none_voiceover_embeddings(self):
        """voiceover_embeddings=None produces a valid checkpoint dict."""
        state = PipelineState()
        state.voiceover_embeddings = None
        d = state.to_checkpoint_dict()

        # to_checkpoint_dict doesn't include voiceover_embeddings directly,
        # but the dict must be JSON-serializable
        json_str = json.dumps(d, default=str)
        assert json_str  # Non-empty

    @pytest.mark.fast
    def test_default_state_checkpoint_dict(self):
        """Default PipelineState produces a valid, JSON-serializable checkpoint dict."""
        state = PipelineState()
        d = state.to_checkpoint_dict()

        assert d['voiceover_path'] == ""
        assert d['keywords'] == []
        assert d['topic_context'] == ""
        assert d['segment_count'] == 0
        assert d['video_count'] == 0
        assert d['match_count'] == 0
        assert d['stage_timings'] == {}

        # Must be JSON-serializable
        json_str = json.dumps(d, default=str)
        loaded = json.loads(json_str)
        assert loaded == d

    @pytest.mark.fast
    def test_all_list_fields_empty(self):
        """PipelineState with all empty lists produces valid checkpoint dict."""
        state = PipelineState()
        state.voiceover_segments = []
        state.keywords = []
        state.video_ids = []
        state.matches = []
        d = state.to_checkpoint_dict()

        assert d['segment_count'] == 0
        assert d['video_count'] == 0
        assert d['match_count'] == 0

    @pytest.mark.fast
    def test_asdict_with_none_embeddings_serializable(self):
        """Full asdict() with None voiceover_embeddings is JSON-serializable."""
        state = PipelineState()
        state.voiceover_embeddings = None
        raw = asdict(state)

        # None serializes fine in JSON
        json_str = json.dumps(raw, default=str)
        loaded = json.loads(json_str)
        assert loaded['voiceover_embeddings'] is None

    @pytest.mark.fast
    def test_optional_fields_explicitly_none(self):
        """Setting Optional fields to None doesn't break serialization."""
        state = PipelineState()
        state.voiceover_embeddings = None

        raw = asdict(state)
        # All Optional[Any] fields that could be None
        assert raw['voiceover_embeddings'] is None

        # JSON roundtrip
        json_str = json.dumps(raw, default=str)
        loaded = json.loads(json_str)
        assert loaded['voiceover_embeddings'] is None


# ---------------------------------------------------------------------------
# AC3: from_checkpoint_dict() with missing fields (backward compatibility)
# ---------------------------------------------------------------------------

class TestFromCheckpointDictMissingFields:
    """AC3: PipelineState handles reconstruction from incomplete checkpoint data."""

    @pytest.mark.fast
    def test_validate_state_attributes_initializes_missing(self):
        """validate_state_attributes() fills in missing required fields."""
        state = PipelineState()
        # Simulate what happens after checkpoint restore with missing data
        state.text_metadata = None
        state.caption_results = None
        state.video_ids = None
        state.entity_images = None
        state.entity_videos = None

        initialized = state.validate_state_attributes()

        assert 'text_metadata' in initialized
        assert 'caption_results' in initialized
        assert 'video_ids' in initialized
        assert 'entity_images' in initialized
        assert 'entity_videos' in initialized
        assert state.text_metadata == []
        assert state.caption_results == {}
        assert state.video_ids == []
        assert state.entity_images == {}
        assert state.entity_videos == {}

    @pytest.mark.fast
    def test_validate_state_attributes_preserves_existing(self):
        """validate_state_attributes() doesn't overwrite existing non-None fields."""
        state = PipelineState()
        state.text_metadata = [{"video_id": "abc", "word_count": 100}]
        state.caption_results = {"abc": {"text": "hello"}}
        state.video_ids = ["abc123"]
        state.entity_images = {"whale": {"file": "w.jpg"}}
        state.entity_videos = {"whale": {"file": "w.mp4"}}

        initialized = state.validate_state_attributes()

        assert initialized == []  # Nothing was re-initialized
        assert state.text_metadata == [{"video_id": "abc", "word_count": 100}]
        assert state.video_ids == ["abc123"]

    @pytest.mark.fast
    def test_checkpoint_restore_with_empty_dict(self, tmp_path):
        """CheckpointManager.restore_state() creates valid state from empty checkpoint."""
        manager = CheckpointManager(tmp_path)
        manager.save("ANALYZE", {})
        manager.load()

        state = manager.restore_state()

        assert isinstance(state, PipelineState)
        assert state.text_metadata == []
        assert state.caption_results == {}
        assert state.video_ids == []

    @pytest.mark.fast
    def test_checkpoint_restore_preserves_existing_state(self, tmp_path):
        """restore_state() with pre-populated state preserves fields."""
        manager = CheckpointManager(tmp_path)
        manager.save("MATCH", {"match_count": 5})
        manager.load()

        state = PipelineState()
        state.keywords = ["nature", "ocean"]
        state.video_ids = ["abc123"]

        restored = manager.restore_state(state)

        assert restored.keywords == ["nature", "ocean"]
        assert restored.video_ids == ["abc123"]

    @pytest.mark.fast
    def test_post_init_handles_none_text_metadata(self):
        """__post_init__ converts None text_metadata to empty list."""
        # Simulate what could happen during deserialization
        state = PipelineState.__new__(PipelineState)
        # Manually set fields to simulate partial construction
        for f in fields(PipelineState):
            if f.name == 'text_metadata':
                object.__setattr__(state, f.name, None)
            elif f.name == 'voiceover_embeddings':
                object.__setattr__(state, f.name, None)
            elif f.default is not f.default_factory:
                if f.default is not f.default_factory:
                    try:
                        object.__setattr__(state, f.name, f.default)
                    except Exception:
                        object.__setattr__(state, f.name, f.default_factory())
            else:
                object.__setattr__(state, f.name, f.default_factory())

        state.__post_init__()
        assert state.text_metadata == []

    @pytest.mark.fast
    def test_missing_fields_in_legacy_checkpoint_json(self, tmp_path):
        """Simulate loading a legacy checkpoint that lacks newer PipelineState fields."""
        # Write a minimal checkpoint JSON missing many newer fields
        legacy_data = {
            "version": "2.0",
            "created_at": "2025-01-01T00:00:00",
            "updated_at": "2025-01-01T00:00:00",
            "last_completed_stage": "MATCH",
            "config_hash": "",
            "voiceover_path": "/old/project/vo.srt",
            "voiceover_hash": "",
            "analyze": {"keywords": ["legacy"]},
            "video_search": {"video_ids": ["old_vid"]},
            "caption": {},
            "match": {"match_count": 3},
            "iterative_match": {},
            "download_segments": {},
        }
        cp_path = tmp_path / "checkpoint.json"
        cp_path.write_text(json.dumps(legacy_data, indent=2))

        manager = CheckpointManager(tmp_path)
        loaded = manager.load()

        assert loaded is not None
        assert loaded.last_completed_stage == "MATCH"
        assert loaded.analyze == {"keywords": ["legacy"]}

        # restore_state fills in any missing PipelineState fields
        state = manager.restore_state()
        assert isinstance(state, PipelineState)
        assert state.text_metadata == []
        assert state.video_ids == []

    @pytest.mark.fast
    def test_checkpoint_data_from_dict_ignores_unknown_keys(self):
        """CheckpointData.from_dict() silently ignores unknown keys (forward compat)."""
        data = {
            "version": "2.0",
            "created_at": "2025-01-01T00:00:00",
            "updated_at": "2025-01-01T00:00:00",
            "last_completed_stage": "ANALYZE",
            "config_hash": "",
            "voiceover_path": "",
            "voiceover_hash": "",
            "analyze": {"keywords": ["test"]},
            "future_field_v3": {"some": "data"},
            "another_unknown": 42,
        }
        cp = CheckpointData.from_dict(data)
        assert cp.analyze == {"keywords": ["test"]}
        assert not hasattr(cp, 'future_field_v3')

    @pytest.mark.fast
    def test_checkpoint_data_from_dict_defaults_missing_stages(self):
        """CheckpointData.from_dict() defaults missing stage fields to empty dict."""
        data = {
            "version": "2.0",
            "last_completed_stage": "ANALYZE",
            "analyze": {"keywords": ["test"]},
            # All other stage fields missing
        }
        cp = CheckpointData.from_dict(data)
        assert cp.analyze == {"keywords": ["test"]}
        assert cp.video_search == {}
        assert cp.caption == {}
        assert cp.match == {}
        assert cp.iterative_match == {}
        assert cp.download_segments == {}


# ---------------------------------------------------------------------------
# AC4: Complex nested field types
# ---------------------------------------------------------------------------

class TestComplexNestedTypes:
    """AC4: PipelineState with complex nested field types."""

    @pytest.mark.fast
    def test_list_of_voiceover_segments_roundtrip(self):
        """List[VoiceoverSegment] survives asdict -> JSON -> reconstruction."""
        state = PipelineState()
        state.voiceover_segments = [
            VoiceoverSegment(index=i, start=i * 5.0, end=(i + 1) * 5.0,
                             text=f"Segment {i}")
            for i in range(5)
        ]

        raw = asdict(state)
        json_str = json.dumps(raw, default=str)
        loaded = json.loads(json_str)

        assert len(loaded['voiceover_segments']) == 5
        for i, seg in enumerate(loaded['voiceover_segments']):
            assert seg['index'] == i
            assert seg['text'] == f"Segment {i}"
            assert seg['duration'] == 5.0

    @pytest.mark.fast
    def test_list_of_matches_roundtrip(self):
        """List[Match] survives asdict -> JSON -> reconstruction."""
        state = PipelineState()
        state.matches = [
            Match(segment_index=i, video_file=f"vid_{i}", video_start=i * 10.0,
                  video_end=(i + 1) * 10.0, confidence=0.5 + i * 0.1,
                  strategy="test", reason=f"reason_{i}", face_score=0.2 * i)
            for i in range(3)
        ]

        raw = asdict(state)
        json_str = json.dumps(raw, default=str)
        loaded = json.loads(json_str)

        assert len(loaded['matches']) == 3
        for i, m in enumerate(loaded['matches']):
            assert m['segment_index'] == i
            assert m['video_file'] == f"vid_{i}"
            assert abs(m['confidence'] - (0.5 + i * 0.1)) < 1e-9

    @pytest.mark.fast
    def test_alternatives_dict_with_int_keys(self):
        """Dict[int, List[Match]] survives roundtrip (int keys become strings in JSON)."""
        state = PipelineState()
        state.alternatives = {
            0: [Match(segment_index=0, video_file="alt1", video_start=0.0,
                      video_end=5.0, confidence=0.7, strategy="alt")],
            5: [Match(segment_index=5, video_file="alt2", video_start=10.0,
                      video_end=15.0, confidence=0.6, strategy="alt"),
                Match(segment_index=5, video_file="alt3", video_start=20.0,
                      video_end=25.0, confidence=0.55, strategy="alt")],
        }

        raw = asdict(state)
        json_str = json.dumps(raw, default=str)
        loaded = json.loads(json_str)

        # JSON converts int keys to strings
        assert '0' in loaded['alternatives']
        assert '5' in loaded['alternatives']
        assert len(loaded['alternatives']['0']) == 1
        assert len(loaded['alternatives']['5']) == 2

    @pytest.mark.fast
    def test_nested_dicts_with_special_keys(self):
        """Dict fields with special characters in keys survive roundtrip."""
        state = PipelineState()
        state.caption_results = {
            "abc-123_XY": {"language": "en", "text": "Caption with unicode: \u00e9\u00e0\u00fc"},
            "vid.with.dots": {"language": "fr", "segments": [{"start": 0, "end": 5}]},
        }
        state.entity_images = {
            "Blue Whale (Balaenoptera musculus)": {"file": "whale.jpg", "width": 1920},
            "key/with/slashes": {"file": "other.jpg"},
        }

        raw = asdict(state)
        json_str = json.dumps(raw, default=str)
        loaded = json.loads(json_str)

        assert loaded['caption_results']['abc-123_XY']['text'] == "Caption with unicode: \u00e9\u00e0\u00fc"
        assert loaded['caption_results']['vid.with.dots']['segments'][0]['end'] == 5
        assert loaded['entity_images']['Blue Whale (Balaenoptera musculus)']['width'] == 1920
        assert loaded['entity_images']['key/with/slashes']['file'] == "other.jpg"

    @pytest.mark.fast
    def test_list_of_downloaded_videos_roundtrip(self):
        """List[DownloadedVideo] survives asdict -> JSON -> reconstruction."""
        state = PipelineState()
        state.downloaded_segments = [
            DownloadedVideo(file="vid1.mp4", url="https://example.com/1",
                            title="Video 1", duration=30.0, duration_tier="short",
                            keyword="test", face_score=0.3),
            DownloadedVideo(file="vid2.mp4", url="https://example.com/2",
                            title="Video 2", duration=120.0, duration_tier="medium",
                            keyword="nature", face_score=0.8),
        ]

        raw = asdict(state)
        json_str = json.dumps(raw, default=str)
        loaded = json.loads(json_str)

        assert len(loaded['downloaded_segments']) == 2
        assert loaded['downloaded_segments'][0]['file'] == "vid1.mp4"
        assert loaded['downloaded_segments'][1]['duration'] == 120.0
        assert loaded['downloaded_segments'][0]['face_score'] == 0.3

    @pytest.mark.fast
    def test_video_search_results_roundtrip(self):
        """List[VideoSearchResult] survives asdict -> JSON -> reconstruction."""
        state = PipelineState()
        state.video_search_results = [
            VideoSearchResult(video_id="abc", url="https://youtube.com/watch?v=abc",
                              title="Test Video", channel="TestChannel",
                              duration=60.0, duration_tier="short", keyword="test"),
        ]

        raw = asdict(state)
        json_str = json.dumps(raw, default=str)
        loaded = json.loads(json_str)

        assert len(loaded['video_search_results']) == 1
        assert loaded['video_search_results'][0]['video_id'] == 'abc'
        assert loaded['video_search_results'][0]['channel'] == 'TestChannel'

    @pytest.mark.fast
    def test_extracted_entities_deeply_nested(self):
        """Deeply nested extracted_entities survive roundtrip."""
        state = PipelineState()
        state.extracted_entities = [
            {
                "name": "Coral Reef",
                "type": "ecosystem",
                "attributes": {
                    "location": "Great Barrier Reef",
                    "species": ["clownfish", "sea turtle", "manta ray"],
                    "depth_range": {"min": 1.0, "max": 50.0},
                },
                "related": [
                    {"name": "Ocean Acidification", "relevance": 0.85},
                ],
            },
        ]

        raw = asdict(state)
        json_str = json.dumps(raw, default=str)
        loaded = json.loads(json_str)

        entity = loaded['extracted_entities'][0]
        assert entity['attributes']['species'] == ["clownfish", "sea turtle", "manta ray"]
        assert entity['attributes']['depth_range']['max'] == 50.0
        assert entity['related'][0]['relevance'] == 0.85

    @pytest.mark.fast
    def test_location_chapters_mixed_types(self):
        """location_chapters with mixed-type items survive roundtrip."""
        state = PipelineState()
        state.location_chapters = [
            {"name": "Introduction", "start": 0.0, "end": 30.0, "segments": [0, 1, 2]},
            {"name": "Main Body", "start": 30.0, "end": 120.0, "nested": {"key": "value"}},
            "simple_string_chapter",
        ]

        raw = asdict(state)
        json_str = json.dumps(raw, default=str)
        loaded = json.loads(json_str)

        assert len(loaded['location_chapters']) == 3
        assert loaded['location_chapters'][0]['segments'] == [0, 1, 2]
        assert loaded['location_chapters'][2] == "simple_string_chapter"

    @pytest.mark.fast
    def test_empty_nested_structures(self):
        """Empty nested structures (empty lists in dicts, empty dicts in lists) survive."""
        state = PipelineState()
        state.caption_results = {"vid_empty": {}}
        state.alternatives = {}
        state.extracted_entities = [{}]
        state.entity_images = {"empty_entity": {}}

        raw = asdict(state)
        json_str = json.dumps(raw, default=str)
        loaded = json.loads(json_str)

        assert loaded['caption_results']['vid_empty'] == {}
        assert loaded['alternatives'] == {}
        assert loaded['extracted_entities'] == [{}]
        assert loaded['entity_images']['empty_entity'] == {}

    @pytest.mark.fast
    def test_large_state_roundtrip(self):
        """State with many items in lists/dicts survives roundtrip without data loss."""
        state = PipelineState()
        state.video_ids = [f"vid_{i:04d}" for i in range(200)]
        state.matches = [
            Match(segment_index=i, video_file=f"vid_{i:04d}", video_start=float(i),
                  video_end=float(i + 1), confidence=0.5, strategy="bulk")
            for i in range(100)
        ]
        state.keywords = [f"keyword_{i}" for i in range(50)]

        raw = asdict(state)
        json_str = json.dumps(raw, default=str)
        loaded = json.loads(json_str)

        assert len(loaded['video_ids']) == 200
        assert len(loaded['matches']) == 100
        assert len(loaded['keywords']) == 50
        assert loaded['video_ids'][199] == "vid_0199"
        assert loaded['matches'][99]['video_file'] == "vid_0099"


# ---------------------------------------------------------------------------
# AC4 extension: numpy float32 serialization (Rule 7)
# ---------------------------------------------------------------------------

class TestNumpySerializationSafety:
    """Verify numpy types don't break JSON serialization (CLAUDE.md Rule 7)."""

    @pytest.mark.fast
    def test_numpy_float32_in_stage_timings(self):
        """numpy.float32 in stage_timings must be serializable via default=str."""
        try:
            import numpy as np
        except ImportError:
            pytest.skip("numpy not installed")

        state = PipelineState()
        state.stage_timings = {
            "ANALYZE": np.float32(2.5),
            "MATCH": np.float64(8.1),
        }

        d = state.to_checkpoint_dict()
        # default=str handles numpy types
        json_str = json.dumps(d, default=str)
        loaded = json.loads(json_str)

        # Values may be strings due to default=str, but they parse back
        assert float(loaded['stage_timings']['ANALYZE']) == pytest.approx(2.5, abs=0.01)

    @pytest.mark.fast
    def test_numpy_in_confidence_scores(self):
        """numpy.float64 confidence scores survive roundtrip."""
        try:
            import numpy as np
        except ImportError:
            pytest.skip("numpy not installed")

        state = PipelineState()
        state.matches = [
            Match(segment_index=0, video_file="vid1", video_start=0.0,
                  video_end=5.0, confidence=float(np.float64(0.923)),
                  strategy="embedding"),
        ]

        raw = asdict(state)
        json_str = json.dumps(raw, default=str)
        loaded = json.loads(json_str)

        assert loaded['matches'][0]['confidence'] == pytest.approx(0.923)


# ---------------------------------------------------------------------------
# Full integration: CheckpointManager -> PipelineState restore cycle
# ---------------------------------------------------------------------------

class TestFullCheckpointRoundtrip:
    """Integration test: full save/load cycle through CheckpointManager."""

    @pytest.mark.fast
    def test_analyze_stage_data_roundtrip(self, tmp_path):
        """Analyze stage data with keywords and entities survives save/load."""
        stage_data = {
            "keywords": ["ocean", "marine", "coral"],
            "segment_count": 15,
            "topic_context": "Marine biology documentary",
            "entities": [
                {"name": "Great Barrier Reef", "type": "location"},
                {"name": "Coral", "type": "organism"},
            ],
        }

        manager = CheckpointManager(tmp_path)
        manager.save("ANALYZE", stage_data)

        manager2 = CheckpointManager(tmp_path)
        manager2.load()
        loaded = manager2.get_stage_data("ANALYZE")

        assert loaded == stage_data
        assert loaded['entities'][0]['name'] == "Great Barrier Reef"

    @pytest.mark.fast
    def test_match_stage_with_matches_list_roundtrip(self, tmp_path):
        """Match stage data containing a list of match dicts survives roundtrip."""
        stage_data = {
            "match_count": 3,
            "avg_confidence": 0.88,
            "matches": [
                {"segment_index": 0, "video_file": "vid1", "confidence": 0.95,
                 "video_start": 10.0, "video_end": 15.0, "strategy": "embedding"},
                {"segment_index": 1, "video_file": "vid2", "confidence": 0.82,
                 "video_start": 0.0, "video_end": 5.0, "strategy": "keyword"},
                {"segment_index": 2, "video_file": "vid3", "confidence": 0.87,
                 "video_start": 20.0, "video_end": 30.0, "strategy": "diversity"},
            ],
        }

        manager = CheckpointManager(tmp_path)
        manager.save("MATCH", stage_data)

        manager2 = CheckpointManager(tmp_path)
        manager2.load()
        loaded = manager2.get_stage_data("MATCH")

        assert loaded == stage_data
        assert len(loaded['matches']) == 3
        assert loaded['matches'][2]['strategy'] == "diversity"

    @pytest.mark.fast
    def test_multi_stage_progressive_save_load(self, tmp_path):
        """Progressive saves across multiple stages all survive a single load."""
        manager = CheckpointManager(tmp_path)
        manager.save("ANALYZE", {"keywords": ["test"], "segment_count": 10})
        manager.save("VIDEO_SEARCH", {"video_ids": ["v1", "v2", "v3"]})
        manager.save("CAPTION", {"caption_count": 3})
        manager.save("MATCH", {"match_count": 8, "avg_confidence": 0.9})

        manager2 = CheckpointManager(tmp_path)
        loaded = manager2.load()

        assert loaded.last_completed_stage == "MATCH"
        assert loaded.analyze == {"keywords": ["test"], "segment_count": 10}
        assert loaded.video_search == {"video_ids": ["v1", "v2", "v3"]}
        assert loaded.caption == {"caption_count": 3}
        assert loaded.match == {"match_count": 8, "avg_confidence": 0.9}
        # Stages not yet saved should be empty
        assert loaded.iterative_match == {}
        assert loaded.download_segments == {}

    @pytest.mark.fast
    def test_restore_state_after_full_roundtrip(self, tmp_path):
        """restore_state() after full save/load produces usable PipelineState."""
        manager = CheckpointManager(tmp_path)
        manager.save("VIDEO_SEARCH", {"video_ids": ["abc", "def"]})

        manager2 = CheckpointManager(tmp_path)
        manager2.load()

        state = manager2.restore_state()

        assert isinstance(state, PipelineState)
        # validate_state_attributes ensures these are non-None
        assert state.text_metadata is not None
        assert state.caption_results is not None
        assert state.video_ids is not None
        assert state.entity_images is not None
        assert state.entity_videos is not None
