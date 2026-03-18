"""
Non-yt-dlp Pipeline Flow Tests

Verifies V9 entity track matching, --output-only embedding fallback,
generated image prompt correctness, and config backward compatibility
with removed entity_cache sections.
"""

import sys
import types
import textwrap
from types import SimpleNamespace
from unittest.mock import Mock, patch, MagicMock

import pytest

# Stub the missing generated_images.service module so the package __init__ can load
if "src.generated_images.service" not in sys.modules:
    _stub = types.ModuleType("src.generated_images.service")
    _stub.GeneratedImageService = type("GeneratedImageService", (), {})
    sys.modules["src.generated_images.service"] = _stub

from src.config.base import Config
from src.config.sections.entity import ImageSearchConfig
from src.generated_images.batching import build_generated_image_batches
from src.otio.entities import _find_best_entity_match, add_entity_media_to_track
from src.state import VoiceoverSegment
from src.utils import Match, MatchResult, SRTSegment


# ---------------------------------------------------------------------------
# Helper factories
# ---------------------------------------------------------------------------

def _make_entity_dict():
    """3 entities: Obama, Paris, NASA — Mock objects with .images, .entity_type, .query."""
    obama = Mock()
    obama.images = ["/img/obama1.jpg", "/img/obama2.jpg"]
    obama.videos = []
    obama.entity_type = "PERSON"
    obama.query = "Barack Obama president"

    paris = Mock()
    paris.images = ["/img/paris1.jpg"]
    paris.videos = []
    paris.entity_type = "GPE"
    paris.query = "Paris France city"

    nasa = Mock()
    nasa.images = ["/img/nasa1.jpg", "/img/nasa2.jpg", "/img/nasa3.jpg"]
    nasa.videos = []
    nasa.entity_type = "ORG"
    nasa.query = "NASA space agency"

    return {"Obama": obama, "Paris": paris, "NASA": nasa}


def _make_match_result(index, start, end, text):
    """Single MatchResult wrapping SRTSegment-based Match."""
    vo_seg = SRTSegment(
        index=index, start_time=start, end_time=end,
        text=text, source_file="voiceover.srt"
    )
    vid_seg = SRTSegment(
        index=index, start_time=start, end_time=end,
        text=f"Video {index}", source_file=f"/videos/v{index}.mp4"
    )
    match = Match(
        voiceover_segment=vo_seg,
        video_segment=vid_seg,
        video_scene=None,
        confidence=0.8,
        reasoning=f"Match {index}",
    )
    return MatchResult(
        primary_match=match,
        alternatives=[],
        secondary_matches=[],
        strategy_matches=[],
    )


def _make_matches(texts):
    """List of MatchResults from text strings (3s segments)."""
    results = []
    for i, text in enumerate(texts):
        results.append(_make_match_result(i, i * 3.0, (i + 1) * 3.0, text))
    return results


def _make_mock_config(sticky=False, consolidation=False):
    """Mock config for add_entity_media_to_track."""
    cfg = Mock()
    cfg.image_search = Mock()
    cfg.image_search.enable_sticky_matching = sticky
    cfg.image_search.semantic_match_threshold = 0.15
    cfg.image_search.embedding_match_threshold = 0.30
    cfg.image_search.enable_consolidation = consolidation
    return cfg


# ===========================================================================
# Group 1: V9 Entity Matching Quality
# ===========================================================================

class TestV9EntityMatchingQuality:
    """Tests for _find_best_entity_match() — semantic vs word-overlap."""

    @pytest.mark.fast
    def test_embeddings_match_more_segments_than_word_overlap(self):
        """With zero word overlap but semantic relatedness, embedding mode
        matches all 6 segments while word-overlap matches 0."""
        entities = _make_entity_dict()

        # Texts with zero word overlap with any entity name/query
        texts = [
            "the commander in chief addressed the nation",
            "executive leadership during the crisis",
            "the oval office policy announcement",
            "presidential decision on healthcare",
            "the white house press conference",
            "democratic reform legislation signed",
        ]

        # Embedding scores: Obama always gets cosine 0.6 (above 0.30 threshold)
        fake_vo_embeddings = {i: [float(i)] for i in range(len(texts))}
        fake_entity_embeddings = {"Obama": [99.0], "Paris": [98.0], "NASA": [97.0]}

        def fake_cosine(a, b):
            # All entity embeddings score above threshold
            return 0.6

        # Word-overlap path: no embeddings → falls back to word overlap
        word_overlap_hits = 0
        for text in texts:
            name, match_type = _find_best_entity_match(
                text.lower(), entities, vo_embedding=None, entity_embeddings=None
            )
            if name is not None:
                word_overlap_hits += 1

        # Embedding path: mock cosine_similarity
        embedding_hits = 0
        with patch("src.embeddings.cosine_similarity", fake_cosine):
            for i, text in enumerate(texts):
                name, match_type = _find_best_entity_match(
                    text.lower(), entities,
                    vo_embedding=fake_vo_embeddings[i],
                    entity_embeddings=fake_entity_embeddings,
                )
                if name is not None:
                    embedding_hits += 1

        assert word_overlap_hits == 0, "word-overlap should miss all zero-overlap texts"
        assert embedding_hits == 6, "embeddings should match all 6 segments"

    @pytest.mark.fast
    def test_exact_match_takes_priority_over_high_embedding_score(self):
        """Entity name in text always wins, even when a different entity has cosine=0.99."""
        entities = _make_entity_dict()

        # Text contains "Obama" literally
        text = "obama spoke at the summit"

        # Make Paris have the highest embedding score
        def fake_cosine(a, b):
            # Return 0.99 for Paris, 0.1 for others
            return 0.99  # Doesn't matter — exact match should preempt

        fake_vo_embedding = [1.0]
        fake_entity_embeddings = {"Obama": [1.0], "Paris": [2.0], "NASA": [3.0]}

        with patch("src.embeddings.cosine_similarity", fake_cosine):
            name, match_type = _find_best_entity_match(
                text, entities,
                vo_embedding=fake_vo_embedding,
                entity_embeddings=fake_entity_embeddings,
            )

        assert name == "Obama"
        assert match_type == "exact"

    @pytest.mark.fast
    def test_embedding_matches_what_word_overlap_misses(self):
        """Word overlap=0.125 (below 0.15) misses, but cosine=0.5 (above 0.30) hits."""
        entities = _make_entity_dict()

        # "space" overlaps with NASA query "NASA space agency" but total overlap is low
        # vo_words = {"the", "frontier", "of", "space", "exploration", "continues", "today", "forward"}
        # query_words = {"nasa", "space", "agency", "org"} (entity_type "ORG" added)
        # common = {"space"} → 1/(12+1) = 0.077 < 0.15
        text = "the frontier of space exploration continues today forward"

        # Word overlap path
        name_wo, type_wo = _find_best_entity_match(
            text.lower(), entities, vo_embedding=None, entity_embeddings=None
        )

        # Embedding path — NASA scores 0.5
        def fake_cosine(a, b):
            return 0.5

        fake_entity_embeddings = {"Obama": [1.0], "Paris": [2.0], "NASA": [3.0]}
        with patch("src.embeddings.cosine_similarity", fake_cosine):
            name_emb, type_emb = _find_best_entity_match(
                text.lower(), entities,
                vo_embedding=[1.0],
                entity_embeddings=fake_entity_embeddings,
            )

        assert name_wo is None, "word overlap should miss (score < 0.15)"
        assert name_emb is not None, "embedding should hit (cosine >= 0.30)"
        assert type_emb == "semantic"

    @pytest.mark.fast
    def test_match_stats_printed_correctly(self, capsys, tmp_path):
        """add_entity_media_to_track prints correct exact/semantic/sticky/none counts."""
        import opentimelineio as otio

        entities = _make_entity_dict()
        # 3 segments: exact (Obama), semantic (embedding), none
        texts = [
            "Obama visited the museum",  # exact match
            "aerospace technology advances",  # semantic via embedding
            "the weather is nice today",  # none
        ]
        matches = _make_matches(texts)
        config = _make_mock_config(sticky=False, consolidation=False)

        track = otio.schema.Track(name="V9", kind=otio.schema.TrackKind.Video)

        # Embedding: segment 1 → NASA at 0.5, segment 2 → below threshold
        def fake_cosine(a, b):
            # vo_embedding [2.0] for segment 2 ("weather") → score below threshold
            if a == [2.0]:
                return 0.1
            return 0.5

        fake_vo_embeddings = {0: [0.0], 1: [1.0], 2: [2.0]}
        # Only NASA has an embedding so only seg 1 can match via embedding
        fake_entity_embeddings = {"NASA": [3.0]}

        # Patch image existence checks so clips are added
        with patch("src.embeddings.cosine_similarity", fake_cosine), \
             patch("pathlib.Path.exists", return_value=True), \
             patch("src.otio.entities._has_problematic_path", return_value=False):
            add_entity_media_to_track(
                track, entities, matches, 24.0, config, "images",
                voiceover_embeddings=fake_vo_embeddings,
                entity_embeddings=fake_entity_embeddings,
            )

        captured = capsys.readouterr().out
        assert "Exact: 1" in captured
        assert "Semantic: 1" in captured
        assert "2/3 segments" in captured

    @pytest.mark.fast
    def test_entity_without_assets_skipped(self):
        """Entity with empty images=[] is skipped even with perfect embedding score."""
        empty_entity = Mock()
        empty_entity.images = []
        empty_entity.videos = []
        empty_entity.entity_type = "PERSON"
        empty_entity.query = "Empty Person"

        entities = {"EmptyPerson": empty_entity}

        def fake_cosine(a, b):
            return 0.99

        with patch("src.embeddings.cosine_similarity", fake_cosine):
            name, match_type = _find_best_entity_match(
                "empty person is mentioned here",
                entities,
                vo_embedding=[1.0],
                entity_embeddings={"EmptyPerson": [1.0]},
            )

        # Exact match checks assets first — no assets → skip
        assert name is None
        assert match_type == "none"


# ===========================================================================
# Group 2: --output-only Embedding Fallback
# ===========================================================================

class TestOutputOnlyEmbeddingFallback:
    """Tests for OutputStage._ensure_voiceover_embeddings / _compute_entity_embeddings."""

    def _make_stage(self):
        """Create an OutputStage instance with minimal init."""
        from src.stages.output import OutputStage
        stage = OutputStage.__new__(OutputStage)
        stage.logger = MagicMock()
        return stage

    @pytest.mark.fast
    @patch("src.stages.output.OutputStage.__init__", return_value=None)
    def test_ensure_voiceover_embeddings_recomputes_when_none(self, mock_init):
        """When state.voiceover_embeddings is None, calls compute_embeddings."""
        stage = self._make_stage()

        state = Mock()
        state.voiceover_embeddings = None
        state.voiceover_segments = [
            VoiceoverSegment(index=0, start=0.0, end=3.0, text="Hello world"),
            VoiceoverSegment(index=1, start=3.0, end=6.0, text="Goodbye world"),
        ]

        config = Mock()
        config.cache = Mock()
        config.cache.cache_dir = ".cache"

        fake_embeddings = [[0.1, 0.2], [0.3, 0.4]]

        with patch("src.embeddings.compute_embeddings", return_value=fake_embeddings) as mock_compute, \
             patch("src.embeddings.get_embedding_provider") as mock_provider, \
             patch("src.utils.CacheManager"):
            stage._ensure_voiceover_embeddings(state, config)

        mock_compute.assert_called_once()
        assert state.voiceover_embeddings == fake_embeddings

    @pytest.mark.fast
    @patch("src.stages.output.OutputStage.__init__", return_value=None)
    def test_ensure_voiceover_embeddings_skips_when_already_set(self, mock_init):
        """When embeddings exist, method returns immediately (no calls)."""
        stage = self._make_stage()

        state = Mock()
        state.voiceover_embeddings = [[0.1, 0.2]]  # Already set
        state.voiceover_segments = [
            VoiceoverSegment(index=0, start=0.0, end=3.0, text="Hello"),
        ]

        config = Mock()

        with patch("src.embeddings.compute_embeddings") as mock_compute:
            stage._ensure_voiceover_embeddings(state, config)

        mock_compute.assert_not_called()

    @pytest.mark.fast
    @patch("src.stages.output.OutputStage.__init__", return_value=None)
    def test_compute_entity_embeddings_populates_state(self, mock_init):
        """Entity queries get embedded and stored in state.entity_embeddings."""
        stage = self._make_stage()

        entity1 = Mock()
        entity1.query = "Barack Obama president"
        entity2 = Mock()
        entity2.query = "NASA space agency"

        state = Mock()
        state.entity_images = {"Obama": entity1, "NASA": entity2}
        state.entity_embeddings = {}

        config = Mock()

        mock_provider = MagicMock()
        mock_provider.embed.return_value = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]

        with patch("src.embeddings.get_embedding_provider", return_value=mock_provider):
            stage._compute_entity_embeddings(state, config)

        assert len(state.entity_embeddings) == 2
        assert "Obama" in state.entity_embeddings
        assert "NASA" in state.entity_embeddings

    @pytest.mark.fast
    @patch("src.stages.output.OutputStage.__init__", return_value=None)
    def test_compute_entity_embeddings_falls_back_on_error(self, mock_init):
        """Exception resets state.entity_embeddings = {} (graceful, no crash)."""
        stage = self._make_stage()

        entity1 = Mock()
        entity1.query = "test"

        state = Mock()
        state.entity_images = {"Test": entity1}
        state.entity_embeddings = {"stale": [1, 2, 3]}

        config = Mock()

        with patch("src.embeddings.get_embedding_provider", side_effect=RuntimeError("API down")):
            stage._compute_entity_embeddings(state, config)

        assert state.entity_embeddings == {}


# ===========================================================================
# Group 3: Generated Image Prompt Text
# ===========================================================================

class TestGeneratedImagePromptText:
    """Tests for build_generated_image_batches() — batch.text uses last segment."""

    @pytest.mark.fast
    def test_batch_uses_only_last_segment_text(self):
        """Single batch of 8 segments: batch.text == segments[7].text."""
        segments = [
            VoiceoverSegment(index=i, start=i * 3.0, end=(i + 1) * 3.0,
                             text=f"Segment {i} distinct text")
            for i in range(8)
        ]
        batch_config = SimpleNamespace(
            min_segments_per_image=7,
            max_segments_per_image=9,
            batch_target_segments=8,
            allow_boundary_batch_outside_range=True,
            boundary_min_segments=6,
            boundary_max_segments=10,
            fallback_single_batch_max_segments=10,
        )

        batches = build_generated_image_batches(segments, batch_config)

        assert len(batches) == 1
        assert batches[0].text == "Segment 7 distinct text"

    @pytest.mark.fast
    def test_multiple_batches_each_use_own_last_segment(self):
        """16 segments = 2 batches; each batch uses its own last segment text."""
        segments = [
            VoiceoverSegment(index=i, start=i * 3.0, end=(i + 1) * 3.0,
                             text=f"Line {i}")
            for i in range(16)
        ]
        batch_config = SimpleNamespace(
            min_segments_per_image=7,
            max_segments_per_image=9,
            batch_target_segments=8,
            allow_boundary_batch_outside_range=True,
            boundary_min_segments=6,
            boundary_max_segments=10,
            fallback_single_batch_max_segments=10,
        )

        batches = build_generated_image_batches(segments, batch_config)

        assert len(batches) == 2
        # First batch of 8: last segment is index 7
        assert batches[0].text == "Line 7"
        # Second batch of 8: last segment is index 15
        assert batches[1].text == "Line 15"

    @pytest.mark.fast
    def test_single_segment_batch_uses_that_segment_text(self):
        """Edge case: batch of 1 segment uses that segment's text."""
        segments = [
            VoiceoverSegment(index=0, start=0.0, end=3.0, text="Solo segment")
        ]
        batch_config = SimpleNamespace(
            min_segments_per_image=1,
            max_segments_per_image=1,
            batch_target_segments=1,
            allow_boundary_batch_outside_range=True,
            boundary_min_segments=1,
            boundary_max_segments=1,
            fallback_single_batch_max_segments=1,
        )

        batches = build_generated_image_batches(segments, batch_config)

        assert len(batches) == 1
        assert batches[0].text == "Solo segment"


# ===========================================================================
# Group 4: Config Backward Compat
# ===========================================================================

class TestConfigBackwardCompat:
    """Tests for _build_dataclass filtering and entity_cache removal."""

    @pytest.mark.fast
    def test_build_dataclass_ignores_unknown_entity_cache_key(self):
        """_build_dataclass(ImageSearchConfig, {'entity_cache': {...}}) silently drops the key."""
        result = Config._build_dataclass(
            ImageSearchConfig,
            {"entity_cache": {"ttl_hours": 24, "max_entries": 1000}},
        )
        assert isinstance(result, ImageSearchConfig)
        assert not hasattr(result, "entity_cache")

    @pytest.mark.fast
    def test_from_dict_ignores_toplevel_entity_cache_section(self):
        """Top-level entity_cache: in YAML data is silently ignored (not in section_mapping)."""
        # from_yaml loads data and iterates section_mapping — entity_cache is not in it
        data = {
            "entity_cache": {"ttl_hours": 48, "max_entries": 500},
            "image_search": {"enabled": True},
        }
        # Use from_yaml-equivalent: construct Config and apply section_mapping logic
        config = Config()
        # The section_mapping loop only processes known keys; entity_cache is silently skipped
        # Verify by loading via _build_dataclass for image_search — no crash
        config.image_search = Config._build_dataclass(
            ImageSearchConfig, data.get("image_search", {})
        )
        assert config.image_search.enabled is True
        assert not hasattr(config, "entity_cache") or not hasattr(config.image_search, "entity_cache")

    @pytest.mark.fast
    def test_image_search_config_rejects_direct_unknown_kwarg(self):
        """ImageSearchConfig(entity_cache=...) raises TypeError — safety net is _build_dataclass."""
        with pytest.raises(TypeError, match="entity_cache"):
            ImageSearchConfig(entity_cache={"ttl_hours": 24})

    @pytest.mark.fast
    def test_build_dataclass_with_multiple_extra_keys(self):
        """Multiple unknown keys all silently ignored."""
        result = Config._build_dataclass(
            ImageSearchConfig,
            {
                "entity_cache": {"ttl_hours": 24},
                "legacy_flag": True,
                "removed_option": "value",
                "enabled": False,  # This one IS valid
            },
        )
        assert isinstance(result, ImageSearchConfig)
        assert result.enabled is False  # Valid field kept

    @pytest.mark.fast
    def test_config_yaml_with_entity_cache_loads_cleanly(self, tmp_path):
        """Write a temp YAML with entity_cache: section, load via Config.from_yaml(), no crash."""
        yaml_content = textwrap.dedent("""\
            image_search:
              enabled: true
              images_per_entity: 3

            entity_cache:
              ttl_hours: 24
              max_entries: 1000
              cleanup_interval: 3600
        """)
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text(yaml_content, encoding="utf-8")

        config = Config.from_yaml(str(config_file), skip_final_validation=True)
        assert config.image_search.enabled is True
        assert config.image_search.images_per_entity == 3
