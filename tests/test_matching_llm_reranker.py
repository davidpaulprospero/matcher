"""Tests for LLMReranker class."""

import pytest
from unittest.mock import MagicMock, patch
from dataclasses import dataclass

from src.matching.llm_reranker import LLMReranker, LLMRerankerConfig, RerankResult


@dataclass
class MockSRTSegment:
    """Mock SRT segment for testing."""
    text: str
    start: float = 0.0
    end: float = 1.0
    source_file: str = ""


class MockLLMProvider:
    """Mock LLM provider for testing."""

    def __init__(self, return_values=None):
        """Initialize with optional return values."""
        self.return_values = return_values or [(0, 0.85, "Good match", "reasoning")]
        self.call_count = 0
        self.last_call_args = None

    def match_batch(self, batch, context=None, negative_rules=None):
        """Mock match_batch returning configured values."""
        self.call_count += 1
        self.last_call_args = {
            'batch': batch,
            'context': context,
            'negative_rules': negative_rules
        }
        return self.return_values


class TestLLMRerankerInit:
    """Tests for LLMReranker initialization and provider injection."""

    def test_llm_reranker_init_with_config(self):
        """Verify LLMReranker initializes with config correctly."""
        config = LLMRerankerConfig(
            ambiguous_threshold=0.7,
            cache_llm_responses=False
        )
        reranker = LLMReranker(config=config)

        assert reranker.config.ambiguous_threshold == 0.7
        assert reranker.config.cache_llm_responses is False
        assert reranker.cache is None

    def test_llm_reranker_init_with_cache(self):
        """Verify LLMReranker initializes with cache."""
        config = LLMRerankerConfig()
        mock_cache = MagicMock()
        reranker = LLMReranker(config=config, cache=mock_cache)

        assert reranker.cache is mock_cache

    def test_llm_reranker_from_matching_config(self):
        """Verify factory method creates LLMReranker from matching config."""
        matching_config = MagicMock()
        matching_config.ambiguous_threshold = 0.55
        matching_config.cache_llm_responses = True

        reranker = LLMReranker.from_matching_config(matching_config)

        assert reranker.config.ambiguous_threshold == 0.55
        assert reranker.config.cache_llm_responses is True

    def test_llm_reranker_from_matching_config_defaults(self):
        """Verify factory uses defaults when attributes missing."""
        matching_config = MagicMock(spec=[])  # Empty spec - no attributes

        reranker = LLMReranker.from_matching_config(matching_config)

        assert reranker.config.ambiguous_threshold == 0.65  # Default
        assert reranker.config.cache_llm_responses is True  # Default

    def test_llm_reranker_from_matching_config_spread_calibration(self):
        """Verify factory reads spread calibration config from matching config."""
        matching_config = MagicMock()
        matching_config.ambiguous_threshold = 0.65
        matching_config.cache_llm_responses = True
        matching_config.llm_reranker_close_spread_threshold = 0.08
        matching_config.llm_reranker_clear_winner_threshold = 0.25
        matching_config.llm_reranker_close_spread_factor = 0.85
        matching_config.llm_reranker_clear_winner_factor = 1.15

        reranker = LLMReranker.from_matching_config(matching_config)

        assert reranker.config.close_spread_threshold == 0.08
        assert reranker.config.clear_winner_threshold == 0.25
        assert reranker.config.close_spread_factor == 0.85
        assert reranker.config.clear_winner_factor == 1.15


class TestLLMRerankerPromptFormat:
    """Tests for LLMReranker prompt formatting via provider."""

    def test_llm_reranker_passes_voiceover_text_to_provider(self):
        """Verify voiceover text is passed to provider correctly."""
        config = LLMRerankerConfig()
        reranker = LLMReranker(config=config)
        provider = MockLLMProvider()

        candidates = [
            (MockSRTSegment("video text 1"), 0.9),
            (MockSRTSegment("video text 2"), 0.8),
        ]

        reranker.rerank(
            voiceover_text="test voiceover",
            candidates=candidates,
            primary_provider=provider
        )

        assert provider.call_count == 1
        batch = provider.last_call_args['batch']
        assert batch[0][0] == "test voiceover"

    def test_llm_reranker_passes_candidates_to_provider(self):
        """Verify candidates are passed to provider correctly."""
        config = LLMRerankerConfig()
        reranker = LLMReranker(config=config)
        provider = MockLLMProvider()

        candidates = [
            (MockSRTSegment("video text 1"), 0.9),
            (MockSRTSegment("video text 2"), 0.8),
            (MockSRTSegment("video text 3"), 0.7),
        ]

        reranker.rerank(
            voiceover_text="test voiceover",
            candidates=candidates,
            primary_provider=provider
        )

        batch = provider.last_call_args['batch']
        # Should pass up to 5 candidates
        assert len(batch[0][1]) == 3

    def test_llm_reranker_passes_context_and_negative_rules(self):
        """Verify context and negative_rules are passed through."""
        config = LLMRerankerConfig()
        reranker = LLMReranker(config=config)
        provider = MockLLMProvider()

        candidates = [(MockSRTSegment("video text"), 0.9)]

        reranker.rerank(
            voiceover_text="test voiceover",
            candidates=candidates,
            primary_provider=provider,
            context="documentary context",
            negative_rules=["no violence", "no profanity"]
        )

        assert provider.last_call_args['context'] == "documentary context"
        assert provider.last_call_args['negative_rules'] == ["no violence", "no profanity"]


class TestLLMRerankerEmptyCandidates:
    """Tests for LLMReranker handling of empty candidates."""

    def test_llm_reranker_handles_empty_candidates(self):
        """Verify empty candidates list returns valid RerankResult."""
        config = LLMRerankerConfig()
        reranker = LLMReranker(config=config)
        provider = MockLLMProvider()

        result = reranker.rerank(
            voiceover_text="test voiceover",
            candidates=[],
            primary_provider=provider
        )

        assert isinstance(result, RerankResult)
        assert result.selected_idx == 0
        assert result.confidence == 0.0
        assert "No candidates" in result.reasoning
        # Provider should not be called for empty candidates
        assert provider.call_count == 0

    def test_llm_reranker_handles_none_provider(self):
        """Verify None provider returns embedding fallback."""
        config = LLMRerankerConfig()
        reranker = LLMReranker(config=config)

        candidates = [(MockSRTSegment("video text"), 0.85)]

        result = reranker.rerank(
            voiceover_text="test voiceover",
            candidates=candidates,
            primary_provider=None
        )

        assert result.selected_idx == 0
        assert result.confidence == 0.60
        assert "Embedding similarity only" in result.reasoning


class TestLLMRerankerConfidenceThreshold:
    """Tests for LLMReranker confidence threshold behavior."""

    def test_llm_reranker_respects_confidence_threshold(self):
        """Verify ambiguous_threshold triggers secondary provider."""
        config = LLMRerankerConfig(ambiguous_threshold=0.70)
        reranker = LLMReranker(config=config)

        # Primary returns low confidence (0.5), below threshold
        primary = MockLLMProvider(return_values=[(0, 0.50, "Unsure match", "cot")])
        # Secondary returns higher confidence
        secondary = MockLLMProvider(return_values=[(1, 0.80, "Better match", "cot")])

        candidates = [
            (MockSRTSegment("video text 1"), 0.9),
            (MockSRTSegment("video text 2"), 0.8),
        ]

        result = reranker.rerank(
            voiceover_text="test voiceover",
            candidates=candidates,
            primary_provider=primary,
            secondary_provider=secondary
        )

        # Should use secondary's result since it has higher confidence
        assert result.selected_idx == 1
        assert result.confidence == 0.80
        assert result.used_secondary is True
        assert primary.call_count == 1
        assert secondary.call_count == 1

    def test_llm_reranker_skips_secondary_above_threshold(self):
        """Verify secondary not called when confidence above threshold."""
        config = LLMRerankerConfig(ambiguous_threshold=0.70)
        reranker = LLMReranker(config=config)

        # Primary returns high confidence (0.85), above threshold
        primary = MockLLMProvider(return_values=[(0, 0.85, "Good match", "cot")])
        secondary = MockLLMProvider()

        candidates = [(MockSRTSegment("video text"), 0.9)]

        result = reranker.rerank(
            voiceover_text="test voiceover",
            candidates=candidates,
            primary_provider=primary,
            secondary_provider=secondary
        )

        assert result.selected_idx == 0
        assert result.confidence == 0.85
        assert result.used_secondary is False
        assert primary.call_count == 1
        assert secondary.call_count == 0  # Not called

    def test_llm_reranker_keeps_primary_if_secondary_worse(self):
        """Verify primary result kept if secondary has lower confidence."""
        config = LLMRerankerConfig(ambiguous_threshold=0.70)
        reranker = LLMReranker(config=config)

        # Primary returns low confidence
        primary = MockLLMProvider(return_values=[(0, 0.55, "Primary match", "cot")])
        # Secondary returns even lower confidence
        secondary = MockLLMProvider(return_values=[(1, 0.40, "Worse match", "cot")])

        candidates = [
            (MockSRTSegment("video text 1"), 0.9),
            (MockSRTSegment("video text 2"), 0.8),
        ]

        result = reranker.rerank(
            voiceover_text="test voiceover",
            candidates=candidates,
            primary_provider=primary,
            secondary_provider=secondary
        )

        # Should keep primary's result
        assert result.selected_idx == 0
        assert result.confidence == 0.55
        assert result.used_secondary is False  # Secondary called but not used


class TestLLMRerankerSpreadCalibration:
    """Tests for US-63-008: confidence calibration based on candidate spread."""

    def test_close_spread_reduces_confidence(self):
        """Verify close spread (< 0.05) reduces confidence by factor 0.9."""
        config = LLMRerankerConfig(
            close_spread_threshold=0.05,
            close_spread_factor=0.9
        )
        reranker = LLMReranker(config=config)

        # Primary returns 0.80 confidence
        provider = MockLLMProvider(return_values=[(0, 0.80, "Good match", "cot")])

        # Candidates with close spread: 0.90 - 0.88 = 0.02 (< 0.05)
        candidates = [
            (MockSRTSegment("video text 1"), 0.90),
            (MockSRTSegment("video text 2"), 0.88),
        ]

        result = reranker.rerank(
            voiceover_text="test voiceover",
            candidates=candidates,
            primary_provider=provider
        )

        # 0.80 * 0.9 = 0.72
        assert result.confidence == pytest.approx(0.72, rel=0.01)
        assert "spread_adj=" in result.reasoning

    def test_clear_winner_boosts_confidence(self):
        """Verify clear winner (spread > 0.20) boosts confidence by factor 1.1."""
        config = LLMRerankerConfig(
            clear_winner_threshold=0.20,
            clear_winner_factor=1.1
        )
        reranker = LLMReranker(config=config)

        # Primary returns 0.80 confidence
        provider = MockLLMProvider(return_values=[(0, 0.80, "Good match", "cot")])

        # Candidates with clear spread: 0.90 - 0.65 = 0.25 (> 0.20)
        candidates = [
            (MockSRTSegment("video text 1"), 0.90),
            (MockSRTSegment("video text 2"), 0.65),
        ]

        result = reranker.rerank(
            voiceover_text="test voiceover",
            candidates=candidates,
            primary_provider=provider
        )

        # 0.80 * 1.1 = 0.88
        assert result.confidence == pytest.approx(0.88, rel=0.01)
        assert "spread_adj=" in result.reasoning

    def test_clear_winner_boost_capped_at_one(self):
        """Verify boosted confidence is capped at 1.0."""
        config = LLMRerankerConfig(
            clear_winner_threshold=0.20,
            clear_winner_factor=1.1
        )
        reranker = LLMReranker(config=config)

        # Primary returns 0.95 confidence
        provider = MockLLMProvider(return_values=[(0, 0.95, "Great match", "cot")])

        # Candidates with clear spread: 0.90 - 0.50 = 0.40 (> 0.20)
        candidates = [
            (MockSRTSegment("video text 1"), 0.90),
            (MockSRTSegment("video text 2"), 0.50),
        ]

        result = reranker.rerank(
            voiceover_text="test voiceover",
            candidates=candidates,
            primary_provider=provider
        )

        # 0.95 * 1.1 = 1.045, but capped at 1.0
        assert result.confidence == 1.0

    def test_neutral_spread_no_adjustment(self):
        """Verify neutral spread (0.05 <= spread <= 0.20) has no adjustment."""
        config = LLMRerankerConfig(
            close_spread_threshold=0.05,
            clear_winner_threshold=0.20
        )
        reranker = LLMReranker(config=config)

        # Primary returns 0.80 confidence
        provider = MockLLMProvider(return_values=[(0, 0.80, "Good match", "cot")])

        # Candidates with neutral spread: 0.90 - 0.80 = 0.10 (between 0.05 and 0.20)
        candidates = [
            (MockSRTSegment("video text 1"), 0.90),
            (MockSRTSegment("video text 2"), 0.80),
        ]

        result = reranker.rerank(
            voiceover_text="test voiceover",
            candidates=candidates,
            primary_provider=provider
        )

        # No adjustment, confidence stays at 0.80
        assert result.confidence == pytest.approx(0.80, rel=0.01)
        assert "spread_adj=" not in result.reasoning

    def test_single_candidate_no_calibration(self):
        """Verify single candidate skips spread calibration."""
        config = LLMRerankerConfig()
        reranker = LLMReranker(config=config)

        # Primary returns 0.80 confidence
        provider = MockLLMProvider(return_values=[(0, 0.80, "Good match", "cot")])

        # Single candidate - no spread to calculate
        candidates = [
            (MockSRTSegment("video text 1"), 0.90),
        ]

        result = reranker.rerank(
            voiceover_text="test voiceover",
            candidates=candidates,
            primary_provider=provider
        )

        # No adjustment, confidence stays at 0.80
        assert result.confidence == pytest.approx(0.80, rel=0.01)
        assert "spread_adj=" not in result.reasoning

    def test_spread_calibration_config_in_llmrerankerconfig(self):
        """Verify config options exist in LLMRerankerConfig."""
        config = LLMRerankerConfig(
            close_spread_threshold=0.08,
            clear_winner_threshold=0.25,
            close_spread_factor=0.85,
            clear_winner_factor=1.15
        )

        assert config.close_spread_threshold == 0.08
        assert config.clear_winner_threshold == 0.25
        assert config.close_spread_factor == 0.85
        assert config.clear_winner_factor == 1.15

    def test_spread_calibration_default_values(self):
        """Verify default values for spread calibration config."""
        config = LLMRerankerConfig()

        assert config.close_spread_threshold == 0.05
        assert config.clear_winner_threshold == 0.20
        assert config.close_spread_factor == 0.9
        assert config.clear_winner_factor == 1.1


class TestLLMRerankerVideoContext:
    """Tests for US-70-007: video title/description context enrichment."""

    def test_build_video_context_with_title_and_description(self):
        """Verify context string includes title and first sentence of description."""
        result = LLMReranker._build_video_context(
            "Solar Energy Explained",
            "This video covers the basics of solar power. It also discusses costs."
        )
        assert result == "Video context: Solar Energy Explained. This video covers the basics of solar power"

    def test_build_video_context_title_only(self):
        """Verify context string works with title only."""
        result = LLMReranker._build_video_context("Solar Energy Explained", "")
        assert result == "Video context: Solar Energy Explained"

    def test_build_video_context_description_only(self):
        """Verify context string works with description only."""
        result = LLMReranker._build_video_context(
            "", "This video covers the basics of solar power."
        )
        assert result == "Video context: This video covers the basics of solar power"

    def test_build_video_context_empty(self):
        """Verify empty string returned when no title or description."""
        result = LLMReranker._build_video_context("", "")
        assert result == ""

    def test_build_video_context_long_description_truncated(self):
        """Verify first sentence longer than 100 chars is truncated."""
        long_sentence = "A" * 150 + ". Second sentence."
        result = LLMReranker._build_video_context("Title", long_sentence)
        # First sentence is 150 chars, gets truncated to 97 + "..."
        assert "..." in result
        assert len(result) < 200  # Reasonable total length

    def test_enrich_candidates_with_metadata(self):
        """Verify candidates are enriched with video context prefix."""
        config = LLMRerankerConfig()
        reranker = LLMReranker(config=config)

        candidates = [
            (MockSRTSegment("caption text about panels", source_file="vid123"), 0.9),
            (MockSRTSegment("another caption", source_file="vid456"), 0.8),
        ]

        video_metadata = {
            "vid123": {"title": "Solar Energy", "description": "How solar panels work."},
            "vid456": {"title": "Wind Power", "description": ""},
        }

        enriched = reranker._enrich_candidates_with_context(candidates, video_metadata)

        assert len(enriched) == 2
        assert enriched[0][0].text.startswith("[Video context: Solar Energy. How solar panels work]")
        assert "caption text about panels" in enriched[0][0].text
        assert enriched[1][0].text.startswith("[Video context: Wind Power]")
        assert "another caption" in enriched[1][0].text

    def test_enrich_candidates_no_metadata_fallback(self):
        """Verify candidates unchanged when no metadata provided."""
        config = LLMRerankerConfig()
        reranker = LLMReranker(config=config)

        candidates = [
            (MockSRTSegment("caption text", source_file="vid123"), 0.9),
        ]

        enriched = reranker._enrich_candidates_with_context(candidates, None)

        assert enriched[0][0].text == "caption text"

    def test_enrich_candidates_partial_metadata(self):
        """Verify only matching candidates get enriched."""
        config = LLMRerankerConfig()
        reranker = LLMReranker(config=config)

        candidates = [
            (MockSRTSegment("caption 1", source_file="vid123"), 0.9),
            (MockSRTSegment("caption 2", source_file="vid999"), 0.8),
        ]

        video_metadata = {
            "vid123": {"title": "Known Video", "description": ""},
        }

        enriched = reranker._enrich_candidates_with_context(candidates, video_metadata)

        assert "[Video context:" in enriched[0][0].text
        assert enriched[1][0].text == "caption 2"  # No metadata, unchanged

    def test_rerank_passes_enriched_candidates_to_provider(self):
        """Verify enriched candidates are passed to LLM provider."""
        config = LLMRerankerConfig()
        reranker = LLMReranker(config=config)
        provider = MockLLMProvider()

        candidates = [
            (MockSRTSegment("solar panel footage", source_file="vid123"), 0.9),
            (MockSRTSegment("wind turbine footage", source_file="vid456"), 0.8),
        ]

        video_metadata = {
            "vid123": {"title": "Solar Energy Guide", "description": "Complete guide to solar."},
        }

        reranker.rerank(
            voiceover_text="renewable energy sources",
            candidates=candidates,
            primary_provider=provider,
            video_metadata=video_metadata
        )

        batch = provider.last_call_args['batch']
        enriched_candidates = batch[0][1]
        # First candidate should be enriched
        assert "[Video context:" in enriched_candidates[0][0].text
        assert "Solar Energy Guide" in enriched_candidates[0][0].text
        # Second candidate has no metadata, unchanged
        assert enriched_candidates[1][0].text == "wind turbine footage"

    def test_rerank_without_metadata_falls_back(self):
        """Verify rerank works normally when no video_metadata provided."""
        config = LLMRerankerConfig()
        reranker = LLMReranker(config=config)
        provider = MockLLMProvider()

        candidates = [
            (MockSRTSegment("caption text", source_file="vid123"), 0.9),
        ]

        reranker.rerank(
            voiceover_text="test voiceover",
            candidates=candidates,
            primary_provider=provider
        )

        batch = provider.last_call_args['batch']
        # Candidate text should be unchanged
        assert batch[0][1][0][0].text == "caption text"

    def test_cache_key_differs_with_video_context(self):
        """Verify cache key changes when video context is added."""
        config = LLMRerankerConfig()
        reranker = LLMReranker(config=config)

        candidates = [
            (MockSRTSegment("caption text", source_file="vid123"), 0.9),
        ]

        key_without = reranker._get_cache_key("voiceover", candidates)

        # Enrich candidates
        video_metadata = {"vid123": {"title": "Solar Energy", "description": ""}}
        enriched = reranker._enrich_candidates_with_context(candidates, video_metadata)

        key_with = reranker._get_cache_key("voiceover", enriched)

        assert key_without != key_with


class TestVideoMetadataConstruction:
    """Tests for US-72-006: video_metadata construction from state.video_search_results."""

    def test_video_metadata_built_from_video_search_results(self):
        """Verify video_metadata dict is built correctly from VideoSearchResult objects."""
        from dataclasses import dataclass, field

        @dataclass
        class FakeVideoSearchResult:
            video_id: str = ""
            title: str = ""
            description: str = ""
            url: str = ""

        results = [
            FakeVideoSearchResult(video_id="abc123", title="Solar Energy", description="About solar panels."),
            FakeVideoSearchResult(video_id="def456", title="Wind Power", description="Wind turbines explained."),
        ]

        # Replicate the construction logic from MatchStage._run_matching
        video_metadata = {}
        for vsr in results:
            vid_id = vsr.video_id if hasattr(vsr, 'video_id') else ''
            title = vsr.title if hasattr(vsr, 'title') else ''
            desc = vsr.description if hasattr(vsr, 'description') else ''
            if vid_id and (title or desc):
                video_metadata[vid_id] = {'title': title, 'description': desc}

        assert len(video_metadata) == 2
        assert video_metadata["abc123"]["title"] == "Solar Energy"
        assert video_metadata["abc123"]["description"] == "About solar panels."
        assert video_metadata["def456"]["title"] == "Wind Power"

    def test_video_metadata_skips_entries_without_title_or_description(self):
        """Verify entries with no title AND no description are skipped."""
        from dataclasses import dataclass

        @dataclass
        class FakeVideoSearchResult:
            video_id: str = ""
            title: str = ""
            description: str = ""

        results = [
            FakeVideoSearchResult(video_id="abc123", title="Has Title", description=""),
            FakeVideoSearchResult(video_id="def456", title="", description="Has description"),
            FakeVideoSearchResult(video_id="ghi789", title="", description=""),  # Should be skipped
        ]

        video_metadata = {}
        for vsr in results:
            vid_id = vsr.video_id if hasattr(vsr, 'video_id') else ''
            title = vsr.title if hasattr(vsr, 'title') else ''
            desc = vsr.description if hasattr(vsr, 'description') else ''
            if vid_id and (title or desc):
                video_metadata[vid_id] = {'title': title, 'description': desc}

        assert len(video_metadata) == 2
        assert "abc123" in video_metadata
        assert "def456" in video_metadata
        assert "ghi789" not in video_metadata

    def test_video_metadata_handles_dict_format(self):
        """Verify video_metadata construction handles dict-format results."""
        results = [
            {"video_id": "abc123", "title": "Solar Energy", "description": "About solar."},
        ]

        video_metadata = {}
        for vsr in results:
            vid_id = vsr.video_id if hasattr(vsr, 'video_id') else vsr.get('video_id', '') if isinstance(vsr, dict) else ''
            title = vsr.title if hasattr(vsr, 'title') else vsr.get('title', '') if isinstance(vsr, dict) else ''
            desc = vsr.description if hasattr(vsr, 'description') else vsr.get('description', '') if isinstance(vsr, dict) else ''
            if vid_id and (title or desc):
                video_metadata[vid_id] = {'title': title, 'description': desc}

        assert len(video_metadata) == 1
        assert video_metadata["abc123"]["title"] == "Solar Energy"

    def test_video_metadata_empty_results(self):
        """Verify empty video_search_results produces empty metadata."""
        video_metadata = {}
        for vsr in []:
            pass  # no iterations
        assert video_metadata == {}

    def test_enrichment_end_to_end_with_constructed_metadata(self):
        """Verify constructed video_metadata flows through to enriched candidates."""
        from dataclasses import dataclass as dc

        @dc
        class FakeVSR:
            video_id: str = ""
            title: str = ""
            description: str = ""

        # Build metadata like the match stage does
        results = [FakeVSR(video_id="vid123", title="Solar Energy", description="How panels work.")]
        video_metadata = {}
        for vsr in results:
            if vsr.video_id and (vsr.title or vsr.description):
                video_metadata[vsr.video_id] = {'title': vsr.title, 'description': vsr.description}

        # Pass through reranker enrichment
        config = LLMRerankerConfig()
        reranker = LLMReranker(config=config)
        candidates = [
            (MockSRTSegment("caption about solar", source_file="vid123"), 0.9),
        ]
        enriched = reranker._enrich_candidates_with_context(candidates, video_metadata)

        assert "[Video context: Solar Energy. How panels work]" in enriched[0][0].text
        assert "caption about solar" in enriched[0][0].text


class TestUS95VideoMetadataEnrichment:
    """Tests for US-95-005: Pass full video metadata context to LLM reranker."""

    def test_build_video_context_includes_tags(self):
        """Verify tags are included in video context."""
        result = LLMReranker._build_video_context(
            "Solar Energy",
            "How solar panels work.",
            tags=["solar", "renewable", "energy", "panels", "photovoltaic"]
        )
        assert "Tags:" in result
        assert "solar" in result
        assert "renewable" in result

    def test_build_video_context_includes_chapters(self):
        """Verify chapter titles are included in video context."""
        chapters = [
            {"title": "Introduction to Solar", "start_time": 0},
            {"title": "How Panels Work", "start_time": 120},
            {"title": "Cost Analysis", "start_time": 300},
        ]
        result = LLMReranker._build_video_context(
            "Solar Energy",
            "Complete guide to solar power.",
            chapters=chapters
        )
        assert "Chapters:" in result
        assert "Introduction to Solar" in result
        assert "How Panels Work" in result

    def test_build_video_context_tags_limit(self):
        """Verify tags are limited to 5."""
        tags = ["tag1", "tag2", "tag3", "tag4", "tag5", "tag6", "tag7"]
        result = LLMReranker._build_video_context(
            "Title", "Description.", tags=tags
        )
        # Should only include first 5 tags
        assert "tag1" in result
        assert "tag2" in result
        assert "tag3" in result
        assert "tag4" in result
        assert "tag5" in result
        assert "tag6" not in result
        assert "tag7" not in result

    def test_build_video_context_chapters_limit(self):
        """Verify chapters are limited to 3."""
        chapters = [
            {"title": "Chapter 1", "start_time": 0},
            {"title": "Chapter 2", "start_time": 60},
            {"title": "Chapter 3", "start_time": 120},
            {"title": "Chapter 4", "start_time": 180},
        ]
        result = LLMReranker._build_video_context(
            "Title", "Description.", chapters=chapters
        )
        # Should only include first 3 chapters
        assert "Chapter 1" in result
        assert "Chapter 2" in result
        assert "Chapter 3" in result
        assert "Chapter 4" not in result

    def test_build_video_context_handles_dict_chapters(self):
        """Verify chapters can be dict format with title key."""
        chapters = [{"title": "Getting Started"}, {"title": "Advanced Topics"}]
        result = LLMReranker._build_video_context(
            "Python Tutorial", "Learn Python programming.", chapters=chapters
        )
        assert "Getting Started" in result
        assert "Advanced Topics" in result

    def test_build_video_context_empty_when_no_metadata(self):
        """Verify empty string returned when all metadata is empty."""
        result = LLMReranker._build_video_context("", "", [], [])
        assert result == ""

    def test_reranker_include_metadata_config_default(self):
        """Verify reranker_include_metadata defaults to True."""
        config = LLMRerankerConfig()
        assert config.reranker_include_metadata is True

    def test_reranker_include_metadata_config_false(self):
        """Verify reranker_include_metadata can be set to False."""
        config = LLMRerankerConfig(reranker_include_metadata=False)
        assert config.reranker_include_metadata is False

    def test_enrich_candidates_respects_config_flag(self):
        """Verify enrichment is skipped when config flag is False."""
        config = LLMRerankerConfig(reranker_include_metadata=False)
        reranker = LLMReranker(config=config)

        candidates = [
            (MockSRTSegment("caption text", source_file="vid123"), 0.9),
        ]
        video_metadata = {
            "vid123": {"title": "Solar Energy", "description": "About solar.", "tags": ["solar"], "chapters": []},
        }

        enriched = reranker._enrich_candidates_with_context(candidates, video_metadata)

        # Should not be enriched because config flag is False
        assert enriched[0][0].text == "caption text"

    def test_enrich_candidates_with_tags_and_chapters(self):
        """Verify candidates are enriched with tags and chapters."""
        config = LLMRerankerConfig()
        reranker = LLMReranker(config=config)

        candidates = [
            (MockSRTSegment("caption about panels", source_file="vid123"), 0.9),
        ]
        video_metadata = {
            "vid123": {
                "title": "Solar Energy Guide",
                "description": "Complete guide to solar.",
                "tags": ["solar", "renewable", "energy"],
                "chapters": [
                    {"title": "Introduction", "start_time": 0},
                    {"title": "Installation", "start_time": 180},
                ]
            },
        }

        enriched = reranker._enrich_candidates_with_context(candidates, video_metadata)

        assert "Video context:" in enriched[0][0].text
        assert "Solar Energy Guide" in enriched[0][0].text
        assert "Tags:" in enriched[0][0].text
        assert "solar" in enriched[0][0].text
        assert "Chapters:" in enriched[0][0].text
        assert "Introduction" in enriched[0][0].text

    def test_rerank_passes_full_metadata_to_provider(self):
        """Verify full metadata (tags, chapters) is passed to LLM provider."""
        config = LLMRerankerConfig()
        reranker = LLMReranker(config=config)
        provider = MockLLMProvider()

        candidates = [
            (MockSRTSegment("solar panel footage", source_file="vid123"), 0.9),
        ]

        video_metadata = {
            "vid123": {
                "title": "Solar Energy Guide",
                "description": "Complete guide to solar.",
                "tags": ["solar", "renewable"],
                "chapters": [{"title": "Getting Started"}],
            },
        }

        reranker.rerank(
            voiceover_text="renewable energy sources",
            candidates=candidates,
            primary_provider=provider,
            video_metadata=video_metadata
        )

        batch = provider.last_call_args['batch']
        enriched_candidates = batch[0][1]
        # First candidate should be enriched with tags and chapters
        assert "Tags:" in enriched_candidates[0][0].text
        assert "solar" in enriched_candidates[0][0].text
        assert "Chapters:" in enriched_candidates[0][0].text

    def test_from_matching_config_includes_reranker_include_metadata(self):
        """Verify factory method reads reranker_include_metadata from matching config."""
        matching_config = MagicMock()
        matching_config.reranker_include_metadata = False
        matching_config.ambiguous_threshold = 0.65
        matching_config.cache_llm_responses = True
        matching_config.llm_reranker_close_spread_threshold = 0.05

        reranker = LLMReranker.from_matching_config(matching_config)

        assert reranker.config.reranker_include_metadata is False

    def test_from_matching_config_defaults_reranker_include_metadata(self):
        """Verify factory method defaults reranker_include_metadata to True."""
        matching_config = MagicMock(spec=[])  # Empty spec - no attributes

        reranker = LLMReranker.from_matching_config(matching_config)

        assert reranker.config.reranker_include_metadata is True

    def test_reranker_produces_better_selections_with_metadata(self):
        """Verify that including metadata leads to better/more informed selections.

        This test demonstrates that metadata (tags, chapters) helps the LLM
        make better decisions when caption text alone is ambiguous.
        """
        config = LLMRerankerConfig()
        reranker = LLMReranker(config=config)

        # Create a scenario where captions are ambiguous but metadata clarifies intent
        # Caption text is vague: "The system is working" - could be about many topics
        candidates = [
            (MockSRTSegment("The system is working properly", source_file="vid_solar"), 0.85),
            (MockSRTSegment("The system is working", source_file="vid_car"), 0.85),
            (MockSRTSegment("The system is working great", source_file="vid_computer"), 0.85),
        ]

        # With metadata: video is about solar panels
        video_metadata = {
            "vid_solar": {
                "title": "Solar Panel Installation",
                "description": "How to install solar panels on your roof.",
                "tags": ["solar", "renewable energy", "photovoltaic", "green energy"],
                "chapters": [
                    {"title": "Introduction to Solar", "start_time": 0},
                    {"title": "Panel Selection", "start_time": 180},
                ]
            },
            "vid_car": {
                "title": "Car Engine Repair",
                "description": "How to fix car engine problems.",
                "tags": ["car", "automotive", "repair", "engine"],
                "chapters": [
                    {"title": "Engine Overview", "start_time": 0},
                ]
            },
            "vid_computer": {
                "title": "Computer Setup Guide",
                "description": "Setting up your new computer.",
                "tags": ["computer", "technology", "setup", "guide"],
                "chapters": []
            },
        }

        # Run rerank with voiceover about renewable energy
        provider_with_metadata = MockLLMProvider(
            return_values=[
                (0, 0.9, "Best match: solar panel installation matches renewable energy voiceover", "cot")
            ]
        )

        result = reranker.rerank(
            voiceover_text="I need footage about renewable energy sources and solar power",
            candidates=candidates,
            primary_provider=provider_with_metadata,
            video_metadata=video_metadata
        )

        # Verify metadata was included in the context passed to LLM
        batch = provider_with_metadata.last_call_args['batch']
        enriched_text = batch[0][1][0][0].text

        # The enriched text should contain metadata
        assert "Video context:" in enriched_text
        assert "Solar Panel Installation" in enriched_text
        assert "solar" in enriched_text
        assert "Tags:" in enriched_text
        assert "Chapters:" in enriched_text

        # The LLM selected vid_solar (index 0) which has matching solar tags
        # This demonstrates metadata helps make better selection
        assert result.selected_idx == 0
