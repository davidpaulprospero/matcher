"""Tests for US-73-012: Caption language confidence and fallback tracking."""

import pytest
from unittest.mock import MagicMock, patch
from src.caption.models import CaptionResult, CaptionSegment


def _make_segments(n=3):
    """Helper to create N simple caption segments."""
    return [
        CaptionSegment(index=i, start_time=i * 10.0, end_time=(i + 1) * 10.0, text=f"segment {i}")
        for i in range(n)
    ]


class TestCaptionResultLanguageConfidence:
    """Test language_confidence and fallback_language fields on CaptionResult."""

    def test_default_language_confidence_is_1(self):
        """New CaptionResult should default to 1.0 language_confidence."""
        result = CaptionResult(video_id="test123", segments=_make_segments())
        assert result.language_confidence == 1.0
        assert result.fallback_language == ""

    def test_manual_captions_confidence(self):
        """Manual captions in target language should have confidence 1.0."""
        result = CaptionResult(
            video_id="test123",
            segments=_make_segments(),
            is_auto_generated=False,
            language="en",
            language_confidence=1.0,
        )
        assert result.language_confidence == 1.0

    def test_auto_generated_target_language_confidence(self):
        """Auto-generated captions in target language should have confidence 0.8."""
        result = CaptionResult(
            video_id="test123",
            segments=_make_segments(),
            is_auto_generated=True,
            language="en",
            language_confidence=0.8,
        )
        assert result.language_confidence == 0.8

    def test_auto_generated_translated_confidence(self):
        """Auto-generated captions translated from different language should have confidence 0.5."""
        result = CaptionResult(
            video_id="test123",
            segments=_make_segments(),
            is_auto_generated=True,
            language="es",
            language_confidence=0.5,
            fallback_language="es",
        )
        assert result.language_confidence == 0.5
        assert result.fallback_language == "es"

    def test_to_dict_includes_language_confidence(self):
        """to_dict should include language_confidence and fallback_language."""
        result = CaptionResult(
            video_id="test123",
            segments=_make_segments(),
            language_confidence=0.5,
            fallback_language="fr",
        )
        d = result.to_dict()
        assert d['language_confidence'] == 0.5
        assert d['fallback_language'] == "fr"

    def test_to_dict_default_values(self):
        """to_dict should include defaults for language_confidence."""
        result = CaptionResult(video_id="test123", segments=_make_segments())
        d = result.to_dict()
        assert d['language_confidence'] == 1.0
        assert d['fallback_language'] == ""


class TestFetchCaptionsLanguageConfidence:
    """Test that caption_fetcher sets language_confidence correctly."""

    def test_manual_captions_in_preferred_language(self):
        """When manual captions exist in preferred language, confidence=1.0."""
        from src.caption_fetcher import CaptionFetcher

        fetcher = CaptionFetcher.__new__(CaptionFetcher)
        fetcher.config = None  # No config -> no language_priority
        # Simulate _select_best_track returning manual captions in preferred lang
        result = fetcher._select_best_track(
            [MagicMock(code='en', is_auto_generated=False)],
            'en', True
        )
        assert result == ('en', False, 0)
        # Manual + matching language -> 1.0

    def test_auto_captions_in_preferred_language(self):
        """When auto captions exist in preferred language, confidence=0.8."""
        from src.caption_fetcher import CaptionFetcher

        fetcher = CaptionFetcher.__new__(CaptionFetcher)
        fetcher.config = None
        result = fetcher._select_best_track(
            [MagicMock(code='en', is_auto_generated=True)],
            'en', True
        )
        assert result == ('en', True, 0)
        # Auto + matching language -> 0.8

    def test_fallback_to_different_language(self):
        """When captions fall back to different language, confidence=0.5."""
        from src.caption_fetcher import CaptionFetcher

        fetcher = CaptionFetcher.__new__(CaptionFetcher)
        fetcher.config = None
        result = fetcher._select_best_track(
            [MagicMock(code='en', is_auto_generated=True)],
            'fr', True
        )
        assert result == ('en', True, -1)
        # Auto + different language -> 0.5


class TestLanguageConfidencePenaltyScoring:
    """Test apply_language_confidence_penalty in scoring."""

    def test_penalty_disabled_by_default(self):
        """When language_confidence_penalty=0.0, no penalty applied."""
        from src.matching.scoring import apply_language_confidence_penalty

        config = MagicMock()
        config.matching.language_confidence_penalty = 0.0
        segment = MagicMock()
        segment.language_confidence = 0.5

        adjusted, reason = apply_language_confidence_penalty(0.8, segment, config)
        assert adjusted == 0.8
        assert reason == ""

    def test_penalty_applied_when_enabled(self):
        """When enabled, penalty = (1.0 - lang_conf) * penalty_factor."""
        from src.matching.scoring import apply_language_confidence_penalty

        config = MagicMock()
        config.matching.language_confidence_penalty = 0.1
        segment = MagicMock()
        segment.language_confidence = 0.5

        adjusted, reason = apply_language_confidence_penalty(0.8, segment, config)
        # penalty = (1.0 - 0.5) * 0.1 = 0.05
        assert adjusted == pytest.approx(0.75, abs=0.001)
        assert "language confidence 0.5" in reason

    def test_no_penalty_for_full_confidence(self):
        """No penalty when language_confidence=1.0."""
        from src.matching.scoring import apply_language_confidence_penalty

        config = MagicMock()
        config.matching.language_confidence_penalty = 0.1
        segment = MagicMock()
        segment.language_confidence = 1.0

        adjusted, reason = apply_language_confidence_penalty(0.8, segment, config)
        assert adjusted == 0.8
        assert reason == ""

    def test_penalty_for_auto_generated_target(self):
        """Auto-generated in target language (0.8) gets small penalty."""
        from src.matching.scoring import apply_language_confidence_penalty

        config = MagicMock()
        config.matching.language_confidence_penalty = 0.1
        segment = MagicMock()
        segment.language_confidence = 0.8

        adjusted, reason = apply_language_confidence_penalty(0.8, segment, config)
        # penalty = (1.0 - 0.8) * 0.1 = 0.02
        assert adjusted == pytest.approx(0.78, abs=0.001)

    def test_penalty_for_translated_captions(self):
        """Translated captions (0.5) get larger penalty."""
        from src.matching.scoring import apply_language_confidence_penalty

        config = MagicMock()
        config.matching.language_confidence_penalty = 0.1
        segment = MagicMock()
        segment.language_confidence = 0.5

        adjusted, reason = apply_language_confidence_penalty(0.8, segment, config)
        # penalty = (1.0 - 0.5) * 0.1 = 0.05
        assert adjusted == pytest.approx(0.75, abs=0.001)

    def test_penalty_does_not_go_below_zero(self):
        """Penalty should not reduce confidence below 0."""
        from src.matching.scoring import apply_language_confidence_penalty

        config = MagicMock()
        config.matching.language_confidence_penalty = 1.0  # Very high
        segment = MagicMock()
        segment.language_confidence = 0.0  # Zero confidence

        adjusted, reason = apply_language_confidence_penalty(0.05, segment, config)
        assert adjusted == 0.0

    def test_missing_language_confidence_attribute(self):
        """When segment has no language_confidence, treat as 1.0."""
        from src.matching.scoring import apply_language_confidence_penalty

        config = MagicMock()
        config.matching.language_confidence_penalty = 0.1
        segment = MagicMock(spec=[])  # No attributes

        adjusted, reason = apply_language_confidence_penalty(0.8, segment, config)
        assert adjusted == 0.8
        assert reason == ""


class TestCachedCaptionLanguageConfidence:
    """Test CachedCaption round-trip preserves language_confidence."""

    def test_cached_caption_round_trip(self):
        """CachedCaption should preserve language_confidence through serialization."""
        from src.caption_fetcher import CachedCaption

        cached = CachedCaption(
            video_id="test123",
            language="en",
            segments=[{'index': 0, 'start': 0.0, 'end': 10.0, 'text': 'hello'}],
            is_auto_generated=True,
            format_source="vtt",
            fetch_timestamp=1000.0,
            language_confidence=0.5,
            fallback_language="fr",
        )

        d = cached.to_dict()
        assert d['language_confidence'] == 0.5
        assert d['fallback_language'] == "fr"

        restored = CachedCaption.from_dict(d)
        assert restored.language_confidence == 0.5
        assert restored.fallback_language == "fr"

    def test_cached_caption_to_result_preserves_confidence(self):
        """to_caption_result should carry language_confidence to CaptionResult."""
        from src.caption_fetcher import CachedCaption

        cached = CachedCaption(
            video_id="test123",
            language="en",
            segments=[{'index': 0, 'start': 0.0, 'end': 10.0, 'text': 'hello'}],
            is_auto_generated=True,
            format_source="vtt",
            fetch_timestamp=1000.0,
            language_confidence=0.8,
            fallback_language="",
        )

        result = cached.to_caption_result()
        assert result.language_confidence == 0.8
        assert result.fallback_language == ""

    def test_backward_compat_no_language_confidence_in_cache(self):
        """Old cache entries without language_confidence should default to 1.0."""
        from src.caption_fetcher import CachedCaption

        old_data = {
            'video_id': 'test123',
            'language': 'en',
            'segments': [],
            'is_auto_generated': False,
            'format_source': 'vtt',
            'fetch_timestamp': 1000.0,
        }
        restored = CachedCaption.from_dict(old_data)
        assert restored.language_confidence == 1.0
        assert restored.fallback_language == ""
