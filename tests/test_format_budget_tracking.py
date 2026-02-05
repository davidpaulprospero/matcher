"""Tests for per-format retry budget tracking (US-59-010).

Verifies that CaptionRetryBudget correctly tracks attempts, successes,
and failures per subtitle format (json3, vtt, srt), supports per-format
exhaustion checks, and survives checkpoint round-trip serialization.
"""

import pytest

from src.caption.retry_budget import CaptionRetryBudget


class TestFormatAttemptTracking:
    """Tests for format_attempts dict tracking."""

    def test_format_attempts_initialized_empty(self):
        """format_attempts starts as an empty dict."""
        budget = CaptionRetryBudget()
        assert budget.format_attempts == {}

    def test_record_attempt_with_format(self):
        """record_attempt() with format param creates and increments format counter."""
        budget = CaptionRetryBudget()
        budget.record_attempt(video_id="v1", format="json3")
        assert "json3" in budget.format_attempts
        assert budget.format_attempts["json3"]["attempts"] == 1
        assert budget.format_attempts["json3"]["failures"] == 0
        assert budget.format_attempts["json3"]["successes"] == 0

    def test_record_attempt_without_format_no_tracking(self):
        """record_attempt() without format doesn't create format entries."""
        budget = CaptionRetryBudget()
        budget.record_attempt(video_id="v1")
        assert budget.format_attempts == {}

    def test_record_success_with_format(self):
        """record_success() with format increments format success counter."""
        budget = CaptionRetryBudget()
        budget.record_attempt(video_id="v1", format="vtt")
        budget.record_success(video_id="v1", format="vtt")
        assert budget.format_attempts["vtt"]["successes"] == 1

    def test_record_failure_with_format(self):
        """record_failure() with format increments format failure counter."""
        budget = CaptionRetryBudget()
        budget.record_attempt(video_id="v1", format="srt")
        budget.record_failure(video_id="v1", format="srt")
        assert budget.format_attempts["srt"]["failures"] == 1

    def test_multiple_formats_tracked_independently(self):
        """Different formats are tracked independently."""
        budget = CaptionRetryBudget()

        # json3: 3 attempts, 2 failures, 1 success
        for i in range(3):
            budget.record_attempt(video_id=f"v{i}", format="json3")
        budget.record_failure(video_id="v0", format="json3")
        budget.record_failure(video_id="v1", format="json3")
        budget.record_success(video_id="v2", format="json3")

        # vtt: 2 attempts, 0 failures, 2 successes
        for i in range(2):
            budget.record_attempt(video_id=f"v{i}", format="vtt")
            budget.record_success(video_id=f"v{i}", format="vtt")

        assert budget.format_attempts["json3"]["attempts"] == 3
        assert budget.format_attempts["json3"]["failures"] == 2
        assert budget.format_attempts["json3"]["successes"] == 1

        assert budget.format_attempts["vtt"]["attempts"] == 2
        assert budget.format_attempts["vtt"]["failures"] == 0
        assert budget.format_attempts["vtt"]["successes"] == 2

    def test_format_accumulates_across_videos(self):
        """Per-format tracking accumulates across multiple videos."""
        budget = CaptionRetryBudget()
        videos = ["vid_a", "vid_b", "vid_c", "vid_d", "vid_e"]

        for vid in videos:
            budget.record_attempt(video_id=vid, format="json3")
            budget.record_failure(video_id=vid, format="json3")

        assert budget.format_attempts["json3"]["attempts"] == 5
        assert budget.format_attempts["json3"]["failures"] == 5
        assert budget.format_attempts["json3"]["successes"] == 0


class TestFormatExhaustion:
    """Tests for per-format exhaustion checks."""

    def test_format_not_exhausted_below_threshold(self):
        """Format with fewer failures than max_format_failures is not exhausted."""
        budget = CaptionRetryBudget(max_format_failures=5)
        for i in range(4):
            budget.record_attempt(video_id=f"v{i}", format="json3")
            budget.record_failure(video_id=f"v{i}", format="json3")

        assert not budget.is_format_exhausted("json3")

    def test_format_exhausted_at_threshold(self):
        """Format is exhausted when failures reach max_format_failures."""
        budget = CaptionRetryBudget(max_format_failures=5)
        for i in range(5):
            budget.record_attempt(video_id=f"v{i}", format="json3")
            budget.record_failure(video_id=f"v{i}", format="json3")

        assert budget.is_format_exhausted("json3")

    def test_format_exhausted_above_threshold(self):
        """Format is exhausted when failures exceed max_format_failures."""
        budget = CaptionRetryBudget(max_format_failures=3)
        for i in range(5):
            budget.record_attempt(video_id=f"v{i}", format="srt")
            budget.record_failure(video_id=f"v{i}", format="srt")

        assert budget.is_format_exhausted("srt")

    def test_unknown_format_not_exhausted(self):
        """A format with no tracking data is not exhausted."""
        budget = CaptionRetryBudget(max_format_failures=5)
        assert not budget.is_format_exhausted("unknown_format")

    def test_format_exhaustion_disabled_when_zero(self):
        """Per-format exhaustion is disabled when max_format_failures=0."""
        budget = CaptionRetryBudget(max_format_failures=0)
        for i in range(100):
            budget.record_attempt(video_id=f"v{i}", format="json3")
            budget.record_failure(video_id=f"v{i}", format="json3")

        assert not budget.is_format_exhausted("json3")

    def test_budget_exhausted_with_format_param(self):
        """budget_exhausted(format=X) returns True when format X is exhausted."""
        budget = CaptionRetryBudget(
            max_attempts=1000,  # Global budget NOT exhausted
            max_backoff_time=1000.0,
            max_format_failures=3,
        )
        for i in range(3):
            budget.record_attempt(video_id=f"v{i}", format="json3")
            budget.record_failure(video_id=f"v{i}", format="json3")

        # Global budget is fine
        assert not budget.budget_exhausted()
        # But json3 specifically is exhausted
        assert budget.budget_exhausted(format="json3")
        # Other formats are fine
        assert not budget.budget_exhausted(format="vtt")

    def test_budget_exhausted_without_format_ignores_format_tracking(self):
        """budget_exhausted() without format param doesn't check format exhaustion."""
        budget = CaptionRetryBudget(
            max_attempts=1000,
            max_format_failures=3,
        )
        for i in range(10):
            budget.record_attempt(video_id=f"v{i}", format="json3")
            budget.record_failure(video_id=f"v{i}", format="json3")

        # Global budget is fine, and no format param passed
        assert not budget.budget_exhausted()


class TestGetFormatStats:
    """Tests for get_format_stats()."""

    def test_empty_stats(self):
        """get_format_stats() returns empty dict when no formats tracked."""
        budget = CaptionRetryBudget()
        assert budget.get_format_stats() == {}

    def test_returns_copy(self):
        """get_format_stats() returns a copy, not a reference."""
        budget = CaptionRetryBudget()
        budget.record_attempt(video_id="v1", format="json3")
        stats = budget.get_format_stats()
        stats["json3"]["attempts"] = 999
        assert budget.format_attempts["json3"]["attempts"] == 1

    def test_multiple_formats(self):
        """get_format_stats() returns stats for all tracked formats."""
        budget = CaptionRetryBudget()
        budget.record_attempt(video_id="v1", format="json3")
        budget.record_attempt(video_id="v1", format="vtt")
        budget.record_attempt(video_id="v1", format="srt")

        stats = budget.get_format_stats()
        assert set(stats.keys()) == {"json3", "vtt", "srt"}


class TestFormatSerializationRoundTrip:
    """Tests for to_dict/from_dict serialization of format_attempts."""

    def test_to_dict_includes_format_attempts(self):
        """to_dict() includes format_attempts field."""
        budget = CaptionRetryBudget()
        budget.record_attempt(video_id="v1", format="json3")
        budget.record_failure(video_id="v1", format="json3")

        data = budget.to_dict()
        assert "format_attempts" in data
        assert data["format_attempts"]["json3"]["attempts"] == 1
        assert data["format_attempts"]["json3"]["failures"] == 1
        assert data["format_attempts"]["json3"]["successes"] == 0

    def test_to_dict_includes_max_format_failures(self):
        """to_dict() includes max_format_failures field."""
        budget = CaptionRetryBudget(max_format_failures=7)
        data = budget.to_dict()
        assert data["max_format_failures"] == 7

    def test_from_dict_restores_format_attempts(self):
        """from_dict() restores format_attempts from checkpoint data."""
        data = {
            "attempts": 10,
            "failures": 5,
            "successes": 5,
            "backoff_time_spent": 0.0,
            "videos_skipped": 0,
            "format_attempts": {
                "json3": {"attempts": 6, "failures": 4, "successes": 2},
                "vtt": {"attempts": 4, "failures": 1, "successes": 3},
            },
            "max_format_failures": 8,
        }

        budget = CaptionRetryBudget.from_dict(data)

        assert len(budget.format_attempts) == 2
        assert budget.format_attempts["json3"]["attempts"] == 6
        assert budget.format_attempts["json3"]["failures"] == 4
        assert budget.format_attempts["json3"]["successes"] == 2
        assert budget.format_attempts["vtt"]["attempts"] == 4
        assert budget.format_attempts["vtt"]["failures"] == 1
        assert budget.format_attempts["vtt"]["successes"] == 3
        assert budget.max_format_failures == 8

    def test_from_dict_defaults_when_missing(self):
        """from_dict() uses defaults when format_attempts is missing."""
        data = {
            "attempts": 5,
            "failures": 2,
            "successes": 3,
            "backoff_time_spent": 0.0,
            "videos_skipped": 0,
        }

        budget = CaptionRetryBudget.from_dict(data)
        assert budget.format_attempts == {}
        assert budget.max_format_failures == 10  # default

    def test_full_round_trip(self):
        """Complete serialize -> deserialize round trip preserves format data."""
        budget = CaptionRetryBudget(max_format_failures=5)

        # Record mixed data across multiple videos and formats
        for vid in ["v1", "v2", "v3"]:
            budget.record_attempt(video_id=vid, format="json3")
            budget.record_failure(video_id=vid, format="json3")
        for vid in ["v1", "v2"]:
            budget.record_attempt(video_id=vid, format="vtt")
            budget.record_success(video_id=vid, format="vtt")
        budget.record_attempt(video_id="v1", format="srt")
        budget.record_failure(video_id="v1", format="srt")

        # Serialize
        data = budget.to_dict()

        # Deserialize
        restored = CaptionRetryBudget.from_dict(data)

        # Verify all format data preserved
        assert restored.format_attempts["json3"]["attempts"] == 3
        assert restored.format_attempts["json3"]["failures"] == 3
        assert restored.format_attempts["json3"]["successes"] == 0

        assert restored.format_attempts["vtt"]["attempts"] == 2
        assert restored.format_attempts["vtt"]["failures"] == 0
        assert restored.format_attempts["vtt"]["successes"] == 2

        assert restored.format_attempts["srt"]["attempts"] == 1
        assert restored.format_attempts["srt"]["failures"] == 1
        assert restored.format_attempts["srt"]["successes"] == 0

        assert restored.max_format_failures == 5

        # Verify format exhaustion checks work after round-trip
        # json3 has 3 failures, max_format_failures=5 -> NOT exhausted (3 < 5)
        assert not restored.is_format_exhausted("json3")
        assert not restored.is_format_exhausted("vtt")
        assert not restored.is_format_exhausted("srt")

    def test_round_trip_exhaustion_survives(self):
        """Format exhaustion state survives checkpoint round-trip."""
        budget = CaptionRetryBudget(max_format_failures=3)

        # Exhaust json3 format
        for vid in ["v1", "v2", "v3"]:
            budget.record_attempt(video_id=vid, format="json3")
            budget.record_failure(video_id=vid, format="json3")

        assert budget.is_format_exhausted("json3")

        # Round-trip
        data = budget.to_dict()
        restored = CaptionRetryBudget.from_dict(data)

        # Exhaustion state preserved
        assert restored.is_format_exhausted("json3")
        assert restored.budget_exhausted(format="json3")


class TestPerFormatExhaustionInBudgetExhausted:
    """Tests for budget_exhausted(format=X) per-format exhaustion checks (criterion 3)."""

    def test_budget_exhausted_returns_true_for_exhausted_format(self):
        """budget_exhausted(format=X) returns True when format X has too many failures."""
        budget = CaptionRetryBudget(
            max_attempts=9999,  # Global budget NOT exhausted
            max_backoff_time=9999.0,
            max_format_failures=5,  # Per-format threshold
        )
        # Accumulate failures for json3 across different videos until exhausted
        for i in range(5):
            budget.record_attempt(video_id=f"video_{i}", format="json3")
            budget.record_failure(video_id=f"video_{i}", format="json3")

        # Global budget check (no format) should be False
        assert not budget.budget_exhausted()
        # Per-format check for json3 should be True (5 failures = exhausted)
        assert budget.budget_exhausted(format="json3")

    def test_budget_exhausted_returns_false_for_healthy_format(self):
        """budget_exhausted(format=X) returns False when format X is below threshold."""
        budget = CaptionRetryBudget(
            max_attempts=9999,
            max_format_failures=10,
        )
        # Only 3 failures for vtt
        for i in range(3):
            budget.record_attempt(video_id=f"v{i}", format="vtt")
            budget.record_failure(video_id=f"v{i}", format="vtt")

        assert not budget.budget_exhausted(format="vtt")


class TestCrossVideoAccumulationAndCheckpointRoundTrip:
    """Test that per-format tracking accumulates across videos AND survives checkpoint (criterion 5)."""

    def test_accumulation_across_multiple_videos_survives_checkpoint_round_trip(self):
        """Per-format tracking accumulates across videos and survives checkpoint serialization."""
        # Create budget with specific max_format_failures
        budget = CaptionRetryBudget(max_format_failures=10)

        # Simulate json3 failing across 8 different videos (accumulation)
        video_ids = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta"]
        for vid in video_ids:
            budget.record_attempt(video_id=vid, format="json3")
            budget.record_failure(video_id=vid, format="json3")

        # Verify accumulation before serialization
        assert budget.format_attempts["json3"]["attempts"] == 8
        assert budget.format_attempts["json3"]["failures"] == 8
        assert budget.format_attempts["json3"]["successes"] == 0
        # 8 failures < 10 threshold, not exhausted yet
        assert not budget.is_format_exhausted("json3")

        # Serialize to dict (checkpoint)
        checkpoint_data = budget.to_dict()

        # Deserialize from dict (resume from checkpoint)
        restored = CaptionRetryBudget.from_dict(checkpoint_data)

        # Verify accumulation survived checkpoint round-trip
        assert restored.format_attempts["json3"]["attempts"] == 8
        assert restored.format_attempts["json3"]["failures"] == 8
        assert restored.format_attempts["json3"]["successes"] == 0
        assert restored.max_format_failures == 10
        assert not restored.is_format_exhausted("json3")

        # Continue accumulating after restore (simulate 2 more failures)
        for vid in ["iota", "kappa"]:
            restored.record_attempt(video_id=vid, format="json3")
            restored.record_failure(video_id=vid, format="json3")

        # Now 10 failures total -> exhausted
        assert restored.format_attempts["json3"]["failures"] == 10
        assert restored.is_format_exhausted("json3")
        assert restored.budget_exhausted(format="json3")


class TestFormatResetBehavior:
    """Tests for format_attempts clearing on reset."""

    def test_reset_clears_format_attempts(self):
        """reset() clears format_attempts."""
        budget = CaptionRetryBudget()
        budget.record_attempt(video_id="v1", format="json3")
        budget.record_failure(video_id="v1", format="json3")
        assert len(budget.format_attempts) > 0

        budget.reset()
        assert budget.format_attempts == {}

    def test_summary_includes_format_attempts(self):
        """get_summary() includes format_attempts."""
        budget = CaptionRetryBudget()
        budget.record_attempt(video_id="v1", format="json3")
        budget.record_failure(video_id="v1", format="json3")

        summary = budget.get_summary()
        assert "format_attempts" in summary
        assert summary["format_attempts"]["json3"]["failures"] == 1


class TestFormatSkippingInCaptionFetcher:
    """Tests for US-62-008: Exhausted format skipping in _fetch_subtitle_formats (criterion 4, 5)."""

    def test_exhausted_format_skipped_in_fetch_loop(self, monkeypatch):
        """When format is exhausted, _fetch_subtitle_formats skips it and tries next format.

        US-62-008 AC4: When format exhausted, skip that format in future attempts.
        US-62-008 AC5: Test verifies format skipped after max_format_failures failures.
        """
        from pathlib import Path
        from unittest.mock import MagicMock
        from src.caption_fetcher import CaptionFetcher, CaptionResult

        # Create budget with json3 exhausted (3 failures with max_format_failures=3)
        budget = CaptionRetryBudget(max_format_failures=3)
        for i in range(3):
            budget.record_attempt(video_id=f"v{i}", format="json3")
            budget.record_failure(video_id=f"v{i}", format="json3")
        assert budget.is_format_exhausted("json3")
        assert not budget.is_format_exhausted("vtt")
        assert not budget.is_format_exhausted("srt")

        # Create fetcher with retry_budget
        fetcher = CaptionFetcher()
        fetcher._preferred_formats = ["json3", "vtt", "srt"]
        fetcher._active_retry_budget = budget

        # Mock _fetch_subtitle_with_format to track which formats are attempted
        formats_attempted = []

        def mock_fetch_with_format(
            video_url, video_id, temp_dir, language, auto_generated, subtitle_format,
            fallback_level=0
        ):
            formats_attempted.append(subtitle_format)
            if subtitle_format == "vtt":
                # vtt succeeds
                result = MagicMock(spec=CaptionResult)
                result.segments = [MagicMock()]
                return result
            # Other formats fail
            from src.caption_fetcher import CaptionFetchError
            raise CaptionFetchError("test_video", "Format failed")

        monkeypatch.setattr(
            fetcher, '_fetch_subtitle_with_format',
            mock_fetch_with_format
        )

        # Call _fetch_subtitle_formats
        result = fetcher._fetch_subtitle_formats(
            "https://youtube.com/watch?v=test",
            "test_video",
            Path("/tmp"),
            "en",
            False
        )

        # Verify json3 was SKIPPED, vtt was tried and succeeded
        assert "json3" not in formats_attempted, "Exhausted format json3 should be skipped"
        assert "vtt" in formats_attempted, "Non-exhausted format vtt should be tried"
        assert result is not None, "Should return successful vtt result"

    def test_all_formats_exhausted_returns_none(self):
        """When all formats are exhausted, _fetch_subtitle_formats returns None."""
        from pathlib import Path
        from src.caption_fetcher import CaptionFetcher

        # Create budget with all formats exhausted
        budget = CaptionRetryBudget(max_format_failures=2)
        for fmt in ["json3", "vtt", "srt"]:
            for i in range(2):
                budget.record_attempt(video_id=f"v{i}", format=fmt)
                budget.record_failure(video_id=f"v{i}", format=fmt)

        assert budget.is_format_exhausted("json3")
        assert budget.is_format_exhausted("vtt")
        assert budget.is_format_exhausted("srt")

        # Create fetcher with retry_budget
        fetcher = CaptionFetcher()
        fetcher._preferred_formats = ["json3", "vtt", "srt"]
        fetcher._active_retry_budget = budget

        # Track if _fetch_subtitle_with_format was called
        call_count = [0]
        original_method = fetcher._fetch_subtitle_with_format

        def mock_fetch(*args, **kwargs):
            call_count[0] += 1
            raise AssertionError("Should not be called when all formats exhausted")

        fetcher._fetch_subtitle_with_format = mock_fetch

        # Call _fetch_subtitle_formats - should return None (all skipped)
        result = fetcher._fetch_subtitle_formats(
            "https://youtube.com/watch?v=test",
            "test_video",
            Path("/tmp"),
            "en",
            False
        )

        # Verify no formats were tried
        assert call_count[0] == 0, "No formats should be attempted when all are exhausted"
        assert result is None

    def test_format_not_skipped_without_retry_budget(self, monkeypatch):
        """When no retry_budget, all formats are attempted normally."""
        from pathlib import Path
        from unittest.mock import MagicMock
        from src.caption_fetcher import CaptionFetcher, CaptionResult

        fetcher = CaptionFetcher()
        fetcher._preferred_formats = ["json3", "vtt", "srt"]
        fetcher._active_retry_budget = None  # No budget

        formats_attempted = []

        def mock_fetch_with_format(
            video_url, video_id, temp_dir, language, auto_generated, subtitle_format,
            fallback_level=0
        ):
            formats_attempted.append(subtitle_format)
            if subtitle_format == "json3":
                result = MagicMock(spec=CaptionResult)
                result.segments = [MagicMock()]
                return result
            from src.caption_fetcher import CaptionFetchError
            raise CaptionFetchError("test_video", "Format failed")

        monkeypatch.setattr(
            fetcher, '_fetch_subtitle_with_format',
            mock_fetch_with_format
        )

        result = fetcher._fetch_subtitle_formats(
            "https://youtube.com/watch?v=test",
            "test_video",
            Path("/tmp"),
            "en",
            False
        )

        # Without retry_budget, json3 should be tried (not skipped)
        assert "json3" in formats_attempted
        assert result is not None
