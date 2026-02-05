"""Tests for persistent adaptive caption format ordering (US-67-006).

Verifies:
- Save/load round-trip with correct JSON structure
- 7-day stale entry exclusion
- compute_order produces correct ordering from stats
- from_metrics_counts helper builds proper stats dict
- Missing/corrupt file handling
"""

import json
import pytest
from datetime import datetime, timezone, timedelta
from pathlib import Path

from src.caption.format_stats import FormatStatsFile, _is_stale, STALE_DAYS


@pytest.fixture
def project_dir(tmp_path):
    """Create a temporary project directory with .cache."""
    cache_dir = tmp_path / ".cache"
    cache_dir.mkdir()
    return tmp_path


@pytest.fixture
def stats_file(project_dir):
    return FormatStatsFile(project_dir)


class TestFormatStatsRoundTrip:
    """Round-trip: save stats, load, confirm structure and values."""

    def test_save_creates_file(self, stats_file):
        """save() creates caption_format_stats.json in .cache/."""
        stats = {
            "vtt": {"successes": 95, "failures": 2},
            "json3": {"successes": 80, "failures": 5},
            "srt": {"successes": 25, "failures": 10},
        }
        assert stats_file.save(stats) is True
        assert stats_file.path.exists()

    def test_round_trip_preserves_counts(self, stats_file):
        """Save then load preserves successes/failures for each format."""
        stats = {
            "vtt": {"successes": 95, "failures": 2},
            "json3": {"successes": 80, "failures": 5},
        }
        stats_file.save(stats)
        loaded = stats_file.load()

        assert loaded["vtt"]["successes"] == 95
        assert loaded["vtt"]["failures"] == 2
        assert loaded["json3"]["successes"] == 80
        assert loaded["json3"]["failures"] == 5

    def test_round_trip_adds_last_updated(self, stats_file):
        """save() adds last_updated ISO timestamp to each entry."""
        stats = {"vtt": {"successes": 10, "failures": 1}}
        stats_file.save(stats)
        loaded = stats_file.load()

        assert "last_updated" in loaded["vtt"]
        # Verify it's a parseable ISO timestamp
        ts = datetime.fromisoformat(loaded["vtt"]["last_updated"])
        assert ts.tzinfo is not None

    def test_json_structure_matches_spec(self, stats_file):
        """File contains {format: {successes, failures, last_updated}}."""
        stats = {
            "vtt": {"successes": 50, "failures": 3},
            "json3": {"successes": 40, "failures": 7},
        }
        stats_file.save(stats)

        raw = json.loads(stats_file.path.read_text(encoding="utf-8"))
        for fmt in ("vtt", "json3"):
            assert "successes" in raw[fmt]
            assert "failures" in raw[fmt]
            assert "last_updated" in raw[fmt]

    def test_compute_order_matches_after_round_trip(self, stats_file):
        """compute_order from saved stats matches expected ordering."""
        stats = {
            "vtt": {"successes": 95, "failures": 2},    # 97.9% rate
            "json3": {"successes": 80, "failures": 20},  # 80% rate
            "srt": {"successes": 25, "failures": 75},    # 25% rate
        }
        stats_file.save(stats)

        # Save, load, compute_order — should be vtt > json3 > srt
        order = stats_file.compute_order()
        assert order == ["vtt", "json3", "srt"]


class TestStaleEntryFiltering:
    """Entries older than 7 days are excluded from ordering calculation."""

    def test_fresh_entries_included(self, stats_file):
        """Entries with recent last_updated are included."""
        now = datetime.now(timezone.utc).isoformat()
        raw = {
            "vtt": {"successes": 50, "failures": 1, "last_updated": now},
        }
        stats_file.path.write_text(json.dumps(raw), encoding="utf-8")

        loaded = stats_file.load()
        assert "vtt" in loaded

    def test_stale_entries_excluded(self, stats_file):
        """Entries older than 7 days are excluded from load()."""
        old = (datetime.now(timezone.utc) - timedelta(days=8)).isoformat()
        fresh = datetime.now(timezone.utc).isoformat()
        raw = {
            "vtt": {"successes": 50, "failures": 1, "last_updated": fresh},
            "json3": {"successes": 40, "failures": 2, "last_updated": old},
        }
        stats_file.path.write_text(json.dumps(raw), encoding="utf-8")

        loaded = stats_file.load()
        assert "vtt" in loaded
        assert "json3" not in loaded

    def test_stale_entries_excluded_from_compute_order(self, stats_file):
        """compute_order ignores stale entries."""
        old = (datetime.now(timezone.utc) - timedelta(days=8)).isoformat()
        fresh = datetime.now(timezone.utc).isoformat()
        raw = {
            "srt": {"successes": 999, "failures": 0, "last_updated": old},
            "vtt": {"successes": 10, "failures": 1, "last_updated": fresh},
        }
        stats_file.path.write_text(json.dumps(raw), encoding="utf-8")

        # srt is stale, so only vtt should appear (srt has highest rate but is excluded)
        order = stats_file.compute_order()
        assert order == ["vtt"]

    def test_all_stale_returns_empty(self, stats_file):
        """When all entries are stale, load returns empty dict."""
        old = (datetime.now(timezone.utc) - timedelta(days=10)).isoformat()
        raw = {
            "vtt": {"successes": 50, "failures": 1, "last_updated": old},
            "json3": {"successes": 40, "failures": 2, "last_updated": old},
        }
        stats_file.path.write_text(json.dumps(raw), encoding="utf-8")

        loaded = stats_file.load()
        assert loaded == {}

    def test_is_stale_boundary(self):
        """_is_stale correctly handles the 7-day boundary."""
        exactly_7_days = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
        just_over = (datetime.now(timezone.utc) - timedelta(days=7, seconds=1)).isoformat()
        just_under = (datetime.now(timezone.utc) - timedelta(days=6, hours=23)).isoformat()

        assert _is_stale(exactly_7_days, STALE_DAYS) is False  # exactly at boundary
        assert _is_stale(just_over, STALE_DAYS) is True  # just over
        assert _is_stale(just_under, STALE_DAYS) is False  # under


class TestComputeOrder:
    """compute_order sorts by success rate and appends missing defaults."""

    def test_sorts_by_success_rate_descending(self, stats_file):
        stats = {
            "srt": {"successes": 10, "failures": 90},   # 10%
            "vtt": {"successes": 90, "failures": 10},   # 90%
            "json3": {"successes": 50, "failures": 50},  # 50%
        }
        stats_file.save(stats)
        order = stats_file.compute_order()
        assert order == ["vtt", "json3", "srt"]

    def test_appends_missing_defaults(self, stats_file):
        stats = {"vtt": {"successes": 10, "failures": 0}}
        stats_file.save(stats)
        order = stats_file.compute_order(default_formats=["json3", "vtt", "srt"])
        assert order[0] == "vtt"
        assert "json3" in order
        assert "srt" in order

    def test_empty_file_returns_defaults(self, stats_file):
        order = stats_file.compute_order(default_formats=["json3", "vtt", "srt"])
        assert order == ["json3", "vtt", "srt"]

    def test_empty_file_no_defaults_returns_empty(self, stats_file):
        order = stats_file.compute_order()
        assert order == []


class TestFromMetricsCounts:
    """from_metrics_counts helper builds proper stats dict."""

    def test_success_only(self):
        result = FormatStatsFile.from_metrics_counts(
            success_counts={"vtt": 95, "json3": 80},
        )
        assert result["vtt"] == {"successes": 95, "failures": 0}
        assert result["json3"] == {"successes": 80, "failures": 0}

    def test_with_failure_counts(self):
        result = FormatStatsFile.from_metrics_counts(
            success_counts={"vtt": 95, "json3": 80},
            failure_counts={"vtt": 5, "srt": 10},
        )
        assert result["vtt"] == {"successes": 95, "failures": 5}
        assert result["json3"] == {"successes": 80, "failures": 0}
        assert result["srt"] == {"successes": 0, "failures": 10}


class TestEdgeCases:
    """Missing file, corrupt JSON, missing .cache directory."""

    def test_load_missing_file(self, stats_file):
        loaded = stats_file.load()
        assert loaded == {}

    def test_load_corrupt_json(self, stats_file):
        stats_file.path.write_text("not valid json{{{", encoding="utf-8")
        loaded = stats_file.load()
        assert loaded == {}

    def test_load_non_dict_json(self, stats_file):
        stats_file.path.write_text("[1, 2, 3]", encoding="utf-8")
        loaded = stats_file.load()
        assert loaded == {}

    def test_save_creates_cache_dir(self, tmp_path):
        """save() creates .cache directory if it doesn't exist."""
        sf = FormatStatsFile(tmp_path / "nonexistent_project")
        assert sf.save({"vtt": {"successes": 1, "failures": 0}}) is True
        assert sf.path.exists()

    def test_missing_last_updated_treated_as_stale(self, stats_file):
        raw = {"vtt": {"successes": 50, "failures": 1}}
        stats_file.path.write_text(json.dumps(raw), encoding="utf-8")
        loaded = stats_file.load()
        assert "vtt" not in loaded  # missing last_updated → stale
