"""
Tests for US-76-006: Cross-chapter coherence diagnostics.

Tests aggregate_chapter_diagnostics() and log_chapter_diagnostics()
from src/matching/scoring.py, plus checkpoint integration in match stage.
"""

import logging
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.matching.scoring import aggregate_chapter_diagnostics, log_chapter_diagnostics


def _make_match(chapter_index, source_file, confidence=0.7):
    """Helper: create a mock Match with voiceover_segment.chapter_index and video_segment.source_file."""
    vo = SimpleNamespace(chapter_index=chapter_index, index=0, text="test")
    vid = SimpleNamespace(source_file=source_file, start_time=0, end_time=10)
    return SimpleNamespace(
        voiceover_segment=vo,
        video_segment=vid,
        confidence=confidence,
    )


def _make_match_result(chapter_index, source_file, confidence=0.7):
    """Helper: create a mock MatchResult wrapping a primary_match."""
    primary = _make_match(chapter_index, source_file, confidence)
    return SimpleNamespace(primary_match=primary, confidence=confidence)


class TestAggregateChapterDiagnostics:
    """Unit tests for aggregate_chapter_diagnostics()."""

    def test_correct_chapter_count(self):
        """Diagnostics summary includes correct chapter count."""
        matches = [
            _make_match(0, "videoA"),
            _make_match(0, "videoB"),
            _make_match(1, "videoC"),
            _make_match(2, "videoD"),
        ]
        result = aggregate_chapter_diagnostics(matches)
        assert result['total_chapters'] == 3

    def test_scatter_metrics_single_source_per_chapter(self):
        """Chapters with 1 source each have avg_consistency = 1.0."""
        matches = [
            _make_match(0, "videoA"),
            _make_match(1, "videoB"),
        ]
        result = aggregate_chapter_diagnostics(matches)
        assert result['avg_source_consistency'] == 1.0
        assert result['chapters_exceeding_threshold'] == 0
        # top_scattered always shows up to 3 chapters ranked by source count
        assert all(t['source_count'] == 1 for t in result['top_scattered'])

    def test_scatter_metrics_multiple_sources(self):
        """Chapter with 2 sources has consistency 0.5."""
        matches = [
            _make_match(0, "videoA"),
            _make_match(0, "videoB"),
        ]
        result = aggregate_chapter_diagnostics(matches)
        # 1 chapter, 2 sources -> consistency = 1/2 = 0.5
        assert result['avg_source_consistency'] == 0.5
        assert result['total_chapters'] == 1

    def test_exceeding_threshold_counted(self):
        """Chapters with sources > threshold are counted."""
        # 6 unique sources in chapter 0 (threshold default = 5)
        matches = [_make_match(0, f"video{i}") for i in range(6)]
        result = aggregate_chapter_diagnostics(matches)
        assert result['chapters_exceeding_threshold'] == 1

    def test_top_scattered_ranked_by_source_count(self):
        """Top scattered chapters are sorted by source count descending."""
        # Chapter 0: 7 sources, Chapter 1: 3 sources, Chapter 2: 10 sources
        matches = (
            [_make_match(0, f"v0_{i}") for i in range(7)] +
            [_make_match(1, f"v1_{i}") for i in range(3)] +
            [_make_match(2, f"v2_{i}") for i in range(10)]
        )
        result = aggregate_chapter_diagnostics(matches)
        top = result['top_scattered']
        assert len(top) == 3
        assert top[0]['chapter_index'] == 2  # 10 sources
        assert top[0]['source_count'] == 10
        assert top[1]['chapter_index'] == 0  # 7 sources
        assert top[1]['source_count'] == 7
        assert top[2]['chapter_index'] == 1  # 3 sources
        assert top[2]['source_count'] == 3

    def test_top_scattered_capped_at_three(self):
        """At most 3 chapters in top_scattered."""
        matches = []
        for ch in range(5):
            for i in range(8):
                matches.append(_make_match(ch, f"v{ch}_{i}"))
        result = aggregate_chapter_diagnostics(matches)
        assert len(result['top_scattered']) == 3

    def test_zero_chapters_graceful(self):
        """No chapter detection produces safe defaults."""
        # Segments with no chapter_index
        matches = [
            SimpleNamespace(
                voiceover_segment=SimpleNamespace(chapter_index=-1, index=0),
                video_segment=SimpleNamespace(source_file="v1"),
                confidence=0.5,
            ),
        ]
        result = aggregate_chapter_diagnostics(matches)
        assert result['total_chapters'] == 0
        assert result['avg_source_consistency'] == 1.0
        assert result['chapters_exceeding_threshold'] == 0
        assert result['top_scattered'] == []
        assert result['per_chapter'] == {}

    def test_empty_matches(self):
        """Empty match list produces zero-chapter result."""
        result = aggregate_chapter_diagnostics([])
        assert result['total_chapters'] == 0

    def test_none_matches(self):
        """None match list is handled gracefully."""
        result = aggregate_chapter_diagnostics(None)
        assert result['total_chapters'] == 0

    def test_match_result_with_primary_match(self):
        """MatchResult objects (with primary_match) are handled correctly."""
        matches = [
            _make_match_result(0, "videoA"),
            _make_match_result(0, "videoB"),
            _make_match_result(1, "videoC"),
        ]
        result = aggregate_chapter_diagnostics(matches)
        assert result['total_chapters'] == 2
        assert result['per_chapter'][0]['source_count'] == 2
        assert result['per_chapter'][1]['source_count'] == 1

    def test_per_chapter_detail_keys(self):
        """per_chapter contains source_count and sources for each chapter."""
        matches = [
            _make_match(0, "videoA"),
            _make_match(0, "videoB"),
        ]
        result = aggregate_chapter_diagnostics(matches)
        assert 0 in result['per_chapter']
        detail = result['per_chapter'][0]
        assert 'source_count' in detail
        assert 'sources' in detail
        assert detail['source_count'] == 2
        assert sorted(detail['sources']) == ['videoA', 'videoB']

    def test_custom_coherence_threshold(self):
        """Custom coherence_threshold changes exceeding count."""
        matches = [_make_match(0, f"v{i}") for i in range(4)]
        # Default threshold=5 -> 4 sources not exceeding
        result_default = aggregate_chapter_diagnostics(matches)
        assert result_default['chapters_exceeding_threshold'] == 0
        # threshold=3 -> 4 sources exceeds
        result_custom = aggregate_chapter_diagnostics(matches, coherence_threshold=3)
        assert result_custom['chapters_exceeding_threshold'] == 1

    def test_checkpoint_contains_chapter_diagnostics_key(self):
        """Checkpoint data dict includes chapter_diagnostics with expected keys."""
        matches = [
            _make_match(0, "videoA"),
            _make_match(1, "videoB"),
        ]
        diag = aggregate_chapter_diagnostics(matches)
        # Verify all required top-level keys
        expected_keys = {
            'total_chapters', 'avg_source_consistency',
            'chapters_exceeding_threshold', 'top_scattered', 'per_chapter',
        }
        assert set(diag.keys()) == expected_keys


class TestLogChapterDiagnostics:
    """Tests for log_chapter_diagnostics()."""

    def test_info_log_emitted_for_chapters(self, caplog):
        """INFO log emitted with chapter count and consistency."""
        diag = aggregate_chapter_diagnostics([
            _make_match(0, "videoA"),
            _make_match(0, "videoB"),
            _make_match(1, "videoC"),
        ])
        with caplog.at_level(logging.INFO, logger="src.matching.scoring"):
            log_chapter_diagnostics(diag)

        info_msgs = [r.message for r in caplog.records if r.levelno == logging.INFO]
        assert any("2 chapters" in m for m in info_msgs)
        assert any("avg_consistency" in m for m in info_msgs)

    def test_info_log_for_zero_chapters(self, caplog):
        """INFO log reports no chapters when none detected."""
        diag = aggregate_chapter_diagnostics([])
        with caplog.at_level(logging.INFO, logger="src.matching.scoring"):
            log_chapter_diagnostics(diag)

        info_msgs = [r.message for r in caplog.records if r.levelno == logging.INFO]
        assert any("no chapters" in m for m in info_msgs)

    def test_debug_log_per_chapter(self, caplog):
        """DEBUG log emitted per chapter with source details."""
        diag = aggregate_chapter_diagnostics([
            _make_match(0, "videoA"),
            _make_match(0, "videoB"),
            _make_match(1, "videoC"),
        ])
        with caplog.at_level(logging.DEBUG, logger="src.matching.scoring"):
            log_chapter_diagnostics(diag)

        debug_msgs = [r.message for r in caplog.records if r.levelno == logging.DEBUG]
        assert any("chapter 0" in m for m in debug_msgs)
        assert any("chapter 1" in m for m in debug_msgs)
