"""
Tests for throughput metrics tracking in batch pipeline stages (US-81-007).

Covers:
- AC1: StageMetrics has items_per_second, peak_items_per_second, throughput_samples
- AC2: compute_throughput() correctly calculates from throughput_samples
- AC3: Sliding window (last 10 items) gives recent throughput, not overall average
- AC4: Throughput in timing summary shows 'N items/sec' for batch stages
- AC5: Serialization roundtrip preserves throughput fields
"""

import logging
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.stages import StageMetrics

pytestmark = pytest.mark.unit


# ============================================================================
# AC1: StageMetrics has new throughput fields
# ============================================================================

class TestThroughputFields:
    """StageMetrics has items_per_second, peak_items_per_second, throughput_samples."""

    @pytest.mark.fast
    def test_default_values(self):
        """New fields default to 0.0 / empty list."""
        m = StageMetrics()
        assert m.items_per_second == 0.0
        assert m.peak_items_per_second == 0.0
        assert m.throughput_samples == []

    @pytest.mark.fast
    def test_fields_settable(self):
        """Fields can be set at construction."""
        m = StageMetrics(
            items_per_second=5.0,
            peak_items_per_second=10.0,
            throughput_samples=[1.0, 2.0, 3.0],
        )
        assert m.items_per_second == 5.0
        assert m.peak_items_per_second == 10.0
        assert m.throughput_samples == [1.0, 2.0, 3.0]


# ============================================================================
# AC2 + AC3: compute_throughput() with sliding window
# ============================================================================

class TestComputeThroughput:
    """Throughput computed using sliding window of last 10 items."""

    @pytest.mark.fast
    def test_empty_samples_fallback(self):
        """With no samples, falls back to items_processed / duration."""
        m = StageMetrics(items_processed=100, duration_seconds=10.0)
        m.compute_throughput()
        assert m.items_per_second == pytest.approx(10.0)

    @pytest.mark.fast
    def test_empty_samples_zero_duration(self):
        """With no samples and zero duration, items_per_second stays 0."""
        m = StageMetrics(items_processed=100, duration_seconds=0.0)
        m.compute_throughput()
        assert m.items_per_second == 0.0

    @pytest.mark.fast
    def test_single_sample(self):
        """Single sample: items_per_second = that sample."""
        m = StageMetrics(throughput_samples=[5.0])
        m.compute_throughput()
        assert m.items_per_second == pytest.approx(5.0)
        assert m.peak_items_per_second == pytest.approx(5.0)

    @pytest.mark.fast
    def test_uniform_samples(self):
        """Uniform samples: all windows have same average."""
        m = StageMetrics(throughput_samples=[2.0] * 20)
        m.compute_throughput()
        assert m.items_per_second == pytest.approx(2.0)
        assert m.peak_items_per_second == pytest.approx(2.0)

    @pytest.mark.fast
    def test_sliding_window_uses_last_10(self):
        """Recent throughput uses last 10 samples, not full history."""
        # First 10 items slow (1 item/sec), last 10 fast (10 items/sec)
        slow = [1.0] * 10
        fast = [10.0] * 10
        m = StageMetrics(throughput_samples=slow + fast)
        m.compute_throughput()
        # Recent throughput should be 10.0 (last 10 samples)
        assert m.items_per_second == pytest.approx(10.0)

    @pytest.mark.fast
    def test_peak_throughput_tracks_max_window(self):
        """Peak throughput captures the highest sliding window average."""
        # Pattern: slow, fast burst, slow again
        samples = [1.0] * 5 + [20.0] * 10 + [1.0] * 5
        m = StageMetrics(throughput_samples=samples)
        m.compute_throughput()
        # Peak should be 20.0 (the window covering the fast burst)
        assert m.peak_items_per_second == pytest.approx(20.0)
        # Recent throughput covers last 10: 5 fast + 5 slow
        expected_recent = (20.0 * 5 + 1.0 * 5) / 10
        assert m.items_per_second == pytest.approx(expected_recent)

    @pytest.mark.fast
    def test_fewer_than_window_size(self):
        """With <10 samples, uses all available."""
        m = StageMetrics(throughput_samples=[2.0, 4.0, 6.0])
        m.compute_throughput()
        assert m.items_per_second == pytest.approx(4.0)  # avg of [2, 4, 6]
        assert m.peak_items_per_second == pytest.approx(4.0)


# ============================================================================
# AC4: Timing summary includes throughput column
# ============================================================================

class TestTimingSummaryThroughput:
    """Pipeline timing summary shows throughput for batch stages."""

    @pytest.fixture
    def mock_orchestrator(self):
        from src.pipeline import PipelineOrchestrator
        with patch.object(PipelineOrchestrator, '__init__', lambda self, *a, **kw: None):
            orch = PipelineOrchestrator.__new__(PipelineOrchestrator)
            orch.stage_timings = {}
            orch.stage_metrics = {}
            orch.stages = []
            orch.checkpoint = MagicMock()
            return orch

    @pytest.fixture
    def make_stage(self):
        def _make(name):
            stage = MagicMock()
            stage.name = name
            return stage
        return _make

    @pytest.mark.fast
    def test_throughput_column_in_header(self, mock_orchestrator, make_stage, caplog):
        """Timing summary header includes Throughput column."""
        mock_orchestrator.stages = [make_stage("CAPTION")]
        mock_orchestrator.stage_timings = {"CAPTION": 10.0}

        with caplog.at_level(logging.INFO, logger='src.pipeline'):
            mock_orchestrator._print_timing_summary(10.0, set())

        messages = [r.message for r in caplog.records]
        header = [m for m in messages if "Stage" in m and "Duration" in m]
        assert len(header) == 1
        assert "Throughput" in header[0]

    @pytest.mark.fast
    def test_throughput_shown_for_batch_stage(self, mock_orchestrator, make_stage, caplog):
        """Batch stage with throughput metrics shows 'N items/sec'."""
        mock_orchestrator.stages = [make_stage("CAPTION")]
        mock_orchestrator.stage_timings = {"CAPTION": 10.0}
        mock_orchestrator.stage_metrics = {
            "CAPTION": StageMetrics(
                items_processed=50,
                duration_seconds=10.0,
                items_per_second=5.0,
            )
        }

        with caplog.at_level(logging.INFO, logger='src.pipeline'):
            mock_orchestrator._print_timing_summary(10.0, set())

        messages = " ".join(r.message for r in caplog.records)
        assert "5.0 items/sec" in messages

    @pytest.mark.fast
    def test_no_throughput_for_non_batch_stage(self, mock_orchestrator, make_stage, caplog):
        """Non-batch stages (no throughput data) show no items/sec."""
        mock_orchestrator.stages = [make_stage("ANALYZE")]
        mock_orchestrator.stage_timings = {"ANALYZE": 2.0}
        mock_orchestrator.stage_metrics = {
            "ANALYZE": StageMetrics(duration_seconds=2.0)
        }

        with caplog.at_level(logging.INFO, logger='src.pipeline'):
            mock_orchestrator._print_timing_summary(2.0, set())

        messages = " ".join(r.message for r in caplog.records)
        assert "items/sec" not in messages


# ============================================================================
# AC5: Serialization roundtrip
# ============================================================================

class TestThroughputSerialization:
    """Throughput fields survive to_dict/from_dict roundtrip."""

    @pytest.mark.fast
    def test_roundtrip(self):
        """Throughput fields preserved through to_dict → from_dict."""
        original = StageMetrics(
            items_processed=100,
            items_per_second=5.5,
            peak_items_per_second=12.3,
            throughput_samples=[1.0, 2.0, 3.0, 4.0, 5.0],
        )
        d = original.to_dict()
        restored = StageMetrics.from_dict(d)
        assert restored.items_per_second == pytest.approx(5.5, abs=0.01)
        assert restored.peak_items_per_second == pytest.approx(12.3, abs=0.01)
        assert len(restored.throughput_samples) == 5

    @pytest.mark.fast
    def test_empty_throughput_not_serialized(self):
        """Zero throughput and empty samples are not included in dict."""
        m = StageMetrics(items_processed=10)
        d = m.to_dict()
        assert 'items_per_second' not in d
        assert 'peak_items_per_second' not in d
        assert 'throughput_samples' not in d

    @pytest.mark.fast
    def test_from_dict_missing_fields(self):
        """from_dict handles missing throughput fields gracefully."""
        d = {'items_processed': 50, 'duration_seconds': 10.0}
        m = StageMetrics.from_dict(d)
        assert m.items_per_second == 0.0
        assert m.peak_items_per_second == 0.0
        assert m.throughput_samples == []
