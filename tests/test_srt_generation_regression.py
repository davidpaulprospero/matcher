"""
Regression snapshots for SRT generation.

Current generation behavior is the baseline. Any formatting/timing drift
must fail these tests unless snapshots are intentionally updated.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.transcription.utils import write_srt
from src.utils import SRTSegment, write_srt_file


SNAPSHOT_DIR = Path(__file__).parent / "snapshots"


def _load_snapshot_text(snapshot_name: str) -> str:
    snapshot_path = SNAPSHOT_DIR / snapshot_name
    assert snapshot_path.exists(), f"Missing snapshot: {snapshot_path}"
    return snapshot_path.read_text(encoding="utf-8")


class TestSRTGenerationRegression:
    @pytest.mark.fast
    def test_transcription_write_srt_snapshot(self, tmp_path):
        segments = [
            {"start": 0.0, "end": 1.234, "text": "  First line  "},
            {"start": 2.5, "end": 4.0, "text": "Second line\nwith break"},
            {"start": 4.75, "end": 6.0, "text": "Third line"},
        ]

        output_path = tmp_path / "transcription_default.srt"
        write_srt(segments, str(output_path), force_contiguous_timing=False)

        current = output_path.read_text(encoding="utf-8")
        expected = _load_snapshot_text("write_srt_default.snap.srt")
        assert current == expected

    @pytest.mark.fast
    def test_transcription_write_srt_contiguous_snapshot(self, tmp_path):
        segments = [
            {"start": 0.0, "end": 1.234, "text": "  First line  "},
            {"start": 2.5, "end": 4.0, "text": "Second line\nwith break"},
            {"start": 4.75, "end": 6.0, "text": "Third line"},
        ]

        output_path = tmp_path / "transcription_contiguous.srt"
        write_srt(segments, str(output_path), force_contiguous_timing=True)

        current = output_path.read_text(encoding="utf-8")
        expected = _load_snapshot_text("write_srt_contiguous.snap.srt")
        assert current == expected

    @pytest.mark.fast
    def test_utils_write_srt_file_snapshot(self, tmp_path):
        segments = [
            SRTSegment(index=10, start_time=0.0, end_time=1.5, text="Utils one"),
            SRTSegment(index=20, start_time=1.5, end_time=3.25, text="Utils two"),
        ]

        output_path = tmp_path / "utils_writer.srt"
        write_srt_file(segments, str(output_path))

        current = output_path.read_text(encoding="utf-8")
        expected = _load_snapshot_text("write_srt_file_default.snap.srt")
        assert current == expected
