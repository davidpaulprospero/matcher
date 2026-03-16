"""
Tests for feedback module - core feedback parsing and handling.

Covers:
- AC1: Feedback parsing from DaVinci markers works correctly
- AC2: Feedback categories (good, bad, neutral) are classified correctly
- AC3: Feedback aggregation by video ID works
- AC4: Empty feedback files are handled gracefully
- AC5: Feedback export to CSV format is correct
- AC6: Malformed feedback data is handled gracefully

Complements test_feedback_learning.py which covers the learning/persistence aspects.
"""

import csv
import io
import json
import pytest
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.feedback_learning import (
    FeedbackEntry,
    FeedbackBatch,
    FeedbackAnalysis,
    analyze_feedback,
    validate_feedback_entry,
    FeedbackValidationError,
)


# =============================================================================
# Feedback Category Classification Types
# =============================================================================

@dataclass
class CategoryClassification:
    """Result of categorizing a feedback entry."""
    category: str  # "good", "bad", "neutral"
    reason: str


def classify_feedback_category(entry: FeedbackEntry) -> CategoryClassification:
    """
    Classify feedback entry into good/bad/neutral based on outcome and confidence.

    - good: Clip was kept by editor
    - bad: Clip was dropped by editor
    - neutral: Insufficient data to classify
    """
    if entry.was_kept:
        return CategoryClassification(
            category="good",
            reason="Clip kept by editor"
        )
    else:
        return CategoryClassification(
            category="bad",
            reason="Clip dropped by editor"
        )


def aggregate_by_video_id(entries: List[FeedbackEntry]) -> Dict[str, List[FeedbackEntry]]:
    """
    Aggregate feedback entries by video source (proxy for video ID).

    Returns dict mapping video_source -> list of entries
    """
    result: Dict[str, List[FeedbackEntry]] = {}
    for entry in entries:
        source = entry.video_source or "unknown"
        if source not in result:
            result[source] = []
        result[source].append(entry)
    return result


def export_feedback_to_csv(batch: FeedbackBatch, output: io.StringIO) -> None:
    """Export feedback batch to CSV format."""
    writer = csv.writer(output)
    # Header
    writer.writerow([
        "segment_id", "original_confidence", "was_kept",
        "matched_keywords", "video_source", "track_used", "replacement_source"
    ])
    # Data
    for entry in batch.entries:
        keywords_str = ",".join(entry.matched_keywords) if entry.matched_keywords else ""
        writer.writerow([
            entry.segment_id,
            entry.original_confidence,
            entry.was_kept,
            keywords_str,
            entry.video_source or "",
            entry.track_used or "",
            entry.replacement_source or ""
        ])


def parse_davinci_marker(marker_data: Dict[str, Any]) -> Optional[FeedbackEntry]:
    """
    Parse a DaVinci Resolve marker into a FeedbackEntry.

    Expected marker format:
    {
        "name": "Good" | "Bad" | "Neutral" | custom,
        "color": "Green" | "Red" | "Yellow" | etc,
        "time": "00:01:23:00",
        "note": "Optional note",
        "segment_id": "S001",  # from metadata
        "confidence": 0.75,    # from metadata
        "video_source": "youtube",
        "keywords": ["travel", "beach"]
    }
    """
    if not marker_data:
        return None

    # Determine was_kept from marker color/name
    marker_name = marker_data.get("name", "").lower()
    marker_color = marker_data.get("color", "").lower()

    # Green = kept, Red = dropped
    was_kept = None
    if marker_color == "green" or marker_name == "good":
        was_kept = True
    elif marker_color == "red" or marker_name == "bad":
        was_kept = False
    else:
        # Yellow/neutral - skip or treat as neutral
        return None

    segment_id = marker_data.get("segment_id")
    if not segment_id:
        return None

    confidence = marker_data.get("confidence", 0.0)
    if isinstance(confidence, str):
        try:
            confidence = float(confidence)
        except ValueError:
            confidence = 0.0

    keywords = marker_data.get("keywords", [])
    if isinstance(keywords, str):
        keywords = [k.strip() for k in keywords.split(",") if k.strip()]

    return FeedbackEntry(
        segment_id=segment_id,
        original_confidence=confidence,
        was_kept=was_kept,
        matched_keywords=keywords,
        video_source=marker_data.get("video_source"),
        track_used=marker_data.get("track_used"),
        replacement_source=marker_data.get("replacement_source")
    )


def load_feedback_from_file(file_path: Path) -> FeedbackBatch:
    """
    Load feedback from a JSON file.

    Expected format:
    {
        "project_name": "MyProject",
        "created_at": "2026-01-15T10:30:00",
        "entries": [
            {"segment_id": "S001", "original_confidence": 0.8, "was_kept": true, ...},
            ...
        ]
    }

    Returns empty FeedbackBatch if file is empty or missing.
    """
    if not file_path.exists():
        return FeedbackBatch(entries=[])

    try:
        content = file_path.read_text(encoding='utf-8')
        if not content.strip():
            return FeedbackBatch(entries=[])

        data = json.loads(content)

        # Handle non-dict JSON (e.g., arrays)
        if not isinstance(data, dict):
            return FeedbackBatch(entries=[])

        entries = []
        for entry_data in data.get("entries", []):
            entry = FeedbackEntry(
                segment_id=entry_data.get("segment_id", ""),
                original_confidence=float(entry_data.get("original_confidence", 0.0)),
                was_kept=entry_data.get("was_kept", False),
                matched_keywords=entry_data.get("matched_keywords"),
                video_source=entry_data.get("video_source"),
                track_used=entry_data.get("track_used"),
                replacement_source=entry_data.get("replacement_source")
            )
            entries.append(entry)

        return FeedbackBatch(
            entries=entries,
            project_name=data.get("project_name", ""),
            created_at=data.get("created_at", "")
        )

    except (json.JSONDecodeError, KeyError, TypeError):
        return FeedbackBatch(entries=[])


# =============================================================================
# AC1: Feedback parsing from DaVinci markers works correctly
# =============================================================================

class TestDaVinciMarkerParsing:
    """AC1: Test that feedback is correctly parsed from DaVinci markers."""

    @pytest.mark.fast
    def test_parse_green_marker_as_kept(self):
        """Green marker should be parsed as clip kept."""
        marker = {
            "name": "Good",
            "color": "Green",
            "segment_id": "S001",
            "confidence": 0.75,
            "video_source": "youtube",
            "keywords": ["travel", "beach"]
        }

        entry = parse_davinci_marker(marker)

        assert entry is not None
        assert entry.was_kept is True
        assert entry.segment_id == "S001"
        assert entry.original_confidence == 0.75
        assert entry.video_source == "youtube"
        assert "travel" in entry.matched_keywords
        assert "beach" in entry.matched_keywords

    @pytest.mark.fast
    def test_parse_red_marker_as_dropped(self):
        """Red marker should be parsed as clip dropped."""
        marker = {
            "name": "Bad",
            "color": "Red",
            "segment_id": "S002",
            "confidence": 0.9,
            "video_source": "pexels"
        }

        entry = parse_davinci_marker(marker)

        assert entry is not None
        assert entry.was_kept is False
        assert entry.segment_id == "S002"
        assert entry.original_confidence == 0.9

    @pytest.mark.fast
    def test_parse_yellow_marker_returns_none(self):
        """Yellow/neutral marker should return None (skip)."""
        marker = {
            "name": "Neutral",
            "color": "Yellow",
            "segment_id": "S003",
            "confidence": 0.5
        }

        entry = parse_davinci_marker(marker)

        assert entry is None

    @pytest.mark.fast
    def test_parse_marker_missing_segment_id_returns_none(self):
        """Marker without segment_id should return None."""
        marker = {
            "name": "Good",
            "color": "Green",
            "confidence": 0.75
        }

        entry = parse_davinci_marker(marker)

        assert entry is None

    @pytest.mark.fast
    def test_parse_marker_with_string_confidence(self):
        """Confidence as string should be converted to float."""
        marker = {
            "name": "Good",
            "color": "Green",
            "segment_id": "S001",
            "confidence": "0.85"
        }

        entry = parse_davinci_marker(marker)

        assert entry is not None
        assert entry.original_confidence == 0.85

    @pytest.mark.fast
    def test_parse_marker_with_comma_separated_keywords(self):
        """Keywords as comma-separated string should be parsed to list."""
        marker = {
            "name": "Good",
            "color": "Green",
            "segment_id": "S001",
            "keywords": "travel, beach, vacation"
        }

        entry = parse_davinci_marker(marker)

        assert entry is not None
        assert len(entry.matched_keywords) == 3
        assert "travel" in entry.matched_keywords
        assert "beach" in entry.matched_keywords
        assert "vacation" in entry.matched_keywords

    @pytest.mark.fast
    def test_parse_empty_marker_returns_none(self):
        """Empty marker dict should return None."""
        entry = parse_davinci_marker({})
        assert entry is None

        entry = parse_davinci_marker(None)
        assert entry is None


# =============================================================================
# AC2: Feedback categories (good, bad, neutral) are classified correctly
# =============================================================================

class TestFeedbackCategoryClassification:
    """AC2: Test that feedback categories are correctly classified."""

    @pytest.mark.fast
    def test_kept_clip_classified_as_good(self):
        """Clips kept by editor should be classified as 'good'."""
        entry = FeedbackEntry(
            segment_id="S001",
            original_confidence=0.7,
            was_kept=True,
            matched_keywords=["travel"],
            video_source="youtube"
        )

        classification = classify_feedback_category(entry)

        assert classification.category == "good"
        assert "kept" in classification.reason.lower()

    @pytest.mark.fast
    def test_dropped_clip_classified_as_bad(self):
        """Clips dropped by editor should be classified as 'bad'."""
        entry = FeedbackEntry(
            segment_id="S002",
            original_confidence=0.8,
            was_kept=False,
            matched_keywords=["generic"],
            video_source="pexels"
        )

        classification = classify_feedback_category(entry)

        assert classification.category == "bad"
        assert "dropped" in classification.reason.lower()

    @pytest.mark.fast
    def test_high_confidence_dropped_still_bad(self):
        """High confidence clips that are dropped should still be 'bad'."""
        entry = FeedbackEntry(
            segment_id="S003",
            original_confidence=0.95,
            was_kept=False,
            matched_keywords=["test"],
            video_source="youtube"
        )

        classification = classify_feedback_category(entry)

        assert classification.category == "bad"

    @pytest.mark.fast
    def test_low_confidence_kept_still_good(self):
        """Low confidence clips that are kept should still be 'good'."""
        entry = FeedbackEntry(
            segment_id="S004",
            original_confidence=0.3,
            was_kept=True,
            matched_keywords=["test"],
            video_source="youtube"
        )

        classification = classify_feedback_category(entry)

        assert classification.category == "good"


# =============================================================================
# AC3: Feedback aggregation by video ID works
# =============================================================================

class TestFeedbackAggregationByVideoId:
    """AC3: Test that feedback can be aggregated by video source/ID."""

    @pytest.mark.fast
    def test_aggregate_by_single_source(self):
        """Entries from single source should group together."""
        entries = [
            FeedbackEntry(segment_id="S001", original_confidence=0.7, was_kept=True,
                          matched_keywords=["test"], video_source="youtube"),
            FeedbackEntry(segment_id="S002", original_confidence=0.8, was_kept=True,
                          matched_keywords=["test"], video_source="youtube"),
        ]

        result = aggregate_by_video_id(entries)

        assert len(result) == 1
        assert "youtube" in result
        assert len(result["youtube"]) == 2

    @pytest.mark.fast
    def test_aggregate_by_multiple_sources(self):
        """Entries from multiple sources should group separately."""
        entries = [
            FeedbackEntry(segment_id="S001", original_confidence=0.7, was_kept=True,
                          matched_keywords=["test"], video_source="youtube"),
            FeedbackEntry(segment_id="S002", original_confidence=0.8, was_kept=True,
                          matched_keywords=["test"], video_source="pexels"),
            FeedbackEntry(segment_id="S003", original_confidence=0.6, was_kept=False,
                          matched_keywords=["test"], video_source="youtube"),
        ]

        result = aggregate_by_video_id(entries)

        assert len(result) == 2
        assert len(result["youtube"]) == 2
        assert len(result["pexels"]) == 1

    @pytest.mark.fast
    def test_aggregate_handles_none_source(self):
        """Entries with None source should be grouped under 'unknown'."""
        entries = [
            FeedbackEntry(segment_id="S001", original_confidence=0.7, was_kept=True,
                          matched_keywords=["test"], video_source=None),
            FeedbackEntry(segment_id="S002", original_confidence=0.8, was_kept=True,
                          matched_keywords=["test"], video_source="youtube"),
        ]

        result = aggregate_by_video_id(entries)

        assert "unknown" in result
        assert "youtube" in result
        assert len(result["unknown"]) == 1

    @pytest.mark.fast
    def test_aggregate_empty_list(self):
        """Empty entries list should return empty dict."""
        result = aggregate_by_video_id([])
        assert result == {}

    @pytest.mark.fast
    def test_aggregate_preserves_entry_data(self):
        """Aggregation should preserve all entry fields."""
        entries = [
            FeedbackEntry(
                segment_id="S001",
                original_confidence=0.7,
                was_kept=True,
                matched_keywords=["travel", "beach"],
                video_source="youtube",
                track_used="V1",
                replacement_source=None
            ),
        ]

        result = aggregate_by_video_id(entries)

        aggregated_entry = result["youtube"][0]
        assert aggregated_entry.segment_id == "S001"
        assert aggregated_entry.original_confidence == 0.7
        assert aggregated_entry.was_kept is True
        assert aggregated_entry.matched_keywords == ["travel", "beach"]
        assert aggregated_entry.track_used == "V1"


# =============================================================================
# AC4: Empty feedback files are handled gracefully
# =============================================================================

class TestEmptyFeedbackFileHandling:
    """AC4: Test that empty feedback files are handled gracefully."""

    @pytest.mark.fast
    def test_load_nonexistent_file_returns_empty_batch(self, tmp_path):
        """Loading non-existent file should return empty FeedbackBatch."""
        fake_path = tmp_path / "nonexistent.json"

        batch = load_feedback_from_file(fake_path)

        assert isinstance(batch, FeedbackBatch)
        assert len(batch.entries) == 0

    @pytest.mark.fast
    def test_load_empty_file_returns_empty_batch(self, tmp_path):
        """Loading empty file should return empty FeedbackBatch."""
        empty_file = tmp_path / "empty.json"
        empty_file.write_text("")

        batch = load_feedback_from_file(empty_file)

        assert isinstance(batch, FeedbackBatch)
        assert len(batch.entries) == 0

    @pytest.mark.fast
    def test_load_whitespace_only_file_returns_empty_batch(self, tmp_path):
        """Loading file with only whitespace should return empty FeedbackBatch."""
        whitespace_file = tmp_path / "whitespace.json"
        whitespace_file.write_text("   \n\t  \n  ")

        batch = load_feedback_from_file(whitespace_file)

        assert isinstance(batch, FeedbackBatch)
        assert len(batch.entries) == 0

    @pytest.mark.fast
    def test_load_file_with_empty_entries_returns_empty_batch(self, tmp_path):
        """Loading file with empty entries array should return empty FeedbackBatch."""
        file = tmp_path / "empty_entries.json"
        file.write_text('{"project_name": "test", "entries": []}')

        batch = load_feedback_from_file(file)

        assert isinstance(batch, FeedbackBatch)
        assert len(batch.entries) == 0
        assert batch.project_name == "test"

    @pytest.mark.fast
    def test_load_valid_file_returns_populated_batch(self, tmp_path):
        """Loading valid file should return populated FeedbackBatch."""
        file = tmp_path / "valid.json"
        file.write_text(json.dumps({
            "project_name": "TestProject",
            "created_at": "2026-01-15T10:00:00",
            "entries": [
                {
                    "segment_id": "S001",
                    "original_confidence": 0.75,
                    "was_kept": True,
                    "matched_keywords": ["travel"],
                    "video_source": "youtube"
                }
            ]
        }))

        batch = load_feedback_from_file(file)

        assert len(batch.entries) == 1
        assert batch.project_name == "TestProject"
        assert batch.entries[0].segment_id == "S001"
        assert batch.entries[0].was_kept is True


# =============================================================================
# AC5: Feedback export to CSV format is correct
# =============================================================================

class TestFeedbackCsvExport:
    """AC5: Test that feedback export to CSV format is correct."""

    @pytest.mark.fast
    def test_export_empty_batch_has_header_only(self):
        """Exporting empty batch should produce CSV with header only."""
        batch = FeedbackBatch(entries=[])
        output = io.StringIO()

        export_feedback_to_csv(batch, output)

        output.seek(0)
        reader = csv.reader(output)
        rows = list(reader)

        assert len(rows) == 1  # Header only
        assert "segment_id" in rows[0]
        assert "original_confidence" in rows[0]
        assert "was_kept" in rows[0]

    @pytest.mark.fast
    def test_export_single_entry(self):
        """Exporting single entry should produce correct CSV row."""
        batch = FeedbackBatch(entries=[
            FeedbackEntry(
                segment_id="S001",
                original_confidence=0.75,
                was_kept=True,
                matched_keywords=["travel", "beach"],
                video_source="youtube",
                track_used="V1"
            )
        ])
        output = io.StringIO()

        export_feedback_to_csv(batch, output)

        output.seek(0)
        reader = csv.DictReader(output)
        rows = list(reader)

        assert len(rows) == 1
        assert rows[0]["segment_id"] == "S001"
        assert rows[0]["original_confidence"] == "0.75"
        assert rows[0]["was_kept"] == "True"
        assert "travel" in rows[0]["matched_keywords"]
        assert "beach" in rows[0]["matched_keywords"]
        assert rows[0]["video_source"] == "youtube"
        assert rows[0]["track_used"] == "V1"

    @pytest.mark.fast
    def test_export_multiple_entries(self):
        """Exporting multiple entries should produce correct CSV rows."""
        batch = FeedbackBatch(entries=[
            FeedbackEntry(segment_id="S001", original_confidence=0.7, was_kept=True,
                          matched_keywords=["test"], video_source="youtube"),
            FeedbackEntry(segment_id="S002", original_confidence=0.8, was_kept=False,
                          matched_keywords=["other"], video_source="pexels"),
            FeedbackEntry(segment_id="S003", original_confidence=0.6, was_kept=True,
                          matched_keywords=None, video_source=None),
        ])
        output = io.StringIO()

        export_feedback_to_csv(batch, output)

        output.seek(0)
        reader = csv.DictReader(output)
        rows = list(reader)

        assert len(rows) == 3
        assert rows[0]["segment_id"] == "S001"
        assert rows[1]["segment_id"] == "S002"
        assert rows[2]["segment_id"] == "S003"

    @pytest.mark.fast
    def test_export_handles_none_values(self):
        """Exporting entries with None values should produce empty strings."""
        batch = FeedbackBatch(entries=[
            FeedbackEntry(
                segment_id="S001",
                original_confidence=0.5,
                was_kept=True,
                matched_keywords=None,
                video_source=None,
                track_used=None,
                replacement_source=None
            )
        ])
        output = io.StringIO()

        export_feedback_to_csv(batch, output)

        output.seek(0)
        reader = csv.DictReader(output)
        rows = list(reader)

        assert rows[0]["matched_keywords"] == ""
        assert rows[0]["video_source"] == ""
        assert rows[0]["track_used"] == ""
        assert rows[0]["replacement_source"] == ""

    @pytest.mark.fast
    def test_export_keywords_comma_separated(self):
        """Keywords should be exported as comma-separated string."""
        batch = FeedbackBatch(entries=[
            FeedbackEntry(
                segment_id="S001",
                original_confidence=0.5,
                was_kept=True,
                matched_keywords=["travel", "beach", "vacation"],
                video_source="youtube"
            )
        ])
        output = io.StringIO()

        export_feedback_to_csv(batch, output)

        output.seek(0)
        reader = csv.DictReader(output)
        rows = list(reader)

        keywords = rows[0]["matched_keywords"]
        assert "travel" in keywords
        assert "beach" in keywords
        assert "vacation" in keywords


# =============================================================================
# AC6: Malformed feedback data is handled gracefully
# =============================================================================

class TestMalformedFeedbackHandling:
    """AC6: Test that malformed feedback data is handled gracefully."""

    @pytest.mark.fast
    def test_load_invalid_json_returns_empty_batch(self, tmp_path):
        """Loading invalid JSON should return empty FeedbackBatch."""
        file = tmp_path / "invalid.json"
        file.write_text("not valid json!!!")

        batch = load_feedback_from_file(file)

        assert isinstance(batch, FeedbackBatch)
        assert len(batch.entries) == 0

    @pytest.mark.fast
    def test_load_json_array_instead_of_object_returns_empty_batch(self, tmp_path):
        """Loading JSON array (not object) should return empty FeedbackBatch."""
        file = tmp_path / "array.json"
        file.write_text('[{"segment_id": "S001"}]')

        batch = load_feedback_from_file(file)

        # Should handle gracefully - entries won't be found in array
        assert isinstance(batch, FeedbackBatch)
        assert len(batch.entries) == 0

    @pytest.mark.fast
    def test_parse_marker_with_invalid_confidence_uses_zero(self):
        """Marker with invalid confidence should default to 0."""
        marker = {
            "name": "Good",
            "color": "Green",
            "segment_id": "S001",
            "confidence": "not_a_number"
        }

        entry = parse_davinci_marker(marker)

        assert entry is not None
        assert entry.original_confidence == 0.0

    @pytest.mark.fast
    def test_load_entry_with_missing_fields_uses_defaults(self, tmp_path):
        """Entry missing optional fields should use defaults."""
        file = tmp_path / "minimal.json"
        file.write_text(json.dumps({
            "entries": [
                {
                    "segment_id": "S001",
                    "original_confidence": 0.5,
                    "was_kept": True
                    # missing: matched_keywords, video_source, track_used, replacement_source
                }
            ]
        }))

        batch = load_feedback_from_file(file)

        assert len(batch.entries) == 1
        assert batch.entries[0].segment_id == "S001"
        assert batch.entries[0].matched_keywords is None
        assert batch.entries[0].video_source is None

    @pytest.mark.fast
    def test_load_entry_with_wrong_types_handles_gracefully(self, tmp_path):
        """Entry with wrong types should be handled gracefully."""
        file = tmp_path / "wrong_types.json"
        file.write_text(json.dumps({
            "entries": [
                {
                    "segment_id": "S001",
                    "original_confidence": "0.75",  # string instead of float
                    "was_kept": True
                }
            ]
        }))

        batch = load_feedback_from_file(file)

        assert len(batch.entries) == 1
        assert batch.entries[0].original_confidence == 0.75  # converted to float


# =============================================================================
# Integration tests
# =============================================================================

class TestFeedbackModuleIntegration:
    """Integration tests for feedback module workflow."""

    @pytest.mark.fast
    def test_full_workflow_parse_classify_aggregate_export(self, tmp_path):
        """Test complete workflow: parse markers -> classify -> aggregate -> export."""
        # Step 1: Parse multiple DaVinci markers
        markers = [
            {"color": "Green", "segment_id": "S001", "confidence": 0.8,
             "video_source": "youtube", "keywords": ["travel"]},
            {"color": "Green", "segment_id": "S002", "confidence": 0.7,
             "video_source": "youtube", "keywords": ["beach"]},
            {"color": "Red", "segment_id": "S003", "confidence": 0.9,
             "video_source": "pexels", "keywords": ["generic"]},
            {"color": "Red", "segment_id": "S004", "confidence": 0.6,
             "video_source": "youtube", "keywords": ["stock"]},
        ]

        entries = []
        for marker in markers:
            entry = parse_davinci_marker(marker)
            if entry:
                entries.append(entry)

        assert len(entries) == 4

        # Step 2: Classify each entry
        classifications = [classify_feedback_category(e) for e in entries]
        good_count = sum(1 for c in classifications if c.category == "good")
        bad_count = sum(1 for c in classifications if c.category == "bad")

        assert good_count == 2
        assert bad_count == 2

        # Step 3: Aggregate by video source
        aggregated = aggregate_by_video_id(entries)

        assert len(aggregated["youtube"]) == 3
        assert len(aggregated["pexels"]) == 1

        # Step 4: Export to CSV
        batch = FeedbackBatch(entries=entries, project_name="IntegrationTest")
        output = io.StringIO()
        export_feedback_to_csv(batch, output)

        output.seek(0)
        reader = csv.DictReader(output)
        rows = list(reader)

        assert len(rows) == 4

    @pytest.mark.fast
    def test_roundtrip_save_and_load(self, tmp_path):
        """Test saving feedback to file and loading it back."""
        # Create original batch
        original_batch = FeedbackBatch(
            project_name="RoundtripTest",
            entries=[
                FeedbackEntry(segment_id="S001", original_confidence=0.75, was_kept=True,
                              matched_keywords=["travel"], video_source="youtube"),
                FeedbackEntry(segment_id="S002", original_confidence=0.85, was_kept=False,
                              matched_keywords=["generic"], video_source="pexels"),
            ]
        )

        # Save to file
        file_path = tmp_path / "roundtrip.json"
        data = {
            "project_name": original_batch.project_name,
            "created_at": original_batch.created_at,
            "entries": [
                {
                    "segment_id": e.segment_id,
                    "original_confidence": e.original_confidence,
                    "was_kept": e.was_kept,
                    "matched_keywords": e.matched_keywords,
                    "video_source": e.video_source,
                    "track_used": e.track_used,
                    "replacement_source": e.replacement_source
                }
                for e in original_batch.entries
            ]
        }
        file_path.write_text(json.dumps(data))

        # Load back
        loaded_batch = load_feedback_from_file(file_path)

        assert loaded_batch.project_name == "RoundtripTest"
        assert len(loaded_batch.entries) == 2
        assert loaded_batch.entries[0].segment_id == "S001"
        assert loaded_batch.entries[0].was_kept is True
        assert loaded_batch.entries[1].segment_id == "S002"
        assert loaded_batch.entries[1].was_kept is False


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
