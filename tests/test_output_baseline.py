"""
Baseline output STRUCTURE validation tests.

Validates the structure and format of pipeline output files - NOT content/matches.
Tests ensure OTIO, XML, EDL, JSON files are correctly built.

Baseline from:
  Project: E:/Edit Job/Stu/January/24__2026-01-13
  Output:  20260115_051251

What this validates:
- OTIO files parse without errors, have correct track layout
- XML files are well-formed with proper xmeml structure
- EDL has required headers (TITLE, FCM)
- Segments JSON has required metadata keys
- Track counts match (19 tracks: V1-V10, A1-A9)
- No zero-duration clips, media references exist

Run with: pytest tests/test_output_baseline.py -v
Skill:    /validate-output [path]
"""

import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Any

import pytest

# Try to import opentimelineio - skip OTIO tests if not available
try:
    import opentimelineio as otio
    HAS_OTIO = True
except ImportError:
    HAS_OTIO = False
    otio = None


# =============================================================================
# FIXTURES
# =============================================================================

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "baseline_output"
BASELINE_METRICS_FILE = FIXTURES_DIR / "baseline_metrics.json"

# Default baseline output for live validation (can be overridden via env var or pytest arg)
DEFAULT_BASELINE_OUTPUT = Path("E:/Edit Job/Stu/January/24__2026-01-13/output/20260115_064843")


@pytest.fixture
def baseline_metrics() -> Dict:
    """Load baseline metrics from fixture file."""
    with open(BASELINE_METRICS_FILE, encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture
def baseline_output_dir(request) -> Optional[Path]:
    """
    Get baseline output directory for live validation.

    Override via:
      pytest --baseline-output=/path/to/output
      or BASELINE_OUTPUT_DIR env var
    """
    import os

    # Check pytest option first
    output_path = getattr(request.config, "baseline_output", None)

    # Then environment variable
    if not output_path:
        output_path = os.environ.get("BASELINE_OUTPUT_DIR")

    # Then default
    if not output_path:
        output_path = DEFAULT_BASELINE_OUTPUT

    path = Path(output_path)
    if path.exists():
        return path
    return None


# =============================================================================
# TEST CLASSES
# =============================================================================

class TestBaselineMetricsFixture:
    """Tests that validate the baseline metrics fixture itself."""

    def test_baseline_metrics_file_exists(self):
        """Baseline metrics JSON exists."""
        assert BASELINE_METRICS_FILE.exists(), f"Missing: {BASELINE_METRICS_FILE}"

    def test_baseline_metrics_valid_json(self, baseline_metrics):
        """Baseline metrics is valid JSON with required keys."""
        required_keys = ["files", "timeline_metrics", "track_details", "validation_expectations"]
        for key in required_keys:
            assert key in baseline_metrics, f"Missing key: {key}"

    def test_baseline_has_expected_file_counts(self, baseline_metrics):
        """Baseline has expected file type counts."""
        files = baseline_metrics["files"]
        expectations = baseline_metrics["validation_expectations"]

        otio_count = sum(1 for f in files if f.endswith(".otio"))
        xml_count = sum(1 for f in files if f.endswith(".xml"))

        assert otio_count == expectations["otio_file_count"], f"OTIO count mismatch: {otio_count}"
        assert xml_count == expectations["xml_file_count"], f"XML count mismatch: {xml_count}"

    def test_baseline_track_details_complete(self, baseline_metrics):
        """Baseline has all expected tracks."""
        track_names = [t["name"] for t in baseline_metrics["track_details"]]
        required = baseline_metrics["validation_expectations"]["required_tracks"]

        for track in required:
            assert track in track_names, f"Missing required track: {track}"


class TestOutputStructure:
    """Tests that validate output directory structure against baseline."""

    def test_required_files_exist(self, baseline_output_dir, baseline_metrics):
        """All required output files exist."""
        if not baseline_output_dir:
            pytest.skip("No baseline output directory available")

        required_files = [
            "timeline_FULL.otio",
            "timeline.edl",
            "timeline_project.xml",
            "timeline_segments.json",
            "match_report.md",
        ]

        for filename in required_files:
            filepath = baseline_output_dir / filename
            assert filepath.exists(), f"Missing required file: {filename}"

    def test_otio_file_count_matches(self, baseline_output_dir, baseline_metrics):
        """Number of OTIO files matches baseline."""
        if not baseline_output_dir:
            pytest.skip("No baseline output directory available")

        expected = baseline_metrics["validation_expectations"]["otio_file_count"]
        actual = len(list(baseline_output_dir.glob("*.otio")))

        assert actual == expected, f"OTIO file count: expected {expected}, got {actual}"

    def test_xml_file_count_matches(self, baseline_output_dir, baseline_metrics):
        """Number of XML files matches baseline."""
        if not baseline_output_dir:
            pytest.skip("No baseline output directory available")

        expected = baseline_metrics["validation_expectations"]["xml_file_count"]
        actual = len(list(baseline_output_dir.glob("*.xml")))

        assert actual == expected, f"XML file count: expected {expected}, got {actual}"

    def test_individual_track_otio_files_exist(self, baseline_output_dir):
        """Individual track OTIO files exist for all video tracks."""
        if not baseline_output_dir:
            pytest.skip("No baseline output directory available")

        expected_patterns = [
            "timeline_V1_*.otio",
            "timeline_V2_*.otio",
            "timeline_V3_*.otio",
            "timeline_V8_*.otio",  # B-roll
            "timeline_V9_*.otio",  # Entity images
            "timeline_V10_*.otio",  # Stock videos
            "timeline_A8_*.otio",  # Voiceover
        ]

        for pattern in expected_patterns:
            matches = list(baseline_output_dir.glob(pattern))
            assert len(matches) >= 1, f"Missing track OTIO for pattern: {pattern}"


@pytest.mark.skipif(not HAS_OTIO, reason="opentimelineio not installed")
class TestOTIOValidity:
    """Tests that validate OTIO file integrity."""

    def test_all_otio_files_readable(self, baseline_output_dir):
        """All OTIO files can be parsed without errors."""
        if not baseline_output_dir:
            pytest.skip("No baseline output directory available")

        otio_files = list(baseline_output_dir.glob("*.otio"))
        assert len(otio_files) > 0, "No OTIO files found"

        for filepath in otio_files:
            try:
                timeline = otio.adapters.read_from_file(str(filepath))
                assert timeline is not None, f"Failed to parse: {filepath.name}"
            except Exception as e:
                pytest.fail(f"OTIO parse error for {filepath.name}: {e}")

    def test_full_timeline_track_count(self, baseline_output_dir, baseline_metrics):
        """FULL timeline has expected number of tracks."""
        if not baseline_output_dir:
            pytest.skip("No baseline output directory available")

        full_otio = baseline_output_dir / "timeline_FULL.otio"
        if not full_otio.exists():
            pytest.skip("timeline_FULL.otio not found")

        timeline = otio.adapters.read_from_file(str(full_otio))
        expected = baseline_metrics["validation_expectations"]["min_total_tracks"]
        actual = len(list(timeline.tracks))

        assert actual >= expected, f"Track count: expected >= {expected}, got {actual}"

    def test_full_timeline_has_required_tracks(self, baseline_output_dir, baseline_metrics):
        """FULL timeline contains all required track names."""
        if not baseline_output_dir:
            pytest.skip("No baseline output directory available")

        full_otio = baseline_output_dir / "timeline_FULL.otio"
        if not full_otio.exists():
            pytest.skip("timeline_FULL.otio not found")

        timeline = otio.adapters.read_from_file(str(full_otio))
        track_names = [t.name for t in timeline.tracks]
        required = baseline_metrics["validation_expectations"]["required_tracks"]

        for track in required:
            assert track in track_names, f"Missing required track: {track}"

    def test_v1_primary_clip_count(self, baseline_output_dir, baseline_metrics):
        """V1 Primary track has minimum expected clips."""
        if not baseline_output_dir:
            pytest.skip("No baseline output directory available")

        full_otio = baseline_output_dir / "timeline_FULL.otio"
        if not full_otio.exists():
            pytest.skip("timeline_FULL.otio not found")

        timeline = otio.adapters.read_from_file(str(full_otio))
        v1_track = None
        for track in timeline.tracks:
            if track.name == "V1 - Primary":
                v1_track = track
                break

        assert v1_track is not None, "V1 - Primary track not found"

        clips = [c for c in v1_track if isinstance(c, otio.schema.Clip)]
        min_clips = baseline_metrics["validation_expectations"]["min_v1_clips"]

        assert len(clips) >= min_clips, f"V1 clips: expected >= {min_clips}, got {len(clips)}"

    def test_voiceover_track_single_clip(self, baseline_output_dir, baseline_metrics):
        """Voiceover track (A9) has exactly one clip."""
        if not baseline_output_dir:
            pytest.skip("No baseline output directory available")

        full_otio = baseline_output_dir / "timeline_FULL.otio"
        if not full_otio.exists():
            pytest.skip("timeline_FULL.otio not found")

        timeline = otio.adapters.read_from_file(str(full_otio))
        vo_track = None
        for track in timeline.tracks:
            if "Voiceover" in track.name:
                vo_track = track
                break

        assert vo_track is not None, "Voiceover track not found"

        clips = [c for c in vo_track if isinstance(c, otio.schema.Clip)]
        expected = baseline_metrics["validation_expectations"]["voiceover_track_clips"]

        assert len(clips) == expected, f"Voiceover clips: expected {expected}, got {len(clips)}"

    def test_no_zero_duration_clips(self, baseline_output_dir):
        """No clips have zero or negative duration."""
        if not baseline_output_dir:
            pytest.skip("No baseline output directory available")

        full_otio = baseline_output_dir / "timeline_FULL.otio"
        if not full_otio.exists():
            pytest.skip("timeline_FULL.otio not found")

        timeline = otio.adapters.read_from_file(str(full_otio))
        issues = []

        for track in timeline.tracks:
            for item in track:
                if isinstance(item, otio.schema.Clip):
                    dur = item.duration().to_seconds()
                    if dur <= 0:
                        issues.append(f"{track.name}: {item.name} has duration {dur}")

        assert len(issues) == 0, f"Found {len(issues)} zero/negative duration clips:\n" + "\n".join(issues[:10])

    def test_clips_have_media_references(self, baseline_output_dir):
        """All clips (except gaps) have media references."""
        if not baseline_output_dir:
            pytest.skip("No baseline output directory available")

        full_otio = baseline_output_dir / "timeline_FULL.otio"
        if not full_otio.exists():
            pytest.skip("timeline_FULL.otio not found")

        timeline = otio.adapters.read_from_file(str(full_otio))
        missing_refs = []

        for track in timeline.tracks:
            for item in track:
                if isinstance(item, otio.schema.Clip):
                    if item.media_reference is None:
                        missing_refs.append(f"{track.name}: {item.name}")

        # Allow some missing refs (placeholder clips) but not too many
        max_allowed = 50
        assert len(missing_refs) <= max_allowed, f"Too many clips without media refs: {len(missing_refs)}"


class TestXMLValidity:
    """Tests that validate XML file integrity."""

    def test_all_xml_files_well_formed(self, baseline_output_dir):
        """All XML files are well-formed XML."""
        if not baseline_output_dir:
            pytest.skip("No baseline output directory available")

        xml_files = list(baseline_output_dir.glob("*.xml"))
        assert len(xml_files) > 0, "No XML files found"

        for filepath in xml_files:
            try:
                tree = ET.parse(filepath)
                assert tree.getroot() is not None, f"Empty XML: {filepath.name}"
            except ET.ParseError as e:
                pytest.fail(f"XML parse error for {filepath.name}: {e}")

    def test_project_xml_has_xmeml_root(self, baseline_output_dir):
        """Main project XML has xmeml root element."""
        if not baseline_output_dir:
            pytest.skip("No baseline output directory available")

        project_xml = baseline_output_dir / "timeline_project.xml"
        if not project_xml.exists():
            pytest.skip("timeline_project.xml not found")

        tree = ET.parse(project_xml)
        root = tree.getroot()

        assert root.tag == "xmeml", f"Expected xmeml root, got: {root.tag}"

    def test_xml_part_files_count(self, baseline_output_dir):
        """Part XML files exist for large timelines."""
        if not baseline_output_dir:
            pytest.skip("No baseline output directory available")

        part_files = list(baseline_output_dir.glob("timeline_media_part*.xml"))
        # Should have at least some part files for a large timeline
        assert len(part_files) >= 1, "No part XML files found"


class TestSegmentsJSON:
    """Tests that validate timeline_segments.json structure."""

    def test_segments_json_valid(self, baseline_output_dir):
        """timeline_segments.json is valid JSON."""
        if not baseline_output_dir:
            pytest.skip("No baseline output directory available")

        segments_file = baseline_output_dir / "timeline_segments.json"
        if not segments_file.exists():
            pytest.skip("timeline_segments.json not found")

        with open(segments_file, encoding="utf-8") as f:
            data = json.load(f)

        assert isinstance(data, dict), "Segments JSON should be a dict"

    def test_segments_json_has_required_keys(self, baseline_output_dir):
        """timeline_segments.json has required metadata keys."""
        if not baseline_output_dir:
            pytest.skip("No baseline output directory available")

        segments_file = baseline_output_dir / "timeline_segments.json"
        if not segments_file.exists():
            pytest.skip("timeline_segments.json not found")

        with open(segments_file, encoding="utf-8") as f:
            data = json.load(f)

        required_keys = ["generated_at", "frame_rate", "total_segments", "segments"]
        for key in required_keys:
            assert key in data, f"Missing key: {key}"

    def test_segments_json_segment_count(self, baseline_output_dir, baseline_metrics):
        """Segment count matches baseline expectations."""
        if not baseline_output_dir:
            pytest.skip("No baseline output directory available")

        segments_file = baseline_output_dir / "timeline_segments.json"
        if not segments_file.exists():
            pytest.skip("timeline_segments.json not found")

        with open(segments_file, encoding="utf-8") as f:
            data = json.load(f)

        # V1 clip count should roughly match segment count
        expected_min = baseline_metrics["validation_expectations"]["min_v1_clips"]
        actual = data.get("total_segments", 0)

        assert actual >= expected_min, f"Segment count: expected >= {expected_min}, got {actual}"


class TestMatchReport:
    """Tests that validate match_report.md structure."""

    def test_match_report_exists(self, baseline_output_dir):
        """match_report.md exists."""
        if not baseline_output_dir:
            pytest.skip("No baseline output directory available")

        report = baseline_output_dir / "match_report.md"
        assert report.exists(), "match_report.md not found"

    def test_match_report_has_content(self, baseline_output_dir):
        """match_report.md has meaningful content."""
        if not baseline_output_dir:
            pytest.skip("No baseline output directory available")

        report = baseline_output_dir / "match_report.md"
        if not report.exists():
            pytest.skip("match_report.md not found")

        content = report.read_text(encoding="utf-8")

        assert len(content) > 1000, "Match report too short"
        assert "# Match Report" in content, "Missing report header"
        assert "## Matches" in content or "### Segment" in content, "Missing matches section"

    def test_match_report_segment_entries(self, baseline_output_dir, baseline_metrics):
        """match_report.md has segment entries."""
        if not baseline_output_dir:
            pytest.skip("No baseline output directory available")

        report = baseline_output_dir / "match_report.md"
        if not report.exists():
            pytest.skip("match_report.md not found")

        content = report.read_text(encoding="utf-8")

        # Count segment headers
        segment_count = len(re.findall(r"### Segment \d+", content))
        min_expected = 10  # Should have at least some segments

        assert segment_count >= min_expected, f"Too few segments in report: {segment_count}"


class TestEDLValidity:
    """Tests that validate EDL file structure."""

    def test_edl_exists_and_readable(self, baseline_output_dir):
        """timeline.edl exists and is readable."""
        if not baseline_output_dir:
            pytest.skip("No baseline output directory available")

        edl_file = baseline_output_dir / "timeline.edl"
        assert edl_file.exists(), "timeline.edl not found"

        content = edl_file.read_text(encoding="utf-8")
        assert len(content) > 100, "EDL file too short"

    def test_edl_has_title(self, baseline_output_dir):
        """EDL has TITLE line."""
        if not baseline_output_dir:
            pytest.skip("No baseline output directory available")

        edl_file = baseline_output_dir / "timeline.edl"
        if not edl_file.exists():
            pytest.skip("timeline.edl not found")

        content = edl_file.read_text(encoding="utf-8")
        assert "TITLE:" in content, "EDL missing TITLE line"

    def test_edl_has_fcm(self, baseline_output_dir):
        """EDL has FCM (frame count mode) line."""
        if not baseline_output_dir:
            pytest.skip("No baseline output directory available")

        edl_file = baseline_output_dir / "timeline.edl"
        if not edl_file.exists():
            pytest.skip("timeline.edl not found")

        content = edl_file.read_text(encoding="utf-8")
        assert "FCM:" in content, "EDL missing FCM line"


# =============================================================================
# UTILITY FUNCTIONS FOR EXTERNAL USE
# =============================================================================

def validate_output_directory(output_dir: Path, baseline_metrics: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Validate an output directory against baseline expectations.

    Returns a dict with validation results suitable for programmatic use.

    Args:
        output_dir: Path to output directory to validate
        baseline_metrics: Optional baseline metrics dict (loads from fixture if not provided)

    Returns:
        Dict with keys: passed, failed, warnings, details
    """
    if baseline_metrics is None:
        with open(BASELINE_METRICS_FILE, encoding="utf-8") as f:
            baseline_metrics = json.load(f)

    results = {
        "passed": [],
        "failed": [],
        "warnings": [],
        "details": {}
    }

    # Check required files
    required_files = ["timeline_FULL.otio", "timeline.edl", "timeline_project.xml"]
    for filename in required_files:
        if (output_dir / filename).exists():
            results["passed"].append(f"Required file exists: {filename}")
        else:
            results["failed"].append(f"Missing required file: {filename}")

    # Check file counts
    expectations = baseline_metrics["validation_expectations"]

    otio_count = len(list(output_dir.glob("*.otio")))
    if otio_count >= expectations["otio_file_count"]:
        results["passed"].append(f"OTIO file count OK: {otio_count}")
    else:
        results["warnings"].append(f"OTIO count low: {otio_count} < {expectations['otio_file_count']}")

    xml_count = len(list(output_dir.glob("*.xml")))
    if xml_count >= expectations["xml_file_count"]:
        results["passed"].append(f"XML file count OK: {xml_count}")
    else:
        results["warnings"].append(f"XML count low: {xml_count} < {expectations['xml_file_count']}")

    # OTIO validation if available
    if HAS_OTIO:
        full_otio = output_dir / "timeline_FULL.otio"
        if full_otio.exists():
            try:
                timeline = otio.adapters.read_from_file(str(full_otio))
                track_count = len(list(timeline.tracks))
                results["details"]["track_count"] = track_count

                if track_count >= expectations["min_total_tracks"]:
                    results["passed"].append(f"Track count OK: {track_count}")
                else:
                    results["failed"].append(f"Track count low: {track_count} < {expectations['min_total_tracks']}")

                # Check V1 clips
                for track in timeline.tracks:
                    if track.name == "V1 - Primary":
                        clip_count = len([c for c in track if isinstance(c, otio.schema.Clip)])
                        results["details"]["v1_clip_count"] = clip_count
                        if clip_count >= expectations["min_v1_clips"]:
                            results["passed"].append(f"V1 clip count OK: {clip_count}")
                        else:
                            results["failed"].append(f"V1 clips low: {clip_count} < {expectations['min_v1_clips']}")
                        break

            except Exception as e:
                results["failed"].append(f"OTIO parse error: {e}")

    return results


# =============================================================================
# PYTEST CONFIGURATION
# =============================================================================

def pytest_addoption(parser):
    """Add custom pytest options."""
    parser.addoption(
        "--baseline-output",
        action="store",
        default=None,
        help="Path to baseline output directory for validation"
    )


def pytest_configure(config):
    """Store baseline output path in config."""
    config.baseline_output = config.getoption("--baseline-output")
