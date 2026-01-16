"""
Output STRUCTURE validation - validates output format/integrity, NOT content.

Designed as a robust boilerplate that won't break with new features/tracks.

Validates:
- File format integrity (OTIO parses, XML well-formed, JSON valid, EDL structure)
- OTIO timeline integrity (no zero-duration clips, valid media refs, no overlaps)
- Track structure (required tracks exist, naming conventions)
- Coverage thresholds (percentage-based, not absolute counts)
- NLE importability (EDL CMX3600, XML XMEML, path compatibility)

Usage:
    python tests/test_output_comparison.py "E:/path/to/output" -v
    OUTPUT_DIR="E:/path/to/output" pytest tests/test_output_comparison.py -v
"""

import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field

import pytest

try:
    import opentimelineio as otio
    HAS_OTIO = True
except ImportError:
    HAS_OTIO = False
    otio = None


# =============================================================================
# STRUCTURAL EXPECTATIONS (extensible - add new tracks/files here)
# =============================================================================

STRUCTURE_SPEC = {
    # Required files (must exist)
    "required_files": [
        "timeline_FULL.otio",
        "timeline.edl",
        "timeline_project.xml",
        "match_report.md",
        "timeline_segments.json",
    ],

    # Minimum file counts by extension
    "min_file_counts": {
        ".otio": 10,  # FULL + individual tracks
        ".xml": 1,    # At least project XML
    },

    # Required tracks (must exist in timeline)
    "required_tracks": [
        "V1 - Primary",
        "A9 - Voiceover",
    ],

    # Track naming patterns (regex) - validates track naming convention
    "track_patterns": [
        r"^V\d+ - .+$",   # Video tracks: V1 - Name, V2 - Name, etc.
        r"^A\d+ - .+$",   # Audio tracks: A1 - Name, A2 - Name, etc.
    ],

    # Coverage thresholds (percentage) - tracks must meet these minimums
    "coverage_thresholds": {
        "V1 - Primary": 90.0,       # Primary must have high coverage
        "V2 - Alternative 1": 50.0,  # Alternatives can have gaps
        "V3 - Alternative 2": 40.0,
        "V8 - B-roll Only": 20.0,    # B-roll may be sparse
    },

    # Segments JSON required keys
    "segments_required_keys": ["generated_at", "total_segments", "segments"],
    "segment_entry_keys": ["id", "start_frame", "end_frame", "voiceover_text"],

    # EDL required sections
    "edl_required_patterns": [
        r"TITLE:",
        r"FCM:",
    ],
}

# =============================================================================
# NLE IMPORTABILITY CONSTANTS
# =============================================================================

# Windows reserved filenames (case-insensitive)
WINDOWS_RESERVED_NAMES = {
    'CON', 'PRN', 'AUX', 'NUL',
    'COM1', 'COM2', 'COM3', 'COM4', 'COM5', 'COM6', 'COM7', 'COM8', 'COM9',
    'LPT1', 'LPT2', 'LPT3', 'LPT4', 'LPT5', 'LPT6', 'LPT7', 'LPT8', 'LPT9',
}

# Characters that cause issues in different NLEs
NLE_PROBLEMATIC_CHARS = {
    'premiere': set('<>:"|?*'),  # Windows path + Premiere-specific
    'resolve': set('<>:"|?*[]'),  # DaVinci Resolve
    'fcpx': set(':'),  # Final Cut Pro X
    'avid': set('<>:"|?*&'),  # Avid Media Composer
    'common': set('<>:"|?*'),  # Common across all NLEs
}

# Valid EDL edit types (CMX3600)
EDL_VALID_EDIT_TYPES = {'C', 'D', 'W', 'K', 'B'}  # Cut, Dissolve, Wipe, Key, Black

# Valid frame rate values for NLEs
VALID_FRAME_RATES = {23.976, 24, 25, 29.97, 30, 50, 59.94, 60}

# XMEML required elements for different NLEs
XMEML_REQUIREMENTS = {
    'davinci_bin': {
        'required_under_xmeml': ['bin'],
        'required_sibling_to_bin': ['sequence'],  # Empty sequence triggers import
        'bin_children': ['name', 'children'],
    },
    'davinci_sequence': {
        'required_under_xmeml': ['sequence'],
        'sequence_children': ['name', 'duration', 'rate', 'media'],
    },
    'premiere': {
        'required_under_xmeml': ['project', 'sequence'],
    },
}


# =============================================================================
# RESULT CONTAINER
# =============================================================================

@dataclass
class ValidationResult:
    """Validation results with categories."""
    passed: List[str] = field(default_factory=list)
    failed: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        return len(self.failed) == 0

    def add_pass(self, msg: str):
        self.passed.append(msg)

    def add_fail(self, msg: str):
        self.failed.append(msg)

    def add_warn(self, msg: str):
        self.warnings.append(msg)

    def summary(self) -> str:
        lines = [
            "Validation Summary:",
            f"  Passed:   {len(self.passed)}",
            f"  Failed:   {len(self.failed)}",
            f"  Warnings: {len(self.warnings)}",
        ]
        if self.failed:
            lines.append("\nFailures:")
            for f in self.failed[:15]:
                lines.append(f"  [FAIL] {f}")
        if self.warnings:
            lines.append("\nWarnings:")
            for w in self.warnings[:10]:
                lines.append(f"  [WARN] {w}")
        return "\n".join(lines)

    def structure_report(self) -> str:
        """Generate NLE import structure report."""
        lines = [
            "",
            "=" * 60,
            "  STRUCTURE REPORT (NLE Import)",
            "=" * 60,
        ]

        # Timeline info
        duration = self.metrics.get("timeline_duration", 0)
        if duration > 0:
            mins, secs = divmod(duration, 60)
            hours, mins = divmod(mins, 60)
            if hours > 0:
                dur_str = f"{int(hours)}h {int(mins)}m {secs:.1f}s"
            else:
                dur_str = f"{int(mins)}m {secs:.1f}s"
            lines.append(f"  Duration: {dur_str} ({duration:.1f}s)")

        frame_rate = self.metrics.get("frame_rate", "Unknown")
        lines.append(f"  Frame Rate: {frame_rate}")

        # File counts
        lines.append("")
        lines.append("  Files:")
        lines.append(f"    OTIO: {self.metrics.get('.otio_count', 0)}")
        lines.append(f"    XML:  {self.metrics.get('.xml_count', 0)}")
        lines.append(f"    EDL entries: {self.metrics.get('edl_edit_count', 0)}")
        lines.append(f"    Segments: {self.metrics.get('segment_count', 0)}")

        # Track breakdown
        tracks = self.metrics.get("tracks", [])
        if tracks:
            lines.append("")
            lines.append("  Tracks:")
            lines.append("  " + "-" * 56)
            lines.append(f"  {'Track':<28} {'Clips':>8} {'Coverage':>10}")
            lines.append("  " + "-" * 56)

            video_tracks = [t for t in tracks if t.startswith("V")]
            audio_tracks = [t for t in tracks if t.startswith("A")]

            for track in video_tracks:
                clips = self.metrics.get(f"{track}_clips", 0)
                coverage = self.metrics.get(f"{track}_coverage", 0)
                lines.append(f"  {track:<28} {clips:>8} {coverage:>9.1f}%")

            if audio_tracks:
                lines.append("  " + "-" * 56)
                for track in audio_tracks:
                    clips = self.metrics.get(f"{track}_clips", 0)
                    coverage = self.metrics.get(f"{track}_coverage", 0)
                    lines.append(f"  {track:<28} {clips:>8} {coverage:>9.1f}%")

        # Media status
        lines.append("")
        lines.append("  Media References:")
        refs_checked = self.metrics.get("media_refs_checked", 0)
        refs_valid = self.metrics.get("media_refs_valid", 0)
        refs_missing = self.metrics.get("media_refs_missing", 0)

        if refs_checked > 0:
            pct = (refs_valid / refs_checked) * 100
            status = "[OK]" if refs_missing == 0 else "[!]"
            lines.append(f"    {status} {refs_valid}/{refs_checked} files found ({pct:.1f}%)")
            if refs_missing > 0:
                lines.append(f"    Missing: {refs_missing} files (may be in E:/v/ or cleaned up)")
        else:
            lines.append("    No media references checked")

        # Import readiness
        lines.append("")
        lines.append("  Import Readiness:")

        # Check for potential issues
        issues = []
        if self.metrics.get("long_paths", 0) > 0:
            issues.append(f"    [!] {self.metrics['long_paths']} paths exceed 200 chars")
        if self.metrics.get("unicode_paths", 0) > 0:
            issues.append(f"    [!] {self.metrics['unicode_paths']} paths have unicode chars")
        if self.metrics.get("reserved_names", 0) > 0:
            issues.append(f"    [!] {self.metrics['reserved_names']} reserved Windows filenames")
        if self.metrics.get("nle_problematic_chars", 0) > 0:
            issues.append(f"    [!] {self.metrics['nle_problematic_chars']} paths with NLE-problematic chars")
        if refs_missing > refs_checked * 0.5:
            issues.append(f"    [!] >50% media files not found on disk")

        # EDL compliance
        edl_issues = self.metrics.get("edl_issues", [])
        if edl_issues:
            for edl_issue in edl_issues[:3]:
                issues.append(f"    [!] EDL: {edl_issue}")

        # XML compliance
        xml_issues = self.metrics.get("xml_import_issues", [])
        if xml_issues:
            for xml_issue in xml_issues[:3]:
                issues.append(f"    [!] XML: {xml_issue}")

        if not issues:
            lines.append("    [OK] EDL: CMX3600 compliant")
            lines.append("    [OK] XML: DaVinci-compatible (bin + sequence)")
            lines.append("    [OK] Paths: No compatibility issues")
            lines.append("    [OK] Ready for NLE import")
        else:
            for issue in issues:
                lines.append(issue)

        # Frame rate consistency
        frame_rates = self.metrics.get("detected_frame_rates", set())
        if len(frame_rates) > 1:
            lines.append(f"    [!] Mixed frame rates detected: {frame_rates}")
        elif frame_rates:
            rate = list(frame_rates)[0]
            lines.append(f"    [OK] Consistent frame rate: {rate} fps")

        lines.append("=" * 60)
        return "\n".join(lines)


# =============================================================================
# VALIDATORS
# =============================================================================

class OutputValidator:
    """Validates pipeline output structure and integrity."""

    def __init__(self, spec: Dict = None):
        self.spec = spec or STRUCTURE_SPEC

    def validate(self, output_dir: Path) -> ValidationResult:
        """Run all validation checks on output directory."""
        result = ValidationResult()

        if not output_dir.exists():
            result.add_fail(f"Output directory does not exist: {output_dir}")
            return result

        # 1. File structure
        self._validate_files(output_dir, result)

        # 2. File format integrity
        self._validate_formats(output_dir, result)

        # 3. OTIO timeline integrity
        if HAS_OTIO:
            self._validate_otio(output_dir, result)

        # 4. Segments JSON structure
        self._validate_segments(output_dir, result)

        # 5. EDL structure
        self._validate_edl(output_dir, result)

        # 6. Cross-file consistency (frame rates, timecodes)
        self._validate_cross_file_consistency(output_dir, result)

        return result

    # -------------------------------------------------------------------------
    # File Structure Validation
    # -------------------------------------------------------------------------

    def _validate_files(self, output_dir: Path, result: ValidationResult):
        """Validate required files exist and minimum counts met."""

        # Required files
        for filename in self.spec["required_files"]:
            filepath = output_dir / filename
            if filepath.exists():
                result.add_pass(f"Required file exists: {filename}")
            else:
                result.add_fail(f"Missing required file: {filename}")

        # Minimum file counts
        for ext, min_count in self.spec["min_file_counts"].items():
            count = len(list(output_dir.glob(f"*{ext}")))
            result.metrics[f"{ext}_count"] = count
            if count >= min_count:
                result.add_pass(f"File count {ext}: {count} >= {min_count}")
            else:
                result.add_fail(f"File count {ext}: {count} < {min_count} minimum")

    # -------------------------------------------------------------------------
    # File Format Validation
    # -------------------------------------------------------------------------

    def _validate_formats(self, output_dir: Path, result: ValidationResult):
        """Validate file formats are valid/parseable with NLE import compatibility."""

        xml_import_issues = []
        detected_frame_rates = set()

        # XML files well-formed + NLE import validation
        for xml_file in output_dir.glob("*.xml"):
            try:
                tree = ET.parse(xml_file)
                result.add_pass(f"XML valid: {xml_file.name}")

                root = tree.getroot()

                # Validate XMEML structure
                if root.tag == 'xmeml':
                    self._validate_xmeml_structure(
                        tree, xml_file, result, xml_import_issues, detected_frame_rates
                    )

            except ET.ParseError as e:
                result.add_fail(f"XML parse error in {xml_file.name}: {e}")

        result.metrics["xml_import_issues"] = xml_import_issues
        result.metrics["detected_frame_rates"] = detected_frame_rates

        # JSON files valid
        for json_file in output_dir.glob("*.json"):
            try:
                with open(json_file, encoding="utf-8") as f:
                    json.load(f)
                result.add_pass(f"JSON valid: {json_file.name}")
            except json.JSONDecodeError as e:
                result.add_fail(f"JSON parse error in {json_file.name}: {e}")
            except UnicodeDecodeError:
                # Try with different encoding
                try:
                    with open(json_file, encoding="utf-8-sig") as f:
                        json.load(f)
                    result.add_pass(f"JSON valid: {json_file.name} (BOM)")
                except Exception as e:
                    result.add_fail(f"JSON encoding error in {json_file.name}: {e}")

    def _validate_xmeml_structure(
        self,
        tree: ET.ElementTree,
        xml_file: Path,
        result: ValidationResult,
        xml_import_issues: List[str],
        detected_frame_rates: Set[float]
    ):
        """Validate XMEML structure for NLE import compatibility."""
        root = tree.getroot()
        filename = xml_file.name

        # Determine XML type based on content
        is_media_bin = "media_part" in filename or "conflict" in filename
        is_project = "project" in filename and "media" not in filename
        has_bin = root.find("bin") is not None or root.find(".//bin") is not None
        has_sequence = root.find("sequence") is not None or root.find(".//sequence") is not None
        has_project = root.find("project") is not None

        # Media bin XMLs: validate DaVinci structure
        if is_media_bin or (has_bin and not has_project):
            bin_node = root.find("bin")
            sequence_node = root.find("sequence")

            if bin_node is not None and sequence_node is not None:
                result.add_pass(f"XML structure: {filename} has bin + sequence (DaVinci-compatible)")
            elif bin_node is not None:
                result.add_warn(f"XML structure: {filename} has bin but missing sequence sibling")
                xml_import_issues.append(f"{filename}: Missing sequence sibling for DaVinci import")
            else:
                # Check if bin is nested under project (wrong for DaVinci bin import)
                nested_bin = root.find(".//bin")
                if nested_bin is not None:
                    parent = root.find("project")
                    if parent is not None:
                        xml_import_issues.append(f"{filename}: bin nested under project (should be direct child of xmeml)")
                else:
                    result.add_fail(f"XML structure: {filename} missing <bin> element")

        # Extract and validate frame rates
        for timebase in root.iter("timebase"):
            try:
                rate = float(timebase.text)
                detected_frame_rates.add(rate)
            except (ValueError, TypeError):
                xml_import_issues.append(f"{filename}: Invalid timebase value: {timebase.text}")

        # Validate pathurl format in file elements
        invalid_paths = 0
        for file_elem in root.iter("file"):
            pathurl = file_elem.find("pathurl")
            if pathurl is not None and pathurl.text:
                url = pathurl.text
                # Check for common issues
                if '\\' in url:
                    invalid_paths += 1  # Should use forward slashes
                elif url.startswith('file:///') and not url[8:].startswith('/'):
                    # Windows path should be file:///C:/... not file:///C:\...
                    pass  # This is actually OK for Windows
                # Check for empty or placeholder paths
                elif not url.strip() or url in ['', 'null', 'None']:
                    invalid_paths += 1

        if invalid_paths > 0:
            xml_import_issues.append(f"{filename}: {invalid_paths} paths with backslashes or invalid format")

        # Validate duration values
        zero_durations = 0
        for duration_elem in root.iter("duration"):
            try:
                dur = int(duration_elem.text)
                if dur <= 0:
                    zero_durations += 1
            except (ValueError, TypeError):
                pass  # Non-integer durations handled elsewhere

        if zero_durations > 0:
            xml_import_issues.append(f"{filename}: {zero_durations} zero/negative duration elements")

        # Validate clip/file ID references
        self._validate_xml_id_references(root, filename, result, xml_import_issues)

        # Validate timecode format
        for timecode_elem in root.iter("timecode"):
            tc_string = timecode_elem.find("string")
            if tc_string is not None and tc_string.text:
                if not self._is_valid_timecode(tc_string.text):
                    xml_import_issues.append(f"{filename}: Invalid timecode format: {tc_string.text}")

    def _validate_xml_id_references(
        self,
        root: ET.Element,
        filename: str,
        result: ValidationResult,
        xml_import_issues: List[str]
    ):
        """Validate that file ID references in XML are consistent."""
        # Collect all file definitions (elements with id attribute and children)
        file_definitions = {}
        for file_elem in root.iter("file"):
            file_id = file_elem.get("id")
            if file_id and len(file_elem) > 0:  # Has children = definition
                file_definitions[file_id] = file_elem

        # Collect all file references (elements with id attribute but no children)
        file_references = []
        for file_elem in root.iter("file"):
            file_id = file_elem.get("id")
            if file_id and len(file_elem) == 0:  # No children = reference
                file_references.append(file_id)

        # Check for orphan references
        orphan_refs = [ref for ref in file_references if ref not in file_definitions]
        if orphan_refs:
            # This is only a warning because references might be to external files
            unique_orphans = list(set(orphan_refs))[:3]
            result.add_warn(f"XML {filename}: {len(orphan_refs)} file refs without definitions (may be external)")

    # -------------------------------------------------------------------------
    # OTIO Timeline Validation
    # -------------------------------------------------------------------------

    def _validate_otio(self, output_dir: Path, result: ValidationResult):
        """Validate OTIO timeline structure and integrity."""
        full_otio = output_dir / "timeline_FULL.otio"
        if not full_otio.exists():
            return

        try:
            timeline = otio.adapters.read_from_file(str(full_otio))
        except Exception as e:
            result.add_fail(f"OTIO parse error: {e}")
            return

        result.add_pass(f"OTIO parses successfully")
        result.metrics["timeline_duration"] = timeline.duration().to_seconds()
        result.metrics["track_count"] = len(list(timeline.tracks))

        # Extract frame rate from timeline
        try:
            rate = timeline.duration().rate
            result.metrics["frame_rate"] = f"{rate} fps"
        except Exception:
            result.metrics["frame_rate"] = "Unknown"

        # Track structure
        self._validate_tracks(timeline, result)

        # Clip integrity
        self._validate_clips(timeline, result)

        # Media references
        self._validate_media_refs(timeline, result)

        # Test export adapter capabilities
        self._validate_otio_export_capability(timeline, result)

    def _validate_tracks(self, timeline, result: ValidationResult):
        """Validate track structure and naming."""
        track_names = [t.name for t in timeline.tracks]
        result.metrics["tracks"] = track_names

        # Required tracks
        for req_track in self.spec["required_tracks"]:
            if req_track in track_names:
                result.add_pass(f"Required track present: {req_track}")
            else:
                result.add_fail(f"Missing required track: {req_track}")

        # Track naming convention
        patterns = self.spec["track_patterns"]
        for track_name in track_names:
            matches_pattern = any(re.match(p, track_name) for p in patterns)
            if not matches_pattern:
                result.add_warn(f"Track name doesn't match convention: {track_name}")

        # Coverage thresholds
        timeline_dur = timeline.duration().to_seconds()
        if timeline_dur <= 0:
            result.add_fail("Timeline has zero duration")
            return

        for track in timeline.tracks:
            clips = [c for c in track if isinstance(c, otio.schema.Clip)]
            clip_dur = sum(c.duration().to_seconds() for c in clips)
            coverage = (clip_dur / timeline_dur) * 100

            result.metrics[f"{track.name}_coverage"] = coverage
            result.metrics[f"{track.name}_clips"] = len(clips)

            # Check threshold if defined
            if track.name in self.spec["coverage_thresholds"]:
                threshold = self.spec["coverage_thresholds"][track.name]
                if coverage >= threshold:
                    result.add_pass(f"{track.name}: {coverage:.1f}% >= {threshold}%")
                elif coverage >= threshold * 0.8:  # 80% of threshold = warning
                    result.add_warn(f"{track.name}: {coverage:.1f}% below {threshold}%")
                else:
                    result.add_fail(f"{track.name}: {coverage:.1f}% far below {threshold}%")

    def _validate_clips(self, timeline, result: ValidationResult):
        """Validate clip integrity (no zero-duration, valid timing, no overlaps)."""
        zero_duration_count = 0
        negative_duration_count = 0
        total_clips = 0
        overlapping_clips = 0
        invalid_speed_effects = 0
        excessive_durations = 0

        # Constants for validation
        MAX_CLIP_DURATION = 24 * 60 * 60  # 24 hours in seconds
        MIN_SPEED = 0.1  # 10%
        MAX_SPEED = 10.0  # 1000%

        for track in timeline.tracks:
            prev_end = 0
            for item in track:
                if isinstance(item, otio.schema.Clip):
                    total_clips += 1
                    dur = item.duration().to_seconds()

                    # Duration checks
                    if dur <= 0:
                        zero_duration_count += 1
                    if dur < 0:
                        negative_duration_count += 1
                    if dur > MAX_CLIP_DURATION:
                        excessive_durations += 1

                    # Overlap detection (within same track)
                    try:
                        # Get clip range in timeline
                        clip_range = item.range_in_parent()
                        if clip_range:
                            start = clip_range.start_time.to_seconds()
                            if start < prev_end - 0.001:  # Allow 1ms tolerance
                                overlapping_clips += 1
                            prev_end = start + dur
                    except Exception:
                        pass  # Skip if range_in_parent not available

                    # Speed effect validation
                    if hasattr(item, 'effects'):
                        for effect in item.effects:
                            if hasattr(effect, 'time_scalar'):
                                scalar = effect.time_scalar
                                if scalar < MIN_SPEED or scalar > MAX_SPEED:
                                    invalid_speed_effects += 1

        result.metrics["total_clips"] = total_clips
        result.metrics["overlapping_clips"] = overlapping_clips

        if zero_duration_count == 0:
            result.add_pass(f"No zero-duration clips ({total_clips} total)")
        else:
            result.add_fail(f"{zero_duration_count} clips have zero/negative duration")

        if negative_duration_count > 0:
            result.add_fail(f"{negative_duration_count} clips have negative duration")

        if overlapping_clips > 0:
            result.add_warn(f"{overlapping_clips} clips overlap within tracks")

        if excessive_durations > 0:
            result.add_warn(f"{excessive_durations} clips exceed 24h duration (likely data error)")

        if invalid_speed_effects > 0:
            result.add_warn(f"{invalid_speed_effects} clips have speed effects outside 10%-1000% range")

    def _validate_media_refs(self, timeline, result: ValidationResult):
        """Validate media references exist and are accessible with NLE compatibility."""
        refs_checked = 0
        refs_valid = 0
        refs_missing = 0
        long_paths = 0
        unicode_paths = 0
        reserved_names = 0
        nle_problematic_chars = 0
        unc_paths = 0
        missing_samples = []
        problematic_samples = []

        for track in timeline.tracks:
            for item in track:
                if isinstance(item, otio.schema.Clip) and item.media_reference:
                    ref = item.media_reference
                    if hasattr(ref, 'target_url') and ref.target_url:
                        refs_checked += 1
                        url = ref.target_url

                        # Convert file:// URL to path
                        if url.startswith('file:///'):
                            path = url[8:]
                        elif url.startswith('file://'):
                            path = url[7:]
                        else:
                            path = url

                        # Normalize for Windows
                        path = path.replace('/', '\\')

                        # Check for potential import issues
                        if len(path) > 200:
                            long_paths += 1

                        # Unicode check
                        try:
                            path.encode('ascii')
                        except UnicodeEncodeError:
                            unicode_paths += 1

                        # Check for Windows reserved names
                        filename_stem = Path(path).stem.upper()
                        if filename_stem in WINDOWS_RESERVED_NAMES:
                            reserved_names += 1
                            if len(problematic_samples) < 3:
                                problematic_samples.append(f"Reserved: {Path(path).name}")

                        # Check for NLE-problematic characters in filename
                        filename = Path(path).name
                        problematic = NLE_PROBLEMATIC_CHARS['common']
                        if any(c in filename for c in problematic):
                            nle_problematic_chars += 1
                            if len(problematic_samples) < 3:
                                bad_chars = [c for c in filename if c in problematic]
                                problematic_samples.append(f"Bad chars '{bad_chars}': {filename[:30]}")

                        # Check for UNC paths (network paths)
                        if path.startswith('\\\\') or path.startswith('//'):
                            unc_paths += 1

                        # Check for overly long path components (some NLEs have 255 char limits per component)
                        for component in Path(path).parts:
                            if len(component) > 255:
                                long_paths += 1
                                break

                        if Path(path).exists():
                            refs_valid += 1
                        else:
                            refs_missing += 1
                            if len(missing_samples) < 3:
                                missing_samples.append(Path(path).name)

        result.metrics["media_refs_checked"] = refs_checked
        result.metrics["media_refs_valid"] = refs_valid
        result.metrics["media_refs_missing"] = refs_missing
        result.metrics["long_paths"] = long_paths
        result.metrics["unicode_paths"] = unicode_paths
        result.metrics["reserved_names"] = reserved_names
        result.metrics["nle_problematic_chars"] = nle_problematic_chars
        result.metrics["unc_paths"] = unc_paths

        if refs_checked == 0:
            result.add_warn("No media references to validate")
        elif refs_missing == 0:
            result.add_pass(f"All {refs_checked} media references valid")
        else:
            # Media refs on disk are NOT a structural requirement
            # Files may be stored elsewhere (E:/v/) or cleaned up
            # This is always a warning, not a failure
            pct_valid = (refs_valid / refs_checked) * 100
            result.add_warn(f"Media refs: {refs_valid}/{refs_checked} found on disk ({pct_valid:.1f}%)")
            if pct_valid < 50 and missing_samples:
                result.add_warn(f"  Sample missing: {', '.join(missing_samples[:3])}")

        # NLE compatibility warnings
        if reserved_names > 0:
            result.add_fail(f"{reserved_names} files use Windows reserved names (CON, PRN, etc.)")
        if nle_problematic_chars > 0:
            result.add_warn(f"{nle_problematic_chars} paths have NLE-problematic chars (<>:\"|?*)")
            if problematic_samples:
                result.add_warn(f"  Samples: {', '.join(problematic_samples[:2])}")
        if unc_paths > 0:
            result.add_warn(f"{unc_paths} UNC (network) paths - may not import in all NLEs")
        if long_paths > 0:
            result.add_warn(f"{long_paths} paths exceed 200 chars - may fail in some NLEs")
        if unicode_paths > 0:
            result.add_warn(f"{unicode_paths} paths have non-ASCII chars - may fail in older NLEs")

        # Check for duplicate files referenced from different paths
        # This causes DaVinci Resolve to hang during OTIO import
        self._check_duplicate_media_paths(timeline, result)

    def _check_duplicate_media_paths(self, timeline, result: ValidationResult):
        """
        Check for same file referenced from multiple paths.

        DaVinci Resolve hangs when importing OTIO with the same video file
        referenced from different paths (e.g., stock/video.mp4 and broll/video.mp4).
        """
        import os

        # Collect all paths: (filename, size) -> list of paths
        file_key_to_paths: Dict[Tuple[str, int], List[str]] = {}

        for track in timeline.tracks:
            for item in track:
                if isinstance(item, otio.schema.Clip) and item.media_reference:
                    ref = item.media_reference
                    if hasattr(ref, 'target_url') and ref.target_url:
                        url = ref.target_url

                        # Convert to path
                        if url.startswith('file:///'):
                            path = url[8:]
                        elif url.startswith('file://'):
                            path = url[7:]
                        else:
                            path = url

                        # Normalize slashes
                        path = path.replace('\\', '/')

                        if os.path.exists(path):
                            try:
                                filename = os.path.basename(path)
                                size = os.path.getsize(path)
                                key = (filename, size)

                                if key not in file_key_to_paths:
                                    file_key_to_paths[key] = []
                                if path not in file_key_to_paths[key]:
                                    file_key_to_paths[key].append(path)
                            except (OSError, IOError):
                                pass

        # Find duplicates
        duplicates = {k: v for k, v in file_key_to_paths.items() if len(v) > 1}

        if duplicates:
            total_dupe_files = len(duplicates)
            sample_file = list(duplicates.keys())[0][0]
            sample_paths = duplicates[list(duplicates.keys())[0]]
            result.add_fail(
                f"{total_dupe_files} files referenced from multiple paths (causes DaVinci hang). "
                f"Example: {sample_file} in {len(sample_paths)} locations"
            )
            result.metrics["duplicate_media_paths"] = total_dupe_files
        else:
            result.metrics["duplicate_media_paths"] = 0

    def _validate_otio_export_capability(self, timeline, result: ValidationResult):
        """Test OTIO export capabilities to common NLE formats."""
        import tempfile
        import os

        export_tests = []

        # Test CMX3600 EDL export
        try:
            with tempfile.NamedTemporaryFile(suffix='.edl', delete=False) as f:
                temp_edl = f.name
            otio.adapters.write_to_file(timeline, temp_edl, adapter_name='cmx_3600')
            # Verify file was created and has content
            if os.path.exists(temp_edl) and os.path.getsize(temp_edl) > 0:
                export_tests.append(('EDL (CMX3600)', True, None))
            else:
                export_tests.append(('EDL (CMX3600)', False, 'Empty output'))
            os.unlink(temp_edl)
        except Exception as e:
            export_tests.append(('EDL (CMX3600)', False, str(e)[:50]))

        # Test FCPXML export (if adapter available)
        try:
            with tempfile.NamedTemporaryFile(suffix='.fcpxml', delete=False) as f:
                temp_fcpxml = f.name
            otio.adapters.write_to_file(timeline, temp_fcpxml, adapter_name='fcpx_xml')
            if os.path.exists(temp_fcpxml) and os.path.getsize(temp_fcpxml) > 0:
                export_tests.append(('FCPXML', True, None))
            else:
                export_tests.append(('FCPXML', False, 'Empty output'))
            os.unlink(temp_fcpxml)
        except otio.exceptions.NoKnownAdapterForExtensionError:
            export_tests.append(('FCPXML', None, 'Adapter not installed'))
        except Exception as e:
            export_tests.append(('FCPXML', False, str(e)[:50]))

        # Report results
        successful = [t[0] for t in export_tests if t[1] is True]
        failed = [(t[0], t[2]) for t in export_tests if t[1] is False]
        unavailable = [t[0] for t in export_tests if t[1] is None]

        result.metrics["otio_export_capable"] = successful
        result.metrics["otio_export_failed"] = [f[0] for f in failed]

        if successful:
            result.add_pass(f"OTIO exports to: {', '.join(successful)}")
        if failed:
            for fmt, err in failed:
                result.add_warn(f"OTIO export failed for {fmt}: {err}")

    # -------------------------------------------------------------------------
    # Segments JSON Validation
    # -------------------------------------------------------------------------

    def _validate_segments(self, output_dir: Path, result: ValidationResult):
        """Validate segments JSON structure."""
        segments_file = output_dir / "timeline_segments.json"
        if not segments_file.exists():
            return

        try:
            with open(segments_file, encoding="utf-8") as f:
                data = json.load(f)
        except UnicodeDecodeError:
            try:
                with open(segments_file, encoding="utf-8-sig") as f:
                    data = json.load(f)
            except Exception as e:
                result.add_fail(f"Segments JSON encoding error: {e}")
                return
        except Exception as e:
            result.add_fail(f"Segments JSON parse error: {e}")
            return

        # Required top-level keys
        missing = [k for k in self.spec["segments_required_keys"] if k not in data]
        if not missing:
            result.add_pass("Segments JSON has required keys")
        else:
            result.add_fail(f"Segments JSON missing keys: {missing}")

        # Segment count
        segment_count = data.get("total_segments", 0)
        result.metrics["segment_count"] = segment_count
        if segment_count > 0:
            result.add_pass(f"Segments JSON contains {segment_count} segments")
        else:
            result.add_warn("Segments JSON has zero segments")

        # Segment entry structure
        segments = data.get("segments", [])
        if segments:
            sample = segments[0]
            entry_keys = self.spec["segment_entry_keys"]
            has_keys = sum(1 for k in entry_keys if k in sample)
            if has_keys >= len(entry_keys) * 0.8:  # 80% of expected keys
                result.add_pass("Segment entries have expected structure")
            else:
                result.add_warn(f"Segment entries missing some keys (has {has_keys}/{len(entry_keys)})")

    # -------------------------------------------------------------------------
    # EDL Validation (CMX3600 Compliance)
    # -------------------------------------------------------------------------

    def _validate_edl(self, output_dir: Path, result: ValidationResult):
        """Validate EDL file structure and CMX3600 compliance."""
        edl_file = output_dir / "timeline.edl"
        if not edl_file.exists():
            return

        try:
            with open(edl_file, encoding="utf-8") as f:
                content = f.read()
        except UnicodeDecodeError:
            try:
                with open(edl_file, encoding="latin-1") as f:
                    content = f.read()
            except Exception as e:
                result.add_fail(f"EDL encoding error: {e}")
                return

        edl_issues = []

        # Required patterns
        for pattern in self.spec["edl_required_patterns"]:
            if re.search(pattern, content):
                result.add_pass(f"EDL has {pattern.rstrip(':')}")
            else:
                result.add_fail(f"EDL missing {pattern.rstrip(':')}")

        # FCM (Frame Code Mode) validation
        # Match full FCM line: "FCM: NON-DROP FRAME" or "FCM: DROP FRAME"
        fcm_match = re.search(r'FCM:\s*(.+?)(?:\r?\n|$)', content)
        if fcm_match:
            fcm_mode = fcm_match.group(1).strip().upper()
            # Normalize: remove hyphens and extra spaces
            fcm_normalized = fcm_mode.replace('-', '').replace(' ', '')
            valid_fcm_normalized = {'DROPFRAME', 'NONDROPFRAME', 'DF', 'NDF'}
            if fcm_normalized not in valid_fcm_normalized:
                edl_issues.append(f"Unknown FCM mode: {fcm_mode}")

        # Parse edit entries for CMX3600 compliance
        # Format: EVENT# REEL CHANNEL EDITTYPE SOURCE_TC RECORD_TC
        edit_pattern = re.compile(
            r'^(\d{3})\s+'  # Event number (001-999)
            r'(\S+)\s+'     # Reel name
            r'([VA12]+)\s+' # Channel (V, A, A2, AA, etc.)
            r'([CDWKB])\s+' # Edit type
            r'(\d{2}:\d{2}:\d{2}[;:]\d{2})\s+'  # Source IN timecode
            r'(\d{2}:\d{2}:\d{2}[;:]\d{2})\s+'  # Source OUT timecode
            r'(\d{2}:\d{2}:\d{2}[;:]\d{2})\s+'  # Record IN timecode
            r'(\d{2}:\d{2}:\d{2}[;:]\d{2})',    # Record OUT timecode
            re.MULTILINE
        )

        edit_matches = list(edit_pattern.finditer(content))
        edit_count = len(edit_matches)
        result.metrics["edl_edit_count"] = edit_count

        if edit_count > 0:
            result.add_pass(f"EDL has {edit_count} edit entries")
        else:
            # Fallback: count basic edit lines
            basic_edit_count = len(re.findall(r'^\d{3}\s+', content, re.MULTILINE))
            result.metrics["edl_edit_count"] = basic_edit_count
            if basic_edit_count > 0:
                result.add_warn(f"EDL has {basic_edit_count} entries (non-standard format)")
            else:
                result.add_warn("EDL has no edit entries")
            return

        # Validate edit entries
        prev_record_out = None
        event_numbers = set()
        invalid_timecodes = 0
        invalid_edit_types = 0
        long_reel_names = 0
        timecode_discontinuities = 0

        for match in edit_matches:
            event_num, reel, channel, edit_type, src_in, src_out, rec_in, rec_out = match.groups()

            # Check for duplicate event numbers
            if event_num in event_numbers:
                edl_issues.append(f"Duplicate event number: {event_num}")
            event_numbers.add(event_num)

            # Validate edit type
            if edit_type not in EDL_VALID_EDIT_TYPES:
                invalid_edit_types += 1

            # Check reel name length (traditional limit is 8 chars, modern NLEs more flexible)
            if len(reel) > 32:
                long_reel_names += 1

            # Validate timecode format (HH:MM:SS:FF or HH:MM:SS;FF for drop frame)
            for tc in [src_in, src_out, rec_in, rec_out]:
                if not self._is_valid_timecode(tc):
                    invalid_timecodes += 1

            # Check for timeline discontinuities (gaps in record timecodes)
            if prev_record_out:
                if rec_in != prev_record_out:
                    timecode_discontinuities += 1
            prev_record_out = rec_out

        # Report findings
        if invalid_timecodes > 0:
            edl_issues.append(f"{invalid_timecodes} invalid timecodes")
        if invalid_edit_types > 0:
            edl_issues.append(f"{invalid_edit_types} invalid edit types")
        if long_reel_names > 0:
            result.add_warn(f"EDL has {long_reel_names} reel names > 32 chars")

        result.metrics["edl_timecode_gaps"] = timecode_discontinuities
        if timecode_discontinuities > 0:
            result.add_warn(f"EDL has {timecode_discontinuities} timeline gaps (may be intentional)")

        # Store issues for report
        result.metrics["edl_issues"] = edl_issues
        if not edl_issues:
            result.add_pass("EDL CMX3600 compliant")
        else:
            for issue in edl_issues[:3]:
                result.add_fail(f"EDL: {issue}")

    def _is_valid_timecode(self, tc: str) -> bool:
        """Validate timecode format HH:MM:SS:FF or HH:MM:SS;FF."""
        # Pattern: HH:MM:SS:FF or HH:MM:SS;FF (drop frame uses semicolon)
        pattern = r'^(\d{2}):(\d{2}):(\d{2})[;:](\d{2})$'
        match = re.match(pattern, tc)
        if not match:
            return False

        hours, mins, secs, frames = map(int, match.groups())
        # Basic sanity checks
        if hours > 23 or mins > 59 or secs > 59 or frames > 59:
            return False
        return True

    # -------------------------------------------------------------------------
    # Cross-File Consistency Validation
    # -------------------------------------------------------------------------

    def _validate_cross_file_consistency(self, output_dir: Path, result: ValidationResult):
        """Validate consistency across OTIO, XML, and EDL files."""
        consistency_issues = []

        # Collect frame rates from all sources
        frame_rates_by_source = {}

        # OTIO frame rate
        otio_rate = result.metrics.get("frame_rate", "").replace(" fps", "")
        if otio_rate and otio_rate != "Unknown":
            try:
                frame_rates_by_source["OTIO"] = float(otio_rate)
            except ValueError:
                pass

        # XML frame rates (already collected in detected_frame_rates)
        xml_rates = result.metrics.get("detected_frame_rates", set())
        if xml_rates:
            frame_rates_by_source["XML"] = list(xml_rates)

        # EDL frame rate (infer from FCM mode if available)
        edl_file = output_dir / "timeline.edl"
        if edl_file.exists():
            try:
                with open(edl_file, encoding="utf-8", errors="ignore") as f:
                    edl_content = f.read()
                fcm_match = re.search(r'FCM:\s*(DROP\s*FRAME|NON.?DROP\s*FRAME|DF|NDF)', edl_content, re.IGNORECASE)
                if fcm_match:
                    fcm = fcm_match.group(1).upper().replace(' ', '')
                    # Drop frame typically means 29.97 or 59.94
                    if 'DROP' in fcm and 'NON' not in fcm:
                        frame_rates_by_source["EDL (inferred)"] = "29.97 or 59.94 (drop frame)"
                    else:
                        frame_rates_by_source["EDL (inferred)"] = "integer fps (non-drop)"
            except Exception:
                pass

        # Check frame rate consistency
        unique_rates = set()
        for source, rate in frame_rates_by_source.items():
            if isinstance(rate, (int, float)):
                unique_rates.add(rate)
            elif isinstance(rate, list):
                unique_rates.update(rate)

        if len(unique_rates) > 1:
            consistency_issues.append(f"Mixed frame rates: {unique_rates}")
            result.add_warn(f"Frame rate inconsistency across files: {unique_rates}")
        elif len(unique_rates) == 1:
            rate = list(unique_rates)[0]
            if rate not in VALID_FRAME_RATES:
                result.add_warn(f"Non-standard frame rate {rate} may cause NLE issues")
            else:
                result.add_pass(f"Consistent frame rate: {rate} fps")

        # Validate clip counts are reasonable across formats
        otio_clips = result.metrics.get("total_clips", 0)
        edl_entries = result.metrics.get("edl_edit_count", 0)

        if otio_clips > 0 and edl_entries > 0:
            # EDL typically has fewer entries than OTIO clips due to single-track export
            # But they should be in the same order of magnitude
            ratio = max(otio_clips, edl_entries) / max(min(otio_clips, edl_entries), 1)
            if ratio > 100:
                consistency_issues.append(f"Large discrepancy: OTIO {otio_clips} clips vs EDL {edl_entries} entries")
                result.add_warn(f"Clip count discrepancy: OTIO={otio_clips}, EDL={edl_entries}")

        # Store consistency results
        result.metrics["consistency_issues"] = consistency_issues
        if not consistency_issues:
            result.add_pass("Cross-file consistency validated")


# =============================================================================
# PATH RESOLUTION
# =============================================================================

def resolve_output_path(path: Path) -> Path:
    """
    Resolve path to output directory.

    Accepts:
    - Output directory (contains timeline_FULL.otio) - returned as-is
    - Project directory (contains output/ folder) - returns latest output

    Args:
        path: Path to output directory or project directory

    Returns:
        Resolved path to output directory
    """
    if not path.exists():
        return path  # Let validator handle non-existent path

    # Check if this is already an output directory
    if (path / "timeline_FULL.otio").exists():
        return path

    # Check if this is a project directory with output/ subfolder
    output_dir = path / "output"
    if output_dir.exists() and output_dir.is_dir():
        # Find timestamped output folders (format: YYYYMMDD_HHMMSS)
        import re
        timestamp_pattern = re.compile(r"^\d{8}_\d{6}$")
        output_folders = [
            f for f in output_dir.iterdir()
            if f.is_dir() and timestamp_pattern.match(f.name)
        ]

        if output_folders:
            # Sort by name (timestamp) descending to get latest
            latest = sorted(output_folders, key=lambda f: f.name, reverse=True)[0]
            print(f"[INFO] Resolved project path to latest output: {latest.name}")
            return latest

        # No timestamped folders, check for any folder with timeline_FULL.otio
        for folder in output_dir.iterdir():
            if folder.is_dir() and (folder / "timeline_FULL.otio").exists():
                print(f"[INFO] Resolved to output folder: {folder.name}")
                return folder

    # Return original path - validator will report appropriate error
    return path


# =============================================================================
# CLI INTERFACE
# =============================================================================

def validate_output(output_path: str) -> ValidationResult:
    """Validate output directory."""
    validator = OutputValidator()
    resolved_path = resolve_output_path(Path(output_path))
    return validator.validate(resolved_path)


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Validate pipeline output structure")
    parser.add_argument("output_dir", help="Path to output directory or project directory")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show all checks")
    parser.add_argument("--json", action="store_true", help="Output as JSON")

    args = parser.parse_args()

    # Resolve path (handles project directories)
    input_path = Path(args.output_dir)
    resolved_path = resolve_output_path(input_path)

    validator = OutputValidator()
    result = validator.validate(resolved_path)

    if args.json:
        import json
        # Convert sets to lists for JSON serialization
        metrics_json = {}
        for k, v in result.metrics.items():
            if isinstance(v, set):
                metrics_json[k] = list(v)
            else:
                metrics_json[k] = v
        print(json.dumps({
            "success": result.success,
            "input_path": str(input_path),
            "resolved_path": str(resolved_path),
            "passed": result.passed,
            "failed": result.failed,
            "warnings": result.warnings,
            "metrics": metrics_json,
        }, indent=2))
    else:
        # Show resolved path if different from input
        if resolved_path != input_path:
            print(f"Validating: {resolved_path}\n")
        print(result.summary())

        if args.verbose:
            print("\nAll passed checks:")
            for p in result.passed:
                print(f"  [OK] {p}")

        # Always show structure report for NLE import info
        print(result.structure_report())

        if result.success:
            print("\n[OK] Output validation passed")
        else:
            print("\n[FAIL] Output validation failed")

    return 0 if result.success else 1


# =============================================================================
# PYTEST TESTS
# =============================================================================

@pytest.fixture
def validator():
    return OutputValidator()


@pytest.fixture
def output_dir(request):
    import os
    path = os.environ.get("OUTPUT_DIR") or getattr(request.config, "output_dir", None)
    return Path(path) if path else None


class TestOutputValidation:
    """Pytest test class for output validation."""

    def test_output_structure(self, validator, output_dir):
        """Output passes structural validation."""
        if not output_dir:
            pytest.skip("Set OUTPUT_DIR environment variable")
        result = validator.validate(output_dir)
        assert result.success, result.summary()

    def test_required_files_exist(self, output_dir):
        """Required files exist in output."""
        if not output_dir:
            pytest.skip("Set OUTPUT_DIR environment variable")
        for filename in STRUCTURE_SPEC["required_files"]:
            assert (output_dir / filename).exists(), f"Missing: {filename}"

    @pytest.mark.skipif(not HAS_OTIO, reason="opentimelineio not installed")
    def test_otio_parses(self, output_dir):
        """OTIO files parse without errors."""
        if not output_dir:
            pytest.skip("Set OUTPUT_DIR environment variable")
        full_otio = output_dir / "timeline_FULL.otio"
        if full_otio.exists():
            timeline = otio.adapters.read_from_file(str(full_otio))
            assert timeline is not None
            assert timeline.duration().to_seconds() > 0

    def test_xml_wellformed(self, output_dir):
        """XML files are well-formed."""
        if not output_dir:
            pytest.skip("Set OUTPUT_DIR environment variable")
        for xml_file in output_dir.glob("*.xml"):
            ET.parse(xml_file)  # Raises on parse error

    def test_json_valid(self, output_dir):
        """JSON files are valid."""
        if not output_dir:
            pytest.skip("Set OUTPUT_DIR environment variable")
        for json_file in output_dir.glob("*.json"):
            with open(json_file, encoding="utf-8") as f:
                json.load(f)  # Raises on parse error


def pytest_addoption(parser):
    parser.addoption("--output-dir", action="store", default=None)


def pytest_configure(config):
    config.output_dir = config.getoption("--output-dir")


if __name__ == "__main__":
    exit(main())
