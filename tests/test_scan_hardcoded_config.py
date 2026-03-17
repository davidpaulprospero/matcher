#!/usr/bin/env python3
"""
Integration test for scan_hardcoded_config.py scanner.

Verifies the scanner finds known hardcoded config values in stages.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest


class TestScanHardcodedConfig:
    """Test suite for scan_hardcoded_config.py"""

    @pytest.fixture
    def scanner_path(self):
        """Path to the scanner script."""
        project_root = Path(__file__).parent.parent
        return project_root / "scripts" / "scan_hardcoded_config.py"

    def test_scanner_finds_known_hardcoded_values(self, scanner_path):
        """Scanner should find known hardcoded values in stages."""
        # Run scanner
        result = subprocess.run(
            [sys.executable, str(scanner_path), "--json"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )

        assert result.returncode in [0, 1], f"Scanner failed: {result.stderr}"
        output = json.loads(result.stdout)

        # Verify output structure
        assert "scanner" in output
        assert output["scanner"] == "scan_hardcoded_config"
        assert "findings" in output
        assert "total_findings" in output

        # Verify it finds at least some hardcoded values
        # We expect to find things like:
        # - bot_floor_threshold = 5 (download_segments.py)
        # - min_duration = 3.0 (entity_videos.py)
        # - max_duration = 30.0 (entity_videos.py)
        # - _socket_timeout = 30 (download_segments.py)
        # - _max_res = 1080 (download_segments.py)

        findings = output["findings"]
        file_paths = {f["file"] for f in findings}

        # Should find at least some known files with hardcoded values
        assert any("download_segments.py" in p for p in file_paths) or any(
            "entity_videos.py" in p for p in file_paths
        ), "Should find hardcoded values in known files"

    def test_scanner_json_output_valid(self, scanner_path):
        """Scanner should produce valid JSON output."""
        result = subprocess.run(
            [sys.executable, str(scanner_path), "--json"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )

        # Should be valid JSON
        output = json.loads(result.stdout)

        # Should have expected fields
        assert "version" in output
        assert "total_findings" in output
        assert isinstance(output["findings"], list)

        # Each finding should have required fields
        for finding in output["findings"]:
            assert "file" in finding
            assert "line" in finding
            assert "pattern" in finding
            assert "content" in finding

    def test_scanner_text_output(self, scanner_path):
        """Scanner should produce readable text output."""
        result = subprocess.run(
            [sys.executable, str(scanner_path)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )

        # Text output should be readable
        assert len(result.stdout) > 0
        # Either shows OK message or lists findings
        assert ("hardcoded" in result.stdout.lower()) or (
            "No hardcoded" in result.stdout
        )

    def test_scanner_verbose_flag(self, scanner_path):
        """Scanner should support --verbose flag."""
        result = subprocess.run(
            [sys.executable, str(scanner_path), "--verbose"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )

        # Should complete without error
        assert result.returncode in [0, 1]

    def test_scanner_finds_entity_videos_hardcoded(self, scanner_path):
        """Scanner should find hardcoded values in entity_videos.py."""
        result = subprocess.run(
            [sys.executable, str(scanner_path), "--json"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )

        output = json.loads(result.stdout)
        findings = output["findings"]

        # Find entity_videos.py findings
        entity_findings = [
            f
            for f in findings
            if "entity_videos.py" in f["file"] and f["line"] in [165, 166]
        ]

        # Should find min_duration and max_duration at lines 165-166
        assert len(entity_findings) >= 2, (
            f"Should find hardcoded min_duration/max_duration in entity_videos.py, "
            f"found: {entity_findings}"
        )

        patterns = {f["pattern"] for f in entity_findings}
        assert any("min_duration" in p for p in patterns)
        assert any("max_duration" in p for p in patterns)

    def test_scanner_finds_download_segments_hardcoded(self, scanner_path):
        """Scanner should find hardcoded values in download_segments.py."""
        result = subprocess.run(
            [sys.executable, str(scanner_path), "--json"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )

        output = json.loads(result.stdout)
        findings = output["findings"]

        # Find download_segments.py findings at known hardcoded lines
        download_findings = [
            f
            for f in findings
            if "download_segments.py" in f["file"]
            and f["line"] in [1315, 1416, 1842, 1843]
        ]

        # Should find known hardcoded values
        assert len(download_findings) >= 4, (
            f"Should find hardcoded values in download_segments.py at known lines, "
            f"found: {download_findings}"
        )

    def test_scanner_exit_code(self, scanner_path):
        """Scanner should return non-zero exit code when findings exist."""
        result = subprocess.run(
            [sys.executable, str(scanner_path)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )

        # Should return 1 when findings exist
        assert result.returncode == 1, (
            "Should return non-zero exit code when hardcoded values found"
        )
