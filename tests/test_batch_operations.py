#!/usr/bin/env python3
"""
Tests for scripts/batch_operations.py export/import functionality (US-139-003).

Tests:
- Export checkpoints to JSON
- Import checkpoints from JSON
- Selective export (by stage)
- Date range filtering
- Validation of imported data
- Dry-run import
- Round-trip data integrity
"""

import json
import os
import shutil
import subprocess
import tempfile
from unittest.mock import MagicMock, patch

import pytest
from pathlib import Path

# Add scripts directory to path
import sys
scripts_dir = Path(__file__).parent.parent / "scripts"
sys.path.insert(0, str(scripts_dir))

# Import the module under test
import batch_operations


@pytest.fixture
def temp_project_dir():
    """Create a temporary project directory."""
    tmpdir = tempfile.mkdtemp()
    yield Path(tmpdir)
    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture
def sample_checkpoint_data():
    """Create sample checkpoint data."""
    return {
        "version": "1.0.0",
        "last_completed_stage": "MATCH",
        "created_at": "2026-01-15T10:00:00",
        "updated_at": "2026-01-16T15:30:00",
        "video_search": {
            "video_ids": ["vid1", "vid2", "vid3"],
            "query": "python tutorial"
        },
        "caption": {
            "captions": {
                "vid1": {"text": "Hello world"},
                "vid2": {"text": "Python is great"}
            }
        },
        "match": {
            "matches": [
                {"video_id": "vid1", "start": 0, "end": 10},
                {"video_id": "vid2", "start": 10, "end": 20}
            ],
            "match_count": 2
        }
    }


class TestExportCheckpoint:
    """Test checkpoint export functionality."""

    def test_export_checkpoint_no_file(self, temp_project_dir):
        """Test export when no checkpoint exists."""
        result = batch_operations.export_checkpoint(temp_project_dir)
        assert result["exists"] is False
        assert "error" in result

    def test_export_checkpoint_with_data(self, temp_project_dir, sample_checkpoint_data):
        """Test export with existing checkpoint."""
        # Create checkpoint file
        checkpoint_file = temp_project_dir / "checkpoint.json"
        with open(checkpoint_file, 'w') as f:
            json.dump(sample_checkpoint_data, f)

        result = batch_operations.export_checkpoint(temp_project_dir)

        assert result["exists"] is True
        assert result["data"]["last_completed_stage"] == "MATCH"
        assert "video_search" in result["data"]
        assert "caption" in result["data"]
        assert "match" in result["data"]

    def test_export_selective_stages(self, temp_project_dir, sample_checkpoint_data):
        """Test selective stage export."""
        # Create checkpoint file
        checkpoint_file = temp_project_dir / "checkpoint.json"
        with open(checkpoint_file, 'w') as f:
            json.dump(sample_checkpoint_data, f)

        result = batch_operations.export_checkpoint(
            temp_project_dir,
            include_stages=["video_search"]
        )

        assert result["exists"] is True
        assert "video_search" in result["data"]
        assert "caption" not in result["data"]
        assert "match" not in result["data"]

    def test_export_since_date_filter(self, temp_project_dir, sample_checkpoint_data):
        """Test date range filtering."""
        # Create checkpoint file
        checkpoint_file = temp_project_dir / "checkpoint.json"
        with open(checkpoint_file, 'w') as f:
            json.dump(sample_checkpoint_data, f)

        # Should include (checkpoint updated at 2026-01-16)
        result = batch_operations.export_checkpoint(
            temp_project_dir,
            since_date="2026-01-01"
        )
        assert result["exists"] is True
        assert result.get("filtered") is not True

        # Should filter out (checkpoint updated at 2026-01-16 < 2026-02-01)
        result = batch_operations.export_checkpoint(
            temp_project_dir,
            since_date="2026-02-01"
        )
        assert result.get("filtered") is True


class TestValidateImportData:
    """Test import validation."""

    def test_validate_valid_data(self, sample_checkpoint_data):
        """Test validation of valid checkpoint data."""
        is_valid, errors = batch_operations.validate_import_data(sample_checkpoint_data)
        assert is_valid is True
        assert len(errors) == 0

    def test_validate_missing_version(self):
        """Test validation fails for missing version."""
        data = {"last_completed_stage": "MATCH"}
        is_valid, errors = batch_operations.validate_import_data(data)
        assert is_valid is False
        assert any("version" in e for e in errors)

    def test_validate_invalid_stage(self):
        """Test validation fails for invalid stage."""
        data = {"version": "1.0.0", "last_completed_stage": "INVALID_STAGE"}
        is_valid, errors = batch_operations.validate_import_data(data)
        assert is_valid is False
        assert any("Invalid stage" in e for e in errors)

    def test_validate_invalid_stage_data(self):
        """Test validation fails when stage data is not a dict."""
        data = {"version": "1.0.0", "video_search": "not a dict"}
        is_valid, errors = batch_operations.validate_import_data(data)
        assert is_valid is False
        assert any("must be a dictionary" in e for e in errors)


class TestImportCheckpoint:
    """Test checkpoint import functionality."""

    def test_import_new_checkpoint(self, temp_project_dir, sample_checkpoint_data):
        """Test importing a new checkpoint."""
        success, message = batch_operations.import_checkpoint(
            temp_project_dir, sample_checkpoint_data, dry_run=False
        )

        assert success is True
        assert (temp_project_dir / "checkpoint.json").exists()

        # Verify imported data
        with open(temp_project_dir / "checkpoint.json") as f:
            imported = json.load(f)

        assert imported["last_completed_stage"] == "MATCH"
        assert "_imported_at" in imported

    def test_import_dry_run(self, temp_project_dir, sample_checkpoint_data):
        """Test dry-run import doesn't modify files."""
        success, message = batch_operations.import_checkpoint(
            temp_project_dir, sample_checkpoint_data, dry_run=True
        )

        assert success is True
        assert "[DRY-RUN]" in message
        assert not (temp_project_dir / "checkpoint.json").exists()

    def test_import_validation_failure(self, temp_project_dir):
        """Test import fails with invalid data."""
        invalid_data = {"version": "1.0.0", "last_completed_stage": "INVALID"}
        success, message = batch_operations.import_checkpoint(
            temp_project_dir, invalid_data, dry_run=False, validate=True
        )

        assert success is False
        assert "Validation failed" in message


class TestBatchExport:
    """Test batch export functionality."""

    def test_batch_export_single_project(self, temp_project_dir, sample_checkpoint_data):
        """Test batch export with single project."""
        # Create checkpoint
        checkpoint_file = temp_project_dir / "checkpoint.json"
        with open(checkpoint_file, 'w') as f:
            json.dump(sample_checkpoint_data, f)

        output_path = temp_project_dir / "export.json"
        projects = [temp_project_dir]

        results = batch_operations.batch_export(projects, output_path)

        assert results["total"] == 1
        assert results["succeeded"] == 1
        assert results["failed"] == 0
        assert output_path.exists()

        # Verify export file contents
        with open(output_path) as f:
            export_data = json.load(f)

        assert "data" in export_data
        assert temp_project_dir.name in export_data["data"]


class TestBatchImport:
    """Test batch import functionality."""

    def test_batch_import_single_project(self, temp_project_dir, sample_checkpoint_data):
        """Test batch import with single project."""
        # Create import file
        import_file_data = {
            "version": "1.0",
            "data": {
                temp_project_dir.name: sample_checkpoint_data
            }
        }
        import_file_path = temp_project_dir / "import.json"
        with open(import_file_path, 'w') as f:
            json.dump(import_file_data, f)

        projects = [temp_project_dir]

        results = batch_operations.batch_import(
            projects, import_file_path, dry_run=False
        )

        assert results["total"] == 1
        assert results["succeeded"] == 1

        # Verify checkpoint was created
        assert (temp_project_dir / "checkpoint.json").exists()

    def test_batch_import_dry_run(self, temp_project_dir, sample_checkpoint_data):
        """Test batch import dry-run."""
        # Create import file
        import_file_data = {
            "version": "1.0",
            "data": {
                temp_project_dir.name: sample_checkpoint_data
            }
        }
        import_file_path = temp_project_dir / "import.json"
        with open(import_file_path, 'w') as f:
            json.dump(import_file_data, f)

        projects = [temp_project_dir]

        results = batch_operations.batch_import(
            projects, import_file_path, dry_run=True
        )

        assert results["succeeded"] == 1
        # File should not be created in dry-run
        assert not (temp_project_dir / "checkpoint.json").exists()


class TestRoundTrip:
    """Test export/import round-trip preserves data."""

    def test_roundtrip_preserves_data(self, temp_project_dir, sample_checkpoint_data):
        """Test that export/import round-trip preserves all data."""
        # Setup: create checkpoint in same-named temp dir
        checkpoint_file = temp_project_dir / "checkpoint.json"
        with open(checkpoint_file, 'w') as f:
            json.dump(sample_checkpoint_data, f)

        # Step 1: Export
        export_path = temp_project_dir / "export.json"
        results = batch_operations.batch_export([temp_project_dir], export_path)
        assert results["succeeded"] == 1

        # Step 2: Import back to same project (overwrite test)
        # First, remove the checkpoint
        checkpoint_file.unlink()

        projects = [temp_project_dir]
        results = batch_operations.batch_import(projects, export_path, dry_run=False)
        assert results["succeeded"] == 1

        # Step 3: Verify data matches
        with open(temp_project_dir / "checkpoint.json") as f:
            imported_data = json.load(f)

        assert imported_data["last_completed_stage"] == sample_checkpoint_data["last_completed_stage"]
        assert imported_data["video_search"] == sample_checkpoint_data["video_search"]
        assert imported_data["caption"] == sample_checkpoint_data["caption"]
        assert imported_data["match"] == sample_checkpoint_data["match"]


class TestFindProjects:
    """Test find_projects with various path patterns."""

    def test_find_projects_with_glob_pattern(self, temp_project_dir):
        """Test finding projects with glob pattern."""
        # Create test project directories
        (temp_project_dir / "Project1__2026-01-01").mkdir()
        (temp_project_dir / "Project1__2026-01-01" / "run.bat").touch()
        (temp_project_dir / "Project2__2026-01-02").mkdir()
        (temp_project_dir / "Project2__2026-01-02" / "checkpoint.json").touch()

        projects = batch_operations.find_projects(str(temp_project_dir / "*"))

        assert len(projects) == 2
        assert sorted([p.name for p in projects]) == ["Project1__2026-01-01", "Project2__2026-01-02"]

    def test_find_projects_single_directory(self, temp_project_dir):
        """Test finding a single project directory."""
        project_dir = temp_project_dir / "MyProject__2026-01-01"
        project_dir.mkdir()
        (project_dir / "run.sh").touch()

        projects = batch_operations.find_projects(str(project_dir))

        assert len(projects) == 1
        assert projects[0].name == "MyProject__2026-01-01"

    def test_find_projects_invalid_directory(self, temp_project_dir):
        """Test with directory that has no project files."""
        (temp_project_dir / "NotAProject").mkdir()

        projects = batch_operations.find_projects(str(temp_project_dir / "*"))

        assert len(projects) == 0

    def test_find_projects_file_path(self, temp_project_dir):
        """Test finding project from a file path."""
        project_dir = temp_project_dir / "Project__2026-01-01"
        project_dir.mkdir()
        (project_dir / "checkpoint.json").touch()

        # Pass a file inside the project directory
        file_path = project_dir / "checkpoint.json"
        projects = batch_operations.find_projects(str(file_path))

        assert len(projects) == 1

    def test_find_projects_sorted_by_name(self, temp_project_dir):
        """Test that projects are sorted by name."""
        (temp_project_dir / "ZebraProject").mkdir()
        (temp_project_dir / "ZebraProject" / "run.bat").touch()
        (temp_project_dir / "AlphaProject").mkdir()
        (temp_project_dir / "AlphaProject" / "run.bat").touch()
        (temp_project_dir / "BetaProject").mkdir()
        (temp_project_dir / "BetaProject" / "run.bat").touch()

        projects = batch_operations.find_projects(str(temp_project_dir / "*"))

        names = [p.name for p in projects]
        assert names == sorted(names)


class TestIsValidProject:
    """Test project validation logic (_is_valid_project)."""

    def test_valid_project_with_run_bat(self, temp_project_dir):
        """Test valid project with run.bat."""
        (temp_project_dir / "run.bat").touch()
        assert batch_operations._is_valid_project(temp_project_dir) is True

    def test_valid_project_with_run_sh(self, temp_project_dir):
        """Test valid project with run.sh."""
        (temp_project_dir / "run.sh").touch()
        assert batch_operations._is_valid_project(temp_project_dir) is True

    def test_valid_project_with_checkpoint(self, temp_project_dir):
        """Test valid project with checkpoint.json."""
        (temp_project_dir / "checkpoint.json").touch()
        assert batch_operations._is_valid_project(temp_project_dir) is True

    def test_invalid_project(self, temp_project_dir):
        """Test invalid project with no project files."""
        assert batch_operations._is_valid_project(temp_project_dir) is False


class TestLoadCheckpoint:
    """Test checkpoint loading and parsing."""

    def test_load_checkpoint_success(self, temp_project_dir):
        """Test successfully loading a checkpoint."""
        checkpoint_data = {
            "version": "4.0.0",
            "last_completed_stage": "MATCH",
            "created_at": "2026-01-01T00:00:00Z"
        }
        checkpoint_path = temp_project_dir / "checkpoint.json"
        checkpoint_path.write_text(json.dumps(checkpoint_data))

        result = batch_operations.load_checkpoint(temp_project_dir)

        assert result == checkpoint_data

    def test_load_checkpoint_no_file(self, temp_project_dir):
        """Test loading checkpoint when file doesn't exist."""
        result = batch_operations.load_checkpoint(temp_project_dir)
        assert result == {}

    def test_load_checkpoint_invalid_json(self, temp_project_dir):
        """Test loading invalid JSON checkpoint."""
        checkpoint_path = temp_project_dir / "checkpoint.json"
        checkpoint_path.write_text("{invalid json")

        result = batch_operations.load_checkpoint(temp_project_dir)
        assert result == {}


class TestGetProjectStatus:
    """Test get_project_status function."""

    def test_get_project_status_with_checkpoint(self, temp_project_dir):
        """Test getting project status with checkpoint data."""
        checkpoint_data = {
            "version": "4.0.0",
            "last_completed_stage": "MATCH",
            "download": {
                "video_paths": ["video1.mp4", "video2.mp4", "video3.mp4"]
            },
            "match": {
                "match_count": 5,
                "avg_confidence": 0.85
            }
        }
        checkpoint_path = temp_project_dir / "checkpoint.json"
        checkpoint_path.write_text(json.dumps(checkpoint_data))

        status = batch_operations.get_project_status(temp_project_dir)

        assert status['name'] == temp_project_dir.name
        assert status['last_stage'] == 'MATCH'
        assert status['completed'] is False
        assert status['has_checkpoint'] is True
        assert status['video_count'] == 3
        assert status['match_count'] == 5
        assert status['avg_confidence'] == 0.85

    def test_get_project_status_output_complete(self, temp_project_dir):
        """Test getting status when pipeline is complete."""
        checkpoint_data = {
            "last_completed_stage": "OUTPUT"
        }
        checkpoint_path = temp_project_dir / "checkpoint.json"
        checkpoint_path.write_text(json.dumps(checkpoint_data))

        status = batch_operations.get_project_status(temp_project_dir)

        assert status['last_stage'] == 'OUTPUT'
        assert status['completed'] is True

    def test_get_project_status_no_checkpoint(self, temp_project_dir):
        """Test getting status when no checkpoint exists."""
        status = batch_operations.get_project_status(temp_project_dir)

        assert status['last_stage'] is None
        assert status['has_checkpoint'] is False
        assert status['video_count'] == 0
        assert status['match_count'] == 0


class TestPrintStatusTable:
    """Test status table formatting."""

    def test_print_status_table_empty(self, capsys):
        """Test printing empty status table."""
        batch_operations.print_status_table([])

        captured = capsys.readouterr()
        assert "No projects found" in captured.out

    def test_print_status_table_with_projects(self, capsys):
        """Test printing status table with projects."""
        projects = [
            {
                'name': 'Project1',
                'last_stage': 'MATCH',
                'video_count': 5,
                'match_count': 3,
                'avg_confidence': 0.85,
                'completed': False
            },
            {
                'name': 'Project2',
                'last_stage': 'OUTPUT',
                'video_count': 10,
                'match_count': 8,
                'avg_confidence': 0.92,
                'completed': True
            }
        ]

        batch_operations.print_status_table(projects)

        captured = capsys.readouterr()
        assert 'Project1' in captured.out
        assert 'Project2' in captured.out
        assert 'MATCH' in captured.out
        assert 'OUTPUT' in captured.out
        assert 'Total: 2' in captured.out


class TestResumeWithMocks:
    """Test resume operations with mocked subprocess calls."""

    @patch('batch_operations.subprocess.run')
    def test_resume_project_success(self, mock_run, temp_project_dir):
        """Test successful project resume."""
        # Create checkpoint
        checkpoint_data = {"last_completed_stage": "MATCH"}
        checkpoint_path = temp_project_dir / "checkpoint.json"
        checkpoint_path.write_text(json.dumps(checkpoint_data))

        # Mock successful subprocess
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_run.return_value = mock_result

        success, message = batch_operations.resume_project(temp_project_dir)

        assert success is True
        assert "Resumed" in message
        mock_run.assert_called_once()

    @patch('batch_operations.subprocess.run')
    def test_resume_project_failure(self, mock_run, temp_project_dir):
        """Test failed project resume."""
        checkpoint_path = temp_project_dir / "checkpoint.json"
        checkpoint_path.write_text(json.dumps({"last_completed_stage": "MATCH"}))

        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_run.return_value = mock_result

        success, message = batch_operations.resume_project(temp_project_dir)

        assert success is False
        assert "Failed" in message

    @patch('batch_operations.subprocess.run')
    def test_resume_project_timeout(self, mock_run, temp_project_dir):
        """Test project resume with timeout."""
        checkpoint_path = temp_project_dir / "checkpoint.json"
        checkpoint_path.write_text(json.dumps({"last_completed_stage": "MATCH"}))

        mock_run.side_effect = subprocess.TimeoutExpired("cmd", 3600)

        success, message = batch_operations.resume_project(temp_project_dir)

        assert success is False
        assert "Timeout" in message


class TestCleanupWithMocks:
    """Test cleanup operations with mocked subprocess calls."""

    @patch('batch_operations.subprocess.run')
    def test_cleanup_project_success(self, mock_run, temp_project_dir):
        """Test successful project cleanup."""
        # Mock successful subprocess
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_run.return_value = mock_result

        success, message = batch_operations.cleanup_project(
            temp_project_dir,
            level="cache",
            dry_run=False
        )

        assert success is True
        assert "Cleaned" in message
        mock_run.assert_called_once()

    @patch('batch_operations.subprocess.run')
    def test_cleanup_project_dry_run(self, mock_run, temp_project_dir):
        """Test cleanup project dry-run."""
        success, message = batch_operations.cleanup_project(
            temp_project_dir,
            level="cache",
            dry_run=True
        )

        assert success is True
        assert "DRY-RUN" in message
        mock_run.assert_not_called()

    @patch('batch_operations.subprocess.run')
    def test_cleanup_project_failure(self, mock_run, temp_project_dir):
        """Test failed project cleanup."""
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_run.return_value = mock_result

        success, message = batch_operations.cleanup_project(
            temp_project_dir,
            level="cache",
            dry_run=False
        )

        assert success is False
        assert "Failed" in message


class TestBatchCleanupWithMocks:
    """Test batch cleanup with mocked cleanup_project."""

    @patch('batch_operations.cleanup_project')
    def test_batch_cleanup(self, mock_cleanup, temp_project_dir):
        """Test batch cleanup calls cleanup for each project."""
        project1 = temp_project_dir / "Project1"
        project1.mkdir()
        (project1 / "checkpoint.json").touch()

        project2 = temp_project_dir / "Project2"
        project2.mkdir()
        (project2 / "checkpoint.json").touch()

        mock_cleanup.side_effect = [
            (True, "Cleaned: Project1"),
            (True, "Cleaned: Project2")
        ]

        results = batch_operations.batch_cleanup([project1, project2])

        assert results['total'] == 2
        assert results['succeeded'] == 2
        assert mock_cleanup.call_count == 2


class TestStageDetection:
    """Test checkpoint-aware stage detection for resume (US-139-009)."""

    def test_get_last_completed_stage_no_checkpoint(self, temp_project_dir):
        """Test stage detection when no checkpoint exists."""
        stage = batch_operations.get_last_completed_stage(temp_project_dir)
        assert stage is None

    def test_get_last_completed_stage_with_checkpoint(self, temp_project_dir):
        """Test stage detection with existing checkpoint."""
        checkpoint_data = {
            "version": "1.0.0",
            "last_completed_stage": "MATCH"
        }
        checkpoint_file = temp_project_dir / "checkpoint.json"
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint_data, f)

        stage = batch_operations.get_last_completed_stage(temp_project_dir)
        assert stage == "MATCH"

    def test_is_stage_completed_true(self, temp_project_dir):
        """Test stage completion check when stage is completed."""
        checkpoint_data = {
            "version": "1.0.0",
            "last_completed_stage": "MATCH"
        }
        checkpoint_file = temp_project_dir / "checkpoint.json"
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint_data, f)

        assert batch_operations.is_stage_completed(temp_project_dir, "VIDEO_SEARCH") is True
        assert batch_operations.is_stage_completed(temp_project_dir, "CAPTION") is True
        assert batch_operations.is_stage_completed(temp_project_dir, "MATCH") is True

    def test_is_stage_completed_false(self, temp_project_dir):
        """Test stage completion check when stage is not completed."""
        checkpoint_data = {
            "version": "1.0.0",
            "last_completed_stage": "CAPTION"
        }
        checkpoint_file = temp_project_dir / "checkpoint.json"
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint_data, f)

        assert batch_operations.is_stage_completed(temp_project_dir, "MATCH") is False
        assert batch_operations.is_stage_completed(temp_project_dir, "ITERATIVE_MATCH") is False

    def test_should_skip_for_resume_output_complete(self, temp_project_dir):
        """Test that OUTPUT stage is skipped for resume."""
        checkpoint_data = {
            "version": "1.0.0",
            "last_completed_stage": "OUTPUT"
        }
        checkpoint_file = temp_project_dir / "checkpoint.json"
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint_data, f)

        should_skip, reason = batch_operations.should_skip_for_resume(temp_project_dir)
        assert should_skip is True
        assert "OUTPUT" in reason

    def test_should_skip_for_resume_in_progress(self, temp_project_dir):
        """Test that in-progress projects are not skipped."""
        checkpoint_data = {
            "version": "1.0.0",
            "last_completed_stage": "MATCH"
        }
        checkpoint_file = temp_project_dir / "checkpoint.json"
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint_data, f)

        should_skip, reason = batch_operations.should_skip_for_resume(temp_project_dir)
        assert should_skip is False

    def test_should_skip_for_resume_no_checkpoint(self, temp_project_dir):
        """Test that projects without checkpoint are not skipped."""
        should_skip, reason = batch_operations.should_skip_for_resume(temp_project_dir)
        assert should_skip is False
        assert "No checkpoint" in reason

    def test_batch_resume_skips_completed(self, temp_project_dir):
        """Test that batch resume skips completed projects by default."""
        # Create checkpoint at OUTPUT stage
        checkpoint_data = {
            "version": "1.0.0",
            "last_completed_stage": "OUTPUT"
        }
        checkpoint_file = temp_project_dir / "checkpoint.json"
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint_data, f)

        results = batch_operations.batch_resume(
            [temp_project_dir],
            parallel=False,
            dry_run=True,
            force=False
        )

        assert results["skipped"] == 1
        assert results["succeeded"] == 0
        assert len(results["skipped_projects"]) == 1
        assert results["skipped_projects"][0]["reason"] == "Already at OUTPUT stage (pipeline complete)"

    def test_batch_resume_force_flag(self, temp_project_dir):
        """Test that --force flag overrides skip behavior."""
        # Create checkpoint at OUTPUT stage
        checkpoint_data = {
            "version": "1.0.0",
            "last_completed_stage": "OUTPUT"
        }
        checkpoint_file = temp_project_dir / "checkpoint.json"
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint_data, f)

        results = batch_operations.batch_resume(
            [temp_project_dir],
            parallel=False,
            dry_run=True,
            force=True  # Force resume
        )

        assert results["skipped"] == 0
        assert results["total"] == 1

    def test_batch_resume_stage_argument(self, temp_project_dir):
        """Test that --stage flag sets target resume stage."""
        checkpoint_data = {
            "version": "1.0.0",
            "last_completed_stage": "CAPTION"
        }
        checkpoint_file = temp_project_dir / "checkpoint.json"
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint_data, f)

        results = batch_operations.batch_resume(
            [temp_project_dir],
            parallel=False,
            dry_run=True,
            force=False,
            resume_from_stage="MATCH"
        )

        assert results["succeeded"] == 1

    def test_stage_order_constant(self):
        """Test that STAGE_ORDER contains expected stages."""
        assert "ANALYZE" in batch_operations.STAGE_ORDER
        assert "VIDEO_SEARCH" in batch_operations.STAGE_ORDER
        assert "CAPTION" in batch_operations.STAGE_ORDER
        assert "MATCH" in batch_operations.STAGE_ORDER
        assert "ITERATIVE_MATCH" in batch_operations.STAGE_ORDER
        assert "DOWNLOAD_SEGMENTS" in batch_operations.STAGE_ORDER
        assert "OUTPUT" in batch_operations.STAGE_ORDER
        assert len(batch_operations.STAGE_ORDER) == 7
