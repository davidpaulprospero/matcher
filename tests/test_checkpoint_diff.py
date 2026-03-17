"""
Tests for checkpoint diff tool (US-115-004)

Verifies diff detection for various scenarios:
- Added/removed/modified stages
- Field-level diffs
- Metadata changes
- JSON output format
"""

import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.checkpoint_diff import (
    compute_checkpoint_diff,
    compute_stage_diff,
    compute_field_diff,
    FieldDiff,
    StageDiff,
    CheckpointDiff,
    format_diff_human,
    format_diff_json,
    STAGE_FIELDS,
)


class TestFieldDiff:
    """Test field-level diff computation"""

    def test_modified_field(self):
        """Detect modified primitive fields"""
        old = {"count": 10}
        new = {"count": 20}
        diffs = compute_field_diff(old, new)
        assert len(diffs) == 1
        assert diffs[0].change_type == 'modified'
        assert diffs[0].old_value == 10
        assert diffs[0].new_value == 20

    def test_added_field(self):
        """Detect new fields"""
        old = {"a": 1}
        new = {"a": 1, "b": 2}
        diffs = compute_field_diff(old, new)
        assert len(diffs) == 1
        assert diffs[0].change_type == 'added'
        assert diffs[0].field == 'b'

    def test_removed_field(self):
        """Detect removed fields"""
        old = {"a": 1, "b": 2}
        new = {"a": 1}
        diffs = compute_field_diff(old, new)
        assert len(diffs) == 1
        assert diffs[0].change_type == 'removed'
        assert diffs[0].field == 'b'

    def test_nested_dict_diff(self):
        """Detect changes in nested dicts"""
        old = {"video": {"id": "abc", "title": "Old"}}
        new = {"video": {"id": "abc", "title": "New"}}
        diffs = compute_field_diff(old, new)
        # Should have one diff for title
        assert any(d.field == 'video.title' and d.new_value == 'New' for d in diffs)

    def test_list_diff_same_length(self):
        """Detect changes in lists of same length"""
        old = {"videos": ["a", "b"]}
        new = {"videos": ["a", "c"]}
        diffs = compute_field_diff(old, new)
        assert len(diffs) >= 1

    def test_list_diff_different_length(self):
        """Detect list length changes"""
        old = {"videos": ["a", "b", "c"]}
        new = {"videos": ["a", "b"]}
        diffs = compute_field_diff(old, new)
        assert any(d.change_type == 'modified' for d in diffs)

    def test_no_changes(self):
        """Identical dicts have no diffs"""
        old = {"a": 1, "b": 2}
        new = {"a": 1, "b": 2}
        diffs = compute_field_diff(old, new)
        assert len(diffs) == 0


class TestStageDiff:
    """Test stage-level diff computation"""

    def test_stage_added(self):
        """Detect new stage data"""
        diff = compute_stage_diff('match', {}, {"videos": []})
        assert diff.change_type == 'added'
        assert diff.has_changes()

    def test_stage_removed(self):
        """Detect removed stage data"""
        diff = compute_stage_diff('match', {"videos": []}, {})
        assert diff.change_type == 'removed'
        assert diff.has_changes()

    def test_stage_unchanged(self):
        """Detect unchanged stages"""
        diff = compute_stage_diff('match', {"videos": []}, {"videos": []})
        assert diff.change_type == 'unchanged'
        assert not diff.has_changes()

    def test_stage_modified(self):
        """Detect modified stage data"""
        old = {"videos": [{"id": "1"}]}
        new = {"videos": [{"id": "2"}]}
        diff = compute_stage_diff('match', old, new)
        assert diff.change_type == 'modified'
        assert diff.has_changes()


class TestCheckpointDiff:
    """Test full checkpoint diff computation"""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test checkpoints"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_identical_checkpoints(self, temp_dir):
        """No diffs for identical checkpoints"""
        cp1_path = temp_dir / "cp1.json"
        cp2_path = temp_dir / "cp2.json"

        data = {
            "version": "2.1",
            "last_completed_stage": "MATCH",
            "match": {"videos": [{"id": "abc"}]}
        }

        cp1_path.write_text(json.dumps(data))
        cp2_path.write_text(json.dumps(data))

        diff = compute_checkpoint_diff(str(cp1_path), str(cp2_path))
        assert not diff.has_changes()

    def test_stage_added_diff(self, temp_dir):
        """Detect new stage added"""
        cp1_path = temp_dir / "cp1.json"
        cp2_path = temp_dir / "cp2.json"

        cp1_data = {"version": "2.1", "analyze": {}, "match": {}}
        cp2_data = {"version": "2.1", "analyze": {"keywords": ["test"]}, "match": {}}

        cp1_path.write_text(json.dumps(cp1_data))
        cp2_path.write_text(json.dumps(cp2_data))

        diff = compute_checkpoint_diff(str(cp1_path), str(cp2_path))
        # Should have analyze stage diff - going from {} to data is 'added'
        analyze_diff = next((sd for sd in diff.stage_diffs if sd.stage_name == 'analyze'), None)
        assert analyze_diff is not None
        # When going from empty {} to non-empty, it's 'added' change type
        assert analyze_diff.change_type in ('added', 'modified')

    def test_metadata_diff(self, temp_dir):
        """Detect metadata changes"""
        cp1_path = temp_dir / "cp1.json"
        cp2_path = temp_dir / "cp2.json"

        cp1_data = {"version": "2.1", "last_completed_stage": "CAPTION"}
        cp2_data = {"version": "2.1", "last_completed_stage": "MATCH"}

        cp1_path.write_text(json.dumps(cp1_data))
        cp2_path.write_text(json.dumps(cp2_data))

        diff = compute_checkpoint_diff(str(cp1_path), str(cp2_path))
        assert len(diff.metadata_diffs) > 0
        assert any(md.field == 'last_completed_stage' for md in diff.metadata_diffs)

    def test_compressed_checkpoint(self, temp_dir):
        """Handle gzipped checkpoints"""
        import gzip

        cp1_path = temp_dir / "cp1.json.gz"
        cp2_path = temp_dir / "cp2.json.gz"

        cp1_data = {"version": "2.1", "match": {"videos": []}}
        cp2_data = {"version": "2.1", "match": {"videos": [{"id": "1"}]}}

        with gzip.open(cp1_path, 'wt', encoding='utf-8') as f:
            json.dump(cp1_data, f)
        with gzip.open(cp2_path, 'wt', encoding='utf-8') as f:
            json.dump(cp2_data, f)

        diff = compute_checkpoint_diff(str(cp1_path), str(cp2_path))
        assert diff.has_changes()

    def test_missing_checkpoint(self, temp_dir):
        """Handle missing checkpoint file"""
        cp1_path = temp_dir / "nonexistent.json"
        cp2_path = temp_dir / "exists.json"

        cp2_data = {"version": "2.1"}
        cp2_path.write_text(json.dumps(cp2_data))

        with pytest.raises(FileNotFoundError):
            compute_checkpoint_diff(str(cp1_path), str(cp2_path))

    def test_to_dict_json_serializable(self, temp_dir):
        """Verify to_dict produces JSON-serializable output"""
        cp1_path = temp_dir / "cp1.json"
        cp2_path = temp_dir / "cp2.json"

        cp1_data = {"version": "2.1", "match": {"count": 1}}
        cp2_data = {"version": "2.1", "match": {"count": 2}}

        cp1_path.write_text(json.dumps(cp1_data))
        cp2_path.write_text(json.dumps(cp2_data))

        diff = compute_checkpoint_diff(str(cp1_path), str(cp2_path))
        result = diff.to_dict()

        # Should be JSON serializable
        json_str = json.dumps(result)
        assert json_str is not None
        assert result['has_changes'] is True


class TestDiffOutput:
    """Test diff output formatting"""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test checkpoints"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_format_diff_human(self, temp_dir):
        """Human-readable output contains expected sections"""
        cp1_path = temp_dir / "cp1.json"
        cp2_path = temp_dir / "cp2.json"

        cp1_data = {"version": "2.1", "last_completed_stage": "CAPTION", "match": {}}
        cp2_data = {"version": "2.1", "last_completed_stage": "MATCH", "match": {"videos": []}}

        cp1_path.write_text(json.dumps(cp1_data))
        cp2_path.write_text(json.dumps(cp2_data))

        diff = compute_checkpoint_diff(str(cp1_path), str(cp2_path))
        output = format_diff_human(diff)

        assert "Checkpoint Diff" in output
        assert "Summary" in output
        assert "match" in output

    def test_format_diff_json(self, temp_dir):
        """JSON output is valid JSON"""
        cp1_path = temp_dir / "cp1.json"
        cp2_path = temp_dir / "cp2.json"

        cp1_data = {"version": "2.1", "match": {}}
        cp2_data = {"version": "2.1", "match": {"videos": []}}

        cp1_path.write_text(json.dumps(cp1_data))
        cp2_path.write_text(json.dumps(cp2_data))

        diff = compute_checkpoint_diff(str(cp1_path), str(cp2_path))
        output = format_diff_json(diff)

        # Should be valid JSON
        parsed = json.loads(output)
        assert 'checkpoint1' in parsed
        assert 'checkpoint2' in parsed
        assert 'has_changes' in parsed
        assert 'stage_diffs' in parsed

    def test_no_changes_message(self, temp_dir):
        """Output when checkpoints are identical"""
        cp1_path = temp_dir / "cp1.json"
        cp2_path = temp_dir / "cp2.json"

        data = {"version": "2.1", "match": {"videos": []}}
        cp1_path.write_text(json.dumps(data))
        cp2_path.write_text(json.dumps(data))

        diff = compute_checkpoint_diff(str(cp1_path), str(cp2_path))
        output = format_diff_human(diff)

        assert "No differences found" in output


class TestCliIntegration:
    """Test CLI integration"""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test checkpoints"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_cli_module_entry(self, temp_dir):
        """Test running as module"""
        cp1_path = temp_dir / "cp1.json"
        cp2_path = temp_dir / "cp2.json"

        cp1_data = {"version": "2.1", "match": {"count": 1}}
        cp2_data = {"version": "2.1", "match": {"count": 2}}

        cp1_path.write_text(json.dumps(cp1_data))
        cp2_path.write_text(json.dumps(cp2_data))

        # Import and run directly
        from src.checkpoint_diff import main
        import sys

        old_argv = sys.argv
        try:
            sys.argv = ['checkpoint_diff', str(cp1_path), str(cp2_path)]
            # Capture output
            import io
            old_stdout = sys.stdout
            sys.stdout = io.StringIO()
            try:
                main()
            finally:
                output = sys.stdout.getvalue()
                sys.stdout = old_stdout

            assert "Checkpoint Diff" in output
        finally:
            sys.argv = old_argv
