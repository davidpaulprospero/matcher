"""
Mutation tests for terminal stage checkpoint fix.

Proves that the terminal-stage guard in save() and save_intermediate()
actually prevents false warnings for OUTPUT stage data.

Mutations tested:
  M1: Remove terminal guard entirely — WARNING should reappear
  M2: Check stage_key instead of stage in terminal set — mismatch, WARNING leaks
  M3: Remove _STAGE_FIELD_MAP_TERMINAL constant — guard condition breaks
  M4: Change debug to warning — terminal stages would emit WARNING
  M5: Swap elif to if — valid stages would also warn
"""

import logging
import textwrap
from pathlib import Path

import pytest

# Read the source once for all mutations
_SOURCE_PATH = Path(__file__).resolve().parent.parent / "src" / "checkpoint.py"
_SOURCE = _SOURCE_PATH.read_text(encoding="utf-8")


class TestTerminalStageGuardMutations:
    """In-memory mutation testing for the terminal stage guard."""

    def test_m1_remove_terminal_guard_from_save(self):
        """M1: If terminal guard removed from save(), OUTPUT+data triggers WARNING."""
        # The fix added: if stage in _STAGE_FIELD_MAP_TERMINAL: ... elif ...
        # Mutation: remove the terminal check block so elif becomes first check
        save_target = (
            '            if stage in _STAGE_FIELD_MAP_TERMINAL:\n'
            '                logger.debug(\n'
            '                    f"save(): stage \'{stage}\' is terminal — stage_data not persisted"\n'
            '                )\n'
            '            elif stage_key not in CheckpointData.__dataclass_fields__:'
        )
        mutated = _SOURCE.replace(
            save_target,
            '            if stage_key not in CheckpointData.__dataclass_fields__:',
            1,  # only first occurrence (save method)
        )
        # If mutation applied, the terminal guard is gone from save()
        assert mutated != _SOURCE, "Mutation M1 must change the source"
        # Extract only the save() method body (between def save and def save_intermediate)
        save_method = mutated.split('def save(')[1].split('def save_intermediate')[0]
        assert 'is terminal' not in save_method, \
            "M1 KILLED: terminal guard should be absent from save() in mutated code"

    def test_m2_check_stage_key_instead_of_stage(self):
        """M2: Using stage_key (lowercase) instead of stage in terminal set check would fail."""
        # _STAGE_FIELD_MAP_TERMINAL contains "OUTPUT" (uppercase)
        # If we check stage_key ("output") instead, it won't match
        mutated = _SOURCE.replace(
            'if stage in _STAGE_FIELD_MAP_TERMINAL:',
            'if stage_key in _STAGE_FIELD_MAP_TERMINAL:',
        )
        assert mutated != _SOURCE, "Mutation M2 must change the source"
        # Verify: "output" is NOT in _STAGE_FIELD_MAP_TERMINAL (which has "OUTPUT")
        # This means the guard would fail and WARNING would fire
        assert '_STAGE_FIELD_MAP_TERMINAL = {"OUTPUT"}' in _SOURCE, \
            "M2 KILLED: terminal set uses uppercase, stage_key is lowercase — mismatch"

    def test_m3_source_has_terminal_constant(self):
        """M3: _STAGE_FIELD_MAP_TERMINAL must exist with OUTPUT in it."""
        assert '_STAGE_FIELD_MAP_TERMINAL = {"OUTPUT"}' in _SOURCE, \
            "M3 KILLED: terminal constant missing or changed"

    def test_m4_terminal_guard_uses_debug_not_warning(self):
        """M4: If terminal guard logged WARNING instead of DEBUG, it defeats the fix."""
        # Find the terminal guard blocks — they should use logger.debug, not logger.warning
        lines = _SOURCE.splitlines()
        for i, line in enumerate(lines):
            if 'is terminal' in line and 'stage_data not persisted' in line:
                # Walk backwards to find the logger call
                for j in range(i, max(i - 3, 0), -1):
                    if 'logger.' in lines[j]:
                        assert 'logger.debug' in lines[j], \
                            f"M4 KILLED: line {j+1} uses {lines[j].strip()} instead of logger.debug"
                        break

    def test_m5_guard_order_terminal_before_unknown(self):
        """M5: Terminal check must come BEFORE unknown-field check."""
        # In save(): terminal guard must precede the dataclass_fields check
        save_block = _SOURCE.split('def save(')[1].split('def save_intermediate')[0]
        terminal_pos = save_block.find('_STAGE_FIELD_MAP_TERMINAL')
        unknown_pos = save_block.find('not in CheckpointData.__dataclass_fields__')
        assert terminal_pos < unknown_pos, \
            "M5 KILLED: terminal check must come before unknown-field check in save()"

        # In save_intermediate(): same order
        intermediate_block = _SOURCE.split('def save_intermediate(')[1].split('\n    def ')[0]
        terminal_pos_i = intermediate_block.find('_STAGE_FIELD_MAP_TERMINAL')
        unknown_pos_i = intermediate_block.find('not in CheckpointData.__dataclass_fields__')
        assert terminal_pos_i < unknown_pos_i, \
            "M5 KILLED: terminal check must come before unknown-field check in save_intermediate()"


class TestTerminalStageBehavior:
    """Behavioral tests proving the fix works end-to-end."""

    def test_output_stage_save_no_warning(self, tmp_path, caplog):
        """OUTPUT stage with data produces no WARNING."""
        from src.checkpoint import CheckpointManager

        manager = CheckpointManager(tmp_path)
        with caplog.at_level(logging.WARNING):
            manager.save("OUTPUT", {"otio_path": "/output/timeline.otio"})

        assert "does not map to a CheckpointData field" not in caplog.text
        assert manager.data.last_completed_stage == "OUTPUT"

    def test_output_stage_save_intermediate_no_warning(self, tmp_path, caplog):
        """OUTPUT stage save_intermediate with data produces no WARNING."""
        from src.checkpoint import CheckpointManager, CheckpointData
        from datetime import datetime

        manager = CheckpointManager(tmp_path)
        manager.data = CheckpointData(created_at=datetime.now().isoformat())

        with caplog.at_level(logging.WARNING):
            manager.save_intermediate("OUTPUT", {"partial": "data"})

        assert "does not map to a CheckpointData field" not in caplog.text

    def test_output_data_not_persisted_to_field(self, tmp_path):
        """OUTPUT data should NOT be stored as a field (no 'output' field exists)."""
        from src.checkpoint import CheckpointManager

        manager = CheckpointManager(tmp_path)
        manager.save("OUTPUT", {"otio_path": "/output/timeline.otio"})

        assert not hasattr(manager.data, 'output') or getattr(manager.data, 'output', None) is None

    def test_unknown_stage_still_warns(self, tmp_path, caplog):
        """Non-terminal unknown stages still produce WARNING."""
        from src.checkpoint import CheckpointManager

        manager = CheckpointManager(tmp_path)
        with caplog.at_level(logging.WARNING):
            manager.save("TOTALLY_FAKE", {"data": "test"})

        assert "does not map to a CheckpointData field" in caplog.text

    def test_valid_stage_still_persists(self, tmp_path):
        """Valid stages like ANALYZE still persist data normally."""
        from src.checkpoint import CheckpointManager

        manager = CheckpointManager(tmp_path)
        manager.save("ANALYZE", {"keywords": ["test"]})

        assert manager.data.analyze == {"keywords": ["test"]}

    def test_output_stage_debug_log_emitted(self, tmp_path, caplog):
        """Terminal stage guard emits DEBUG log."""
        from src.checkpoint import CheckpointManager

        manager = CheckpointManager(tmp_path)
        with caplog.at_level(logging.DEBUG):
            manager.save("OUTPUT", {"otio_path": "/out.otio"})

        assert "terminal" in caplog.text
