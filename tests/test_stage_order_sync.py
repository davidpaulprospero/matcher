"""
Test that STAGE_ORDER in checkpoint.py stays in sync with ralph-config.json.

CLAUDE.md Rule 31: src/checkpoint.py:STAGE_ORDER is the source of truth;
ralph-config.json must mirror it. This test catches drift automatically.
"""

import json
from pathlib import Path

import pytest

from src.checkpoint import STAGE_ORDER, LEGACY_STAGES

RALPH_CONFIG_PATH = Path(__file__).resolve().parents[1] / "scripts" / "ralph" / "config" / "ralph-config.json"


@pytest.fixture
def ralph_config():
    """Load ralph-config.json and return the pipelineStages section."""
    assert RALPH_CONFIG_PATH.exists(), f"ralph-config.json not found at {RALPH_CONFIG_PATH}"
    with open(RALPH_CONFIG_PATH, "r", encoding="utf-8") as f:
        config = json.load(f)
    assert "pipelineStages" in config, "ralph-config.json missing 'pipelineStages' key"
    return config["pipelineStages"]


class TestStageOrderSync:
    """Verify checkpoint.py and ralph-config.json stay in sync (Rule 31)."""

    def test_stage_order_matches(self, ralph_config):
        """STAGE_ORDER in checkpoint.py must match pipelineStages.order in ralph-config.json."""
        ralph_order = ralph_config["order"]
        assert ralph_order == STAGE_ORDER, (
            f"STAGE_ORDER mismatch!\n"
            f"  checkpoint.py: {STAGE_ORDER}\n"
            f"  ralph-config:  {ralph_order}\n"
            f"Update ralph-config.json to match checkpoint.py (Rule 31: checkpoint.py is truth)"
        )

    def test_stage_order_length(self, ralph_config):
        """Both lists must have the same number of stages."""
        assert len(ralph_config["order"]) == len(STAGE_ORDER)

    def test_stage_order_exact_sequence(self, ralph_config):
        """Order matters — stages must appear in the same sequence."""
        for i, (checkpoint_stage, ralph_stage) in enumerate(
            zip(STAGE_ORDER, ralph_config["order"])
        ):
            assert checkpoint_stage == ralph_stage, (
                f"Stage mismatch at index {i}: "
                f"checkpoint.py has '{checkpoint_stage}', "
                f"ralph-config.json has '{ralph_stage}'"
            )

    def test_legacy_stages_match(self, ralph_config):
        """LEGACY_STAGES in checkpoint.py must match pipelineStages.legacy in ralph-config.json."""
        ralph_legacy = ralph_config["legacy"]
        assert sorted(ralph_legacy) == sorted(LEGACY_STAGES), (
            f"LEGACY_STAGES mismatch!\n"
            f"  checkpoint.py: {sorted(LEGACY_STAGES)}\n"
            f"  ralph-config:  {sorted(ralph_legacy)}\n"
            f"Update ralph-config.json to match checkpoint.py (Rule 31)"
        )

    def test_legacy_stages_exact_content(self, ralph_config):
        """Every legacy stage in checkpoint.py must appear in ralph-config.json and vice versa."""
        checkpoint_set = set(LEGACY_STAGES)
        ralph_set = set(ralph_config["legacy"])
        missing_from_ralph = checkpoint_set - ralph_set
        extra_in_ralph = ralph_set - checkpoint_set
        assert not missing_from_ralph, f"Legacy stages in checkpoint.py missing from ralph-config: {missing_from_ralph}"
        assert not extra_in_ralph, f"Extra legacy stages in ralph-config not in checkpoint.py: {extra_in_ralph}"
