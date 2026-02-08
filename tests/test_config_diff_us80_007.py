"""
Tests for config_diff() utility — US-80-007.

Verifies recursive dataclass comparison, field path formatting,
and correct exclusion of equal values.
"""

import copy
import pytest
from dataclasses import dataclass, field

from src.config.utils import config_diff
from src.config.base import Config


class TestConfigDiffBasic:
    """Core config_diff behaviour."""

    def test_identical_configs_return_empty_dict(self):
        """AC: diff between identical configs returns empty dict."""
        a = Config()
        b = Config()
        assert config_diff(a, b) == {}

    def test_single_field_change_detected(self):
        """AC: diff with matching.min_confidence=0.8 shows only that field."""
        a = Config()
        b = copy.deepcopy(a)
        b.matching.min_confidence = 0.8

        diff = config_diff(a, b)

        assert "matching.min_confidence" in diff
        assert diff["matching.min_confidence"]["old"] == a.matching.min_confidence
        assert diff["matching.min_confidence"]["new"] == 0.8
        # Only that field should differ
        assert len(diff) == 1

    def test_diff_includes_field_path_old_new(self):
        """AC: Diff output includes field path, old value, and new value."""
        a = Config()
        b = copy.deepcopy(a)
        b.embedding.batch_size = 999

        diff = config_diff(a, b)

        assert "embedding.batch_size" in diff
        entry = diff["embedding.batch_size"]
        assert "old" in entry
        assert "new" in entry
        assert entry["new"] == 999

    def test_nested_dataclass_recursion(self):
        """AC: Nested dataclass fields are recursively compared."""
        a = Config()
        b = copy.deepcopy(a)
        # Change a deeply nested field
        b.download.caption_first.enabled = not a.download.caption_first.enabled

        diff = config_diff(a, b)

        assert any("download.caption_first" in k for k in diff)

    def test_equal_values_excluded(self):
        """AC: Fields with equal values are excluded from the diff."""
        a = Config()
        b = copy.deepcopy(a)
        # No changes
        diff = config_diff(a, b)
        assert len(diff) == 0

        # One change — only that field appears
        b.transcription.use_gpu = not a.transcription.use_gpu
        diff = config_diff(a, b)
        assert len(diff) == 1
        assert "transcription.use_gpu" in diff


class TestConfigDiffEdgeCases:
    """Edge-case and multi-field scenarios."""

    def test_multiple_field_changes(self):
        """Multiple changes across sections all appear."""
        a = Config()
        b = copy.deepcopy(a)
        b.matching.min_confidence = 0.99
        b.embedding.batch_size = 42
        b.transcription.use_gpu = not a.transcription.use_gpu

        diff = config_diff(a, b)

        assert len(diff) == 3
        assert "matching.min_confidence" in diff
        assert "embedding.batch_size" in diff
        assert "transcription.use_gpu" in diff

    def test_top_level_scalar_field(self):
        """Top-level non-dataclass fields (e.g., project_dir) are compared."""
        a = Config()
        b = copy.deepcopy(a)
        b.project_dir = "/some/other/path"

        diff = config_diff(a, b)

        assert "project_dir" in diff
        assert diff["project_dir"]["new"] == "/some/other/path"

    def test_plain_dataclass_comparison(self):
        """config_diff works on arbitrary dataclasses, not just Config."""

        @dataclass
        class Inner:
            x: int = 1

        @dataclass
        class Outer:
            inner: Inner = field(default_factory=Inner)
            name: str = "default"

        a = Outer()
        b = Outer(inner=Inner(x=5), name="changed")

        diff = config_diff(a, b)

        assert diff == {
            "inner.x": {"old": 1, "new": 5},
            "name": {"old": "default", "new": "changed"},
        }

    def test_non_dataclass_inputs(self):
        """Non-dataclass inputs compared as plain values."""
        assert config_diff(1, 2, _prefix="val") == {"val": {"old": 1, "new": 2}}
        assert config_diff("a", "a", _prefix="val") == {}
