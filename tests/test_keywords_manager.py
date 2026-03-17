"""
Unit tests for src/keywords/manager.py — the new standalone KeywordManager module.

Validates that keyword save/load/list operations work correctly when imported
from src.keywords (the canonical location after extraction from checkpoint.py).
"""

import pytest
import json
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

# Import from new canonical location
from src.keywords import KeywordManager, SavedKeywords, format_keyword_prompt
# Also verify backward-compatible re-export from checkpoint
from src.checkpoint import KeywordManager as CheckpointKeywordManager
from src.checkpoint import SavedKeywords as CheckpointSavedKeywords


@pytest.mark.fast
class TestNewModuleImports:
    """Verify imports from src.keywords resolve to same classes as checkpoint re-exports."""

    def test_keyword_manager_is_same_class(self):
        assert KeywordManager is CheckpointKeywordManager

    def test_saved_keywords_is_same_class(self):
        assert SavedKeywords is CheckpointSavedKeywords


@pytest.mark.fast
class TestSavedKeywordsFromNewModule:
    """Test SavedKeywords dataclass imported from src.keywords."""

    def test_create_defaults(self):
        sk = SavedKeywords()
        assert sk.name == ""
        assert sk.keywords == []
        assert sk.num_keywords == 0

    def test_create_with_data(self):
        sk = SavedKeywords(
            name="test",
            keywords=["python", "coding"],
            topic_context="programming",
            num_keywords=2
        )
        assert sk.name == "test"
        assert sk.keywords == ["python", "coding"]
        assert sk.topic_context == "programming"

    def test_to_dict(self):
        sk = SavedKeywords(name="test", keywords=["a", "b"], num_keywords=2)
        d = sk.to_dict()
        assert isinstance(d, dict)
        assert d["name"] == "test"
        assert d["keywords"] == ["a", "b"]

    def test_from_dict(self):
        data = {"name": "test", "keywords": ["x"], "num_keywords": 1}
        sk = SavedKeywords.from_dict(data)
        assert sk.name == "test"
        assert sk.keywords == ["x"]

    def test_from_dict_ignores_extra_keys(self):
        data = {"name": "test", "keywords": [], "unknown_field": True}
        sk = SavedKeywords.from_dict(data)
        assert sk.name == "test"

    def test_round_trip(self):
        original = SavedKeywords(
            name="round",
            keywords=["a", "b", "c"],
            topic_context="test topic",
            num_keywords=3,
            created_at="2026-01-01T00:00:00"
        )
        restored = SavedKeywords.from_dict(original.to_dict())
        assert restored.name == original.name
        assert restored.keywords == original.keywords
        assert restored.topic_context == original.topic_context


@pytest.mark.fast
class TestKeywordManagerFromNewModule:
    """Test KeywordManager imported from src.keywords."""

    def test_init_empty(self, tmp_path):
        km = KeywordManager(tmp_path)
        assert not km.has_presets()
        assert km.list_presets() == []

    def test_save_and_retrieve(self, tmp_path):
        km = KeywordManager(tmp_path)
        name = km.save_keywords(
            keywords=["alpha", "beta"],
            topic_context="greek letters",
            name="greek"
        )
        assert name == "greek"
        assert km.has_presets()

        preset = km.get_preset("greek")
        assert preset is not None
        assert preset.keywords == ["alpha", "beta"]
        assert preset.topic_context == "greek letters"
        assert preset.num_keywords == 2

    def test_save_auto_name(self, tmp_path):
        km = KeywordManager(tmp_path)
        name = km.save_keywords(keywords=["test"])
        assert name  # auto-generated timestamp name
        assert km.has_presets()

    def test_get_latest(self, tmp_path):
        km = KeywordManager(tmp_path)
        km.save_keywords(keywords=["old"], name="old")
        km.presets["old"].created_at = "2026-01-01T00:00:00"
        km.save_keywords(keywords=["new"], name="new")
        km.presets["new"].created_at = "2026-01-02T00:00:00"

        latest = km.get_latest()
        assert latest is not None
        assert latest.name == "new"

    def test_get_preset_not_found(self, tmp_path):
        km = KeywordManager(tmp_path)
        assert km.get_preset("nonexistent") is None

    def test_get_preset_none_when_empty(self, tmp_path):
        km = KeywordManager(tmp_path)
        assert km.get_preset() is None

    def test_list_presets_sorted(self, tmp_path):
        km = KeywordManager(tmp_path)
        km.presets["a"] = SavedKeywords(name="a", created_at="2026-01-01")
        km.presets["b"] = SavedKeywords(name="b", created_at="2026-01-03")
        km.presets["c"] = SavedKeywords(name="c", created_at="2026-01-02")

        listed = km.list_presets()
        assert [p.name for p in listed] == ["b", "c", "a"]

    def test_delete_preset(self, tmp_path):
        km = KeywordManager(tmp_path)
        km.save_keywords(keywords=["del"], name="todelete")
        assert km.delete_preset("todelete")
        assert not km.has_presets()

    def test_delete_nonexistent(self, tmp_path):
        km = KeywordManager(tmp_path)
        assert not km.delete_preset("nope")

    def test_persistence_across_instances(self, tmp_path):
        """Saved presets persist when a new KeywordManager is created for the same dir."""
        km1 = KeywordManager(tmp_path)
        km1.save_keywords(keywords=["persist"], name="p1")

        km2 = KeywordManager(tmp_path)
        assert km2.has_presets()
        assert km2.get_preset("p1").keywords == ["persist"]

    def test_load_old_format(self, tmp_path):
        """Old single-preset format is loaded as 'default'."""
        old_data = {"keywords": ["legacy"], "topic_context": "old format"}
        with open(tmp_path / "saved_keywords.json", "w") as f:
            json.dump(old_data, f)

        km = KeywordManager(tmp_path)
        assert km.has_presets()
        assert "default" in km.presets
        assert km.presets["default"].keywords == ["legacy"]

    def test_get_summary_empty(self, tmp_path):
        km = KeywordManager(tmp_path)
        assert "No saved" in km.get_summary()

    def test_get_summary_with_data(self, tmp_path):
        km = KeywordManager(tmp_path)
        km.presets["test"] = SavedKeywords(
            name="test",
            keywords=["a", "b", "c", "d"],
            created_at="2026-01-01T12:00:00",
            topic_context="test topic"
        )
        summary = km.get_summary()
        assert "test" in summary
        assert "a, b, c" in summary


@pytest.mark.fast
class TestFormatKeywordPromptFromNewModule:
    """Test format_keyword_prompt imported from src.keywords."""

    def test_format_prompt(self, tmp_path):
        km = KeywordManager(tmp_path)
        km.presets["t"] = SavedKeywords(
            name="t", keywords=["x"], created_at="2026-01-01"
        )
        result = format_keyword_prompt(km)
        assert "SAVED KEYWORDS FOUND" in result
        assert "[U]" in result
        assert "[L]" in result
        assert "[N]" in result
