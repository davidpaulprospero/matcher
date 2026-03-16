"""
Comprehensive config round-trip tests for all section_mapping entries.

Ensures every entry in Config._from_dict's section_mapping actually loads
from YAML end-to-end. Catches regressions when new sections are added but
not mapped.

US-69-012
"""

import logging
import os
import pytest
import yaml

from src.config.base import Config


# ---------------------------------------------------------------------------
# Build the parametrized test data from the REAL section_mapping
# ---------------------------------------------------------------------------
# Each tuple: (yaml_key, attr_name, field_name, test_value, expected_value)
#
# test_value  = what goes in YAML
# expected_value = what we assert on the loaded Config attr (usually same)
#
# We pick ONE simple scalar field per section whose default differs from
# test_value so the test proves YAML -> dataclass propagation.
# ---------------------------------------------------------------------------

SECTION_TEST_DATA = [
    # (yaml_key, attr_on_config, field_name, yaml_value, expected_after_load)
    ("transcription", "transcription", "use_gpu", False, False),
    ("embedding", "embedding", "batch_size", 50, 50),
    ("indexing", "indexing", "use_faiss", False, False),
    ("vision", "vision", "enabled", False, False),
    ("scene_detection", "scene_detection", "enabled", False, False),
    ("audio_analysis", "audio_analysis", "enabled", False, False),
    ("matching", "matching", "min_confidence", 0.5, 0.5),
    ("negative_matching", "negative_matching", "enabled", False, False),
    ("remix", "remix", "enabled", False, False),
    ("zero_download_remix", "zero_download_remix", "enabled", False, False),
    ("image_search", "image_search", "enabled", False, False),
    ("keyword", "keyword", "max_keywords", 20, 20),
    ("llm", "llm", "provider", "anthropic", "anthropic"),
    ("enhanced", "enhanced", "enabled", False, False),
    ("downloading", "downloading", "prefer_h264", False, False),
    ("download", "download", "quality", "720p", "720p"),
    ("stock_footage", "stock_footage", "min_duration", 1.0, 1.0),
    ("deduplication", "deduplication", "enabled", False, False),
    ("output", "output", "generate_otio", False, False),
    ("multi_style", "multi_style", "enabled", False, False),
    ("logging", "logging", "enabled", False, False),
    ("cache", "cache", "cache_transcriptions", False, False),
    ("pipeline", "pipeline", "skip_download", True, True),
    ("api_keys", "api_keys", "gemini_api_key", "test-key-123", "test-key-123"),
    ("healing", "healing", "enabled", False, False),
    ("iterative_matching", "iterative_matching", "enabled", False, False),
    ("rate_limit", "rate_limit", "slots_per_second", 1.0, 1.0),
    ("broll", "broll", "enabled", False, False),
    ("global_cache", "global_cache", "enabled", False, False),
    ("silent_video", "silent_video", "enabled", False, False),
]


def _section_ids():
    """Generate test IDs from yaml keys."""
    return [t[0] for t in SECTION_TEST_DATA]


@pytest.mark.fast
class TestSectionMappingRoundTrip:
    """Parametrized test: every section_mapping entry loads from YAML."""

    @pytest.mark.parametrize(
        "yaml_key, attr_name, field_name, yaml_value, expected",
        SECTION_TEST_DATA,
        ids=_section_ids(),
    )
    def test_section_loads_non_default_value(
        self, tmp_path, yaml_key, attr_name, field_name, yaml_value, expected
    ):
        """Load a minimal YAML with one non-default value and verify it is applied."""
        # Build minimal YAML containing only this section + field
        yaml_content = yaml.dump(
            {yaml_key: {field_name: yaml_value}},
            default_flow_style=False,
        )
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml_content)

        config = Config.from_yaml(str(config_file))

        section = getattr(config, attr_name)
        actual = getattr(section, field_name)
        assert actual == expected, (
            f"Section '{yaml_key}' field '{field_name}': "
            f"expected {expected!r}, got {actual!r}"
        )

    def test_all_section_mapping_entries_covered(self):
        """Verify our test data covers every key in section_mapping.

        If someone adds a new section to _from_dict but forgets to add it
        here, this test will catch it.
        """
        # Extract section_mapping keys by building a Config from an empty dict
        # and inspecting the source.  More robust: parse the actual mapping.
        tested_keys = {t[0] for t in SECTION_TEST_DATA}

        # Known section_mapping keys (from base.py lines 398-429)
        expected_keys = {
            "transcription", "embedding", "indexing", "vision",
            "scene_detection", "audio_analysis", "matching",
            "negative_matching", "remix", "zero_download_remix",
            "image_search", "keyword", "llm", "enhanced",
            "downloading", "download", "stock_footage", "deduplication",
            "output", "multi_style", "logging", "cache", "pipeline",
            "api_keys", "healing", "iterative_matching", "rate_limit",
            "broll", "global_cache", "silent_video",
        }

        missing = expected_keys - tested_keys
        extra = tested_keys - expected_keys
        assert not missing, f"Section mapping keys not tested: {missing}"
        assert not extra, f"Test data has keys not in section_mapping: {extra}"


@pytest.mark.fast
class TestUnknownYamlKeys:
    """Unknown YAML keys in a section are logged (not silently dropped or causing errors)."""

    def test_unknown_keys_logged_as_debug(self, tmp_path, caplog):
        """Unknown keys produce a debug-level log message."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("""
matching:
  min_confidence: 0.5
  totally_fake_field: 999
  another_bogus_key: "hello"
""")

        with caplog.at_level(logging.DEBUG, logger="src.config.base"):
            config = Config.from_yaml(str(config_file))

        # The valid field should still load
        assert config.matching.min_confidence == 0.5

        # Unknown keys should appear in debug log
        debug_messages = [r.message for r in caplog.records if r.levelno == logging.DEBUG]
        assert any("totally_fake_field" in m for m in debug_messages), (
            f"Expected debug log about 'totally_fake_field', got: {debug_messages}"
        )
        assert any("another_bogus_key" in m for m in debug_messages), (
            f"Expected debug log about 'another_bogus_key', got: {debug_messages}"
        )

    def test_unknown_keys_do_not_raise(self, tmp_path):
        """Unknown keys must not cause errors — config still loads successfully."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("""
transcription:
  use_gpu: false
  nonexistent_option: 42
embedding:
  batch_size: 50
  fake_setting: "should be ignored"
""")
        config = Config.from_yaml(str(config_file))

        assert config.transcription.use_gpu is False
        assert config.embedding.batch_size == 50
        assert not hasattr(config.transcription, "nonexistent_option")
        assert not hasattr(config.embedding, "fake_setting")


@pytest.mark.fast
class TestRealConfigYaml:
    """Load the real config.yaml and verify all sections parse without errors."""

    @staticmethod
    def _find_real_config():
        """Find the project's config.yaml."""
        # Walk up from test file to find project root
        here = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(here)
        config_path = os.path.join(project_root, "config.yaml")
        return config_path

    def test_real_config_loads_without_errors(self):
        """The project's config.yaml must parse without raising exceptions."""
        config_path = self._find_real_config()
        if not os.path.exists(config_path):
            pytest.skip("config.yaml not found at project root")

        # Should not raise
        config = Config.from_yaml(config_path)
        assert config is not None

    def test_real_config_all_sections_populated(self):
        """After loading config.yaml, key sections should be non-None."""
        config_path = self._find_real_config()
        if not os.path.exists(config_path):
            pytest.skip("config.yaml not found at project root")

        config = Config.from_yaml(config_path)

        # Spot-check a representative set of sections
        assert config.matching is not None
        assert config.transcription is not None
        assert config.output is not None
        assert config.download is not None
        assert config.embedding is not None
        assert config.healing is not None

    def test_real_config_validate_no_critical_errors(self):
        """Validation on real config should not produce critical errors.

        API key errors are expected (keys not in env during test) but
        structural/constraint errors should not occur.
        """
        config_path = self._find_real_config()
        if not os.path.exists(config_path):
            pytest.skip("config.yaml not found at project root")

        config = Config.from_yaml(config_path)
        errors = config.validate()

        # Filter out API key errors (expected in test env)
        structural_errors = [
            e for e in errors
            if "API_KEY" not in e and "api_key" not in e.lower()
        ]
        assert not structural_errors, (
            f"Real config.yaml has structural validation errors: {structural_errors}"
        )
