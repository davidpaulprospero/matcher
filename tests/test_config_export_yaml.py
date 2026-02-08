"""
Tests for Config.to_dict() and Config.to_yaml() — effective config export.

US-80-008: Add effective config export to YAML.

Acceptance criteria:
- to_dict() returns config as nested dict suitable for YAML serialization
- to_yaml(path) writes the effective config to a YAML file
- Exported YAML can be loaded back with from_yaml() and produce equivalent config (roundtrip)
- Sensitive fields (api_keys) are redacted by default, with option to include them
- Roundtrip through to_yaml/from_yaml preserves all non-sensitive field values
"""

import os
import pytest
import yaml

from dataclasses import fields, is_dataclass, asdict
from src.config.base import Config


@pytest.mark.fast
class TestToDict:
    """Config.to_dict() returns a complete nested dict."""

    def test_returns_dict(self):
        config = Config()
        result = config.to_dict()
        assert isinstance(result, dict)

    def test_contains_all_section_mapping_keys(self):
        """to_dict() must include every section from _from_dict's section_mapping."""
        config = Config()
        result = config.to_dict()

        expected_sections = {
            'project', 'project_dir',
            'transcription', 'embedding', 'indexing', 'vision',
            'scene_detection', 'audio_analysis', 'matching',
            'negative_matching', 'remix', 'zero_download_remix',
            'image_search', 'keyword', 'llm', 'enhanced',
            'downloading', 'download', 'stock_footage', 'deduplication',
            'output', 'multi_style', 'logging', 'cache', 'pipeline',
            'api_keys', 'healing', 'iterative_matching', 'rate_limit',
            'broll', 'global_cache', 'silent_video',
        }

        for section in expected_sections:
            assert section in result, f"Missing section '{section}' in to_dict() output"

    def test_duration_tiers_exported(self):
        """Duration tiers should be in the export with YAML-style keys."""
        config = Config()
        result = config.to_dict()
        assert 'duration_tiers' in result
        dt = result['duration_tiers']
        for tier_name in ['short', 'medium', 'long', 'longer']:
            assert tier_name in dt, f"Missing tier '{tier_name}'"
            assert 'min' in dt[tier_name]
            assert 'max' in dt[tier_name]
            assert 'count' in dt[tier_name]

    def test_redacts_api_keys_by_default(self):
        """API keys should be redacted when redact_sensitive=True (default)."""
        config = Config()
        config.api_keys.gemini_api_key = "my-secret-key"
        result = config.to_dict()

        assert result['api_keys']['gemini_api_key'] == '***REDACTED***'

    def test_empty_api_keys_stay_empty(self):
        """Empty API key strings should remain empty, not get redacted."""
        config = Config()
        config.api_keys.gemini_api_key = ""
        result = config.to_dict()

        assert result['api_keys']['gemini_api_key'] == ''

    def test_include_sensitive_when_requested(self):
        """redact_sensitive=False should include actual API key values."""
        config = Config()
        config.api_keys.gemini_api_key = "real-key-value"
        result = config.to_dict(redact_sensitive=False)

        assert result['api_keys']['gemini_api_key'] == 'real-key-value'

    def test_dict_values_are_yaml_serializable(self):
        """The dict from to_dict() should be serializable to YAML without errors."""
        config = Config()
        result = config.to_dict()
        # Should not raise
        yaml_str = yaml.dump(result, default_flow_style=False)
        assert isinstance(yaml_str, str)
        assert len(yaml_str) > 0


@pytest.mark.fast
class TestToYaml:
    """Config.to_yaml() writes effective config to YAML."""

    def test_returns_yaml_string(self):
        config = Config()
        yaml_str = config.to_yaml()
        assert isinstance(yaml_str, str)
        # Verify it's valid YAML
        loaded = yaml.safe_load(yaml_str)
        assert isinstance(loaded, dict)

    def test_writes_to_file(self, tmp_path):
        config = Config()
        output_file = tmp_path / "exported.yaml"
        config.to_yaml(str(output_file))

        assert output_file.exists()
        content = output_file.read_text(encoding='utf-8')
        loaded = yaml.safe_load(content)
        assert isinstance(loaded, dict)
        assert 'matching' in loaded

    def test_redacts_by_default_in_file(self, tmp_path):
        config = Config()
        config.api_keys.gemini_api_key = "secret"
        output_file = tmp_path / "exported.yaml"
        config.to_yaml(str(output_file))

        content = output_file.read_text(encoding='utf-8')
        assert 'secret' not in content
        assert 'REDACTED' in content

    def test_unredacted_export(self, tmp_path):
        config = Config()
        config.api_keys.gemini_api_key = "my-key"
        output_file = tmp_path / "exported.yaml"
        config.to_yaml(str(output_file), redact_sensitive=False)

        content = output_file.read_text(encoding='utf-8')
        assert 'my-key' in content


@pytest.mark.fast
class TestRoundTrip:
    """Exported YAML can be loaded back and produce an equivalent config."""

    def test_roundtrip_preserves_non_sensitive_fields(self, tmp_path):
        """Core roundtrip test: export -> reload -> compare non-sensitive values."""
        # Create config with non-default values
        original = Config()
        original.matching.min_confidence = 0.42
        original.transcription.use_gpu = False
        original.embedding.batch_size = 77
        original.keyword.max_keywords = 15
        original.output.generate_otio = False
        original.download.quality = "1080p"

        # Export to YAML (unredacted so api_keys round-trip too)
        export_path = tmp_path / "roundtrip.yaml"
        original.to_yaml(str(export_path), redact_sensitive=False)

        # Reload
        reloaded = Config.from_yaml(str(export_path))

        # Compare key fields
        assert reloaded.matching.min_confidence == original.matching.min_confidence
        assert reloaded.transcription.use_gpu == original.transcription.use_gpu
        assert reloaded.embedding.batch_size == original.embedding.batch_size
        assert reloaded.keyword.max_keywords == original.keyword.max_keywords
        assert reloaded.output.generate_otio == original.output.generate_otio
        assert reloaded.download.quality == original.download.quality

    def test_roundtrip_all_sections_present(self, tmp_path):
        """After roundtrip, all sections should still be present and non-None."""
        original = Config()
        export_path = tmp_path / "roundtrip_all.yaml"
        original.to_yaml(str(export_path), redact_sensitive=False)

        reloaded = Config.from_yaml(str(export_path))

        sections = [
            'transcription', 'embedding', 'indexing', 'vision',
            'scene_detection', 'audio_analysis', 'matching',
            'negative_matching', 'remix', 'zero_download_remix',
            'image_search', 'keyword', 'llm', 'enhanced',
            'downloading', 'download', 'stock_footage', 'deduplication',
            'output', 'multi_style', 'logging', 'cache', 'pipeline',
            'api_keys', 'healing', 'iterative_matching', 'rate_limit',
            'broll', 'global_cache', 'silent_video',
        ]

        for section_name in sections:
            orig_section = getattr(original, section_name)
            reload_section = getattr(reloaded, section_name)
            assert reload_section is not None, (
                f"Section '{section_name}' is None after roundtrip"
            )
            assert type(reload_section) == type(orig_section), (
                f"Section '{section_name}' type mismatch: "
                f"{type(orig_section).__name__} vs {type(reload_section).__name__}"
            )

    def test_roundtrip_duration_tiers(self, tmp_path):
        """Duration tiers survive the roundtrip."""
        original = Config()
        original.duration_tiers.short.min_seconds = 5
        original.duration_tiers.short.max_seconds = 30
        original.duration_tiers.short.videos_per_keyword = 10

        export_path = tmp_path / "roundtrip_dt.yaml"
        original.to_yaml(str(export_path), redact_sensitive=False)

        reloaded = Config.from_yaml(str(export_path))

        assert reloaded.duration_tiers.short.min_seconds == 5
        assert reloaded.duration_tiers.short.max_seconds == 30
        assert reloaded.duration_tiers.short.videos_per_keyword == 10

    def test_roundtrip_with_redacted_keys_loads_safely(self, tmp_path):
        """Redacted export loads without error (keys become redacted strings)."""
        original = Config()
        original.api_keys.gemini_api_key = "real-secret"

        export_path = tmp_path / "roundtrip_redacted.yaml"
        original.to_yaml(str(export_path), redact_sensitive=True)

        # Should load without error
        reloaded = Config.from_yaml(str(export_path))
        # The key value will be the redacted placeholder, not the original
        assert reloaded.api_keys.gemini_api_key == '***REDACTED***'

    def test_roundtrip_section_fields_match(self, tmp_path):
        """Spot-check: every scalar field in a few sections matches after roundtrip."""
        original = Config()
        export_path = tmp_path / "roundtrip_fields.yaml"
        original.to_yaml(str(export_path), redact_sensitive=False)

        reloaded = Config.from_yaml(str(export_path))

        # Check a few representative sections field-by-field
        check_sections = ['healing', 'iterative_matching', 'rate_limit', 'broll']
        for section_name in check_sections:
            orig = getattr(original, section_name)
            reloaded_sec = getattr(reloaded, section_name)
            for f in fields(type(orig)):
                orig_val = getattr(orig, f.name)
                reload_val = getattr(reloaded_sec, f.name)
                # Only compare simple scalars (str, int, float, bool)
                if isinstance(orig_val, (str, int, float, bool)):
                    assert reload_val == orig_val, (
                        f"{section_name}.{f.name}: {orig_val!r} != {reload_val!r}"
                    )
