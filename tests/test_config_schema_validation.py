"""
Tests for US-85-006: Config schema validation that fails fast on invalid YAML.

Verifies that validate_config_schema() catches unknown sections (warning)
and type mismatches (ConfigValidationError) before pipeline execution.
"""

import logging
import pytest

from src.config.schema_validation import (
    ConfigValidationError,
    validate_config_schema,
    _KNOWN_SECTION_NAMES,
    _get_primitive_type,
    _check_field_type,
    _validate_section_fields,
)


# =============================================================================
# Unknown section names → warning (not error, for forward compat)
# =============================================================================

@pytest.mark.fast
class TestUnknownSectionWarning:
    """Unknown top-level YAML keys produce warnings, not errors."""

    def test_unknown_section_triggers_warning(self, caplog):
        """A YAML key not in _KNOWN_SECTION_NAMES logs a warning."""
        data = {
            'matching': {'min_confidence': 0.5},
            'totally_unknown_section': {'foo': 'bar'},
        }

        with caplog.at_level(logging.WARNING):
            result = validate_config_schema(data, raise_on_error=False)

        assert any('totally_unknown_section' in msg for msg in result)
        assert any('totally_unknown_section' in r.message for r in caplog.records)

    def test_multiple_unknown_sections(self):
        """Multiple unknown sections each produce their own warning."""
        data = {
            'fake_section_a': {'x': 1},
            'fake_section_b': {'y': 2},
        }

        result = validate_config_schema(data, raise_on_error=False)

        unknown_warnings = [r for r in result if 'Unknown config section' in r]
        assert len(unknown_warnings) == 2

    def test_known_sections_no_warning(self, caplog):
        """All known section names produce zero warnings."""
        data = {
            'matching': {'min_confidence': 0.5},
            'download': {'max_retries': 3},
        }

        with caplog.at_level(logging.WARNING):
            result = validate_config_schema(data, raise_on_error=False)

        unknown_warnings = [r for r in result if 'Unknown config section' in r]
        assert len(unknown_warnings) == 0

    def test_unknown_section_does_not_raise(self):
        """Unknown sections warn but do NOT raise even with raise_on_error=True."""
        data = {
            'nonexistent_module': {'a': 1},
        }

        # Should not raise — unknown sections are warnings only
        result = validate_config_schema(data, raise_on_error=True)
        assert any('nonexistent_module' in r for r in result)


# =============================================================================
# Type mismatch → ConfigValidationError
# =============================================================================

@pytest.mark.fast
class TestTypeMismatchError:
    """String where number expected, etc. raises ConfigValidationError."""

    def test_string_where_int_expected_raises(self):
        """A string value for an int field raises ConfigValidationError."""
        data = {
            'transcription': {'max_workers': 'not_a_number'},
        }

        with pytest.raises(ConfigValidationError, match="max_workers.*expected int.*str"):
            validate_config_schema(data, raise_on_error=True)

    def test_string_where_float_expected_raises(self):
        """A string value for a float field raises ConfigValidationError."""
        data = {
            'matching': {'min_confidence': 'high'},
        }

        with pytest.raises(ConfigValidationError, match="min_confidence.*expected number.*str"):
            validate_config_schema(data, raise_on_error=True)

    def test_string_where_bool_expected_raises(self):
        """A string value for a bool field raises ConfigValidationError."""
        data = {
            'output': {'generate_otio': 'yes'},
        }

        with pytest.raises(ConfigValidationError, match="generate_otio.*expected bool.*str"):
            validate_config_schema(data, raise_on_error=True)

    def test_int_where_float_ok(self):
        """Int value for float field is acceptable (numeric promotion)."""
        data = {
            'matching': {'min_confidence': 1},
        }

        # Should not raise — int is valid for float fields
        result = validate_config_schema(data, raise_on_error=True)
        type_errors = [r for r in result if 'expected' in r]
        assert len(type_errors) == 0

    def test_valid_types_no_error(self):
        """Correct types produce no errors."""
        data = {
            'transcription': {'max_workers': 4},
            'matching': {'min_confidence': 0.7},
            'output': {'generate_otio': True},
        }

        result = validate_config_schema(data, raise_on_error=True)
        type_errors = [r for r in result if 'expected' in r]
        assert len(type_errors) == 0

    def test_raise_on_error_false_returns_errors(self):
        """With raise_on_error=False, type errors are returned not raised."""
        data = {
            'transcription': {'max_workers': 'bad'},
        }

        result = validate_config_schema(data, raise_on_error=False)
        assert any('max_workers' in r for r in result)

    def test_none_value_is_acceptable(self):
        """None values are OK (Optional fields)."""
        data = {
            'download': {'cookies_path': None},
        }

        result = validate_config_schema(data, raise_on_error=True)
        type_errors = [r for r in result if 'expected' in r]
        assert len(type_errors) == 0


# =============================================================================
# Nested dataclass validation
# =============================================================================

@pytest.mark.fast
class TestNestedValidation:
    """Nested config objects are recursively validated."""

    def test_nested_type_mismatch_raises(self):
        """Type error in nested section raises ConfigValidationError."""
        data = {
            'download': {
                'caption_first': {
                    'enabled': 'not_bool',
                },
            },
        }

        with pytest.raises(ConfigValidationError, match="caption_first.*enabled.*expected bool"):
            validate_config_schema(data, raise_on_error=True)


# =============================================================================
# Empty / missing data
# =============================================================================

@pytest.mark.fast
class TestEdgeCases:
    """Edge cases: empty data, None, non-dict."""

    def test_empty_dict_no_error(self):
        """Empty config data produces no errors."""
        result = validate_config_schema({}, raise_on_error=True)
        assert result == []

    def test_none_data_no_error(self):
        """None data produces no errors."""
        result = validate_config_schema(None, raise_on_error=True)
        assert result == []

    def test_empty_section_no_error(self):
        """A section with empty dict value is fine."""
        data = {'matching': {}}
        result = validate_config_schema(data, raise_on_error=True)
        assert result == []


# =============================================================================
# Helper function tests
# =============================================================================

@pytest.mark.fast
class TestHelpers:
    """Test internal helper functions."""

    def test_get_primitive_type_int(self):
        assert _get_primitive_type(int) is int

    def test_get_primitive_type_float(self):
        assert _get_primitive_type(float) is float

    def test_get_primitive_type_str(self):
        assert _get_primitive_type(str) is str

    def test_get_primitive_type_bool(self):
        assert _get_primitive_type(bool) is bool

    def test_get_primitive_type_complex_returns_none(self):
        from typing import List
        assert _get_primitive_type(List[str]) is None

    def test_check_field_type_bool_rejects_int(self):
        """bool field should reject int (even though bool is subclass of int)."""
        err = _check_field_type('test', 'flag', 1, bool)
        assert err is not None
        assert 'expected bool' in err

    def test_check_field_type_int_rejects_bool(self):
        """int field should reject bool."""
        err = _check_field_type('test', 'count', True, int)
        assert err is not None
        assert 'expected int' in err

    def test_known_section_names_includes_core(self):
        """Core sections are in the known set."""
        assert 'matching' in _KNOWN_SECTION_NAMES
        assert 'download' in _KNOWN_SECTION_NAMES
        assert 'output' in _KNOWN_SECTION_NAMES
        assert 'project' in _KNOWN_SECTION_NAMES


# =============================================================================
# Integration: validate_config_schema called from load_config
# =============================================================================

@pytest.mark.fast
class TestLoadConfigIntegration:
    """validate_config_schema is called during Config.from_yaml()."""

    def test_from_yaml_with_bad_type_raises(self, tmp_path):
        """Config.from_yaml() raises ConfigValidationError on type mismatch."""
        import yaml
        config_file = tmp_path / "bad_config.yaml"
        config_file.write_text(yaml.dump({
            'transcription': {'max_workers': 'not_a_number'},
        }))

        from src.config.base import Config
        with pytest.raises(ConfigValidationError):
            Config.from_yaml(str(config_file))

    def test_from_yaml_with_unknown_section_warns(self, tmp_path, caplog):
        """Config.from_yaml() warns on unknown sections but succeeds."""
        import yaml
        config_file = tmp_path / "unknown_section.yaml"
        config_file.write_text(yaml.dump({
            'mystery_section': {'x': 1},
        }))

        from src.config.base import Config
        with caplog.at_level(logging.WARNING):
            config = Config.from_yaml(str(config_file))

        # Should succeed (unknown sections are warnings only)
        assert config is not None
        assert any('mystery_section' in r.message for r in caplog.records)

    def test_from_yaml_valid_config_succeeds(self, tmp_path):
        """Valid config loads without error."""
        import yaml
        config_file = tmp_path / "good_config.yaml"
        config_file.write_text(yaml.dump({
            'matching': {'min_confidence': 0.5},
            'transcription': {'max_workers': 2},
        }))

        from src.config.base import Config
        config = Config.from_yaml(str(config_file))
        assert config is not None
        assert config.matching.min_confidence == 0.5
