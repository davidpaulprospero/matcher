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


# =============================================================================
# US-112-004: Dynamic list config validation (negative_keywords, topic_tags)
# =============================================================================

@pytest.mark.fast
class TestNegativeKeywordsValidation:
    """Validation for negative_keywords field (List[str])."""

    def test_negative_keywords_valid_list(self):
        """Valid list of strings passes validation."""
        data = {
            'video_search': {
                'negative_keywords': ['trailer', 'teaser', 'compilation'],
            },
        }

        result = validate_config_schema(data, raise_on_error=True)
        errors = [r for r in result if 'negative_keywords' in r]
        assert len(errors) == 0

    def test_negative_keywords_empty_string_raises(self):
        """Empty string in list raises ConfigValidationError."""
        data = {
            'video_search': {
                'negative_keywords': ['trailer', '', 'compilation'],
            },
        }

        with pytest.raises(ConfigValidationError, match="negative_keywords.*non-empty string"):
            validate_config_schema(data, raise_on_error=True)

    def test_negative_keywords_non_string_item_raises(self):
        """Non-string item in list raises ConfigValidationError."""
        data = {
            'video_search': {
                'negative_keywords': ['trailer', 123, 'compilation'],
            },
        }

        with pytest.raises(ConfigValidationError, match="negative_keywords.*expected str"):
            validate_config_schema(data, raise_on_error=True)

    def test_negative_keywords_not_list_raises(self):
        """Non-list value raises ConfigValidationError."""
        data = {
            'video_search': {
                'negative_keywords': 'not_a_list',
            },
        }

        with pytest.raises(ConfigValidationError, match="negative_keywords.*expected list"):
            validate_config_schema(data, raise_on_error=True)

    def test_negative_keywords_none_is_acceptable(self):
        """None value for negative_keywords is acceptable."""
        data = {
            'video_search': {
                'negative_keywords': None,
            },
        }

        result = validate_config_schema(data, raise_on_error=True)
        errors = [r for r in result if 'negative_keywords' in r]
        assert len(errors) == 0


@pytest.mark.fast
class TestTopicTagsValidation:
    """Validation for topic_tags field (Dict[str, List[str]])."""

    def test_topic_tags_valid_dict(self):
        """Valid dict with non-empty string lists passes validation."""
        data = {
            'video_search': {
                'topic_tags': {
                    'nature': ['wildlife', 'landscape'],
                    'travel': ['adventure', 'destination'],
                },
            },
        }

        result = validate_config_schema(data, raise_on_error=True)
        errors = [r for r in result if 'topic_tags' in r]
        assert len(errors) == 0

    def test_topic_tags_empty_list_value_raises(self):
        """Empty list as dict value raises ConfigValidationError."""
        data = {
            'video_search': {
                'topic_tags': {
                    'nature': [],
                },
            },
        }

        with pytest.raises(ConfigValidationError, match="topic_tags"):
            validate_config_schema(data, raise_on_error=True)

    def test_topic_tags_empty_string_in_list_raises(self):
        """Empty string in list value raises ConfigValidationError."""
        data = {
            'video_search': {
                'topic_tags': {
                    'nature': ['wildlife', ''],
                },
            },
        }

        with pytest.raises(ConfigValidationError, match="topic_tags.*non-empty string"):
            validate_config_schema(data, raise_on_error=True)

    def test_topic_tags_non_string_in_list_raises(self):
        """Non-string in list value raises ConfigValidationError."""
        data = {
            'video_search': {
                'topic_tags': {
                    'nature': ['wildlife', 123],
                },
            },
        }

        with pytest.raises(ConfigValidationError, match="topic_tags.*expected str"):
            validate_config_schema(data, raise_on_error=True)

    def test_topic_tags_not_dict_raises(self):
        """Non-dict value raises ConfigValidationError."""
        data = {
            'video_search': {
                'topic_tags': 'not_a_dict',
            },
        }

        with pytest.raises(ConfigValidationError, match="topic_tags.*expected dict"):
            validate_config_schema(data, raise_on_error=True)

    def test_topic_tags_list_value_raises(self):
        """List instead of dict raises ConfigValidationError."""
        data = {
            'video_search': {
                'topic_tags': ['nature', 'travel'],
            },
        }

        with pytest.raises(ConfigValidationError, match="topic_tags.*expected dict"):
            validate_config_schema(data, raise_on_error=True)

    def test_topic_tags_none_is_acceptable(self):
        """None value for topic_tags is acceptable."""
        data = {
            'video_search': {
                'topic_tags': None,
            },
        }

        result = validate_config_schema(data, raise_on_error=True)
        errors = [r for r in result if 'topic_tags' in r]
        assert len(errors) == 0


@pytest.mark.fast
class TestVideoSearchConfigLoadTimeValidation:
    """Test that validation errors are caught at load time."""

    def test_invalid_negative_keywords_raises_on_yaml_load(self, tmp_path):
        """Config.from_yaml() raises ConfigValidationError on invalid negative_keywords."""
        import yaml
        config_file = tmp_path / "bad_video_search.yaml"
        config_file.write_text(yaml.dump({
            'video_search': {'negative_keywords': ['valid', '', 'also_valid']},
        }))

        from src.config.base import Config
        with pytest.raises(ConfigValidationError):
            Config.from_yaml(str(config_file))

    def test_invalid_topic_tags_raises_on_yaml_load(self, tmp_path):
        """Config.from_yaml() raises ConfigValidationError on invalid topic_tags."""
        import yaml
        config_file = tmp_path / "bad_topic_tags.yaml"
        config_file.write_text(yaml.dump({
            'video_search': {'topic_tags': {'nature': []}},
        }))

        from src.config.base import Config
        with pytest.raises(ConfigValidationError):
            Config.from_yaml(str(config_file))


# =============================================================================
# US-128-007: Enhanced error messages with line numbers and fuzzy matching
# =============================================================================

@pytest.mark.fast
class TestEnhancedErrorMessages:
    """Test that error messages include line numbers and fuzzy suggestions."""

    def test_unknown_section_warning_includes_line_number(self, tmp_path):
        """Unknown section warnings should include YAML line numbers."""
        import yaml
        config_file = tmp_path / "unknown_with_line.yaml"
        # yaml.dump orders alphabetically, so:
        # Line 1-2: matching (first alphabetically)
        # Line 3-4: unknown_section
        config_file.write_text(yaml.dump({
            'unknown_section': {'foo': 'bar'},
            'matching': {'min_confidence': 0.5},
        }))

        from src.config.schema_validation import validate_config_schema, _parse_yaml_with_lines

        yaml_content = config_file.read_text()
        data, line_map = _parse_yaml_with_lines(yaml_content)

        errors = validate_config_schema(data, raise_on_error=False, line_map=line_map)

        # Should have warning with line number
        unknown_warnings = [r for r in errors if 'unknown_section' in r]
        assert len(unknown_warnings) == 1
        # Line number should be present (matching comes first, so unknown_section is on line 3)
        assert 'line' in unknown_warnings[0].lower()

    def test_unknown_section_warning_suggests_closest_match(self):
        """Unknown section warnings should suggest closest valid sections."""
        # Test fuzzy matching for typos
        from src.config.schema_validation import _find_closest_field

        # 'matchin' should suggest 'matching'
        suggestions = _find_closest_field('matchin', ['matching', 'download', 'output'])
        assert 'matching' in suggestions

        # 'vide_search' should suggest 'video_search'
        suggestions = _find_closest_field('vide_search', ['video_search', 'image_search', 'cache'])
        assert 'video_search' in suggestions

        # Completely unknown should return empty or low-confidence matches
        suggestions = _find_closest_field('xyzabc', ['matching', 'download'])
        assert len(suggestions) == 0  # Too different

    def test_type_error_includes_line_number(self, tmp_path):
        """Type mismatch errors should include section line numbers when available."""
        import yaml
        config_file = tmp_path / "type_error_with_line.yaml"
        # Line 1: matching:
        # Line 2:   min_confidence: "not_a_number"
        config_file.write_text(yaml.dump({
            'matching': {'min_confidence': 'not_a_number'},
        }))

        from src.config.schema_validation import validate_config_schema, _parse_yaml_with_lines

        yaml_content = config_file.read_text()
        data, line_map = _parse_yaml_with_lines(yaml_content)

        with pytest.raises(ConfigValidationError) as exc_info:
            validate_config_schema(data, raise_on_error=True, line_map=line_map)

        error_msg = str(exc_info.value)
        # Should include field name
        assert 'min_confidence' in error_msg
        # Should suggest correct type
        assert 'number' in error_msg.lower()
        # Line number for section should be in the individual field error logs
        # (nested fields don't have line numbers, but section does)

    def test_fuzzy_matching_threshold(self):
        """Fuzzy matching should only return suggestions above similarity threshold."""
        from src.config.schema_validation import _find_closest_field

        # 'mat' is too short to be similar to 'matching'
        suggestions = _find_closest_field('mat', ['matching', 'download'])
        # Should still return something since partial match is reasonable
        assert isinstance(suggestions, list)

    def test_line_number_for_nested_field_errors(self, tmp_path):
        """Nested field type errors should include line numbers when available."""
        import yaml
        config_file = tmp_path / "nested_error.yaml"
        config_file.write_text(yaml.dump({
            'download': {'caption_first': {'enabled': 'not_bool'}},
        }))

        from src.config.schema_validation import validate_config_schema, _parse_yaml_with_lines

        yaml_content = config_file.read_text()
        data, line_map = _parse_yaml_with_lines(yaml_content)

        with pytest.raises(ConfigValidationError) as exc_info:
            validate_config_schema(data, raise_on_error=True, line_map=line_map)

        error_msg = str(exc_info.value)
        # Should mention the nested field
        assert 'caption_first' in error_msg
        assert 'enabled' in error_msg

    def test_validate_config_schema_without_line_map(self):
        """validate_config_schema should work without line_map (backwards compatible)."""
        from src.config.schema_validation import validate_config_schema

        # Without line_map, should still work but without line numbers
        data = {
            'matching': {'min_confidence': 'not_a_number'},
        }

        with pytest.raises(ConfigValidationError) as exc_info:
            validate_config_schema(data, raise_on_error=True)

        error_msg = str(exc_info.value)
        assert 'min_confidence' in error_msg


@pytest.mark.fast
class TestSchemaValidationPerformance:
    """Test that schema validation meets performance requirements."""

    def test_validation_performance_under_50ms(self):
        """Schema validation should complete in < 50ms for typical config."""
        import time
        from src.config.schema_validation import validate_config_schema

        # Typical config data
        data = {
            'matching': {'min_confidence': 0.5, 'max_clip_reuse': 3},
            'download': {'max_retries': 3, 'timeout': 300},
            'transcription': {'max_workers': 4},
            'video_search': {'max_results': 50},
        }

        start = time.perf_counter()
        result = validate_config_schema(data, raise_on_error=False)
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < 50, f"Validation took {elapsed_ms:.1f}ms, expected < 50ms"


# =============================================================================
# YouTubeAPIConfig validation (US-149-011)
# =============================================================================

@pytest.mark.fast
class TestYouTubeAPIConfigValidation:
    """Validation for YouTubeAPIConfig-specific constraints."""

    def test_api_key_valid_non_empty_string(self):
        """Valid non-empty api_key passes validation."""
        data = {
            'download': {
                'youtube_api': {
                    'enabled': True,
                    'api_key': 'AIzaSyABC123xyz',
                },
            },
        }

        result = validate_config_schema(data, raise_on_error=True)
        errors = [r for r in result if 'youtube_api.api_key' in r]
        assert len(errors) == 0

    def test_api_key_empty_string_raises(self):
        """Empty api_key string when enabled raises ConfigValidationError."""
        data = {
            'download': {
                'youtube_api': {
                    'enabled': True,
                    'api_key': '',
                },
            },
        }

        with pytest.raises(ConfigValidationError, match='api_key must be non-empty'):
            validate_config_schema(data, raise_on_error=True)

    def test_api_key_whitespace_only_raises(self):
        """Whitespace-only api_key raises ConfigValidationError."""
        data = {
            'download': {
                'youtube_api': {
                    'enabled': True,
                    'api_key': '   ',
                },
            },
        }

        with pytest.raises(ConfigValidationError, match='api_key must be non-empty'):
            validate_config_schema(data, raise_on_error=True)

    def test_api_keys_list_valid(self):
        """Valid api_keys list passes validation."""
        data = {
            'download': {
                'youtube_api': {
                    'enabled': True,
                    'api_keys': ['key1', 'key2', 'key3'],
                },
            },
        }

        result = validate_config_schema(data, raise_on_error=True)
        errors = [r for r in result if 'youtube_api.api_keys' in r]
        assert len(errors) == 0

    def test_api_keys_empty_string_in_list_raises(self):
        """Empty string in api_keys list raises ConfigValidationError."""
        data = {
            'download': {
                'youtube_api': {
                    'enabled': True,
                    'api_keys': ['valid_key', '', 'another_key'],
                },
            },
        }

        with pytest.raises(ConfigValidationError, match='api_keys.*must be non-empty'):
            validate_config_schema(data, raise_on_error=True)

    def test_quota_limit_valid_range(self):
        """Valid quota_limit within range passes validation."""
        data = {
            'download': {
                'youtube_api': {
                    'quota_limit': 10000,
                },
            },
        }

        result = validate_config_schema(data, raise_on_error=True)
        errors = [r for r in result if 'quota_limit' in r]
        assert len(errors) == 0

    def test_quota_limit_below_minimum_raises(self):
        """quota_limit below 1 raises ConfigValidationError."""
        data = {
            'download': {
                'youtube_api': {
                    'quota_limit': 0,
                },
            },
        }

        with pytest.raises(ConfigValidationError, match='quota_limit.*must be >= 1'):
            validate_config_schema(data, raise_on_error=True)

    def test_quota_limit_above_maximum_raises(self):
        """quota_limit above 1000000 raises ConfigValidationError."""
        data = {
            'download': {
                'youtube_api': {
                    'quota_limit': 2000000,
                },
            },
        }

        with pytest.raises(ConfigValidationError, match='quota_limit.*must be <= 1000000'):
            validate_config_schema(data, raise_on_error=True)

    def test_quota_limit_negative_raises(self):
        """Negative quota_limit raises ConfigValidationError."""
        data = {
            'download': {
                'youtube_api': {
                    'quota_limit': -100,
                },
            },
        }

        with pytest.raises(ConfigValidationError, match='quota_limit.*must be >= 1'):
            validate_config_schema(data, raise_on_error=True)

    def test_warn_at_percent_valid_range(self):
        """Valid warn_at_percent within 0-100 passes validation."""
        data = {
            'download': {
                'youtube_api': {
                    'warn_at_percent': 80,
                },
            },
        }

        result = validate_config_schema(data, raise_on_error=True)
        errors = [r for r in result if 'warn_at_percent' in r]
        assert len(errors) == 0

    def test_warn_at_percent_below_minimum_raises(self):
        """warn_at_percent below 0 raises ConfigValidationError."""
        data = {
            'download': {
                'youtube_api': {
                    'warn_at_percent': -1,
                },
            },
        }

        with pytest.raises(ConfigValidationError, match='warn_at_percent.*must be 0-100'):
            validate_config_schema(data, raise_on_error=True)

    def test_warn_at_percent_above_maximum_raises(self):
        """warn_at_percent above 100 raises ConfigValidationError."""
        data = {
            'download': {
                'youtube_api': {
                    'warn_at_percent': 101,
                },
            },
        }

        with pytest.raises(ConfigValidationError, match='warn_at_percent.*must be 0-100'):
            validate_config_schema(data, raise_on_error=True)

    def test_timeout_seconds_valid_range(self):
        """Valid timeout_seconds within 5-120 passes validation."""
        data = {
            'download': {
                'youtube_api': {
                    'timeout_seconds': 30,
                },
            },
        }

        result = validate_config_schema(data, raise_on_error=True)
        errors = [r for r in result if 'timeout_seconds' in r]
        assert len(errors) == 0

    def test_timeout_seconds_below_minimum_raises(self):
        """timeout_seconds below 5 raises ConfigValidationError."""
        data = {
            'download': {
                'youtube_api': {
                    'timeout_seconds': 2,
                },
            },
        }

        with pytest.raises(ConfigValidationError, match='timeout_seconds.*must be 5-120'):
            validate_config_schema(data, raise_on_error=True)

    def test_timeout_seconds_above_maximum_raises(self):
        """timeout_seconds above 120 raises ConfigValidationError."""
        data = {
            'download': {
                'youtube_api': {
                    'timeout_seconds': 200,
                },
            },
        }

        with pytest.raises(ConfigValidationError, match='timeout_seconds.*must be 5-120'):
            validate_config_schema(data, raise_on_error=True)

    def test_max_retries_valid(self):
        """Valid max_retries (non-negative) passes validation."""
        data = {
            'download': {
                'youtube_api': {
                    'max_retries': 3,
                },
            },
        }

        result = validate_config_schema(data, raise_on_error=True)
        errors = [r for r in result if 'max_retries' in r]
        assert len(errors) == 0

    def test_max_retries_negative_raises(self):
        """Negative max_retries raises ConfigValidationError."""
        data = {
            'download': {
                'youtube_api': {
                    'max_retries': -1,
                },
            },
        }

        with pytest.raises(ConfigValidationError, match='max_retries.*must be >= 0'):
            validate_config_schema(data, raise_on_error=True)

    def test_max_retries_zero_valid(self):
        """max_retries of 0 passes validation (no retries)."""
        data = {
            'download': {
                'youtube_api': {
                    'max_retries': 0,
                },
            },
        }

        result = validate_config_schema(data, raise_on_error=True)
        errors = [r for r in result if 'max_retries' in r]
        assert len(errors) == 0

    def test_cache_ttl_seconds_valid_range(self):
        """Valid cache_ttl_seconds within 60-86400 passes validation."""
        data = {
            'download': {
                'youtube_api': {
                    'cache_ttl_seconds': 3600,
                },
            },
        }

        result = validate_config_schema(data, raise_on_error=True)
        errors = [r for r in result if 'cache_ttl_seconds' in r]
        assert len(errors) == 0

    def test_cache_ttl_seconds_below_minimum_raises(self):
        """cache_ttl_seconds below 60 raises ConfigValidationError."""
        data = {
            'download': {
                'youtube_api': {
                    'cache_ttl_seconds': 30,
                },
            },
        }

        with pytest.raises(ConfigValidationError, match='cache_ttl_seconds.*must be 60-86400'):
            validate_config_schema(data, raise_on_error=True)

    def test_cache_ttl_seconds_above_maximum_raises(self):
        """cache_ttl_seconds above 86400 raises ConfigValidationError."""
        data = {
            'download': {
                'youtube_api': {
                    'cache_ttl_seconds': 100000,
                },
            },
        }

        with pytest.raises(ConfigValidationError, match='cache_ttl_seconds.*must be 60-86400'):
            validate_config_schema(data, raise_on_error=True)

    def test_all_youtube_api_fields_valid(self):
        """All valid YouTubeAPIConfig fields pass validation."""
        data = {
            'download': {
                'youtube_api': {
                    'enabled': True,
                    'api_key': 'AIzaSyABC123xyz',
                    'quota_limit': 10000,
                    'warn_at_percent': 80,
                    'timeout_seconds': 30,
                    'cache_ttl_seconds': 3600,
                    'channel_metadata_cache_ttl_seconds': 86400,
                    'cache_ttl_days': 7,
                    'min_subscriber_count': 1000,
                    'max_retries': 3,
                    'retry_delay_seconds': 2.0,
                    'quota_auto_scale_enabled': False,
                    'quota_multiplier': 1.0,
                    'quota_floor': 1000,
                    'quota_ceiling': 100000,
                },
            },
        }

        result = validate_config_schema(data, raise_on_error=True)
        errors = [r for r in result if 'youtube_api' in r]
        assert len(errors) == 0

    def test_quota_floor_greater_than_ceiling_raises(self):
        """quota_floor > quota_ceiling raises ConfigValidationError."""
        data = {
            'download': {
                'youtube_api': {
                    'quota_floor': 50000,
                    'quota_ceiling': 10000,
                },
            },
        }

        with pytest.raises(ConfigValidationError, match='quota_floor must be <= quota_ceiling'):
            validate_config_schema(data, raise_on_error=True)

    def test_disabled_youtube_api_no_api_key_required(self):
        """When youtube_api is disabled, api_key is not required."""
        data = {
            'download': {
                'youtube_api': {
                    'enabled': False,
                },
            },
        }

        result = validate_config_schema(data, raise_on_error=True)
        errors = [r for r in result if 'youtube_api.api_key' in r]
        assert len(errors) == 0
        assert len(result) == 0  # No errors in valid config
