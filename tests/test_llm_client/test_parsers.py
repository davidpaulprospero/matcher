"""
Unit tests for JSON parsing utilities.
"""

import pytest
from src.llm_client.parsers import (
    parse_json,
    parse_json_array,
    repair_json,
    extract_json_by_keys
)


class TestParseJson:
    """Test parse_json function with various inputs."""

    def test_valid_json(self):
        """Test parsing valid JSON."""
        text = '{"key": "value", "number": 42}'
        result = parse_json(text)

        assert result == {"key": "value", "number": 42}

    def test_json_with_markdown(self):
        """Test parsing JSON wrapped in markdown code blocks."""
        text = '```json\n{"key": "value"}\n```'
        result = parse_json(text)

        assert result == {"key": "value"}

    def test_json_with_text_before_and_after(self):
        """Test extracting JSON from text with surrounding content."""
        text = 'Here is the result:\n{"key": "value"}\nThat was it.'
        result = parse_json(text)

        assert result == {"key": "value"}

    def test_json_with_trailing_comma(self):
        """Test repairing JSON with trailing comma."""
        text = '{"key": "value",}'
        result = parse_json(text)

        assert result == {"key": "value"}

    def test_nested_json(self):
        """Test parsing nested JSON objects."""
        text = '{"outer": {"inner": "value"}}'
        result = parse_json(text)

        assert result == {"outer": {"inner": "value"}}

    def test_empty_object(self):
        """Test parsing empty JSON object."""
        text = '{}'
        result = parse_json(text)

        assert result == {}

    def test_invalid_json_returns_none(self):
        """Test that completely invalid JSON returns None."""
        text = 'this is not json at all'
        result = parse_json(text)

        assert result is None

    def test_empty_string_returns_none(self):
        """Test that empty string returns None."""
        result = parse_json("")
        assert result is None

        result = parse_json("   ")
        assert result is None

    def test_json_with_special_characters(self):
        """Test parsing JSON with special characters."""
        text = '{"text": "Hello\\"world\\nNew line"}'
        result = parse_json(text)

        assert result is not None
        assert "text" in result

    def test_multiple_json_objects_returns_first(self):
        """Test that multiple JSON objects returns the first valid one."""
        text = '{"first": 1} {"second": 2}'
        result = parse_json(text)

        # Should get the first complete object
        assert result is not None
        assert "first" in result or result == {"first": 1}


class TestParseJsonArray:
    """Test parse_json_array function."""

    def test_valid_array(self):
        """Test parsing valid JSON array."""
        text = '[{"id": 1}, {"id": 2}, {"id": 3}]'
        result = parse_json_array(text)

        assert result == [{"id": 1}, {"id": 2}, {"id": 3}]

    def test_array_with_markdown(self):
        """Test parsing array wrapped in markdown."""
        text = '```json\n[{"id": 1}]\n```'
        result = parse_json_array(text)

        assert result == [{"id": 1}]

    def test_array_with_text_around_it(self):
        """Test extracting array from text with surrounding content."""
        text = 'Results:\n[{"id": 1}, {"id": 2}]\nDone.'
        result = parse_json_array(text)

        assert result == [{"id": 1}, {"id": 2}]

    def test_empty_array(self):
        """Test parsing empty array."""
        text = '[]'
        result = parse_json_array(text)

        assert result == []

    def test_array_with_trailing_comma(self):
        """Test repairing array with trailing comma."""
        text = '[{"id": 1}, {"id": 2},]'
        result = parse_json_array(text)

        assert len(result) == 2

    def test_expected_count_validation(self):
        """Test expected_count parameter."""
        text = '[{"id": 1}, {"id": 2}]'

        # Should pass with correct count
        result = parse_json_array(text, expected_count=2)
        assert result == [{"id": 1}, {"id": 2}]

        # Should still return with wrong count (no strict validation)
        result = parse_json_array(text, expected_count=3)
        assert result is not None  # Parser is lenient

    def test_extract_individual_objects(self):
        """Test extracting individual objects when array parsing fails."""
        # Malformed array but valid objects
        text = '{"id": 1} {"id": 2} {"id": 3}'
        result = parse_json_array(text)

        assert result is not None
        assert len(result) >= 2  # Should extract at least some objects

    def test_invalid_array_returns_none(self):
        """Test that invalid array returns None."""
        text = 'not an array at all'
        result = parse_json_array(text)

        assert result is None

    def test_array_of_primitives(self):
        """Test parsing array of primitive values."""
        text = '["apple", "banana", "cherry"]'
        result = parse_json_array(text)

        # Parser is designed for arrays of objects, but should handle primitives
        assert result is not None


class TestRepairJson:
    """Test repair_json function."""

    def test_remove_markdown(self):
        """Test removing markdown code blocks."""
        text = '```json\n{"key": "value"}\n```'
        repaired = repair_json(text)

        assert '```' not in repaired
        assert '{"key": "value"}' in repaired

    def test_remove_trailing_commas(self):
        """Test removing trailing commas."""
        text = '{"key": "value",}'
        repaired = repair_json(text)

        assert repaired == '{"key": "value"}'

    def test_fix_missing_comma_between_objects(self):
        """Test adding missing comma between objects."""
        text = '{"a": 1}{"b": 2}'
        repaired = repair_json(text)

        assert '},{'  in repaired

    def test_fix_missing_comma_between_arrays(self):
        """Test adding missing comma between arrays."""
        text = '[1, 2][3, 4]'
        repaired = repair_json(text)

        assert '],[' in repaired

    def test_extract_json_boundaries(self):
        """Test extracting JSON from text with extra content."""
        text = 'Here is some text before {"key": "value"} and after'
        repaired = repair_json(text)

        # Should start with { and end with }
        assert repaired.startswith('{')
        assert repaired.endswith('}')

    def test_handles_nested_braces(self):
        """Test handling nested braces correctly."""
        text = '{"outer": {"inner": "value"}}'
        repaired = repair_json(text)

        # Should preserve nested structure
        assert repaired == text

    def test_no_changes_for_valid_json(self):
        """Test that valid JSON is not changed."""
        text = '{"key": "value", "number": 42}'
        repaired = repair_json(text)

        assert repaired == text


class TestExtractJsonByKeys:
    """Test extract_json_by_keys function."""

    def test_extract_simple_values(self):
        """Test extracting values by keys."""
        text = '"name": "John", "age": 30'
        result = extract_json_by_keys(text, ["name", "age"])

        assert result == {"name": "John", "age": 30}

    def test_extract_with_quotes(self):
        """Test extracting string values with quotes."""
        text = '"title": "Hello World"'
        result = extract_json_by_keys(text, ["title"])

        assert result == {"title": "Hello World"}

    def test_extract_numbers(self):
        """Test extracting numeric values."""
        text = '"count": 42, "price": 19.99'
        result = extract_json_by_keys(text, ["count", "price"])

        assert result == {"count": 42, "price": 19.99}

    def test_extract_booleans(self):
        """Test extracting boolean values."""
        text = '"active": true, "deleted": false'
        result = extract_json_by_keys(text, ["active", "deleted"])

        assert result == {"active": True, "deleted": False}

    def test_missing_keys_return_none(self):
        """Test that missing keys result in None."""
        text = '"name": "John"'
        result = extract_json_by_keys(text, ["name", "missing"])

        # Should return dict with found keys
        assert result is not None
        assert "name" in result

    def test_no_keys_found_returns_none(self):
        """Test that no matching keys returns None."""
        text = '"name": "John"'
        result = extract_json_by_keys(text, ["age", "email"])

        assert result is None or result == {}

    def test_complex_text(self):
        """Test extracting from text with lots of noise."""
        text = 'The person is "name": "Alice" and they are "age": 25 years old'
        result = extract_json_by_keys(text, ["name", "age"])

        assert result is not None
        assert "name" in result
        assert "age" in result


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_parse_json_with_none(self):
        """Test parsing None input."""
        result = parse_json(None)
        assert result is None

    def test_parse_json_array_with_none(self):
        """Test parsing None input for array."""
        result = parse_json_array(None)
        assert result is None

    def test_very_large_json(self):
        """Test parsing very large JSON object."""
        # Create large JSON
        large_obj = {f"key_{i}": f"value_{i}" for i in range(1000)}
        import json
        text = json.dumps(large_obj)

        result = parse_json(text)

        assert result is not None
        assert len(result) == 1000

    def test_deeply_nested_json(self):
        """Test parsing deeply nested JSON."""
        text = '{"a": {"b": {"c": {"d": {"e": "value"}}}}}'
        result = parse_json(text)

        assert result is not None
        assert result["a"]["b"]["c"]["d"]["e"] == "value"

    def test_unicode_in_json(self):
        """Test parsing JSON with Unicode characters."""
        text = '{"message": "Hello 世界 🌍"}'
        result = parse_json(text)

        assert result is not None
        assert "世界" in result["message"]
        assert "🌍" in result["message"]

    def test_json_with_newlines_and_tabs(self):
        """Test parsing JSON with whitespace."""
        text = '''
        {
            "key": "value",
            "number": 42
        }
        '''
        result = parse_json(text)

        assert result == {"key": "value", "number": 42}
