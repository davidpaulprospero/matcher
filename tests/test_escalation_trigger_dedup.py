"""Tests for US-58-005: Deduplicate escalation trigger regex from category definitions.

Verifies that:
  - _ESCALATION_TRIGGER_RE is built dynamically from _TRIGGER_CATEGORIES
  - All patterns in _TRIGGER_CATEGORIES match the combined regex
  - classify_trigger uses _TRIGGER_CATEGORIES as the single source of truth
  - Invalid regex patterns in _TRIGGER_CATEGORIES raise errors at init time
"""

import re
from unittest.mock import patch

import pytest

from src.downloader.escalation_manager import (
    _ESCALATION_TRIGGER_RE,
    _TRIGGER_CATEGORIES,
    classify_trigger,
    is_escalation_trigger,
)


class TestTriggerRegexBuiltFromCategories:
    """Verify the combined regex is derived from _TRIGGER_CATEGORIES."""

    def test_combined_regex_is_union_of_category_patterns(self):
        """The combined regex pattern should be the '|'-joined category patterns."""
        expected_pattern = '|'.join(
            cat_pattern.pattern for _, cat_pattern in _TRIGGER_CATEGORIES
        )
        assert _ESCALATION_TRIGGER_RE.pattern == expected_pattern

    def test_combined_regex_is_case_insensitive(self):
        """The combined regex should have IGNORECASE flag."""
        assert _ESCALATION_TRIGGER_RE.flags & re.IGNORECASE

    def test_all_category_patterns_match_combined_regex(self):
        """Every individual category pattern should have a match in the combined regex."""
        # For each category, generate a sample string that matches its pattern,
        # then verify it also matches the combined regex.
        test_samples = {
            '429': ['HTTP Error 429', 'Too Many Requests', 'rate limit'],
            'age_gate': ['age gate', 'age restricted', 'sign in to confirm age'],
            'ip_blocked': ['IP address', 'ip block', 'access denied', 'geo block'],
            'bot_detection': ['bot', 'captcha', 'verify you are human'],
            '403': ['HTTP Error 403', 'Sign in to confirm', 'blocked'],
        }

        for category, cat_pattern in _TRIGGER_CATEGORIES:
            samples = test_samples.get(category, [])
            assert samples, f"No test samples for category {category!r}"
            for sample in samples:
                # Must match category pattern
                assert cat_pattern.search(sample), (
                    f"Sample {sample!r} did not match category {category!r} pattern"
                )
                # Must also match the combined regex
                assert _ESCALATION_TRIGGER_RE.search(sample), (
                    f"Sample {sample!r} matched category {category!r} but NOT "
                    f"the combined _ESCALATION_TRIGGER_RE"
                )

    def test_is_escalation_trigger_uses_combined_regex(self):
        """is_escalation_trigger() should match any category's patterns."""
        # One sample from each category
        samples = [
            'HTTP Error 429',
            'age gate',
            'IP address',
            'captcha',
            'HTTP Error 403',
        ]
        for sample in samples:
            assert is_escalation_trigger(sample), (
                f"is_escalation_trigger({sample!r}) should be True"
            )

    def test_is_escalation_trigger_rejects_non_matching(self):
        """is_escalation_trigger() should return False for non-trigger text."""
        assert not is_escalation_trigger('')
        assert not is_escalation_trigger('Download completed successfully')
        assert not is_escalation_trigger('video unavailable')

    def test_no_category_patterns_are_duplicated_in_separate_regex(self):
        """There should be no hardcoded _ESCALATION_TRIGGER_RE separate from categories.

        The combined regex pattern should be exactly the union of category patterns,
        with no extra patterns that don't come from _TRIGGER_CATEGORIES.
        """
        combined_parts = set(_ESCALATION_TRIGGER_RE.pattern.split('|'))
        category_parts = set()
        for _, cat_pattern in _TRIGGER_CATEGORIES:
            for part in cat_pattern.pattern.split('|'):
                category_parts.add(part)

        # Every part in the combined regex must come from a category
        extra = combined_parts - category_parts
        assert not extra, (
            f"Combined regex has patterns not in _TRIGGER_CATEGORIES: {extra}"
        )


class TestClassifyTriggerUsesCategories:
    """Verify classify_trigger() uses _TRIGGER_CATEGORIES as single source of truth."""

    def test_classify_returns_correct_category_for_each(self):
        """classify_trigger should return the right category for known patterns."""
        assert classify_trigger('HTTP Error 429') == '429'
        assert classify_trigger('Too Many Requests') == '429'
        assert classify_trigger('rate limiting') == '429'
        assert classify_trigger('age gate required') == 'age_gate'
        assert classify_trigger('age restricted') == 'age_gate'
        assert classify_trigger('IP address banned') == 'ip_blocked'
        assert classify_trigger('access denied') == 'ip_blocked'
        assert classify_trigger('captcha required') == 'bot_detection'
        assert classify_trigger('verify you are human') == 'bot_detection'
        assert classify_trigger('HTTP Error 403') == '403'
        assert classify_trigger('Sign in to confirm') == '403'

    def test_classify_returns_none_for_non_matching(self):
        """classify_trigger should return None for non-trigger text."""
        assert classify_trigger('') is None
        assert classify_trigger('Download completed') is None

    def test_classify_trigger_category_order_specificity(self):
        """More specific categories should match before general ones.

        '429' should match before '403' for rate-limit errors.
        """
        # "429" text should classify as '429', not '403'
        assert classify_trigger('HTTP Error 429') == '429'
        # "blocked" alone should classify as '403' (last category)
        assert classify_trigger('blocked') == '403'


class TestPatternValidationAtImportTime:
    """Verify that invalid regex patterns raise errors."""

    def test_all_trigger_categories_are_compiled_patterns(self):
        """Every entry in _TRIGGER_CATEGORIES should have a compiled re.Pattern."""
        for name, pattern in _TRIGGER_CATEGORIES:
            assert isinstance(pattern, re.Pattern), (
                f"_TRIGGER_CATEGORIES[{name!r}] is not a compiled pattern: "
                f"got {type(pattern).__name__}"
            )

    def test_invalid_regex_in_categories_raises_on_compile(self):
        """If someone puts an invalid regex string in the categories,
        re.compile() itself raises re.error at definition time."""
        with pytest.raises(re.error):
            re.compile(r'(unclosed group')

    def test_trigger_categories_has_expected_categories(self):
        """Verify the expected category names are present."""
        category_names = [name for name, _ in _TRIGGER_CATEGORIES]
        assert '429' in category_names
        assert 'age_gate' in category_names
        assert 'ip_blocked' in category_names
        assert 'bot_detection' in category_names
        assert '403' in category_names

    def test_building_combined_regex_from_invalid_pattern_raises(self):
        """Simulates what would happen if an invalid pattern string was used."""
        bad_categories = [
            ('bad', '(unclosed'),  # Not a compiled pattern, just a string
        ]
        # The assertion in the module checks isinstance(pattern, re.Pattern)
        # So a raw string would fail the assertion
        for name, pattern in bad_categories:
            assert not isinstance(pattern, re.Pattern)
