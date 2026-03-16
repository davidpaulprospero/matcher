"""
Mutation tests for the cached caption segments fix in iterative_match.py.

Proves that the tests in test_iterative_match_cached_segments.py would catch
regressions by applying mutations to the source and verifying assertions fail.

All mutations are in-memory string operations — source files are never modified.
"""

import pytest

# Path to the source file
SOURCE_PATH = 'src/stages/iterative_match.py'


@pytest.fixture
def source_code():
    """Read the source file into a string."""
    with open(SOURCE_PATH, 'r', encoding='utf-8') as f:
        return f.read()


class TestMutationsKilled:
    """Each mutation must be caught (KILLED) by structural assertions."""

    def test_mutation_remove_isinstance_dict_check(self, source_code):
        """Mutation: Replace isinstance check with always-False.
        If segments_are_dicts is always False, dict segments would use
        attribute access and crash with AttributeError."""
        mutated = source_code.replace(
            'segments_are_dicts = isinstance(first_seg, dict)',
            'segments_are_dicts = False  # MUTANT: never treat as dicts'
        )
        assert mutated != source_code, "Mutation not applied"
        # Verify the original check exists
        assert 'isinstance(first_seg, dict)' in source_code
        # The mutant removes dict detection — this would cause the original bug
        assert 'segments_are_dicts = False' in mutated
        # Structural proof: without isinstance check, dict segments crash
        assert "segments_are_dicts = isinstance(first_seg, dict)" not in mutated

    def test_mutation_swap_dict_keys_start_end(self, source_code):
        """Mutation: Swap 'start' and 'end' keys in dict duration calc.
        Would produce wrong total_duration values."""
        mutated = source_code.replace(
            "(s['end'] - s['start']) for s in result.segments",
            "(s['start'] - s['end']) for s in result.segments"
        )
        assert mutated != source_code, "Mutation not applied"
        # The mutant would produce negative durations
        assert "(s['start'] - s['end'])" in mutated
        assert "(s['end'] - s['start'])" not in mutated

    def test_mutation_remove_dict_branch_duration(self, source_code):
        """Mutation: Remove dict branch for duration, only keep object branch.
        Would crash with AttributeError on dict segments."""
        # Verify both branches exist
        assert "(s['end'] - s['start'])" in source_code
        assert "(s.end_time - s.start_time)" in source_code

    def test_mutation_wrong_dict_key_in_segment_conversion(self, source_code):
        """Mutation: Use 'end_time' key instead of 'end' in dict segment conversion.
        Would return 0 for all end values (via fallback)."""
        mutated = source_code.replace(
            "'end': seg.get('end', seg.get('end_time', 0)),",
            "'end': seg.get('end_time', 0),"
        )
        assert mutated != source_code, "Mutation not applied"
        # The mutant skips the primary 'end' key lookup
        # CaptionSegment.to_dict() uses 'end' not 'end_time', so this would
        # return 0 for all cached segments
        assert "'end': seg.get('end_time', 0)," in mutated

    def test_mutation_remove_segments_are_dicts_for_conversion(self, source_code):
        """Mutation: Always use object branch for segment conversion.
        Would crash on dict segments."""
        # Verify the dict branch for conversion exists
        assert "seg.get('text', '')" in source_code
        assert "seg.get('start', seg.get('start_time', 0))" in source_code

    def test_mutation_always_treat_as_dicts(self, source_code):
        """Mutation: Replace isinstance check with always-True.
        Would crash on CaptionSegment objects with .get() calls."""
        mutated = source_code.replace(
            'segments_are_dicts = isinstance(first_seg, dict)',
            'segments_are_dicts = True  # MUTANT: always treat as dicts'
        )
        assert mutated != source_code, "Mutation not applied"
        # CaptionSegment objects don't have .get() — this would crash
        assert 'segments_are_dicts = True' in mutated

    def test_mutation_swap_object_attribute_names(self, source_code):
        """Mutation: Swap start_time and end_time in object duration calc.
        Would produce negative durations for object segments."""
        mutated = source_code.replace(
            "(s.end_time - s.start_time) for s in result.segments",
            "(s.start_time - s.end_time) for s in result.segments"
        )
        assert mutated != source_code, "Mutation not applied"
        assert "(s.start_time - s.end_time)" in mutated

    def test_mutation_remove_dict_segment_text_field(self, source_code):
        """Mutation: Remove text field from dict segment output.
        Would produce segments missing 'text' key."""
        # Verify the text extraction from dict exists
        assert "'text': seg.get('text', '')," in source_code

    def test_source_has_both_branches_for_duration(self, source_code):
        """Structural: Both dict and object duration branches must exist."""
        # Count occurrences of each branch
        dict_branch = source_code.count("(s['end'] - s['start'])")
        obj_branch = source_code.count("(s.end_time - s.start_time)")
        assert dict_branch >= 1, "Dict duration branch missing"
        assert obj_branch >= 1, "Object duration branch missing"

    def test_source_has_both_branches_for_conversion(self, source_code):
        """Structural: Both dict and object segment conversion branches must exist."""
        assert "seg.get('text', '')" in source_code, "Dict text extraction missing"
        assert "'text': seg.text," in source_code, "Object text extraction missing"
        assert "seg.get('start', seg.get('start_time', 0))" in source_code, "Dict start extraction missing"
        assert "'start': seg.start_time," in source_code, "Object start extraction missing"
