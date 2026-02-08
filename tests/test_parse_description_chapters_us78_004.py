"""Tests for parse_description_chapters in src/caption/models.py (US-78-004).

Covers all 4 timestamp formats plus edge cases:
- '0:00 Title' (M:SS)
- '00:00 Title' (MM:SS)
- '0:00:00 Title' (H:MM:SS)
- '[0:00] Title' (bracketed)
"""

import pytest

from src.caption.models import parse_description_chapters


class TestParseDescriptionChaptersFormats:
    """Tests for all 4 timestamp formats."""

    def test_m_ss_format(self):
        """Parses M:SS format like '0:00 Title'."""
        description = "0:00 Introduction\n2:30 Middle\n5:00 End"
        chapters = parse_description_chapters(description)

        assert len(chapters) == 3
        assert chapters[0] == {'title': 'Introduction', 'start_time': 0.0, 'end_time': 150.0}
        assert chapters[1] == {'title': 'Middle', 'start_time': 150.0, 'end_time': 300.0}
        assert chapters[2] == {'title': 'End', 'start_time': 300.0, 'end_time': None}

    def test_mm_ss_format(self):
        """Parses MM:SS format like '00:00 Title'."""
        description = "00:00 Intro\n01:30 Topic\n10:00 Outro"
        chapters = parse_description_chapters(description)

        assert len(chapters) == 3
        assert chapters[0]['title'] == 'Intro'
        assert chapters[0]['start_time'] == 0.0
        assert chapters[1]['title'] == 'Topic'
        assert chapters[1]['start_time'] == 90.0
        assert chapters[2]['title'] == 'Outro'
        assert chapters[2]['start_time'] == 600.0

    def test_h_mm_ss_format(self):
        """Parses H:MM:SS format like '0:00:00 Title'."""
        description = "0:00:00 Start\n1:02:30 Long Section\n2:15:00 End"
        chapters = parse_description_chapters(description)

        assert len(chapters) == 3
        assert chapters[0] == {'title': 'Start', 'start_time': 0.0, 'end_time': 3750.0}
        assert chapters[1] == {'title': 'Long Section', 'start_time': 3750.0, 'end_time': 8100.0}
        assert chapters[2] == {'title': 'End', 'start_time': 8100.0, 'end_time': None}

    def test_bracketed_format(self):
        """Parses [0:00] bracketed format."""
        description = "[0:00] Introduction\n[3:45] Main Content\n[8:20] Summary"
        chapters = parse_description_chapters(description)

        assert len(chapters) == 3
        assert chapters[0] == {'title': 'Introduction', 'start_time': 0.0, 'end_time': 225.0}
        assert chapters[1] == {'title': 'Main Content', 'start_time': 225.0, 'end_time': 500.0}
        assert chapters[2] == {'title': 'Summary', 'start_time': 500.0, 'end_time': None}

    def test_bracketed_hh_mm_ss_format(self):
        """Parses [H:MM:SS] bracketed format."""
        description = "[0:00:00] Opening\n[0:05:30] Discussion\n[1:00:00] Closing"
        chapters = parse_description_chapters(description)

        assert len(chapters) == 3
        assert chapters[0]['title'] == 'Opening'
        assert chapters[0]['start_time'] == 0.0
        assert chapters[1]['title'] == 'Discussion'
        assert chapters[1]['start_time'] == 330.0
        assert chapters[2]['title'] == 'Closing'
        assert chapters[2]['start_time'] == 3600.0


class TestParseDescriptionChaptersEdgeCases:
    """Edge case tests."""

    def test_empty_string_returns_empty(self):
        """Returns empty list for empty description."""
        assert parse_description_chapters("") == []

    def test_none_returns_empty(self):
        """Returns empty list for None."""
        assert parse_description_chapters(None) == []

    def test_no_timestamps_returns_empty(self):
        """Returns empty list when no timestamps found."""
        description = "This is a regular description with no timestamps at all."
        assert parse_description_chapters(description) == []

    def test_single_chapter_returns_list(self):
        """Single chapter is still returned (not filtered like old implementation)."""
        description = "0:00 Only Chapter"
        chapters = parse_description_chapters(description)
        # Single chapter is valid per acceptance criteria
        assert len(chapters) == 1
        assert chapters[0] == {'title': 'Only Chapter', 'start_time': 0.0, 'end_time': None}

    def test_malformed_input_mixed_with_valid(self):
        """Handles malformed lines mixed with valid timestamps."""
        description = (
            "Check out my channel!\n"
            "0:00 Introduction\n"
            "Some random text\n"
            "3:45 Good Stuff\n"
            "Thanks for watching!"
        )
        chapters = parse_description_chapters(description)

        assert len(chapters) == 2
        assert chapters[0]['title'] == 'Introduction'
        assert chapters[1]['title'] == 'Good Stuff'

    def test_end_time_inferred_from_next_start(self):
        """End times are inferred from next chapter's start_time."""
        description = "0:00 A\n1:00 B\n2:00 C\n3:00 D"
        chapters = parse_description_chapters(description)

        assert chapters[0]['end_time'] == chapters[1]['start_time']  # 60.0
        assert chapters[1]['end_time'] == chapters[2]['start_time']  # 120.0
        assert chapters[2]['end_time'] == chapters[3]['start_time']  # 180.0

    def test_last_chapter_end_time_is_none(self):
        """Last chapter end_time defaults to None."""
        description = "0:00 First\n5:00 Last"
        chapters = parse_description_chapters(description)

        assert chapters[-1]['end_time'] is None

    def test_leading_whitespace(self):
        """Handles timestamps with leading spaces."""
        description = "  0:00 Introduction\n  2:30 Topic\n  5:00 End"
        chapters = parse_description_chapters(description)

        assert len(chapters) == 3
        assert chapters[0]['title'] == 'Introduction'

    def test_mixed_formats_in_same_description(self):
        """Parses mix of MM:SS and HH:MM:SS formats."""
        description = (
            "0:00 Introduction\n"
            "5:30 First Part\n"
            "1:00:00 Second Part\n"
            "1:30:15 Wrap Up"
        )
        chapters = parse_description_chapters(description)

        assert len(chapters) == 4
        assert chapters[0]['start_time'] == 0.0
        assert chapters[1]['start_time'] == 330.0
        assert chapters[2]['start_time'] == 3600.0
        assert chapters[3]['start_time'] == 5415.0
        assert chapters[3]['end_time'] is None


class TestParseDescriptionChaptersConfig:
    """Tests for config gating (matching.context_enrichment.parse_description_chapters)."""

    def test_config_flag_exists(self):
        """Config flag parse_description_chapters exists in matching config."""
        from src.config.sections.matching import ContextEnrichmentConfig
        config = ContextEnrichmentConfig()
        assert hasattr(config, 'parse_description_chapters')
        assert config.parse_description_chapters is True  # Default true
