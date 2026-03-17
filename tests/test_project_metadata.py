"""Tests for src.utils.project_metadata — sanitize_filename and get_card_title."""

import json
import pytest
from pathlib import Path

from src.utils.project_metadata import sanitize_filename, get_card_title


class TestSanitizeFilename:
    def test_basic_spaces_to_underscores(self):
        assert sanitize_filename("UKRAINE Russia Drone Strikes") == "UKRAINE_Russia_Drone_Strikes"

    def test_strips_unsafe_characters(self):
        assert sanitize_filename('Breaking: US/Russia "Peace"') == "Breaking_USRussia_Peace"

    def test_strips_apostrophes_and_commas(self):
        assert sanitize_filename("Iran's 40,000 Troops") == "Irans_40000_Troops"

    def test_replaces_em_dash(self):
        assert sanitize_filename("Big News \u2014 Breaking") == "Big_News_Breaking"

    def test_replaces_en_dash(self):
        assert sanitize_filename("2020\u20132025") == "2020_2025"

    def test_dashes_to_underscores(self):
        assert sanitize_filename("some-dashed-name") == "some_dashed_name"

    def test_collapses_repeated_underscores(self):
        assert sanitize_filename("foo   bar") == "foo_bar"

    def test_empty_string(self):
        assert sanitize_filename("") == ""

    def test_none_input(self):
        assert sanitize_filename(None) == ""

    def test_truncates_long_string(self):
        long_name = "A" * 200
        result = sanitize_filename(long_name)
        assert len(result) <= 80

    def test_truncates_at_word_boundary(self):
        name = "Word " * 20  # 100 chars
        result = sanitize_filename(name, max_length=80)
        assert len(result) <= 80
        assert not result.endswith("_")

    def test_strips_leading_trailing_underscores(self):
        assert sanitize_filename("  hello  ") == "hello"

    def test_mixed_special_characters(self):
        result = sanitize_filename("Test <video> | file?name*.mp4")
        assert "<" not in result
        assert ">" not in result
        assert "|" not in result
        assert "?" not in result
        assert "*" not in result

    def test_all_unsafe_chars_removed(self):
        result = sanitize_filename('<>:"/\\|?*')
        assert result == ""


class TestGetCardTitle:
    def test_reads_card_title(self, tmp_path):
        card_data = {"card_id": "abc123", "name": "Ukraine Drone Strikes"}
        (tmp_path / "trello_card.json").write_text(json.dumps(card_data), encoding="utf-8")

        result = get_card_title(tmp_path)
        assert result == "Ukraine_Drone_Strikes"

    def test_no_file_returns_none(self, tmp_path):
        assert get_card_title(tmp_path) is None

    def test_missing_name_field(self, tmp_path):
        card_data = {"card_id": "abc123"}
        (tmp_path / "trello_card.json").write_text(json.dumps(card_data), encoding="utf-8")

        assert get_card_title(tmp_path) is None

    def test_empty_name_field(self, tmp_path):
        card_data = {"card_id": "abc123", "name": ""}
        (tmp_path / "trello_card.json").write_text(json.dumps(card_data), encoding="utf-8")

        assert get_card_title(tmp_path) is None

    def test_invalid_json(self, tmp_path):
        (tmp_path / "trello_card.json").write_text("not valid json", encoding="utf-8")

        assert get_card_title(tmp_path) is None

    def test_sanitizes_name(self, tmp_path):
        card_data = {"name": 'Breaking: US/Russia "Peace" Deal'}
        (tmp_path / "trello_card.json").write_text(json.dumps(card_data), encoding="utf-8")

        result = get_card_title(tmp_path)
        assert result == "Breaking_USRussia_Peace_Deal"
        assert ":" not in result
        assert "/" not in result
        assert '"' not in result
