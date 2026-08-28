"""Unit tests for src/chapter_detection/listicle_seed.py.

Covers:
- format_listicle_seed_block: empty, single, multiple groups, expected_count
- select_seed_groups: confidence threshold, require_count_match toggle
"""

from src.chapter_detection.listicle_seed import (
    format_listicle_seed_block,
    select_seed_groups,
)
from src.chapter_detection.models import ListicleGroup


def _mk(group_id: int, label: str, start: int, end: int, *,
        conf: float = 0.8, expected: int = None,
        topics=("alpha", "beta"), marker: str = "ordinal") -> ListicleGroup:
    return ListicleGroup(
        group_id=group_id,
        item_label=label,
        marker_type=marker,
        start_segment_idx=start,
        end_segment_idx=end,
        topic_keywords=list(topics),
        expected_count=expected,
        confidence=conf,
    )


class TestFormatListicleSeedBlock:
    def test_empty_input_returns_empty_string(self):
        assert format_listicle_seed_block([]) == ""
        assert format_listicle_seed_block(iter([])) == ""

    def test_single_group_renders_one_line(self):
        g = _mk(0, "first", 0, 3, expected=5)
        out = format_listicle_seed_block([g])
        assert "PRE-DETECTED LIST STRUCTURE" in out
        assert "Item 1 [first] segs 0-3" in out
        assert "Expected total items per header: 5" in out
        # Rules block always present
        assert "Treat each item's end_segment_idx" in out
        assert "Do NOT introduce a boundary in the MIDDLE" in out

    def test_multiple_groups_render_in_order(self):
        groups = [_mk(i, f"item-{i}", i * 4, i * 4 + 3, expected=3) for i in range(3)]
        out = format_listicle_seed_block(groups)
        # Items appear in declaration order
        pos1 = out.find("Item 1 [item-0]")
        pos2 = out.find("Item 2 [item-1]")
        pos3 = out.find("Item 3 [item-2]")
        assert 0 <= pos1 < pos2 < pos3

    def test_topic_keywords_truncated_to_three(self):
        g = _mk(0, "first", 0, 2, topics=("a", "b", "c", "d", "e"))
        out = format_listicle_seed_block([g])
        # Only first 3 appear, comma-joined
        assert "a, b, c" in out
        assert " d," not in out
        assert " e," not in out

    def test_topic_keywords_fallback_when_empty(self):
        g = _mk(0, "first", 0, 2, topics=())
        out = format_listicle_seed_block([g])
        assert "(no keywords)" in out

    def test_expected_count_absent_when_none(self):
        g = _mk(0, "first", 0, 3, expected=None)
        out = format_listicle_seed_block([g])
        assert "Expected total items" not in out

    def test_uses_first_group_expected_count(self):
        # When groups[0] has no expected_count, none is reported
        groups = [_mk(0, "a", 0, 1, expected=None),
                  _mk(1, "b", 2, 3, expected=5)]
        out = format_listicle_seed_block(groups)
        assert "Expected total items" not in out

    def test_label_fallback_when_empty(self):
        g = _mk(0, "", 0, 2, expected=1)
        out = format_listicle_seed_block([g])
        assert "Item 1 [item-1]" in out  # group_id 0 → fallback "item-1"


class TestSelectSeedGroups:
    def test_returns_empty_when_input_empty(self):
        assert select_seed_groups([]) == []
        assert select_seed_groups(iter([])) == []

    def test_filters_below_min_confidence(self):
        groups = [_mk(0, "a", 0, 1, conf=0.9),
                  _mk(1, "b", 2, 3, conf=0.4),
                  _mk(2, "c", 4, 5, conf=0.71)]
        out = select_seed_groups(groups, min_confidence=0.7)
        assert [g.group_id for g in out] == [0, 2]

    def test_default_threshold_is_0_7(self):
        groups = [_mk(0, "a", 0, 1, conf=0.69),
                  _mk(1, "b", 2, 3, conf=0.71)]
        assert select_seed_groups(groups) == [groups[1]]

    def test_require_count_match_passes_when_match(self):
        groups = [_mk(0, "a", 0, 1, conf=0.9, expected=2),
                  _mk(1, "b", 2, 3, conf=0.9, expected=2)]
        out = select_seed_groups(groups, require_count_match=True)
        assert len(out) == 2

    def test_require_count_match_drops_mismatch(self):
        groups = [_mk(0, "a", 0, 1, conf=0.9, expected=5),
                  _mk(1, "b", 2, 3, conf=0.9, expected=5)]
        out = select_seed_groups(groups, require_count_match=True)
        assert out == []

    def test_require_count_match_ignored_when_expected_none(self):
        groups = [_mk(0, "a", 0, 1, conf=0.9, expected=None),
                  _mk(1, "b", 2, 3, conf=0.9, expected=None)]
        out = select_seed_groups(groups, require_count_match=True)
        assert len(out) == 2
