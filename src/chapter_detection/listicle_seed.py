"""Helpers to pre-seed LLM chapter detection with listicle structure.

Standalone utilities that translate `ListicleGroup` boundaries into a prompt
fragment the LLM topic prompt can accept as a strong prior. Designed for use
outside the in-pipeline detector (e.g., `scripts/chapters/detect_chapters.py`
and any future per-chapter pipeline).

This module intentionally does NOT import from `.detector` to keep it
portable across callers that may not have a full `EnhancedChapterDetector`
config available.
"""

from __future__ import annotations

from typing import Iterable, List, Optional

from .models import ListicleGroup


SEED_HEADER = "PRE-DETECTED LIST STRUCTURE (use as anchor):"
SEED_RULES = (
    "- Treat each item's end_segment_idx as a strong prior for a chapter boundary.",
    "- Do NOT introduce a boundary in the MIDDLE of an item.",
    "- Titles should align with the item topic when reasonable.",
    "- You MAY merge two adjacent items if both are very short and clearly the same topic.",
    "- You MAY split an item if the spoken text clearly indicates a sub-topic within it.",
)


def select_seed_groups(
    groups: Iterable[ListicleGroup],
    min_confidence: float = 0.7,
    require_count_match: bool = False,
) -> List[ListicleGroup]:
    """Return high-confidence groups eligible for LLM seeding.

    Filters by per-group `confidence >= min_confidence`. If
    `require_count_match` is True, all groups are dropped when the detected
    count disagrees with the list-header's `expected_count`.
    """
    eligible = [g for g in groups if g.confidence >= min_confidence]
    if require_count_match and eligible:
        ec = eligible[0].expected_count
        if ec is not None and len(eligible) != ec:
            return []
    return eligible


def format_listicle_seed_block(
    groups: Iterable[ListicleGroup],
    *,
    header: str = SEED_HEADER,
    rules: Iterable[str] = SEED_RULES,
) -> str:
    """Render the listicle seed block as a string for appending to an LLM prompt.

    Empty input returns an empty string. The block looks like::

        PRE-DETECTED LIST STRUCTURE (use as anchor):
          Item 1 [first] segs 0-3: topic, keywords
          Item 2 [second] segs 4-7: another topic
        Expected total items per header: 5

        - Treat each item's end_segment_idx as a strong prior...
        ...
    """
    items = list(groups)
    if not items:
        return ""

    lines: List[str] = [header]
    for g in items:
        topic = ", ".join((g.topic_keywords or [])[:3]) or "(no keywords)"
        lines.append(
            f"  Item {g.group_id + 1} [{g.item_label or f'item-{g.group_id + 1}'}] "
            f"segs {g.start_segment_idx}-{g.end_segment_idx}: {topic}"
        )

    expected_count: Optional[int] = items[0].expected_count
    if expected_count is not None:
        lines.append(f"Expected total items per header: {expected_count}")

    lines.append("")
    lines.extend(rules)

    return "\n".join(lines)
