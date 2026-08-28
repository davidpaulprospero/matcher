"""Unit tests for scripts/lower_thirds.py.

Covers:
- parse_ollama_entities edge cases (fences, prose, string indices, out-of-range)
- render_lower_third_png creates a valid RGBA file, truncates long names
- build_lower_thirds_otio track name, FreezeFrame, forward-slash target_url
- find_srt_path prefers trimmed variant

All tests are pure unit tests — no Ollama network, no FFmpeg, no real SRT.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from lower_thirds import (  # noqa: E402  (path injection above)
    ANIMATION_STYLES,
    Candidate,
    EntitySpan,
    EntityType,
    Layout,
    LowerThirdTemplate,
    TEMPLATE_REGISTRY,
    TRACK_NAME_BY_TYPE,
    TRACK_ORDER,
    _find_entity_segments,
    _normalize_text,
    _repair_flat_array_of_dicts,
    _repair_missing_open_brace,
    _word_onset,
    audit_first_minute,
    build_html_template,
    build_lower_thirds_otio,
    clamp_entity_durations,
    dedupe_entities,
    dedupe_substring_entities,
    extract_candidates,
    find_srt_path,
    load_chapter_entities,
    parse_candidate_response,
    parse_ollama_entities,
    parse_template_by_type,
    redistribute_stacked_entities,
    render_lower_third_png,
)
from src.utils import SRTSegment  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# parse_ollama_entities
#
# The LLM only classifies (name, role, entity_type). Timing is derived purely
# from the SRT by substring search — see _find_entity_segments.
# ─────────────────────────────────────────────────────────────────────────────

def _seg(index: int, text: str) -> SRTSegment:
    return SRTSegment(
        index=index,
        start_time=(index - 1) * 2.0,
        end_time=index * 2.0,
        text=text,
    )


def _segments(entries: list[str] | int | None = None, n: int = 5) -> list[SRTSegment]:
    """Build synthetic SRT segments. Each entry becomes a segment's text.

    Accepts:
    - a list of strings (preferred for tests that need specific text),
    - a bare int (default: n=5) to build generic "segment N text" entries,
    - None to default to n generic entries.
    """
    if isinstance(entries, int):
        n = entries
        entries = [f"segment {i} text" for i in range(1, n + 1)]
    elif entries is None:
        entries = [f"segment {i} text" for i in range(1, n + 1)]
    return [
        SRTSegment(
            index=i + 1,
            start_time=i * 2.0,
            end_time=(i + 1) * 2.0,
            text=text,
        )
        for i, text in enumerate(entries)
    ]


def test_parse_ollama_entities_strips_fences():
    raw = '```json\n[{"name": "Alice", "role": "Mayor", "entity_type": "person"}]\n```'
    result = parse_ollama_entities(raw, _segments(["Alice says hello"]))
    assert len(result) == 1
    assert result[0].name == "Alice"
    assert result[0].role == "Mayor"
    assert result[0].srt_indices == [1]
    assert result[0].start_time == 0.0
    assert result[0].end_time == 2.0


def test_parse_ollama_entities_prose_around_array():
    raw = 'Sure! Here is the array: [{"name": "Bob", "role": "", "entity_type": "person"}] hope that helps.'
    result = parse_ollama_entities(raw, _segments([
        "first segment", "Bob appears here", "Bob continues", "Bob signs off",
    ]))
    assert len(result) == 1
    assert result[0].name == "Bob"
    assert result[0].role == ""
    # First match wins, so the entity sits inside the first segment that mentions Bob.
    assert result[0].srt_indices == [2]
    assert result[0].start_time == 2.0
    assert result[0].end_time == 4.0


def test_parse_ollama_entities_top_level_object_with_array_values():
    """Mistral 7B sometimes returns a flat object — keys are names, values are lists."""
    raw = '{"Paris": ["France"], "Llanelli": ["Wales"]}'
    result = parse_ollama_entities(raw, _segments([
        "opening", "Welcome to Paris today", "later we visit Llanelli",
    ]))
    assert [e.name for e in result] == ["Paris", "Llanelli"]
    # Values are list-shaped, so the str() fallback kicks in and role ends up "['France']"
    # — this is just a smoke check that we didn't crash; the real-world case is plain strings.
    assert all(e.srt_indices for e in result)


def test_parse_ollama_entities_top_level_object_with_string_values():
    """The Mistral 7B failure mode: flat object with string values."""
    raw = (
        '{\n'
        '"event": "Reintroduction of a wetland habitat in Llanelli, Wales",\n'
        '"precedence": "Follows the work of Peter Scott who advocated for decades",\n'
        '"Llanelli": "Wales coastal town"\n'
        '}'
    )
    result = parse_ollama_entities(raw, _segments([
        "opening narration",
        "Llanelli is a Welsh coastal town where decades of conservation happened",
    ]))
    names = [e.name for e in result]
    # Only entries whose names appear in the SRT survive.
    assert "Llanelli" in names
    # "event" and "precedence" aren't proper nouns from the SRT, so they're dropped as hallucinations.
    assert "event" not in names
    assert "precedence" not in names
    llanelli = next(e for e in result if e.name == "Llanelli")
    assert llanelli.role == "Wales coastal town"
    assert llanelli.srt_indices == [2]


def test_parse_ollama_entities_top_level_object_strips_em_dash_prefix():
    """When the model embeds ' — context' in the value, the parser strips it for role."""
    raw = '{"1946": "Trust founded", "1969": " — Moon landing"}'
    result = parse_ollama_entities(raw, _segments([
        "in 1946 the Trust was founded", "by 1969 we had arrived",
    ]))
    by_name = {e.name: e for e in result}
    assert by_name["1946"].role == "Trust founded"
    assert by_name["1969"].role == "Moon landing"


def test_parse_ollama_entities_truncated_array_salvages_complete_objects():
    """When the model hits max_tokens mid-array, salvage the complete
    { ... } objects from before the truncation point."""
    raw = (
        '[\n'
        '  {"name": "Burian Lake", "entity_type": "place", "role": ""},\n'
        '  {"name": "Carmarthenshire", "entity_type": "place", "role": "Wales"},\n'
        '  {"name": "Scotland", "entity_type":"place", "role":""},\n'
        '  {"'  # truncated mid-name
    )
    result = parse_ollama_entities(raw, _segments([
        "opening",
        "Burian Lake is a Welsh site",
        "Scotland is mentioned here too",
        "Carmarthenshire is the region in Wales",
    ]))
    names = {e.name for e in result}
    assert "Burian Lake" in names
    assert "Carmarthenshire" in names
    assert "Scotland" in names
    # The truncated half-named entry must NOT have leaked in as a junk entity.
    assert all(len(n) > 1 for n in names)


def test_parse_ollama_entities_truncated_array_ignores_prose_braces():
    """The truncated-array fallback must not pull random {...} blocks out
    of surrounding prose (e.g. an English sentence in braces)."""
    raw = (
        'Here is the result you wanted:\n'
        '[{"name": "Llanelli", "entity_type": "place", "role": ""},\n'
        '   (some explanatory note {not an entity})\n'  # junk in braces
    )
    result = parse_ollama_entities(raw, _segments([
        "opening", "Llanelli is the town in question",
    ]))
    assert [e.name for e in result] == ["Llanelli"]


def test_parse_ollama_entities_drops_empty_name():
    raw = '[{"name": "", "entity_type": "person"}, {"name": "Dave", "entity_type": "person"}]'
    result = parse_ollama_entities(raw, _segments(["Dave walks in"]))
    assert len(result) == 1
    assert result[0].name == "Dave"


def test_parse_ollama_entities_empty_response():
    assert parse_ollama_entities("", _segments(3)) == []
    assert parse_ollama_entities("   \n  ", _segments(3)) == []
    assert parse_ollama_entities("no json here at all", _segments(3)) == []
    assert parse_ollama_entities("[not valid json", _segments(3)) == []


def test_parse_ollama_entities_drops_when_not_in_srt():
    """Entity name the LLM invented — substring search fails, entry dropped."""
    raw = '[{"name": "Alice", "entity_type": "person"}, {"name": "Frank", "entity_type": "person"}]'
    result = parse_ollama_entities(raw, _segments(["Alice greets the crowd"]))
    assert [e.name for e in result] == ["Alice"]


def test_parse_ollama_entities_ignores_models_srt_indices():
    """The model may still emit srt_indices — we ignore them and search instead.

    This prevents the LLM from hallucinating indices (regression for the bug
    found on welsh-farmer-birds where the model picked wrong/off-by-one indices).
    """
    raw = json.dumps([
        {"name": "Alice", "role": "Mayor", "entity_type": "person", "srt_indices": [99]},
    ])
    result = parse_ollama_entities(raw, _segments([
        "Alice greets the crowd",
        "second segment", "third segment",
    ]))
    assert len(result) == 1
    # Even though the model said [99], we found Alice in segment 1.
    assert result[0].srt_indices == [1]
    assert result[0].start_time == 0.0


def test_parse_ollama_entities_sorts_by_start_time():
    raw = json.dumps([
        {"name": "Carol", "entity_type": "person"},
        {"name": "Alice", "entity_type": "person"},
    ])
    result = parse_ollama_entities(raw, _segments([
        "Alice appears first", "middle filler", "Carol shows up later",
    ]))
    assert [e.name for e in result] == ["Alice", "Carol"]


def test_entity_span_defaults_to_person():
    """No entity_type field → Person (legacy responses stay back-compatible)."""
    e = EntitySpan(name="Alice", role="Mayor", start_time=0.0, end_time=2.0, srt_indices=[1])
    assert e.entity_type == EntityType.PERSON


def test_parse_ollama_entities_handles_each_type():
    """One entity of each canonical type round-trips with correct enum value."""
    raw = json.dumps([
        {"name": "Alice", "role": "Mayor", "entity_type": "person"},
        {"name": "Paris", "role": "France", "entity_type": "place"},
        {"name": "1869", "role": "", "entity_type": "date"},
        {"name": "99% solar", "role": "NASA", "entity_type": "info"},
    ])
    result = parse_ollama_entities(raw, _segments([
        "Alice walks into Paris",     # contains both Alice (person) and Paris (place)
        "second segment",
        "the year 1869 was eventful", # date
        "the panel reaches 99% solar generation efficiency",  # info (first word "99%" matches)
    ]))
    assert [e.entity_type for e in result] == [
        EntityType.PERSON, EntityType.PLACE, EntityType.DATE, EntityType.INFO,
    ]
    assert [e.name for e in result] == ["Alice", "Paris", "1869", "99% solar"]
    assert result[2].role == ""
    assert result[3].role == "NASA"


def test_parse_ollama_entities_type_aliases():
    """Common aliases map to canonical EntityType values."""
    raw = json.dumps([
        {"name": "Alice", "entity_type": "people"},
        {"name": "Paris", "entity_type": "location"},
        {"name": "1869", "entity_type": "year"},
        {"name": "99% solar", "entity_type": "fact"},
    ])
    result = parse_ollama_entities(raw, _segments([
        "Alice in Paris", "filler", "1869 saw revolution", "99% solar is impressive",
    ]))
    assert [e.entity_type for e in result] == [
        EntityType.PERSON, EntityType.PLACE, EntityType.DATE, EntityType.INFO,
    ]


def test_parse_ollama_entities_missing_type_defaults_to_person():
    """Back-compat: legacy responses without entity_type still parse as Person."""
    raw = '[{"name": "Alice", "role": ""}]'
    result = parse_ollama_entities(raw, _segments(["Alice waves"]))
    assert len(result) == 1
    assert result[0].entity_type == EntityType.PERSON


def test_parse_ollama_entities_unknown_type_drops_entry():
    """Unparseable entity_type (not a canonical or aliased value) drops the entry cleanly."""
    raw = json.dumps([
        {"name": "Alice", "entity_type": "person"},
        {"name": "Weird", "entity_type": "weird_type"},
        {"name": "Bob", "entity_type": "place"},
    ])
    result = parse_ollama_entities(raw, _segments([
        "Alice meets Bob",
    ]))
    # "Weird" has unparseable type → dropped. Alice and Bob both found in seg 1.
    assert [e.name for e in result] == ["Alice", "Bob"]


def test_parse_ollama_entities_accepts_type_alias_for_entity_type():
    """Some models use `type` instead of `entity_type` — both should work."""
    raw = json.dumps([{"name": "Paris", "type": "place"}])
    result = parse_ollama_entities(raw, _segments(["trip to Paris"]))
    assert len(result) == 1
    assert result[0].entity_type == EntityType.PLACE


def test_parse_ollama_entities_promotes_role_to_entity_type_when_missing():
    """When the LLM omits entity_type but puts a type value in `role`, promote it.

    This is the safety net for small models (llama3.2, mistral) that often dump
    classification info into the wrong field. The role must end up cleared so
    it doesn't render as on-screen text.
    """
    raw = json.dumps([
        {"name": "Burian Lake", "role": "place"},
        {"name": "Scott Peter", "role": "person"},
    ])
    result = parse_ollama_entities(raw, _segments([
        "At Burian Lake in 1991",
        "filler",
        "Scott Peter arrived",
    ]))
    assert [e.name for e in result] == ["Burian Lake", "Scott Peter"]
    assert result[0].entity_type == EntityType.PLACE
    assert result[0].role == ""  # cleared after promotion
    assert result[1].entity_type == EntityType.PERSON
    assert result[1].role == ""


def test_parse_ollama_entities_promotes_role_aliases_to_entity_type():
    """Region/country/landmark aliases in role are also detected and promoted."""
    raw = json.dumps([
        {"name": "Carmarthenshire", "role": "region"},
        {"name": "Wales", "role": "country"},
    ])
    result = parse_ollama_entities(raw, _segments([
        "Carmarthenshire is in Wales",
    ]))
    assert [e.entity_type for e in result] == [EntityType.PLACE, EntityType.PLACE]
    assert all(e.role == "" for e in result)


def test_parse_ollama_entities_keeps_role_when_also_type_typed():
    """When entity_type IS present, don't touch role — it's real on-screen text."""
    raw = json.dumps([
        {"name": "Alice", "entity_type": "person", "role": "Mayor"},
    ])
    result = parse_ollama_entities(raw, _segments(["Alice speaks"]))
    assert len(result) == 1
    assert result[0].entity_type == EntityType.PERSON
    assert result[0].role == "Mayor"


# ─────────────────────────────────────────────────────────────────────────────
# _find_entity_segments (substring search → SRT indices)
# ─────────────────────────────────────────────────────────────────────────────

def test_find_entity_segments_word_boundary():
    """Word-boundary match — 'Burian Lake' should not match 'Burianlets'."""
    segments = _segments([
        "1991, on the north shore of the Burian Lake",
        "filler",
        "The way most of the Burianlets edge had been",
    ])
    idx = _find_entity_segments("Burian Lake", segments, EntityType.PLACE)
    assert idx == [1]


def test_find_entity_segments_first_match_wins():
    """When an entity is mentioned multiple times, we use the FIRST mention."""
    segments = _segments([
        "second mention here",  # index 1
        "Bob arrives first",    # index 2 (first match)
        "Bob is back",
    ])
    idx = _find_entity_segments("Bob", segments, EntityType.PERSON)
    assert idx == [2]


def test_find_entity_segments_case_insensitive():
    segments = _segments(["BURTON arrives", "filler"])
    idx = _find_entity_segments("burton", segments, EntityType.PERSON)
    assert idx == [1]


def test_find_entity_segments_date_year_fallback():
    """A date entity named '1869' finds a segment where '1869' is mentioned as a year."""
    segments = _segments([
        "filler",
        "the Suez Canal opened in 1869",  # "1869" appears as a bare year
    ])
    idx = _find_entity_segments("1869 — Suez Canal opens", segments, EntityType.DATE)
    assert idx == [2]


def test_find_entity_segments_info_first_word_fallback():
    """An info entity with paraphrased name falls back to first word / first-2-words match."""
    segments = _segments([
        "filler",
        "the panel reaches 99% solar generation efficiency today",
    ])
    idx = _find_entity_segments("99% solar powered by 2010", segments, EntityType.INFO)
    # "99% solar" first 2 words match segment 2.
    assert idx == [2]


def test_find_entity_segments_no_match_returns_empty():
    segments = _segments(["nothing relevant here"])
    assert _find_entity_segments("Imaginary Place", segments, EntityType.PLACE) == []
    assert _find_entity_segments("9999", segments, EntityType.DATE) == []
    assert _find_entity_segments("some fact", segments, EntityType.INFO) == []


def test_parse_ollama_entities_off_by_one_segment_corrected():
    """Regression test for welsh-farmer-birds bug: model picks wrong SRT index,
    substring search uses the actual segment instead."""
    segments = _segments([
        "1991, on the north shore of the Burian Lake",  # seg 1
        "The way most of the Burianlets edge had been", # seg 2 — model would say [3]
        "managed for over a century",                   # seg 3
    ])
    raw = json.dumps([
        {"name": "Burianlets", "entity_type": "place", "srt_indices": [3]},  # WRONG index
    ])
    result = parse_ollama_entities(raw, segments)
    # The model's [3] is ignored; we find Burianlets in segment 2.
    assert len(result) == 1
    assert result[0].srt_indices == [2]
    assert result[0].start_time == 2.0


def test_parse_ollama_entities_hallucinated_entity_dropped():
    """If the entity name isn't anywhere in the SRT, drop it (no fallback to model indices)."""
    segments = _segments([
        "Built to carry visitors out into the reserve",
        "filler segment",
    ])
    raw = json.dumps([
        {"name": "Burian Lake", "entity_type": "place", "srt_indices": [1]},
    ])
    result = parse_ollama_entities(raw, segments)
    # Burian Lake is NOT mentioned in any segment → entry dropped.
    assert result == []


def test_otio_clip_metadata_includes_entity_type(tmp_path: Path):
    """Each clip's metadata carries the entity_type for downstream tools."""
    import opentimelineio as otio

    png_path = tmp_path / "a.png"
    Image.new("RGBA", (320, 180), (0, 0, 0, 0)).save(str(png_path), format="PNG")

    entities = [
        EntitySpan(
            name="Alice", role="Mayor",
            start_time=0.0, end_time=2.0, srt_indices=[1],
            entity_type=EntityType.PERSON,
        ),
        EntitySpan(
            name="Paris", role="France",
            start_time=4.0, end_time=6.0, srt_indices=[2],
            entity_type=EntityType.PLACE,
        ),
        EntitySpan(
            name="1869", role="",
            start_time=8.0, end_time=10.0, srt_indices=[3],
            entity_type=EntityType.DATE,
        ),
    ]
    pngs = [png_path] * 3
    out_otio = tmp_path / "out.otio"
    timeline = build_lower_thirds_otio(entities, pngs, ["classic"] * 3, out_otio)

    clips = [c for tr in timeline.tracks for c in tr if isinstance(c, otio.schema.Clip)]
    # Tracks are emitted in fixed TRACK_ORDER (PERSON, DATE, INFO, PLACE) — not
    # the entity input order — so the flat clip order reflects track ordering.
    assert [c.metadata["entity"]["entity_type"] for c in clips] == [
        "person", "date", "place",
    ]


# ─────────────────────────────────────────────────────────────────────────────
# clamp_entity_durations
# ─────────────────────────────────────────────────────────────────────────────

def test_clamp_entity_durations_preserves_entity_type_when_extending():
    """Regression: clamp_entity_durations was dropping entity_type for short
    entities (those whose (end - start) < duration_min), defaulting them back
    to PERSON. The reconstructed EntitySpan must carry the original type.

    Behaviour: every entity with duration < duration_min extends to the
    minimum unless doing so would overlap the next neighbour. Tail entities
    with room extend to the full minimum."""
    entities = [
        EntitySpan(
            name="Arctic", role="",
            start_time=10.0, end_time=11.5,  # 1.5s, < 2.0s — extendable
            srt_indices=[138], entity_type=EntityType.PLACE,
        ),
        EntitySpan(
            name="Burian Lake", role="",
            start_time=0.0, end_time=6.98,  # 6.98s, > 2.0s — not extended
            srt_indices=[1], entity_type=EntityType.PLACE,
        ),
        EntitySpan(
            name="Lanelli", role="",
            start_time=30.0, end_time=31.2,  # 1.2s, tail entity — extend to 32.0
            srt_indices=[170], entity_type=EntityType.PLACE,
        ),
    ]
    clamped = clamp_entity_durations(entities, duration_min=2.0)
    assert [e.entity_type for e in clamped] == [EntityType.PLACE] * 3
    # Sorted by start_time: Burian Lake (0), Arctic (10), Lanelli (30)
    assert clamped[0].end_time == 6.98  # Burian Lake unchanged
    # Arctic has room before Lanelli (gap = 18.5s) → extended to 12.0s
    assert clamped[1].start_time == 10.0
    assert clamped[1].end_time == 12.0
    # Lanelli is the tail entity → extended to 2.0s minimum (32.0s)
    assert clamped[2].end_time == 32.0


def test_clamp_entity_durations_does_not_extend_into_next_entity():
    """Two short entities in sequence: extending the first to 2s would overlap
    the second, so the first is capped at next_start - 1ms (no visual stacking).
    The second has no neighbour and extends to the full minimum."""
    entities = [
        _span("A", start=10.0, end=10.5),  # 0.5s, < 2.0s
        _span("B", start=11.0, end=12.5),  # 1.5s, < 2.0s, tail
    ]
    clamped = clamp_entity_durations(entities, duration_min=2.0)
    # A would need to extend to 12.0 — overlaps B (starts 11.0). Cap at 10.999.
    assert clamped[0].end_time == 10.999
    # B is the tail entity → extended to 2.0s minimum (13.0s)
    assert clamped[1].end_time == 13.0


def test_clamp_entity_durations_extends_when_room_available():
    """Two entities with a wide gap: the first can safely extend to min."""
    entities = [
        _span("A", start=10.0, end=10.5),
        _span("B", start=50.0, end=52.0),
    ]
    clamped = clamp_entity_durations(entities, duration_min=2.0)
    assert clamped[0].end_time == 12.0  # extended
    assert clamped[1].end_time == 52.0  # unchanged (already >= 2s)


def test_clamp_entity_durations_no_change_when_all_already_long():
    entities = [
        _span("A", start=0.0, end=5.0),
        _span("B", start=10.0, end=15.0),
    ]
    clamped = clamp_entity_durations(entities, duration_min=2.0)
    assert [(e.start_time, e.end_time) for e in clamped] == [(0.0, 5.0), (10.0, 15.0)]


def test_clamp_entity_durations_empty_input():
    assert clamp_entity_durations([], duration_min=2.0) == []


def test_redistribute_stacked_entities_splits_shared_srt_indices():
    """4 entities all on srt_indices=[26] (SRT 85.74→92.79, 7.05s) split
    into 4 sequential sub-windows within the same segment."""
    entities = [
        _span("Slimbridge", 85.74, 92.79, srt=[26]),
        _span("Gloucestershire", 85.74, 92.79, srt=[26]),
        _span("Wetlands", 85.74, 92.79, srt=[26]),
        _span("1946 — Trust", 85.74, 92.79, srt=[26]),
    ]
    out = redistribute_stacked_entities(entities)
    out_sorted = sorted(out, key=lambda e: e.start_time)
    # No time overlaps between redistributed entities
    for a, b in zip(out_sorted, out_sorted[1:]):
        assert a.end_time <= b.start_time + 1e-6, \
            f"{a.name} ends {a.end_time} > {b.name} starts {b.start_time}"
    # Union covers [85.74, 92.79]
    assert out_sorted[0].start_time == pytest.approx(85.74)
    assert out_sorted[-1].end_time == pytest.approx(92.79)
    # All four keep srt_indices=[26]
    assert [e.srt_indices for e in out] == [[26]] * 4


def test_redistribute_stacked_entities_no_op_for_singletons():
    """A single entity on an SRT index is left untouched."""
    entities = [
        _span("Wales", 30.0, 36.0, srt=[10]),
    ]
    out = redistribute_stacked_entities(entities)
    assert len(out) == 1
    assert out[0].start_time == 30.0
    assert out[0].end_time == 36.0
    assert out[0].srt_indices == [10]


def test_redistribute_stacked_entities_preserves_distinct_groups():
    """Two groups of stacked entities stay independent — entities in group A
    are not interleaved with entities in group B."""
    entities = [
        _span("Wales", 0.0, 10.0, srt=[1]),
        _span("Lake", 0.0, 10.0, srt=[1]),
        _span("Europe", 50.0, 55.0, srt=[25]),
        _span("Britain", 50.0, 55.0, srt=[25]),
    ]
    out = redistribute_stacked_entities(entities)
    # Group A still occupies 0–10s; group B occupies 50–55s
    assert out[0].start_time >= 0.0 and out[-1].end_time <= 10.0 or \
           out[2].start_time >= 50.0 and out[3].end_time <= 55.0
    # Specifically: entities 0+1 are in [0,10], entities 2+3 are in [50,55]
    assert out[0].end_time <= 10.0
    assert out[1].end_time <= 10.0
    assert out[2].start_time >= 50.0
    assert out[3].start_time >= 50.0


def test_redistribute_stacked_entities_empty_input():
    assert redistribute_stacked_entities([]) == []


def test_redistribute_stacked_entities_skips_empty_srt_indices():
    """An entity with empty srt_indices is passed through unchanged."""
    entities = [
        EntitySpan(name="X", role="", start_time=5.0, end_time=10.0,
                   srt_indices=[], entity_type=EntityType.PLACE),
        _span("Y", 0.0, 4.0, srt=[1]),
        _span("Z", 0.0, 4.0, srt=[1]),
    ]
    out = redistribute_stacked_entities(entities)
    assert out[0].srt_indices == []
    assert out[0].start_time == 5.0
    # Y and Z split the 4s segment
    assert out[1].end_time <= out[2].start_time + 1e-6


def test_redistribute_then_clamp_produces_no_overlap():
    """Integration: redistribute first (stacking fix) then clamp (timing fix).
    Combined pipeline produces non-overlapping, non-overshooting clips."""
    # Simulate the welsh-farmer-birds tail problem: two short segments at end
    entities = [
        _span("Arctic", 451.89, 453.08, srt=[138]),
        _span("Europe", 453.08, 454.30, srt=[139]),
        # And 4 stacked entities in the middle
        _span("Slimbridge", 85.74, 92.79, srt=[26]),
        _span("Gloucestershire", 85.74, 92.79, srt=[26]),
        _span("Wetlands", 85.74, 92.79, srt=[26]),
        _span("1946 — Trust", 85.74, 92.79, srt=[26]),
    ]
    entities = dedupe_entities(entities)
    entities = redistribute_stacked_entities(entities)
    entities = clamp_entity_durations(entities, duration_min=2.0)

    # Sort by start_time
    entities.sort(key=lambda e: e.start_time)

    # No overlap between adjacent clips
    for a, b in zip(entities, entities[1:]):
        assert a.end_time <= b.start_time + 1e-6, \
            f"overlap: {a.name} ({a.start_time:.2f}-{a.end_time:.2f}) -> " \
            f"{b.name} ({b.start_time:.2f}-{b.end_time:.2f})"
    # Tail (Europe) is now extended to the 2.0s minimum (453.08 + 2.0 = 455.08)
    assert entities[-1].end_time == pytest.approx(455.08, abs=1e-6)
    # Arctic is capped at next_start - 1ms to avoid overlapping Europe at 453.08
    arctic = next(e for e in entities if e.name == "Arctic")
    assert arctic.end_time == pytest.approx(453.079, abs=1e-3)


# ─────────────────────────────────────────────────────────────────────────────
# dedupe_entities
#
# Type-agnostic, name-only dedupe. Keeps the earliest occurrence and drops
# later duplicates — no semantic merge (Peter Scott / Scott, UK / United
# Kingdom, Wales / New South Wales all stay separate).
# ─────────────────────────────────────────────────────────────────────────────

def _span(name: str, start: float, end: float = 2.0,
          role: str = "", typ: EntityType = EntityType.PLACE,
          srt: list[int] | None = None) -> EntitySpan:
    return EntitySpan(
        name=name, role=role,
        start_time=start, end_time=end,
        srt_indices=srt if srt is not None else [1],
        entity_type=typ,
    )


def test_dedupe_entities_keeps_earliest_duplicate():
    entities = [
        _span("Wales", 10.0),
        _span("Wales", 30.0),
        _span("Wales", 50.0),
    ]
    result = dedupe_entities(entities)
    assert len(result) == 1
    assert result[0].start_time == 10.0


def test_dedupe_entities_case_and_whitespace_insensitive():
    entities = [
        _span("Wales", 5.0),
        _span("  wales ", 6.0),
        _span("WALES", 7.0),
    ]
    result = dedupe_entities(entities)
    assert len(result) == 1
    assert result[0].start_time == 5.0


def test_dedupe_entities_strips_leading_the():
    entities = [
        _span("The Wales", 20.0),
        _span("Wales", 30.0),
        _span("the wales", 40.0),
    ]
    result = dedupe_entities(entities)
    assert len(result) == 1
    assert result[0].start_time == 20.0


def test_dedupe_entities_is_type_agnostic():
    """Same name with different entity_type must collapse (user requirement:
    'any type of name should be considered')."""
    entities = [
        _span("Birds", 5.0, typ=EntityType.PLACE),
        _span("Birds", 9.0, typ=EntityType.INFO),
    ]
    result = dedupe_entities(entities)
    assert len(result) == 1
    # Earliest (place) wins, even though info is more "abstract" semantically.
    assert result[0].start_time == 5.0
    assert result[0].entity_type == EntityType.PLACE


def test_dedupe_entities_preserves_distinct_names():
    entities = [
        _span("Peter Scott", 5.0),
        _span("Scott", 6.0),
    ]
    result = dedupe_entities(entities)
    assert [e.name for e in result] == ["Peter Scott", "Scott"]


def test_dedupe_entities_preserves_distinct_synonyms():
    entities = [
        _span("UK", 5.0),
        _span("United Kingdom", 6.0),
    ]
    result = dedupe_entities(entities)
    assert [e.name for e in result] == ["UK", "United Kingdom"]


def test_dedupe_entities_preserves_substring_overlap():
    """Substring matches (Wales / New South Wales) are not deduped."""
    entities = [
        _span("Wales", 5.0),
        _span("New South Wales", 6.0),
    ]
    result = dedupe_entities(entities)
    assert [e.name for e in result] == ["Wales", "New South Wales"]


def test_dedupe_entities_empty_input():
    assert dedupe_entities([]) == []


def test_dedupe_entities_unsorted_input_returns_timeline_order():
    """Input in random order must come out sorted by start_time."""
    entities = [
        _span("Llanelli", 566.0),
        _span("Burian Lake", 0.0),
        _span("Slimbridge", 85.0),
    ]
    result = dedupe_entities(entities)
    assert [e.name for e in result] == ["Burian Lake", "Slimbridge", "Llanelli"]


def test_dedupe_entities_keeps_winning_role_and_indices():
    """The first occurrence's role + srt_indices must survive the merge."""
    entities = [
        _span("Wales", 5.0, role="Cymru", srt=[1]),
        _span("Wales", 30.0, role="different role", srt=[42]),
    ]
    result = dedupe_entities(entities)
    assert len(result) == 1
    assert result[0].role == "Cymru"
    assert result[0].srt_indices == [1]


# ─────────────────────────────────────────────────────────────────────────────
# render_lower_third_png
# ─────────────────────────────────────────────────────────────────────────────

def test_render_lower_third_png_creates_rgba_file(tmp_path: Path):
    template = LowerThirdTemplate(width=320, height=180, bar_height=40)
    entity = EntitySpan(name="Alice", role="Mayor", start_time=0.0, end_time=2.0, srt_indices=[1])
    out = render_lower_third_png(entity, tmp_path / "lt.png", template)
    assert out.exists()
    assert out.stat().st_size > 0
    from PIL import Image
    with Image.open(out) as img:
        assert img.mode == "RGBA"
        assert img.size == (320, 180)


def test_render_lower_third_png_role_only_or_name_only(tmp_path: Path):
    template = LowerThirdTemplate(width=320, height=180, bar_height=40)
    role_only = EntitySpan(name="Bob", role="", start_time=0.0, end_time=2.0, srt_indices=[1])
    out1 = render_lower_third_png(role_only, tmp_path / "lt1.png", template)
    assert out1.exists()

    no_role = EntitySpan(name="Carol", role="Senator", start_time=0.0, end_time=2.0, srt_indices=[1])
    out2 = render_lower_third_png(no_role, tmp_path / "lt2.png", template)
    assert out2.exists()


def test_render_long_name_truncates(tmp_path: Path):
    """A 500-char name must not crash the renderer and must produce a valid PNG.

    We don't assert exact pixel widths (font metrics vary); we only assert that
    the truncation + ellipsis path runs to completion and yields a valid image.
    """
    template = LowerThirdTemplate(
        width=320, height=180, bar_height=40, name_font_size=24, role_font_size=14,
        accent_width=0, pad_x=10,
    )
    long_name = "A" * 500
    entity = EntitySpan(name=long_name, role="X", start_time=0.0, end_time=2.0, srt_indices=[1])
    out = render_lower_third_png(entity, tmp_path / "lt_long.png", template)
    assert out.exists()
    assert out.stat().st_size > 0

    with Image.open(out) as img:
        assert img.mode == "RGBA"
        assert img.size == (320, 180)
        # Image should have some non-transparent pixels (the rendered text)
        alpha_band = img.split()[-1]
        assert alpha_band.getbbox() is not None


# ─────────────────────────────────────────────────────────────────────────────
# build_lower_thirds_otio
# ─────────────────────────────────────────────────────────────────────────────

def test_build_lower_thirds_otio_track_name_and_clip_count(tmp_path: Path):
    import opentimelineio as otio

    entities = [
        EntitySpan(name="Alice", role="Mayor", start_time=0.0, end_time=2.0, srt_indices=[1]),
        EntitySpan(name="Bob", role="Senator", start_time=4.0, end_time=6.0, srt_indices=[2, 3]),
    ]
    pngs = [
        tmp_path / "lowerthird_001.png",
        tmp_path / "lowerthird_002.png",
    ]
    for p in pngs:
        Image.new("RGBA", (320, 180), (0, 0, 0, 0)).save(str(p), format="PNG")

    out_otio = tmp_path / "lowerthirds.otio"
    timeline = build_lower_thirds_otio(entities, pngs, ["classic"] * 2, out_otio)

    assert timeline.name == "lowerthirds_timeline"
    # Two person entities → one track ("V1 - Lower Thirds · Names")
    assert len(timeline.tracks) == 1
    track = timeline.tracks[0]
    assert track.name == "V1 - Lower Thirds · Names"
    assert track.kind == otio.schema.TrackKind.Video

    clips = [c for c in track if isinstance(c, otio.schema.Clip)]
    assert len(clips) == 2
    assert clips[0].name.startswith("LT:001:Alice")
    assert clips[1].name.startswith("LT:002:Bob")
    assert "Alice" in clips[0].metadata["entity"]["name"]
    assert clips[0].metadata["entity"]["template"] == "classic"


def test_build_lower_thirds_otio_includes_freeze_frame_and_gaps(tmp_path: Path):
    import opentimelineio as otio

    entities = [
        EntitySpan(name="X", role="", start_time=2.0, end_time=4.0, srt_indices=[2]),
        EntitySpan(name="Y", role="", start_time=8.0, end_time=10.0, srt_indices=[4]),
    ]
    pngs = [tmp_path / "a.png", tmp_path / "b.png"]
    for p in pngs:
        Image.new("RGBA", (320, 180), (0, 0, 0, 0)).save(str(p), format="PNG")

    out_otio = tmp_path / "out.otio"
    timeline = build_lower_thirds_otio(entities, pngs, ["classic"] * 2, out_otio)
    track = timeline.tracks[0]

    clips = [c for c in track if isinstance(c, otio.schema.Clip)]
    gaps = [c for c in track if isinstance(c, otio.schema.Gap)]
    assert len(clips) == 2
    assert len(gaps) >= 1

    for clip in clips:
        assert any(isinstance(e, otio.schema.FreezeFrame) for e in clip.effects)


def test_build_lower_thirds_otio_target_url_uses_forward_slashes(tmp_path: Path, monkeypatch):
    """On Windows, the target_url must use forward slashes (Resolve import safety)."""
    import opentimelineio as otio

    png_path = tmp_path / "a.png"
    Image.new("RGBA", (320, 180), (0, 0, 0, 0)).save(str(png_path), format="PNG")

    entities = [EntitySpan(name="A", role="", start_time=0.0, end_time=2.0, srt_indices=[1])]
    out_otio = tmp_path / "out.otio"
    build_lower_thirds_otio(entities, [png_path], ["classic"], out_otio)

    track = out_otio  # file path; we just check the file exists and round-trips
    assert track.exists()

    timeline = otio.adapters.read_from_file(str(out_otio))
    clip = timeline.tracks[0][0]
    assert isinstance(clip, otio.schema.Clip)
    target = clip.media_reference.target_url
    assert "\\" not in target
    assert "/" in target


def test_build_lower_thirds_otio_empty_entities(tmp_path: Path):
    import opentimelineio as otio

    out_otio = tmp_path / "empty.otio"
    timeline = build_lower_thirds_otio([], [], [], out_otio)
    assert len(timeline.tracks) == 0
    assert out_otio.exists()


def test_build_lower_thirds_otio_video_clip_has_no_freeze_frame(tmp_path: Path):
    """Rendered .mov clips carry animation and MUST play as video, not freeze.

    Regression: OTIO applied FreezeFrame to every clip (images and videos
    alike), which made DaVinci Resolve import the animated .movs as still
    frames — wiping out the slide/pop/fade animation. FreezeFrame is now
    only applied to image media; video media plays naturally.
    """
    import opentimelineio as otio

    mov_path = tmp_path / "anim.mov"
    mov_path.write_bytes(b"")  # placeholder; just checking OTIO metadata, not real file

    entities = [EntitySpan(name="A", role="", start_time=0.0, end_time=2.0, srt_indices=[1])]
    out_otio = tmp_path / "out.otio"
    timeline = build_lower_thirds_otio(entities, [mov_path], ["classic"], out_otio)
    track = timeline.tracks[0]
    clip = track[0]
    assert isinstance(clip, otio.schema.Clip)
    assert not any(isinstance(e, otio.schema.FreezeFrame) for e in clip.effects)
    assert clip.metadata["entity"]["media_kind"] == "video"
    assert clip.metadata["entity"]["media_file"] == "anim.mov"


def test_build_lower_thirds_otio_png_clip_still_has_freeze_frame(tmp_path: Path):
    """PNG references SHOULD still get a FreezeFrame (regression check)."""
    import opentimelineio as otio

    png_path = tmp_path / "a.png"
    Image.new("RGBA", (320, 180), (0, 0, 0, 0)).save(str(png_path), format="PNG")

    entities = [EntitySpan(name="A", role="", start_time=0.0, end_time=2.0, srt_indices=[1])]
    out_otio = tmp_path / "out.otio"
    timeline = build_lower_thirds_otio(entities, [png_path], ["classic"], out_otio)
    clip = timeline.tracks[0][0]
    assert isinstance(clip, otio.schema.Clip)
    assert any(isinstance(e, otio.schema.FreezeFrame) for e in clip.effects)
    assert clip.metadata["entity"]["media_kind"] == "image"


# ─────────────────────────────────────────────────────────────────────────────
# parse_template_by_type
# ─────────────────────────────────────────────────────────────────────────────

def test_parse_template_by_type_basic():
    """`person=classic,date=boxed,info=modern` → typed dict with 3 entries."""
    result = parse_template_by_type("person=classic,date=boxed,info=modern")
    assert result == {
        EntityType.PERSON: "classic",
        EntityType.DATE:   "boxed",
        EntityType.INFO:   "modern",
    }


def test_parse_template_by_type_strips_whitespace():
    """Surrounding whitespace on keys and values is tolerated."""
    result = parse_template_by_type(" person = classic , date=boxed ")
    assert result == {
        EntityType.PERSON: "classic",
        EntityType.DATE:   "boxed",
    }


def test_parse_template_by_type_unknown_entity_type_logs_and_skips(caplog):
    """Unknown entity types log an error and are skipped (no fail)."""
    with caplog.at_level("ERROR", logger="lower_thirds"):
        result = parse_template_by_type("weird=classic,person=boxed")
    # `weird` was skipped, `person` made it through.
    assert result == {EntityType.PERSON: "boxed"}
    assert any("unknown entity_type" in rec.message for rec in caplog.records)


def test_parse_template_by_type_unknown_template_logs_and_skips(caplog):
    """Unknown template names log an error and are skipped (no fail)."""
    with caplog.at_level("ERROR", logger="lower_thirds"):
        result = parse_template_by_type("person=nonexistent,date=boxed")
    assert result == {EntityType.DATE: "boxed"}
    assert any("unknown template" in rec.message for rec in caplog.records)


def test_parse_template_by_type_empty_string():
    """Empty spec → empty dict."""
    assert parse_template_by_type("") == {}


def test_parse_template_by_type_alias_resolves():
    """Aliases (e.g. `location` → PLACE) work the same as `_normalize_entity_type`."""
    result = parse_template_by_type("location=classic,person=newsroom")
    assert result == {
        EntityType.PLACE:  "classic",
        EntityType.PERSON: "newsroom",
    }


# ─────────────────────────────────────────────────────────────────────────────
# build_lower_thirds_otio — multi-track behavior
# ─────────────────────────────────────────────────────────────────────────────

def _png(tmp_path: Path, name: str = "a.png") -> Path:
    p = tmp_path / name
    Image.new("RGBA", (320, 180), (0, 0, 0, 0)).save(str(p), format="PNG")
    return p


def _mixed_entities() -> list[EntitySpan]:
    """Five entities, one per category, with non-sorted times to test track-order."""
    return [
        EntitySpan(name="Wales",    role="",       start_time=12.0, end_time=14.0, srt_indices=[1], entity_type=EntityType.PLACE),
        EntitySpan(name="Alice",    role="Mayor",  start_time=0.0,  end_time=2.0,  srt_indices=[1], entity_type=EntityType.PERSON),
        EntitySpan(name="1946",     role="",       start_time=4.0,  end_time=6.0,  srt_indices=[2], entity_type=EntityType.DATE),
        EntitySpan(name="Wetlands", role="",       start_time=8.0,  end_time=10.0, srt_indices=[3], entity_type=EntityType.INFO),
        EntitySpan(name="Opening",  role="intro",  start_time=16.0, end_time=20.0, srt_indices=[5], entity_type=EntityType.CHAPTER),
    ]


def test_build_lower_thirds_otio_emits_one_track_per_category(tmp_path: Path):
    """One entity per category → 5 tracks, in TRACK_ORDER."""
    import opentimelineio as otio

    entities = _mixed_entities()
    pngs = [_png(tmp_path, f"p{i}.png") for i in range(len(entities))]
    out_otio = tmp_path / "out.otio"

    timeline = build_lower_thirds_otio(entities, pngs, ["classic"] * 5, out_otio)

    assert len(timeline.tracks) == 5
    expected_names = [TRACK_NAME_BY_TYPE[t] for t in TRACK_ORDER]
    assert [tr.name for tr in timeline.tracks] == expected_names

    # Each track has exactly 1 clip.
    for tr in timeline.tracks:
        clips = [c for c in tr if isinstance(c, otio.schema.Clip)]
        assert len(clips) == 1


def test_build_lower_thirds_otio_omits_empty_category_tracks(tmp_path: Path):
    """Categories with zero entities are skipped entirely (no empty track)."""
    import opentimelineio as otio

    entities = [
        EntitySpan(name="Alice", role="Mayor", start_time=0.0, end_time=2.0, srt_indices=[1], entity_type=EntityType.PERSON),
        EntitySpan(name="1946",  role="",      start_time=4.0, end_time=6.0, srt_indices=[2], entity_type=EntityType.DATE),
    ]
    pngs = [_png(tmp_path, f"p{i}.png") for i in range(len(entities))]
    out_otio = tmp_path / "out.otio"

    timeline = build_lower_thirds_otio(entities, pngs, ["classic", "boxed"], out_otio)

    # Only Names + Dates tracks — no Key Points, no Places.
    assert len(timeline.tracks) == 2
    assert [tr.name for tr in timeline.tracks] == [
        "V1 - Lower Thirds · Names",
        "V2 - Lower Thirds · Dates",
    ]


def test_build_lower_thirds_otio_carries_template_in_metadata(tmp_path: Path):
    """Each clip's metadata['entity']['template'] matches its per-type template."""
    import opentimelineio as otio

    # Input entities are in mixed order; templates are aligned to that input order.
    entities = _mixed_entities()  # order: PLACE, PERSON, DATE, INFO, CHAPTER
    pngs = [_png(tmp_path, f"p{i}.png") for i in range(len(entities))]
    out_otio = tmp_path / "out.otio"

    # PLACE=classic, PERSON=boxed, DATE=modern, INFO=corner, CHAPTER=newsroom.
    template_names = ["classic", "boxed", "modern", "corner", "newsroom"]
    timeline = build_lower_thirds_otio(entities, pngs, template_names, out_otio)

    # Walk all clips across all tracks and verify their template matches input.
    clips_with_template = {
        c.metadata["entity"]["entity_type"]: c.metadata["entity"]["template"]
        for tr in timeline.tracks
        for c in tr if isinstance(c, otio.schema.Clip)
    }
    assert clips_with_template == {
        "place":   "classic",
        "person":  "boxed",
        "date":    "modern",
        "info":    "corner",
        "chapter": "newsroom",
    }


def test_build_lower_thirds_otio_track_order_is_fixed(tmp_path: Path):
    """Input order is irrelevant: tracks come out as Names→Dates→KeyPoints→Places→Chapters."""
    entities = _mixed_entities()
    pngs = [_png(tmp_path, f"p{i}.png") for i in range(len(entities))]
    out_otio = tmp_path / "out.otio"

    timeline = build_lower_thirds_otio(entities, pngs, ["classic"] * 5, out_otio)

    # TRACK_ORDER is the source of truth — assert we honor it.
    assert [tr.name for tr in timeline.tracks] == [TRACK_NAME_BY_TYPE[t] for t in TRACK_ORDER]


def test_build_lower_thirds_otio_gaps_per_track(tmp_path: Path):
    """Gaps live WITHIN a track, never BETWEEN tracks (different from single-track)."""
    import opentimelineio as otio

    # Two person entities separated by a big gap; one date entity on its own track.
    entities = [
        EntitySpan(name="Alice", role="Mayor", start_time=0.0,  end_time=2.0,  srt_indices=[1], entity_type=EntityType.PERSON),
        EntitySpan(name="Bob",   role="Chef",  start_time=10.0, end_time=12.0, srt_indices=[2], entity_type=EntityType.PERSON),
        EntitySpan(name="1946",  role="",      start_time=4.0,  end_time=6.0,  srt_indices=[3], entity_type=EntityType.DATE),
    ]
    pngs = [_png(tmp_path, f"p{i}.png") for i in range(len(entities))]
    out_otio = tmp_path / "out.otio"

    timeline = build_lower_thirds_otio(entities, pngs, ["classic"] * 3, out_otio)
    assert len(timeline.tracks) == 2

    # V1 (Names) carries both Alice and Bob with an inter-clip gap; no leading
    # gap is needed because Alice starts at 0.
    names_track = timeline.tracks[0]
    assert names_track.name == "V1 - Lower Thirds · Names"
    kinds = [type(c).__name__ for c in names_track]
    assert kinds.count("Gap") >= 1   # the gap between Alice and Bob
    assert kinds.count("Clip") == 2

    # V2 (Dates) is independent — its own clip, not stretched to fill V1's gap.
    dates_track = timeline.tracks[1]
    assert dates_track.name == "V2 - Lower Thirds · Dates"
    assert any(isinstance(c, otio.schema.Clip) for c in dates_track)


# ─────────────────────────────────────────────────────────────────────────────
# build_html_template
# ─────────────────────────────────────────────────────────────────────────────

def _entity() -> EntitySpan:
    return EntitySpan(name="Alice", role="Mayor", start_time=0.0, end_time=2.0, srt_indices=[1])


def test_build_html_template_includes_entity_text():
    entity = _entity()
    template = LowerThirdTemplate(width=1920, height=1080)
    html = build_html_template(entity, template, "slide", anim_duration_s=0.4)
    assert "Alice" in html
    assert "Mayor" in html
    assert "data-style=\"slide\"" in html
    assert "@keyframes slide-in" in html
    # All keyframes are defined; only the active style's selectors are wired.
    assert "body[data-style=\"slide\"] .bar" in html
    assert "body[data-style=\"pop\"] .bar" in html
    assert "body[data-style=\"fade\"] .bar" in html


def test_build_html_template_pop_includes_pop_keyframes():
    entity = _entity()
    template = LowerThirdTemplate(width=1920, height=1080)
    html = build_html_template(entity, template, "pop", anim_duration_s=0.5)
    assert "data-style=\"pop\"" in html
    assert "@keyframes pop-in" in html


def test_build_html_template_fade_includes_fade_keyframes():
    entity = _entity()
    template = LowerThirdTemplate(width=1920, height=1080)
    html = build_html_template(entity, template, "fade", anim_duration_s=0.4)
    assert "data-style=\"fade\"" in html
    assert "@keyframes fade-in" in html


def test_build_html_template_no_role_omits_role_div():
    entity = EntitySpan(name="Bob", role="", start_time=0.0, end_time=2.0, srt_indices=[1])
    template = LowerThirdTemplate(width=1920, height=1080)
    html = build_html_template(entity, template, "slide", anim_duration_s=0.4)
    assert "Bob" in html
    assert "class=\"role\"" not in html


def test_build_html_template_rejects_unknown_style():
    entity = _entity()
    template = LowerThirdTemplate(width=1920, height=1080)
    with pytest.raises(ValueError, match="Unknown animation style"):
        build_html_template(entity, template, "spin", anim_duration_s=0.4)


def test_build_html_template_uses_template_dimensions():
    entity = _entity()
    template = LowerThirdTemplate(width=1280, height=720, bar_height=120, accent_width=4)
    html = build_html_template(entity, template, "slide", anim_duration_s=0.4)
    assert "1280px" in html
    assert "720px" in html
    assert "120px" in html  # bar height
    assert "4px" in html  # accent width


def test_build_html_template_html_escapes_special_chars():
    entity = EntitySpan(
        name='<script>"hi"</script>', role="A & B", start_time=0.0, end_time=2.0, srt_indices=[1]
    )
    template = LowerThirdTemplate()
    html = build_html_template(entity, template, "slide", anim_duration_s=0.4)
    assert "<script>" not in html
    assert "&lt;script&gt;" in html
    assert "A &amp; B" in html


def test_animation_styles_constant():
    assert set(ANIMATION_STYLES) == {"slide", "pop", "fade"}


def test_build_html_template_slide_scales_text_fade_in_with_short_anim():
    """When anim_duration_s is very short (e.g., 0.1s for a 0.2s clip),
    the text-wrap fade-in duration must scale DOWN from the 0.3s default
    so the text reaches opacity 1 before the clip ends."""
    entity = _entity()
    template = LowerThirdTemplate(width=1920, height=1080)
    html = build_html_template(entity, template, "slide", anim_duration_s=0.1)
    # text-wrap fade-in duration is clamped to anim*0.5 = 0.05s
    assert "fade-in 0.050s ease-out 0.050s forwards" in html


def test_build_html_template_slide_keeps_text_fade_in_cap_at_0_3s():
    """For default-length anims, the text-wrap fade-in stays at 0.3s."""
    entity = _entity()
    template = LowerThirdTemplate(width=1920, height=1080)
    html = build_html_template(entity, template, "slide", anim_duration_s=0.4)
    # anim*0.5 = 0.2s — but min(0.3, 0.2) = 0.2s, so fade-in duration is 0.2s
    assert "fade-in 0.200s ease-out 0.200s forwards" in html


def test_build_html_template_pop_scales_text_fade_in_with_short_anim():
    """Pop style uses the same fade-in scale-down as slide."""
    entity = _entity()
    template = LowerThirdTemplate(width=1920, height=1080)
    html = build_html_template(entity, template, "pop", anim_duration_s=0.1)
    assert "fade-in 0.050s ease-out 0.050s forwards" in html


def test_build_html_template_fade_scales_text_fade_with_short_anim():
    """Fade style: text-wrap fade-in start delay and duration must both
    scale down so text reaches opacity 1 inside a short clip."""
    entity = _entity()
    template = LowerThirdTemplate(width=1920, height=1080)
    # 0.1s anim → duration min(0.4, max(0.05, 0.1-0.1))=0.05, delay min(0.1, 0.025)=0.025
    html = build_html_template(entity, template, "fade", anim_duration_s=0.1)
    assert "fade-in 0.050s ease-out 0.025s forwards" in html


def test_build_html_template_fade_text_fade_for_default_anim():
    """For 0.4s default anim, fade style text-wrap fade-in delay=0.1s
    and duration=0.3s (min(0.4, max(0.05, 0.3)) = 0.3)."""
    entity = _entity()
    template = LowerThirdTemplate(width=1920, height=1080)
    html = build_html_template(entity, template, "fade", anim_duration_s=0.4)
    assert "fade-in 0.300s ease-out 0.100s forwards" in html


def test_render_lower_third_video_clamps_anim_for_short_entity(monkeypatch, tmp_path: Path):
    """render_lower_third_video must clamp anim_duration_s to half the
    entity duration so the fade-in completes inside the clip."""
    from lower_thirds import render_lower_third_video  # noqa: E402
    # 0.3s entity, 0.4s requested anim — should clamp to 0.15s
    short_entity = EntitySpan(
        name="China", role="", start_time=0.0, end_time=0.3, srt_indices=[1]
    )
    template = LowerThirdTemplate(width=1920, height=1080)

    captured_calls = []

    def fake_capture(html_path, frame_dir, anim_frames, fps):
        frame_dir.mkdir(parents=True, exist_ok=True)
        captured_calls.append(("capture", anim_frames))

    def fake_build(frame_dir, output_path, total_duration_s, fps, anim_duration_s):
        captured_calls.append(("build", total_duration_s, anim_duration_s))
        output_path.touch()

    monkeypatch.setattr("lower_thirds.capture_animation_frames", fake_capture)
    monkeypatch.setattr("lower_thirds.build_lower_third_video", fake_build)

    out = tmp_path / "lt.mov"
    render_lower_third_video(
        short_entity, out, template, style="slide",
        anim_duration_s=0.4, fps=30,
    )
    # The clamp should have made anim=0.15s (total * 0.5)
    # ceil(0.15 * 30) + 1 = 6
    assert ("capture", 6) in captured_calls
    # build sees total=0.3, anim=0.15
    build_call = next(c for c in captured_calls if c[0] == "build")
    assert abs(build_call[1] - 0.3) < 0.01
    assert abs(build_call[2] - 0.15) < 0.01


def test_render_lower_third_video_keeps_long_anim_unchanged(monkeypatch, tmp_path: Path):
    """For entities long enough to accommodate the full anim, no clamp."""
    from lower_thirds import render_lower_third_video  # noqa: E402
    long_entity = EntitySpan(
        name="United States", role="", start_time=0.0, end_time=3.0, srt_indices=[1]
    )
    template = LowerThirdTemplate(width=1920, height=1080)

    captured = []

    def fake_capture(html_path, frame_dir, anim_frames, fps):
        frame_dir.mkdir(parents=True, exist_ok=True)
        captured.append(("capture", anim_frames))

    def fake_build(frame_dir, output_path, total_duration_s, fps, anim_duration_s):
        captured.append(("build", anim_duration_s))
        output_path.touch()

    monkeypatch.setattr("lower_thirds.capture_animation_frames", fake_capture)
    monkeypatch.setattr("lower_thirds.build_lower_third_video", fake_build)

    out = tmp_path / "lt.mov"
    render_lower_third_video(
        long_entity, out, template, style="slide",
        anim_duration_s=0.4, fps=30,
    )
    build_call = next(c for c in captured if c[0] == "build")
    # anim stays at 0.4s (clip is 3.0s, half = 1.5s)
    assert abs(build_call[1] - 0.4) < 0.01


def test_sanitize_chapter_title_strips_duplicated_prefix():
    """LLM sometimes emits 'second - second quarter ride'. Strip the
    duplicated first word."""
    from lower_thirds import _sanitize_chapter_title  # noqa: E402
    assert _sanitize_chapter_title("second - second quarter ride") == "second quarter ride"
    assert _sanitize_chapter_title("first - robotaxis first concentrate") == "robotaxis first concentrate"
    assert _sanitize_chapter_title("number to - important number watch") == "important number watch"


def test_sanitize_chapter_title_keeps_normal_titles():
    """Normal chapter titles pass through unchanged."""
    from lower_thirds import _sanitize_chapter_title  # noqa: E402
    assert _sanitize_chapter_title("Introduction") == "Introduction"
    assert _sanitize_chapter_title("Conclusion") == "Conclusion"
    assert _sanitize_chapter_title("Expanding Global Reach") == "Expanding Global Reach"


def test_sanitize_chapter_title_truncates_overlong():
    """Titles over 40 chars get truncated at word boundary."""
    from lower_thirds import _sanitize_chapter_title  # noqa: E402
    long = "Zagreb's Role in Autonomous Vehicle Development and Testing"
    out = _sanitize_chapter_title(long)
    assert len(out) <= 40
    # Must end on a word, not mid-word
    assert not out.endswith((".", ",", ";", ":"))


def test_sanitize_chapter_title_handles_empty():
    from lower_thirds import _sanitize_chapter_title  # noqa: E402
    assert _sanitize_chapter_title("") == ""
    assert _sanitize_chapter_title("   ") == ""  # whitespace strips to empty


def test_render_lower_third_video_captures_enough_frames_for_short_anim(monkeypatch, tmp_path: Path):
    """Very short anim (0.085s for a 0.17s clip) must still capture at
    least ceil(anim*fps)+1 frames so the last frame lands AT or past the
    animation end. Otherwise the fade-in never reaches opacity 1 in any
    captured frame."""
    from lower_thirds import render_lower_third_video  # noqa: E402
    yes_entity = EntitySpan(
        name="Yes", role="", start_time=0.0, end_time=0.17, srt_indices=[1]
    )
    template = LowerThirdTemplate(width=1920, height=1080)

    captured = []

    def fake_capture(html_path, frame_dir, anim_frames, fps):
        frame_dir.mkdir(parents=True, exist_ok=True)
        captured.append(anim_frames)

    def fake_build(frame_dir, output_path, total_duration_s, fps, anim_duration_s):
        output_path.touch()

    monkeypatch.setattr("lower_thirds.capture_animation_frames", fake_capture)
    monkeypatch.setattr("lower_thirds.build_lower_third_video", fake_build)

    out = tmp_path / "lt.mov"
    render_lower_third_video(
        yes_entity, out, template, style="slide",
        anim_duration_s=0.4, fps=30,
    )
    # Clamp: anim = min(0.4, 0.17*0.5) = 0.085
    # ceil(0.085*30) + 1 = ceil(2.55)+1 = 4
    assert captured == [4]


# ─────────────────────────────────────────────────────────────────────────────
# find_srt_path
# ─────────────────────────────────────────────────────────────────────────────

def test_find_srt_path_prefers_trimmed(tmp_path: Path):
    voiceover = tmp_path / "voiceover"
    voiceover.mkdir()
    (voiceover / "voiceover.srt").write_text("1\n00:00:00,000 --> 00:00:01,000\nA\n", encoding="utf-8")
    (voiceover / "voiceover_trimmed.srt").write_text("1\n00:00:00,000 --> 00:00:01,000\nA\n", encoding="utf-8")

    result = find_srt_path(tmp_path)
    assert result is not None
    assert result.name == "voiceover_trimmed.srt"


def test_find_srt_path_falls_back_to_untrimmed(tmp_path: Path):
    voiceover = tmp_path / "voiceover"
    voiceover.mkdir()
    (voiceover / "voiceover.srt").write_text("1\n00:00:00,000 --> 00:00:01,000\nA\n", encoding="utf-8")

    result = find_srt_path(tmp_path)
    assert result is not None
    assert result.name == "voiceover.srt"


def test_find_srt_path_no_srt_returns_none(tmp_path: Path):
    assert find_srt_path(tmp_path) is None


# ─────────────────────────────────────────────────────────────────────────────
# Template registry + per-layout rendering
# ─────────────────────────────────────────────────────────────────────────────

def test_template_registry_has_six_named_templates():
    assert set(TEMPLATE_REGISTRY.keys()) == {
        "classic", "minimal", "boxed", "modern", "corner", "newsroom",
    }
    assert len(TEMPLATE_REGISTRY) == 6


def test_template_registry_classic_matches_default_template():
    """classic must be byte-identical to today's default output."""
    default = LowerThirdTemplate()
    classic = TEMPLATE_REGISTRY["classic"]
    for f in default.__dataclass_fields__:
        assert getattr(default, f) == getattr(classic, f), f"field {f!r} diverged"


@pytest.mark.parametrize("template_name", sorted(TEMPLATE_REGISTRY.keys()))
def test_render_lower_third_png_each_template_produces_valid_png(tmp_path: Path, template_name: str):
    template = TEMPLATE_REGISTRY[template_name]
    entity = EntitySpan(name="Alice", role="Mayor", start_time=0.0, end_time=2.0, srt_indices=[1])
    out = render_lower_third_png(entity, tmp_path / f"lt_{template_name}.png", template)
    assert out.exists()
    assert out.stat().st_size > 0
    with Image.open(out) as img:
        assert img.mode == "RGBA"
        assert img.size == (template.width, template.height)
        alpha = img.split()[-1]
        assert alpha.getbbox() is not None, f"{template_name}: blank image"


def test_template_layouts_have_distinct_layouts():
    """All 5 Layout enum values must be exercised by the registry.

    (classic and boxed both use Layout.BAR but with different field values —
    the difference is in shadow/radius/padding, not the layout enum.)
    """
    layouts = {t.layout for t in TEMPLATE_REGISTRY.values()}
    assert layouts == set(Layout), f"layouts missing from registry: {set(Layout) - layouts}"


def test_build_html_template_minimal_omits_bar_background():
    """minimal layout has no bar background — it relies on underline only."""
    template = TEMPLATE_REGISTRY["minimal"]
    entity = _entity_dummy()
    html = build_html_template(entity, template, "slide", anim_duration_s=0.4)
    # The .bar class is not used by minimal layout; underline element is
    assert 'class="bar"' not in html
    assert "underline" in html
    # No background-color on the bar; the .underline element has the color
    assert ".underline" in html


def test_build_html_template_modern_includes_border_radius():
    """modern layout uses border-radius for the centered pill."""
    template = TEMPLATE_REGISTRY["modern"]
    entity = _entity_dummy()
    html = build_html_template(entity, template, "slide", anim_duration_s=0.4)
    assert "border-radius" in html
    # The pill width should be capped at pill_max_width
    assert f"width: {template.pill_max_width}px" in html


def test_build_html_template_corner_positions_bar_at_top():
    """corner layout positions the bar near the top, not the bottom."""
    template = TEMPLATE_REGISTRY["corner"]
    entity = _entity_dummy()
    html = build_html_template(entity, template, "slide", anim_duration_s=0.4)
    # .bar should have top: Ypx where Y is small (y_offset_from_top)
    # and should NOT use the bottom-anchored layout
    assert f"top: {template.y_offset_from_top}px" in html
    # And no `top: {bar_top_px}` where bar_top = height - y_offset - bar_h
    bar_bottom_top = template.height - template.y_offset_from_bottom - template.bar_height
    assert f"top: {bar_bottom_top}px" not in html


def test_build_html_template_newsroom_includes_three_lines():
    """newsroom layout splits role on ':' into role + org lines."""
    template = TEMPLATE_REGISTRY["newsroom"]
    entity = EntitySpan(
        name="Alice Smith", role="Mayor: City Council",
        start_time=0.0, end_time=2.0, srt_indices=[1],
    )
    html = build_html_template(entity, template, "slide", anim_duration_s=0.4)
    assert "Alice Smith" in html
    assert "Mayor" in html
    assert "City Council" in html
    assert "frame" in html


def _entity_dummy():
    return EntitySpan(name="Alice", role="Mayor", start_time=0.0, end_time=2.0, srt_indices=[1])


def test_classic_template_is_byte_identical_to_default():
    """classic is the unspecialized default — must not have any field overridden."""
    classic = TEMPLATE_REGISTRY["classic"]
    assert classic.layout == Layout.BAR
    assert classic.bar_height == 180
    assert classic.accent_color == (255, 200, 40)


def test_template_field_overrides_apply_after_registry_lookup():
    """When main() applies --resolution WxH after registry lookup, width/height change."""
    template = TEMPLATE_REGISTRY["classic"]
    template.width = 1280
    template.height = 720
    assert template.width == 1280
    assert template.height == 720
    # Other fields remain at classic defaults
    assert template.bar_height == 180
    assert template.accent_color == (255, 200, 40)


def test_render_lower_third_png_modern_no_shadow_when_disabled(tmp_path: Path):
    """Modern template by default has no shadow; rendering must still produce valid PNG."""
    template = TEMPLATE_REGISTRY["modern"]
    assert template.shadow is False
    entity = _entity_dummy()
    out = render_lower_third_png(entity, tmp_path / "lt_modern.png", template)
    assert out.exists()
    with Image.open(out) as img:
        assert img.mode == "RGBA"
        assert img.size == (1920, 1080)


def test_render_lower_third_png_boxed_with_shadow(tmp_path: Path):
    """boxed template has shadow enabled and rounded corners."""
    template = TEMPLATE_REGISTRY["boxed"]
    assert template.shadow is True
    entity = _entity_dummy()
    out = render_lower_third_png(entity, tmp_path / "lt_boxed.png", template)
    assert out.exists()
    with Image.open(out) as img:
        assert img.mode == "RGBA"
        assert img.size == (1920, 1080)


# ─────────────────────────────────────────────────────────────────────────────
# load_chapter_entities
# ─────────────────────────────────────────────────────────────────────────────

def _write_checkpoint(project_dir: Path, chapters: list, *, gzip_compress: bool = False):
    """Helper: drop a fake checkpoint.json with chapter_data inside project_dir."""
    payload = {
        "chapter_data": {
            "chapters": chapters,
            "listicle_groups": [],
        }
    }
    raw = json.dumps(payload).encode("utf-8")
    out = project_dir / "checkpoint.json"
    if gzip_compress:
        import gzip as _gz
        raw = _gz.compress(raw)
    out.write_bytes(raw)


def _ch(idx: int, title: str, *, start_idx: int, end_idx: int | None = None,
        location_name: str = "", topics: list | None = None) -> dict:
    """Helper: build a minimal ChapterCandidate dict."""
    return {
        "chapter_id": idx,
        "title": title,
        "start_segment_idx": start_idx,
        "end_segment_idx": end_idx or start_idx,
        "location_name": location_name,
        "topics": topics or [],
    }


def test_load_chapter_entities_from_checkpoint(tmp_path: Path):
    """Loads ChapterCandidate dicts from checkpoint.json into EntitySpan(type=CHAPTER)."""
    segs = _segments(n=30)
    _write_checkpoint(tmp_path, [
        _ch(0, "Opening",         start_idx=1,  end_idx=8),
        _ch(1, "Farmland history", start_idx=9,  end_idx=22, location_name="Burian Lake"),
    ])
    chapters = load_chapter_entities(tmp_path, segs)
    assert len(chapters) == 2
    assert chapters[0].entity_type == EntityType.CHAPTER
    assert chapters[0].name == "Opening"
    # First two segments of the chapter → pinned to segs[0..1].
    assert chapters[0].start_time == segs[0].start_time
    assert chapters[0].end_time == segs[1].end_time
    assert chapters[0].srt_indices == [1, 2]
    # Second chapter has location_name → surfaces as role.
    assert chapters[1].role == "Burian Lake"
    assert chapters[1].srt_indices == [9, 10]


def test_load_chapter_entities_from_gzipped_checkpoint(tmp_path: Path):
    """Real checkpoints are gzip-compressed — must work without explicit decode."""
    segs = _segments(n=10)
    _write_checkpoint(tmp_path, [_ch(0, "Only Chapter", start_idx=1)], gzip_compress=True)
    chapters = load_chapter_entities(tmp_path, segs)
    assert len(chapters) == 1
    assert chapters[0].name == "Only Chapter"


def test_load_chapter_entities_no_checkpoint_returns_empty(tmp_path: Path):
    """Missing checkpoint.json → no chapters, no failure."""
    chapters = load_chapter_entities(tmp_path, _segments(n=5))
    assert chapters == []


def test_load_chapter_entities_empty_chapter_data_returns_empty(tmp_path: Path):
    """Checkpoint exists but chapter_data is empty → no chapters."""
    (tmp_path / "checkpoint.json").write_text(json.dumps({"chapter_data": {}}))
    assert load_chapter_entities(tmp_path, _segments(n=5)) == []


def test_load_chapter_entities_drops_invalid_entries(tmp_path: Path):
    """Chapters missing title or with bad start_idx are skipped, not failed."""
    segs = _segments(n=20)
    _write_checkpoint(tmp_path, [
        _ch(0, "OK chapter", start_idx=5),
        {"chapter_id": 1, "title": "",   "start_segment_idx": 6},   # empty title
        {"chapter_id": 2, "title": "X",  "start_segment_idx": 0},   # invalid idx
        {"chapter_id": 3, "title": "Y",  "start_segment_idx": 99},  # out of range
    ])
    chapters = load_chapter_entities(tmp_path, segs)
    assert len(chapters) == 1
    assert chapters[0].name == "OK chapter"


def test_load_chapter_entities_uses_topics_as_role_fallback(tmp_path: Path):
    """When location_name is empty, the first few topics become the role."""
    segs = _segments(n=10)
    _write_checkpoint(tmp_path, [
        _ch(0, "Restoration", start_idx=1, topics=["labor", "replanting", "patience"]),
    ])
    chapters = load_chapter_entities(tmp_path, segs)
    assert chapters[0].role == "labor, replanting, patience"


def test_build_lower_thirds_otio_includes_chapter_track(tmp_path: Path):
    """Chapter entity lands on V5 - Lower Thirds · Chapters."""
    import opentimelineio as otio

    entities = [
        EntitySpan(name="Section A", role="", start_time=0.0, end_time=2.0,
                   srt_indices=[1], entity_type=EntityType.CHAPTER),
    ]
    pngs = [_png(tmp_path, "chap0.png")]
    out_otio = tmp_path / "out.otio"

    timeline = build_lower_thirds_otio(entities, pngs, ["classic"], out_otio)

    assert len(timeline.tracks) == 1
    assert timeline.tracks[0].name == "V5 - Lower Thirds · Chapters"
    clips = [c for c in timeline.tracks[0] if isinstance(c, otio.schema.Clip)]
    assert len(clips) == 1
    assert clips[0].metadata["entity"]["entity_type"] == "chapter"
    assert clips[0].metadata["entity"]["template"] == "classic"


def test_parse_template_by_type_chapter_alias_resolves():
    """`section=classic` → CHAPTER via the alias map."""
    out = parse_template_by_type("section=classic,chapter=newsroom")
    assert out == {
        EntityType.CHAPTER: "newsroom",
    }


# ─────────────────────────────────────────────────────────────────────────────
# extract_candidates (Python pre-screening)
#
# These patterns run BEFORE Ollama so the LLM only classifies/filters; timing
# is owned by Python and the LLM never invents SRT indices.
# ─────────────────────────────────────────────────────────────────────────────


def test_extract_candidates_finds_year():
    segs = _segments(["The trust was founded in 1946.", "It continues today."])
    cands = extract_candidates(segs)
    years = [c for c in cands if c.hint == "date"]
    assert any(c.text == "1946" and c.srt_index == 1 for c in years)


def test_extract_candidates_finds_percentage():
    segs = _segments(["99% of homes use this.", "Other segment."])
    cands = extract_candidates(segs)
    infos = [c for c in cands if c.hint == "info" and "%" in c.text]
    assert any(c.text.replace(" ", "") == "99%" for c in infos)


def test_extract_candidates_finds_currency():
    segs = _segments(["Cost was $50,000.", "Later $5 million more."])
    cands = extract_candidates(segs)
    infos = [c for c in cands if c.hint == "info" and "$" in c.text]
    texts = {c.text.replace(" ", "") for c in infos}
    assert "$50,000" in texts
    assert "$5million" in texts or "$5 million" in texts


def test_extract_candidates_finds_person_with_title():
    segs = _segments(["Dr. Alice Smith led the team.", "She won awards."])
    cands = extract_candidates(segs)
    people = [c for c in cands if c.hint == "person"]
    assert any("Alice Smith" in c.text for c in people)


def test_extract_candidates_finds_place_after_preposition():
    segs = _segments(["We traveled to Paris last year.", "It was great."])
    cands = extract_candidates(segs)
    places = [c for c in cands if c.hint == "place"]
    assert any(c.text == "Paris" for c in places)


def test_extract_candidates_finds_place_suffix():
    segs = _segments(["We visited Gloucestershire today.", "Lovely county."])
    cands = extract_candidates(segs)
    places = [c for c in cands if c.hint == "place"]
    assert any(c.text == "Gloucestershire" for c in places)


def test_extract_candidates_dedupes_same_text_in_same_segment():
    segs = _segments(["We visited New York, New York was great."])
    cands = extract_candidates(segs)
    ny = [c for c in cands if c.text == "New York"]
    assert len(ny) == 1


def test_extract_candidates_returns_sorted_by_srt_index():
    segs = _segments([
        "First segment here.",
        "Then Dr. Bob Jones arrived.",
        "Finally in 1999.",
    ])
    cands = extract_candidates(segs)
    indices = [c.srt_index for c in cands]
    assert indices == sorted(indices)


def test_extract_candidates_empty_segments_returns_empty():
    cands = extract_candidates(_segments(n=0))
    assert cands == []


def test_extract_candidates_context_window_populated():
    segs = _segments(["The trust was founded in 1946 when the family arrived."])
    cands = extract_candidates(segs)
    dates = [c for c in cands if c.hint == "date"]
    assert dates, "expected a date candidate"
    assert "1946" in dates[0].context
    assert "trust" in dates[0].context


# ─────────────────────────────────────────────────────────────────────────────
# parse_candidate_response
#
# The model returns [{idx, include, entity_type, role}, ...]. Each confirmed
# candidate inherits timing from Python's pre-computed srt_index — the LLM
# cannot influence timing.
# ─────────────────────────────────────────────────────────────────────────────


def _candidate(text: str, srt_index: int, hint: str = "person") -> Candidate:
    return Candidate(text=text, srt_index=srt_index, hint=hint, context=text)


def test_parse_candidate_response_basic_include():
    segs = _segments([
        "First segment",
        "Dr. Alice Smith led the team",
        "Third segment",
    ])
    cands = [_candidate("Alice Smith", 2, "person")]
    raw = '[{"idx": 1, "include": true, "entity_type": "person", "role": "Mayor"}]'
    out = parse_candidate_response(raw, cands, segs)
    assert len(out) == 1
    assert out[0].name == "Alice Smith"
    assert out[0].role == "Mayor"
    assert out[0].entity_type == EntityType.PERSON
    assert out[0].srt_indices == [2]
    assert out[0].start_time == 2.0
    assert out[0].end_time == 4.0


def test_parse_candidate_response_excludes_when_include_false():
    segs = _segments(["Alice says hi", "Bob arrives"])
    cands = [_candidate("Alice", 1, "person"), _candidate("Bob", 2, "person")]
    raw = '[{"idx": 1, "include": false}, {"idx": 2, "include": true, "entity_type": "person", "role": ""}]'
    out = parse_candidate_response(raw, cands, segs)
    assert len(out) == 1
    assert out[0].name == "Bob"


def test_parse_candidate_response_falls_back_to_hint_when_type_missing():
    """When the model omits entity_type, use the candidate's hint."""
    segs = _segments(["Dr. Carol Chen arrived"])
    cands = [_candidate("Carol Chen", 1, "person")]
    raw = '[{"idx": 1, "include": true, "role": "Doctor"}]'  # no entity_type
    out = parse_candidate_response(raw, cands, segs)
    assert len(out) == 1
    assert out[0].entity_type == EntityType.PERSON


def test_parse_candidate_response_drops_unknown_entity_type():
    segs = _segments(["Some segment"])
    cands = [_candidate("Foo", 1, "person")]
    raw = '[{"idx": 1, "include": true, "entity_type": "alien", "role": ""}]'
    out = parse_candidate_response(raw, cands, segs)
    assert out == []


def test_parse_candidate_response_skips_unknown_idx():
    segs = _segments(["Alice", "Bob"])
    cands = [_candidate("Alice", 1, "person")]
    raw = '[{"idx": 99, "include": true, "entity_type": "person", "role": ""}]'
    out = parse_candidate_response(raw, cands, segs)
    assert out == []


def test_parse_candidate_response_uses_string_include():
    """Some models return include as "true"/"false" strings — accept them."""
    segs = _segments(["Alice says hi"])
    cands = [_candidate("Alice", 1, "person")]
    raw = '[{"idx": 1, "include": "yes", "entity_type": "person", "role": ""}]'
    out = parse_candidate_response(raw, cands, segs)
    assert len(out) == 1
    assert out[0].name == "Alice"


def test_parse_candidate_response_timing_owned_by_python():
    """Python owns srt_index; the LLM only chooses include/type/role.

    When the model provides a parseable entity_type, we trust it (the model
    can correct Python's hint). When the model provides an unparseable type,
    the entry is dropped. When the model omits type entirely, Python's hint
    is used.
    """
    segs = _segments([
        "First",
        "We met in 1946 at the trust.",
        "Third",
    ])
    cands = [_candidate("1946", 2, "date")]

    # Case 1: model returns an unparseable entity_type → drop
    raw = '[{"idx": 1, "include": true, "entity_type": "alien", "role": ""}]'
    out = parse_candidate_response(raw, cands, segs)
    assert out == []

    # Case 2: model omits entity_type → Python's hint wins
    raw = '[{"idx": 1, "include": true, "role": "Trust founded"}]'
    out = parse_candidate_response(raw, cands, segs)
    assert len(out) == 1
    assert out[0].entity_type == EntityType.DATE
    assert out[0].srt_indices == [2]
    assert out[0].start_time == 2.0
    assert out[0].end_time == 4.0

    # Case 3: model supplies a parseable entity_type → model wins
    raw = '[{"idx": 1, "include": true, "entity_type": "info", "role": "Trust founded"}]'
    out = parse_candidate_response(raw, cands, segs)
    assert len(out) == 1
    assert out[0].entity_type == EntityType.INFO
    # timing still comes from Python (srt_index=2)
    assert out[0].srt_indices == [2]


def test_parse_candidate_response_handles_unwrapping_string_objects():
    """Small models sometimes wrap object literals as strings in the array.

    The parser unwraps them transparently.
    """
    segs = _segments(["Alice says hi"])
    cands = [_candidate("Alice", 1, "person")]
    raw = (
        '["{\\"idx\\": 1, \\"include\\": true, '
        '\\"entity_type\\": \\"person\\", \\"role\\": \\"Mayor\\"}"]'
    )
    out = parse_candidate_response(raw, cands, segs)
    assert len(out) == 1
    assert out[0].name == "Alice"
    assert out[0].role == "Mayor"


def test_parse_candidate_response_concatenates_multiple_arrays():
    """Small models sometimes emit multiple JSON arrays instead of one."""
    segs = _segments(["Alice says hi", "Bob arrives"])
    cands = [
        _candidate("Alice", 1, "person"),
        _candidate("Bob", 2, "person"),
    ]
    raw = (
        '[{"idx": 1, "include": true, "entity_type": "person", "role": ""}] '
        '[{"idx": 2, "include": true, "entity_type": "person", "role": ""}]'
    )
    out = parse_candidate_response(raw, cands, segs)
    assert len(out) == 2
    assert {e.name for e in out} == {"Alice", "Bob"}


def test_parse_candidate_response_empty_response_returns_empty():
    segs = _segments(["Alice says hi"])
    cands = [_candidate("Alice", 1, "person")]
    assert parse_candidate_response("", cands, segs) == []
    assert parse_candidate_response("not json at all", cands, segs) == []


def test_parse_candidate_response_repair_flat_array_of_dicts():
    """When a small model (llama3.2) emits a single flat `[...]` with every
    candidate's keys/values inline and no `{}` wrappers, the repair should
    insert `{` after `[` and `}, {` at each `, "key":` boundary so the parser
    yields the same entities as a well-formed response.

    Regression: prior to the fix the repair only inserted one `{` and the
    parser returned `[]` (silently dropping all 8 candidates).
    """
    segs = _segments([
        "opening",
        "Alice Smith led the project",
        "middle",
        "Bob Jones arrived",
        "Carol Chen reviewed it",
        "more middle",
        "Dave Brown signed off",
        "even more middle",
    ])
    cands = [
        _candidate("Alice Smith", 2, "person"),
        _candidate("Bob Jones", 4, "person"),
        _candidate("Carol Chen", 5, "person"),
        _candidate("Dave Brown", 7, "person"),
    ]
    # Real-world malformed response from llama3.2 — no `{}` wrappers anywhere.
    raw = (
        '["idx": 1, "include": true, "entity_type": "person", "role": " ", '
        '"idx": 2, "include": false, "entity_type": " ", '
        '"idx": 3, "include": true, "entity_type": "person", "role": " ", '
        '"idx": 4, "include": true, "entity_type": "person", "role": " "]'
    )
    out = parse_candidate_response(raw, cands, segs)
    names = [e.name for e in out]
    # idx 1 (Alice) and idx 3 (Carol), idx 4 (Dave) include=True; idx 2 (Bob) excluded.
    assert "Alice Smith" in names
    assert "Carol Chen" in names
    assert "Dave Brown" in names
    assert "Bob Jones" not in names
    assert len(out) == 3


def test_parse_candidate_response_repair_truncated_flat_array():
    """Flat-array repair must also work when the model's response is truncated
    mid-stream (no closing `]`)."""
    segs = _segments(["Alice led", "Bob arrived", "Carol reviewed"])
    cands = [_candidate("Alice", 1, "person"), _candidate("Bob", 2, "person")]
    raw = (
        '["idx": 1, "include": true, "entity_type": "person", "role": " ", '
        '"idx": 2, "include": true, "entity_type": "person"'  # truncated
    )
    out = parse_candidate_response(raw, cands, segs)
    assert [e.name for e in out] == ["Alice", "Bob"]


def test_repair_flat_array_of_dicts_direct():
    """Direct unit test: `_repair_flat_array_of_dicts` must convert the
    flat-array pattern into a valid JSON array of objects."""
    raw = (
        '["idx": 1, "include": true, "entity_type": "person", "role": " ", '
        '"idx": 2, "include": false, "entity_type": " ", '
        '"idx": 3, "include": true, "entity_type": "date", "role": "1946"]'
    )
    repaired = _repair_missing_open_brace(raw)
    parsed = json.loads(repaired)
    assert isinstance(parsed, list)
    assert len(parsed) == 3
    assert parsed[0] == {"idx": 1, "include": True, "entity_type": "person", "role": " "}
    assert parsed[1]["include"] is False
    assert parsed[2]["entity_type"] == "date"
    assert parsed[2]["role"] == "1946"


def test_repair_flat_array_of_dicts_returns_unchanged_when_irrelevant():
    """No leading `[` or already has `}` → returned unchanged (let other paths
    handle it)."""
    assert _repair_missing_open_brace('{"idx": 1}') == '{"idx": 1}'
    assert _repair_missing_open_brace("not json at all") == "not json at all"
    assert _repair_missing_open_brace("") == ""


def test_parse_candidate_response_sorted_by_start_time():
    segs = _segments([
        "First segment",
        "Bob arrives",
        "Middle",
        "Alice enters",
    ])
    cands = [
        _candidate("Bob", 2, "person"),
        _candidate("Alice", 4, "person"),
    ]
    raw = '[{"idx": 1, "include": true, "entity_type": "person", "role": ""}, {"idx": 2, "include": true, "entity_type": "person", "role": ""}]'
    out = parse_candidate_response(raw, cands, segs)
    assert [e.name for e in out] == ["Bob", "Alice"]  # start_time order


# ──────────────────────────────────────────────────────────────────
# audit_first_minute
# ──────────────────────────────────────────────────────────────────


def _audit_seg(idx: int, start: float, end: float, text: str) -> SRTSegment:
    return SRTSegment(index=idx, start_time=start, end_time=end, text=text)


def _audit_span(name: str, start: float, end: float, srt_indices: list,
                entity_type: EntityType = EntityType.PERSON) -> EntitySpan:
    return EntitySpan(
        name=name, role="", start_time=start, end_time=end,
        srt_indices=list(srt_indices), entity_type=entity_type,
    )


def test_audit_first_minute_flags_misplaced_first_occurrence():
    """2026 is mentioned in SRT 1 (0.00s) but emitted at SRT 167 (513s)."""
    segs = [
        _audit_seg(1, 0.0, 5.0, "On August 19th, 2026, something important happened in Zagreb."),
        _audit_seg(167, 513.0, 518.0, "Honi .ai's commercial service began in April, 2026."),
    ]
    ents = [_audit_span("2026", 513.0, 517.0, [167], entity_type=EntityType.DATE)]
    report = audit_first_minute(ents, segs, print_report=False)
    assert report["coverage_score"] < 1.0
    misplaced = {m["name"] for m in report["first_occurrence_misplaced"]}
    assert "2026" in misplaced
    rec = next(m for m in report["first_occurrence_misplaced"] if m["name"] == "2026")
    assert rec["first_srt_idx"] == 1
    assert rec["emitted_at_srt_idx"] == 167


def test_audit_first_minute_flags_missing_news_hook():
    """Date 2026 mentioned in opener but never extracted as an entity."""
    segs = [
        _audit_seg(1, 0.0, 5.0, "On August 19th, 2026, something important happened in Zagreb."),
        _audit_seg(2, 5.0, 10.0, "Uber began offering autonomous rides using technology."),
    ]
    # Empty entities — both opener mentions are missing
    report = audit_first_minute([], segs, print_report=False)
    phrases = {h["phrase"] for h in report["missing_news_hook"]}
    assert "2026" in phrases


def test_audit_first_minute_flags_missing_place_in_window():
    """Place 'Zagreb' in opener not extracted → flagged as missing_news_hook."""
    segs = [
        _audit_seg(2, 3.0, 6.0, "happened in Zagreb, Croatia."),
    ]
    report = audit_first_minute([], segs, print_report=False)
    phrases = {h["phrase"] for h in report["missing_news_hook"]}
    assert "Zagreb" in phrases
    assert any(h["hint"] == "place" for h in report["missing_news_hook"])


def test_audit_first_minute_flags_missing_stats_with_space_comma():
    """Voiceover transcriptions produce '2 ,000' (space-comma). Audit should still detect it."""
    segs = [
        _audit_seg(6, 16.0, 21.0, "for more than 2 ,000 robotaxies across five European cities."),
    ]
    report = audit_first_minute([], segs, print_report=False)
    phrases = [s["phrase"] for s in report["missing_stats"]]
    assert any("robotaxies" in p for p in phrases), f"expected robotaxies in stats: {phrases}"
    # The detected phrase should still contain '2 ,000' (not stripped to '000')
    detected = next(p for p in phrases if "robotaxies" in p)
    assert "2 ,000 robotaxies" in detected


def test_audit_first_minute_anchored_in_window():
    """An entity starting before 60s shows up in anchored_in_window."""
    segs = [
        _audit_seg(10, 31.0, 37.0, "For years, autonomous taxis looked like a race."),
    ]
    ents = [_audit_span("Race", 31.5, 37.0, [10], entity_type=EntityType.CHAPTER)]
    report = audit_first_minute(ents, segs, print_report=False)
    assert len(report["anchored_in_window"]) == 1
    assert report["anchored_in_window"][0]["name"] == "Race"
    assert report["anchored_in_window"][0]["start_time"] == 31.5


def test_audit_first_minute_full_coverage_scores_high():
    """When all opener entities are anchored in window AND no missing hooks → score >= 0.85."""
    segs = [
        _audit_seg(1, 0.0, 5.0, "On August 19th, 2026, something important happened in Zagreb."),
        _audit_seg(2, 5.0, 10.0, "Uber began offering autonomous rides."),
    ]
    ents = [
        _audit_span("2026", 0.0, 5.0, [1], entity_type=EntityType.DATE),
        _audit_span("Zagreb", 3.0, 8.0, [1], entity_type=EntityType.PLACE),
        _audit_span("Uber", 5.0, 10.0, [2], entity_type=EntityType.INFO),
    ]
    report = audit_first_minute(ents, segs, print_report=False)
    assert report["coverage_score"] >= 0.85
    assert report["missing_news_hook"] == []
    assert report["first_occurrence_misplaced"] == []


def test_audit_first_minute_does_not_double_flag_present_entities():
    """An entity anchored in window must NOT also appear in missing_news_hook."""
    segs = [
        _audit_seg(1, 0.0, 5.0, "On August 19th, 2026, something important happened in Zagreb."),
    ]
    ents = [_audit_span("2026", 0.0, 5.0, [1], entity_type=EntityType.DATE)]
    report = audit_first_minute(ents, segs, print_report=False)
    hook_phrases = {h["phrase"] for h in report["missing_news_hook"]}
    assert "2026" not in hook_phrases


def test_audit_first_minute_score_weights_chapter_present():
    """A chapter starting in 0<start<60s boosts score."""
    segs = [
        _audit_seg(10, 31.0, 37.0, "For years, autonomous taxis looked like a race."),
        _audit_seg(20, 60.0, 65.0, "Chapter opens here."),
    ]
    # Without chapter
    ents_no_chapter = [_audit_span("Race", 31.5, 37.0, [10], entity_type=EntityType.INFO)]
    r1 = audit_first_minute(ents_no_chapter, segs, print_report=False)
    # With chapter at same time
    ents_chapter = [_audit_span("Race", 31.5, 37.0, [10], entity_type=EntityType.CHAPTER)]
    r2 = audit_first_minute(ents_chapter, segs, print_report=False)
    assert r2["coverage_score"] > r1["coverage_score"]


def test_audit_first_minute_custom_window_size():
    """opening_s argument changes the window boundary."""
    segs = [
        _audit_seg(1, 25.0, 30.0, "On August 19th, 2026, something important happened in Zagreb."),
        _audit_seg(20, 65.0, 70.0, "Another 2026 mention here."),
    ]
    ents = [_audit_span("2026", 65.0, 70.0, [20], entity_type=EntityType.DATE)]
    # With 60s window, the SRT 1 mention (25s) is in window — flagged as misplaced
    r60 = audit_first_minute(ents, segs, opening_s=60, print_report=False)
    misplaced_60 = {m["name"] for m in r60["first_occurrence_misplaced"]}
    assert "2026" in misplaced_60
    # With 20s window, SRT 1 (25s) is OUTSIDE window — nothing to flag
    r20 = audit_first_minute(ents, segs, opening_s=20, print_report=False)
    misplaced_20 = {m["name"] for m in r20["first_occurrence_misplaced"]}
    assert "2026" not in misplaced_20


def test_audit_first_minute_returns_recommendation():
    """Recommendation string is always non-empty and varies by score."""
    segs = [_audit_seg(1, 0.0, 5.0, "On August 19th, 2026, something important happened in Zagreb.")]
    # Empty coverage → recommendation warns about poor coverage
    r_poor = audit_first_minute([], segs, print_report=False)
    assert r_poor["recommendation"]
    assert "weak" in r_poor["recommendation"].lower() or "poor" in r_poor["recommendation"].lower()
    # Full coverage → "solid"
    full_ents = [
        _audit_span("2026", 0.0, 5.0, [1], entity_type=EntityType.DATE),
        _audit_span("Zagreb", 3.0, 5.0, [1], entity_type=EntityType.PLACE),
    ]
    r_good = audit_first_minute(full_ents, segs, print_report=False)
    assert "solid" in r_good["recommendation"].lower() or "ok" in r_good["recommendation"].lower()


# ──────────────────────────────────────────────────────────────────
# audit_first_minute — regression tests for the 4 bug fixes
# ──────────────────────────────────────────────────────────────────


def test_audit_first_minute_zagreb_not_suppressed_by_chapter_substring():
    """Bug #1: 'Zagreb' (a missing place) was previously suppressed because
    the chapter 'Zagreb's Role...' contained 'Zagreb' as a substring. The
    fix uses type-aware matching — only same-type entities count.
    """
    segs = [
        _audit_seg(2, 3.0, 6.0, "happened in Zagreb, Croatia."),
        _audit_seg(16, 52.0, 57.0, "Now those boundaries are starting to disappear."),
    ]
    ents = [
        # Chapter mentions Zagreb in its title, but it's a CHAPTER entity,
        # not a PLACE entity. So "Zagreb" as a missing PLACE should still
        # be flagged.
        _audit_span("Zagreb's Role in Autonomous Vehicle Development",
                    52.0, 57.0, [16], entity_type=EntityType.CHAPTER),
    ]
    r = audit_first_minute(ents, segs, print_report=False)
    hook_phrases = {h["phrase"] for h in r["missing_news_hook"]}
    assert "Zagreb" in hook_phrases


def test_audit_first_minute_america_not_misplaced_via_substring_match():
    """Bug #2: 'America' was flagged as misplaced because the loose-fallback
    substring regex matched 'American' in SRT 12. Word-boundary fix
    prevents this false positive.
    """
    segs = [
        _audit_seg(12, 37.0, 39.0, "between American technology companies"),
        _audit_seg(14, 43.0, 47.0, "Waymo built a large driverless service in the United States."),
    ]
    ents = [
        _audit_span("America", 847.0, 849.0, [279], entity_type=EntityType.PLACE),
    ]
    r = audit_first_minute(ents, segs, print_report=False)
    misplaced_names = {m["name"] for m in r["first_occurrence_misplaced"]}
    assert "America" not in misplaced_names


def test_audit_first_minute_catches_sentence_start_brands():
    """Bug #3: 'Uber' at SRT 3 start and 'Waymo' at SRT 14 start were missed
    because brand_re requires verb+name (verb comes AFTER for these cases).
    Sentence-start proper noun regex catches them.
    """
    segs = [
        _audit_seg(3, 5.0, 9.0, "Uber began offering autonomous rides using technology"),
        _audit_seg(14, 43.0, 47.0, "Waymo built a large driverless service in the United States."),
    ]
    r = audit_first_minute([], segs, print_report=False)
    hook_phrases = {h["phrase"] for h in r["missing_news_hook"]}
    assert "Uber" in hook_phrases
    assert "Waymo" in hook_phrases


def test_audit_first_minute_sentence_start_skips_stopwords():
    """Sentence-start 'Now', 'And', 'On' should NOT be flagged as missing
    news-hook entities even though they're capitalized."""
    segs = [
        _audit_seg(7, 21.0, 26.0, "And two days ago, deployment had grown."),
        _audit_seg(17, 55.0, 57.0, "Now those boundaries are starting to disappear."),
        _audit_seg(1, 0.0, 3.0, "On August 19th, something important."),
    ]
    r = audit_first_minute([], segs, print_report=False)
    hook_phrases = {h["phrase"] for h in r["missing_news_hook"]}
    # Stopwords should never appear
    for stop in ("And", "Now", "On"):
        assert stop not in hook_phrases, f"{stop!r} should be filtered as stopword"


def test_audit_first_minute_score_zero_for_empty_entities():
    """Bug #4: With entities=[], score previously could reach 0.55+ via
    vacuous-truth. Should now be 0.0.
    """
    segs = [
        _audit_seg(1, 0.0, 5.0, "On August 19th, 2026, something important happened in Zagreb."),
        _audit_seg(2, 5.0, 10.0, "Uber began offering autonomous rides."),
    ]
    r = audit_first_minute([], segs, print_report=False)
    assert r["coverage_score"] == 0.0


def test_audit_first_minute_token_sequence_fallback_for_pony_dot_ai():
    """Token-sequence fallback handles 'pony.ai' matching 'pony .ai' in SRT."""
    segs = [
        _audit_seg(4, 9.0, 12.0, "from pony .ai, a company founded in China."),
    ]
    ents = [
        _audit_span("pony.ai", 9.0, 12.0, [4], entity_type=EntityType.INFO),
    ]
    # Entity IS in window — not misplaced, but should also not be flagged
    # as missing (it has a lower-third).
    r = audit_first_minute(ents, segs, print_report=False)
    misplaced_names = {m["name"] for m in r["first_occurrence_misplaced"]}
    hook_phrases = {h["phrase"] for h in r["missing_news_hook"]}
    assert "pony.ai" not in misplaced_names  # it's in the window, anchored correctly
    assert "pony.ai" not in hook_phrases  # it's covered


def test_audit_first_minute_present_in_entities_type_aware():
    """_present_in_entities should only consider same-type entities when hint is given.

    Regression for bug #1: 'Zagreb' (place hint) should NOT be suppressed by
    chapter 'Zagreb's Role...' (chapter type).
    """
    segs = [_audit_seg(2, 3.0, 6.0, "happened in Zagreb, Croatia.")]
    # Chapter entity that contains "Zagreb" in name
    chapter_ent = _audit_span(
        "Zagreb's Role in Autonomous Vehicle Development",
        52.0, 57.0, [16], entity_type=EntityType.CHAPTER,
    )
    # Zagreb is NOT in any PLACE entity. Type-aware check should consider it missing.
    r = audit_first_minute([chapter_ent], segs, print_report=False)
    hook_phrases = {h["phrase"] for h in r["missing_news_hook"]}
    assert "Zagreb" in hook_phrases, f"expected Zagreb in missing_news_hook, got {hook_phrases}"


# ─────────────────────────────────────────────────────────────────────────────
# Regression tests for first-minute coverage fixes (2026-08-24)
#
# Bug fix verifications:
#   - dedup-by-text keeps earliest srt_index (so 2026 anchors at SRT 1,
#     not SRT 167 where the LLM picked the wrong-numbered candidate)
#   - brand extraction catches sentence-start "Uber began" and verb-following
#     "Apollo GO launched"
#   - big-number stat extraction catches "2 ,000 robotaxies"
#   - first-detection-wins on hint (Zagreb stays `place`, not demoted to
#     `person` by a later SENTENCE_START_RE match)
#   - sentence-start stopword filter excludes "And", "Now", etc.
# ─────────────────────────────────────────────────────────────────────────────


def test_extract_candidates_dedupes_by_text_keeps_earliest():
    """Two occurrences of "2026" (SRT 1 and SRT 167) → one candidate at SRT 1."""
    segs = [
        _seg(1, "On August 19th, 2026, something important"),
        _seg(167, "Honi .ai's commercial service there began in April 2026"),
    ]
    cands = extract_candidates(segs)
    by_text = {c.text: c for c in cands}
    assert "2026" in by_text
    assert by_text["2026"].srt_index == 1, (
        f"expected SRT 1 (earliest), got {by_text['2026'].srt_index}"
    )
    # Hint should be `date` (year regex), not `info` (big-number)
    assert by_text["2026"].hint == "date"


def test_extract_candidates_dedup_first_hint_wins():
    """First detection wins on hint — generic SENTENCE_START match must NOT
    demote a more-specific PLACE_PREP match to PERSON.
    """
    segs = [
        _seg(2, "happened in Zagreb, Croatia."),       # → place (prep)
        _seg(225, "Zagreb, London, or Singapore,"),    # → would match SENTENCE_START as person
    ]
    cands = extract_candidates(segs)
    by_text = {c.text: c for c in cands}
    assert "Zagreb" in by_text
    assert by_text["Zagreb"].hint == "place", (
        f"first-detection hint should win; got {by_text['Zagreb'].hint}"
    )
    assert by_text["Zagreb"].srt_index == 2


def test_extract_candidates_catches_sentence_start_brand():
    """Sentence-start single-word brand ("Uber began") is captured as person."""
    segs = [_seg(3, "Uber began offering autonomous rides using technology")]
    cands = extract_candidates(segs)
    by_text = {c.text: c for c in cands}
    assert "Uber" in by_text
    assert by_text["Uber"].hint == "person"
    assert by_text["Uber"].srt_index == 3


def test_extract_candidates_catches_verb_following_brand():
    """Multi-word brand + verb ("Apollo GO launched") is captured as person."""
    segs = [_seg(307, "Apollo GO launched a new service in Wuhan")]
    cands = extract_candidates(segs)
    by_text = {c.text: c for c in cands}
    assert "Apollo GO" in by_text
    assert by_text["Apollo GO"].hint == "person"


def test_extract_candidates_catches_big_number_stat():
    """Big-number stat ("2 ,000 robotaxies") is captured as info."""
    segs = [_seg(6, "for more than 2 ,000 robotaxies across five European cities.")]
    cands = extract_candidates(segs)
    by_text = {c.text: c for c in cands}
    assert "2 ,000 robotaxies" in by_text, (
        f"expected big-number stat, got texts: {list(by_text.keys())}"
    )
    assert by_text["2 ,000 robotaxies"].hint == "info"


def test_extract_candidates_skips_sentence_start_stopwords():
    """Common sentence-start words (And, Now) are filtered out."""
    segs = [
        _seg(10, "And then we saw the results."),
        _seg(11, "Now this is the part that matters."),
    ]
    cands = extract_candidates(segs)
    texts = {c.text.lower() for c in cands}
    assert "and" not in texts
    assert "now" not in texts
    # And the first content word ("then", "this") is also filtered or
    # not matched by _SENTENCE_START_RE (regex requires capitalized).
    assert "then" not in texts
    assert "this" not in texts


# ─────────────────────────────────────────────────────────────────────────────
# Word-onset anchoring: start = spoken word onset, end = seg.end_time
# ─────────────────────────────────────────────────────────────────────────────

def _w(word: str, start: float, end: float, confidence: float = 0.95) -> dict:
    """Helper: build a sidecar word dict."""
    return {"word": word, "start": start, "end": end, "confidence": confidence}


def test_normalize_text_strips_punctuation_and_collapses_whitespace():
    assert _normalize_text("Apollo GO") == "apollo go"
    assert _normalize_text("U.S.") == "u s"
    assert _normalize_text("  AI33  ") == "ai33"
    assert _normalize_text("On August 19th, 2026") == "on august 19th 2026"
    assert _normalize_text("") == ""


def test_word_onset_finds_single_token():
    """Entity "2026" anchors to the spoken word within seg 1."""
    seg_words = [
        _w("On", 0.0, 0.2),
        _w("August", 0.2, 0.5),
        _w("19th", 0.5, 0.8),
        _w("2026", 0.85, 1.2),  # "twenty twenty-six"
        _w("something", 1.3, 1.7),
        _w("important", 1.8, 2.4),
    ]
    onset = _word_onset("2026", seg_words, seg_start=0.0)
    assert onset == pytest.approx(0.85, abs=0.001)


def test_word_onset_finds_multi_token():
    """Multi-token entity "United States" anchors to the first token."""
    seg_words = [
        _w("introducing", 0.0, 0.3),
        _w("the", 0.3, 0.4),
        _w("United", 0.5, 0.9),
        _w("States", 0.9, 1.4),
        _w("policy", 1.5, 1.9),
    ]
    onset = _word_onset("United States", seg_words, seg_start=0.0)
    assert onset == pytest.approx(0.5, abs=0.001)


def test_word_onset_skips_low_confidence():
    """Low-confidence (hallucinated) words are skipped."""
    seg_words = [
        _w("On", 0.0, 0.2),
        _w("twenty", 0.3, 0.5, confidence=0.05),  # hallucination
        _w("twenty", 0.5, 0.8, confidence=0.05),  # hallucination
        _w("2026", 0.85, 1.2),                    # confident match
    ]
    onset = _word_onset("2026", seg_words, seg_start=0.0)
    assert onset == pytest.approx(0.85, abs=0.001)


def test_word_onset_punctuation_normalization_fallback():
    """Punctuation-fragmented entity falls back to first-token match."""
    seg_words = [
        _w("introducing", 0.0, 0.3),
        _w("u", 0.3, 0.4),
        _w("s", 0.4, 0.5),
        _w("policy", 0.6, 1.0),
    ]
    # Multi-token match for "u s" should find both tokens in order.
    onset = _word_onset("U.S.", seg_words, seg_start=0.0)
    assert onset == pytest.approx(0.3, abs=0.001)


def test_word_onset_returns_none_when_no_match():
    """Empty seg words or no confident match returns None."""
    assert _word_onset("2026", [], seg_start=0.0) is None
    assert _word_onset("", [_w("x", 0.0, 0.1)], seg_start=0.0) is None
    assert _word_onset("missing", [_w("foo", 0.0, 0.1), _w("bar", 0.1, 0.2)], seg_start=0.0) is None


def test_word_onset_handles_contraction_in_word():
    """Entity with apostrophe ("Baidu's") aligns to the source word's start.

    Regression: a single sidecar word " Baidu's" normalizes to ["baidu", "s"]
    (two tokens after punctuation strip) but the entity tokenizes the same
    way. The flat-tokenize algorithm must align them.
    """
    seg_words = [
        _w("In", 0.0, 0.2),
        _w("London,", 0.2, 0.5),
        _w("Baidu's", 1.08, 1.49, confidence=0.67),
        _w("Apollo", 1.49, 1.69, confidence=0.89),
        _w("Go", 1.69, 2.09, confidence=0.54),
        _w("is", 2.09, 2.41, confidence=0.98),
    ]
    onset = _word_onset("Baidu's Apollo Go", seg_words, seg_start=0.0)
    assert onset == pytest.approx(1.08, abs=0.001)


def test_clamp_extends_to_minimum_regardless_of_word_anchored():
    """The 2s minimum applies to every clip regardless of word-anchoring mode.

    Previously `word_anchored=True` was a no-op (clip kept its natural end
    even if < 2s), but the user wants every lower-third on screen for at
    least 2s. The word_anchored parameter is now back-compat only."""
    e = EntitySpan(
        name="2026",
        role="",
        start_time=0.85,  # word onset
        end_time=3.462,   # seg end — already 2.6s, no clamp needed
        srt_indices=[1],
        entity_type=EntityType.DATE,
    )
    out = clamp_entity_durations([e], duration_min=2.0, word_anchored=True)
    assert out[0].end_time == 3.462

    # Edge: very short entity (0.5s) — word_anchored flag is ignored;
    # the function extends to honour duration_min=2.0.
    short = EntitySpan(
        name="X", role="", start_time=1.0, end_time=1.5,
        srt_indices=[1], entity_type=EntityType.INFO,
    )
    out = clamp_entity_durations([short], duration_min=2.0, word_anchored=True)
    assert out[0].end_time == 3.0  # extended to 1.0 + 2.0


def test_clamp_zero_duration_guard():
    """Zero/negative-duration entity gets bumped to start + 0.05s."""
    e = EntitySpan(
        name="X", role="", start_time=1.0, end_time=1.0,  # zero duration
        srt_indices=[1], entity_type=EntityType.INFO,
    )
    out = clamp_entity_durations([e], duration_min=2.0)
    assert out[0].end_time == pytest.approx(1.05, abs=0.001)


def test_redistribute_stacked_entities_word_anchored_no_op():
    """When word-anchored, redistribute does NOT split shared-segment entities
    into sub-windows — each entity already has a unique word-onset start."""
    from lower_thirds import redistribute_stacked_entities
    # Two entities in seg 14 with distinct word-onset starts.
    waymo = EntitySpan(
        name="Waymo", role="", start_time=41.28, end_time=44.40,  # seg 14 end
        srt_indices=[14], entity_type=EntityType.PERSON,
    )
    united_states = EntitySpan(
        name="United States", role="", start_time=42.84, end_time=44.40,  # seg 14 end
        srt_indices=[14], entity_type=EntityType.PLACE,
    )
    out = redistribute_stacked_entities([waymo, united_states], word_anchored=True)
    # Word-anchored mode should preserve the exact start/end of each entity.
    assert out[0].start_time == 41.28
    assert out[0].end_time == 44.40
    assert out[1].start_time == 42.84
    assert out[1].end_time == 44.40


# ─────────────────────────────────────────────────────────────────────────────
# Global substring dedup: "United" dropped because "United States" exists
# ─────────────────────────────────────────────────────────────────────────────

def _ent(name: str, start: float = 0.0, end: float = 2.0) -> EntitySpan:
    return EntitySpan(
        name=name, role="", start_time=start, end_time=end,
        srt_indices=[1], entity_type=EntityType.PLACE,
    )


def test_dedupe_substring_drops_shorter():
    """'United' is dropped because 'United States' is a longer superset name."""
    entities = [
        _ent("United", start=10.0),
        _ent("United States", start=11.0),
        _ent("China", start=20.0),
    ]
    out = dedupe_substring_entities(entities)
    names = {e.name for e in out}
    assert names == {"United States", "China"}
    assert _ent("China", start=20.0) in out


def test_dedupe_substring_keeps_short_names():
    """3-char floor prevents masking 'AI' inside 'AI33' or 'UK' inside 'UK Ltd'."""
    entities = [
        _ent("AI", start=1.0),
        _ent("AI33", start=2.0),
        _ent("UK", start=3.0),
        _ent("UK Ltd", start=4.0),
    ]
    out = dedupe_substring_entities(entities)
    assert {e.name for e in out} == {"AI", "AI33", "UK", "UK Ltd"}


def test_dedupe_substring_case_and_article_insensitive():
    """Normalization: case-insensitive + leading-article strip + whitespace collapse."""
    entities = [
        _ent("United", start=1.0),
        _ent("the  united  states", start=2.0),  # normalized → "united states"
    ]
    out = dedupe_substring_entities(entities)
    assert len(out) == 1
    assert out[0].name == "the  united  states"


def test_dedupe_substring_no_change_when_no_superset():
    """If no entity is a strict superset of another, nothing is dropped."""
    entities = [
        _ent("China", start=1.0),
        _ent("Japan", start=2.0),
        _ent("Switzerland", start=3.0),
    ]
    out = dedupe_substring_entities(entities)
    assert len(out) == 3


def test_dedupe_substring_chapter_suppresses_substrings():
    """Regression: chapter titles must run through substring dedup.

    "Zagreb" is a substring of the chapter title "Zagreb's Role in
    Autonomous Vehicle Development"; "Autonomous" is also a substring.
    Pipeline order must call dedupe_substring_entities AFTER chapter
    entities are appended so the chapter title suppresses them.
    """
    place = EntitySpan(name="Zagreb", role="", start_time=3.87, end_time=4.89,
                       srt_indices=[2], entity_type=EntityType.PLACE)
    person = EntitySpan(name="Autonomous", role="", start_time=593.96, end_time=596.56,
                        srt_indices=[180], entity_type=EntityType.PERSON)
    chapter = EntitySpan(name="Zagreb's Role in Autonomous Vehicle Development",
                         role="", start_time=49.82, end_time=55.77,
                         srt_indices=[14, 15, 16], entity_type=EntityType.CHAPTER)
    entities = [place, person, chapter]
    out = dedupe_substring_entities(entities)
    names = {e.name for e in out}
    assert names == {"Zagreb's Role in Autonomous Vehicle Development"}
