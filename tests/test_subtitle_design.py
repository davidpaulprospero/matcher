"""
Unit tests for scripts/subtitle_design.py
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import pytest

# Make scripts/ importable
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import opentimelineio as otio

from scripts.subtitle_design import (  # noqa: E402
    Animation,
    Position,
    PRESETS,
    SRTSegment,
    SubtitleStyle,
    _ass_dialogue,
    _ass_header,
    _animation_overrides,
    _apply_overrides,
    _fmt_ass_time,
    _karaoke_for_segment,
    apply_preset,
    build_subtitle_otio,
    find_srt_path,
    hex_to_ass_color,
    load_word_timestamps,
    parse_args,
    render_transparent_video,
    split_long_cues,
    words_for_segment,
    write_ass,
    write_karaoke_ass,
    write_styled_srt,
)


# ============================================================
# Module-level mock fixtures (repo convention from test_otio_timeline.py)
# ============================================================

_TEXT_KIND = getattr(otio.schema.TrackKind, "Text", otio.schema.TrackKind.Video)


def _get_subtitle_tracks(timeline):
    """Find tracks marked as subtitle tracks (by name + metadata flag).

    Works across OTIO versions whether or not TrackKind.Text exists.
    """
    return [
        t for t in timeline.tracks
        if t.name == "V1 - Subtitles"
        or t.metadata.get("is_subtitle_track") is True
        or getattr(t, "kind", None) == _TEXT_KIND and t.name.startswith("V1 - ")
    ]


@dataclass
class MockWord:
    word: str
    start: float
    end: float
    confidence: float = 0.95


@dataclass
class MockWordSegment:
    start: float
    end: float
    text: str
    words: List[dict] = field(default_factory=list)


def _make_words_data() -> List[dict]:
    """Return a small WhisperClient-shaped word-timestamps dataset."""
    return [
        MockWordSegment(
            start=0.0, end=2.5, text="Hello world",
            words=[
                {"word": "Hello", "start": 0.0, "end": 0.8, "confidence": 0.99},
                {"word": "world", "start": 0.9, "end": 2.4, "confidence": 0.95},
            ],
        ).__dict__,
        MockWordSegment(
            start=3.0, end=5.0, text="Foo bar baz",
            words=[
                {"word": "Foo", "start": 3.0, "end": 3.5, "confidence": 0.95},
                {"word": "bar", "start": 3.6, "end": 4.1, "confidence": 0.97},
                {"word": "baz", "start": 4.2, "end": 4.9, "confidence": 0.92},
            ],
        ).__dict__,
    ]


def _make_segments() -> List[SRTSegment]:
    return [
        SRTSegment(index=1, start_time=0.0, end_time=2.5, text="Hello world", source_file="vo.srt"),
        SRTSegment(index=2, start_time=3.0, end_time=5.0, text="Foo bar baz", source_file="vo.srt"),
        SRTSegment(index=3, start_time=6.0, end_time=8.0,
                   text="This is a deliberately long segment that should be wrapped at the configured threshold",
                   source_file="vo.srt"),
    ]


# ============================================================
# find_srt_path
# ============================================================

@pytest.mark.fast
class TestFindSrtPath:
    def test_prefers_trimmed_variant(self, tmp_path: Path):
        vo_dir = tmp_path / "voiceover"
        vo_dir.mkdir()
        (vo_dir / "voiceover.srt").write_text("1\n00:00:00,000 --> 00:00:01,000\nhi\n\n")
        (vo_dir / "voiceover_trimmed.srt").write_text("1\n00:00:00,000 --> 00:00:01,000\nhi\n\n")
        result = find_srt_path(tmp_path)
        assert result is not None
        assert "trimmed" in result.name

    def test_falls_back_to_root(self, tmp_path: Path):
        (tmp_path / "voiceover.srt").write_text("1\n00:00:00,000 --> 00:00:01,000\nhi\n\n")
        result = find_srt_path(tmp_path)
        assert result is not None
        assert result.name == "voiceover.srt"

    def test_returns_none_when_missing(self, tmp_path: Path):
        assert find_srt_path(tmp_path) is None


# ============================================================
# Presets
# ============================================================

@pytest.mark.fast
class TestPresets:
    @pytest.mark.parametrize("preset_name", list(PRESETS.keys()))
    def test_known_preset_returns_fresh_copy(self, preset_name: str):
        s1 = apply_preset(preset_name)
        s2 = apply_preset(preset_name)
        assert isinstance(s1, SubtitleStyle)
        assert s1 == s2
        # Mutating s1 must not leak back to PRESETS.
        s1.font_size = 999
        s3 = apply_preset(preset_name)
        assert s3.font_size != 999

    def test_unknown_preset_raises(self):
        with pytest.raises(ValueError, match="Unknown style preset"):
            apply_preset("not_a_real_preset")

    def test_mrbeast_has_yellow_pop(self):
        s = apply_preset("mrbeast")
        assert s.primary_color == "&H0000FFFF"  # BGR yellow
        assert s.animation == Animation.POP

    def test_tiktok_has_karaoke(self):
        s = apply_preset("tiktok")
        assert s.animation == Animation.KARAOKE

    def test_hormozi_is_top_fade(self):
        s = apply_preset("hormozi")
        assert s.position == Position.TOP
        assert s.animation == Animation.FADE

    def test_three_word_karaoke_uses_window_size_3(self):
        """The user's default go-back-to style. Must:
          - animate KARAOKE (one Dialogue line per word from .words.json)
          - show a 3-word sliding window (default karaoke_window_size=3)
          - highlight the current word in the karaoke color, others in primary
        """
        s = apply_preset("three_word_karaoke")
        assert s.animation == Animation.KARAOKE
        assert s.karaoke_window_size == 3
        assert s.primary_color == "&H00FFFFFF"      # white
        assert s.karaoke_color == "&H0000FFFF"       # yellow
        assert s.position == Position.BOTTOM

    def test_davinci_caps_uses_thick_outline_not_background_box(self):
        """Sampled from 1caps_DAVINCI.mov: thick black outline around white text,
        first word in cotton-candy pink #F080C0, all-caps rendering. The
        'pill' look is the outline merging letters, NOT an opaque back-box."""
        s = apply_preset("davinci_caps")
        assert s.font_family == "Anton"
        assert s.primary_color == "&H00FFFFFF"     # white
        assert s.karaoke_color == "&H00C080F0"     # pink #F080C0 in BGR
        assert s.outline_color == "&H00000000"     # black
        assert s.outline_thickness >= 5.0          # thick outline (the pill effect)
        assert s.background_box is False           # NOT an opaque back-box
        assert s.force_uppercase is True
        assert s.highlight_position == "first"
        assert s.position == Position.BOTTOM

    def test_gold_on_black_preset(self):
        """Gold #FFD700 text on opaque black back-box; cue holds then fades
        out + drifts up at the end (Animation.FADE_UP)."""
        s = apply_preset("gold_on_black")
        assert s.primary_color == "&H0000D7FF"     # gold #FFD700 in BGR+alpha
        assert s.karaoke_color == "&H0000D7FF"
        assert s.background_box is True            # opaque back-box
        assert s.background_color == "&H00000000"  # opaque black
        assert s.animation == Animation.FADE_UP    # hold-then-fly-away
        assert s.anim_duration_ms > 0              # tail window length
        assert s.outline_thickness == 0.0          # box covers glyphs
        assert s.position == Position.BOTTOM
        assert s.highlight_position == "none"      # no per-word color overrides


# ============================================================
# Color + time helpers
# ============================================================

@pytest.mark.fast
class TestHelpers:
    def test_hex_to_ass_color_white(self):
        assert hex_to_ass_color("#FFFFFF") == "&H00FFFFFF"

    def test_hex_to_ass_color_yellow(self):
        # RGB(255,255,0) -> ASS BGR (0x00FFFF + alpha 0x00)
        assert hex_to_ass_color("#FFFF00") == "&H0000FFFF"

    def test_hex_to_ass_color_no_hash(self):
        assert hex_to_ass_color("FF0000") == "&H000000FF"

    def test_hex_to_ass_color_invalid_length(self):
        with pytest.raises(ValueError, match="6 chars"):
            hex_to_ass_color("#FFF")

    def test_fmt_ass_time_zero(self):
        assert _fmt_ass_time(0.0) == "0:00:00.00"

    def test_fmt_ass_time_minutes_seconds(self):
        assert _fmt_ass_time(72.34) == "0:01:12.34"

    def test_fmt_ass_time_hours(self):
        assert _fmt_ass_time(3725.5) == "1:02:05.50"

    def test_fmt_ass_time_clamped_non_negative(self):
        assert _fmt_ass_time(-1.0) == "0:00:00.00"


# ============================================================
# Long-line splitting
# ============================================================

@pytest.mark.fast
class TestSplitLongCues:
    def test_short_lines_unchanged(self):
        segs = [SRTSegment(index=1, start_time=0.0, end_time=1.0, text="Short text")]
        out = split_long_cues(segs, max_chars=42)
        assert len(out) == 1
        assert out[0].text == "Short text"

    def test_long_line_wrapped_at_word_boundary(self):
        segs = [SRTSegment(index=1, start_time=0.0, end_time=1.0,
                            text="alpha beta gamma delta epsilon zeta eta")]
        out = split_long_cues(segs, max_chars=20)
        assert "\\N" in out[0].text
        for line in out[0].text.split("\\N"):
            assert len(line) <= 20

    def test_long_single_word_not_split(self):
        segs = [SRTSegment(index=1, start_time=0.0, end_time=1.0,
                            text="supercalifragilisticexpialidocious")]
        out = split_long_cues(segs, max_chars=10)
        # No spaces, no wrap possible — text preserved.
        assert out[0].text == "supercalifragilisticexpialidocious"

    def test_max_chars_zero_disables(self):
        segs = [SRTSegment(index=1, start_time=0.0, end_time=1.0,
                            text="alpha beta gamma")]
        out = split_long_cues(segs, max_chars=0)
        assert out[0].text == "alpha beta gamma"


# ============================================================
# SRT writer
# ============================================================

@pytest.mark.fast
class TestWriteStyledSrt:
    def test_writes_parseable_srt(self, tmp_path: Path):
        segs = _make_segments()
        out = tmp_path / "subtitles.srt"
        write_styled_srt(segs, out)
        assert out.exists()
        text = out.read_text(encoding="utf-8")
        assert "Hello world" in text
        assert "00:00:00,000 --> 00:00:02,500" in text

    def test_creates_parent_dirs(self, tmp_path: Path):
        out = tmp_path / "nested" / "dir" / "subtitles.srt"
        write_styled_srt(_make_segments(), out)
        assert out.exists()


# ============================================================
# ASS writers
# ============================================================

@pytest.mark.fast
class TestWriteAss:
    def test_header_has_required_sections(self, tmp_path: Path):
        segs = _make_segments()[:1]
        style = apply_preset("capcut_default")
        out = tmp_path / "subtitles.ass"
        write_ass(segs, style, out)
        text = out.read_text(encoding="utf-8")
        assert "[Script Info]" in text
        assert "[V4+ Styles]" in text
        assert "[Events]" in text
        assert "Style: Default" in text
        assert "Dialogue: 0," in text

    def test_dialogue_count_matches_cues(self, tmp_path: Path):
        segs = _make_segments()
        style = apply_preset("capcut_default")
        out = tmp_path / "subtitles.ass"
        write_ass(segs, style, out)
        text = out.read_text(encoding="utf-8")
        dialogue_count = text.count("Dialogue:")
        assert dialogue_count == len(segs)

    def test_alignment_reflects_position(self, tmp_path: Path):
        segs = _make_segments()[:1]
        style = apply_preset("hormozi")  # TOP -> NUMPAD 8
        out = tmp_path / "subtitles.ass"
        write_ass(segs, style, out)
        text = out.read_text(encoding="utf-8")
        # NUMPAD 8 is the alignment value for top-center.
        style_line = [line for line in text.splitlines() if line.startswith("Style:")][0]
        fields = [f.strip() for f in style_line.split(",")]
        # Format has 23 fields; Alignment is field[18] (0-indexed).
        assert fields[18] == "8"

    def test_pop_animation_overrides_appear(self):
        style = apply_preset("mrbeast")  # POP
        overrides = _animation_overrides(style)
        assert "\\fscy110" in overrides
        assert "\\t(" in overrides

    def test_fade_animation_overrides_appear(self):
        style = apply_preset("hormozi")  # FADE
        overrides = _animation_overrides(style)
        assert "\\fad(" in overrides

    def test_no_animation_overrides_empty(self):
        style = apply_preset("capcut_default")  # NONE
        overrides = _animation_overrides(style)
        assert overrides == ""


@pytest.mark.fast
class TestKaraoke:
    def test_per_word_k_tags_land_on_each_word(self):
        seg = SRTSegment(index=1, start_time=0.0, end_time=2.5, text="Hello world")
        words = [
            {"word": "Hello", "start": 0.0, "end": 0.8, "confidence": 0.99},
            {"word": "world", "start": 0.9, "end": 2.4, "confidence": 0.95},
        ]
        records = _karaoke_for_segment(seg, words, apply_preset("tiktok"))
        # Sliding-window karaoke: one Dialogue line PER WORD, each showing a
        # window of `karaoke_window_size` words centred on the current word.
        # For a 2-word cue with window_size=3, the window is the whole cue.
        assert len(records) == len(words)
        # Current word is the karaoke (highlight) colour; non-current words
        # are primary (white). With tiktok, karaoke=yellow &H0000FFFF.
        hello_start, hello_end, hello_text = records[0]
        assert hello_start == 0.0 and hello_end == 0.8
        assert "Hello" in hello_text
        assert "world" in hello_text  # both words visible in this window
        # Current word "Hello" is wrapped in karaoke colour (yellow).
        assert f"{{\\c&H0000FFFF}}Hello" in hello_text
        # Other word "world" is wrapped in primary colour (white).
        assert f"{{\\c&H00FFFFFF}}world" in hello_text
        world_start, world_end, world_text = records[1]
        assert world_start == 0.9 and world_end == 2.4
        assert "world" in world_text
        # Now "world" is the current word — wrapped in karaoke colour.
        assert f"{{\\c&H0000FFFF}}world" in world_text
        # "Hello" is no longer current — wrapped in primary colour.
        assert f"{{\\c&H00FFFFFF}}Hello" in world_text
        # No {\k} tag (libass misrenders it).
        for _, _, txt in records:
            assert "\\k" not in txt

    def test_words_for_segment_filters_by_time(self):
        words = _make_words_data()
        seg = SRTSegment(index=1, start_time=0.0, end_time=2.5, text="Hello world")
        result = words_for_segment(words, seg)
        # Segment 1 contains "Hello" (0.0-0.8) and "world" (0.9-2.4).
        assert len(result) == 2
        assert result[0]["word"] == "Hello"
        assert result[1]["word"] == "world"

    def test_falls_back_to_plain_when_no_words(self):
        segs = _make_segments()[:1]
        out_path = Path(__file__).parent / "_tmp_fallback_test.ass"
        try:
            write_karaoke_ass(segs, words_data=[], style=apply_preset("tiktok"), path=out_path)
            text = out_path.read_text(encoding="utf-8")
            # When no usable word timestamps, emit ONE Dialogue line per segment
            # (the background) — no per-word overlay lines. The static
            # first-word highlight from highlight_position="first" is allowed
            # but no per-word karaoke overlays should appear.
            assert text.count("Dialogue:") == len(segs)
        finally:
            if out_path.exists():
                out_path.unlink()

    def test_karaoke_write_emits_k_tags_when_words_present(self, tmp_path: Path):
        segs = _make_segments()[:1]
        out = tmp_path / "k.ass"
        write_karaoke_ass(segs, _make_words_data(), apply_preset("tiktok"), out)
        text = out.read_text(encoding="utf-8")
        # Sliding-window karaoke: one Dialogue line per WORD. Segment 1
        # has 2 words → 2 Dialogue lines.
        assert text.count("Dialogue:") == 2
        assert "\\c&H" in text
        assert "Hello" in text

    def test_sliding_window_shows_karaoke_window_size_words(self):
        """A 7-word cue with batch_size=3 should produce 3 batches:
        batch 1 = words 1-3, batch 2 = words 4-6, batch 3 = word 7.
        Within each batch the entire batch stays visible while the karaoke
        (yellow) current word moves left-to-right."""
        from dataclasses import replace
        seg = SRTSegment(index=1, start_time=0.0, end_time=7.0,
                          text="ONE TWO THREE FOUR FIVE SIX SEVEN")
        words = [
            {"word": "ONE", "start": 0.0, "end": 1.0},
            {"word": "TWO", "start": 1.0, "end": 2.0},
            {"word": "THREE", "start": 2.0, "end": 3.0},
            {"word": "FOUR", "start": 3.0, "end": 4.0},
            {"word": "FIVE", "start": 4.0, "end": 5.0},
            {"word": "SIX", "start": 5.0, "end": 6.0},
            {"word": "SEVEN", "start": 6.0, "end": 7.0},
        ]
        style = replace(apply_preset("tiktok"), karaoke_window_size=3)
        records = _karaoke_for_segment(seg, words, style)
        assert len(records) == 7
        # Batch 1 (words 1-3): all three Dialogue lines show "ONE TWO THREE",
        # only the current word is wrapped in karaoke (yellow).
        for i in range(3):
            txt = records[i][2]
            assert "ONE" in txt
            assert "TWO" in txt
            assert "THREE" in txt
            assert "FOUR" not in txt
            assert "FIVE" not in txt
        # Batch 2 (words 4-6): lines 3-5 show "FOUR FIVE SIX".
        for i in range(3, 6):
            txt = records[i][2]
            assert "FOUR" in txt
            assert "FIVE" in txt
            assert "SIX" in txt
            assert "ONE" not in txt
            assert "THREE" not in txt
        # Batch 3 (word 7): line 6 shows just "SEVEN".
        txt = records[6][2]
        assert "SEVEN" in txt
        assert "ONE" not in txt
        # Within each line, the current word is karaoke (yellow) and the
        # others are primary (white). The last batch may have only 1 word
        # (no non-current words to be white).
        for i, (_, _, t) in enumerate(records):
            karaoke_count = t.count(f"{{\\c&H0000FFFF}}")
            primary_count = t.count(f"{{\\c&H00FFFFFF}}")
            assert karaoke_count >= 1, f"Line {i}: current word should be karaoke colour"
            assert primary_count >= 1, f"Line {i}: at least the reset tag should be present"

    def test_pause_breaks_batch_into_separate_windows(self):
        """A long pause between two words should split the batch boundary
        so the later word lives in its own batch, even with batch_size=3."""
        from dataclasses import replace
        seg = SRTSegment(index=1, start_time=0.0, end_time=10.0,
                          text="ALPHA BETA GAMMA DELTA")
        # 0.1s gap between ALPHA-BETA-GAMMA (continuous), then a 2.0s pause,
        # then DELTA on its own.
        words = [
            {"word": "ALPHA", "start": 0.0, "end": 1.0},
            {"word": "BETA",  "start": 1.1, "end": 2.0},
            {"word": "GAMMA", "start": 2.1, "end": 3.0},
            {"word": "DELTA", "start": 5.0, "end": 6.0},  # 2.0s after GAMMA
        ]
        style = replace(
            apply_preset("tiktok"),
            karaoke_window_size=3,
            karaoke_pause_threshold=0.5,
        )
        records = _karaoke_for_segment(seg, words, style)
        assert len(records) == 4
        # Lines 0-2 (ALPHA/BETA/GAMMA) all show the same 3-word batch
        for i in range(3):
            txt = records[i][2]
            assert "ALPHA" in txt
            assert "BETA" in txt
            assert "GAMMA" in txt
            assert "DELTA" not in txt
        # Line 3 (DELTA) sits alone — the pause forced a batch break.
        last_txt = records[3][2]
        assert "DELTA" in last_txt
        assert "ALPHA" not in last_txt
        assert "BETA" not in last_txt
        assert "GAMMA" not in last_txt

    def test_short_pause_does_not_break_batch(self):
        """A gap shorter than the pause threshold should NOT split a batch."""
        from dataclasses import replace
        seg = SRTSegment(index=1, start_time=0.0, end_time=6.0,
                          text="ONE TWO THREE FOUR")
        words = [
            {"word": "ONE",   "start": 0.0, "end": 1.0},
            {"word": "TWO",   "start": 1.1, "end": 2.0},  # 0.1s gap
            {"word": "THREE", "start": 2.1, "end": 3.0},  # 0.1s gap
            {"word": "FOUR",  "start": 3.1, "end": 4.0},  # 0.1s gap
        ]
        style = replace(
            apply_preset("tiktok"),
            karaoke_window_size=3,
            karaoke_pause_threshold=0.5,
        )
        records = _karaoke_for_segment(seg, words, style)
        # All 4 words fit in one batch (size=3 holds THREE, FOUR is the
        # single-word remainder — no pause break).
        for i in range(3):
            txt = records[i][2]
            assert "ONE" in txt and "TWO" in txt and "THREE" in txt
        last_txt = records[3][2]
        assert "FOUR" in last_txt


# ============================================================
# Word timestamps loader
# ============================================================

@pytest.mark.fast
class TestLoadWordTimestamps:
    def test_returns_data_when_sidecar_exists(self, tmp_path: Path):
        srt = tmp_path / "voiceover.srt"
        srt.write_text("1\n00:00:00,000 --> 00:00:01,000\nhi\n\n")
        words = tmp_path / "voiceover.words.json"
        words.write_text(json.dumps(_make_words_data()), encoding="utf-8")
        result = load_word_timestamps(srt)
        assert len(result) == 2
        assert result[0]["text"] == "Hello world"

    def test_returns_empty_when_sidecar_missing(self, tmp_path: Path):
        srt = tmp_path / "voiceover.srt"
        srt.write_text("1\n00:00:00,000 --> 00:00:01,000\nhi\n\n")
        result = load_word_timestamps(srt)
        assert result == []

    def test_returns_empty_when_json_malformed(self, tmp_path: Path):
        srt = tmp_path / "voiceover.srt"
        srt.write_text("1\n00:00:00,000 --> 00:00:01,000\nhi\n\n")
        words = tmp_path / "voiceover.words.json"
        words.write_text("not valid json", encoding="utf-8")
        result = load_word_timestamps(srt)
        assert result == []


# ============================================================
# OTIO Text track builder
# ============================================================

@pytest.mark.fast
class TestBuildSubtitleOtio:
    def test_clip_count_matches_cues(self, tmp_path: Path):
        ass_path = tmp_path / "subtitles.ass"
        ass_path.write_text("[Script Info]\n", encoding="utf-8")
        out = tmp_path / "subtitles.otio"
        segs = _make_segments()
        result_path = build_subtitle_otio(ass_path, segs, out)
        assert result_path == out
        assert out.exists()

        timeline = otio.adapters.read_from_file(str(out))
        sub_tracks = _get_subtitle_tracks(timeline)
        assert len(sub_tracks) == 1
        sub_track = sub_tracks[0]
        assert sub_track.name == "V1 - Subtitles"

        # Gaps may be inserted between cues; count Clip children only.
        clips = [item for item in sub_track if isinstance(item, otio.schema.Clip)]
        assert len(clips) == len(segs)

    def test_clip_source_range_matches_cue_duration(self, tmp_path: Path):
        ass_path = tmp_path / "subtitles.ass"
        ass_path.write_text("stub", encoding="utf-8")
        out = tmp_path / "subtitles.otio"
        segs = _make_segments()
        build_subtitle_otio(ass_path, segs, out)
        timeline = otio.adapters.read_from_file(str(out))
        sub_track = _get_subtitle_tracks(timeline)[0]
        clips = [item for item in sub_track if isinstance(item, otio.schema.Clip)]
        for clip, seg in zip(clips, segs):
            assert abs(clip.source_range.duration.value - (seg.end_time - seg.start_time) * 30.0) < 1

    def test_external_reference_uses_forward_slash_path(self, tmp_path: Path):
        ass_path = tmp_path / "subtitles.ass"
        ass_path.write_text("stub", encoding="utf-8")
        out = tmp_path / "subtitles.otio"
        segs = _make_segments()[:1]
        build_subtitle_otio(ass_path, segs, out)
        timeline = otio.adapters.read_from_file(str(out))
        sub_track = _get_subtitle_tracks(timeline)[0]
        clip = next(item for item in sub_track if isinstance(item, otio.schema.Clip))
        url = clip.media_reference.target_url
        assert "\\" not in url, f"DaVinci hangs on backslash paths: {url!r}"

    def test_empty_segments_produces_empty_track(self, tmp_path: Path):
        ass_path = tmp_path / "subtitles.ass"
        ass_path.write_text("stub", encoding="utf-8")
        out = tmp_path / "subtitles.otio"
        build_subtitle_otio(ass_path, [], out)
        timeline = otio.adapters.read_from_file(str(out))
        sub_tracks = _get_subtitle_tracks(timeline)
        assert len(sub_tracks) == 1
        assert len(sub_tracks[0]) == 0

    def test_track_marked_as_subtitle(self, tmp_path: Path):
        ass_path = tmp_path / "subtitles.ass"
        ass_path.write_text("stub", encoding="utf-8")
        out = tmp_path / "subtitles.otio"
        build_subtitle_otio(ass_path, _make_segments()[:1], out)
        timeline = otio.adapters.read_from_file(str(out))
        sub_track = _get_subtitle_tracks(timeline)[0]
        assert sub_track.metadata.get("is_subtitle_track") is True
        assert sub_track.metadata.get("Resolve_OTIO", {}).get("IsSubtitle") is True


# ============================================================
# CLI
# ============================================================

@pytest.mark.fast
class TestApplyOverrides:
    def test_animation_override(self):
        from dataclasses import replace
        base = apply_preset("capcut_default")  # animation=NONE
        args = parse_args(["--project", "/tmp/x", "--animation", "pop"])
        overridden = _apply_overrides(base, args)
        assert overridden.animation == Animation.POP

    def test_color_override_uses_hex_to_ass(self):
        base = apply_preset("capcut_default")
        args = parse_args(["--project", "/tmp/x", "--primary-color", "#FF8800"])
        overridden = _apply_overrides(base, args)
        assert overridden.primary_color == "&H000088FF"  # RGB(255,136,0) -> BGR(0,136,255)

    def test_position_override(self):
        base = apply_preset("capcut_default")
        args = parse_args(["--project", "/tmp/x", "--position", "top"])
        overridden = _apply_overrides(base, args)
        assert overridden.position == Position.TOP

    def test_no_overrides_returns_equivalent(self):
        base = apply_preset("capcut_default")
        args = parse_args(["--project", "/tmp/x"])
        overridden = _apply_overrides(base, args)
        assert overridden == base

    def test_background_box_flag_enables_box(self):
        """--background-box flips background_box on; --background-color overrides
        the box color (parsed as hex → ASS BGR)."""
        base = apply_preset("capcut_default")  # background_box=False
        args = parse_args(["--project", "/tmp/x", "--background-box"])
        overridden = _apply_overrides(base, args)
        assert overridden.background_box is True
        assert overridden.background_color == "&H00000000"  # default unchanged

        args = parse_args([
            "--project", "/tmp/x",
            "--background-box",
            "--background-color", "#000000",
        ])
        overridden = _apply_overrides(base, args)
        assert overridden.background_box is True
        assert overridden.background_color == "&H00000000"  # #000000 → opaque black


# ============================================================
# Random highlight color
# ============================================================

@pytest.mark.fast
class TestRandomHighlightColor:
    def test_palette_is_bright_only(self):
        """Every entry must be opaque and saturated (no dark/pastel hues)."""
        from scripts.subtitle_design import BRIGHT_HIGHLIGHT_COLORS
        for color in BRIGHT_HIGHLIGHT_COLORS:
            # Format &HAABBGGRR — strip &H + AA=00 prefix
            assert color.startswith("&H00"), f"{color} not fully opaque"
            bgr = color[4:]  # BGR hex string, 6 chars
            assert len(bgr) == 6, f"{color} wrong length"
            b = int(bgr[0:2], 16)
            g = int(bgr[2:4], 16)
            r = int(bgr[4:6], 16)
            # At least one channel >= 200 to guarantee visibility, and at
            # least one channel < 200 to avoid pure white/pastel greys.
            assert max(r, g, b) >= 200, f"{color} too dark (max={max(r,g,b)})"
            assert min(r, g, b) < 200, f"{color} too white/pastel"

    def test_picker_returns_palette_member(self):
        from scripts.subtitle_design import BRIGHT_HIGHLIGHT_COLORS, pick_random_highlight_color
        for _ in range(50):
            assert pick_random_highlight_color() in BRIGHT_HIGHLIGHT_COLORS

    def test_picker_accepts_deterministic_rng(self):
        import random as _random
        from scripts.subtitle_design import pick_random_highlight_color
        rng = _random.Random(42)
        a = pick_random_highlight_color(rng)
        rng = _random.Random(42)
        b = pick_random_highlight_color(rng)
        assert a == b  # same seed → same pick

    def test_main_skips_random_when_flag_set(self, tmp_path: Path, capsys):
        from scripts.subtitle_design import main
        vo_dir = tmp_path / "voiceover"
        vo_dir.mkdir()
        srt = vo_dir / "voiceover.srt"
        srt.write_text(
            "1\n00:00:00,000 --> 00:00:02,500\nHello world\n\n",
            encoding="utf-8",
        )
        rc = main([
            "--project", str(tmp_path),
            "--style", "davinci_caps",
            "--no-random-highlight",
            "--dry-run",
        ])
        assert rc == 0
        out = capsys.readouterr().out
        assert "Highlight color (random)" not in out

    def test_main_applies_explicit_karaoke_color(self, tmp_path: Path, capsys):
        from scripts.subtitle_design import main
        vo_dir = tmp_path / "voiceover"
        vo_dir.mkdir()
        srt = vo_dir / "voiceover.srt"
        srt.write_text(
            "1\n00:00:00,000 --> 00:00:02,500\nHello world\n\n",
            encoding="utf-8",
        )
        rc = main([
            "--project", str(tmp_path),
            "--style", "davinci_caps",
            "--karaoke-color", "#00FF00",
            "--dry-run",
        ])
        assert rc == 0
        # The explicit override should win; no random pick logged.
        out = capsys.readouterr().out
        assert "Highlight color (random)" not in out

    def test_main_skips_random_for_none_animation(self, tmp_path: Path, capsys):
        """For Animation.NONE (gold_on_black), the random highlight picker is
        meaningless — there are no per-word colors to override. Verify the
        picker is skipped so the ASS output keeps the gold primary color
        across the whole cue (no {\c...} override on the first word)."""
        from scripts.subtitle_design import main
        vo_dir = tmp_path / "voiceover"
        vo_dir.mkdir()
        srt = vo_dir / "voiceover.srt"
        srt.write_text(
            "1\n00:00:00,000 --> 00:00:02,500\nHello world\n\n",
            encoding="utf-8",
        )
        rc = main([
            "--project", str(tmp_path),
            "--style", "gold_on_black",
            "--dry-run",
        ])
        assert rc == 0
        out = capsys.readouterr().out
        # No random pick should be logged for NONE animation.
        assert "Highlight color (random)" not in out


# ============================================================
# End-to-end smoke (CLI invocation)
# ============================================================

@pytest.mark.fast
class TestMainEndToEnd:
    def test_dry_run_writes_nothing(self, tmp_path: Path, capsys):
        from scripts.subtitle_design import main

        vo_dir = tmp_path / "voiceover"
        vo_dir.mkdir()
        srt = vo_dir / "voiceover.srt"
        srt.write_text(
            "1\n00:00:00,000 --> 00:00:02,500\nHello world\n\n"
            "2\n00:00:03,000 --> 00:00:05,000\nFoo bar baz\n\n",
            encoding="utf-8",
        )

        rc = main([
            "--project", str(tmp_path),
            "--style", "mrbeast",
            "--animation", "pop",
            "--dry-run",
        ])
        assert rc == 0
        assert not (tmp_path / "subtitle_design").exists()

    def test_full_run_writes_all_formats(self, tmp_path: Path):
        from scripts.subtitle_design import main

        vo_dir = tmp_path / "voiceover"
        vo_dir.mkdir()
        srt = vo_dir / "voiceover.srt"
        srt.write_text(
            "1\n00:00:00,000 --> 00:00:02,500\nHello world\n\n"
            "2\n00:00:03,000 --> 00:00:05,000\nFoo bar baz\n\n",
            encoding="utf-8",
        )

        rc = main([
            "--project", str(tmp_path),
            "--style", "tiktok",
            "--animation", "karaoke",
        ])
        assert rc == 0
        out_dir = tmp_path / "subtitle_design"
        assert (out_dir / "subtitles.srt").exists()
        assert (out_dir / "subtitles.ass").exists()
        assert (out_dir / "subtitles.otio").exists()

        # SRT is parseable
        srt_text = (out_dir / "subtitles.srt").read_text(encoding="utf-8")
        assert "Hello world" in srt_text

        # ASS is valid libass
        ass_text = (out_dir / "subtitles.ass").read_text(encoding="utf-8")
        assert "[Script Info]" in ass_text
        assert "[V4+ Styles]" in ass_text
        assert "[Events]" in ass_text
        assert "Dialogue:" in ass_text

        # OTIO loads with a subtitle track
        timeline = otio.adapters.read_from_file(str(out_dir / "subtitles.otio"))
        sub_tracks = _get_subtitle_tracks(timeline)
        assert len(sub_tracks) == 1

    def test_full_run_with_karaoke_writes_k_tags(self, tmp_path: Path):
        from scripts.subtitle_design import main

        vo_dir = tmp_path / "voiceover"
        vo_dir.mkdir()
        srt = vo_dir / "voiceover.srt"
        srt.write_text(
            "1\n00:00:00,000 --> 00:00:02,500\nHello world\n\n",
            encoding="utf-8",
        )
        words = vo_dir / "voiceover.words.json"
        words.write_text(json.dumps(_make_words_data()), encoding="utf-8")

        rc = main([
            "--project", str(tmp_path),
            "--style", "tiktok",
            "--animation", "karaoke",
        ])
        assert rc == 0
        ass_text = (tmp_path / "subtitle_design" / "subtitles.ass").read_text(encoding="utf-8")
        # Sliding-window karaoke: one Dialogue line per WORD, each showing
        # a window of words with the current word in pink. 2 words → 2 lines.
        assert "\\c&H" in ass_text
        assert ass_text.count("Dialogue:") == 2


# ============================================================
# Transparent video render
# ============================================================

@pytest.mark.fast
class TestRenderTransparentVideo:
    """Integration tests for ffmpeg-driven alpha-channel render.

    These call ffmpeg directly, so they are marked `@pytest.mark.fast` only if
    ffmpeg is on PATH. The test is skipped automatically when ffmpeg is missing.
    """

    @pytest.fixture(autouse=True)
    def _require_ffmpeg(self):
        import shutil
        if shutil.which("ffmpeg") is None:
            pytest.skip("ffmpeg not on PATH")

    def _write_minimal_ass(self, path: Path) -> None:
        """Trivial 2-second ASS with one visible cue."""
        path.write_text(
            "[Script Info]\n"
            "ScriptType: v4.00+\n"
            "PlayResX: 320\nPlayResY: 180\n"
            "\n[V4+ Styles]\n"
            "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
            "Style: Default,Arial,40,&H00FFFFFF,&H0000FFFF,&H00000000,&H00000000,-1,0,0,0,100,100,0,0,1,2.0,0,2,10,10,10,1\n"
            "\n[Events]\n"
            "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
            "Dialogue: 0,0:00:00.10,0:00:01.90,Default,,0,0,0,,HELLO\n",
            encoding="utf-8",
        )

    def test_writes_alpha_pix_fmt_mov(self, tmp_path: Path):
        import subprocess
        ass = tmp_path / "sub.ass"
        self._write_minimal_ass(ass)
        out = tmp_path / "sub.mov"
        render_transparent_video(
            ass_path=ass, output_path=out, duration_seconds=2.0,
            width=320, height=180, fps=25,
        )
        assert out.exists()
        assert out.stat().st_size > 0

        # Probe pixel format via ffprobe.
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=pix_fmt", "-of", "csv=p=0", str(out)],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
        )
        assert "yuva" in r.stdout, f"Expected alpha pix_fmt, got {r.stdout!r}"

    def test_rendered_mov_has_transparent_background(self, tmp_path: Path):
        """Extract a frame from the middle of the clip and verify:
          - Multiple corner / edge pixels are alpha=0 (background transparent).
          - The frame is NOT entirely transparent (text actually rendered).
          - At least one opaque pixel exists (libass drew something).

        The original regression bug was alpha=255 everywhere (libass default
        with no `alpha=1` flag) — a single corner-pixel assertion missed this
        because it was checking the wrong direction (zero alpha on a frame
        that should have SOME non-zero alpha too).
        """
        import subprocess
        ass = tmp_path / "sub.ass"
        self._write_minimal_ass(ass)
        out = tmp_path / "sub.mov"
        render_transparent_video(
            ass_path=ass, output_path=out, duration_seconds=2.0,
            width=320, height=180, fps=25,
        )
        # Extract a frame as rgba PNG.
        frame = tmp_path / "frame.png"
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error",
             "-ss", "1.0", "-i", str(out),
             "-pix_fmt", "rgba", "-frames:v", "1", "-update", "1", str(frame)],
            check=True, capture_output=True, text=True, encoding="utf-8", errors="replace",
        )
        from PIL import Image
        img = Image.open(frame).convert("RGBA")
        w, h = img.size

        # Sample 4 corners + 4 mid-edge points — all must be fully transparent.
        sample_points = [
            (2, 2), (w - 3, 2), (2, h - 3), (w - 3, h - 3),
            (w // 2, 2), (w // 2, h - 3), (2, h // 2), (w - 3, h // 2),
        ]
        alphas = []
        for x, y in sample_points:
            _r, _g, _b, a = img.getpixel((x, y))
            alphas.append(a)
            assert a == 0, (
                f"Background at ({x},{y}) should be alpha=0, got alpha={a}. "
                f"Probable regression: libass filter is dropping alpha or "
                f"lavfi source isn't being chained through format=rgba."
            )

        # And confirm SOMETHING was drawn (frame isn't fully transparent).
        # Sample a strip of pixels along the vertical center line — if the cue
        # is rendered, at least one of these will be opaque (text fill / outline).
        center_x = w // 2
        center_alphas = [img.getpixel((center_x, y))[3] for y in range(0, h, 2)]
        assert max(center_alphas) > 0, (
            f"Frame is entirely transparent (all alphas along centerline = 0). "
            f"Probable regression: lavfi source forced alpha=0 across the frame."
        )

    def test_opaque_background_renders_visible_black(self, tmp_path: Path):
        """When opaque_background=True, the lavfi source is solid black (no
        alpha), so the rendered frame has alpha=255 everywhere AND the RGB
        channels are visibly black in the empty regions (not transparent)."""
        import subprocess
        ass = tmp_path / "sub.ass"
        self._write_minimal_ass(ass)
        out = tmp_path / "sub.mov"
        render_transparent_video(
            ass_path=ass, output_path=out, duration_seconds=2.0,
            width=320, height=180, fps=25,
            pix_fmt="yuv444p", opaque_background=True,
        )
        assert out.exists()
        assert out.stat().st_size > 0

        # Probe pixel format — must NOT be yuva* (no alpha plane).
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=pix_fmt", "-of", "csv=p=0", str(out)],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
        )
        assert "yuva" not in r.stdout, (
            f"Expected non-alpha pix_fmt for opaque_background=True, got {r.stdout!r}"
        )

        # Extract a frame and verify the background is visibly black (not
        # transparent, not grey). Use rgb24 (no alpha) since the alpha plane
        # is gone.
        frame = tmp_path / "frame.png"
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error",
             "-ss", "1.0", "-i", str(out),
             "-pix_fmt", "rgb24", "-frames:v", "1", "-update", "1", str(frame)],
            check=True, capture_output=True, text=True, encoding="utf-8", errors="replace",
        )
        from PIL import Image
        img = Image.open(frame).convert("RGB")
        w, h = img.size

        # Sample the four corners — all must be solid black.
        for x, y in [(2, 2), (w - 3, 2), (2, h - 3), (w - 3, h - 3)]:
            r_, g_, b_ = img.getpixel((x, y))
            assert (r_, g_, b_) == (0, 0, 0), (
                f"Background at ({x},{y}) should be solid black, got "
                f"RGB=({r_},{g_},{b_}). Check that opaque_background=True "
                f"dropped the alpha pipeline."
            )

        # And confirm SOMETHING was drawn (text actually rendered on top).
        center_x = w // 2
        center_row = [img.getpixel((center_x, y)) for y in range(0, h, 2)]
        # At least one pixel along the centerline is NOT pure black
        # (the text fill is white in our minimal ASS).
        has_non_black = any(px != (0, 0, 0) for px in center_row)
        assert has_non_black, (
            f"Frame is entirely black (no text drawn). Probable regression: "
            f"opaque_background=True dropped the ass filter."
        )


# ============================================================
# Background-box renderer (per-cue opaque back-boxes)
# ============================================================

@pytest.mark.fast
class TestRenderWithBackgroundBoxes:
    """Verifies `render_with_background_boxes` produces a video where:
      - The canvas outside cue text is fully transparent (alpha=0).
      - The box region around each cue is opaque black (alpha=255, RGB=0,0,0).
      - Gold text glyphs sit on top of the box.

    Bypasses libass BorderStyle=3 (which doesn't write alpha=255 for the box)
    and ffmpeg drawbox (which preserves source alpha) by drawing each box
    with Pillow and overlaying via ffmpeg's overlay filter.
    """

    @pytest.fixture(autouse=True)
    def _require_ffmpeg(self):
        import shutil
        if shutil.which("ffmpeg") is None:
            pytest.skip("ffmpeg not on PATH")

    def _write_minimal_ass(self, path: Path) -> None:
        path.write_text(
            "[Script Info]\n"
            "ScriptType: v4.00+\n"
            "PlayResX: 320\nPlayResY: 180\n"
            "\n[V4+ Styles]\n"
            "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
            "Style: Default,Arial,24,&H00FFFFFF,&H0000FFFF,&H00000000,&H00000000,-1,0,0,0,100,100,0,0,3,0.0,0,2,10,10,10,1\n"
            "\n[Events]\n"
            "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
            "Dialogue: 0,0:00:00.10,0:00:01.90,Default,,0,0,0,,HELLO\n",
            encoding="utf-8",
        )

    def test_box_region_opaque_corners_transparent(self, tmp_path: Path):
        """At cue time the canvas has an opaque black box around the text and
        transparent pixels at the corners (no full-screen black)."""
        from scripts.subtitle_design import render_with_background_boxes, SRTSegment
        import subprocess

        ass = tmp_path / "sub.ass"
        self._write_minimal_ass(ass)
        out = tmp_path / "sub.mov"
        segments = [SRTSegment(index=1, start_time=0.0, end_time=2.0, text="HELLO", source_file="vo.srt")]
        render_with_background_boxes(
            ass_path=ass, segments=segments, output_path=out,
            duration_seconds=2.0, width=320, height=180, fps=25,
            font_size=24, margin_v=10,
        )
        assert out.exists() and out.stat().st_size > 0

        # Extract a frame at t=1.0 (cue is active throughout).
        frame = tmp_path / "frame.png"
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error",
             "-ss", "1.0", "-i", str(out),
             "-pix_fmt", "rgba", "-frames:v", "1", "-update", "1", str(frame)],
            check=True, capture_output=True, text=True,
            encoding="utf-8", errors="replace",
        )
        from PIL import Image
        img = Image.open(frame).convert("RGBA")
        w, h = img.size

        # 1. Corners must be transparent (canvas outside the box).
        for x, y in [(2, 2), (w - 3, 2), (2, h - 3), (w - 3, h - 3)]:
            _r, _g, _b, a = img.getpixel((x, y))
            assert a == 0, f"Corner ({x},{y}) should be alpha=0 (transparent), got {a}"

        # 2. Somewhere in the frame there must be an opaque black pixel
        # (the back-box). Sample a horizontal strip in the bottom area
        # (margin_v=10 places the box near the bottom).
        bottom_strip = [
            img.getpixel((x, h - 30))[3] for x in range(0, w, 4)
        ]
        assert max(bottom_strip) == 255, (
            f"Expected at least one alpha=255 pixel in the box region, "
            f"max alpha in bottom strip = {max(bottom_strip)}"
        )

        # 3. At least one of those opaque pixels must be pure black RGB
        # (the box fill). Some opaque pixels could be the text glyphs.
        bottom_pixels = [
            img.getpixel((x, h - 30)) for x in range(0, w, 4)
        ]
        opaque_black = [p for p in bottom_pixels if p[3] == 255 and p[:3] == (0, 0, 0)]
        assert len(opaque_black) > 0, (
            f"Expected at least one opaque black pixel in the box region. "
            f"Sample = {bottom_pixels[:8]}"
        )

    def test_outside_cue_time_canvas_is_transparent(self, tmp_path: Path):
        """Before the first cue starts (and after the last cue ends), the
        canvas must be fully transparent — the box overlay is gated by
        enable='between(t,start,end)' and the ass filter only emits glyph
        pixels during its Dialogue window, so neither can write alpha=255
        pixels outside the cue time."""
        from scripts.subtitle_design import render_with_background_boxes, SRTSegment
        import subprocess

        # ASS with NO Dialogue lines at all — so the ass filter never writes
        # any pixel. The segment is at 2.5-3.5; at t=0.5 neither the box
        # overlay nor the ass filter is active.
        ass = tmp_path / "sub.ass"
        ass.write_text(
            "[Script Info]\n"
            "ScriptType: v4.00+\n"
            "PlayResX: 320\nPlayResY: 180\n"
            "\n[V4+ Styles]\n"
            "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
            "Style: Default,Arial,24,&H00FFFFFF,&H0000FFFF,&H00000000,&H00000000,-1,0,0,0,100,100,0,0,1,0.0,0,2,10,10,10,1\n"
            "\n[Events]\n"
            "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n",
            encoding="utf-8",
        )
        out = tmp_path / "sub.mov"
        segments = [SRTSegment(index=1, start_time=2.5, end_time=3.5, text="HELLO", source_file="vo.srt")]
        render_with_background_boxes(
            ass_path=ass, segments=segments, output_path=out,
            duration_seconds=4.0, width=320, height=180, fps=25,
            font_size=24, margin_v=10,
        )

        # Sample at t=0.5 — before both the segment and any ass dialogue.
        # Every pixel should be alpha=0.
        frame = tmp_path / "frame.png"
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error",
             "-ss", "0.5", "-i", str(out),
             "-pix_fmt", "rgba", "-frames:v", "1", "-update", "1", str(frame)],
            check=True, capture_output=True, text=True,
            encoding="utf-8", errors="replace",
        )
        from PIL import Image
        img = Image.open(frame).convert("RGBA")
        w, h = img.size

        all_alphas = [
            img.getpixel((x, y))[3]
            for y in range(0, h, 4)
            for x in range(0, w, 4)
        ]
        assert max(all_alphas) == 0, (
            f"Expected all-transparent frame before cue, got max alpha = "
            f"{max(all_alphas)}. Box overlay may not be gated by enable=."
        )


# ============================================================
# Audio voiceover input (transcription)
# ============================================================

@pytest.mark.fast
class TestAudioInput:
    """Audio paths through main() — exercises detection + cache logic without
    invoking the actual faster-whisper model (mocked)."""

    def test_audio_extensions_constant(self):
        from scripts.subtitle_design import AUDIO_EXTENSIONS
        assert ".mp3" in AUDIO_EXTENSIONS
        assert ".wav" in AUDIO_EXTENSIONS
        assert ".m4a" in AUDIO_EXTENSIONS
        assert ".srt" not in AUDIO_EXTENSIONS  # SRT must NOT be treated as audio

    def test_srt_ts_formatting(self):
        from scripts.subtitle_design import _srt_ts
        assert _srt_ts(0.0) == "00:00:00,000"
        assert _srt_ts(1.5) == "00:00:01,500"
        assert _srt_ts(3725.123) == "01:02:05,123"

    def test_cache_hit_skips_transcription(self, tmp_path: Path, monkeypatch):
        """When a fresh .srt already exists next to the audio, the transcriber
        must NOT be called (we patch it to raise if invoked)."""
        from scripts.subtitle_design import transcribe_audio_to_srt

        audio = tmp_path / "clip.mp3"
        audio.write_bytes(b"fake mp3")  # mtime = now
        srt = tmp_path / "clip.srt"
        srt.write_text("1\n00:00:00,000 --> 00:00:01,000\nHi\n\n", encoding="utf-8")
        # Make srt newer than audio
        import os
        os.utime(srt, (audio.stat().st_mtime + 5, audio.stat().st_mtime + 5))
        words = tmp_path / "clip.words.json"
        words.write_text("[]", encoding="utf-8")

        called = {"n": 0}
        def fake_model(*a, **kw):
            called["n"] += 1
            raise RuntimeError("should not be invoked on cache hit")
        monkeypatch.setattr(
            "faster_whisper.WhisperModel", fake_model, raising=False,
        )

        result = transcribe_audio_to_srt(audio, srt, words, model_size="base")
        assert result == srt
        assert called["n"] == 0

    def test_cache_bypass_when_flag_set(self, tmp_path: Path, monkeypatch):
        """use_cache=False always invokes the model, even when a fresh .srt
        exists. We verify by mocking WhisperModel and checking call count."""
        from scripts.subtitle_design import transcribe_audio_to_srt

        audio = tmp_path / "clip.mp3"
        audio.write_bytes(b"fake mp3")
        srt = tmp_path / "clip.srt"
        srt.write_text("1\n00:00:00,000 --> 00:00:01,000\nHi\n\n", encoding="utf-8")
        import os
        os.utime(srt, (audio.stat().st_mtime + 5, audio.stat().st_mtime + 5))
        words = tmp_path / "clip.words.json"

        # Stub the model + transcription iterator.
        class FakeWord:
            def __init__(self, w, s, e, p):
                self.word, self.start, self.end, self.probability = w, s, e, p
        class FakeSeg:
            def __init__(self, s, e, t, ws):
                self.start, self.end, self.text, self.words = s, e, t, ws
        class FakeModel:
            def __init__(self, *a, **kw): pass
            def transcribe(self, *a, **kw):
                return iter([
                    FakeSeg(0.0, 1.0, "hello", [FakeWord("hello", 0.0, 1.0, 0.9)]),
                ]), None

        monkeypatch.setattr("faster_whisper.WhisperModel", FakeModel, raising=False)

        result = transcribe_audio_to_srt(
            audio, srt, words, model_size="base", use_cache=False,
        )
        assert result == srt
        # The SRT was rewritten with the new content.
        assert "hello" in srt.read_text(encoding="utf-8")
        # And the words.json sidecar was written.
        import json as _json
        data = _json.loads(words.read_text(encoding="utf-8"))
        assert isinstance(data, list) and len(data) == 1
        assert data[0]["text"] == "hello"
