---
name: subtitle-design
description: Generate CapCut-style styled subtitles from a project SRT. Produces an editable .ass sidecar (with optional word-by-word karaoke + pop/fade animations) and a DaVinci-compatible OTIO file with a TrackKind.Text track. Use when the user says "make subtitles pop", "CapCut style subs", "MrBeast subtitles", "styled subtitles", "burn subtitles" (note: this skill does NOT burn into video — only sidecar/OTIO), or wants karaoke word-highlight on voiceover.
allowed-tools:
  - Read
  - Write
  - Edit
  - Glob
  - Grep
  - Bash
  - Agent
---

# Subtitle Design Skill

CapCut's subtitle styling is proprietary — there's no clean "extract" path. This skill
is the practical alternative: a standalone tool that produces the *output* CapCut would
produce (styled, optionally animated subtitles) using free/local tooling (Pillow, ffmpeg,
ASS libass format). Per `feedback_no_paid_software.md`, defaults are free/local — no
CapCut Pro / Gemini / Imagen needed.

Reads an existing project SRT (or its `.words.json` sidecar for karaoke) and emits:

- `subtitles.ass` — styled, optionally animated ASS subtitle file (libass format; readable
  by DaVinci's subtitle editor, VLC, MPV, OBS, ffmpeg `subtitles=` filter, and most modern
  video players)
- `subtitles.srt` — styled plain SRT (fallback for tools that don't read ASS)
- `subtitles.otio` — DaVinci-compatible OTIO with `TrackKind.Text` track, one Clip per cue,
  `ExternalReference` to the `.ass` file

The output is **not burned into video pixels** — it stays editable. To burn in, either:

- Pass `--render-video` to emit a transparent ProRes 4444 MOV (DaVinci-compatible
  alpha overlay you can drop above any main V1), or
- Run the output through `ffmpeg -vf "ass=subtitles.ass"` separately to bake
  onto a colored background.

## When to Use

- User asks for styled/animated subtitles on an existing voiceover SRT
- User says "CapCut style", "MrBeast style", "Hormozi style", "TikTok style" subtitles
- User wants karaoke word-by-word highlight (uses the existing `.words.json` sidecar)
- User wants pop-in or fade animations on each line
- User wants a DaVinci-compatible OTIO that brings styled subs into Resolve

## Invocation

```
/subtitle-design --style mrbeast --project <path>
```

## Direct Script Invocation (recommended)

```bash
python scripts/subtitle_design.py \
  --project "E:\Edit Job\samples\smoke-test-ollama" \
  --style mrbeast \
  --animation pop
```

### Flags

| Flag | Default | Description |
|---|---|---|
| `--project PATH` | (required) | Project directory; auto-discovers `voiceover*.srt` (prefers `_trimmed` variant) |
| `--voiceover PATH` | auto | Override SRT path |
| `--style NAME` | `capcut_default` | Preset: `capcut_default`, `mrbeast`, `hormozi`, `tiktok`, `minimal`, `news_broadcast` |
| `--animation {none,karaoke,pop,fade}` | (from preset) | Force a specific line animation |
| `--font NAME` | (from preset) | Override font family (e.g. `Impact`, `Arial Black`) |
| `--font-size N` | (from preset) | Override font size in points |
| `--primary-color HEX` | (from preset) | Override text color (e.g. `#FFFF00`) |
| `--outline-color HEX` | (from preset) | Override outline color |
| `--position {top,center,bottom}` | (from preset) | Override NUMPAD alignment |
| `--max-chars N` | (from preset) | Override line wrap threshold |
| `--output-dir PATH` | `<project>/subtitle_design` | Where to write outputs |
| `--formats {srt,ass,otio}` | `srt,ass,otio` | Comma-separated list of outputs |
| `--dry-run` | False | Print plan without writing |
| `--render-video` | False | Also burn the `.ass` into a transparent ProRes 4444 MOV (DaVinci-compatible alpha overlay) |
| `--video-codec` | `prores_ks` | ffmpeg video codec for the transparent render |
| `--video-container` | `mov` | Output container extension |
| `--video-resolution` | `1920x1080` | WidthxHeight for the transparent render |
| `--video-fps` | `30` | Frame rate for the transparent render |

## Workflow

### Phase 1 — Locate the SRT

`find_srt_path(project_dir)` (mirrors `scripts/title_color_video.py:33-58`) walks
`<project>/voiceover/*.srt` and `<project>/*.srt`, preferring `_trimmed` variants. Pass
`--voiceover` to override.

### Phase 2 — Load word timestamps

Reads the sibling `<srt>.words.json` sidecar (written by
`src/transcription/parallel_processor.py:1622-1630` during Whisper transcription).
If missing, karaoke animations degrade gracefully to per-segment timing.

### Phase 3 — Apply preset + overrides

`apply_preset(name)` returns a fresh `SubtitleStyle` (copy so caller mutations don't
leak). CLI flags `--font`, `--font-size`, `--primary-color`, `--outline-color`,
`--position`, `--max-chars`, `--animation` apply `dataclasses.replace()` overrides
on top of the preset.

### Phase 4 — Wrap long cues

`split_long_cues(segments, max_chars_per_line)` does word-aware wrapping using `\N`
(ASS hard line break). Lines that fit within `max_chars_per_line` are untouched.

### Phase 5 — Write ASS (styled + optionally animated)

`write_ass()` emits a static-styled `.ass`. `write_karaoke_ass()` adds per-word `{\k}`
karaoke tags. The POP animation uses `\t(0,200,\fscy110)\t(200,400,\fscy100)` for a
quick scale-bounce. The FADE animation uses `\fad(300,200)`.

### Phase 6 — Write OTIO Text track

`build_subtitle_otio()` creates a standalone OTIO with one `TrackKind.Text` track
named "V1 - Subtitles". Each cue becomes a Clip with `ExternalReference(target_url=
<ass_path>)` and `source_range.duration` matching the cue window. Forward-slash
Windows paths are mandatory (DaVinci Resolve hangs on backslash paths).

### Phase 7 — Output

Default output: `<project>/subtitle_design/{subtitles.srt, subtitles.ass, subtitles.otio}`.

## Presets

| Preset | Font | Size | Color | Outline | Animation | Notes |
|---|---|---|---|---|---|---|
| `capcut_default` | Arial bold | 56 | white | 2px black | none | Sensible default |
| `mrbeast` | Impact | 80 | yellow `#FFFF00` | 4px black | POP | The "huge yellow text" look |
| `hormozi` | Arial bold | 60 | white on red | 3px dark blue | FADE | Top-third placement |
| `tiktok` | Arial bold | 52 | white | 2px black | KARAOKE | Word-by-word yellow highlight |
| `minimal` | Arial | 40 | white | none | FADE | Quiet, documentary-style |
| `news_broadcast` | Arial bold | 44 | white on black bar | none | none | Lower-third news bar |
| `davinci_caps` | Anton | 78 | white, first word pink `#F080C0` | 6px black | none | Sampled from `1caps_DAVINCI.mov` — heavy condensed all-caps with thick black outline (the "pill" look comes from a thick outline, not an opaque back-box) |

## Key File Locations

- `scripts/subtitle_design.py` — main CLI (~600 lines)
- `.claude/skills/subtitle-design/skill.md` — this file
- `src/utils.py:993` — `parse_srt_file` (reused, not reimplemented)
- `src/utils.py:289` — `SRTSegment` dataclass (reused)
- `src/transcription/parallel_processor.py:1622-1630` — produces the `.words.json` sidecar
- `src/otio/utils.py:113` — `_to_windows_path` (inlined in this skill for portability)

## Related

- `.claude/skills/lower-thirds/` — sibling skill; generates name/title graphics with
  the same "standalone OTIO overlay" pattern
- `.claude/skills/capcut-export/` — sibling skill; cuts top-most OTIO clips as MP4s for
  CapCut drag-in. Use `subtitle-design` first to get styled subs, then `capcut-export`
  to cut clips if you want burned-in video.
- `.claude/skills/improve-otio/` — re-transcribes voiceover with word-level timing if
  your `.words.json` sidecar is stale or missing.
