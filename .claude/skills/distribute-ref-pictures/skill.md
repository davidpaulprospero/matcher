---
name: distribute-ref-pictures
description: Download N reference pictures of a person/entity and evenly distribute them as still-image clips across the span of an SRT onto a NEW single-track OTIO file, without mutating the source edited OTIO. The subject is auto-inferred from the narration by a local LLM (Ollama mistral:7b) when not given explicitly. Each image's duration is matched to the source OTIO clip covering that point (falling back to the SRT segment). Use `--fill-gaps` to instead place one image per SRT segment that lacks video coverage in the source OTIO (coverage detection mirrors /re-download-otio's topmost-visible rule). Use `--all` to run all three passes (fill-gaps MiniMax → normal MiniMax → normal pyimagedl) in one shot, sharing the parsed SRT, source OTIO, and LLM-derived subject, writing three separate OTIO files. Use when the user says "evenly distribute reference pictures over an OTIO", "spray reference photos across the timeline", "lay N images over a finished OTIO", "distribute ref pics across the SRT", "add a reference-photo track to an edited timeline", "fill the gaps in this OTIO with reference images", "do all three at once", "run gapfill, MiniMax, and pyimagedl back-to-back", or "all at once".
allowed-tools:
  - Read
  - Write
  - Edit
  - Glob
  - Grep
  - Bash
---

# Distribute Reference Pictures

Downloads N reference pictures of a person/entity (using the same `pyimagedl 0.4.9`
machinery as the `/download-pictures` skill) and lays them as still-image clips
across the span of an SRT. Output is a **separate new OTIO file**; the source
OTIO is never modified.

The N image centers are evenly spaced across the SRT span. Each image's
**duration is taken from the SRT segment whose `[start, end)` contains
the image's center** (priority order: SRT segment → source OTIO clip →
nearest SRT segment → uniform slice). This way a still picture ends up
with the exact same on-screen length as the line of voiceover it sits
over.

Unlike `/shot-otio` and `/imagen-otio` (AI-generated stills), this skill
defaults to spraying **real reference photos** (thumbnails) onto a new
track and never touches the existing edit. Think "here are reference
shots you can drop into Resolve alongside your edit" rather than
"auto-generate a full AI edit".

It also supports an **AI-generation mode** via `--provider minimax` for
when you want generated stills instead of downloaded thumbnails — useful
for biopic/documentary edits where the SRT topic has no stock-photo
hits, or for visual consistency across the track. The same per-slot
SRT-text query logic applies; only the image source changes.

## When to Use

- User says "evenly distribute reference pictures over an OTIO", "spray
  reference photos across the timeline", "lay N images over a finished
  OTIO", "distribute ref pics across the SRT", or "add a reference-photo
  track to an edited timeline".
- User wants reference imagery of a single named subject
  (person, place, entity) laid out alongside an existing edit.
- User wants the source OTIO preserved (this skill always produces a new
  separate file).
- User does not know the exact subject name (the SRT-derived name is
  ambiguous, e.g. "oscar") — the LLM-inference step figures it out from
  the narration itself.

## Invocation

```
/distribute-ref-pictures <srt> <otio> [--subject "..."] [--max-images N] [--output <path>]
/distribute-ref-pictures <srt> <otio> --no-llm            # skip LLM, use filename
/distribute-ref-pictures <srt> <otio> --dry-run

# NEW: fill gaps in the source OTIO with one image per uncovered SRT segment
/distribute-ref-pictures <srt> <otio> --fill-gaps [--subject "..."]

# NEW: fill gaps with MiniMax AI stills (shorthand for --fill-gaps --provider minimax)
/distribute-ref-pictures <srt> <otio> --minimax-fill

# NEW: run all three passes back-to-back (fill-gaps MiniMax, normal MiniMax, normal pyimagedl)
/distribute-ref-pictures <srt> <otio> --all
/distribute-ref-pictures <srt> <otio> --all --all-skip-minimax
/distribute-ref-pictures <srt> <otio> --all --all-minimax-images 30
```

## Direct Script Invocation (recommended — runs without prompting)

```bash
python scripts/distribute_ref_pictures.py \
  --srt "E:/Edit Job/fransisco/voiceover/voiceover.srt" \
  --otio "E:/Edit Job/fransisco/edit.otio" \
  --subject "Mario Moreno Reyes" \
  --context "Golden Age Mexican cinema" \
  --max-images 25
```

## Arguments

| Flag | Default | Notes |
| --- | --- | --- |
| `--srt` | required | SRT file (narration schedule). |
| `--otio` | required | Source edited OTIO (read-only). |
| `--output` | `<otio_stem>_ref_pictures.otio` next to source | New OTIO path. Refuses to equal `--otio`. |
| `--subject` | LLM-inferred (see below) | pyimagedl search query. Explicit value overrides LLM + heuristic. |
| `--no-llm` | off | Skip LLM inference; use the SRT-filename heuristic for `--subject`. |
| `--llm-provider` | `ollama` | LLM for subject inference: `ollama`, `gemini`, or `anthropic`. |
| `--llm-model` | `mistral:7b` | Model name. llama3.2 (3.2B) is too weak to follow the query instruction. |
| `--llm-host` | `$OLLAMA_HOST` or `http://localhost:11434` | Ollama host URL. |
| `--llm-samples` | 12 | Number of SRT segments fed to the LLM for inference. |
| `--context` | _none_ | Optional scope phrase (passed to downloader). |
| `--max-images` | 25 | Target image count. Ignored in `--fill-gaps` mode (count is determined by the OTIO+SRT). |
| `--fill-gaps` | off | Switch to gap-fill mode: place one image per SRT segment that lacks video coverage in the source OTIO. See [Fill-gaps mode](#fill-gaps-mode). |
| `--gap-coverage-mode` | `topmost` | How to decide if an SRT segment is covered. `topmost` (default) uses `/re-download-otio`'s rule: a moment is covered iff the highest-track enabled non-gap non-audio clip covers it. `track` looks only at `--track-index`. |
| `--gap-threshold` | 0.5 | Coverage fraction below which an SRT segment counts as a gap (default 0.5 = majority uncovered). 0.0 = require zero coverage (strict); 1.0 = flag every segment with any uncovered portion (matches `--gap-coverage-mode track` count); values > 1.0 force every segment to be a gap. |
| `--max-gap-duration` | _none_ | Skip SRT segments longer than this many seconds in `--fill-gaps` mode. Useful when one SRT line is 60s+ and you only want to patch small gaps. |
| `--min-size-kb` | 8 | Lower-bound size for saved images. |
| `--sources` | downloader default (Google+Bing+Yandex+Wikipedia) | Comma-separated imagedl source IDs. |
| `--per-source-limit` | 40 | Max candidates per (query, source). |
| `--exclude-domains` | Alamy/Shutterstock/Getty stock-photo hosts | Comma-separated domains to skip (substring match on URL). Pass `""` to disable. Applied at downloader and as a post-download safety net that re-checks each sidecar JSON. |
| `--tmp-dir` | `<output>_images` | Image download dir. |
| `--keep-tmp` | off | Keep image dir after the run. |
| `--frame-rate` | 30.0 | Timeline rate. |
| `--track-name` | `V-Ref` | New track name. |
| `--track-index` | 0 | Which video track of source OTIO to read durations from. |
| `--match-duration` | off | Trailing gap so new track equals source OTIO total duration. |
| `--dry-run` | off | Plan only — no download, no write. |
| `--verbose` | off | Extra logging. |
| `--candidates` | `1` | Per-gap candidate count. Only valid with `--fill-gaps`. Builds `N` parallel `V-Ref-Cand00..NN` tracks with identical frame timing so Resolve can flip among candidates. See [Multi-candidate mode](#multi-candidate-mode). |
| `--candidate-track-prefix` | `f"{track_name}-Cand"` | Prefix used to name each candidate track (`{prefix}{NN:02d}`). |
| `--detect-names` | off | In `--fill-gaps` mode, scan each gap's overlapping SRT line(s) for a named entity and prepend it to the pyimagedl query, so the first candidate becomes a portrait/biopic-style picture of the actual person/place rather than a generic scene. Falls back to subject-only when no name is found. See [Name detection](#name-detection). |
| `--name-provider` | `ollama` | LLM for name extraction: `ollama` (with heuristic fallback) or `heuristic` only. |
| `--provider` | `pyimagedl` | Image source: `pyimagedl` (download thumbnails) or `minimax` (generate AI stills via `src/minimax_image.py`). `--provider minimax` requires `MINIMAX_API_KEY` in env. |
| `--minimax-size` | `1792x1024` | `WxH` for MiniMax output. Mapped to aspect ratio via `ASPECT_MAP`; unknown sizes fall back to `16:9`. Ignored unless `--provider minimax`. |
| `--minimax-aspect` | auto from size | Force an explicit MiniMax `aspect_ratio` (`16:9`, `9:16`, `1:1`, `4:3`, `3:4`). Wins over `--minimax-size`. Ignored unless `--provider minimax`. |
| `--minimax-fill` | off | Shorthand for `--fill-gaps --provider minimax`: writes one MiniMax AI still per uncovered SRT segment to `<otio_stem>_fill_gaps.otio`. Mutually exclusive with `--fill-gaps`, `--provider`, `--all`, and `--existing-images-dir`. Requires `MINIMAX_API_KEY`. See [MiniMax-fill shorthand](#minimax-fill-shorthand). |
| `--all` | off | Run three sequential passes against the same SRT/OTIO pair and write three new OTIO files: (1) `--fill-gaps` with `minimax`, (2) normal distribution with `minimax` (20 images), (3) normal distribution with `pyimagedl` (50 images). Mutually exclusive with `--fill-gaps`, `--provider`, and `--minimax-fill`. See [All-at-once mode](#all-at-once-mode). |
| `--all-minimax-images` | 20 | MiniMax image count for pass 2 of `--all`. Only valid with `--all`. |
| `--all-skip-minimax` | off | Skip the MiniMax pass in `--all` mode (useful when `MINIMAX_API_KEY` is unset). Only valid with `--all`. |

## Workflow

### Phase 1 — Resolve + validate inputs

- SRT and source OTIO must exist.
- `--output` must differ from `--otio` (refuses to overwrite source).
- **Subject precedence:**
  1. `--subject "..."` (explicit) — always wins, skips LLM.
  2. **LLM inference** (default) — feeds the first `--llm-samples` SRT
     segments to `--llm-model` via `--llm-provider` and asks for a concise
     image-search query. Skips any leading preamble/disclaimer the model
     emits. Uses `mistral:7b` on local Ollama by default (llama3.2 is too
     weak). Caching is disabled and the model name is part of the cache
     prefix, so switching models never returns a stale answer.
  3. **SRT-filename heuristic** — fallback when `--no-llm` is set or the LLM
     call fails: SRT stem with `_` → space and common suffixes
     (`voiceover`, `vo`, language tags, version digits) stripped.
- Log the chosen query loudly so the user can override if it's wrong.

### Phase 2 — Parse source OTIO → `existing_slots`

- Read source OTIO with `otio.adapters.read_from_file()`.
- Pick the video track at `--track-index` (default 0). Match
  `otio.schema.TrackKind.Video` with a string fallback for older
  versions.
- Walk the track with a running frame cursor. For each `Clip` with
  `enabled=True`, record `(start_sec, end_sec, dur_frames, name)`.
  Gaps advance the cursor but are not recorded.
- `source_total_frames` is the total duration (used by
  `--match-duration`).

### Phase 3 — Parse SRT

- Inline regex-based parser (same approach as `scripts/imagen_otio.py`).
- Produces `[(start_sec, end_sec, text), ...]`.
- Total SRT span = `srt_end - srt_start`.

### Phase 4 — Plan N image slots

For `i in 0..N-1`:

- Center: `srt_start + (i + 0.5) * (total_srt_dur / N)`.
- Duration priority (**SRT-first**):
  1. **SRT segment** containing center → `dur_source="srt"`. Each image
     takes the exact duration of the line of voiceover it sits over.
  2. **Source OTIO clip** containing center → `dur_source="otio"`,
     capture clip name. Fallback when the image center lands in an
     OTIO gap (no SRT coverage there).
  3. **Nearest SRT segment** by center distance → `dur_source="srt_nearest"`.
  4. **Uniform slice** → `dur_source="uniform"`.
- `dur_frames = max(1, round(dur_sec * frame_rate))`.
- Placement: `start_frames = max(prev_end, round(center_sec*rate) - dur_frames//2)`.
  Center-anchored, with right-clamp to prevent overlap.

In dry-run, print a table: `idx | center (HH:MM:SS:FF) | center (s) |
dur_f | start_f | dur_source | slot_name`.

### Phase 5 — Download images (skipped in dry-run)

Shell out via `subprocess.run([sys.executable,
"scripts/download_reference_pictures.py", ...])` with the same flags
the existing `/download-pictures` skill uses. Exit codes:

- `0` = ≥1 image saved → proceed.
- `1` = bad args / no candidates → hard fail, exit 3.
- `2` = candidates found but zero passed size filter → warn, continue.

Collect results by globbing `<tmp>/<slug>_NN.<ext>` (slug matches the
downloader's `_slugify` exactly: `re.sub(r"[^A-Za-z0-9]+", "_",
name).strip("_").lower()[:40] or "subject"`), sort by the numeric
`NN` suffix, drop files below `--min-size-kb` bytes or with
problematic unicode (`_has_problematic_path` from
`src/otio/utils.py`).

### Phase 6 — Reconcile N vs downloaded count

- `actual == 0` → warn; write a valid OTIO with an empty `V-Ref`
  track; exit 0.
- `0 < actual < N` → warn; **re-plan with N = actual** (re-spread
  evenly over the actual count).
- `actual >= N` → use the first N, keep the original plan.

### Phase 7 — Build the new OTIO

- `otio.schema.Timeline(name=output_path.stem)`,
  `metadata["Resolve_OTIO"] = {"Resolve OTIO Meta Version": "1.0"}`,
  `global_start_time = RationalTime(0, rate)`.
- Single `Track(name=args.track_name, kind=Video)`.
- For each slot + image file (sorted by `start_frames`):
  - Leading **Gap** if `start_frames > cursor`.
  - `_to_windows_path(file)` for `target_url` (forward-slash absolute
    path, matches `src/otio/entities.py:549`).
  - **ExternalReference** with `available_range == source_range ==
    TimeRange(start=0, duration=dur_frames)` — required for DaVinci
    still import (see `src/otio/entities.py:558-564`).
  - **Clip** with `FreezeFrame()` effect (PNG/JPEG have 1 actual
    frame; without this Resolve shows "Media Offline").
  - Metadata: `Resolve_OTIO={}`, `is_still_image=True`,
    `image_index`, `image_path`, `srt_center_sec`, `dur_source`,
    `subject`, `source_otio_clip` (when present).
- Trailing **Gap** if `--match-duration` and `cursor <
  source_total_frames`.

### Phase 8 — Write outputs

- `otio.adapters.write_to_file(new_tl, output_path)`.
- JSON summary at `<output_path>.summary.json` with subject, paths,
  frame rate, requested/downloaded counts, and per-image
  `index/path/center_sec/start_frame/dur_frames/dur_source/slot_name`.

## AI generation mode (`--provider minimax`)

Replaces the pyimagedl downloader with `src.minimax_image.py`
(MiniMax `image-01`, endpoint `https://api.minimax.io/v1/image_generation`).
Each plan entry becomes one MiniMax call with prompt
`"<subject>, <SRT text or slot_name>, <context>"` (comma-joined,
context optional) — see `build_minimax_prompt` at
`scripts/distribute_ref_pictures.py:715`.

### Setup

`MINIMAX_API_KEY` must be in env. The script does **not** load `.env`
on its own — `MINIMAX_API_KEY` lives in `D:/_Projects/voiceover-matcher-dev/.env`,
so invoke via `python -c "from dotenv import load_dotenv; load_dotenv('.env'); ..."`
or `set MINIMAX_API_KEY=...` in the shell. The client raises
`RuntimeError("MINIMAX_API_KEY not set…")` at construction if missing,
and the script logs that and exits rc=1.

### Behavior vs `pyimagedl`

| Aspect | `pyimagedl` | `minimax` |
| --- | --- | --- |
| Network | 4 sources, candidate scrape | 1 POST per slot |
| Hit rate | Variable (34/50–50/50 typical) | Deterministic per-prompt (one image back) |
| Speed | 30s+ per query | 2–4s per slot |
| File format | `.jpg`/`.webp`/`.png` (whichever the source served) | `.jpg` or `.png` (sniffed from magic bytes; `generate_image_to_file` auto-corrects extension) |
| Filenames | `<slug>_NN.<ext>` | `minimax_NNN.jpg` (zero-padded per slot index) |
| Domain filtering | `--exclude-domains` blacklist | None — generated pixels |
| Failure mode | Returns 0 images, rc=2 | Per-slot `requests.HTTPError` is logged, slot skipped; remaining slots continue |

### Per-slot failures are silent

`generate_minimax_for_slots` (line 738) catches `Exception` per slot
and continues. A 401/429 from MiniMax on one prompt does **not** stop
the run. If `len(image_paths) < len(plan)`, the post-build replan logic
from the pyimagedl branch still applies — you get a valid OTIO with
fewer clips, not a hard fail.

### Example

```bash
python scripts/distribute_ref_pictures.py \
  --srt "E:/Edit Job/Oscar/oscar-vo-7__2026-08-03/voiceover/oscar-vo-7_trimmed.srt" \
  --otio "E:/Edit Job/Oscar/oscar-vo-7__2026-08-03/7real.otio" \
  --fill-gaps --no-llm \
  --subject "Mexican traditional nixtamalization tortilla making" \
  --provider minimax \
  --max-gap-duration 30
# -> 41 MiniMax calls, ~2-3 min, ~41 generated jpgs.
```

Useful flags:

- `--minimax-size 1280x720` — smaller/faster if 1792x1024 is overkill.
- `--minimax-aspect 1:1` — square crops for thumbnails/portrait reels.

## `--minimax-fill` shorthand

`--minimax-fill` is a one-flag alias for `--fill-gaps --provider minimax`. It
writes one MiniMax AI still per uncovered SRT segment to
`<otio_stem>_fill_gaps.otio` — same OTIO shape as `--fill-gaps`, but with
generated pixels instead of downloaded thumbnails. Requires `MINIMAX_API_KEY`
in env. Useful when stock-photo hits for the gap topic are sparse
(biographical / obscure historical content, fictional re-creations) and AI
stills are preferred over thumbnails.

Inherits every `--fill-gaps` option (`--gap-coverage-mode`, `--gap-threshold`,
`--max-gap-duration`, `--detect-names`, `--candidates`, ...) plus every
MiniMax option (`--minimax-size`, `--minimax-aspect`, ...).

```bash
python scripts/distribute_ref_pictures.py \
  --srt "E:/Edit Job/Oscar/oscar-el-amish-ezekiel-explica-como-dejar-algo-a-tus-hijos-sin-quedarte-tu-sin-nada__2026-08-12/voiceover/el-amish-ezekiel-explica-como-dejar-algo-a-tus-hijos-sin-quedarte-tu-sin-nada.en.srt" \
  --otio "E:/Edit Job/Oscar/oscar-el-amish-ezekiel-explica-como-dejar-algo-a-tus-hijos-sin-quedarte-tu-sin-nada__2026-08-12/oscar-el-amish-ezekiel-explica-como-dejar-algo-a-tus-hijos-sin-quedarte-tu-sin-nada__2026-08-12.otio" \
  --minimax-fill
# -> 16 MiniMax calls (~30-90s), <otio>_fill_gaps.otio with 16 AI stills.
```

Mutually exclusive with `--fill-gaps`, `--provider`, `--all`, and `--existing-images-dir`.

## Output Locations

- New OTIO: `<source_otio_dir>/<source_otio_stem>_ref_pictures.otio` (or
  wherever `--output` points).
- Summary: `<output>.summary.json`.
- Images: `<output_stem>_images/` (default), unless `--tmp-dir` is set.
  Not deleted; the script just logs the path. Use `--keep-tmp` to opt
  into retaining the imagedl intermediate `_workdir/` as well.

## Complete Example

Subject auto-inferred by the local LLM (recommended when the SRT filename is
ambiguous, e.g. "oscar" or "hqpls"):

```bash
python scripts/distribute_ref_pictures.py \
  --srt "E:/Edit Job/Oscar/oscar-vo-1__2026-07-21/voiceover/oscar-vo-1_trimmed.srt" \
  --otio "E:/Edit Job/Oscar/oscar-vo-1__2026-07-21/hqpls_hq.otio" \
  --max-images 50
# -> LLM infers subject from the narration (e.g. "Corn husking scene in winter")
#    instead of searching for random people named "Oscar".
```

> **Caveat:** `mistral:7b` returns slightly different phrasings across runs
> (e.g. "Corn husking scene in winter" vs "Corn harvest in Mixteca Guajaqueña"
> vs "Breaking corn sounds in Mixteca Guajaque"). All are on-topic for the
> same SRT, but if the LLM's pick isn't what you want, override with
> `--subject`.

Override the LLM entirely (e.g. when the SRT is being re-recorded with
different content — oscar-vo-1 was originally about Mexican corn harvest but
is being re-narrated as Amish lifestyle):

```bash
python scripts/distribute_ref_pictures.py \
  --srt "E:/Edit Job/Oscar/oscar-vo-1__2026-07-21/voiceover/oscar-vo-1_trimmed.srt" \
  --otio "E:/Edit Job/Oscar/oscar-vo-1__2026-07-21/hqpls_hq.otio" \
  --subject "Amish lifestyle" \
  --context "rural farming, simple living, traditional work" \
  --max-images 50
# -> Both --subject and --context feed the search: 6 base variants plus one
#    pair per context fragment split on commas (12 total for 3 fragments).
```

Output:
- `E:/Edit Job/fransisco/edit_ref_pictures.otio`
- `E:/Edit Job/fransisco/edit_ref_pictures.summary.json`
- `E:/Edit Job/fransisco/edit_ref_pictures_images/mario_moreno_reyes_01.jpg` … `_25.jpg`

The source `edit.otio` is **not** modified (mtime/size unchanged). Open the
new `edit_ref_pictures.otio` in DaVinci Resolve; `V-Ref` is a fresh
top-level video track with still-image clips whose durations line up with
the source's V1 clips.

## All-at-once mode

`--all` runs three sequential passes against the same SRT/OTIO pair inside
one process and writes three new OTIO files. All three passes share the
resolved subject (LLM is called **once**), the parsed SRT, and the source
OTIO, so this avoids re-running the LLM and re-parsing the OTIO three
times.

### Pass order

1. **Fill-gaps with MiniMax** → `<otio_stem>_fill_gaps.otio`
   One AI still per uncovered SRT segment. `--provider minimax`; runs only
   if `MINIMAX_API_KEY` is set.
2. **Normal distribution with MiniMax** → `<otio_stem>_ref_pictures_minimax.otio`
   20 AI-generated stills (default; `--all-minimax-images N` to override).
3. **Normal distribution with pyimagedl** → `<otio_stem>_ref_pictures.otio`
   50 downloaded thumbnails (script default `--max-images=25` is
   auto-bumped to 50 when `--all` is used and the user didn't pass
   `--max-images` explicitly).

Each pass writes its own `*.summary.json` next to its OTIO and uses its
own `<stem>_images/` tmp dir — no pass overwrites another. The process
exit code is the **worst** of the three pass exit codes; banners on the
log show per-pass status:

```
[--all]  Pass 1/3: fill-gaps, MiniMax
[--all]  Pass 2/3: normal, MiniMax (20 images)
[--all]  Pass 3/3: normal, pyimagedl (50 images)
[--all] === Pass 1/3: fill-gaps with MiniMax ===
[--all] Pass 1/3 complete: rc=0, output=...\2_fill_gaps.otio
[--all] === Pass 2/3: normal distribution with MiniMax ===
[--all] Pass 2/3 complete: rc=0, output=...\2_ref_pictures_minimax.otio
[--all] === Pass 3/3: normal distribution with pyimagedl ===
[--all] Pass 3/3 complete: rc=0, output=...\2_ref_pictures.otio
[--all]  Pass 1: OK -> ...\2_fill_gaps.otio
[--all]  Pass 2: OK -> ...\2_ref_pictures_minimax.otio
[--all]  Pass 3: OK -> ...\2_ref_pictures.otio
```

### Flags

| Flag | Default | Notes |
| --- | --- | --- |
| `--all` | off | Run all three passes. Mutually exclusive with `--fill-gaps`, `--provider`, and `--minimax-fill`. |
| `--all-minimax-images` | 20 | MiniMax image count for pass 2. Only valid with `--all`. |
| `--all-skip-minimax` | off | Skip pass 2 (useful when `MINIMAX_API_KEY` is unset). Pass 1 (fill-gaps with MiniMax) still runs. Only valid with `--all`. |

`--all` implies these per-pass settings:

- Pass 1 (`minimax` + `--fill-gaps`): one MiniMax call per uncovered SRT segment; `--max-images` is set automatically by the fill-gaps planner.
- Pass 2 (`minimax` normal): `--max-images 20` (or `--all-minimax-images`).
- Pass 3 (`pyimagedl` normal): `--max-images 50` (auto-bumped from the script default 25); `--sources GoogleImageClient,BingImageClient,YandexImageClient` (Wikipedia excluded).

Any flag not listed above is passed through to all three passes
unchanged (e.g. `--subject`, `--keep-tmp`, `--dry-run`, `--exclude-domains`).

### Example

```bash
python scripts/distribute_ref_pictures.py \
  --srt "E:/Edit Job/Oscar/8-4/oscar-vo-10/voiceover/oscar-vo10-half.srt" \
  --otio "E:/Edit Job/Oscar/8-4/oscar-vo-10/2.otio" \
  --all
# -> Writes 3 OTIOs:
#    E:/Edit Job/Oscar/8-4/oscar-vo-10/2_fill_gaps.otio           (≈29 gap-fill stills)
#    E:/Edit Job/Oscar/8-4/oscar-vo-10/2_ref_pictures_minimax.otio (20 AI stills)
#    E:/Edit Job/Oscar/8-4/oscar-vo-10/2_ref_pictures.otio         (≈50 real photos)
#    LLM is called once; SRT + source OTIO are parsed once.
```

Skip the MiniMax pass when the API key is unset:

```bash
python scripts/distribute_ref_pictures.py \
  --srt "E:/Edit Job/Oscar/8-4/oscar-vo-10/voiceover/oscar-vo10-half.srt" \
  --otio "E:/Edit Job/Oscar/8-4/oscar-vo-10/2.otio" \
  --all --all-skip-minimax
```

### Notes / caveats

- `MINIMAX_API_KEY` is required for passes 1 and 2 unless
  `--all-skip-minimax` is passed. Without the key, the script exits 1
  on the first MiniMax pass it tries.
- **Dry-run does not stop MiniMax API calls.** The MiniMax branch sends
  one HTTP request per planned slot even under `--dry-run` (pre-existing
  behavior of the per-pass code path). Use `--all-skip-minimax` for a
  true free smoke test.
- The three passes write to **distinct filenames** by default; running
  `--all` repeatedly overwrites the previous run's outputs.
- `--all` shares the LLM subject across passes — if you want a
  per-pass subject override (rare), run each pass separately.

## Fill-gaps mode

`--fill-gaps` flips the planning strategy: instead of evenly distributing N
images across the SRT span, the script groups consecutive uncovered SRT
segments into **continuous gap regions** and places **one image per gap
region** (not per SRT segment). Use this when a previous edit left gaps
under the narration and you want a still image at every uncovered stretch.

### Fill-gaps mode requirements

When running `--fill-gaps`, treat each uncovered SRT segment as its own voiceover-specific image request—not as a generic subject-image batch:

- **Perfect inline timing is mandatory.** Each still must start exactly at the uncovered SRT segment's `start` and end exactly at its `end`, using the same frame-rate conversion as the source timeline. Do not center, pad, shorten, stretch, or shift a fill clip. Before reporting success, re-read the generated OTIO and double-check every fill clip's start frame and duration against the corresponding SRT segment; report any frame mismatch instead of silently accepting it.
- **Search queries must combine topic and narration.** For every gap, build the image query from the resolved subject/topic plus the actual SRT voiceover text for that gap (and any explicitly supplied context). The query must describe imagery for that specific spoken line, not merely the person/entity or broad project topic. Preserve the per-gap association in the slot name/summary so downloaded images can be audited against the voiceover.
- If a gap's SRT text is too vague by itself, retain the exact voiceover text and add only enough resolved topic/context to make the visual subject unambiguous. Never replace the line with a generic subject-only query.

### How a "gap" is detected

The script walks every enabled track in the source OTIO at every unique
start moment, picks the highest-track enabled non-gap non-audio clip
with an `ExternalReference` (mirroring `/re-download-otio`'s
`find_top_most_clips` rule), and merges those clips into a set of
coverage regions. An SRT segment is considered a "gap" if its
**covered fraction** of those regions is below `--gap-threshold`
(default `0.5` = majority uncovered). `--gap-coverage-mode track`
narrows the check to `--track-index` (default V1) only.

### Output shape

A **brand-new single-track OTIO** (matches the default mode's "don't
mutate source" rule). One `V-Ref` track with **one still per continuous
gap region in the source OTIO**. Each gap region produces one plan entry
with `dur_source: "continuous_gap"`; the clip's `[start_frame,
start_frame + dur_frames]` covers the on-screen window where the source
OTIO had nothing, even if that window spans multiple SRT segments.

`V-Ref` mirrors source coverage — where the source OTIO has video, V-Ref
has a `Gap` item. The result is still useful: every source OTIO gap is
covered by a still, and the rest of the timeline stays in sync with the
source edit.

A 20-second OTIO gap that covers three 6–7s SRT segments produces one
~20s gap entry (not three) — use `--max-gap-duration N` to drop very
long regions.

When downloads return fewer images than plan entries, the script
**pads with replicated successful images** (cycling through them) so
every gap region gets a still rather than being dropped.

For each gap entry, the downloader query must be specific to that
voiceover line: `<resolved topic/subject> + <exact SRT text> +
<optional context>`. Do not use one generic subject query for all gap
images. The generated summary must retain the SRT text and query for
every slot so relevance can be checked.

### Verifying the output

For each gap entry, `clip.metadata.dur_source` is `"continuous_gap"`.
The `summary.json` records the *planned* `start_frame` and `dur_frames`,
but the on-disk OTIO stores each clip's `source_range` as
`TimeRange(0, dur_frames)` — so reading `clip.source_range.start_time`
returns `0.0` for every still, not the in-timeline position. To verify
placement, walk the track with a running cursor:

```python
import opentimelineio as otio
tl = otio.adapters.read_from_file(out_path)
track = tl.tracks[0]
cursor = 0
for c in track:
    if isinstance(c, otio.schema.Gap):
        cursor += c.source_range.duration.value
    elif isinstance(c, otio.schema.Clip):
        # start_frame=cursor, dur=c.source_range.duration.value
        cursor += c.source_range.duration.value
# Each Clip should land on a frame window where the source OTIO had no
# video; each Gap should land on a frame window where the source had
# video. Compare against the source OTIO's coverage to confirm.
```

The "perfect inline timing" rule in the requirements section still
holds *at the gap-region level* — each fill clip's `(start, end)` is
the in/out of one continuous gap from the source OTIO. Track boundaries
match the source OTIO's gap boundaries exactly modulo 1-frame
frame-rate rounding; the build step absorbs that 1-frame slop by
nudging the clip rather than inserting a new Gap item.

### Example

```bash
python scripts/distribute_ref_pictures.py \
  --srt "E:/Edit Job/Oscar/7-30/campo-mexicano__2026-07-30/voiceover.en.srt" \
  --otio "E:/Edit Job/Oscar/7-30/pics pls.otio" \
  --fill-gaps \
  --subject "Mexican rural village daily life"
# -> Finds the 33 SRT segments with no topmost-visible video coverage,
#    downloads ~33 reference images via pyimagedl, writes
#    pics pls_fill_gaps.otio with one still per gap.
```

Dry-run to see which segments will be filled (no download, no write):

```bash
python scripts/distribute_ref_pictures.py \
  --srt "E:/Edit Job/Oscar/7-30/campo-mexicano__2026-07-30/voiceover.en.srt" \
  --otio "E:/Edit Job/Oscar/7-30/pics pls.otio" \
  --fill-gaps --dry-run --no-llm --subject "Mexican rural village"
```

Reuse already-downloaded images from a folder (skip the imagedl step):

```bash
python scripts/distribute_ref_pictures.py \
  --srt "E:/Edit Job/Oscar/.../voiceover.en.srt" \
  --otio "E:/Edit Job/Oscar/.../pics pls.otio" \
  --fill-gaps --no-llm --subject "Mexican rural village" \
  --existing-images-dir "E:/Edit Job/Oscar/7-30/ref_imgs"
```

Default output naming switches to `<otio_stem>_fill_gaps.otio` in this
mode (instead of `<otio_stem>_ref_pictures.otio`), and the JSON summary
records each slot's `dur_source: "srt_gap"` plus the SRT text used as
the slot name.

### Differences from default mode

| Aspect | Default | `--fill-gaps` |
| --- | --- | --- |
| Image count | `--max-images` (default 25) | Number of uncovered SRT segments (auto) |
| Placement | Evenly spaced across SRT span | One per gap, starting at SRT segment start |
| Per-image duration | SRT segment / source OTIO clip / uniform | Full SRT segment duration |
| `--gap-threshold` | n/a | Coverage fraction below which a segment is a gap (default 0.5) |
| `--gap-coverage-mode` | n/a | `topmost` (default) or `track` |
| `--max-gap-duration` | n/a | Skip segments longer than N seconds (e.g. 60) |
| Output filename suffix | `_ref_pictures.otio` | `_fill_gaps.otio` |
| `dur_source` in summary | `srt` / `otio` / `srt_nearest` / `uniform` | `continuous_gap` |

## Multi-candidate mode

`--candidates N` (only valid with `--fill-gaps`) builds `N` parallel OTIO
tracks rather than packing every image on one track. All candidates share
**identical gap start/end frames** so the user can flip tracks on/off in
Resolve to compare portraits/biopic pictures and pick the best one.

### Behavior

- Tracks are named `<candidate-track-prefix>{NN:02d}` (default prefix
  `V-Ref-Cand`, so `V-Ref-Cand00`, `V-Ref-Cand01`, …).
- Track 0 is the primary, tracks 1..N-1 are alternates.
- Each track holds exactly `len(plan)` clips — one per gap — with the
  same `start_frame`, `dur_frames`, and trailing gap.
- Per-clip `metadata` carries `candidate_index` and `candidate_track_name`
  for Resolve-side auditing.
- Downloads are isolated under `<tmp>/cand_NN/` so the downloader's
  `<slug>_NN.<ext>` filenames never collide between candidates. Candidate
  0 uses the base subject/query; candidates 1..N-1 append a short SRT
  snippet (or `cand{NN}` when none) as a defensive disambiguator.

### Summary JSON shape

For `candidates > 1`, the JSON switches to:

```json
{
  "subject": "...",
  "candidates": 3,
  "track_names": ["V-Ref-Cand00", "V-Ref-Cand01", "V-Ref-Cand02"],
  "gaps": [
    {
      "index": 0,
      "start_frame": 30,
      "dur_frames": 120,
      "center_sec": 1.5,
      "srt_text": "...",
      "query": "...",
      "candidates": [
        {"track": "V-Ref-Cand00", "path": "..."},
        {"track": "V-Ref-Cand01", "path": "..."},
        {"track": "V-Ref-Cand02", "path": "..."}
      ]
    }
  ]
}
```

For `candidates == 1` the legacy flat `images` array is preserved for
backward compatibility.

### Validation

- `candidates < 1` → error + exit 2.
- `candidates > 1` without `--fill-gaps` → error + exit 2.
- Resolve treats the parallel tracks as stacked video layers; the
  identical frame ranges read as overlap on a single track but are fine
  on distinct tracks.

## Name detection

`--detect-names` (only meaningful with `--fill-gaps`) tells the planner
to scan each gap's overlapping SRT line(s) for a named entity — person,
place, or organization — and prepend it to the per-gap download query.
This produces portraits / location shots for the actual subject of that
voiceover line, instead of a generic scene.

### Behavior

- Uses Ollama (`--name-provider ollama`, default) with a one-shot JSON
  prompt; falls back to the heuristic named-entity extractor at
  `src.matching/scoring.py:3616` when Ollama is unavailable or returns
  no names.
- The downloaded candidate picture for each gap is biased toward the
  detected name; gaps with no name fall back to the global
  `--subject` + SRT snippet (unchanged from default FILL behavior).
- Detected entities are recorded on each `gaps[*].detected_name` entry
  and surfaced as a top-level `name_detection` block:

```json
{
  "name_detection": {
    "enabled": true,
    "provider": "ollama",
    "providers_used": ["ollama", "heuristic"],
    "detected": [
      {"index": 0, "name": "Ada Lovelace", "provider": "ollama"},
      {"index": 2, "name": "Babbage", "provider": "heuristic"}
    ]
  }
}
```

### When to use

- Biopic / documentary style projects where the named subject drives
  the visual identity of each line.
- Combine with `--candidates N` to get multiple pictures *per named
  entity*, stacked on N tracks for Resolve-side A/B'ing.

## Error Handling

## Error Handling

| Condition | Handling |
| --- | --- |
| Missing SRT / OTIO | error + exit 2. |
| `--output` equals `--otio` | error "refusing to overwrite source OTIO" + exit 2. |
| Source OTIO no video track / zero clips | warn; durations fall back to SRT / uniform. |
| SRT empty or `total_srt_dur <= 0` | error + exit 2. |
| Downloader rc 1 | hard fail + exit 3. |
| Downloader rc 2 | warn; proceed (likely 0 → empty track). |
| `pyimagedl` ImportError | surfaces inside the downloader subprocess (rc≠0). |
| `actual == 0` | warn; write valid OTIO with empty track; exit 0. |
| `0 < actual < N` | warn; re-plan over the actual count. |
| Image below `--min-size-kb` (double-check) | skip file, log. |
| `_has_problematic_path` | skip clip, log (matches `entities.py:537`). |

## Re-running

- If the first run pulled wrong images, refine with `--subject`,
  `--context`, `--sources`, or `--min-size-kb` and re-run.
- The script writes a new OTIO + summary by default; the source is never
  touched. To replace an existing output, pass `--output` explicitly.
- Re-running over a populated `--tmp-dir` is safe — the downloader
  overwrites `<slug>_NN.<ext>` files.
- **Switching subjects leaves orphans.** If you change `--subject`, the
  old subject's `<old_slug>_NN.<ext>` files stay in `--tmp-dir`. The
  post-download safety net will skip any with blacklisted-domain sidecars,
  but otherwise they'll silently sit in the directory unused (the new OTIO
  only references the current slug). Delete them with
  `rm --tmp-dir/old_slug_*` to clean up, or pass `--tmp-dir` to a fresh
  path.
- **The on-screen summary table reflects the post-replan plan.** If the
  download yielded fewer than `--max-images`, the script re-spreads the
  centers over `N = actual` before building the OTIO. The post-build
  table at the end of the run prints that final N rows, not the original
  N. The dry-run table (with `--dry-run`) prints the original N.

## Caveats

- **Identity is not verified.** Like the `/download-pictures` skill,
  results are thumbnails. Always eyeball the saved files (or load the
  new OTIO in Resolve) before trusting the layout.
- **Stock-photo watermarks filtered by default.** Alamy, Shutterstock, and
  Getty domains are blocked at two layers: the downloader rejects candidates
  before download, and `collect_downloaded` re-checks each sidecar JSON
  as a safety net (catches files whose final resolved host revealed Alamy
  even if the candidate URL didn't). Override with
  `--exclude-domains ""` to disable, or extend the list with your own.
- **The duration rule is SRT-first.** An image whose center lands inside
  a 49-second SRT segment becomes a 49-second still. Center-anchored
  placement with right-clamp prevents overlap but may push later clips
  past their ideal centers in dense regions. If you want shorter stills,
  either re-record the SRT or trim durations post-import in Resolve.
- The new OTIO's `V-Ref` track starts at frame 0 (not 1:00:00:00 as
  the matcher-pipeline OTIOs do). DaVinci imports both fine; pick
  whichever is convenient.
- Source OTIO must be a vanilla `otio.schema.Timeline` with at least
  one video track. Exported-from-DaVinci OTIOs with stacks/compounds
  are not yet auto-flattened.
- **SRT must match the project you intend.** If the trimmed SRT describes
  one topic (e.g. Mexican corn harvest) but you're re-narrating with
  different content (e.g. Amish lifestyle), the LLM will faithfully read
  the SRT and pull the wrong images. Either regenerate the trimmed SRT
  first, or override `--subject` to force the new topic.
- **Yields are inconsistent.** Pyimagedl hit rates vary wildly by query
  topic (mixteca-Guajaqueña queries → ~50/50; US-government-PDF-heavy
  queries → ~34/50). The script auto-replans when short, so the OTIO is
  always valid but may have fewer clips than `--max-images`.

## Script Location

`scripts/distribute_ref_pictures.py` — standalone script. Imports limited
to `opentimelineio`, `src.otio.utils` (`_to_windows_path`,
`_has_problematic_path`), `src.llm_client` (`create_client`, `LLMRequest`,
`ResponseFormat`) for subject inference, and inline SRT parsing. All image
fetching is delegated to `scripts/download_reference_pictures.py` via
subprocess.
