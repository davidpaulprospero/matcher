---
name: lower-thirds
description: Generate clean, professional lower-third graphics from a voiceover SRT using local Ollama (entity classification) and Pillow/Playwright/FFmpeg (rendering). Python pre-screens the SRT for entity candidates (proper nouns, dates, percentages, currency, places) so Ollama only classifies/filters — timing is owned by Python and the LLM cannot hallucinate SRT indices. Supports per-category styling — names, dates, and key points can each use a distinct template and live on their own OTIO track. Auto-loads chapter titles from the project's chapter_detection output (<project>/checkpoint.json → chapter_data.chapters) and emits them on V5. Emits a separate OTIO file (lowerthirds.otio) with one V track per entity category (Names / Dates / Key Points / Places / Chapters). Free, local — no paid software. Use when user says "make lower thirds", "name graphics from SRT", "speaker titles", or wants broadcast-style name/title overlays.
allowed-tools:
  - Read
  - Write
  - Edit
  - Glob
  - Grep
  - Bash
  - Agent
---

# Lower-Thirds Skill

Generate **lower-third graphics** (the name/title bar you see on news, docs, and corporate video) from a voiceover SRT using **local Ollama** for entity extraction and **Pillow** for PNG rendering. Emits a **separate OTIO file** (`lowerthirds.otio`) containing **one V track per category** (Names / Dates / Key Points / Places) so DaVinci Resolve can import the graphics as still-image clips with `FreezeFrame` — no editing inside Resolve required, and each category can be soloed/muted independently.

Extracts four entity types — **people**, **places**, **important dates**, **short info callouts** — and an optional fifth: **chapter titles** from the project's `chapter_detection` output. Renders each using one of six named templates (`classic`, `minimal`, `boxed`, `modern`, `corner`, `newsroom`). The four on-screen categories (names / dates / key points / chapters) can each use a **distinct template** and live on their **own V track** in the output OTIO, so the editor can solo/mute each category independently in Resolve.

This is the **free, local** alternative to After Effects / CapCut Pro for adding on-screen name graphics.

## When to Use

- User says "make lower thirds", "name graphics from SRT", "speaker titles", "add name overlays"
- User has a project's voiceover SRT and wants broadcast-style name/title cards
- User wants DaVinci Resolve to show "John Smith — Mayor" overlays aligned to the voiceover
- User explicitly does NOT want to use paid software (After Effects, CapCut Pro)

## When NOT to Use

- Project has no SRT (use shot-otio first to transcribe)
- User wants full-screen title cards (use imagen-otio with a wider prompt)
- User wants burned-in output MP4 (this skill produces editable .otio, not a baked video)

## Invocation

```
/lower-thirds <project_or_srt> [--ollama-model llama3.2] [--output-otio <path>] [--dry-run]
```

### Examples

```
# Standard run on a project dir
/lower-thirds "E:\Edit Job\Degold\DeepSeaReports\3dWWwtJc"

# Point at a specific SRT file
/lower-thirds "E:\Edit Job\client\project\voiceover\voiceover_trimmed.srt"

# Preview extracted entities without writing any files
/lower-thirds "E:\Edit Job\client\project" --dry-run

# Custom resolution and font
/lower-thirds "E:\Edit Job\client\project" --resolution 1920x1080 --font C:/Windows/Fonts/segoeuib.ttf

# Use a different Ollama model
/lower-thirds "E:\Edit Job\client\project" --ollama-model mistral
```

## Recommended Presets for Clean, Professional Output

The 3 important categories (names, dates, key points) can each use a distinct template **and** live on their own V track. Use `--template-by-type` to set per-category templates; unspecified types (and `place`) fall back to `--template`.

```bash
# Broadcast news (gold-accent names, boxed dates, pill callouts)
python scripts/lower_thirds.py "<project>" \
  --template-by-type "person=classic,date=boxed,info=modern"

# Documentary (newsroom 3-line chapter titles + boxed dates)
python scripts/lower_thirds.py "<project>" \
  --template-by-type "person=classic,date=boxed,info=modern,chapter=newsroom"

# Modern minimalist (clean underline for every category)
python scripts/lower_thirds.py "<project>" \
  --template-by-type "person=minimal,date=minimal,info=minimal,chapter=minimal"

# Newsroom formal (3-line stack names, boxed dates, top-left badge info)
python scripts/lower_thirds.py "<project>" \
  --template-by-type "person=newsroom,date=boxed,info=corner,chapter=newsroom"
```

| Preset              | Names              | Dates            | Key points          | Chapters           | Best for                                  |
|---------------------|--------------------|------------------|---------------------|--------------------|-------------------------------------------|
| Broadcast news      | `classic` (gold bar) | `boxed` (shadow) | `modern` (pill)     | `minimal` (title)  | Documentary, news, corporate              |
| Documentary         | `classic` (gold bar) | `boxed` (shadow) | `modern` (pill)     | `newsroom` (3-line)| Long-form with section titles             |
| Modern minimalist   | `minimal` (underline) | `minimal`     | `minimal`           | `minimal`          | Indie film, vlog, modern brand            |
| Newsroom formal     | `newsroom` (3-line) | `boxed` (shadow) | `corner` (top-left) | `newsroom` (3-line)| Magazine, formal news, archival           |

Output structure: one V track per category, in fixed order — Names → Dates → Key Points → Places. Empty categories are omitted (no placeholder tracks). Per-category tracks let the editor solo or mute any category in Resolve independently.

## Arguments

- `<project_or_srt>` (required): Path to a project directory (will auto-discover `voiceover/*_trimmed.srt` or `voiceover.srt`) or to a specific `.srt` file.
- `--srt <path>` (optional): Override SRT auto-discovery.
- `--ollama-model <name>` (optional): Ollama model name. Default: `llama3.2`.
- `--ollama-host <url>` (optional): Ollama server URL. Default: `http://localhost:11434`.
- `--output-otio <path>` (optional): Output OTIO path. Default: `<project>/lowerthirds.otio`.
- `--output-dir <path>` (optional): PNG output directory. Default: `<project>/lowerthirds/`.
- `--resolution <WxH>` (optional): Canvas size. Default: `1920x1080`.
- `--font <path>` (optional): TTF font path. Default: auto-detect arial.
- `--duration-min <sec>` (optional): Minimum on-screen seconds per entity. Default: `2.0`.
- `--template <name>` (optional): Visual template style. One of `classic`, `minimal`, `boxed`, `modern`, `corner`, `newsroom`. Default: `classic`. Acts as the fallback when `--template-by-type` doesn't cover a category.
- `--template-by-type <spec>` (optional): Per-category template override. Format: `person=classic,date=boxed,info=modern,chapter=newsroom`. Aliases like `name`, `location`, and `section` resolve the same way as `_normalize_entity_type`. Unspecified types (and `place`) fall back to `--template`. Unknown entity types or template names are logged + skipped (no fail). Empty string = no overrides. Default: empty.
- `--no-chapters` (optional): Skip auto-loading chapter title overlays from `<project>/checkpoint.json → chapter_data.chapters`. By default chapters are auto-loaded when the checkpoint has them (the on-screen category appears on `V5 - Lower Thirds · Chapters`). Use `--no-chapters` if you've already published the project and don't want chapter overlays.
- `--no-pre-screening` (optional): Disable Python pre-screening. Sends the full SRT to Ollama (legacy mode, prone to hallucinated names on long transcripts). Default: pre-screening on — Python extracts entity candidates first, Ollama only classifies/filters them.
- `--entity-types <list>` (optional): Comma-separated entity types to include. Choices: `person`, `place`, `date`, `info`. Default: `person,place,date,info` (all four).
- `--mode <image|video>` (optional): Render mode. `video` = animated `.mov` (default, requires Playwright + FFmpeg). `image` = static `.png` (Pillow only).
- `--animation-style <slide|pop|fade>` (optional): CSS animation style for video mode. Default: `slide`.
- `--animation-duration <sec>` (optional): Animation length before the lower-third holds. Default: `0.4`.
- `--fps <n>` (optional): Frame rate for video mode. Default: `30`.
- `--first-minute-only` (optional): Render only entities that anchor in the first 60 seconds of the SRT. Output paths change to `<project>/first_minute_lowerthirds.otio` and `<project>/first_minute_lowerthirds/` so the file is distinguishable from a full run (which writes `lowerthirds.otio` + `lowerthirds/`). Use this when you only need the news-hook opener graphics (e.g., for a teaser/promo cut) without the full 12–20 minute timeline of overlays.
- `--dry-run` (optional): Extract entities and print a preview table; do not write PNGs or OTIO.
- `--audit-first-minute` (optional): Audit first-minute lower-third coverage (anchored, misplaced first occurrences, missing news-hook, missing stats, coverage_score 0–1). Print report and write `<project>/first_minute_coverage_audit.json`. Exits before any rendering. Useful as a pre-render sanity check on broadcast/news edits. The audit lives at project root, distinct from the main run's `lowerthirds.otio` + `lowerthirds/entities.json` outputs.

## Templates

Six visual templates are bundled. All work in both image (`.png`) and video (`.mov`) modes.

| Name      | Style                                                  | Best category         | Best for                       |
|-----------|--------------------------------------------------------|-----------------------|--------------------------------|
| `classic` | Bottom bar, gold accent stripe (current default)       | `person` (names)      | Broadcast news, documentaries  |
| `minimal` | Transparent bg, thin gold underline below name         | any (clean default)   | Modern corporate, indie        |
| `boxed`   | Thicker bottom bar with rounded corners + soft shadow  | `date`                | Documentary, journalism        |
| `modern`  | Centered rounded pill at bottom with drop shadow       | `info` (key point)    | Streaming, podcast, callouts   |
| `corner`  | Top-left badge with red accent square                  | `info` (or `place`)   | Talk show, interview           |
| `newsroom`| Light bar with red frame, 3-line stack (name/role/org) | `person` (formal)     | News magazine, archival        |

Use `role: "Mayor: City Council"` to populate the org line for `newsroom`. Other templates ignore the `: org` suffix.

```
# Default (classic) — backwards compatible
python scripts/lower_thirds.py <project>

# Try each visual style
python scripts/lower_thirds.py <project> --template minimal
python scripts/lower_thirds.py <project> --template boxed
python scripts/lower_thirds.py <project> --template modern
python scripts/lower_thirds.py <project> --template corner
python scripts/lower_thirds.py <project> --template newsroom
```

## Output Location

- **OTIO**: `<project_dir>/lowerthirds.otio` (separate file, not merged into main project timeline)
- **PNGs**: `<project_dir>/lowerthirds/lowerthird_NNN.png` (transparent RGBA, 1920×1080 default)
- **Summary JSON**: `<project_dir>/lowerthirds/entities.json`

The OTIO contains one V track per category, in fixed order — `V1 - Lower Thirds · Names` (PERSON), `V2 - Lower Thirds · Dates` (DATE), `V3 - Lower Thirds · Key Points` (INFO), `V4 - Lower Thirds · Places` (PLACE), `V5 - Lower Thirds · Chapters` (CHAPTER, only when `<project>/checkpoint.json → chapter_data.chapters` is populated). Empty categories are omitted (no placeholder tracks). Each Clip has a `FreezeFrame` effect, so DaVinci holds the PNG for the full entity duration.

## Workflow

### Phase 1: Discover SRT

```python
from pathlib import Path
from scripts.lower_thirds import find_srt_path

# If --srt not given, find the best SRT
srt_path = find_srt_path(project_dir)
# Prefers voiceover/voiceover_trimmed.srt over voiceover/voiceover.srt
```

### Phase 2: Parse SRT

```python
from src.utils import parse_srt_file
segments = parse_srt_file(str(srt_path))
# List[SRTSegment] with fields: index, start_time, end_time, text
```

### Phase 3: Extract Entities via Ollama (with Python pre-screening)

The pipeline uses a **two-stage** approach so Ollama cannot hallucinate timings or names:

1. **Python pre-screening** (`extract_candidates`) — regex/heuristics scan the SRT text and produce candidates, each tagged with its `srt_index` and a `hint`:
   - **person** — `Mr/Dr/Senator + name`, multi-word proper nouns that aren't sentence-starters
   - **place** — preposition + capitalized (`in Paris`), place suffixes (`-shire`, `-burg`, `-ville`, `-land`, `-town`)
   - **date** — 4-digit years (1800-2099), decades (`1990s`)
   - **info** — percentages, currency (`$50,000`), number + unit (`5 miles`, `10 years`)
2. **Ollama classification** — sees ONLY the candidate list (numbered, with each candidate's surrounding context). The prompt asks the model to decide `include`/`exclude`, pick the `entity_type`, and provide a `role`/tag. The model never sees SRT indices — Python owns where each candidate lives.

**Output schema expected from the model:**
```json
[
  {"idx": 1, "include": true, "entity_type": "person", "role": "Mayor"},
  {"idx": 2, "include": false},
  {"idx": 3, "include": true, "entity_type": "info", "role": "2010 census"}
]
```

Each confirmed candidate becomes an `EntitySpan` with timing derived from Python's pre-computed `srt_index`. If the model emits an invalid `entity_type` value the entry is dropped. If the model omits `entity_type` Python's hint is used. If the model emits `entity_type=person` for a date-like candidate, the model's call wins (Python's hint is just a starting point).

**Edge cases the parser handles:**
- Multiple JSON arrays concatenated (small models sometimes emit one per candidate)
- Object literals wrapped as strings inside the array
- `include` as `"true"`/`"yes"` strings or bools
- Truncated JSON: salvaged via `_extract_truncated_array_objects`
- Unknown `idx` values: silently skipped
- Empty/missing candidates: pre-screening returns `[]` and the pipeline falls back to the legacy full-SRT prompt

Legacy `--no-pre-screening` mode: sends the full SRT to Ollama and lets the model emit names; Python searches the SRT to verify each one before timing. Brittle on long transcripts (>300 segments). Use only for short SRTs where pre-screening fails.

The Ollama provider does **NOT** forward `system_prompt` separately — always concatenate. Duplicate names (case-insensitive, leading-article insensitive, type-agnostic) are dropped automatically; the earliest occurrence wins, later copies are silently dropped.

```python
from src.llm_client.providers.ollama import OllamaClient
from src.llm_client.base import LLMRequest, ResponseFormat
from scripts.lower_thirds import build_ollama_prompt, parse_ollama_entities

system_prompt, user_prompt = build_ollama_prompt(segments)
client = OllamaClient(model="llama3.2", host="http://localhost:11434")
request = LLMRequest(
    prompt=system_prompt + "\n\n" + user_prompt,
    max_tokens=2048,
    temperature=0.1,
    cache_key_prefix="lowerthirds_entities",
)
response = client.generate(request)
entities = parse_ollama_entities(response.text, segments)
```

**Output schema expected from the model:**
```json
[
  {"name": "Alice Smith",   "role": "Mayor",         "entity_type": "person", "srt_indices": [3, 4]},
  {"name": "Pearl Harbor",  "role": "Hawaii, USA",   "entity_type": "place",  "srt_indices": [7]},
  {"name": "1869",          "role": "",              "entity_type": "date",   "srt_indices": [2]},
  {"name": "99% solar",     "role": "2010 census",   "entity_type": "info",   "srt_indices": [12]}
]
```

**Entity types and per-type guidance:**
| Type      | What goes here                                                                  | Source                                |
|-----------|----------------------------------------------------------------------------------|---------------------------------------|
| `person`  | Real people the narrator names; `role` = title or affiliation                    | Ollama extraction                     |
| `place`   | Proper-noun locations (city, landmark, building, country); `role` = optional region | Ollama extraction                 |
| `date`    | A specific year or named historical date the narrator emphasizes; `role` = empty | Ollama extraction                     |
| `info`    | A short on-screen fact callout (stat / number / surprising claim), ≤ 8 words; `role` = optional source/unit | Ollama extraction         |
| `chapter` | Chapter / section title (e.g., "Project Genesis and Opening"); `role` = topics or location | Auto-loaded from `<project>/checkpoint.json → chapter_data.chapters` (`--no-chapters` to disable) |

**Edge cases the parser handles:**
- ```` ```json ... ``` ```` fences
- Surrounding prose ("Here is the result: [...]")
- String-typed `srt_indices: "3"` (coerced to `[3]`)
- Out-of-range SRT indices (clamped/dropped)
- Empty names (dropped)
- Missing fields (skipped)
- Missing `entity_type` → defaults to `person` (back-compat for legacy responses)
- Aliases (`"people"`, `"location"`, `"year"`, `"fact"`, ...) → mapped to canonical type
- Alias `"type"` accepted in place of `"entity_type"`
- Unknown `entity_type` value → entry dropped with warning

### Phase 4: Render PNGs / Render Video

For each entity, render a transparent RGBA lower-third at 1920×1080 (configurable). Each entity picks its template from `--template-by-type` (falling back to `--template` for unspecified types and `place`):

```python
from scripts.lower_thirds import render_lower_third_png, LowerThirdTemplate, EntitySpan

def pick_template(entity, template_overrides, default_name):
    name = template_overrides.get(entity.entity_type, default_name)
    return TEMPLATE_REGISTRY[name]

default_template = pick_template(entities[0], {}, "classic")
for idx, entity in enumerate(entities, start=1):
    template = pick_template(entity, template_overrides, "classic")
    png_path = output_dir / f"lowerthird_{idx:03d}.png"
    render_lower_third_png(entity, png_path, template)
```

**Default template (LowerThirdTemplate dataclass):**
- Canvas: 1920×1080 RGBA
- Bar: 180px tall, full-width, semi-transparent black `(0,0,0,200)`, near bottom
- Accent stripe: 8px gold `(255,200,40)` flush against left edge of bar
- Name: 64px white on top line
- Role: 36px light gray `(220,220,220)` below name
- Truncates with ellipsis if name exceeds bar width
- Falls back to PIL default font if arial.ttf is missing

### Phase 5: Build OTIO Timeline

Standalone timeline with one V track per category — mirrors `scripts/shot_otio.py:273` (`build_v12_otio`) but emits **multiple V tracks** (one per `EntityType`) sourced from the lower-thirds directory:

```python
from scripts.lower_thirds import build_lower_thirds_otio

template_names = [pick_template_name(e, overrides, "classic") for e in entities]
timeline = build_lower_thirds_otio(entities, png_paths, template_names, output_otio_path)
# timeline.name = "lowerthirds_timeline"
# timeline.tracks:
#   "V1 - Lower Thirds · Names"      (PERSON)
#   "V2 - Lower Thirds · Dates"      (DATE)
#   "V3 - Lower Thirds · Key Points" (INFO)
#   "V4 - Lower Thirds · Places"     (PLACE)
#   "V5 - Lower Thirds · Chapters"   (CHAPTER — only if checkpoint has chapters)
# Empty categories are omitted entirely.
# Each clip: FreezeFrame, ExternalReference (forward-slash Windows path)
# Each clip's metadata.entity.template records the per-type template used.
# Gaps fill the spaces between non-contiguous entities WITHIN each track.
```

**Critical:** OTIO `target_url` uses **forward slashes** (`E:/path/to/file.png`). Use `src.otio.utils._to_windows_path()` — DaVinci Resolve can hang on backslash OTIO imports on Windows.

### Phase 6: Write Outputs + Summary

- `lowerthirds.otio` — the importable file (multi-track, see Phase 5)
- `lowerthirds/lowerthird_NNN.png` — RGBA assets
- `lowerthirds/entities.json` — list of `{idx, name, role, start_time, end_time, srt_indices, entity_type, template, png_file}` plus a top-level `tracks` field `{ "V1 - Lower Thirds · Names": N, ... }`

## First-Minute Coverage Audit

The first 60 seconds of a news/documentary video is the news hook — the dense opener where the narrator drops the date, location, brands, and big stats. This skill ships an audit function that checks whether the extracted entities actually cover that opener.

```bash
# Pre-render sanity check — exits before writing any PNG/MOV/OTIO
python scripts/lower_thirds.py "<project>" --audit-first-minute
```

What it reports:
- **`anchored_in_window`** — entities placed before t=60s (the chapter titles typically land here).
- **`first_occurrence_misplaced`** — entities the narrator mentions in the opener but that were emitted at a later SRT instead. Example: `2026` first appears at SRT 1 (0.00s) but the LLM anchored it at SRT 167 (8.5 minutes in).
- **`missing_news_hook`** — date/place/brand phrases the narrator says in the opener that didn't get extracted at all (e.g., `Zagreb`, `Croatia`, `Uber`, `pony.ai`).
- **`missing_stats`** — large numeric callouts the narrator mentions but that weren't extracted (e.g., `2,000 robotaxies`, `4,000 vehicles`). Voiceover `2 ,000` (space-comma) patterns are handled.
- **`coverage_score`** — 0.0–1.0 weighted blend (anchored count, no misplacement, no missing hooks/stats, chapter present).
- **`recommendation`** — one-line verdict (`solid`, `OK but has gaps`, `weak`, `poor`).

The function is also importable:

```python
from lower_thirds import audit_first_minute
report = audit_first_minute(entities, segments, opening_s=60)
# report["coverage_score"], report["first_occurrence_misplaced"], ...
```

When `--audit-first-minute` is used, the report is also persisted to `<project>/first_minute_coverage_audit.json` for diffing across runs.

## Verification

After running:

```bash
# 1. Confirm files exist
ls lowerthirds.otio
ls lowerthirds/lowerthird_*.png

# 2. Round-trip the OTIO — multi-track structure
python -c "
import opentimelineio as otio
t = otio.adapters.read_from_file('lowerthirds.otio')
print('tracks:', [tr.name for tr in t.tracks])
for tr in t.tracks:
    print(f'  {tr.name}: {len(tr)} children')
"

# 3. Confirm entity types AND per-type templates round-tripped into clip metadata
python -c "
import opentimelineio as otio
t = otio.adapters.read_from_file('lowerthirds.otio')
for tr in t.tracks:
    for c in tr:
        if hasattr(c, 'metadata') and c.metadata.get('entity'):
            e = c.metadata['entity']
            print(f\"{tr.name:32s}  {e.get('name','')[:20]:20s}  template={e.get('template','?'):9s}  type={e.get('entity_type','?')}\")
"

# and (entities.json includes the same per-entity template + a per-track count):
python -c "
import json
d = json.load(open('lowerthirds/entities.json'))
for e in d['entities']:
    print(f\"{e.get('entity_type','?'):7s} {e.get('name','')[:20]:20s}  template={e.get('template','?'):9s}\")
print('---')
print('tracks:', d.get('tracks', {}))
"

# 4. Visual check
#    Open 3 .mov / .png files (one per category) in an image viewer:
#    - transparent background
#    - visual matches the chosen --template-by-type value per category
#      (e.g. classic gold bar for person, boxed shadow for date, pill for info)
#    - name + (optional) role text visible

# 5. DaVinci import test
#    In Resolve, drag lowerthirds.otio into Media Pool, drop on a timeline.
#    Verify all 4 (or fewer) V tracks appear with the expected clip counts.
```

**Visual differentiation:** Per-category templates (when set via `--template-by-type`) make names, dates, and key points look visually distinct — gold-accent bar for names, boxed shadow for dates, centered pill for key points, etc. The entity_type is preserved in OTIO clip metadata (`entity.entity_type`) for downstream filtering in Resolve.

## Prerequisites

- **Ollama running locally** with `llama3.2` (or chosen model) pulled:
  ```bash
  ollama serve           # in another terminal
  ollama pull llama3.2
  ```
- **Pillow** >= 9.0.0 (already in `requirements.txt:75`)
- **OpenTimelineIO** (already a core dep)
- A voiceover SRT file (use `voiceover_trimmed.srt` if available — see `find_srt_path`)

## Key Files

| File | Purpose |
|------|---------|
| `scripts/lower_thirds.py` | Main script (argparse, no main.py changes) |
| `src/utils.py:993` | `parse_srt_file()`, returns `List[SRTSegment]` |
| `src/utils.py:290` | `SRTSegment` dataclass (index, start_time, end_time, text) |
| `src/llm_client/providers/ollama.py` | `OllamaClient`, `check_ollama_model_available` |
| `src/llm_client/base.py` | `LLMRequest`, `ResponseFormat` |
| `src/otio/utils.py:113` | `_to_windows_path()` (use forward slashes) |
| `scripts/title_color_video.py:33` | `find_srt_path()` reference (copy of trimmed-prefer logic) |
| `scripts/shot_otio.py:273` | `build_v12_otio()` reference for ExternalReference+FreezeFrame pattern |

## Common Issues

- **"Ollama not running"**: Run `ollama serve` in a terminal, then `ollama pull llama3.2`.
- **"No entities extracted"**: Pre-screening found 0 candidates AND the fallback prompt also returned nothing. Some narrations don't surface any people/places/dates/info. Try a larger model (`--ollama-model mistral:7b` or `--ollama-model granite3.2:8b`), or filter to only specific types (`--entity-types person` if your narration has lots of irrelevant place hits). If pre-screening is the bottleneck (very sparse narration), pass `--no-pre-screening` to fall back to the legacy full-SRT prompt.
- **"All entities dropped to unknown type"**: The model emitted an `entity_type` that isn't canonical and isn't an alias. Check `lowerthirds/entities.json` is missing → expect a warning in the log naming the dropped entry. Either update `_ENTITY_TYPE_ALIASES` in `scripts/lower_thirds.py:74` or accept the drop.
- **PNG looks opaque black**: Alpha was dropped somewhere. The script always saves with `mode="RGBA"` — re-render and confirm the file is 8-bit/channel.
- **DaVinci shows broken links**: Path has special characters or backslashes. Confirm `target_url` uses forward slashes via `_to_windows_path()`.
- **Track has wrong duration in Resolve**: Check that `available_range.duration == source_range.duration` (both set to the same value in `build_lower_thirds_otio`).
- **External script reads `timeline.tracks[0]` only**: This worked when `lowerthirds.otio` had one V1 track. It now has up to 5 tracks (Names / Dates / Key Points / Places / Chapters); iterate all tracks instead. Tracks with zero entities in a category are omitted entirely, so track positions vary per project. The Chapters track (`V5`) is also conditional on the project's `chapter_detection` output — use `--no-chapters` to suppress it.

## Differences from `shot-otio` and `imagen-otio`

| Feature | `shot-otio` | `imagen-otio` | `lower-thirds` |
|---|---|---|---|
| Input | Video (auto-transcribes) | SRT or video | SRT only |
| LLM cost | Gemini Flash (paid) | Gemini Flash (paid) | Ollama (free, local) |
| Output track | V12 Generated Images | V12 Generated Images | V1-V5 Lower Thirds (one per category; V5 only if chapters exist) |
| Asset type | AI still (Gemini image gen) | AI still (Imagen) | Pillow-rendered lower-third PNG |
| Track intent | Full-frame visual content | Full-frame visual content | Overlay graphics (above V1 in Resolve) |
| Per-type styling | n/a | n/a | Yes — `--template-by-type` (names / dates / key points / chapters can each use a distinct template) |
| Multi-track output | No (single V12 track) | No (single V12 track) | Yes (up to 5 V tracks; empty categories omitted; chapters auto-loaded from checkpoint) |
| Entity types | n/a | n/a | person, place, date, info, chapter |
| Pre-screening | n/a | n/a | Yes — Python regex extracts candidates from SRT; Ollama only classifies. Timing is fully owned by Python (no hallucinations). `--no-pre-screening` to opt out. |
| Chapter source | n/a | n/a | `<project>/checkpoint.json → chapter_data.chapters` (auto-loaded; `--no-chapters` opt-out) |
| Use case | One image per narration line | Batched images per narration | On-screen name / place / date / fact / chapter-title callouts |
