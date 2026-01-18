# Claude Code Project Guide

> **LLM Editing Guide:** This file is optimized for token efficiency. When editing:
> - **Rules 1-9**: Add to the table, not as new verbose sections
> - **Rules 10+**: Keep to ~10 lines max with table + 1-2 key points
> - **Session History**: Keep only last 6 entries, archive older to CHANGELOG.md
> - **Never**: Add verbose code examples (put in code comments instead)
> - **Format**: Tables > prose, one-liners > paragraphs

This file helps Claude Code sessions understand project conventions and avoid common mistakes.

## Quick Reference

### CLI Flags

| Flag | Description |
|------|-------------|
| `--voiceover`, `-v` | Path to voiceover file (SRT, MP3, WAV, MP4) |
| `--project`, `-p` | Project directory path |
| `--config`, `-c` | Path to config file |
| `--keywords`, `-k` | Number of keywords to extract |
| `--match-only` | Skip download/transcribe, use cached data |
| `--resume` | Resume from checkpoint |
| `--fresh` | Force fresh start, ignore checkpoint |
| `--force-rematch` | Force rematch all videos |
| `--use-keywords [NAME]` | Use saved keywords (latest or named preset) |
| `--save-keywords [NAME]` | Save keywords as preset |
| `--list-keywords` | List saved keyword presets |
| `--validate-config` | Validate config and exit |
| `--refresh-entities` | Re-download entity images (ignore local cache) |
| `--non-interactive` | Skip prompts, use defaults |
| `--save-matching-fixtures` | Save matching inputs for testing |

### Common Commands

```bash
# Basic run
python main.py --voiceover script.srt --project "E:\Projects\MyDoc"

# Resume interrupted run
python main.py --resume

# Re-run matching only (after config tweaks)
python main.py --match-only

# Use saved keywords
python main.py --use-keywords mypreset
```

### Skill Commands

| Command | Description |
|---------|-------------|
| `/logcheck <project>` | Check project log for errors/warnings and fix them immediately |
| `/match-only <project>` | Re-run matching and output generation (skips download/transcribe) |
| `/newproject <name> <client> <doc_url>` | Create project, download voiceover from Google Doc links, start pipeline |
| `/gdrive-download <url> <output> [--oauth]` | Download file from Google Drive (public or OAuth) |
| `/validate-output [path]` | Validate output **structure** (OTIO/XML/EDL format, track layout) |
| `/ollama-check` | Diagnose Ollama setup, check models, start server with correct path. Use for keyword mode script generation issues. |

```bash
# Check project log and auto-fix issues
/logcheck E:/Edit Job/Stu/January/6__2026-01-15

# Re-run matching after config tweaks
/match-only E:/Edit Job/Project/MyDoc__2026-01-15

# Create new project with Google Doc voiceover links
/newproject 67 Stu https://docs.google.com/document/d/1abc123/edit

# Download from Google Drive (public file, no auth)
/gdrive-download 'https://drive.google.com/open?id=1ABC...' voiceover.mp3

# Download with OAuth (any file you have access to)
/gdrive-download 'my_audio_file.mp3' voiceover.mp3 --oauth

# Validate baseline itself
/validate-output

# Validate specific output folder
/validate-output E:/Edit Job/Project/output/20260115_123456

# Validate project (auto-finds latest output)
/validate-output E:/Edit Job/Project/MyDoc__2026-01-15

# Check Ollama setup for keyword mode
/ollama-check
```

`/logcheck` reads latest run log, detects errors (downloads, API, OTIO), and applies fixes automatically.
`/newproject` creates `E:\Edit Job\{client}\{month}\{name}__{date}`, downloads+combines Drive audio, starts pipeline.
`/match-only` uses cached transcriptions/embeddings - fast iteration on matching settings.
`/gdrive-download` supports URLs, file IDs, or file names; use `--oauth` for private files.
`/validate-output` accepts output paths OR project paths (auto-resolves to latest output).
`/ollama-check` verifies Ollama installation, starts server with `D:\ollama\models` path, lists available models.

## Refactoring Roadmap

**IMPORTANT:** This codebase has a documented refactoring roadmap. Before starting new work:
- Read [REFACTORING.md](REFACTORING.md) to understand ongoing refactoring efforts
- Check if your work intersects with planned refactorings
- Follow established patterns from completed refactors (e.g., LLM client abstraction)

Current status:
- ✅ Pipeline stages architecture (completed)
- ✅ LLM client abstraction (completed - Jan 6, 2026)
- 📋 See REFACTORING.md for full roadmap

When writing new code, prefer using modern abstractions:
- Use `src/llm_client/` for all LLM calls (see Rule 9 below)
- Use stage classes in `src/stages/` for pipeline logic
- Use `PipelineState` for shared data (not scattered instance vars)

## Architecture

### File Structure

| File | Purpose |
|------|---------|
| `main.py` | Entry point, pipeline orchestration |
| `config.yaml` | User-editable settings |
| `src/config.py` | Config dataclasses with defaults |
| `src/state.py` | PipelineState + dataclasses (CANONICAL location) |
| `src/llm_client/` | **Unified LLM abstraction** (Gemini, Anthropic, Ollama) |
| `src/gdrive/` | Google Drive download utilities (OAuth + anonymous) |
| `src/downloader.py` | Video/audio download logic |
| `src/downloader/caption_fetcher.py` | YouTube caption fetching (yt-dlp) |
| `src/matching.py` | Video-to-voiceover matching |
| `src/transcription.py` | Whisper transcription |
| `src/checkpoint.py` | Resume/checkpoint management |
| `src/keyword_extractor.py` | LLM-based keyword extraction |
| `src/entity_images.py` | Entity image search (Google, Bing) |
| `src/entity_cache.py` | Cross-project entity cache |
| `src/location_service.py` | GeoNames geocoding |
| `src/otio_builder.py` | Timeline generation |
| `src/pipeline.py` | PipelineOrchestrator |
| `src/stages/` | Modular stage classes |
| `src/agents/` | **Self-healing pipeline agents** (ResilientRunner, Healers) |
| `setup_project.py` | Creates project folders with run scripts |

### Pipeline Stages

| Stage | Class | Purpose |
|-------|-------|---------|
| ANALYZE | AnalyzeStage | Keywords, topics, entities, location chapters |
| ENTITY_IMAGES | EntityImagesStage | Download entity images (Google, Bing, Pexels) |
| ENTITY_VIDEOS | EntityVideosStage | Download stock videos for entities |
| DOWNLOAD | DownloadStage | YouTube video/audio download |
| STOCK | StockVideoStage | Download generic stock footage (B-roll) |
| BROLL_DOWNLOAD | BrollDownloadStage | Download B-roll with keyword suffixes |
| REMIX | RemixStage | Filter videos by keyword relevance |
| CAPTION | CaptionStage | Fetch YouTube captions (skip Whisper if available) |
| TRANSCRIBE | TranscribeStage | Whisper + embeddings (fallback for uncaptioned) |
| SCENE_DETECTION | SceneDetectionStage | Scene boundaries + face detection for B-roll |
| MATCH | MatchStage | Embedding + LLM matching |
| BROLL_MATCH | BrollMatchStage | Match silent scenes to voiceover (V8) |
| OUTPUT | OutputStage | OTIO, EDL, XML generation |

### OTIO Track Layout

| Track | Purpose | Default State |
|-------|---------|---------------|
| V1 | Primary video | Enabled |
| V2-V3 | Alternatives 1-2 | Disabled |
| V4-V6 | Secondary (diversity-scored, strict different source) | Disabled |
| V7 | Embedding-Diversity strategy | Disabled |
| V8 | B-roll Only (silent footage) | Disabled |
| V9 | Entity Images (Google stills) | Disabled |
| V10 | Stock Videos (Pexels/Pixabay) | Disabled |
| A1-A8 | Corresponding audio | Matches video |

### V8-V10 Track Requirements

**V8 - B-roll Only:**
- Two detection methods (both contribute to V8):
  1. **Face detection** (SceneDetectionStage): `is_broll=True` when face_score < 0.3
  2. **Silent detection** (BrollMatchStage): word_count < `broll.min_words_threshold` (default: 10)
- Config: `broll.enabled: true`, `scene_detection.detect_faces_per_scene: true`
- **BrollMatchStage**: Vision API enrichment (or filename keywords in audio-first mode), multi-strategy scoring
- **Scoring weights**: embedding (0.4) + keyword (0.35) + entity (0.25) + source_boost
- **Troubleshooting**: If V8 empty, check logs for "B-roll: X/Y scenes" or "BrollMatchStage: Detected X silent scenes"

**V9 - Entity Images:**
- Requires: Entity extraction (ANALYZE stage) + EntityImagesStage
- Config: `pipeline.skip_image_search: false`
- Sources: Google Images, Bing Images, Pexels, Pixabay
- **Troubleshooting**: If using `--match-only`, ensure `EntityImagesStage` is in the pipeline (fixed in src/pipeline.py:275)

**V10 - Stock Videos:**
- Requires: Entity extraction + EntityVideosStage + API keys
- Config: `pipeline.skip_image_search: false`, `image_search.use_stock_apis: true`
- Sources: Pexels, Pixabay stock video APIs
- **Troubleshooting**: Same as V9. EntityVideosStage runs independently from YouTube downloads (fixed: removed incorrect `skip_download` check)

### Project Directory Structure

```
ProjectName__2026-01-03/
├── run.bat              # Launcher script
├── project_config.yaml  # Project-specific overrides
├── voiceover/           # User puts audio here
├── output/              # Generated OTIO, EDL, XML
├── checkpoint.json      # Resume state
├── saved_keywords.json  # Keyword presets
└── .cache/              # Transcriptions, embeddings
```

## Caching

### Cache Locations

| Cache | Location | Purpose |
|-------|----------|---------|
| Checkpoint | `checkpoint.json` | Resume state |
| Checkpoint backup | `checkpoint.backup.json` | Corruption recovery |
| Transcriptions | `.cache/transcriptions/` | Whisper output |
| Captions | `.cache/captions/` | YouTube caption files (SRT) |
| Embeddings | `.cache/embeddings/` | Vector embeddings |
| LLM responses | `.cache/llm_responses/` | Cached LLM calls |
| Locations | `.cache/locations/` | GeoNames geocoding |
| Global cache | `~/.matcher_global_cache/` | Cross-project videos |
| Entity cache | `~/.matcher_entity_cache/` | Cross-project images |
| Saved keywords | `saved_keywords.json` | Keyword presets |

### Clear/Refresh

```bash
# Force fresh start (clears checkpoint)
python main.py --fresh

# Re-download entity images
python main.py --refresh-entities

# Delete caches manually (bash/Git Bash)
rm -rf .cache/

# Delete specific caches to force re-transcription (for V8 B-roll fix)
rm -rf .cache/transcriptions
rm -rf .cache/scene_detection
```

**Note for run.bat users**: When using `run.bat`, the working directory is the project folder (e.g., `E:\Projects\MyDoc__2026-01-03\`). Use relative paths:
```bash
# From project directory via run.bat
rm -rf .cache/transcriptions
rm -rf .cache/scene_detection
```

## Google Drive Utilities

### Module: `src/gdrive/`

Downloads voiceover and media files from Google Drive with two strategies:

| Method | Module | Auth | Best For |
|--------|--------|------|----------|
| **Anonymous (gdown)** | `fallback_downloader.py` | None | Public Drive links |
| **OAuth** | `oauth_downloader.py` | Browser | Private files, owned content |
| **Unified** | `skill.py` | Auto-detect | CLI integration |

**Usage in code:**
```python
from src.gdrive import gdown_download, oauth_download, get_drive_service

# Anonymous download
success = gdown_download('https://drive.google.com/open?id=1ABC...', 'output.mp3')

# OAuth download (auto-detects file by ID, URL, or name)
service = get_drive_service()  # Opens browser first time only
success = oauth_download(service, 'my_file.mp3', 'output.mp3')
```

**Skill usage:** `/gdrive-download <url_or_id> <output> [--oauth]`

**Credentials:** OAuth creds stored in `~/.matcher_drive_auth/` (never committed to repo)

## Configuration

### Config Access Pattern

```python
# Safe config access with fallback:
value = getattr(self.download_config, 'field_name', default_value)

# For nested configs that might be dicts:
if isinstance(config_section, dict):
    value = config_section.get('field_name', default)
else:
    value = getattr(config_section, 'field_name', default)
```

### Adding New Config Options

1. Add field to dataclass in `src/config.py`
2. Add `__post_init__` if nested dataclass (see Rule 2 below)
3. Add to `config.yaml` with comment
4. Access in code with `getattr()` fallback
5. Verify: `python -m py_compile src/config.py`

### Project Overrides

Create `project_config.yaml` in project folder:
```yaml
keyword:
  max_keywords: 5
pipeline:
  skip_image_search: true
```

**IMPORTANT**: The project_config.yaml is automatically loaded as overrides when using `--project`. Do NOT pass it to `--config`:
```bash
# CORRECT - project_config.yaml is auto-loaded as overrides
python main.py --project "E:\Projects\MyDoc"

# ALSO CORRECT - explicit --config with base config, project overrides auto-applied
python main.py --project "E:\Projects\MyDoc" --config custom_base.yaml

# HANDLED - system detects this and uses default config as base
python main.py --project "E:\Projects\MyDoc" --config project_config.yaml
# Shows: "⚠ --config points to project_config.yaml, using default config as base"
```

### Self-Healing Configuration

Self-healing is **enabled by default**. The pipeline automatically recovers from common errors.

```yaml
# config.yaml
healing:
  enabled: true              # Disable to use standard pipeline
  strategy: "conservative"   # aggressive, conservative, interactive, minimal
  max_attempts_per_stage: 3  # Heal attempts before failing stage
  max_total_heals: 20        # Total heals before aborting pipeline
  print_report: true         # Print healing summary after run
```

**Strategies:** `aggressive` (5/stage), `conservative` (3/stage, default), `interactive` (3/stage, asks user), `minimal` (1/stage, fail fast)

**Usage:** `run_pipeline_with_healing(config, project_dir)` or see Self-Healing Agents section for manual control.

### Short Path Configuration (E:/v, E:/i)

To avoid Windows path length limits and improve NLE import performance, you can use short root paths:

```yaml
# config.yaml
download:
  root_dir: "E:/v"  # Videos stored in E:/v/ProjectName
  folder_name: "videos"  # Subfolder within root_dir

image_search:
  root_dir: "E:/i"  # Images stored in E:/i/ProjectName
  folder_name: "images"  # Subfolder within root_dir
```

**How it works:**
1. Pipeline validates that `root_dir` exists (or can be created) at startup
2. Creates project subdirectory: `{root_dir}/{ProjectName[:15]}/`
3. Example: `E:/v/MyProject__202/` instead of `E:/Projects/MyProject__2026-01-10/videos/`

**Benefits:**
- ✅ Avoids 260-char Windows path limit
- ✅ Faster NLE imports (shorter file paths)
- ✅ Shared drive organization (all projects in E:/v/)
- ✅ Easier to manually browse/clean up

**Error handling:**
- Invalid path: "download.root_dir must be an absolute path"
- Missing drive: "Cannot create download.root_dir: E:/v - Check drive exists"
- No permissions: "Check you have write permissions"

**Validation:** The pipeline automatically creates base directories on startup and prints confirmation:
```
✓ Videos root directory: E:/v
✓ Images root directory: E:/i
```

**Troubleshooting:**
- Ensure drive letter exists (E:/, D:/, etc.)
- Use forward slashes: `E:/v` not `E:\v`
- Must be absolute path (not relative like `./videos`)
- Run as administrator if permission errors occur

## Development Rules

| Rule | Summary | Key Point |
|------|---------|-----------|
| 1 | Config sync | Update BOTH `src/config.py` AND `config.yaml` |
| 2 | `__post_init__` | Nested dataclass fields load as `dict` - convert in `__post_init__` |
| 3 | DownloadedVideo | Use `file=`, `duration_tier=` NOT `path=`, `tier=` |
| 4 | Regex lookbehind | Python needs fixed-width - use capture groups instead |
| 5 | Live streams | Add `!is_live` to yt-dlp match-filter |
| 6 | Dict/Object config | Handle both: `vc.get()` if dict, `getattr()` if object |
| 7 | Embeddings truthiness | Use `is_embeddings_empty()` - numpy arrays fail bool check |
| 8 | B-roll propagation | SceneDetectionStage → text_metadata → MatchStage restores is_broll |
| 9 | Test non-interactive | Tests invoking main.py MUST use `--non-interactive` to avoid prompt hangs |
| 18 | Healer config updates | Use `getattr`/`setattr` - never replace config object |
| 19 | HealerResult returns | `.fixed()` for retry, `.failed()` to abort |
| 20 | Healer logging | Use `self.log_attempt()` and `self.log_success()` |

### Rule 10: LLM Client
Use `src/llm_client/` for ALL LLM calls. Never directly initialize provider SDKs.

```python
from src.llm_client import create_client, LLMRequest, ResponseFormat
client = create_client("gemini", api_key=api_key, model=model)
response = client.generate(LLMRequest(prompt=prompt, response_format=ResponseFormat.JSON))
```

**Benefits:** Retry, caching, JSON parsing, provider switching (Gemini/Anthropic/Ollama)

### Rule 11: Dataclass Imports
Import dataclasses from canonical locations - never redefine locally.

| Type | Location | Examples |
|------|----------|----------|
| State | `src/state.py` | VoiceoverSegment, DownloadedVideo, AudioDownload, CaptionDownload, Match, PipelineState |
| Config | `src/config.py` | Config, LLMConfig, DownloadConfig |

**Never**: Duplicate dataclass definitions (causes field name mismatches at runtime)

### Rule 12: VAD Filter (CRITICAL - Recurring Bug)
Voiceover and videos need OPPOSITE VAD settings. Config only affects voiceover.

| Context | VAD | File | Why |
|---------|-----|------|-----|
| Voiceover | ON (config) | analyze.py:256 | Gap detection |
| Videos | OFF (hardcoded) | parallel_processor.py:64 | Audio quality varies |

**Symptoms:** "VAD filter removed XX:XX of audio" = VAD wrongly enabled for videos

**Never**: Read VAD from config for video transcription

### Rule 13: XML Media Bin Import (DaVinci Resolve)
DaVinci Resolve media bin XML requires specific structure:

| Requirement | Detail |
|-------------|--------|
| `<bin>` placement | Directly under `<xmeml>` (NO `<project>` wrapper) |
| `<file>` placement | Inside `<clipitem>`, not at clip level |
| Empty `<sequence>` | Required sibling to trigger import |
| Path format | `file:///E:/path/file.mp4` (file:/// prefix required) |
| Audio-only files | Skip them - DaVinci XML import fails on .mp3/.wav |

**File:** `src/otio/xml_export.py:_write_media_xml_part()`

### Rule 14: OTIO Path Compatibility (DaVinci Resolve)
DaVinci Resolve OTIO import hangs on certain path issues:

| Issue | Solution | Files |
|-------|----------|-------|
| Backslashes in paths | Use forward slashes: `E:/v/file.mp4` | `utils.py:_to_windows_path()` |
| Audio-only files (.mp3, .wav) | Skip them, add gap instead | `timeline.py`, `tracks.py` |
| Unicode in paths (ñ, ü, etc.) | Skip clips with non-ASCII chars | `timeline.py`, `tracks.py`, `entities.py` |
| Corrupted unicode (�) | Skip clips with replacement char | All OTIO generation files |

**Helper function:** `_has_problematic_path()` in timeline.py, tracks.py, entities.py

### Rule 15: DaVinci OTIO Caching & Gap Optimization
DaVinci Resolve caches OTIO import state by filename. Failed imports can corrupt this cache.

| Issue | Solution |
|-------|----------|
| Import hangs after fixing issues | Rename OTIO file or use new DaVinci project |
| Many consecutive gaps in track | `optimize_timeline_gaps()` merges them automatically |
| Trailing gaps after last clip | `optimize_track_gaps()` removes them automatically |

**Gap optimization** is applied automatically in `create_timeline()`. Manual: `optimize_timeline_gaps(timeline)` from `src/otio/utils`.

### Rule 16: DaVinci OTIO Duplicate Media Paths
DaVinci Resolve hangs when the same video file is referenced from **multiple different paths**.

| Scenario | Example | Result |
|----------|---------|--------|
| Same file, two paths | `stock/video.mp4` + `broll/video.mp4` | ❌ Hang |
| Same file, one path | `stock/video.mp4` (used 5x) | ✅ OK |

**Root cause:** "Automatically import source clips into media pool" tries to import both paths, causing infinite loop.

**Fix:** `MediaPathNormalizer` in `src/otio/utils.py` deduplicates paths automatically:
- Identifies duplicates by filename + file size
- Prefers `stock/` over `broll/` paths
- Applied in `create_timeline()` before clip creation

**Files:** `src/otio/utils.py` (MediaPathNormalizer), `src/otio/timeline.py` (integration)

### Rule 17: DaVinci OTIO Scale Limit & Auto-Splitting
DaVinci Resolve crashes/hangs when importing OTIO timelines with 10k+ items. Auto-splitting solves this.

| Item Count | Result | Solution |
|------------|--------|----------|
| < 3,000 | ✅ OK | No splitting needed |
| 3,000-10,000 | ⚠️ Slow | Auto-split into 4 parts |
| > 10,000 | ❌ Crash | Must use split parts |

**Config:** `output.auto_split_threshold: 3000`, `output.split_parts: 4`, `output.always_generate_full: true`

**Output:** Generates `timeline_PART1.otio` through `timeline_PART4.otio` (each has ALL tracks, subset of time range).

**OTIO vs XML path handling:**
- OTIO: DaVinci extracts filename only, searches configured directories (media often offline)
- XML `<bin>`: Full `<pathurl>` paths, media links correctly

**Workaround for offline media:** Import XML media bins first → populate Media Pool → then import OTIO timeline (auto-links).

**Files:** `src/otio/export.py` (split_timeline_by_time_ranges), `src/config/sections/output.py`

### Testing Checklist

Compile check: `python -m py_compile main.py src/config.py src/downloader.py src/matching.py`

Also verify: Config in both `config.py` AND `config.yaml`, nested configs have `__post_init__`

### Output Structure Validation

Validates output **structure, integrity, and NLE importability**, NOT content/matches.

```bash
# Validate any output directory
python tests/test_output_comparison.py "E:/path/to/output" -v

# JSON output for CI
python tests/test_output_comparison.py "E:/path/to/output" --json
```

**What It Validates:**

| Category | Checks |
|----------|--------|
| **File Structure** | Required files exist, minimum file counts |
| **Format Integrity** | OTIO parses, XML well-formed, JSON valid |
| **Track Structure** | Required tracks present, naming convention (V#/A#) |
| **Coverage Thresholds** | V1 >= 90%, V2 >= 50%, V3 >= 40%, V8 >= 20% |
| **Clip Integrity** | No zero-duration/overlapping clips, valid speed effects |
| **EDL (CMX3600)** | Valid timecodes, event numbers, edit types, reel names |
| **XML (XMEML)** | DaVinci bin structure, pathurl format, frame rate consistency |
| **Media Paths** | Reserved names, NLE-problematic chars, path length, UNC paths |
| **Cross-File** | Frame rate consistency across OTIO/XML/EDL |
| **Export Capability** | Tests OTIO → EDL/FCPXML export |

**Extending for New Features:**

Edit `STRUCTURE_SPEC` in `tests/test_output_comparison.py`:

```python
STRUCTURE_SPEC = {
    "required_files": [...],           # Add new required files
    "required_tracks": [...],          # Add new required tracks
    "coverage_thresholds": {...},      # Add new track thresholds
    "segment_entry_keys": [...],       # Add new segment keys
}
```

## Key Features

### Audio-First Mode
Downloads audio only, transcribes, matches, then downloads only matched video segments (~95% bandwidth savings).
```yaml
download:
  audio_first:
    enabled: true
    buffer_seconds: 30.0
    merge_gap_seconds: 15.0
```

### Caption-First Mode
Fetches YouTube captions instead of downloading audio + Whisper transcription. Falls back to Whisper for uncaptioned videos.
```yaml
download:
  caption_first:
    enabled: true
    prefer_manual_captions: true  # Manual > auto-generated
    languages: ["en", "en-US", "en-GB"]
    fallback_to_audio: true       # Whisper for uncaptioned
    confidence_boost_manual: 0.1  # Boost for manual captions
```
**Benefits:**
- ✅ Faster (no Whisper transcription for captioned videos)
- ✅ Lower bandwidth (skip audio download)
- ✅ Better accuracy for professional videos with manual captions

**Transcript sources tracked:** `manual_caption`, `auto_caption`, `whisper`, `metadata`

**Note:** Can coexist with audio-first mode (separate configs).

### Location-Aware Matching
Filters video candidates by geographic proximity for travel content.
```yaml
matching:
  location_matching:
    enabled: true
    geonames_username: "your_username"  # Must enable web services at geonames.org
    hard_filter_level: "city"  # city, state, country, continent
```
**Note:** GeoNames web services must be enabled at https://www.geonames.org/login (401 errors = not enabled).

### Pause-Split Segments
```yaml
transcription:
  pause_split:
    enabled: true
    split_at_sentences: true
    split_at_list_markers: true
    split_at_locations: true
```

### Vision API for Silent Videos
Generates semantic descriptions of silent stock footage using Gemini Vision API.
```yaml
vision:
  enabled: true
  provider: gemini
  model: gemini-2.0-flash
  max_scenes_per_video: 50
  estimated_cost_per_call: 0.001
```
**Behavior:**
- **Auto-detection**: SceneDetectionStage identifies silent videos (no transcripts)
- **Vision processing**: Extracts mid-frame from each scene and describes it via Gemini Vision API
- **Embedding improvement**: Visual descriptions enable better semantic matching vs placeholder text `[Silent video: name]`
- **B-roll enhancement**: B-roll scenes get meaningful descriptions for context-aware matching
- **Cost control**: Only processes silent videos, skips videos with transcripts
- **Caching**: Vision responses cached in `.cache/vision_cache/` to avoid redundant API calls
- **Fallback**: If API unavailable (no GEMINI_API_KEY), uses placeholder text

**Setup:**
1. Set `GEMINI_API_KEY` environment variable
2. Enable in config: `vision.enabled: true`
3. Run pipeline normally - vision processing is automatic for silent videos

**Cost Estimate**: ~$0.001 per scene (configurable via `estimated_cost_per_call`)

## Self-Healing Agents

Location: `src/agents/` - Auto-recovery for pipeline errors.

### Quick Start

```python
# Simple: create_resilient_pipeline(config, project_dir) -> (pipeline, runner)
# With orchestrator (recommended): create_orchestrated_pipeline(config, project_dir, strategy=HealingStrategy.aggressive())
#   -> (pipeline, orchestrator, runner)
# Then: runner.run_pipeline(pipeline); orchestrator.print_report()
```

### Healing Strategies

| Strategy | Attempts | Behavior |
|----------|----------|----------|
| `HealingStrategy.aggressive()` | 5/stage, 50 total | Try everything, minimal user interaction |
| `HealingStrategy.conservative()` | 3/stage, 20 total | Safe fixes only, preserve config |
| `HealingStrategy.interactive()` | 3/stage, 30 total | Ask user before major changes |
| `HealingStrategy.minimal()` | 1/stage, 5 total | Fail fast, critical fixes only |

### Healer Registry

| Healer | Detects | Auto-Fix |
|--------|---------|----------|
| `CheckpointHealer` | JSON parse errors, corrupt checkpoint | Restore from backup, rebuild from cache |
| `APIHealer` | Rate limits (429), auth errors, quota | Backoff + retry, switch provider |
| `CaptionHealer` | Caption fetch failures, no captions, parse errors | Fallback to audio, try alt languages |
| `DownloadHealer` | YouTube 429, unavailable videos | Backoff, skip video, try alt format |
| `DiskHealer` | Disk full, permission denied | Clean caches, suggest short paths |
| `PathHealer` | Windows 260 char limit, unicode | Switch to E:/v, sanitize filenames |
| `OTIOHealer` | Timeline generation failures | Fix gaps, resolve paths, simplify |

### OTIOHealer Details

**Clip Timing Fixes:**

| Issue | Detection | Fix |
|-------|-----------|-----|
| Zero/negative duration | `end <= start` | Clamp to MIN_DURATION (0.04s) |
| Excessive duration | `duration > 24h` | Clamp to MAX_DURATION |
| Clip overlaps | `clip[i].end > clip[i+1].start` | Trim earlier clip with 20ms buffer |
| Invalid speed | `time_scalar < 0.1 or > 10` | Clamp to 10%-1000% range |
| Negative start_time | `start < 0` | Set to 0 |
| Gap overflow | Total gaps > available time | Switch gap_mode progressively |

**Media Reference Fixes:**

| Issue | Detection | Fix |
|-------|-----------|-----|
| Missing file | `not Path(video_path).exists()` | Search caches by filename, video_id |
| Path encoding | Unicode chars in path | Sanitize to ASCII, rename file |
| Invalid URL | Backslash/extended path issues | Convert to forward slashes |
| Broken reference | File moved/deleted | Search project/.cache, global cache, E:/v |

**Preflight Check:** Call `healer.preflight_check(state)` before timeline generation to detect issues early.

**Safe Mode:** Last resort applies minimal settings (V1 only, no EDL/XML, gap_mode=none, 30fps).

### HealingOrchestrator

Coordinates all healers with:

| Feature | Description |
|---------|-------------|
| **Preflight checks** | Disk space, API keys, paths, media files before run |
| **Smart healer selection** | Priority ordering, picks best healer for error |
| **Config rollback** | Snapshot before each stage, restore on failure |
| **Cross-healer coordination** | Notifies healers when others make changes |
| **User escalation** | Asks user for critical decisions (interactive mode) |
| **Metrics tracking** | Success rates, time spent, issues found/fixed |

**Manual preflight:** `orchestrator.run_preflight(state)` → list of issues, `orchestrator.fix_preflight_issues(issues, state)` → (fixed, remaining)

### Adding New Healers

Create `src/agents/healers/my_healer.py`, extend `Healer`, set `error_patterns`/`exception_types`, implement `fix()` → `HealerResult`, add to `HEALER_REGISTRY`.

## Git Conventions

### Branch Strategy

**Main branch protection:**
- Keep `main` stable and production-ready
- Never push major refactors directly to main
- Use feature branches for all significant changes

**Branch naming:**
- Feature branches: `feature/descriptive-name`
- Bug fixes: `fix/issue-description`
- Refactors: `refactor/component-name`
- Claude Code branches: `claude/auto-generated-name`

**Workflow:** `git checkout -b feature/name` → commit incrementally → `git push -u origin feature/name` → `gh pr create --base main`

**Commit prefixes:**
- `feat:` - New features
- `fix:` - Bug fixes
- `docs:` - Documentation only
- `refactor:` - Code restructuring without behavior change
- `test:` - Test additions/modifications
- `chore:` - Build/config changes

## Session History

| Date | Changes |
|------|---------|
| 2026-01-19 | Keyword mode planning: `docs/plans/PLAN_KEYWORD_MODE.md` - pipeline without voiceover using Ollama for script synthesis |
| 2026-01-19 | `/ollama-check` skill: Diagnose Ollama setup, verify models path `D:\ollama\models`, start server |
| 2026-01-18 | OTIO auto-splitting (Rule 17): Split large timelines into 4 parts when items > 3000 to prevent DaVinci crash |
| 2026-01-16 | Caption-first mode: `CaptionStage` fetches YouTube captions via yt-dlp, skips Whisper for captioned videos |
| 2026-01-16 | Duplicate media paths fix (Rule 16): `MediaPathNormalizer` deduplicates same files in different folders |
| 2026-01-16 | DaVinci caching fix (Rule 14): Rename OTIO files after fixing issues to bypass corrupted cache |
*Older entries archived to [CHANGELOG.md](CHANGELOG.md#session-history-archive)*
