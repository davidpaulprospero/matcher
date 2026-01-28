# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

> **LLM Editing Guide:** Tables > prose, one-liners > paragraphs. Keep rules to ~10 lines max. Session history: last 6 entries only.

## MCP Tools (Use Proactively)

| Tool | When to Use | Example |
|------|-------------|---------|
| `mcp__context7__*` | **Any library/API question** - always check docs first | "How does X work?" → search Context7 |
| `mcp__remotion-documentation__*` | Remotion-specific questions | Video rendering, React compositions |
| `mcp__markitdown__*` | Convert URLs/PDFs to readable markdown | Fetch and parse external docs |

**Rule:** When answering questions about libraries, APIs, or external tools, **search documentation first** using Context7 or the relevant MCP tool before responding from memory. This ensures up-to-date, accurate answers.

## Quick Reference

### CLI Flags

| Flag | Description |
|------|-------------|
| `--voiceover`, `-v` | Path to voiceover file (SRT, MP3, WAV, MP4) |
| `--keywords`, `-k` | Number of keywords to extract (default: from config.yaml) |
| `--project`, `-p` | Project directory path |
| `--config`, `-c` | Path to config file |
| `--match-only` | Skip download/transcribe, use cached data |
| `--output-only` | Regenerate OTIO/EDL/XML only (fastest, no re-matching) |
| `--resume` / `--fresh` | Resume from checkpoint / Force fresh start |
| `--force-rematch` | Force rematch all videos, ignoring cached matches |
| `--non-interactive` | Skip prompts, use defaults |
| `--use-keywords [PRESET]` | Use saved keywords ("latest" or preset name) |
| `--save-keywords [NAME]` | Save extracted keywords as preset |
| `--list-keywords` | List saved keyword presets and exit |
| `--validate-config` | Validate config file and exit |
| `--refresh-entities` | Force re-download entity images (ignore cache) |
| `--export-metrics PATH` | Export rate limit metrics to JSON after pipeline |
| `--caption-first` | Enable caption-first mode (fetch YouTube captions before download) |
| `--caption-language CODE` | Preferred caption language (ISO 639-1, e.g., "en", "es") |
| `--no-caption-fallback` | Disable Whisper fallback when captions unavailable |
| `--validate-captions` | Validate caption config without running pipeline |
| `--test-fetch N` | Test fetch captions for N sample videos (use with `--validate-captions`) |
| `--export-caption-metrics PATH` | Export caption metrics to JSON after run |
| `--cleanup-caption-cache` | Remove stale caption cache entries |
| `--cleanup-caption-cache-days DAYS` | Override max_cache_age_days for cleanup |
| `--cleanup-caption-cache-dry-run` | Preview cleanup without deleting |

### Common Commands

```bash
python main.py --voiceover script.srt --project "E:\Projects\MyDoc"
python main.py --match-only                    # Re-run matching only
python main.py --output-only                   # Regenerate OTIO only (fastest, needs checkpoint)
python main.py --voiceover script.srt --caption-first  # Use YouTube captions instead of Whisper
python main.py --save-keywords mypreset        # Save keywords after extraction
python main.py --use-keywords mypreset         # Reuse saved keywords
python main.py --force-rematch                 # Ignore cached matches, re-match all

# Standalone OTIO regeneration (bypasses checkpoint - works even when corrupted)
python scripts/regenerate_otio.py "E:\Edit Job\client\project"
python scripts/regenerate_otio.py "E:\Edit Job\client\project" --output-folder 20260124_052654
```

### Skill Commands

| Command | Description |
|---------|-------------|
| `/logcheck <project>` | Check log for errors, auto-fix |
| `/match-only <project>` | Re-run matching (skips download/transcribe) |
| `/newproject <name> <client> <doc_url>` | Create project from Google Doc links |
| `/validate-output [path]` | Validate output structure |
| `/ollama-check` | Diagnose Ollama setup for keyword mode |
| `/watch <project>` | Monitor pipeline progress (5-min intervals, maintains `PIPELINE_STATUS.md`) |
| `/research <topic>` | Research a topic using Perplexity AI |
| `/import-feedback <project> [csv]` | Import DaVinci Resolve marker feedback |

**Use `/research` proactively** for API docs, library usage, error debugging, best practices, or any unfamiliar topic. Don't guess—research first.

**Pipeline monitoring:** Don't ask "is the pipeline progressing?" - use `/watch` or check logs directly. Proactively monitor when user shares pipeline output. The `/watch` skill maintains `PIPELINE_STATUS.md` in the project folder with current status, problems being investigated, and solutions in progress.

**PIPELINE_STATUS.md maintenance:** When investigating/fixing pipeline issues, ALWAYS update `PIPELINE_STATUS.md` in the project folder with:
- Current stage and progress
- Problems found and root causes
- Fixes applied (with code snippets)
- Remaining issues to investigate
- Recent activity log

**Output ≠ Success:** Pipeline completing with output files doesn't mean quality is acceptable. Always verify: match counts, confidence scores, video variety, empty source_file warnings. Low segment downloads or many gaps = investigate root cause.

**Checkpoint file:** Located at `<project>/checkpoint.json` (NOT `.matcher_checkpoint.json`). To force a stage to re-run:
```python
import json
cp_path = r'E:/Edit Job/client/project/checkpoint.json'
with open(cp_path, 'r') as f: cp = json.load(f)
cp['last_completed_stage'] = 'STAGE_BEFORE_TARGET'  # e.g., 'ITERATIVE_MATCH' to re-run DOWNLOAD_SEGMENTS
with open(cp_path, 'w') as f: json.dump(cp, f, indent=2)
```
Then run with `--resume`. Stage order: ANALYZE → ENTITY_IMAGES → ENTITY_VIDEOS → VIDEO_METADATA → CAPTION → DOWNLOAD → STOCK → BROLL_DOWNLOAD → REMIX → TRANSCRIBE → PREMISE → SCENE_DETECTION → MATCH → BROLL_MATCH → ITERATIVE_MATCH → DOWNLOAD_SEGMENTS → OUTPUT

## Refactoring Roadmap

**IMPORTANT:** This codebase has a documented refactoring roadmap. Before starting new work:
- Read [REFACTORING.md](REFACTORING.md) to understand ongoing refactoring efforts
- Check if your work intersects with planned refactorings
- Follow established patterns from completed refactors (e.g., LLM client abstraction)

Current status:
- ✅ All 12 major refactorings completed (~17,200+ lines refactored into modular packages)
- ✅ See REFACTORING.md for details (note: some "Proposed Solution" sections are stale - the work is done)

When writing new code, prefer using modern abstractions:
- Use `src/llm_client/` for all LLM calls (see Rule 9 below)
- Use stage classes in `src/stages/` for pipeline logic
- Use `PipelineState` for shared data (not scattered instance vars)

## Architecture

### Setup

```bash
# Python 3.9+ required, FFmpeg required (for video/audio processing)
pip install -r requirements.txt      # Runtime dependencies
pip install -r requirements-dev.txt  # Test/dev dependencies (pytest, pester, etc.)
```

### File Structure

**Entry points:**

| File | Purpose |
|------|---------|
| `main.py` | Entry point, pipeline orchestration |
| `config.yaml` | User-editable settings |
| `setup_project.py` | Creates project folders with run scripts |

**Core packages** (all under `src/`):

| Package | Purpose |
|---------|---------|
| `stages/` | 14 modular pipeline stage classes |
| `config/` | Config dataclasses split by section (`sections/download.py`, `sections/keywords.py`, etc.) |
| `cli/` | CLI arg parsing, config loading, project config merge |
| `llm_client/` | **Unified LLM abstraction** (Gemini, Anthropic, Ollama) |
| `downloader/` | YouTube download: `core.py`, `audio_first.py`, `impersonation.py`, `escalation_manager.py`, `cookie_rotator.py`, `cookie_method_fallback.py`, `rate_limit_budget.py` |
| `matching/` | Tiered video-to-voiceover matching (embedding + LLM strategies) |
| `keyword_extractor/` | LLM-based keyword extraction with validation |
| `otio/` | OTIO/EDL/XML timeline generation, track builders |
| `media_sources/` | Entity image/video search (Google, Bing, Pexels, Pixabay) |
| `transcription/` | Whisper transcription + embeddings |
| `agents/` | **Self-healing pipeline agents** (ResilientRunner, Healers) |
| `cache/` | Unified BaseCache abstraction (6 caches migrated) |
| `compilation/` | Keyword compilation and montage features |

**Key single-file modules** (under `src/`):

| File | Purpose |
|------|---------|
| `state.py` | PipelineState + dataclasses (CANONICAL import location) |
| `pipeline.py` | PipelineOrchestrator, `create_healing_pipeline()` |
| `checkpoint.py` | Resume/checkpoint management |
| `caption_fetcher.py` | YouTube caption fetching |
| `entity_cache.py` | Cross-project entity cache |
| `location_service.py` | GeoNames geocoding |

### Pipeline Stages

| Stage | Class | Purpose |
|-------|-------|---------|
| ANALYZE | AnalyzeStage | Keywords, topics, entities, location chapters |
| ENTITY_IMAGES | EntityImagesStage | Download entity images (Google, Bing, Pexels) |
| ENTITY_VIDEOS | EntityVideosStage | Download stock videos for entities |
| VIDEO_METADATA | VideoMetadataStage | Fetch video metadata for caption-first mode |
| CAPTION | CaptionStage | Fetch YouTube captions (before TRANSCRIBE) |
| DOWNLOAD | DownloadStage | YouTube video/audio download |
| STOCK | StockVideoStage | Download generic stock footage (B-roll) |
| BROLL_DOWNLOAD | BrollDownloadStage | Download B-roll with keyword suffixes |
| REMIX | RemixStage | Filter videos by keyword relevance |
| TRANSCRIBE | TranscribeStage | Whisper + embeddings |
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

**Strategies:**

| Strategy | Attempts | Behavior |
|----------|----------|----------|
| `aggressive` | 5/stage | Try everything, auto-fix, minimal interaction |
| `conservative` | 3/stage | Safe fixes only, preserve user config (default) |
| `interactive` | 3/stage | Ask user before major changes |
| `minimal` | 1/stage | Fail fast, critical fixes only |

**Usage in code:**
```python
from src.pipeline import create_healing_pipeline, run_pipeline_with_healing

# Option 1: Get components for manual control
pipeline, orchestrator, runner = create_healing_pipeline(config, project_dir)
success = runner.run_pipeline(pipeline)
orchestrator.print_report()

# Option 2: One-liner convenience function
success = run_pipeline_with_healing(config, project_dir)
```

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

### Bypass & Escalation System

3-tier yt-dlp bypass using curl_cffi TLS fingerprint spoofing. Always-on, per-keyword, thread-safe.

**Tier Progression:**

| Tier | Trigger | yt-dlp Args Added | Cookie Rotation |
|------|---------|-------------------|-----------------|
| 1 (always) | Every call | `--impersonate Chrome-136:Macos-15` (rotating) | No |
| 2 (on 403) | 2 consecutive 403s | Tier 1 + `--extractor-args youtube:player_client=web_safari,tv_downgraded,web` | No |
| 3 (max) | 2 more 403s at Tier 2 | Tier 2 args + cookie rotation | Yes |

**Key classes:**

| Class | File | Purpose |
|-------|------|---------|
| `ImpersonationManager` | `src/downloader/impersonation.py` | Auto-detect targets, round-robin rotation |
| `EscalationManager` | `src/downloader/escalation_manager.py` | Per-keyword tier progression, metrics |
| `CookieMethodFallback` | `src/downloader/cookie_method_fallback.py` | browser:firefox -> file:main.txt -> file:backup1.txt -> none |
| `RateLimitBudget` | `src/downloader/rate_limit_budget.py` | Cross-keyword recovery resource tracking |
| `is_escalation_trigger()` | `src/downloader/escalation_manager.py` | Detect 403/bot patterns in stderr |

**Escalation behavior:**
- **Sticky:** Tiers never de-escalate mid-session (success resets 403 counter but keeps tier)
- **Cooldown:** 300s between escalations per keyword (`ExtractorArgsConfig.cooldown_seconds`)
- **Thread-safe:** Per-keyword locks for concurrent downloads
- **All 8 call sites wired:** core.py (x2), audio_first.py (x3), title_filter.py, speech_screening.py, caption_fetcher.py

**Config:**
```yaml
download:
  impersonation:
    enabled: true              # Master switch for Tier 1
    preferred_targets: []      # Filter targets (empty = use all detected)
  extractor_args:
    enabled: true              # Master switch for Tier 2+
    player_clients: ["web_safari", "tv_downgraded", "web"]
    escalation_threshold: 2    # Consecutive 403s before escalating
    cooldown_seconds: 300      # Cooldown between escalations
```

**Adding new yt-dlp call sites:** Use `escalation_manager.get_escalation_args(keyword)` instead of direct `impersonation_manager.get_impersonate_args()`. Handle `result.rotate_cookies` flag for Tier 3.

### SABR Anti-Stall Strategy

YouTube's SABR streaming causes fragments to stall indefinitely (TCP connection stays open, zero data). Our subprocess stall detector kills the process and retries with resume.

**Current flags on both download commands (`core.py`):**

| Flag | Purpose |
|------|---------|
| `--socket-timeout 10` | Give up dead connections after 10s (forces yt-dlp internal retry) |
| `--retries 10` | yt-dlp whole-download retries |
| `--fragment-retries 10` | yt-dlp per-fragment retries |
| `--throttled-rate 100K` | Re-extract when throttled below 100KB/s |
| `--force-ipv4` | IPv6 causes silent hangs on Windows |
| `--http-chunk-size 10M` | YouTube throttles chunks >10MB |
| `--skip-unavailable-fragments` | Don't hang on broken fragments |
| `--ignore-config` | Prevent user config conflicts |
| `--progress --newline` | Output progress lines for stall detection |

**Flags intentionally REMOVED:**
- `--no-continue`: Removed so retries resume from last fragment instead of restarting
- `--quiet`: Removed so metadata extraction output resets stall timer
- `--no-warnings`: Removed so retry warning messages reset stall timer
- `--concurrent-fragments`: Removed — triggers 403 errors on YouTube

**Config settings:** `stall_timeout: 60`, `max_retries: 6`, `retry_backoff: 2.0`

**Subprocess encoding:** ALL subprocess calls that process yt-dlp output MUST use `encoding='utf-8', errors='replace'` to prevent charmap crash on Windows (see Rule 27).

## Development Rules

| Rule | Summary | Key Point |
|------|---------|-----------|
| 1 | Config sync | Update BOTH `src/config.py` AND `config.yaml` |
| 2 | `__post_init__` | Nested dataclass fields load as `dict` - convert |
| 3 | DownloadedVideo | Use `file=`, `duration_tier=` NOT `path=`, `tier=` |
| 4 | Regex lookbehind | Python needs fixed-width - use capture groups |
| 5 | Live streams | Add `!is_live & !was_live` to yt-dlp match-filter |
| 6 | Dict/Object config | Handle both: `vc.get()` if dict, `getattr()` if object |
| 7 | Embeddings truthiness | Use `is_embeddings_empty()` - numpy fails bool |
| 8 | B-roll propagation | SceneDetection → text_metadata → Match restores is_broll |
| 9 | LLM Client | Use `src/llm_client/` for ALL LLM calls |
| 10 | Test non-interactive | Tests MUST use `--non-interactive` |
| 11 | Dataclass imports | Import from `src/state.py` or `src/config.py` only |
| 12 | VAD Filter | Videos: OFF (hardcoded). Voiceover: ON (config) |
| 21 | Project vs Global config | Use `project_config.yaml` for project-specific settings |
| 22 | Caption-first paths | Video IDs not file paths - don't filter as `caption_only` |
| 23 | Project config merge | Deep merge preserves sibling sections (`_deep_merge_section`) |
| 24 | Running pipeline = latest code | Python imports dynamically - fixes take effect immediately |
| 25 | `--output-only` needs stage data | Checkpoint must have populated `stages` dict, not just `last_completed_stage` |
| 27 | Subprocess encoding (Windows) | ALL `subprocess.Popen`/`run` with `text=True` MUST add `encoding='utf-8', errors='replace'` |

### DaVinci Rules (13-17)

| Rule | Issue | Solution |
|------|-------|----------|
| 13 | XML bin structure | `<bin>` under `<xmeml>`, `file:///` paths, skip audio-only |
| 14 | OTIO paths | Use `file:///E:/...` URLs (not plain paths), forward slashes, skip unicode/audio-only |
| 15 | OTIO caching | Rename file after fixes to bypass corrupt cache |
| 16 | Duplicate paths | `MediaPathNormalizer` dedupes same file in different folders |
| 17 | Large timelines | Auto-split at 3000 items into PART1-4 |

### Healer Rules (18-20)

| Rule | Summary |
|------|---------|
| 18 | Use `getattr`/`setattr` - never replace config object |
| 19 | `.fixed()` for retry, `.failed()` to abort |
| 20 | Use `self.log_attempt()` and `self.log_success()` |

### Rule 21: Project vs Global Config

**NEVER modify `config.yaml` for project-specific changes.** Use `project_config.yaml` in the project folder instead.

| Change Type | File to Edit |
|-------------|--------------|
| Project-specific settings | `E:\Edit Job\client\project\project_config.yaml` |
| New default features | `D:\_Projects\voiceover-matcher-subtitle\config.yaml` |
| New config fields | `src/config.py` AND `config.yaml` (Rule 1) |

**Warning signs you're editing the wrong file:**
- User mentions a specific project path (e.g., `E:\Edit Job\theresa\...`)
- Change is based on client feedback for one project
- Setting would be too restrictive as a global default

### Rule 22: Caption-First Mode Paths

In **caption-first mode**, `video_path` in `text_metadata` contains YouTube video IDs (11 chars like `EPcZIso6bHw`), NOT file paths. Any filtering logic that:
- Checks if path exists as file
- Detects "video ID pattern" (11 alphanumeric chars)
- Marks entries as `caption_only` for later filtering

**MUST first check `config.download.caption_first.enabled`**. If enabled, caption entries ARE the primary candidates and should NOT be filtered out.

**Symptom:** "no candidates" for most segments, 0% confidence, empty `source_file` in matches, V2-V6 tracks empty.

### Rule 23: Project Config Deep Merge

`project_config.yaml` merges **deeply** into `config.yaml` via `_deep_merge_section()` in `src/cli/config_utils.py`.

**What this means:** When project_config.yaml has:
```yaml
download:
  fallback:
    proxy:
      enabled: true
```

It merges INTO `config.download.fallback.proxy` - it does NOT replace the entire `fallback` section. Sibling sections like `fallback.caption` are preserved from `config.yaml`.

**When adding new nested config sections:**
1. Defaults in dataclass (`src/config/sections/*.py`) are used if not in YAML
2. `config.yaml` values override dataclass defaults
3. `project_config.yaml` values override `config.yaml` (deep merge)

**Symptom of broken merge:** Config values from `config.yaml` ignored when `project_config.yaml` touches a sibling section. Check `_deep_merge_section()` handles nested objects correctly.

### Rule 25: `--output-only` Checkpoint Requirements

`--output-only` mode relies on checkpoint data restoration. It works when:
- Checkpoint has populated `stages` dict with data from each stage
- Pipeline completed normally before (wrote stage data to checkpoint)

**Fails when:**
- `stages` dict is empty `{}` (even if `last_completed_stage` is set correctly)
- Checkpoint was manually edited or corrupted
- Pipeline was interrupted before writing stage data

**Symptom:** `--output-only` runs through earlier stages (CAPTION, TRANSCRIBE, etc.) instead of jumping to OUTPUT.

**Workaround when checkpoint corrupted:**
1. Use `--match-only` instead (re-runs MATCH + OUTPUT, slower but works)
2. Or restore from a backup checkpoint that has stage data

### Rule 26: Caption-First Segment Resolution

In **caption-first mode**, matches store **video IDs** (e.g., `DDi-Swd7Qcw`) NOT file paths. The `source_file` field contains:
- YouTube video ID (11 chars) - needs segment resolution
- Full path (if from audio-first cache) - already resolved

**DOWNLOAD_SEGMENTS** only downloads segments for videos in `video_candidates`. Videos from the **global caption cache** (other projects) won't have segments downloaded.

**Symptom:** OTIO has invalid paths like `file:///D:/_Projects/.../DDi-Swd7Qcw` (video ID used as filename, no `.mp4` extension).

**Analysis pattern:**
```python
# Check segment coverage
segment_ids = {extract_video_id(f) for f in glob('E:/v/project/*_segments/*.mp4')}
match_ids = {m['video_file'] for m in checkpoint['match']['matches']}
missing = match_ids - segment_ids  # These won't have video in OTIO
```

**Fix options:**
1. Re-run DOWNLOAD_SEGMENTS to download missing segments (needs video_candidates update)
2. Limit matching to project's candidates only (loses global cache benefit)
3. Add segment download for global cache videos (significant change)

### Testing

```bash
# Quick verification (single test)
pytest tests/test_cache.py::TestCacheStats::test_hit_rate -v

# Fast unit tests only
pytest tests/ -m fast -v

# All tests (skips integration)
pytest tests/ -v --tb=short

# With coverage
pytest tests/ --cov=src --cov-report=html

# Run specific module tests
pytest tests/test_llm_client/ tests/test_keyword_extractor/ -v
```

**Test markers:** `fast` (unit), `integration`, `stress`, `simulation`, `requires_api`, `requires_network`

### Pre-commit Checklist

- [ ] `python -m py_compile main.py`
- [ ] `python -m py_compile src/config.py`
- [ ] `pytest tests/ -v --tb=short -x`
- [ ] Config in both `config.py` AND `config.yaml`
- [ ] Nested configs have `__post_init__`

## DaVinci API Integration

**See:** [DAVINCI_INTEGRATION_ROADMAP.md](DAVINCI_INTEGRATION_ROADMAP.md) for full automation plan.

| Principle | Implementation |
|-----------|----------------|
| **JSON I/O** | Analysis results → JSON for DaVinci scripts |
| **Modular** | Each stage callable independently |
| **Feedback-ready** | Hooks for editor decisions → learning |
| **API-first** | Core logic headless; CLI optional |

**Scripts:** `scripts/davinci/` - quick_setup, track_manager, diagnose, relink, import, export

**Requires DaVinci Resolve Studio** (scripting is Studio-only).

## Ralph Loop (Autonomous Development)

Location: `scripts/ralph/` - Autonomous development assistant.

```powershell
# Unified launcher (recommended)
.\scripts\ralph\launcher.ps1

# Direct execution
.\scripts\ralph\ralph.ps1 [-TrueAuto] [-Resume] [-FocusArea <area>]

# Watch dashboard (separate terminal)
.\scripts\ralph\watch.ps1 [-Interval 5]
```

**Launcher modes:**

| Mode | Flag | Description |
|------|------|-------------|
| Standard | (default) | Work through stories, pause on sprint complete |
| TrueAuto | `-TrueAuto` | Continuous improvement, auto-generate new sprints |
| Resume | `-Resume` | Continue where left off |
| Smart Queue | (interactive) | Describe work in natural language, Ralph picks focus areas |
| Ralph's Choice | `-RalphsChoice` | Ralph scores all areas, user confirms each decision |
| Ralph's Choice Auto | `-RalphsChoiceAuto` | Ralph scores and continues autonomously (fully unattended) |

**Ralph's Choice Auto features:**
- Scores all focus areas, picks highest (or stay/switch decision after sprint 1)
- 10s countdown between sprints (press any key to pause/override/quit)
- `maxConsecutiveSprints` cap (default: 20) + `maxIterations` cap (default: 115)
- Archives completed sprints, updates queue progress, checks graceful stop

**Module structure** (`scripts/ralph/lib/`):

| Module | Purpose |
|--------|---------|
| `claude.ps1` | Claude subprocess execution, result resolution |
| `display.ps1` | Banners, iteration display |
| `loops.ps1` | Standard, TrueAuto, RalphsChoice, RalphsChoiceAuto loops |
| `metrics.ps1` | CSV recording, health comparison, fast-fail detection |
| `prompts.ps1` | Prompt building for PRD generation and story work |
| `quality.ps1` | Diff scoring, test regression, code review, rollback |
| `queue.ps1` | Queue management, interview context, progress tracking |
| `scoring.ps1` | Focus area scoring, stay/switch decisions, story ordering |
| `sprint.ps1` | PRD read/write, sprint archive, sprint history |

**Focus areas:** pipeline, config, rate-limiting, caption, download, quality, speed, compilation, otio, agents, client-learning, testing, unit-tests, integration-tests, mutation-tests, documentation, ux

**Testing Ralph Loop:**
```powershell
# Run all Pester tests (~403 tests)
Invoke-Pester -Path 'scripts/ralph/tests' -Output Detailed

# Run specific test file
Invoke-Pester -Path 'scripts/ralph/tests/Ralph.Tests.ps1' -Output Detailed

# Run by tag
Invoke-Pester -Path 'scripts/ralph/tests' -Tag 'Unit' -Output Detailed
```

See `scripts/ralph/README.md` for full documentation.

## Known Issues & Solutions

Documented shortcomings encountered and how they were resolved:

### Ralph Loop

| Issue | Symptom | Solution |
|-------|---------|----------|
| Unicode arrows in PowerShell | Parse error: `The string is missing the terminator` | Use ASCII `->` instead of `→` in .ps1 files |
| Queue mode skipped planning | Claude jumped into old stories without generating new PRD | Added `-GeneratePRD` flag to `Invoke-ClaudeForFocusArea` in queue loop |
| Fresh queue reused old PRD | Same focus area name meant no new PRD generated | Added `$isFreshQueue` check - if 0 completed areas, always regenerate |
| Standard mode ignored queue | PRD focus area matched queue, but old stories used | Check queue/PRD mismatch OR fresh queue at `Start-StandardLoop` entry |
| Vague prompt in queue mode | "Focus on quality" gave Claude no direction | Added optional work description prompt after area selection |
| Interview context not actionable | `details = "Direct queue: quality, agents..."` | Changed to user-provided description or sensible default |
| Queue didn't advance to next area | `Get-NextQueuedFocusArea` returned empty despite valid queue.json | Replaced function calls with inline queue file reading in `Start-StandardLoop` |
| Ralph's Choice didn't update queue | Queue stayed stale after Choice/ChoiceAuto sprints | Added `Update-QueueProgress` to both Choice loops after sprint completion |
| Metrics CSV "Stream was not readable" | `Add-Content` fails after timeout on Windows | Added `-Encoding UTF8` + try/catch retry in `Record-Metric` |

### Pipeline

| Issue | Symptom | Solution |
|-------|---------|----------|
| Caption-first used audio-first | `--caption-first` still ran DOWNLOAD AUDIO stage | `_is_caption_first_enabled()` must handle dict configs (Rule 6) |
| Caption-first segment gaps | OTIO paths like `file:///DDi-Swd7Qcw` (no .mp4) | Rule 26: DOWNLOAD_SEGMENTS only covers `video_candidates`, not global cache |
| V8 track empty | B-roll matching produced no results | Two detection methods: face_score < 0.3 OR word_count < threshold |
| `--output-only` re-ran stages | Jumped to CAPTION instead of OUTPUT | Rule 25: Checkpoint needs populated `stages` dict, not just `last_completed_stage` |
| Project config overwrote siblings | Setting `download.fallback.proxy` cleared `fallback.caption` | Deep merge via `_deep_merge_section()` preserves sibling keys |
| 403 errors in caption-first | No cookie rotation in `audio_first.py` | CookieRotator now shared from VideoDownloader to AudioFirstPipeline |
| charmap codec error (Windows) | `UnicodeDecodeError: 'charmap'` crashes reader thread → phantom stall | Rule 27: `encoding='utf-8', errors='replace'` on all subprocess calls |
| SABR download stalling | Downloads stall 50-90%, no yt-dlp output for 60s+ | Resume-on-retry + stall detector + 6 retries (see SABR Anti-Stall Strategy) |
| info.json read crash | `open(info_file, 'r')` fails on non-ASCII metadata | `open(info_file, 'r', encoding='utf-8')` + catch `UnicodeDecodeError` |

### General Patterns

| Pattern | When It Happens | Prevention |
|---------|-----------------|------------|
| Config dict vs object | YAML loads as dict, dataclass as object | Always use `getattr(obj, 'field', default)` or check `isinstance` |
| Numpy bool ambiguity | `if embeddings:` fails on numpy arrays | Use `is_embeddings_empty()` helper |
| PowerShell falsy arrays | `@()` is falsy but `.Count` works | Check `.Count -gt 0` not just truthiness |
| MagicMock auto-attributes | `getattr(MagicMock(), 'anything', default)` returns Mock not default | Set `mock.attr = None` explicitly for attributes that should be absent |
| yt-dlp user config conflict | `~/.config/yt-dlp/config` overrides pipeline's escalation args | Keep user config minimal; avoid `--extractor-args` (pipeline handles this) |

## Git Conventions

| Type | Format |
|------|--------|
| Features | `feature/descriptive-name` |
| Fixes | `fix/issue-description` |
| Commits | `feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:` |

**Ignored files:** Most `*.md` files are gitignored. Exceptions: root-level docs (`CLAUDE.md`, `README.md`, etc.) and `docs/**/*.md`. If `git add` fails with "ignored by .gitignore", either:
1. Add an exception to `.gitignore` (e.g., `!docs/plans/*.md`)
2. Use `git add -f` if it's a one-off
3. Place the file in an already-allowed location

## Session History

| Date | Changes |
|------|---------|
| 2026-01-28 | Fix: Ralph's Choice loops now update queue.json on sprint completion; metrics.csv encoding fix |
| 2026-01-28 | CLAUDE.md: Added Ralph's Choice modes, lib/ module structure to Ralph Loop section |
| 2026-01-27 | Fix: SABR anti-stall strategy — resume-on-retry, stall detector, socket-timeout 10, max_retries 6, removed --no-continue/--quiet/--no-warnings/--concurrent-fragments |
| 2026-01-27 | Fix: charmap encoding crash — added `encoding='utf-8', errors='replace'` to 15 subprocess call sites + info.json reading |
| 2026-01-27 | Fix: Added `--ignore-config` to all 13 yt-dlp call sites to prevent user config conflicts |
| 2026-01-27 | CLAUDE.md: Removed 10 phantom CLI flags, added 8 real undocumented ones |

*Full history in [CHANGELOG.md](CHANGELOG.md#session-history-archive)*
