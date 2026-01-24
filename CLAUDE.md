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
| `--project`, `-p` | Project directory path |
| `--config`, `-c` | Path to config file |
| `--match-only` | Skip download/transcribe, use cached data |
| `--output-only` | Regenerate OTIO/EDL/XML only (fastest, no re-matching) |
| `--resume` / `--fresh` | Resume from checkpoint / Force fresh start |
| `--keyword-list "a,b,c"` | Keyword mode: comma-separated keywords (no voiceover) |
| `--keyword-mode MODE` | montage, script, or collection |
| `--duration SECS` | Target duration for keyword mode |
| `--non-interactive` | Skip prompts, use defaults |
| `--client CLIENT_ID` | Client ID for cross-project learning (e.g., "theresa", "stu") |
| `--evolve-preset` | Generate evolved preset from project history (requires `--client`) |
| `--list-clients` | List all client profiles and exit |
| `--client-stats [ID]` | Show client statistics (specific client or "all") |
| `--high-matches` | Enable iterative matching until target confidence achieved |
| `--target-confidence SCORE` | Target confidence for high matches mode (default: 0.90) |
| `--coverage-target RATIO` | Coverage target for high matches mode (default: 0.85) |

### Common Commands

```bash
python main.py --voiceover script.srt --project "E:\Projects\MyDoc"
python main.py --match-only                    # Re-run matching only
python main.py --output-only                   # Regenerate OTIO only (fastest, needs checkpoint)
python main.py --keyword-list "sunset,ocean" --keyword-mode montage --duration 60
python main.py --project "E:\Edit Job\theresa\Project" --client theresa  # Cross-project learning
python main.py --evolve-preset --client theresa  # Generate evolved preset from history
python main.py --voiceover script.srt --high-matches  # Iterate until 90%+ confidence

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
| `src/downloader.py` | Video/audio download logic |
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
| 9 | Test non-interactive | Tests MUST use `--non-interactive` |
| 10 | LLM Client | Use `src/llm_client/` for ALL LLM calls |
| 11 | Dataclass imports | Import from `src/state.py` or `src/config.py` only |
| 12 | VAD Filter | Videos: OFF (hardcoded). Voiceover: ON (config) |
| 21 | Project vs Global config | Use `project_config.yaml` for project-specific settings |
| 22 | Caption-first paths | Video IDs not file paths - don't filter as `caption_only` |
| 23 | Project config merge | Deep merge preserves sibling sections (`_deep_merge_section`) |
| 24 | Running pipeline = latest code | Python imports dynamically - fixes take effect immediately |
| 25 | `--output-only` needs stage data | Checkpoint must have populated `stages` dict, not just `last_completed_stage` |

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

### Testing Checklist

- [ ] `python -m py_compile main.py`
- [ ] `python -m py_compile src/config.py`
- [ ] `pytest tests/ -v --tb=short -x` (quick verification)
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

## Configuration

### Short Paths (E:/v, E:/i)

```yaml
download:
  root_dir: "E:/v"    # Videos: E:/v/ProjectName/
image_search:
  root_dir: "E:/i"    # Images: E:/i/ProjectName/
```

Avoids 260-char Windows limit, faster NLE imports.

### Key Feature Configs

```yaml
# Audio-first: download audio only, then matched video segments
download.audio_first.enabled: true

# Caption-first: YouTube captions instead of Whisper
download.caption_first.enabled: true

# Self-healing
healing.enabled: true
healing.strategy: "conservative"  # aggressive, conservative, interactive, minimal
```

### PO Token Server (YouTube Auth)

YouTube requires PO Tokens for subtitle/video access. The pipeline auto-starts the server when needed.

**Setup (one-time):**
```bash
pip install bgutil-ytdlp-pot-provider
git clone --branch 1.2.2 https://github.com/Brainicism/bgutil-ytdlp-pot-provider.git ~/bgutil-ytdlp-pot-provider
cd ~/bgutil-ytdlp-pot-provider/server && npm install && npx tsc
```

**Manual start:** `scripts\start_pot_server.bat` or `node ~/bgutil-ytdlp-pot-provider/server/build/main.js`

The pipeline automatically checks and starts the PO Token server before running.

## Caching

| Cache | Location |
|-------|----------|
| Transcriptions | `.cache/transcriptions/` |
| Embeddings | `.cache/embeddings/` |
| Captions | `.cache/captions/` |
| Global videos | `~/.matcher_global_cache/` |
| Entity images | `~/.matcher_entity_cache/` |

Clear with `--fresh` or `rm -rf .cache/`

## Self-Healing Agents

Location: `src/agents/` - Auto-recovery for pipeline errors.

| Healer | Detects | Auto-Fix |
|--------|---------|----------|
| CheckpointHealer | Corrupt JSON | Restore backup |
| APIHealer | Rate limits, auth | Backoff, switch provider |
| DownloadHealer | YouTube 429 | Backoff, skip, alt format |
| PathHealer | Windows 260 char | Switch to E:/v |
| OTIOHealer | Timeline failures | Fix gaps, resolve paths |

## Git Conventions

| Type | Format |
|------|--------|
| Features | `feature/descriptive-name` |
| Fixes | `fix/issue-description` |
| Commits | `feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:` |

## Session History

| Date | Changes |
|------|---------|
| 2026-01-25 | Ralph startup improvements: Claude CLI path resolution, startup validation, -Resume flag, queue persistence, fast-fail detection |
| 2026-01-24 | Global cache segment download: VideoCandidate creation for cache videos, strategy_matches remapping (V7+), duration clamping fix |
| 2026-01-24 | Caption-first segment gap fix: 65% of matches from global cache lack segments. Rule 26 added |
| 2026-01-24 | MatchResultWrapper voiceover timing fix: Was using video_start/end (wrong), now uses segment timing |
| 2026-01-24 | V10 spam fix: Skip populating V10 track if <5 entity videos (prevents same video repeating) |
| 2026-01-24 | Force DOWNLOAD_SEGMENTS re-run: Edit checkpoint.json, set last_completed_stage to ITERATIVE_MATCH |

*Older entries archived to [CHANGELOG.md](CHANGELOG.md#session-history-archive)*
