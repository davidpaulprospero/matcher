# Claude Code Project Guide

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
python main.py --keyword-list "sunset,ocean" --keyword-mode montage --duration 60
python main.py --project "E:\Edit Job\theresa\Project" --client theresa  # Cross-project learning
python main.py --evolve-preset --client theresa  # Generate evolved preset from history
python main.py --voiceover script.srt --high-matches  # Iterate until 90%+ confidence
python main.py --voiceover script.srt --high-matches --target-confidence 0.85  # Custom target
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

## Architecture

### Key Files

| File | Purpose |
|------|---------|
| `src/state.py` | PipelineState + dataclasses (CANONICAL) |
| `src/llm_client/` | Unified LLM abstraction (Gemini/Anthropic/Ollama) |
| `src/stages/` | Modular pipeline stages |
| `src/agents/` | Self-healing pipeline (ResilientRunner, Healers) |
| `src/otio/` | Timeline generation (OTIO, EDL, XML) |

### Pipeline Stages

ANALYZE → ENTITY_IMAGES → ENTITY_VIDEOS → VIDEO_METADATA → CAPTION → DOWNLOAD → STOCK → REMIX → TRANSCRIBE → PREMISE → SCENE_DETECTION → MATCH → BROLL_MATCH → DOWNLOAD_SEGMENTS → ITERATIVE_MATCH → OUTPUT

**Note:** In caption-first/audio-first mode, DOWNLOAD gets audio only. Actual video segments are downloaded in DOWNLOAD_SEGMENTS after matching.

### OTIO Track Layout

| Track | Purpose |
|-------|---------|
| V1 | Primary video (enabled) |
| V2-V3 | Alternatives (disabled) |
| V4-V7 | Diversity/embedding strategies (disabled) |
| V8 | B-roll only - silent footage (disabled) |
| V9 | Entity images (disabled) |
| V10 | Stock videos (disabled) |

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
| 2026-01-23 | SOCKS fallback DONE: Added to `core.py:_run_download_cmd()` - removes proxy and retries on SOCKS/WinError 10061 |
| 2026-01-23 | Preserve-based iterative matching: Rewrote `_rematch()` to keep high-confidence matches, added `pre_populate()` to GlobalClipTracker |
| 2026-01-23 | Empty source_file fix: Filter candidates with empty source_file in `src/matching/main.py` |
| 2026-01-23 | fresh_keywords fix: Changed `_generate_fresh_keywords` to use LLM-based `generate_recovery_keywords` |
| 2026-01-23 | `/watch` skill: Added ITERATIVE_MATCH handling, fix verification patterns, coverage extraction |
| 2026-01-23 | Coverage analyzer fix: `MatchResultWrapper._FakeSegment.index` hardcoded to 0, fixed to use actual `segment_index` |
