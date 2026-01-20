# Claude Code Project Guide

> **LLM Editing Guide:** Tables > prose, one-liners > paragraphs. Keep rules to ~10 lines max. Session history: last 6 entries only.

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

### Common Commands

```bash
python main.py --voiceover script.srt --project "E:\Projects\MyDoc"
python main.py --match-only                    # Re-run matching only
python main.py --keyword-list "sunset,ocean" --keyword-mode montage --duration 60
python main.py --project "E:\Edit Job\theresa\Project" --client theresa  # Cross-project learning
python main.py --evolve-preset --client theresa  # Generate evolved preset from history
```

### Skill Commands

| Command | Description |
|---------|-------------|
| `/logcheck <project>` | Check log for errors, auto-fix |
| `/match-only <project>` | Re-run matching (skips download/transcribe) |
| `/newproject <name> <client> <doc_url>` | Create project from Google Doc links |
| `/validate-output [path]` | Validate output structure |
| `/ollama-check` | Diagnose Ollama setup for keyword mode |
| `/watch <project>` | Monitor pipeline progress |
| `/research <topic>` | Research a topic using Perplexity AI |
| `/import-feedback <project> [csv]` | Import DaVinci Resolve marker feedback |

**Use `/research` proactively** for API docs, library usage, error debugging, best practices, or any unfamiliar topic. Don't guess—research first.

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

ANALYZE → ENTITY_IMAGES → ENTITY_VIDEOS → DOWNLOAD → CAPTION → TRANSCRIBE → SCENE_DETECTION → MATCH → BROLL_MATCH → OUTPUT

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
| 5 | Live streams | Add `!is_live` to yt-dlp match-filter |
| 6 | Dict/Object config | Handle both: `vc.get()` if dict, `getattr()` if object |
| 7 | Embeddings truthiness | Use `is_embeddings_empty()` - numpy fails bool |
| 8 | B-roll propagation | SceneDetection → text_metadata → Match restores is_broll |
| 9 | Test non-interactive | Tests MUST use `--non-interactive` |
| 10 | LLM Client | Use `src/llm_client/` for ALL LLM calls |
| 11 | Dataclass imports | Import from `src/state.py` or `src/config.py` only |
| 12 | VAD Filter | Videos: OFF (hardcoded). Voiceover: ON (config) |

### DaVinci Rules (13-17)

| Rule | Issue | Solution |
|------|-------|----------|
| 13 | XML bin structure | `<bin>` under `<xmeml>`, `file:///` paths, skip audio-only |
| 14 | OTIO paths | Forward slashes, skip unicode/audio-only clips |
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
| 2026-01-20 | Phase 4 Cross-Project Learning: `--client` flag, client profiles (`src/feedback/client_profiles.py`), preset evolution algorithm |
| 2026-01-20 | DaVinci API roadmap + scripts: `DAVINCI_INTEGRATION_ROADMAP.md`, `scripts/davinci/` |
| 2026-01-19 | Keyword mode: `docs/plans/PLAN_KEYWORD_MODE.md`, `/ollama-check` skill |
| 2026-01-18 | OTIO auto-splitting (Rule 17): 4 parts when items > 3000 |
| 2026-01-16 | Caption-first mode, duplicate paths fix (Rule 16), OTIO caching fix |
