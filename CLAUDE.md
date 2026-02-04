# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## MCP Tools (Use Proactively)

| Tool | When to Use |
|------|-------------|
| `mcp__context7__*` | **Any library/API question** - always check docs first |
| `mcp__remotion-documentation__*` | Remotion-specific questions |
| `mcp__markitdown__*` | Convert URLs/PDFs to markdown |

**Rule:** Search documentation first using Context7 before responding from memory.

## Quick Reference

### Common Commands

```bash
# Basic pipeline
python main.py --voiceover script.srt --project "E:\Projects\MyDoc"

# Fast modes
python main.py --match-only                    # Re-run matching only
python main.py --output-only                   # Regenerate OTIO only (needs checkpoint)
python main.py --resume                        # Resume from checkpoint

# Keywords
python main.py --save-keywords mypreset        # Save after extraction
python main.py --use-keywords mypreset         # Reuse saved

# Standalone OTIO regeneration (bypasses checkpoint)
python scripts/regenerate_otio.py "E:\Edit Job\client\project"
```

### CLI Flags

| Flag | Description |
|------|-------------|
| `--voiceover`, `-v` | Path to voiceover file (SRT, MP3, WAV, MP4) |
| `--project`, `-p` | Project directory path |
| `--config`, `-c` | Path to config file |
| `--match-only` | Skip download/transcribe, use cached data |
| `--output-only` | Regenerate OTIO/EDL/XML only (fastest) |
| `--resume` / `--fresh` | Resume from checkpoint / Force fresh start |
| `--force-rematch` | Force rematch all videos |
| `--non-interactive` | Skip prompts, use defaults |
| `--use-keywords [PRESET]` | Use saved keywords |
| `--save-keywords [NAME]` | Save extracted keywords |

### Skill Commands

| Command | Description |
|---------|-------------|
| `/logcheck <project>` | Check log for errors, auto-fix |
| `/match-only <project>` | Re-run matching (skips download/transcribe) |
| `/newproject <name> <client> <doc_url>` | Create project from Google Doc links |
| `/watch <project>` | Monitor pipeline progress (maintains `PIPELINE_STATUS.md`) |
| `/research <topic>` | Research using Perplexity AI |
| `/import-feedback <project> [csv]` | Import DaVinci Resolve marker feedback |

**Use `/research` proactively** for API docs, library usage, error debugging. Don't guess—research first.

## Development Rules (Consolidated)

| # | Rule | Key Point |
|---|------|-----------|
| 1 | Config sync | Update BOTH `src/config/sections/*.py` AND `config.yaml` |
| 2 | `__post_init__` | Nested dataclass fields load as `dict` - convert in post_init |
| 3 | DownloadedVideo | Use `file=`, `duration_tier=` NOT `path=`, `tier=` |
| 6 | Dict/Object config | Handle both: `vc.get()` if dict, `getattr()` if object |
| 7 | Embeddings truthiness | Use `is_embeddings_empty()` - numpy fails bool |
| 9 | **LLM Client** | Use `src/llm_client/` for ALL LLM calls |
| 10 | Test non-interactive | Tests MUST use `--non-interactive` |
| 11 | Dataclass imports | Import from `src/state.py` or `src/config/` only |
| 14 | OTIO paths | Use `file:///E:/...` URLs, forward slashes, skip audio-only |
| 22 | Caption-first (always on) | Video IDs not file paths - don't filter as `caption_only` |
| 27 | Subprocess encoding | ALL `subprocess.Popen/run` with `text=True` MUST add `encoding='utf-8', errors='replace'` |
| 28 | Download one-at-a-time | `_download_by_ids` must loop per video, NOT batch |
| 29 | LLM response format | Use `ResponseFormat.TEXT` for non-JSON responses, not `JSON_ARRAY` |
| 30 | Per-project config | Use `--config custom.yaml` flag — project_config.yaml auto-merge removed |
| 31 | Pipeline stages source | `src/checkpoint.py:STAGE_ORDER` is truth; sync `scripts/ralph/config/ralph-config.json` |
| 32 | Ralph file paths | Always use `state/prd.json`, `state/queue.json`, `session/prompt.md`, `config/ralph-config.json` — never root-level |
| 33 | Ralph seed story prompts | In `Build-StoryPrompt`, seed stories emit instructions+criteria BEFORE notes (notes are background context, safe to truncate) |
| 34 | Ralph prompt maxLength | `ralph-config.json` `prompts.maxLength` (default 10000) truncates story prompts — if stories fail mysteriously, check truncation first |
| 35 | PowerShell threading | Never use `[System.Threading.Thread]` with PS cmdlets — use `[powershell]::Create()` with `.AddArgument()` for background work; raw threads crash the host process |
| 36 | Console Quick Edit | Ralph disables Quick Edit Mode at startup (`Disable-QuickEditMode`) — clicking the console window freezes ALL `Write-Host` calls, blocking the monitoring loop, stall detection, and heartbeat while Claude keeps running |

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

1. Add field to dataclass in `src/config/sections/<section>.py`
2. Add `__post_init__` if nested dataclass
3. Add to `config.yaml` with comment
4. Access in code with `getattr()` fallback
5. Verify: `python -m py_compile src/config/sections/<section>.py`

## Architecture

### Setup

```bash
# Python 3.9+ required, FFmpeg required
pip install -r requirements.txt      # Runtime dependencies
pip install -r requirements-dev.txt  # Test/dev (pytest, pester, coverage)
```

### Core Packages (under `src/`)

| Package | Purpose |
|---------|---------|
| `stages/` | 7 modular pipeline stage classes |
| `config/` | Config dataclasses by section |
| `cli/` | CLI arg parsing, config loading |
| `llm_client/` | **Unified LLM abstraction** (Gemini, Anthropic, Ollama) |
| `downloader/` | YouTube download with bypass/escalation |
| `matching/` | Tiered video-to-voiceover matching |
| `otio/` | OTIO/EDL/XML timeline generation |
| `media_sources/` | Entity image/video search |
| `transcription/` | Whisper transcription + embeddings |
| `agents/` | Self-healing pipeline (ResilientRunner) |
| `cache/` | Unified BaseCache abstraction |
| `iterative_match/` | Gap analysis and query learning |

### Pipeline Stages (Execution Order)

```
ANALYZE → VIDEO_SEARCH → CAPTION → MATCH → ITERATIVE_MATCH → DOWNLOAD_SEGMENTS → OUTPUT
```

### OTIO Track Layout

| Track | Purpose | Default State |
|-------|---------|---------------|
| V1 | Primary video | Enabled |
| V2-V3 | Alternatives 1-2 | Disabled |
| V4-V6 | Secondary (diversity-scored) | Disabled |
| V7 | Embedding-Diversity strategy | Disabled |
| V8 | B-roll Only (silent footage) | Disabled |

### Project Directory Structure

```
ProjectName__2026-01-03/
├── run.bat              # Launcher script
├── voiceover/           # User puts audio here
├── output/              # Generated OTIO, EDL, XML
├── checkpoint.json      # Resume state
└── .cache/              # Transcriptions, embeddings
```

## Checkpoint & Caching

### Cache Locations

| Cache | Location |
|-------|----------|
| Checkpoint | `<project>/checkpoint.json` |
| Transcriptions | `.cache/transcriptions/` |
| Embeddings | `.cache/embeddings/` |
| LLM responses | `.cache/llm_responses/` |
| Global videos | `~/.matcher_global_cache/` |
| Entity images | `~/.matcher_entity_cache/` |

### Force Stage Re-run

```python
import json
cp_path = r'E:/Edit Job/client/project/checkpoint.json'
with open(cp_path, 'r') as f: cp = json.load(f)
cp['last_completed_stage'] = 'STAGE_BEFORE_TARGET'  # e.g., 'MATCH' to re-run BROLL_MATCH
with open(cp_path, 'w') as f: json.dump(cp, f, indent=2)
```
Then run with `--resume`.

### Clear Caches

```bash
python main.py --fresh                 # Clear checkpoint
rm -rf .cache/transcriptions           # Force re-transcription
rm -rf .cache/scene_detection          # Force re-detection
```

## Configuration

### Self-Healing

Enabled by default. Strategies: `aggressive`, `conservative` (default), `interactive`, `minimal`

```yaml
healing:
  enabled: true
  strategy: "conservative"
  max_attempts_per_stage: 3
```

### Bypass & Escalation (yt-dlp)

4-tier system using curl_cffi TLS fingerprint spoofing + VPN:
- **Tier 1**: `--impersonate Chrome-136:Macos-15` (always on)
- **Tier 2**: + `--extractor-args youtube:player_client=web_safari,tv_downgraded,web` (on 403)
- **Tier 3**: + cookie rotation (on continued 403s)
- **Tier 4**: + Mullvad VPN IP rotation (when cookies exhausted)

Key classes: `ImpersonationManager`, `EscalationManager`, `CookieMethodFallback`, `MullvadVPN` in `src/downloader/`

### Mullvad VPN Integration (Tier 4)

**Prerequisites:**
```powershell
# Install Mullvad VPN client (Windows)
winget install Mullvad.Mullvad

# Verify CLI is available
mullvad status
```

**Configuration in config.yaml:**
```yaml
download:
  mullvad:
    enabled: true                          # Enable Tier 4 VPN rotation
    preferred_countries: ['us', 'gb', 'de', 'nl']  # Server locations to rotate through
    rotation_strategy: 'random'            # 'random', 'sequential', or 'nearest'
    max_rotations_per_session: 5           # Limit VPN rotations per run
    verification_timeout: 10               # Timeout for am.i.mullvad.net check
    rotation_delay_seconds: 5              # Delay after rotation before retrying
```

**Behavior:**
- VPN rotation triggers when cookie rotation is exhausted (Tier 3 fails)
- After VPN rotation: circuit breaker resets, cookie/backoff budgets refresh
- Verification via `am.i.mullvad.net/json` confirms Mullvad exit IP
- Falls back to ping verification if API unreachable

**Troubleshooting:**
| Issue | Solution |
|-------|----------|
| `mullvad: command not found` | Add Mullvad to PATH or reinstall |
| VPN rotation fails | Check `mullvad status`, ensure account is active |
| Verification fails | Firewall may block am.i.mullvad.net; rotation still works |

### Caption Retry Budget

Controls how many caption fetch attempts are allowed before falling back to transcription. Prevents infinite retry loops when YouTube is rate-limiting or captions are unavailable.

**Defaults:**
- `max_attempts`: 100 (maximum total fetch attempts across batch)
- `auto_scale`: true (scales max_attempts based on batch size)
- `attempts_per_video`: 2.0 (1 attempt + 1 retry per video average)
- `max_backoff_time_seconds`: 300.0 (5 min cumulative backoff limit)

**Configuration in config.yaml:**
```yaml
download:
  caption_first:
    retry_budget:
      enabled: true                   # Enable retry budget tracking
      max_attempts: 100               # Maximum total fetch attempts (0 = unlimited)
      max_backoff_time_seconds: 300.0 # Maximum cumulative backoff time (5 min)
      auto_scale: true                # Scale max_attempts based on batch size
      attempts_per_video: 2.0         # 2.0 = 1 attempt + 1 retry per video average
```

**Auto-Scaling Behavior:**
- When `auto_scale: true`, max_attempts scales UP for large batches
- Formula: `scaled_max = max(original_max, batch_size * attempts_per_video)`
- Example: batch of 200 videos → scaled max = 200 × 2.0 = 400 attempts
- Only scales UP, never reduces below configured max_attempts

**Debugging Budget Exhaustion:**
1. Check logs for `CaptionRetryBudget: EXHAUSTED` - shows which limit was hit
2. Check logs for `scaled max_attempts` - shows if auto-scaling triggered
3. Look for `budget_exhausted` in skipped video reasons
4. Review `error_counts` in budget summary for error category breakdown

**Related classes:** `CaptionRetryBudget`, `CaptionRetryBudgetConfig` in `src/caption/retry_budget.py`

## Testing

```bash
# Quick verification
pytest tests/test_cache.py::TestCacheStats::test_hit_rate -v

# Fast unit tests only
pytest tests/ -m fast -v

# All tests
pytest tests/ -v --tb=short

# With coverage
pytest tests/ --cov=src --cov-report=html

# Run specific module tests
pytest tests/test_llm_client/ -v

# Stop on first failure
pytest tests/ -v -x
```

**Test markers:** `fast`, `integration`, `stress`, `simulation`, `requires_api`, `requires_network`

### Pre-commit Checklist

```bash
python -m py_compile main.py
python -m py_compile src/config/base.py
pytest tests/ -v --tb=short -x
```

## Troubleshooting

### Common Issues

| Issue | Symptom | Fix |
|-------|---------|-----|
| V8 track empty | No B-roll matches | Check `face_score < 0.3` OR `word_count < threshold` |
| 403 Forbidden | Download failures | Wait 1 hour, check cookies, or use VPN |
| Empty source_file | Caption-first mode | Rule 22: Video IDs not file paths |
| `--output-only` runs stages | Checkpoint corrupted | Use `--match-only` instead |
| Checkpoint corrupted | Various | Restore from `checkpoint.backup.json` |
| Subprocess crash | Windows encoding | Rule 27: Add `encoding='utf-8', errors='replace'` |
| Videos skipped (budget_exhausted) | Logs show `EXHAUSTED` | Increase `retry_budget.max_attempts` or check for rate limiting |

### Debug Workflows

**Pipeline not progressing?** Use `/watch <project>` or check logs directly.

**Match quality issues?** Check: match counts, confidence scores, video variety, empty source_file warnings.

**Output ≠ Success:** Pipeline completing doesn't mean quality is acceptable. Always verify results.

## Ralph Loop (Autonomous Development)

Location: `scripts/ralph/` — See [scripts/ralph/README.md](scripts/ralph/README.md) for full documentation.

### Ralph Directory Structure

| Directory | Purpose | Key Files |
|-----------|---------|-----------|
| `config/` | Static config | `ralph-config.json` |
| `state/` | Persistent state | `prd.json`, `queue.json`, `sprint_history.json` |
| `session/` | Volatile per-session | `prompt.md`, `metrics.csv`, `healing_log.jsonl` |
| `archive/` | Completed sprints | `sprint-N.json`, `sessions/` |

**CRITICAL:** When generating stories or updating PRD, always use `scripts/ralph/state/prd.json` — NOT `scripts/ralph/prd.json` (root level doesn't exist).

```powershell
# Unified launcher
.\scripts\ralph\launcher.ps1

# Direct execution
.\scripts\ralph\ralph.ps1 [-TrueAuto] [-Resume] [-FocusArea <area>]

# Watch dashboard
.\scripts\ralph\watch.ps1 [-Interval 5]
```

**Modes:** Standard, TrueAuto, Resume, Smart Queue, Ralph's Choice, Ralph's Choice Auto

**Testing Ralph:**
```powershell
Invoke-Pester -Path 'scripts/ralph/tests' -Output Detailed
```

## Git Conventions

| Type | Format |
|------|--------|
| Branches | `feature/descriptive-name`, `fix/issue-description` |
| Commits | `feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:` |

**Ignored files:** Most `*.md` files are gitignored except root-level docs and `docs/**/*.md`.

## Additional Documentation

- [REFACTORING.md](REFACTORING.md) - Refactoring roadmap (12 major refactorings completed)
- [DAVINCI_INTEGRATION_ROADMAP.md](DAVINCI_INTEGRATION_ROADMAP.md) - DaVinci API automation plan
- [scripts/ralph/README.md](scripts/ralph/README.md) - Ralph Loop full documentation

## Session History

| Date | Changes |
|------|---------|
| 2026-02-05 | Fix: Early exit race condition — added 30s periodic re-check of prd.json for `passes:true` regardless of file write detection; catches cases where `passes:true` written in same polling interval as prior write, reducing worst-case stall from 360s to ~45s |
| 2026-02-05 | Fix: Phantom revalidation — exit code override in `Invoke-ClaudeSubprocess` now re-reads `prd.json` before overriding; prevents false phantom when Claude reverts `passes:true` during grace period, which caused permanent hard-story ban |
| 2026-02-05 | Fix: PHANTOM false-positive — `Log-StoryVerification` return value leaked to pipeline in timeout/failure paths of `Resolve-ClaudeResult`, making `$resolution.Success` truthy via array member enumeration; suppressed with `$null =` |
| 2026-02-05 | Fix: Evidence gate diff excludes Ralph metadata dirs (`state/`, `session/`, `archive/`) — `prd.json` (40K chars) consumed entire 8K truncation budget, causing 0% criteria verified on valid implementations |
| 2026-02-04 | Feat: Replace keyword evidence gate with LLM-based criteria verification — `Confirm-CriteriaEvidence` sends single haiku call instead of per-criterion keyword matching; falls back to `Search-CriterionEvidence` when CLI unavailable |
| 2026-02-04 | Feat: Ralph evidence threshold gate — stories rejected when <90% of acceptance criteria have verifiable evidence; previously exit code 0 alone was sufficient to pass regardless of actual evidence |
| 2026-02-04 | Fix: Ralph queue not advancing — TrueAutoLoop and AdaptiveOvernightLoop missing `Update-QueueProgress` after sprint completion, causing infinite re-sprints on same focus area |
| 2026-02-04 | Fix: Ralph early exit code override — `taskkill /T /F` produces exit code 1, causing false "story failed" → FAST-FAIL abort; now overrides to 0 when `storyCompletionDetected` is true |
| 2026-02-04 | Feat: Ralph story completion early exit — when prd.json shows `passes:true`, kill Claude after 15s grace period instead of waiting for stall timeout (was up to 45min for quality focus area) |
| 2026-02-04 | Fix: Bot-abort tests use `stats.field` attribute access (not `stats['field']`) after `SegmentDownloadStats` dataclass migration |
| 2026-02-04 | Fix: Ralph `Invoke-ClaudeWithInfiniteRetry` timeout/exception returns missing `ExecutionStart`/`ExecutionEnd` — caused `Cannot convert null to System.DateTime` crash in `Log-ClaudeInvocation` |
| 2026-02-04 | Fix: Ralph monitoring loop freeze — disable Console Quick Edit Mode at startup (Rule 36) |
| 2026-02-02 | Removed project_config.yaml auto-merge — use `--config` flag for per-project settings |
| 2026-02-01 | Caption-first always on, removed `--caption-first` flag, cookie rotation in caption fetcher, removed max_keywords param, fixed LLM text response format |
| 2026-01-29 | Streamlined CLAUDE.md: consolidated rules, reduced verbosity |
| 2026-01-29 | Fix: Download timeout — `_download_by_ids` now downloads one-at-a-time (Rule 28) |
| 2026-01-28 | Feat: Ralph Loop self-healing — T1/T2/T3 health checks, sprint freeze |
| 2026-01-28 | Fix: Ralph's Choice queue auto-sync |
| 2026-01-27 | Fix: SABR anti-stall — resume-on-retry, stall detector |

*Full history in [CHANGELOG.md](CHANGELOG.md#session-history-archive)*
