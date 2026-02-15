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
| `--dry-run` | Preview pipeline execution plan without running stages. Shows stage estimates, input/output counts, checkpoint status |
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
| `/bugfix <error>` | Fast bug fix: reproduce, diagnose, fix, verify, scan for similar |
| `/ralph-insights` | Analyze sessions split by Ralph vs interactive — true success rates |

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
| 38 | Segment file format | DOWNLOAD_SEGMENTS saves flat `{vid}_{start}_{end}.mp4` in `config.downloaded_videos_dir`; `_scan_video_segments` must check BOTH legacy `*_segments` dirs AND this flat dir |
| 39 | Extension detection | Extensionless `source_file` (video IDs) needs `if not file_ext: is_video = True` — exists in `xml_export.py` (2) and `otio_builder.py` (2) |
| 40 | project.name vs folder | `config.downloaded_videos_dir` uses `project.name` (e.g., `matcher-alt`), NOT the project folder name (e.g., `6__2026-01-15`) — path mismatches are silent |
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
| 37 | Non-locking file reads | Python `open()` on Windows blocks other writers — use `os.open(path, os.O_RDONLY \| os.O_BINARY)` for files Ralph writes concurrently (metrics.csv, prd.json, sprint_history.json) |
| 41 | PS string interpolation | Em dashes and `$()` in double-quoted `Write-Host` strings cause cascading parse errors — use `--` and extract to `$var` first |
| 42 | PS syntax check | Use `[Parser]::ParseFile()` in a separate `.ps1` script to validate — inline `powershell -Command` quoting is unreliable |
| 43 | PS regex lookbehinds | .NET regex doesn't support variable-length lookbehinds — use sequential `if`/`elseif` instead of `(?<!pattern?)` |

### Bug Fixing

**Fix first, explain second.** Always attempt a fix within the first 2-3 messages, even if speculative. Do not spend extended time investigating without producing an actionable patch.

**Dict-vs-object pattern:** When hitting `AttributeError` on a dict (e.g., `'dict' object has no attribute 'end_time'`):
1. Check if the variable is a `dict` where a dataclass/object was expected
2. Search for where the variable is assigned — look for JSON loads, cache returns, or dict literals
3. Fix with dict access (`obj['key']`) or convert upstream to the correct type
4. This is the #1 recurring bug class in this codebase

**Fix one, find all:** After fixing any bug, always `grep` for the same anti-pattern across the entire codebase and fix all instances in one pass.

**Add types at fix time:** When fixing attribute-related bugs, add type annotations to the function and its callers so mypy would catch the issue at lint time.

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

**CRITICAL: Config Rule #1** - Always update BOTH files when adding config:
- Update `src/config/sections/<section>.py` (dataclass definition)
- Update `config.yaml` (actual values)

**CRITICAL: Config Rule #2** - Nested dataclass fields load as dict from YAML:
- Must add `__post_init__` to convert dict → dataclass instance
- Pattern: `if isinstance(self.field, dict): self.field = FieldClass(**(self.field or {}))`

**Pattern: dataclass + __post_init__ + yaml + exports**

1. Add field to dataclass in `src/config/sections/<section>.py`
2. Add `__post_init__` if nested dataclass (nested fields load as `dict` - convert in post_init)
3. Add to `config.yaml` with comment explaining the purpose
4. Export from `src/config/sections/__init__.py` if new section
5. Access in code with `getattr()` fallback for backward compatibility
6. Verify: `python -m py_compile src/config/sections/<section>.py`

**Example - Adding a new simple field:**
```python
# 1. In src/config/sections/video_search.py
@dataclass
class VideoSearchConfig:
    max_results: int = 50  # Default value

# 2. In config.yaml
video_search:
  max_results: 50  # Maximum videos to search per query

# 3. In code (use getattr for backward compatibility)
max_results = getattr(config.video_search_config, 'max_results', 50)
```

**Example - Adding a nested config section:**
```python
# 1. In src/config/sections/video_search.py
@dataclass
class SearchBudgetConfig:
    max_searches: int = 100
    auto_scale: bool = True

    def __post_init__(self):
        # Nested dataclass fields load as dict from YAML
        if isinstance(self.max_searches, dict):
            self.max_searches = self.max_searches.get('value', 100)
        if isinstance(self.auto_scale, dict):
            self.auto_scale = self.auto_scale.get('value', True)

@dataclass
class VideoSearchConfig:
    search_budget: SearchBudgetConfig = None

    def __post_init__(self):
        if self.search_budget is None or isinstance(self.search_budget, dict):
            self.search_budget = SearchBudgetConfig(**(self.search_budget or {}))

# 2. In config.yaml
video_search:
  search_budget:
    max_searches: 100
    auto_scale: true
```

**Config migration checklist:**
- [ ] Add field to dataclass in `src/config/sections/<section>.py`
- [ ] Add `__post_init__` for nested dataclasses (converts dict → object)
- [ ] Add to `config.yaml` with descriptive comment
- [ ] Export from `src/config/sections/__init__.py` if new section
- [ ] Update `src/config/base.py` to include new section in `Config` dataclass
- [ ] Run `python -m py_compile` to verify syntax
- [ ] Test with `--dry-run` to ensure loading works

### Claude Code Session Data

| Source | Location | Notes |
|--------|----------|-------|
| Session conversations | `~/.claude/projects/<project-key>/*.jsonl` | First `type: "user"` entry classifies session |
| Prompt history | `~/.claude/history.jsonl` | Per-prompt: display, timestamp, sessionId, project |
| Daily aggregates | `~/.claude/stats-cache.json` | Messages, sessions, tool calls, tokens by model |
| Telemetry | `~/.claude/telemetry/*.json` | Sparse (failed events only); `tengu_init` has `print` flag |
| Session insights script | `scripts/insights/analyze_sessions.py` | Classifies Ralph vs interactive, generates split report |

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

### Chapter and Listicle Configuration

This section documents all configuration options for chapter detection, chapter-aware matching, and listicle topic extraction.

#### Video Search - Chapter/Topic Queries

| Config | Location | Description |
|--------|----------|-------------|
| `use_chapter_queries` | `video_search.use_chapter_queries` | Use chapter topics for targeted search queries (US-98-005) |
| `listicle_topic_as_search_terms` | `video_search.listicle_topic_as_search_terms` | Use listicle topics for targeted search queries (US-98-008) |

#### Chapter Detection

| Config | Location | Description |
|--------|----------|-------------|
| `chapter_detection.enabled` | `matching.chapter_detection.enabled` | Enable enhanced multi-pass chapter detection |
| `chapter_detection.use_validation_pass` | Pass 3 | Cross-validate chapters with LLM |
| `chapter_detection.use_boundary_refinement` | Pass 2 | Refine boundaries with embeddings |
| `chapter_detection.default_strategy` | `'topic'` or `'location'` | Detection strategy |
| `chapter_detection.min_chapter_confidence` | 0.5 | Filter low-confidence chapters |
| `chapter_detection.min_chapter_segments` | 3 | Minimum segments per chapter |
| `chapter_detection.max_chapters` | 20 | Maximum chapters to detect |

#### Video Metadata Extraction

| Config | Location | Description |
|--------|----------|-------------|
| `extract_video_chapters` | `matching.metadata.extract_video_chapters` | Extract chapter markers from video |
| `parse_description_chapters` | `matching.metadata.parse_description_chapters` | Parse chapter timestamps from description text |
| `chapter_enriched_embeddings` | `matching.metadata.chapter_enriched_embeddings` | Include chapter title in embedding text |

#### Chapter-Aware Matching

| Config | Location | Description |
|--------|----------|-------------|
| `chapter_matching_enabled` | `matching.chapter_matching_enabled` | Enable chapter-based topic filtering |
| `enforce_chapter_boundaries` | `matching.enforce_chapter_boundaries` | Penalize cross-chapter matches (US-95-004) |
| `cross_chapter_penalty` | `matching.cross_chapter_penalty` | Penalty amount for cross-chapter matches |
| `prefer_chapter_aligned_segments` | `matching.prefer_chapter_aligned_segments` | Prefer segments aligned with chapter boundaries (US-95-011) |
| `chapter_alignment_boost` | `matching.chapter_alignment_boost` | Boost amount for chapter-aligned segments |

#### Chapter Coherence Scoring

| Config | Location | Description |
|--------|----------|-------------|
| `chapter_coherence_enabled` | `matching.scoring.chapter_coherence_enabled` | Enable chapter coherence scoring (US-98-006) |
| `chapter_coherence_weights` | `scoring.chapter_coherence_weights` | Weights for count similarity, topic overlap, transition pattern |
| `chapter_coherence_boost_max` | 0.08 | Maximum boost for coherent chapter structure |
| `chapter_coherence_penalty_max` | -0.05 | Maximum penalty for incoherent structure |
| `chapter_match_confidence_min` | `matching.scoring.chapter_match_confidence_min` | Minimum confidence for chapter-aligned boost (US-105-004) |

#### Chapter Grouping

| Config | Location | Description |
|--------|----------|-------------|
| `chapter_grouping.enabled` | `matching.chapter_grouping.enabled` | Enable chapter-level source grouping (US-70-011) |
| `chapter_grouping.source_consistency_boost` | 0.03 | Boost when same source reused within chapter |
| `chapter_grouping.coherence_penalty_threshold` | 5 | Max unique sources per chapter before penalty |
| `chapter_grouping.min_source_diversity` | 2 | Min unique sources per chapter (US-77-007) |
| `chapter_grouping.chapter_topic_match_boost` | [0.05, 0.15] | [min, max] boost for topic match within chapter |
| `chapter_grouping.chapter_topic_mismatch_penalty` | -0.10 | Penalty when video topic doesn't match chapter |
| `chapter_grouping.multi_chapter_assignment_strategy` | `'best_match'` | Strategy for segments spanning chapters (US-105-009): `'first'`, `'split'`, `'best_match'` |

#### Listicle Topic Extraction

| Config | Location | Description |
|--------|----------|-------------|
| `listicle_topic.use_llm_topic_extraction` | `matching.listicle_topic.use_llm_topic_extraction` | Use LLM when simple extraction yields <3 keywords (US-105-005) |
| `listicle_topic.min_keywords_for_simple` | 3 | Minimum keywords before LLM fallback triggers |

#### Iterative Matching - Chapter Awareness

| Config | Location | Description |
|--------|----------|-------------|
| `iterative_chapter_boost` | `iterative.iterative_chapter_boost` | Boost for chapter-aligned videos during iterative matching (US-105-007) |

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

**Test markers:** `fast`, `integration`, `stress`, `simulation`, `requires_api`, `requires_network`, `flaky`

### Flaky Test Handling

Tests marked with `@pytest.mark.flaky` are automatically retried using `pytest-rerunfailures`:

```bash
# Run flaky tests with retries (default behavior)
pytest tests/ -v

# Disable retries for debugging
pytest tests/ --reruns 0

# Run only flaky tests
pytest tests/ -m flaky -v
```

**Adding flaky marker to tests:**
```python
import pytest

@pytest.mark.flaky(reruns=2, reruns_delay=0.5)
def test_sometimes_fails():
    # This test will retry up to 2 times on failure
    # with 0.5 second delay between attempts
    assert some_condition
```

**Configuration in pytest.ini:**
- Tests marked with `@pytest.mark.flaky` are retried up to 2 times by default
- Use `--reruns N` to override the number of retries
- Use `--reruns-delay N` to set delay between retries

**Flaky test patterns:**
- See `tests/test_flaky_detection.py` for examples
- Mark tests that fail due to timing, network, or race conditions
- Document the flakiness reason in the test docstring

### Coverage Configuration

Coverage is configured in `.coveragerc`:

```ini
[report]
fail_under = 85  # CI enforces 85% minimum
```

**Coverage commands:**
```bash
# With coverage report (term + missing lines)
pytest tests/ --cov=src --cov-report=term-missing

# Generate HTML report
pytest tests/ --cov=src --cov-report=html  # Outputs to htmlcov/

# XML report for CI integration
pytest tests/ --cov=src --cov-report=xml

# View summary
coverage report
coverage html
```

**Coverage goals:**
- Overall: 85% minimum (enforced by `fail_under = 85` in `.coveragerc`)
- Core modules (matching, LLM client): 80%+
- New code: 90%+

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
| `--output-only` no effect | Was dead code before 2026-02-05 fix | Now wired in main.py |
| XML clips "not found" | Segments in wrong dir | Check `project.name` vs folder name; scan both flat + `*_segments` dirs |
| Empty `<media>` in XML | Extensionless source_file | Rule 39: add `if not file_ext: is_video = True` |
| Checkpoint corrupted | Various | Restore from `checkpoint.backup.json` |
| Subprocess crash | Windows encoding | Rule 27: Add `encoding='utf-8', errors='replace'` |
| Videos skipped (budget_exhausted) | Logs show `EXHAUSTED` | Increase `retry_budget.max_attempts` or check for rate limiting |
| Evidence 0/N rejected | LLM evidence returns all-false | `Confirm-CriteriaEvidence` returns array (not `$null`) when LLM output unparseable — keyword fallback triggers on `parsedCount == 0` or all-false safety net |

### Config Troubleshooting

**"AttributeError: 'dict' object has no attribute 'X'"**
- **Cause:** Nested dataclass field loaded as dict instead of object
- **Fix:** Add `__post_init__` to convert dict to dataclass instance
- **Pattern:** `if isinstance(self.field, dict): self.field = FieldClass(**(self.field or {}))`

**"Missing config section 'X'" or fields not loading**
- **Cause:** Config section not added to main `Config` dataclass in `base.py`
- **Fix:** Add field to `Config` dataclass: `search_budget: SearchBudgetConfig = None`
- **Verify:** Check exports in `src/config/sections/__init__.py`

**Config values not taking effect**
- **Cause:** Hardcoded values in stages override config (Rule #1 violation)
- **Fix:** Replace hardcoded values with `getattr(config, 'field', default)`
- **Scan:** `grep -r "min_duration\s*=" src/stages/` to find hardcoded values

**YAML parsing errors on startup**
- **Cause:** Indentation mismatch or invalid YAML syntax in config.yaml
- **Fix:** Use `python -c "import yaml; yaml.safe_load(open('config.yaml'))"` to validate

**TypeError on config access**
- **Cause:** Field type changed but old config.yaml has wrong type
- **Fix:** Add backward compatibility with `getattr()`: `getattr(cfg, 'field', default_value)`

**New config option not in --help output**
- **Cause:** Config not wired in CLI argument parsing
- **Fix:** Check `src/cli/config_utils.py` for how options are exposed

### Debug Workflows

**Pipeline not progressing?** Use `/watch <project>` or check logs directly.

**Match quality issues?** Check: match counts, confidence scores, video variety, empty source_file warnings.

**Output ≠ Success:** Pipeline completing doesn't mean quality is acceptable. Always verify results.

### MiniMax Model (Alternative to Claude)

MiniMax M2.5 (released Feb 2026) provides Anthropic-compatible API via Claude Code CLI:
- **SWE-Bench:** 80.2%, **Context:** 204.8K tokens, **Speed:** matches Claude Opus 4.6
- **Pricing:** $0.30/M input, $1.20/M output (10% of Claude Sonnet 4.5)

**Setup in `~/.claude/settings.json`:**
```json
{
  "env": {
    "ANTHROPIC_BASE_URL": "https://api.minimax.io/anthropic",
    "ANTHROPIC_AUTH_TOKEN": "YOUR_MINIMAX_API_KEY",
    "ANTHROPIC_MODEL": "MiniMax-M2.5"
  }
}
```
Then update `scripts/ralph/config/ralph-config.json` to use `MiniMax-M2.5` model.

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

See [SESSION_HISTORY.md](SESSION_HISTORY.md) for dated changelog. Recent: Quick Edit Mode periodic re-check safeguard (console output freeze prevention), Pre-flight fast-path for pre-implemented stories (infinite loop fix), evidence gate tolerant parsing + all-false safety net, Ollama embedding provider (nomic-embed-text) with asymmetric search and provider-specific cache keys, Carlini-inspired Ralph improvements, Terminal OUTPUT stage checkpoint false WARNING fix, FAISS index space mismatch fix.
