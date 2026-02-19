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
| `--dry-run` | Preview pipeline execution plan without running stages |
| `--force-rematch` | Force rematch all videos, ignoring cached matches |
| `--non-interactive` | Skip prompts, use defaults |
| `--use-keywords [PRESET]` | Use saved keywords |
| `--save-keywords [NAME]` | Save extracted keywords |
| `--keywords`, `-k` | Number of keywords to extract |
| `--list-keywords` | List saved keyword presets and exit |
| `--validate-config` | Validate config file and exit |
| `--validate-config-json` | Validate config and output as JSON |
| `--refresh-entities` | Force re-download entity images |
| `--export-metrics` | Export rate limit metrics to JSON file |
| `--export-caption-metrics` | Export caption fetch metrics to JSON |
| `--export-resource-metrics` | Export pipeline resource monitoring metrics (CPU/memory per stage) |
| `--export-pipeline-graph` | Export pipeline dependency graph (supports .dot, .png, .svg) |
| `--caption-language` | Preferred caption language (ISO 639-1) |
| `--validate-captions` | Validate caption configuration and exit |
| `--test-fetch N` | With --validate-captions: fetch N sample captions |
| `--cleanup-caption-cache` | Remove stale caption cache entries |

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

> Rules 1-2 detailed in "Adding New Config Options"; Rule 6 in "Config Access Pattern".

| # | Rule | Key Point | Details |
|---|------|-----------|---------|
| 1 | Config sync | Update BOTH `src/config/sections/*.py` AND `config.yaml` | See "Adding New Config Options" section |
| 2 | `__post_init__` | Nested dataclass fields load as `dict` - convert in post_init | See "Adding New Config Options" section |
| 3 | DownloadedVideo | Use `file=`, `duration_tier=` NOT `path=`, `tier=` | |
| 6 | Dict/Object config | Handle both: `vc.get()` if dict, `getattr()` if object | See "Config Access Pattern" section |
| 7 | Embeddings truthiness | Use `is_embeddings_empty()` - numpy fails bool | |
| 9 | **LLM Client** | Use `src/llm_client/` for ALL LLM calls | |
| 10 | Test non-interactive | Tests MUST use `--non-interactive` | |
| 11 | Dataclass imports | Import from `src/state.py` or `src/config/` only | |
| 14 | OTIO paths | Use `file:///E:/...` URLs, forward slashes, skip audio-only | |
| 22 | Caption-first (always on) | Video IDs not file paths - don't filter as `caption_only` | |
| 27 | Subprocess encoding | ALL `subprocess.Popen/run` with `text=True` MUST add `encoding='utf-8', errors='replace'` | |
| 28 | Download one-at-a-time | `_download_by_ids` must loop per video, NOT batch | |
| 29 | LLM response format | Use `ResponseFormat.TEXT` for non-JSON responses, not `JSON_ARRAY` | |
| 30 | Per-project config | Use `--config custom.yaml` flag — project_config.yaml auto-merge removed | |
| 31 | Pipeline stages source | `src/checkpoint.py:STAGE_ORDER` is truth; sync `scripts/ralph/config/ralph-config.json` | |
| 32 | Ralph file paths | Always use `state/prd.json`, `state/queue.json`, `session/prompt.md`, `config/ralph-config.json` — never root-level | |
| 33 | Ralph seed story prompts | In `Build-StoryPrompt`, seed stories emit instructions+criteria BEFORE notes (notes are background context, safe to truncate) | |
| 34 | Ralph prompt maxLength | `ralph-config.json` `prompts.maxLength` (default 10000) truncates story prompts — if stories fail mysteriously, check truncation first | |
| 35 | PowerShell threading | Never use `[System.Threading.Thread]` with PS cmdlets — use `[powershell]::Create()` with `.AddArgument()` for background work; raw threads crash the host process | |
| 36 | Console Quick Edit | Ralph disables Quick Edit Mode at startup (`Disable-QuickEditMode`) — clicking the console window freezes ALL `Write-Host` calls, blocking the monitoring loop, stall detection, and heartbeat while Claude keeps running | |
| 37 | Non-locking file reads | Python `open()` on Windows blocks other writers — use `os.open(path, os.O_RDONLY \| os.O_BINARY)` for files Ralph writes concurrently (metrics.csv, prd.json, sprint_history.json) | |
| 38 | Segment file format | DOWNLOAD_SEGMENTS saves flat `{vid}_{start}_{end}.mp4` in `config.downloaded_videos_dir`; `_scan_video_segments` must check BOTH legacy `*_segments` dirs AND this flat dir | |
| 39 | Extension detection | Extensionless `source_file` (video IDs) needs `if not file_ext: is_video = True` — exists in `xml_export.py` (2) and `otio_builder.py` (2) | |
| 40 | project.name vs folder | `config.downloaded_videos_dir` uses `project.name` (e.g., `matcher-alt`), NOT the project folder name (e.g., `6__2026-01-15`) — path mismatches are silent | |
| 41 | PS string interpolation | Em dashes and `$()` in double-quoted `Write-Host` strings cause cascading parse errors — use `--` and extract to `$var` first | |
| 42 | PS syntax check | Use `[Parser]::ParseFile()` in a separate `.ps1` script to validate — inline `powershell -Command` quoting is unreliable | |
| 43 | PS regex lookbehinds | .NET regex doesn't support variable-length lookbehinds — use sequential `if`/`elseif` instead of `(?<!pattern?)` | |

### Script Logging Standard

All scripts in `scripts/` directory must use `script_utils` for standardized output.

**Import pattern:**
```python
from script_utils import print_ok, print_warn, print_error, print_info, print_header
```

**Function usage:**
| Function | When to use |
|----------|-------------|
| `print_ok(msg)` | Success messages, positive results |
| `print_warn(msg)` | Non-fatal issues, expected edge cases |
| `print_error(msg, exit_code)` | Fatal errors, failures |
| `print_info(msg)` | Verbose-only info (shown with -v) |
| `print_header(title)` | Section headers |

**Output format:**
- `[OK]` - Success messages
- `[WARN]` - Warning messages
- `[ERROR]` - Error messages
- No emoji
- 2-space indent for content

**Migration:** See `docs/script_migration_guide.md` for detailed step-by-step instructions.

### Script Standards Validation

Run `scripts/validate_script_standards.py` to check if scripts meet standards:

```bash
# Check all scripts
python scripts/validate_script_standards.py

# Verbose output
python scripts/validate_script_standards.py --verbose

# Auto-fix issues
python scripts/validate_script_standards.py --fix

# Pre-commit mode (quiet, exit code only)
python scripts/validate_script_standards.py --pre-commit
```

**Checks performed:**
- Shebang line (`#!/usr/bin/env python3`)
- Module-level docstring with usage
- Import from `script_utils`
- Proper `sys.path` setup
- `os.chdir(project_root)` for relative path consistency
- `argparse` import

**Pre-commit integration:**
```bash
pip install pre-commit
pre-commit install
```

The validation runs automatically in CI (see `.github/workflows/tests.yml` - `script-standards` job).

### Standard Argument Patterns (US-132-008)

All scripts in `scripts/` directory should use standard argument patterns for consistency.

**Helper functions in script_utils:**
```python
from script_utils import (
    add_project_argument,
    add_verbose_argument,
    add_quiet_argument,
    add_standard_arguments,
    set_verbosity,
    get_verbosity
)
```

**Standard arguments:**
| Argument | Short | Description | Default |
|----------|-------|-------------|---------|
| `--project` | `-p` | Path to project directory | Optional |
| `--verbose` | `-v` | Enable verbose output | `False` |
| `--quiet` | `-q` | Suppress non-essential output | `False` |
| `--json` | `-j` | Output results as JSON | `False` |
| `--yes` | `-y` | Skip confirmation prompts | `False` |

**Verbosity levels:**
- `set_verbosity(0)` - Quiet: only errors shown
- `set_verbosity(1)` - Normal: errors + warnings (default)
- `set_verbosity(2)` - Verbose: all output including info

**Usage patterns:**

1. **Individual arguments:**
```python
import argparse
from script_utils import add_project_argument, add_verbose_argument, add_quiet_argument, set_verbosity

parser = argparse.ArgumentParser(description="My script")
add_project_argument(parser, required=True)
add_verbose_argument(parser)
add_quiet_argument(parser)

args = parser.parse_args()

# Set verbosity based on flags
if args.quiet:
    set_verbosity(0)
elif args.verbose:
    set_verbosity(2)
else:
    set_verbosity(1)
```

2. **All standard arguments at once:**
```python
from script_utils import add_standard_arguments, set_verbosity

parser = argparse.ArgumentParser(description="My script")
add_standard_arguments(parser, add_project=True, add_verbose=True, add_quiet=True, project_required=True)

args = parser.parse_args()

# Set verbosity
if args.quiet:
    set_verbosity(0)
elif args.verbose:
    set_verbosity(2)
else:
    set_verbosity(1)
```

3. **Using cli_helpers.parse_args (convenience wrapper):**
```python
from utils.cli_helpers import parse_args
from script_utils import set_verbosity

args = parse_args(
    description="My script",
    add_project_arg=True,
    add_verbose=True,
    add_quiet=True
)

# Set verbosity
if args.quiet:
    set_verbosity(0)
elif args.verbose:
    set_verbosity(2)
else:
    set_verbosity(1)
```

**Scripts using standard patterns:**
- `cleanup_project.py` - Uses `--project/-p`, `--verbose/-v`, `--quiet/-q`, `--json/-j`, `--yes/-y`
- `project_health.py` - Uses `--project/-p`, `--verbose/-v`, `--quiet/-q`, `--output/-o`
- `benchmark.py` - Uses `--project/-p`, `--verbose/-v`, `--quiet/-q`, `--json/-j` in all subcommands
- `validate_config.py` - Uses `--verbose/-v`, `--quiet/-q`, `--quick`

### Progress Bar Support

Scripts can use tqdm progress bars via `script_utils`:

```python
from script_utils import progress_bar, track_progress, spinner

# Method 1: Wrap iterable (simplest)
for item in progress_bar(items, desc="Processing"):
    process(item)

# Method 2: Manual tracking with context manager
with track_progress(total=100, desc="Downloading") as pbar:
    for i in range(100):
        download(i)
        pbar.update(1)

# Method 3: Indeterminate spinner (loading state)
with spinner("Loading data") as sp:
    data = fetch_data()
# Spinner auto-closes on exit
```

All functions gracefully degrade if tqdm is not installed.

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

### Major Version Config Migration (v3 -> v4)

The config system supports automatic migration from v3 to v4 via `ConfigMigration` class in `src/config/utils.py`.

**What migration does (v3 -> v4):**
- Adds `region_backoff` section with default values
- Adds `search_budget` section with default values
- Adds `search_budget_aware` and `auto_distribute_budget` to `video_search`
- Updates `project.version` to "4.0.0"

**Auto-migration:**
- Triggered automatically in `Config.from_yaml()` when version mismatch detected
- Creates backup in `.config_backups/` directory before migration
- Raises `ConfigMigrationError` if migration fails

**Manual migration:**
```python
from src.config.utils import ConfigMigration

migrator = ConfigMigration()

# Check if migration needed
if migrator.needs_migration("config.yaml"):
    migrator.backup_config("config.yaml")
    migrator.migrate_config("config.yaml", "4.0.0")

# Rollback if needed
migrator.rollback("config.yaml", "path/to/backup.yaml")
```

**Adding new migrations:**
1. Add migration function: `def _migrate_v4_to_v5(self, config: Dict) -> Dict`
2. Register in `MIGRATIONS` dict: `("4.0.0", "5.0.0"): _migrate_v4_to_v5`
3. Update `CURRENT_CONFIG_VERSION` in `src/config/utils.py`

### Adding Config Migrations (Step-by-Step Guide)

This section documents how to add new config migrations for future major versions. See `src/config/utils.py` for the implementation.

#### MIGRATIONS Dict Pattern

The `MIGRATIONS` dict maps version tuples to migration functions:

```python
self.MIGRATIONS = {
    ("3.0.0", "4.0.0"): self._migrate_v3_to_v4,
    ("4.0.0", "5.0.0"): self._migrate_v4_to_v5,
}
```

#### Migration Function Template

```python
def _migrate_v4_to_v5(self, config: Dict) -> Dict:
    config = copy.deepcopy(config)
    if "project" not in config: config["project"] = {}
    config["project"]["version"] = "5.0.0"
    # Add migrations here...
    return config
```

**Adding new sections:**
```python
if "new_section" not in config:
    config["new_section"] = {"enabled": False, "option1": "default"}
```

**Migrating deprecated fields:**
```python
if "old_field" in config.get("matching", {}):
    config["matching"]["new_field"] = config["matching"].pop("old_field")
```

**Rollback:**
```python
migrator.rollback("config.yaml", "path/to/backup.yaml")
```

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

### Resource Monitoring Export (US-125-003)

`python main.py --project "..." --export-resource-metrics resource_metrics.json`

**Output:** `history[]` (before/after each stage), `summary` (aggregates), `per_stage` (per stage)

### Pipeline Dependency Graph Export (US-125-012)

`python main.py --export-pipeline-graph pipeline.dot|png|svg`

DOT: No deps; PNG/SVG: requires `graphviz` package + binary

## Configuration

### Self-Healing

Enabled by default. Strategies: `aggressive`, `conservative` (default), `interactive`, `minimal`

```yaml
healing:
  enabled: true
  strategy: "conservative"
  max_attempts_per_stage: 3
```

### Runtime Config Hot-Reload with Callbacks (US-112-011)

Config supports runtime reload with change callback notifications.

```python
config = get_config()
config.register_change_callback(my_callback)  # Global
config.register_change_callback(my_callback, section='matching')  # Section-specific
reloaded = config.reload()  # Returns True if file changed
```

**API:** `register_change_callback(callback, section=None)`, `unregister_change_callback()`, `reload()`

### Environment Variable Overrides (US-128-002)

Config values can be overridden at runtime using environment variables with the `MATCHER_` prefix. This enables CI/CD pipelines and container deployments to customize config without modifying files.

**Naming Convention:**
- `MATCHER_<SECTION>_<FIELD>=value` for top-level section fields
- `MATCHER_<SECTION>_<NESTED>_<FIELD>=value` for nested fields (e.g., download.mullvad.enabled)

**Examples:**
```bash
# Override matching threshold
export MATCHER_MATCHING_MIN_CONFIDENCE=0.85

# Enable Mullvad VPN
export MATCHER_DOWNLOAD_MULLVAD_ENABLED=true

# Override video search settings (section with underscore)
export MATCHER_VIDEO_SEARCH_MAX_TOTAL_RESULTS=150
```

**Behavior:**
- Environment variables override config.yaml values
- Validation runs after environment overrides are applied
- Invalid values will cause validation errors
- Nested config sections use double underscore in the env var name

### Config Value Precedence (US-142-010)

**Precedence:** env > profile > yaml > default

**Query:** `python main.py --show-precedence [field]`

### Bypass & Escalation (yt-dlp)

5-tier system using curl_cffi TLS fingerprint spoofing + VPN + circuit breakers:
- **Tier 1**: `--impersonate Chrome-136:Macos-15` (always on)
- **Tier 2**: + `--extractor-args youtube:player_client=web_safari,tv_downgraded,web` (on 403)
- **Tier 3**: + cookie rotation (on continued 403s)
- **Tier 4**: + Mullvad VPN IP rotation (when cookies exhausted)
- **Tier 5**: + Per-keyword circuit breaker (isolates rate-limited keywords)

**Circuit Breaker Integration:**
- `CircuitBreakerCoordinator`: Coordinates across download/caption/video_search components
- `PerKeywordCircuitBreaker` (US-109-004): Per-keyword failure tracking with global fallback
- `RateLimitPredictor` (US-109-008): Predictive detection using historical patterns
- `PerKeywordSpeedTracker`: Speed-based rate limit detection (US-113-007)

Key classes: `ImpersonationManager`, `EscalationManager`, `CookieMethodFallback`, `MullvadVPN`, `PerKeywordCircuitBreaker`, `RateLimitPredictor` in `src/downloader/`

### Mullvad VPN Integration (Tier 4)

**Prereqs:** Install Mullvad (winget), verify with `mullvad status`

**Config:** `download.mullvad.enabled`, `preferred_countries`, `rotation_strategy`, `max_rotations_per_session`, `verification_timeout`

**Behavior:** Triggers when cookie rotation exhausted; resets circuit breaker, refreshes budgets; verifies via `am.i.mullvad.net/json`

### Browser-Based Cookie Extraction (US-143-005)

Auto-extract cookies from browser profiles (Chrome, Firefox, Edge, Brave, Opera, Safari) instead of manual export.

**Config:** `download.cookie_rotation.auto_extract_cookies`, `browser_priority_order`, `browser_profiles[]`, `cookie_freshness_validation`, `auto_refresh_cookies`

**Browser cookie paths:** Chrome: `%LOCALAPPDATA%/Google/Chrome/User Data/Default/Cookies`, Firefox: `%APPDATA%/Mozilla/Firefox/Profiles`, Edge: `%LOCALAPPDATA%/Microsoft/Edge/User Data/Default/Cookies`, Brave: `%LOCALAPPDATA%/BraveSoftware/Brave-Browser/User Data/Default/Cookies`

**Manual export:** Use "Get cookies.txt LOCALLY" browser extension → save to `cookies/main.txt`

### Per-Keyword Circuit Breaker (US-109-004)

Per-keyword circuit breaker isolates rate-limited keywords while allowing others to continue. Global fallback trips when >50% keywords are rate-limited.

**Key features:** Per-keyword failure tracking, adaptive pause (based on keyword history), global fallback, speed-based triggering (US-113-007)

**Config:** `download.per_keyword_circuit_breaker.enabled`, `consecutive_failures_threshold`, `pause_seconds`, `max_pause_seconds`, `jitter_factor`, `global_fallback_threshold`, `speed_threshold_mbps`

**Related:** `PerKeywordCircuitBreaker` in `src/downloader/per_keyword_circuit_breaker.py`

### Predictive Rate Limit Detection (US-109-008)

Predictive rate limit detection using historical patterns from time-of-day and day-of-week.

**Features:** Track events by time window (MORNING 6-12, AFTERNOON 12-18, EVENING 18-22, OVERNIGHT 22-6), day-of-week; predict likelihood (0.0-1.0); auto-increase budget when likelihood > 0.6

**Prediction:** Evening highest (40% base), weekends 1.3x, morning moderate (15%), overnight lowest (10%)

**Related:** `RateLimitPredictor` in `src/downloader/rate_limit_predictor.py`

### Region-Specific Backoff (US-109-011)

Apply region-specific backoff multipliers when using VPN rotation.

**Config:** `download.region_backoff.enabled`, `us_multiplier`, `eu_multiplier` (1.2), `asia_multiplier` (1.5), `other_multiplier` (2.0)

**Region mapping:** US/CA → US; GB, DE, NL, etc. → EU; JP, SG, KR, etc. → ASIA; others → OTHER

**Requires:** `mullvad.enabled=true`

### Caption Retry Budget

Controls caption fetch attempts before fallback to transcription. Prevents infinite retry loops.

**Defaults:** `max_attempts`: 100, `auto_scale`: true, `attempts_per_video`: 2.0, `max_backoff_time_seconds`: 300.0

**Config:** `download.caption_first.retry_budget.enabled`, `max_attempts`, `max_backoff_time_seconds`, `auto_scale`

**Auto-scaling:** Scales UP for large batches: `scaled_max = max(original_max, batch_size * attempts_per_video)`

**Debug:** Check logs for `CaptionRetryBudget: EXHAUSTED`, `budget_exhausted` in skipped reasons

### Data Drift Detection (US-88-006, US-125-005)

Configurable cross-stage data drift detection that warns when output counts drop unexpectedly.

**Config:** `pipeline.drift_rules.enabled`, `track_history`, `max_history`, `rules[]`

**Default rules:** CAPTION: caption_results >= 80% of video_ids; MATCH: text_metadata >= 80%; OUTPUT: matches >= 50% of voiceover_segments

**Threshold types:** `ratio` (0.0-1.0), `absolute_count`, `percentage` (0-100)

**Severity:** `warning` (non-blocking) or `error` (stops pipeline)

### Chapter and Listicle Configuration

| Config | Location | Description |
|--------|----------|-------------|
| `use_chapter_queries` | `video_search.use_chapter_queries` | Use chapter topics for targeted search (US-98-005) |
| `listicle_topic_as_search_terms` | `video_search.listicle_topic_as_search_terms` | Use listicle topics for search (US-98-008) |
| `extract_video_chapters` | `matching.metadata.extract_video_chapters` | Extract chapter markers from video |
| `parse_description_chapters` | `matching.metadata.parse_description_chapters` | Parse chapter timestamps from description |
| `chapter_enriched_embeddings` | `matching.metadata.chapter_enriched_embeddings` | Include chapter title in embedding text |
| `chapter_matching_enabled` | `matching.chapter_matching_enabled` | Enable chapter-based topic filtering |
| `enforce_chapter_boundaries` | `matching.enforce_chapter_boundaries` | Penalize cross-chapter matches (US-95-004) |
| `cross_chapter_penalty` | `matching.cross_chapter_penalty` | Penalty for cross-chapter matches |
| `prefer_chapter_aligned_segments` | `matching.prefer_chapter_aligned_segments` | Prefer segments aligned with chapter boundaries |
| `chapter_alignment_boost` | `matching.chapter_alignment_boost` | Boost for chapter-aligned segments |
| `no_chapter_fallback_strategy` | `matching.no_chapter_fallback_strategy` | Fallback when no chapters: 'global' or 'segment' |
| `chapter_match_confidence_min` | `matching.scoring.chapter_match_confidence_min` | Min confidence for chapter boost |
| `chapter_grouping.enabled` | `matching.chapter_grouping.enabled` | Enable chapter-level source grouping |
| `chapter_grouping.source_consistency_boost` | 0.03 | Boost when same source reused within chapter |
| `chapter_grouping.coherence_penalty_threshold` | 5 | Max unique sources per chapter before penalty |
| `chapter_grouping.min_source_diversity` | 2 | Min unique sources per chapter (US-77-007) |
| `chapter_grouping.chapter_topic_match_boost` | [0.05, 0.15] | [min, max] boost for topic match |
| `chapter_grouping.multi_chapter_assignment_strategy` | `'best_match'` | Strategy for segments spanning chapters |
| `listicle_topic.use_llm_topic_extraction` | `matching.listicle_topic.use_llm_topic_extraction` | Use LLM when simple extraction yields <3 keywords |
| `iterative_chapter_boost` | `iterative.iterative_chapter_boost` | Boost for chapter-aligned videos during iterative matching |
| `temporal_coherence_enabled` | `matching.temporal_coherence_enabled` | Enable temporal coherence scoring (US-77-003) |
| `temporal_coherence_same_source_boost` | `matching.temporal_coherence_same_source_boost` | Boost for clips from same source as adjacent |
| `temporal_coherence_context_switch_penalty` | `matching.temporal_coherence_context_switch_penalty` | Penalty for jarring context switches |

**Temporal Coherence:** Same-source boost (+5%) when adjacent segments from same video. Jarring context switch penalty (-5%) when moving to unrelated topic between adjacent segments.

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
pytest tests/ -m fast -v --tb=short -x
```

## Troubleshooting

### Common Issues

| Issue | Symptom | Fix |
|-------|---------|-----|
| V8 track empty | No B-roll matches | Check `face_score < 0.3` OR `word_count < threshold` |
| 403 Forbidden | Download failures | Wait 1 hour, check cookies, or use VPN |
| Geo-blocked (E601) | Content not available in region | Use VPN (Tier 4 escalation) |
| Device limit (E701) | Too many devices streaming | Wait for streams to finish |
| Login required (E801) | Content requires authentication | Provide YouTube cookies |
| Empty source_file | Caption-first mode | Rule 22 |
| `--output-only` no effect | Dead code fixed 2026-02-05 | Now wired in main.py |
| XML clips "not found" | Segments in wrong dir | Rule 40: check `project.name` vs folder |
| Empty `<media>` in XML | Extensionless source_file | Rule 39 |
| Checkpoint corrupted | Various | Restore from `checkpoint.backup.json` |
| Subprocess crash | Windows encoding | Rule 27 |
| Videos skipped (budget_exhausted) | Logs show `EXHAUSTED` | Increase `retry_budget.max_attempts` |
| Evidence 0/N rejected | LLM returns all-false | Keyword fallback on `parsedCount == 0` |

### Config Troubleshooting

| Issue | Cause | Fix |
|-------|-------|-----|
| `AttributeError: 'dict' object has no attribute 'X'` | Nested dataclass loads as dict | Rule 2: Add `__post_init__` to convert |
| `Missing config section 'X'` | Section not in `Config` dataclass | Add field to `base.py`, export from `__init__.py` |
| Config values not taking effect | Hardcoded values in stages | Rule 1: Use `getattr(config, 'field', default)` |
| YAML parsing errors | Indentation/syntax in config.yaml | Validate: `python -c "import yaml; yaml.safe_load(open('config.yaml'))"` |
| TypeError on config access | Field type changed, old config has wrong type | Use `getattr()` with defaults |
| New config not in --help | Not wired in CLI | Check `src/cli/config_utils.py` |

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

### Health Check Configuration (US-125-011)

Health checks validate external dependencies before stage execution. Configurable intervals for long-running stages.

**Config:** `pipeline.health_check_interval.default` (seconds), stage-specific overrides (VIDEO_SEARCH: 120, CAPTION: 300, MATCH: 300, DOWNLOAD_SEGMENTS: 600)

**Behavior:** With interval > 0, checks run only after elapsed time; set to 0 for check before each stage.

**Checks:** Network, disk space, memory, embedding provider, FFmpeg, yt-dlp, LLM provider

## Ralph Loop (Autonomous Development)

Location: `scripts/ralph/` — See [scripts/ralph/README.md](scripts/ralph/README.md) for full docs.

**Directory structure:** `config/` (ralph-config.json), `state/` (prd.json, queue.json, sprint_history.json), `session/` (prompt.md, metrics.csv), `archive/`

**CRITICAL:** Always use `scripts/ralph/state/prd.json` — NOT root-level.

**Commands:**
```powershell
.\scripts\ralph\launcher.ps1           # Unified launcher
.\scripts\ralph\ralph.ps1 [-TrueAuto] [-Resume] [-FocusArea <area>]
.\scripts\ralph\watch.ps1 [-Interval 5]  # Dashboard
```
**Modes:** Standard, TrueAuto, Resume, Smart Queue, Ralph's Choice, Ralph's Choice Auto

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

See [SESSION_HISTORY.md](SESSION_HISTORY.md) for dated changelog. Recent fixes: PowerShell switch parameter null crash, PowerShell return value unboxing, grace period exit code, Quick Edit Mode safeguard, pre-flight fast-path, evidence gate parsing + all-false safety net, Ollama embeddings, Carlini improvements, Terminal checkpoint warning, FAISS index mismatch.
