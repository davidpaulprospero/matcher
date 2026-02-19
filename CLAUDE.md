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

This section documents how to add new config migrations for future major versions.

#### 1. MIGRATIONS Dict Pattern

The `MIGRATIONS` dict in `_register_migrations()` maps version tuples to migration functions:

```python
def _register_migrations(self):
    """Register all available migrations."""
    self.MIGRATIONS = {
        ("3.0.0", "4.0.0"): self._migrate_v3_to_v4,
        # Add new migrations here:
        ("4.0.0", "5.0.0"): self._migrate_v4_to_v5,
    }
```

#### 2. Migration Function Signature

Each migration function follows this pattern:

```python
def _migrate_v4_to_v5(self, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Migrate config from v4.0.0 to v5.0.0.

    Args:
        config: Raw config dict (v4.0.0 format)

    Returns:
        Migrated config dict (v5.0.0 format)
    """
    config = copy.deepcopy(config)  # Always deepcopy to avoid mutating input

    # Update project version FIRST
    if "project" not in config:
        config["project"] = {}
    config["project"]["version"] = "5.0.0"

    # Add migrations here...

    return config
```

#### 3. Adding a New Required Section

To add a new required section with defaults:

```python
# Add new section if not present
if "new_section" not in config:
    config["new_section"] = {
        "enabled": False,
        "option1": "default_value",
        "option2": 100,
    }
    logger.info("Added new_section section with defaults")
```

#### 4. Adding a Field to Existing Section

To add a new field to an existing section without overwriting existing values:

```python
if "video_search" in config:
    if "new_option" not in config["video_search"]:
        config["video_search"]["new_option"] = True
        logger.info("Added new_option to video_search")
```

#### 5. Migrating Deprecated Fields

To rename or restructure deprecated fields:

```python
# Migrate old_field to new_field
if "old_field" in config.get("matching", {}):
    if "new_field" not in config.get("matching", {}):
        config["matching"]["new_field"] = config["matching"]["old_field"]
    del config["matching"]["old_field"]
    logger.info("Migrated old_field to new_field in matching")
```

#### 6. Updating CURRENT_CONFIG_VERSION

After implementing the migration, update the version constant:

```python
# In src/config/utils.py
CURRENT_CONFIG_VERSION = "5.0.0"
```

#### 7. Backward Compatibility

For backward compatibility in code, use `getattr()` with defaults:

```python
# In code (not migration)
value = getattr(config.video_search_config, 'new_option', default_value)

# For dict/object compatibility:
if isinstance(config_section, dict):
    value = config_section.get('field_name', default)
else:
    value = getattr(config_section, 'field_name', default)
```

#### 8. Rollback Procedure

If migration fails or needs to be reverted:

```python
from src.config.utils import ConfigMigration

migrator = ConfigMigration()

# Rollback to backup
success = migrator.rollback("config.yaml", "path/to/backup.yaml")

# Or manually restore from .config_backups/ directory
```

Backups are stored in `.config_backups/config.yaml.backup_<timestamp>`

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

Export pipeline resource monitoring metrics using `--export-resource-metrics`:

```bash
python main.py --project "E:\Edit Job\client\project" --export-resource-metrics resource_metrics.json
```

**Output schema:**
```json
{
  "resource_metrics": {
    "format": "json",
    "generated_at": "2026-02-17T12:00:00Z",
    "history": [
      {"phase": "before", "stage_name": "VIDEO_SEARCH", "memory_percent": 50.0, "cpu_percent": 30.0},
      {"phase": "after", "stage_name": "VIDEO_SEARCH", "memory_percent": 70.0, "cpu_percent": 50.0}
    ],
    "summary": {
      "stages_tracked": 2,
      "total_measurements": 4,
      "memory_percent_avg": 60.0,
      "memory_percent_max": 80.0,
      "cpu_percent_avg": 45.0,
      "cpu_percent_max": 60.0
    },
    "per_stage": {
      "VIDEO_SEARCH": {
        "measurements": 2,
        "phases": ["before", "after"],
        "memory_percent_avg": 60.0,
        "memory_percent_max": 70.0,
        "cpu_percent_avg": 40.0,
        "cpu_percent_max": 50.0
      }
    }
  }
}
```

**Fields:**
- `history`: Raw per-stage measurements (before/after each stage)
- `summary`: Aggregate statistics across all stages
- `per_stage`: Aggregated metrics per stage name

### Pipeline Dependency Graph Export (US-125-012)

Export pipeline dependency graph to visualize stage relationships:

```bash
# Export as DOT (Graphviz format)
python main.py --export-pipeline-graph pipeline.dot

# Export as PNG (requires graphviz installed)
python main.py --export-pipeline-graph pipeline.png

# Export as SVG (requires graphviz installed)
python main.py --export-pipeline-graph pipeline.svg

# To specific project directory
python main.py --project "E:\Edit Job\client\project" --export-pipeline-graph pipeline.png
```

**Requirements:**
- DOT format: No dependencies
- PNG/SVG format: Requires `graphviz` Python package and system binary
  - `pip install graphviz`
  - Install graphviz from https://graphviz.org/download/

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

The Config class supports runtime hot-reload with change callback notifications. This allows pipeline stages to react to config changes without restarting the pipeline.

**Registering callbacks:**
```python
from src.config import get_config

config = get_config()

# Global callback - fires on ANY config change
def on_any_change(config, changed_sections):
    print(f"Changed: {changed_sections}")

config.register_change_callback(on_any_change)

# Section-specific callback - fires only when that section changes
def on_matching_change(config, changed_sections):
    print("Matching config changed!")

config.register_change_callback(on_matching_change, section='matching')
```

**Performing reload:**
```python
# Reload checks file hash and only reloads if changed
reloaded = config.reload()

if reloaded:
    print("Config was reloaded")
```

**Behavior:**
- Callbacks are invoked AFTER the config is reloaded with the list of changed sections
- Callbacks are preserved across reloads (callbacks persist in the Config object)
- If config is frozen (after pipeline starts), reload raises `FrozenConfigError`
- Use `config.unfreeze()` in tests to allow reload
- Callback errors are caught and logged, not propagated

**API Reference:**
| Method | Description |
|--------|-------------|
| `register_change_callback(callback, section=None)` | Register callback. `section` can be `'*'` for global or specific section name |
| `unregister_change_callback(callback, section=None)` | Unregister callback. Returns True if found |
| `reload()` | Reload from file if changed. Returns True if reloaded |

**Callback signature:** `callback(config: Config, changed_sections: List[str])`

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

The config system follows a clear precedence order for resolving config values:

| Precedence | Source | Description |
|------------|--------|-------------|
| 1 (highest) | Environment variables | `MATCHER_*` env vars override all other sources |
| 2 | Profile | Profile files in `config/profiles/` override base config |
| 3 | config.yaml | Base configuration file |
| 4 (lowest) | Default values | Dataclass field defaults |

**Querying Value Sources:**

```python
# Get source for a specific field
source = config.get_value_source('matching.min_confidence')
# Returns: {'source': 'env', 'value': 0.85, 'section': 'matching'}

# Get all tracked sources
all_sources = config.get_all_sources()
# Returns dict of {field_path: {source, value, section}}
```

**CLI Usage:**

```bash
# Show all config values with their sources
python main.py --show-precedence

# Show specific field
python main.py --show-precedence matching.min_confidence
```

**Output Example:**
```
============================================================
  CONFIG VALUE PRECEDENCE
  Precedence order: env > profile > yaml > default
============================================================

  [ENV] (2 fields)
  ----------------------------------------
    matching.min_confidence: 0.85
    download.mullvad.enabled: true

  [YAML] (45 fields)
  ----------------------------------------
    project.version: 4.0.0
    video_search.max_results: 50
    ...
```

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

### Browser-Based Cookie Extraction (US-143-005)

Automatically extract cookies from browser profiles instead of manually exporting cookie files. Supports Chrome, Firefox, Edge, Brave, Opera, and Safari.

**Configuration in config.yaml:**
```yaml
download:
  cookie_rotation:
    enabled: true
    # Enable automatic cookie extraction from browsers
    auto_extract_cookies: true
    # Browser priority order (higher = tried first)
    browser_priority_order:
      - "chrome"
      - "firefox"
      - "edge"
      - "brave"
      - "opera"
      - "safari"
    # Browser profiles to extract from
    browser_profiles:
      - browser: "chrome"
        profile: "Default"
        domain: ".youtube.com"
      - browser: "firefox"
        profile: "default-release"
        domain: ".youtube.com"
    # Cookie freshness validation (auto-rotate expired cookies)
    cookie_freshness_validation: true
    # Auto-refresh expired cookies from browser
    auto_refresh_cookies: true
    # Extraction timeout (seconds)
    browser_extraction_timeout: 30
```

**Supported Browsers:**
| Browser | Linux Path | macOS Path | Windows Path |
|---------|------------|------------|--------------|
| Chrome | `~/.config/google-chrome/Default/Cookies` | `~/Library/Application Support/Google/Chrome/Default/Cookies` | `%LOCALAPPDATA%/Google/Chrome/User Data/Default/Cookies` |
| Firefox | `~/.mozilla/firefox/` | `~/Library/Application Support/Firefox/Profiles` | `%APPDATA%/Mozilla/Firefox/Profiles` |
| Edge | `~/.config/microsoft-edge/Default/Cookies` | `~/Library/Application Support/Microsoft Edge/Default/Cookies` | `%LOCALAPPDATA%/Microsoft/Edge/User Data/Default/Cookies` |
| Brave | `~/.config/BraveSoftware/Brave-Browser/Default/Cookies` | `~/Library/Application Support/BraveSoftware/Brave-Browser/Default/Cookies` | `%LOCALAPPDATA%/BraveSoftware/Brave-Browser/User Data/Default/Cookies` |

**Manual Cookie Export (alternative to auto-extraction):**
If you prefer manual export, use the "Get cookies.txt LOCALLY" browser extension:
1. Install the extension in your browser
2. Navigate to YouTube
3. Click the extension icon and export as Netscape format
4. Save to `cookies/main.txt`, `cookies/backup1.txt`, etc.

**Features:**
- Extracts cookies directly from browser SQLite databases
- Supports multiple browser profiles (Default, Profile1, etc.)
- Domain filtering (only extracts YouTube cookies by default)
- Automatic cookie freshness validation before use
- Auto-refresh when cookies expire

### Per-Keyword Circuit Breaker (US-109-004)

Per-keyword circuit breaker isolates rate-limited keywords while allowing other keywords to continue. Also includes a global fallback that trips when >50% of keywords are rate-limited.

**Key features:**
- Per-keyword failure tracking: each keyword has its own circuit breaker
- Per-keyword pause duration: based on keyword's own failure history (adaptive)
- Global fallback: trips when >50% keywords are rate-limited
- Speed-based triggering (US-113-007): trip circuit on sustained slow downloads

**Configuration in config.yaml:**
```yaml
download:
  per_keyword_circuit_breaker:
    enabled: true
    consecutive_failures_threshold: 3  # Failures per keyword before trip
    pause_seconds: 30.0               # Base pause per keyword
    max_pause_seconds: 120.0         # Maximum pause cap
    jitter_factor: 0.2                # Random jitter (0.0-1.0)
    global_fallback_threshold: 0.5    # 50% - trip global when exceeded
    global_pause_seconds: 60.0
    global_max_pause_seconds: 300.0
    # Speed-based triggering (US-113-007)
    speed_threshold_mbps: 0.5          # Minimum speed in Mbps
    sustained_degradation_threshold: 3  # Consecutive slow downloads
    enable_speed_trigger: true
```

**Behavior:**
- Each keyword tracks its own consecutive failures independently
- On trip: pause based on keyword's history (adaptive backoff)
- Global fallback prevents cascade when many keywords are rate-limited
- Speed-based triggering detects rate limits before actual failures (US-113-007)

**Related class:** `PerKeywordCircuitBreaker` in `src/downloader/per_keyword_circuit_breaker.py`

### Predictive Rate Limit Detection (US-109-008)

Predictive rate limit detection using historical patterns from time-of-day and day-of-week.

**Features:**
- Track historical rate limit events by time window and day-of-week
- Time windows: MORNING (6-12), AFTERNOON (12-18), EVENING (18-22), OVERNIGHT (22-6)
- Predict rate limit likelihood (0.0-1.0) based on historical patterns
- Auto-increase budget allocation when likelihood > 0.6

**Usage:**
```python
predictor = RateLimitPredictor()

# Record attempts and events
predictor.record_attempt()
predictor.record_rate_limit_event(trigger_category='429', keyword='python tutorial')

# Get prediction
likelihood = predictor.predict_rate_limit_likelihood()

# Auto-increase budgets if high likelihood
if likelihood > 0.6:
    predictor.increase_budget_allocation(budget)
```

**Prediction details:**
- Evening hours have highest rate limits (40% base)
- Weekend days have higher usage (1.3x multiplier)
- Morning hours are moderate (15% base)
- Overnight is lowest (10% base)

**Related class:** `RateLimitPredictor` in `src/downloader/rate_limit_predictor.py`

### Region-Specific Backoff (US-109-011)

When using VPN rotation, different geographic regions have different rate limit tolerance from YouTube. This config allows applying region-specific backoff multipliers based on the current VPN exit region.

**Configuration in config.yaml:**
```yaml
download:
  region_backoff:
    enabled: false                    # Enable region-specific backoff multipliers
    us_multiplier: 1.0                # US region (baseline = 1.0)
    eu_multiplier: 1.2                # European regions (20% longer backoff)
    asia_multiplier: 1.5              # Asian regions (50% longer backoff)
    other_multiplier: 2.0             # Other regions (100% longer backoff)
    track_per_region: false           # Maintain separate rate limit counters per region
```

**Requirements:**
- `region_backoff.enabled=true` requires `mullvad.enabled=true` (VPN must be configured)
- All multipliers must be >= 1.0

**Region mapping:**
- US/CA → US region (multiplier 1.0)
- GB, DE, NL, SE, CH, FR, IT, ES, etc. → EU region (multiplier 1.2)
- JP, SG, KR, IN, ID, MY, TH, VN, PH, TW, HK → ASIA region (multiplier 1.5)
- All others → OTHER region (multiplier 2.0)

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

### Data Drift Detection (US-88-006, US-125-005)

Configurable cross-stage data drift detection that warns when output counts drop unexpectedly between pipeline stages.

**Configuration in config.yaml:**
```yaml
pipeline:
  drift_rules:
    enabled: true              # Globally enable/disable drift detection
    track_history: true       # Track drift events for trend analysis
    max_history: 100          # Maximum drift events to keep in history
    rules:
      # After CAPTION: caption_results should have >= 80% of video_ids
      - trigger_stage: "CAPTION"
        source_field: "video_ids"
        target_field: "caption_results"
        threshold_type: "ratio"  # ratio, absolute_count, percentage
        threshold: 0.8
        severity: "warning"       # warning (log) or error (stop pipeline)
        enabled: true
        remediation: "Check caption fetch failures, rate limiting, or video availability."
      # After MATCH: text_metadata should have >= 80% of video_ids
      - trigger_stage: "MATCH"
        source_field: "video_ids"
        target_field: "text_metadata"
        threshold_type: "ratio"
        threshold: 0.8
        severity: "warning"
        enabled: true
      # After OUTPUT: matches should have >= 50% of voiceover_segments
      - trigger_stage: "OUTPUT"
        source_field: "voiceover_segments"
        target_field: "matches"
        threshold_type: "ratio"
        threshold: 0.5
        severity: "warning"
        enabled: true
```

**Threshold types:**
| Type | Description | Example |
|------|-------------|---------|
| `ratio` | Minimum ratio of target/source (0.0-1.0) | 0.8 = 80% |
| `absolute_count` | Minimum absolute count for target | 50 = at least 50 items |
| `percentage` | Minimum percentage of source (0-100) | 80 = 80% |

**Custom thresholds per stage pair:**
Override default thresholds by specifying custom `threshold` values in the rule. The pipeline uses the configured threshold rather than hardcoded defaults.

**Severity levels:**
- `warning`: Non-blocking - logs warning but pipeline continues
- `error`: Blocks pipeline progression

**Related classes:** `DriftRuleConfig`, `DriftRulesConfig` in `src/config/sections/infrastructure.py`

### Chapter and Listicle Configuration

This section documents all configuration options for chapter detection, chapter-aware matching, and listicle topic extraction.

#### Video Search - Chapter/Topic Queries

| Config | Location | Description |
|--------|----------|-------------|
| `use_chapter_queries` | `video_search.use_chapter_queries` | Use chapter topics for targeted search queries (US-98-005) |
| `listicle_topic_as_search_terms` | `video_search.listicle_topic_as_search_terms` | Use listicle topics for targeted search queries (US-98-008) |

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
| `no_chapter_fallback_strategy` | `matching.no_chapter_fallback_strategy` | Fallback when no chapters detected: 'global' or 'segment' (US-105-011) |
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

#### Temporal Coherence Scoring

| Config | Location | Description |
|--------|----------|-------------|
| `temporal_coherence_enabled` | `matching.temporal_coherence_enabled` | Enable temporal coherence scoring (US-77-003) |
| `temporal_coherence_same_source_boost` | `matching.temporal_coherence_same_source_boost` | Boost for clips from same source as adjacent (+5% default) |
| `temporal_coherence_context_switch_penalty` | `matching.temporal_coherence_context_switch_penalty` | Penalty for jarring context switches (-5% default) |

**How it works:**
- Same-source boost: When adjacent segments (previous/next) are from the same video source, apply a small confidence boost (+5% by default)
- Jarring context switch: When moving from one topic to a completely unrelated topic between adjacent segments, apply a small penalty (-5% by default)
- Topic overlap detection: Uses keywords and topics from video segments to determine if the transition is "jarring" (no overlap = jarring)

**Related functions:** `compute_temporal_coherence()`, `_is_jarring_context_switch()` in `src/matching/scoring.py`

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

### Health Check Configuration (US-125-011)

Health checks validate external dependencies before stage execution. The pipeline supports configurable health check intervals for long-running stages.

**Configuration in config.yaml:**
```yaml
pipeline:
  # US-125-011: Health check interval configuration
  # Controls how often health checks run during long-running pipeline stages
  # Set to 0 to disable interval-based health checks (runs before each stage only)
  health_check_interval:
    default: 300                  # Default interval in seconds (5 minutes)
    VIDEO_SEARCH: 120             # More frequent checks for short stages
    CAPTION: 300                  # Medium frequency
    MATCH: 300                    # LLM stages need regular checks
    DOWNLOAD_SEGMENTS: 600        # Less frequent for long downloads
```

**How it works:**
- Health checks run before each stage by default (when interval = 0)
- With interval > 0: health checks run only after the specified time has elapsed since last check
- Stage-specific intervals override the default for that stage type
- Health check timing metrics are included in stage output (`duration_ms` per check)

**Health checks validated:**
- Network connectivity
- Disk space
- Memory usage
- Embedding provider availability
- FFmpeg availability
- yt-dlp availability
- LLM provider connectivity

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

See [SESSION_HISTORY.md](SESSION_HISTORY.md) for dated changelog. Recent: PowerShell switch parameter null crash fix (periodic exploration failed with "parameter 'FullExplore' not found" when null was passed to switch; fixed with explicit conditional in Invoke-FocusAreaExploration), PowerShell return value unboxing fix (hashtable return from Invoke-ClaudeSubprocess converted to Object[]; fixed with explicit conversion in Invoke-ClaudeWithInfiniteRetry and ralph.ps1), Grace period exit code fix (default exitCode=1 ensures non-null, override sets to 0 if story still passes), Quick Edit Mode periodic re-check safeguard (console output freeze prevention), Pre-flight fast-path for pre-implemented stories (infinite loop fix), evidence gate tolerant parsing + all-false safety net, Ollama embedding provider (nomic-embed-text) with asymmetric search and provider-specific cache keys, Carlini-inspired Ralph improvements, Terminal OUTPUT stage checkpoint false WARNING fix, FAISS index space mismatch fix.
