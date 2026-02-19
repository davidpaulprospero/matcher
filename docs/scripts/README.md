# Scripts Documentation

This directory contains utility scripts for the Matcher project. All scripts follow standardized patterns for logging, argument parsing, and output formatting.

## Overview

The scripts directory contains 35+ utility scripts organized into categories:

| Category | Count | Description |
|----------|-------|-------------|
| **Project Management** | 4 | Setup, cleanup, health checks |
| **Configuration** | 4 | Validation, export, diff |
| **Benchmarking** | 2 | Performance testing and comparison |
| **Analysis** | 8 | Data analysis, testing, diagnostics |
| **Utilities** | 6 | Fixes, conversions, transformations |
| **Ralph** | 9 | Autonomous development loop |
| **Utils** | 2 | Shared utility modules |

## Standard Patterns

All scripts follow these standards (see [CLAUDE.md](../CLAUDE.md#script-logging-standard)):

### Import Pattern

```python
#!/usr/bin/env python3
"""
Script description.
"""
import os
import sys
from pathlib import Path

# Add project root and scripts directory to path for imports
_script_path = os.path.abspath(__file__)
project_root = Path(_script_path).parent.parent
scripts_dir = Path(_script_path).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(scripts_dir))

# Change to project root so relative paths work correctly
os.chdir(project_root)

# Import standardized output functions
from script_utils import print_ok, print_warn, print_error, print_info, print_header
from utils.cli_helpers import confirm
```

### Output Functions

| Function | Use Case |
|----------|----------|
| `print_ok(msg)` | Success messages, positive results |
| `print_warn(msg)` | Non-fatal issues, expected edge cases |
| `print_error(msg, exit_code)` | Fatal errors, failures |
| `print_info(msg)` | Verbose-only info (shown with `-v`) |
| `print_header(title)` | Section headers |

### Standard Arguments

Scripts support these standard arguments:

| Argument | Short | Description |
|----------|-------|-------------|
| `--project` | `-p` | Path to project directory |
| `--verbose` | `-v` | Enable verbose output |
| `--quiet` | `-q` | Suppress non-essential output |
| `--json` | `-j` | Output results as JSON |
| `--yes` | `-y` | Skip confirmation prompts |

### Verbosity Levels

- `0` (Quiet): Only errors shown
- `1` (Normal): Errors + warnings (default)
- `2` (Verbose): All output including info

---

## Script Reference

### Project Management

#### setup_project.py
Initializes a new Matcher project with proper directory structure.

```bash
python scripts/setup_project.py "E:/Projects/MyProject"
python scripts/setup_project.py --client "ClientName" --template documentary
```

**Features:**
- Creates voiceover/, output/, .cache/ directories
- Generates run.bat launcher script
- Copies config.yaml template

#### cleanup_project.py
Removes cache files and temporary data from a project.

```bash
python scripts/cleanup_project.py --project "E:/Projects/MyProject"
python scripts/cleanup_project.py --project "E:/Projects/MyProject" --all  # Remove all caches
python scripts/cleanup_project.py --project "E:/Projects/MyProject" --dry-run
```

**Options:**
- `--all`: Remove all cache files
- `--transcriptions`: Remove only transcription cache
- `--embeddings`: Remove only embedding cache
- `--dry-run`: Preview what would be deleted

#### health_check.py
Validates development environment dependencies.

```bash
python scripts/health_check.py
python scripts/health_check.py --verbose
python scripts/health_check.py --json
```

**Checks:**
- Python version and dependencies
- FFmpeg availability
- yt-dlp availability
- Network connectivity
- Disk space

#### project_health.py
Analyzes project health and identifies issues.

```bash
python scripts/project_health.py --project "E:/Projects/MyProject"
python scripts/project_health.py --project "E:/Projects/MyProject" --verbose
```

---

### Configuration

#### validate_config.py
Validates config.yaml syntax and schema.

```bash
python scripts/validate_config.py
python scripts/validate_config.py --config custom.yaml
python scripts/validate_config.py --verbose
```

#### export_config.py
Exports current configuration with defaults.

```bash
python scripts/export_config.py
python scripts/export_config.py --output exported.yaml
python scripts/export_config.py --json
```

#### config_diff.py
Compares two config files.

```bash
python scripts/config_diff.py config1.yaml config2.yaml
python scripts/config_diff.py config1.yaml config2.yaml --verbose
```

#### validate_otio_media.py
Validates OTIO output and checks media file references.

```bash
python scripts/validate_otio_media.py --project "E:/Projects/MyProject"
python scripts/validate_otio_media.py --project "E:/Projects/MyProject" --fix
```

---

### Benchmarking

#### benchmark.py
Runs standardized pipeline benchmarks.

```bash
# Run benchmark on a test project
python scripts/benchmark.py run --project "E:/Projects/TestProject" --output results.json

# Compare two benchmark runs
python scripts/benchmark.py compare --baseline results_old.json --current results_new.json

# Generate performance trend report
python scripts/benchmark.py trends --project "E:/Projects/TestProject"

# Run with config comparison
python scripts/benchmark.py run --project "E:/Projects/TestProject" --config-a base.yaml --config-b improved.yaml --compare-configs
```

**Features:**
- Measures timing per pipeline stage
- Stores results in SQLite database
- Compares results between config versions
- Generates performance trend reports

#### benchmark_runner.py
Low-level benchmark execution harness.

```bash
python scripts/benchmark_runner.py --project "E:/Projects/TestProject" --iterations 5
```

---

### Analysis

#### check_script_health.py
Validates scripts follow project coding standards.

```bash
python scripts/check_script_health.py
python scripts/check_script_health.py --verbose
python scripts/check_script_health.py --fix
```

**Validates:**
- Shebang line (`#!/usr/bin/env python3`)
- Module docstring with usage instructions
- Uses script_utils for output
- Has argparse with --help support

#### analyze_trace.py
Analyzes pipeline execution traces.

```bash
python scripts/analyze_trace.py --project "E:/Projects/MyProject"
python scripts/analyze_trace.py --project "E:/Projects/MyProject" --verbose
```

#### analyze_test_parallelization.py
Analyzes test execution parallelization opportunities.

```bash
python scripts/analyze_test_parallelization.py
python scripts/analyze_test_parallelization.py --verbose
```

#### categorize_tests.py
Categorizes tests by type (unit, integration, etc.).

```bash
python scripts/categorize_tests.py
python scripts/categorize_tests.py --verbose
python scripts/categorize_tests.py --output categories.json
```

#### assign_test_markers.py
Assigns pytest markers to tests based on categorization.

```bash
python scripts/assign_test_markers.py
python scripts/assign_test_markers.py --dry-run
```

#### analyze_final_edit.py
Analyzes final EDL/OTIO output for quality metrics.

```bash
python scripts/analyze_final_edit.py --project "E:/Projects/MyProject"
python scripts/analyze_final_edit.py --project "E:/Projects/MyProject" --verbose
```

#### diagnostic_viewer.py
View and analyze pipeline diagnostics.

```bash
python scripts/diagnostic_viewer.py --project "E:/Projects/MyProject"
python scripts/diagnostic_viewer.py --project "E:/Projects/MyProject" --json
```

#### coverage_by_module.py
Reports coverage metrics broken down by module.

```bash
python scripts/coverage_by_module.py
python scripts/coverage_by_module.py --output coverage.json
```

---

### Utilities

#### batch_operations.py
Perform batch operations on multiple projects.

```bash
python scripts/batch_operations.py --projects "E:/Projects/P1,E:/Projects/P2" --operation cleanup
python scripts/batch_operations.py --projects "E:/Projects/P1,E:/Projects/P2" --operation validate
```

#### checkpoint_diff.py
Compare checkpoint files between runs.

```bash
python scripts/checkpoint_diff.py --old checkpoint_old.json --new checkpoint_new.json
python scripts/checkpoint_diff.py --old checkpoint_old.json --new checkpoint_new.json --verbose
```

#### fix_special_chars.py
Fix special characters in video metadata.

```bash
python scripts/fix_special_chars.py --project "E:/Projects/MyProject"
```

#### convert_vp9_to_h264.py
Convert VP9 video files to H264.

```bash
python scripts/convert_vp9_to_h264.py --input video.mp4 --output video_h264.mp4
python scripts/convert_vp9_to_h264.py --input-dir videos/ --output-dir converted/
```

#### download_list.py
Download videos from a list of URLs/IDs.

```bash
python scripts/download_list.py --list videos.txt --output-dir downloads/
python scripts/download_list.py --list videos.txt --parallel 4
```

#### generate_coverage_badge.py
Generate coverage badge for CI/CD.

```bash
python scripts/generate_coverage_badge.py --coverage 85 --output badge.svg
```

#### generate_entity_json.py
Generate entity JSON for media sources.

```bash
python scripts/generate_entity_json.py --channel "ChannelName" --output entities.json
```

#### regenerate_otio.py
Regenerate OTIO output from checkpoint.

```bash
python scripts/regenerate_otio.py "E:/Projects/MyProject"
python scripts/regenerate_otio.py "E:/Projects/MyProject" --format edl
```

#### edl_to_scene_video.py
Convert EDL to scene video.

```bash
python scripts/edl_to_scene_video.py --input input.edl --output output.mp4
```

#### tools.py
Unified CLI entry point for all scripts.

```bash
python scripts/tools.py --help
python scripts/tools.py benchmark run --project "E:/Projects/TestProject"
python scripts/tools.py cleanup --project "E:/Projects/MyProject"
```

---

### Ralph Scripts

The Ralph autonomous development loop scripts are in `scripts/ralph/`. See [scripts/ralph/README.md](ralph/README.md) for full documentation.

**Key Ralph scripts:**
- `ralph.ps1` - Main Ralph loop execution
- `watch.ps1` - Ralph progress monitoring
- `config/ralph-config.json` - Ralph configuration

---

### Utils

#### utils/cli_helpers.py
Shared CLI helper functions.

```python
from utils.cli_helpers import confirm, parse_args, json_output

# Confirmation prompt
if not confirm("Continue?"):
    return

# Standard argument parsing
args = parse_args(description="My script", add_project_arg=True)

# JSON output helper
result = {"status": "ok"}
print(json_output(result, args.json))
```

**Functions:**
- `confirm(prompt)` - Yes/no confirmation prompt
- `parse_args()` - Standard argument parsing
- `json_output()` - Format output as JSON
- `format_table()` - Format data as table

#### utils/diagnose_checkpoint.py
Diagnose checkpoint issues.

```bash
python scripts/utils/diagnose_checkpoint.py --project "E:/Projects/MyProject"
python scripts/utils/diagnose_checkpoint.py --project "E:/Projects/MyProject" --fix
```

#### utils/fix_channel_ids.py
Fix channel ID references in checkpoint.

```bash
python scripts/utils/fix_channel_ids.py --project "E:/Projects/MyProject"
```

#### utils/fix_channel_ids_fast.py
Fast channel ID fixing (batch mode).

```bash
python scripts/utils/fix_channel_ids_fast.py --project "E:/Projects/MyProject"
```

---

## Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| ImportError: No module named 'script_utils' | Ensure you're running from project root and scripts directory is in path |
| Config validation fails | Run `python scripts/validate_config.py` to see detailed errors |
| Scripts hang on Windows | Check Quick Edit mode is disabled in PowerShell |
| Path issues with --project | Use absolute paths; relative paths resolved from project root |

### Debug Mode

Most scripts support `--verbose` or `-v` for detailed output:

```bash
python scripts/cleanup_project.py --project "E:/Projects/MyProject" --verbose
```

### Getting Help

```bash
# Show script help
python scripts/script_name.py --help

# Validate config first
python scripts/validate_config.py

# Check script health
python scripts/check_script_health.py --verbose
```

---

## Contribution Guidelines

### Adding New Scripts

1. **Follow the import pattern** - Use the standard sys.path setup
2. **Add shebang** - `#!/usr/bin/env python3` on first line
3. **Add docstring** - Include "Usage:" section with examples
4. **Use script_utils** - Import and use standardized output functions
5. **Add argparse** - Support `--help` and standard arguments

### Required Template

```python
#!/usr/bin/env python3
"""
Script Description

Usage:
    python scripts/new_script.py --arg value
    python scripts/new_script.py --verbose
"""
import argparse
import os
import sys
from pathlib import Path

# Standard path setup
_script_path = os.path.abspath(__file__)
project_root = Path(_script_path).parent.parent
scripts_dir = Path(_script_path).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(scripts_dir))
os.chdir(project_root)

from script_utils import print_ok, print_warn, print_error, print_info, print_header


def main():
    parser = argparse.ArgumentParser(description="Script description")
    parser.add_argument("--project", "-p", help="Project path")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    # Your code here
    print_ok("Script completed")


if __name__ == "__main__":
    main()
```

### Running Tests

```bash
# Test all scripts
python scripts/check_script_health.py

# Run specific test
pytest tests/test_script_utils.py -v

# Validate config
python scripts/validate_config.py
```

### Code Quality

- Use type hints where appropriate
- Add docstrings to functions
- Handle errors gracefully with proper exit codes
- Support `--json` output for machine consumption
