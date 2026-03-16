# Script Migration Guide

This guide explains how to migrate scripts in the `scripts/` directory to use the standardized `script_utils` module for consistent output and argument handling.

## Why Use script_utils?

- **Consistency**: All scripts use the same output format (`[OK]`, `[WARN]`, `[ERROR]`)
- **Verbosity Control**: Scripts automatically support `--verbose` and `--quiet` flags
- **Argument Parsing**: Standard arguments like `--project`, `--config` are built-in
- **Progress Bars**: Built-in support for tqdm progress bars
- **Config Loading**: Unified config loading with error handling

## Quick Start

### Step 1: Add the Standard Import Block

Add this at the top of your script:

```python
#!/usr/bin/env python3
"""
Script description here.
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

# Change to project root so relative paths (config.yaml) work correctly
os.chdir(project_root)

# Import standardized output functions
from script_utils import print_ok, print_warn, print_error, print_info, print_header

# Import CLI helpers (if needed)
from utils.cli_helpers import confirm
```

### Step 2: Replace Logging Calls

Replace custom logging with script_utils functions:

| Old Pattern | New Pattern |
|-------------|-------------|
| `logger.info(msg)` | `print_ok(msg)` or `print_info(msg)` |
| `logger.warning(msg)` | `print_warn(msg)` |
| `logger.error(msg)` | `print_error(msg)` |
| `logger.error(msg, 1)` | `print_error(msg, exit_code=1)` |
| `logger.debug(msg)` | `print_info(msg)` |
| `print("Section")` | `print_header("Section")` |
| `print(f"Result: {x}")` | `print_ok(f"Result: {x}")` |

### Step 3: Add Standard Arguments

```python
import argparse
from script_utils import add_standard_arguments, set_verbosity

parser = argparse.ArgumentParser(description="My script")
add_standard_arguments(parser, add_project=True, add_verbose=True, add_quiet=True, project_required=False)

args = parser.parse_args()

# Set verbosity based on flags
if args.quiet:
    set_verbosity(0)
elif args.verbose:
    set_verbosity(2)
else:
    set_verbosity(1)
```

## Before/After Examples

### Example 1: Simple Script Migration

**Before:**
```python
#!/usr/bin/env python3
"""My script that does something."""
import logging
import sys

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def main():
    logger.info("Starting process")
    logger.warning("No input provided, using defaults")
    result = process_data()
    logger.info(f"Processed {result} items")
    logger.error("Failed to write output")
    sys.exit(1)

if __name__ == "__main__":
    main()
```

**After:**
```python
#!/usr/bin/env python3
"""My script that does something."""
import os
import sys
from pathlib import Path

_script_path = os.path.abspath(__file__)
project_root = Path(_script_path).parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

from script_utils import print_ok, print_warn, print_error

def main():
    print_ok("Starting process")
    print_warn("No input provided, using defaults")
    result = process_data()
    print_ok(f"Processed {result} items")
    print_error("Failed to write output", exit_code=1)

if __name__ == "__main__":
    main()
```

### Example 2: With argparse and project argument

**Before:**
```python
#!/usr/bin/env python3
"""Script with project argument."""
import argparse
import logging
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description="Process a project")
parser.add_argument("-p", "--project", required=True, help="Project directory")
parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

args = parser.parse_args()

if args.verbose:
    logging.getLogger().setLevel(logging.DEBUG)

logger.info(f"Processing project: {args.project}")
```

**After:**
```python
#!/usr/bin/env python3
"""Script with project argument."""
import os
import sys
import argparse
from pathlib import Path

_script_path = os.path.abspath(__file__)
project_root = Path(_script_path).parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

from script_utils import print_ok, add_project_argument, add_verbose_argument, set_verbosity

parser = argparse.ArgumentParser(description="Process a project")
add_project_argument(parser, required=True)
add_verbose_argument(parser)

args = parser.parse_args()

if args.verbose:
    set_verbosity(2)
else:
    set_verbosity(1)

print_ok(f"Processing project: {args.project}")
```

### Example 3: With config loading

**Before:**
```python
#!/usr/bin/env python3
"""Script that loads config."""
import os
import sys
import yaml
from pathlib import Path

_script_path = os.path.abspath(__file__)
project_root = Path(_script_path).parent.parent

config_path = project_root / "config.yaml"
if not config_path.exists():
    print(f"Error: Config file not found: {config_path}")
    sys.exit(1)

with open(config_path) as f:
    config = yaml.safe_load(f)

print(f"Loaded config with {len(config)} sections")
```

**After:**
```python
#!/usr/bin/env python3
"""Script that loads config."""
import os
import sys
from pathlib import Path

_script_path = os.path.abspath(__file__)
project_root = Path(_script_path).parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

from script_utils import print_ok, print_error, load_config_for_script

config = load_config_for_script()
print_ok(f"Loaded config")
```

### Example 4: With progress bar

**Before:**
```python
#!/usr/bin/env python3
"""Script with progress."""
import time

items = range(10)
print(f"Processing {len(items)} items...")

for i, item in enumerate(items):
    time.sleep(0.1)  # Simulate work
    if i % 3 == 0:
        print(f"  Progress: {i+1}/{len(items)}")

print("Done!")
```

**After:**
```python
#!/usr/bin/env python3
"""Script with progress."""
import time
import os
import sys
from pathlib import Path

_script_path = os.path.abspath(__file__)
project_root = Path(_script_path).parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

from script_utils import print_ok, progress_bar

items = range(10)

for item in progress_bar(items, desc="Processing"):
    time.sleep(0.1)  # Simulate work

print_ok("Done!")
```

### Example 5: Verbose-only output

**Before:**
```python
#!/usr/bin/env python3
"""Script with verbose output."""
import logging
import sys

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("-v", "--verbose", action="store_true")
args = parser.parse_args()

if args.verbose:
    logger.setLevel(logging.DEBUG)

logger.debug("This is debug info")
logger.info("This is important")
```

**After:**
```python
#!/usr/bin/env python3
"""Script with verbose output."""
import os
import sys
import argparse
from pathlib import Path

_script_path = os.path.abspath(__file__)
project_root = Path(_script_path).parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

from script_utils import print_info, add_verbose_argument, set_verbosity

parser = argparse.ArgumentParser()
add_verbose_argument(parser)
args = parser.parse_args()

set_verbosity(2 if args.verbose else 1)

print_info("This is debug info")  # Only shown with -v
print_ok("This is important")     # Always shown
```

## Function Reference

### Output Functions

| Function | Description | Shown in Quiet Mode |
|----------|-------------|---------------------|
| `print_ok(msg)` | Success messages | No |
| `print_warn(msg)` | Warning messages | Yes |
| `print_error(msg, exit_code)` | Error messages (optional exit) | Yes |
| `print_info(msg)` | Verbose info (only with `-v`) | No |
| `print_header(title)` | Section headers | No |

### Argument Helpers

| Function | Description |
|----------|-------------|
| `add_project_argument(parser, required=False)` | Add `--project/-p` |
| `add_verbose_argument(parser)` | Add `--verbose/-v` |
| `add_quiet_argument(parser)` | Add `--quiet/-q` |
| `add_config_argument(parser)` | Add `--config/-c` |
| `add_standard_arguments(parser)` | Add all standard args |

### Progress Bar Functions

| Function | Description |
|----------|-------------|
| `progress_bar(iterable, desc)` | Wrap iterable with progress bar |
| `track_progress(total, desc)` | Context manager for manual tracking |
| `spinner(message)` | Context manager for indeterminate ops |

### Config Functions

| Function | Description |
|----------|-------------|
| `load_config_for_script(path, required, skip_validation)` | Load config with error handling |
| `add_config_argument(parser)` | Add `--config/-c` argument |

## Common Patterns

### Full-featured script template

```python
#!/usr/bin/env python3
"""Script description."""
import os
import sys
import argparse
from pathlib import Path

# Path setup
_script_path = os.path.abspath(__file__)
project_root = Path(_script_path).parent.parent
scripts_dir = Path(_script_path).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(scripts_dir))
os.chdir(project_root)

# Imports
from script_utils import (
    print_ok, print_warn, print_error, print_info, print_header,
    set_verbosity, add_standard_arguments
)
from utils.cli_helpers import confirm


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="My script")
    add_standard_arguments(parser, add_project=True, project_required=True)
    args = parser.parse_args()

    # Set verbosity
    if args.quiet:
        set_verbosity(0)
    elif args.verbose:
        set_verbosity(2)
    else:
        set_verbosity(1)

    # Validate project directory
    if args.project and not Path(args.project).exists():
        print_error(f"Project directory not found: {args.project}", exit_code=1)

    # Your logic here
    print_header("My Script")
    print_ok("Starting...")

    print_info("This is verbose output")

    print_ok("Completed successfully")


if __name__ == "__main__":
    main()
```

## Verification

After migration, verify your script works:

```bash
# Normal mode
python scripts/your_script.py --project /path/to/project

# Verbose mode
python scripts/your_script.py --project /path/to/project --verbose

# Quiet mode
python scripts/your_script.py --project /path/to/project --quiet
```

## CI Validation

The project includes `scripts/validate_script_standards.py` which checks:

- Shebang line present (`#!/usr/bin/env python3`)
- Docstring present
- Proper import of script_utils
- Proper sys.path setup
- Proper os.chdir to project_root

Run validation locally:

```bash
python scripts/validate_script_standards.py
```

Or add to your pre-commit hook (see `.pre-commit-config.yaml`).
