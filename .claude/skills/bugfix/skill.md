---
name: bugfix
description: Fast bug fix workflow - reproduce, diagnose, fix, verify, scan. Enforces fix-first approach with time discipline.
allowed-tools:
  - Read
  - Edit
  - Write
  - Grep
  - Glob
  - Bash(python:*)
  - Bash(python -m pytest:*)
  - Bash(pytest:*)
  - Bash(python -m py_compile:*)
  - Bash(python -m mypy:*)
  - Task
---

# Bug Fix Skill

Fast, disciplined bug fix workflow. Fix first, explain second.

## Invocation

```
/bugfix <error message or description>
/bugfix AttributeError: 'dict' object has no attribute 'end_time' in src/caption/timeout.py
/bugfix tests/test_cache.py::TestCacheStats failing with KeyError
```

## Workflow

When invoked, follow these phases strictly. Do NOT spend more than 5 minutes investigating before proposing a concrete fix.

### Phase 1: Reproduce (max 2 minutes)

- If a traceback is provided, read the failing file at the error line
- If a test is failing, run it: `python -m pytest <test_path> -v -x`
- If it's a runtime error, identify the minimal reproduction path
- **Output:** Confirmed error with exact traceback

### Phase 2: Root Cause (max 3 minutes)

- Use Grep to find where the bad value originates
- Trace the data flow ONE level upstream — find the assignment, not the full call chain
- For `AttributeError` on dicts: check if JSON/cache return is dict where object was expected
- For `TypeError`: check function signature vs actual arguments
- For `KeyError`: check dict keys at construction site
- **Output:** Root cause in one sentence

### Phase 3: Fix (immediate)

- Apply the minimal code change with Edit
- Show what changed and why
- If the fix touches a dataclass or config, check Rule 1 (config sync) and Rule 2 (__post_init__)
- **Output:** Applied diff

### Phase 4: Verify

- Re-run the failing test or scenario
- Run `python -m py_compile <changed_file>` on every modified file
- Run `python -m pytest <relevant_test_file> -v -x` if a test exists
- **Output:** Passing test or confirmed fix

### Phase 5: Scan for Similar (always do this)

- Grep the entire codebase for the same anti-pattern
- Fix ALL instances found, not just the one that errored
- **Output:** Count of additional fixes or "No similar patterns found"

### Phase 6: Harden (if attribute-related)

- Add type annotations to the fixed function and its callers
- Run `python -m mypy --ignore-missing-imports <changed_files>` to verify
- **Output:** Type annotations added or "Already typed"

## Common Patterns in This Codebase

### Dict-vs-Object (Most Common)

```python
# BAD: Cache/JSON returns dict, code expects object
segment.end_time  # AttributeError: 'dict' has no attribute 'end_time'

# FIX: Check type and branch
if isinstance(segment, dict):
    end_time = segment.get('end_time', 0)
else:
    end_time = segment.end_time
```

### Missing getattr Fallback

```python
# BAD: Config field might not exist in older checkpoints
value = self.config.new_field  # AttributeError

# FIX: Use getattr with default (Rule 6)
value = getattr(self.config, 'new_field', default_value)
```

### Subprocess Encoding (Rule 27)

```python
# BAD: Missing encoding on Windows
subprocess.run(cmd, text=True)

# FIX: Always add encoding
subprocess.run(cmd, text=True, encoding='utf-8', errors='replace')
```

## Parallel Investigation (for complex bugs)

For bugs requiring multi-file investigation, spawn parallel agents:

```
Task 1: Trace the data flow — where is the variable assigned?
Task 2: Search for similar patterns — does this anti-pattern exist elsewhere?
```

This halves investigation time. Merge findings and apply a unified fix.
