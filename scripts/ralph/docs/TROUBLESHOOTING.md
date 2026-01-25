# Ralph Loop Troubleshooting Guide

Quick reference for diagnosing and fixing common Ralph Loop issues.

## Decision Trees

### "Why Did the Iteration Time Out?"

```
Iteration timed out
    |
    +-- Is iterationTimeout too low?
    |       |
    |       +-- Yes -> Increase in config: autonomy.iterationTimeout: 900
    |       +-- No -> Continue
    |
    +-- Check claude_out_N.log - did Claude get stuck?
    |       |
    |       +-- Lots of Read/Grep calls -> Complex exploration, increase timeout
    |       +-- Waiting for tool result -> Tool hung, check system resources
    |       +-- Empty output -> Process crashed, check claude_err_N.log
    |       +-- Loop pattern in output -> Claude stuck in loop, check prompt
    |
    +-- Check resource_usage_N.json - memory issues?
    |       |
    |       +-- Memory > 2GB -> System resources exhausted
    |       +-- Normal memory -> Timeout is execution time issue
    |
    +-- Is this a specific story pattern?
            |
            +-- Same story times out repeatedly -> Story is too complex, split it
            +-- Random timeouts -> System/network instability
```

### "Why Did the Story Fail?"

```
Story failed (success=false)
    |
    +-- Check error_category in metrics.csv
    |       |
    |       +-- TestFailure -> Tests failed, check test_details_N.json
    |       +-- SyntaxError -> Code has syntax errors, check claude_out_N.log
    |       +-- ToolError -> Tool execution failed, check permissions
    |       +-- APIError -> Rate limiting or network, wait and retry
    |       +-- Unknown -> Check claude_err_N.log for details
    |
    +-- If TestFailure:
    |       |
    |       +-- Check test_details_N.json for specific failing tests
    |       +-- Is it a flaky test? Run manually: pytest tests/specific_test.py -v
    |       +-- Is it a real regression? Review claude_out_N.log for changes
    |
    +-- If SyntaxError:
    |       |
    |       +-- Check which file has the error in claude_out_N.log
    |       +-- Validate: python -m py_compile <file>
    |       +-- Common: Missing comma, unclosed bracket, indentation
    |
    +-- Check story_*_verification.json for acceptance criteria
            |
            +-- Which criteria failed?
            +-- Was the implementation partial?
```

### "Why Are Metrics Showing Different Numbers?"

```
Metrics mismatch (e.g., "10 errors but I see 5")
    |
    +-- Are you looking at the right session?
    |       |
    |       +-- Check session ID in watch dashboard
    |       +-- Compare with metrics.csv session column
    |
    +-- Time range mismatch?
    |       |
    |       +-- Watch dashboard shows current session only
    |       +-- metrics.csv contains all historical data
    |
    +-- Multiple error types being combined?
    |       |
    |       +-- Check error_category breakdown
    |       +-- Some "failures" are retries that eventually succeeded
    |
    +-- Metric calculation difference?
            |
            +-- tokens_used is estimated, not exact
            +-- duration_min is rounded
            +-- test_results is a summary string
```

---

## Common Error Patterns

### Pattern: "Consecutive Failures > 5"

**Symptoms:** Ralph stops with "Too many consecutive failures"

**Causes:**
1. Claude API rate limiting
2. Malformed PRD or prompt
3. Test environment broken

**Solutions:**
```powershell
# Check for API issues
Get-Content scripts/ralph/logs/latest_session/claude_err*.log | Select-String "rate|limit|429"

# Check PRD validity
Get-Content scripts/ralph/prd.json | ConvertFrom-Json

# Reset and retry
Remove-Item scripts/ralph/prd.json
.\ralph.ps1 -FocusArea "testing"
```

### Pattern: "PRD Generation Failed"

**Symptoms:** No prd.json created, exits early

**Causes:**
1. Invalid focus area
2. Claude permissions issue
3. Prompt file missing or corrupt

**Solutions:**
```powershell
# Check prompt file exists
Test-Path scripts/ralph/prompt.md

# Try with explicit permissions
.\ralph.ps1 -FocusArea "testing" -SkipPlanApproval

# Check claude_err for specific error
Get-Content scripts/ralph/logs/latest_session/claude_err_1.log
```

### Pattern: "Tests Pass Locally but Fail in Ralph"

**Symptoms:** Story marked failed for tests, but `pytest` works manually

**Causes:**
1. Different working directory
2. Missing environment variables
3. Concurrent test interference

**Solutions:**
```powershell
# Run tests from project root (same as Ralph)
cd D:\_Projects\voiceover-matcher-subtitle
pytest tests/ -v

# Check for env vars
$env:PYTHONPATH

# Run specific failing test with verbose
pytest tests/test_specific.py::test_name -v --tb=long
```

### Pattern: "Queue Stuck on One Area"

**Symptoms:** Same focus area runs repeatedly, never advances

**Causes:**
1. PRD always fails validation
2. All stories failing
3. queue.json not being updated

**Solutions:**
```powershell
# Check queue state
Get-Content scripts/ralph/queue.json | ConvertFrom-Json | Format-List

# Force advance to next area
$q = Get-Content scripts/ralph/queue.json | ConvertFrom-Json
$current = $q.focusAreas | Where-Object { -not $_.completed } | Select-Object -First 1
$current.completed = $true
$q | ConvertTo-Json -Depth 10 | Set-Content scripts/ralph/queue.json
```

---

## Log File Quick Reference

| File | Purpose | Key Things to Look For |
|------|---------|------------------------|
| `claude_out_N.log` | Claude's output | Tool calls, code changes, errors |
| `claude_err_N.log` | Claude's stderr | API errors, crashes |
| `prompt_N.txt` | Input prompt | What was asked |
| `iteration_N_manifest.json` | Structured summary | Git changes, status, timing |
| `test_details_N.json` | Test results | Individual test pass/fail |
| `resource_usage_N.json` | Memory/CPU | Resource exhaustion |
| `session_timeline.jsonl` | Event log | What happened when |
| `state_transitions.jsonl` | State changes | Debugging flow issues |
| `error_evolution.jsonl` | Error patterns | Recurring error types |

---

## Health Check Commands

```powershell
# Overall system health
Get-Process | Where-Object { $_.WorkingSet64 -gt 1GB } | Select-Object Name, @{N='MB';E={[int]($_.WorkingSet64/1MB)}}

# Check git state
git status
git log --oneline -3

# Validate Python environment
python -c "import src; print('OK')"

# Check Claude CLI
claude --version

# Verify config files
python -c "import yaml; yaml.safe_load(open('config.yaml'))"
```

---

## Quick Fixes

| Issue | Quick Fix |
|-------|-----------|
| Stuck process | `Get-Process claude* \| Stop-Process -Force` |
| Corrupt PRD | `Remove-Item scripts/ralph/prd.json` |
| Stale queue | Reset `completed` flags in queue.json |
| Test pollution | `rm -rf .pytest_cache __pycache__` |
| Git conflicts | `git checkout -- .` (careful!) |
| Memory issues | Close other apps, reduce iteration count |
