---
name: watch
description: Monitor pipeline progress with periodic updates
allowed-tools:
  - Bash(python:*)
  - Bash(sleep:*)
  - Bash(tail:*)
  - Bash(ls:*)
  - Bash(ps:*)
  - Bash(find:*)
  - Bash(wc:*)
  - Bash(stat:*)
  - Grep
  - Read
  - Write
---

Monitor pipeline progress for a project.

## Step 1: Quick Check (always do this first)
```bash
python "D:/_Projects/voiceover-matcher-subtitle/scripts/watch_pipeline.py" "PROJECT_PATH" --once 2>&1
```

Report the current status to the user (stage, progress, any errors).

## Step 2: Find Latest Log and Extract Key Metrics
Find the most recent non-verbose log:
```bash
find "PROJECT_PATH/logs" -name "run_*.log" ! -name "*_verbose*" -type f -printf "%T@ %p\n" 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-
```

Then grep for key metrics (run in parallel):

**Coverage numbers:**
```bash
grep -E "Coverage ratio|High confidence|coverage.*%" "LATEST_LOG" | tail -5
```

**Fix verification patterns** (check if applied fixes are working):
```bash
grep -E "Pre-populated|preserve|BACKUP|rollback|filtered.*EMPTY" "LATEST_LOG" | head -10
```

**Errors and warnings:**
```bash
grep -E "ERROR|WARNING|❌|Failed" "LATEST_LOG" | tail -10
```

## Step 3: Update Status File
Read and update the status file at `PROJECT_PATH/PIPELINE_STATUS.md`:

```markdown
# Pipeline Status

## Current Status
- **Stage:** [current stage name]
- **Progress:** [X/Y segments, videos, etc.]
- **Running:** Yes/No
- **Last Updated:** [timestamp]
- **Log File:** [latest log filename]

## Key Metrics (from logs)
- **Coverage:** [X%] ([Y]/[Z] high confidence)
- **Matches:** [count]
- **Videos:** [count]

## Things to Watch For
Checklist to verify on this run:
- [ ] [Issue - expected log pattern]

## Fixes Applied
[What fixes, their status]

## Recent Activity
- [timestamp] [event]
```

**Template verification items for iterative matching fixes:**
```markdown
- [ ] **Pre-populate active:** `"Pre-populated X protected video IDs"`
- [ ] **Backup before iteration:** `"Backing up X matches"`
- [ ] **No coverage decrease:** Coverage should stay same or improve
- [ ] **Preserve logic:** `"Preserving X high-confidence matches"`
```

## Step 4: Set Up Background Monitor
```bash
sleep 900 && echo "=== 5-MIN CHECK: $(date) ===" && python "D:/_Projects/voiceover-matcher-subtitle/scripts/watch_pipeline.py" "PROJECT_PATH" --once 2>&1
```

Use parameters: `timeout: 360000`, `run_in_background: true`

Tell user: "Background check scheduled - update in 15 minutes."

## ITERATIVE_MATCH Stage (Special Handling)
This is the critical stage for coverage fixes. When pipeline reaches ITERATIVE_MATCH:

1. **Before iteration starts** - note initial coverage
2. **During iteration** - watch for:
   - `"Pre-populated X protected video IDs"` (confirms preserve fix)
   - `"Backing up X matches"` (confirms backup)
   - `"Weak segments: X"` (segments being re-matched)
3. **After iteration** - compare coverage:
   - Should NOT decrease (old bug was 27% → 20%)
   - Check `"COVERAGE HISTORY"` in logs
4. **If coverage decreased** - should see `"Rolling back to backup"`

## Coverage Extraction Pattern
To extract coverage from logs:
```
grep "Coverage ratio" LATEST_LOG | tail -3
```
Format: `Coverage ratio: X.X%`

## If Pipeline Finishes
1. Report final coverage and match count
2. Grep for all verification patterns, mark checkboxes
3. Compare coverage before/after if ITERATIVE_MATCH ran
4. Update status file with final results
5. Don't set up another monitor

## If Pipeline Errors
1. Run `/logcheck PROJECT_PATH`
2. Check if it's a known issue (proxy, rate limit, etc.)
3. Offer to resume with `--resume` flag

Replace PROJECT_PATH with actual path. Use forward slashes (E:/Edit Job/...).
