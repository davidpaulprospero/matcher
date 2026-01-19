---
name: logcheck
description: Check a project's log for errors/warnings and fix them immediately. Auto-detects issues and applies fixes.
allowed-tools:
  - Read
  - Glob
  - Grep
  - Bash(python:*)
  - Bash(rm:*)
  - Edit
---

# Log Check & Fix

Analyzes a project's run logs for errors/warnings and automatically fixes common issues.

## What It Detects & Fixes

| Issue | Detection | Auto-Fix |
|-------|-----------|----------|
| Download failures | `videos_failed > 0`, "429", "unavailable" | Re-run download stage |
| API rate limits | 429 errors, quota exceeded | Wait + retry, suggest provider switch |
| Transcription errors | Failed transcriptions in log | Clear cache, re-transcribe |
| OTIO errors | Timeline generation failures | Run OTIO healer |
| Missing media | "FileNotFoundError", broken refs | Search caches, re-download |
| Duplicate media paths | Same file in stock/ and broll/ | Auto-normalized (Rule 16) |
| Low match confidence | `avg_confidence < 0.5` | Suggest config tweaks |
| Disk space | "No space left", disk errors | Clean old caches |
| Path length | Windows 260 char errors | Enable short paths (E:/v) |

## Instructions

When this skill is invoked:

### 1. Parse the project path

Extract the project path from the user's request:
- Example: `/logcheck E:\Edit Job\Stu\January\6__2026-01-15`
- Path should be a project directory (contains `output/` folder)

### 2. Check if pipeline is currently running

**IMPORTANT**: Before analyzing logs, check if the pipeline is still running.

```bash
# Check for python main.py processes with this project
ps aux | grep -E "python.*main.py.*<project_name>" | grep -v grep

# Check if checkpoint.json was modified in last 60 seconds
python -c "import os, time; p='<project_path>/checkpoint.json'; print('RUNNING' if os.path.exists(p) and time.time() - os.path.getmtime(p) < 60 else 'IDLE')"

# Check if log file is being actively written (modified in last 30 seconds)
python -c "import os, time, glob; logs=glob.glob('<project_path>/logs/run_*.log'); print('RUNNING' if logs and time.time() - os.path.getmtime(max(logs)) < 30 else 'IDLE')"
```

**If pipeline is running:**
```
## Log Check: <project_name>

⚠️ **Pipeline is currently running**

The pipeline appears to be actively running (checkpoint/logs updated within last 60 seconds).

**Current stage**: [from checkpoint if available]
**Running since**: [timestamp]

Wait for the pipeline to complete, or use Ctrl+C in the terminal to stop it first.
```

**Do NOT apply any fixes if the pipeline is running** - just report the status and exit.

### 3. Find the latest logs

```bash
# Find latest output directory
ls -td "<project_path>/output"/*/ | head -1
```

Then locate log files in that output's `logs/` folder (may be in project root `logs/` or inside output dir):
- `run_*.json` - Structured log with errors/warnings arrays
- `run_*.log` - Human-readable log
- `run_*_verbose.log` - Debug details

### 4. Analyze the JSON log

Read the latest `run_*.json` and extract:

```python
# Key fields to check:
{
    "summary": {
        "videos_failed": 0,      # > 0 = download issues
        "avg_confidence": 0.75,  # < 0.5 = poor matching
    },
    "warnings": [...],          # List of warning messages
    "errors": [...],            # List of error messages
}
```

### 5. Categorize issues

Check for these patterns in errors/warnings:

**Download Issues:**
- `"429"`, `"rate limit"`, `"too many requests"`
- `"unavailable"`, `"video unavailable"`, `"private video"`
- `"videos_failed"` > 0 in summary

**API Issues:**
- `"429"`, `"quota"`, `"rate limit"`
- `"API error"`, `"authentication"`

**Transcription Issues:**
- `"transcription failed"`, `"whisper error"`
- `"CUDA"`, `"out of memory"`

**OTIO/Timeline Issues:**
- `"otio"`, `"timeline"`, `"gap"`, `"overlap"`
- `"media reference"`, `"missing file"`

**Path Issues:**
- `"path too long"`, `"260"`, `"WindowsError"`
- `"unicode"`, `"encoding"`

**Disk Issues:**
- `"no space"`, `"disk full"`, `"ENOSPC"`

### 6. Apply fixes automatically

For each issue category, apply the appropriate fix:

**Download failures:**
```bash
cd "D:/_Projects/voiceover-matcher" && python main.py --project "<project_path>" --resume --voiceover "<voiceover_file>"
```

**API rate limits (429s):**
- Report to user: "API rate limited. Wait 5-10 minutes before retrying."
- If Gemini: suggest adding `ANTHROPIC_API_KEY` as fallback

**Transcription cache issues:**
```bash
rm -rf "<project_path>/.cache/transcriptions"
cd "D:/_Projects/voiceover-matcher" && python main.py --project "<project_path>" --resume --voiceover "<voiceover_file>"
```

**OTIO/Timeline issues:**
```bash
cd "D:/_Projects/voiceover-matcher" && python main.py --project "<project_path>" --match-only --voiceover "<voiceover_file>"
```

**Low confidence (< 50%):**
- Report to user with suggestions:
  - Increase `matching.embedding_candidates`
  - Enable `matching.llm_rerank`
  - Check voiceover quality

**Path length issues:**
- Edit `project_config.yaml` to add:
```yaml
download:
  root_dir: "E:/v"
```
- Then re-run: `python main.py --project "<project_path>" --fresh --voiceover "<voiceover_file>"`

**Disk space:**
```bash
# Clean old caches
rm -rf "<project_path>/.cache/llm_responses"
rm -rf "<project_path>/.cache/vision_cache"
```

### 7. Report results

After analysis, report:

```
## Log Check: <project_name>

### Issues Found
- X errors, Y warnings

### Applied Fixes
1. [Fix description] - DONE/FAILED
2. ...

### Manual Action Required
- [Any issues that couldn't be auto-fixed]

### Recommendations
- [Suggestions for config changes or workflow improvements]
```

## Usage Examples

```
/logcheck E:\Edit Job\Stu\January\6__2026-01-15
/logcheck "E:/path/with spaces/ProjectName__2026-01-10"
```

## Finding the voiceover file

When running fixes that need `--voiceover`, find the SRT file:
```bash
ls "<project_path>/voiceover/"*.srt
```

Use the first `.srt` file found. If none, check for `.mp3` or `.wav`.

## Quick Reference: Fix Commands

| Issue | Command |
|-------|---------|
| Resume failed run | `python main.py --project "<path>" --resume --voiceover "<vo>"` |
| Re-run matching only | `python main.py --project "<path>" --match-only --voiceover "<vo>"` |
| Fresh start | `python main.py --project "<path>" --fresh --voiceover "<vo>"` |
| Clear transcriptions | `rm -rf "<path>/.cache/transcriptions"` |
| Clear all cache | `rm -rf "<path>/.cache"` |
