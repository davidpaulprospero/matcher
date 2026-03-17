---
name: fix-ytdlp
description: Use when yt-dlp downloads fail with "No video formats found", "Requested format is not available", 403 errors, rate limiting, or any video download failure. Also use when segment downloads succeed for cached files but fail for new ones.
allowed-tools:
  - Read
  - Edit
  - Write
  - Grep
  - Glob
  - Bash(yt-dlp:*)
  - Bash(pip install:*)
  - Bash(python:*)
  - Bash(python -m pytest:*)
  - Bash(python -m py_compile:*)
  - Bash(gh:*)
  - Task
  - WebSearch
---

# yt-dlp Fix

Diagnose and fix yt-dlp download failures. **CLI-first, code-second.**

## Invocation

```
/ytdlp-fix "No video formats found!" in Stage 6
/ytdlp-fix 403 Forbidden during segment downloads
/ytdlp-fix segments failing but cached files work
```

## The #1 Rule

**NEVER touch Python code before reproducing at the CLI.**

The Python API runs with `quiet: True` and `no_warnings: True`, hiding critical diagnostic output. The CLI shows everything. A 30-second CLI test beats 30 minutes of wrong hypotheses.

## Workflow

### Phase 1: CLI Reproduction (MANDATORY, do this FIRST)

Run the failing video directly with yt-dlp CLI — full verbosity, no quiet flags:

```bash
# List formats to see what YouTube actually returns
yt-dlp --list-formats "https://www.youtube.com/watch?v=VIDEO_ID" 2>&1

# If that works, test actual download with time range
yt-dlp -f "best[height<=1080]/best" --download-sections "*0-15" -o test.mp4 "URL" 2>&1
```

**Read EVERY line of output.** The root cause is almost always visible in the warnings:

| CLI Output | Root Cause | Fix |
|------------|-----------|-----|
| `n challenge solving failed` | EJS solver outdated | Update yt-dlp + `--remote-components ejs:github` |
| `Challenge solver lib script version X is not supported` | EJS version mismatch | `--remote-components ejs:github` |
| `Only images are available` | n-sig broken, all format URLs invalid | Update yt-dlp |
| `SABR` in format URLs | YouTube SABR streaming (not downloadable) | Use `tv` player client |
| `HTTP Error 403` | IP blocked / cookies expired | Rotate cookies, VPN, wait |
| `rate-limited by YouTube` | Too many requests | Add sleep intervals, wait 1hr |
| `Requested format is not available` | Format filter too strict OR no formats at all | Check `--list-formats` first |

### Phase 2: Search GitHub Issues (Parallel with CLI)

While CLI reproduction is running (or if it fails quickly), search the yt-dlp GitHub issues using `gh`:

```bash
# Search for specific error patterns
gh issue list --repo yt-dlp/yt-dlp --search "No video formats found" --limit 5
gh issue list --repo yt-dlp/yt-dlp --search "youtube n sig" --limit 3
gh issue list --repo yt-dlp/yt-dlp --search "403" --label youtube --limit 3

# View a specific issue for details
gh issue view ISSUE_NUMBER --repo yt-dlp/yt-dlp
```

**Key issue patterns to look for:**

| Error Message | Common Causes |
|--------------|----------------|
| "No video formats found" | n-sig extractor broken, YouTube SABR, player version outdated |
| "n challenge solving failed" | EJS solver outdated, need `--remote-components ejs:github` |
| "Requested format is not available" | Format filter too strict, YouTube API change |
| "HTTP Error 403" | IP ban, cookies expired, impersonation needed |
| "Only images are available" | n-sig completely broken |

If you find a recent issue with the same error:
1. Check the resolution (any PR merges, workarounds mentioned)
2. Look at the yt-dlp version mentioned vs. current version
3. Note any extractor_args or CLI flags that worked for others

### Phase 3: Identify the Layer

yt-dlp failures happen at distinct layers. Fix the RIGHT layer:

```
EXTRACTION (get video info) → FORMAT SELECTION (pick quality) → DOWNLOAD (fetch bytes)
```

| Error | Layer | Don't Touch |
|-------|-------|-------------|
| "No video formats found" | EXTRACTION | Format strings, fallback chains |
| "n challenge solving failed" | EXTRACTION | Player clients, extractor_args |
| "Requested format not available" | FORMAT SELECTION | Extraction config |
| "403 Forbidden" on download | DOWNLOAD | Format selection |
| "rate-limited" | EXTRACTION | Everything else until rate limit clears |

**Key insight:** "No video formats found" is NEVER a format selection problem. It means extraction returned zero formats. Don't add format fallbacks — fix extraction.

### Phase 4: Apply Fix

#### Extraction failures (most common in 2025-2026)

YouTube frequently changes their player JS, breaking yt-dlp's signature extraction. Check:

```bash
# 1. Check yt-dlp version
yt-dlp --version

# 2. Update if not latest
pip install --upgrade yt-dlp

# 3. Update EJS challenge solver
yt-dlp --remote-components ejs:github --skip-download "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

# 4. Verify fix
yt-dlp --list-formats "https://www.youtube.com/watch?v=FAILING_VIDEO_ID"
```

For the Python API, add to ydl_opts:
```python
'remote_components': {'ejs:github'},
```

#### Format selection failures

Only fix AFTER confirming `--list-formats` shows formats. Add fallback chain:
```python
'format': 'best[height<=1080]/best/bestvideo+bestaudio',
```

#### Download failures (403, rate limit)

These are network/auth issues. Check cookies, impersonation, VPN. See CLAUDE.md Bypass & Escalation section.

### Phase 5: Verify at CLI, THEN Fix Code

After identifying the fix at CLI:
1. Confirm the CLI command succeeds
2. Translate the fix to Python API ydl_opts
3. Run `python -m py_compile` on changed files
4. Run relevant tests

### Phase 6: Check All yt-dlp Call Sites

This codebase has 3 separate yt-dlp integration points:

| Location | Method | Notes |
|----------|--------|-------|
| `src/stages/download_segments.py` | Python API (`ydl_opts` dict) | Segment downloads |
| `src/downloader/core.py` | CLI subprocess | Video search & full downloads |
| `src/caption_fetcher.py` | CLI subprocess | Caption/subtitle fetching |

If the fix is version-related (update yt-dlp, EJS solver), all sites benefit automatically.
If the fix is config-related (new ydl_opts key), check each site.

## Python API vs CLI Translation

| CLI Flag | Python API ydl_opts Key |
|----------|------------------------|
| `--impersonate X` | `'impersonate': ImpersonateTarget.from_str('X')` |
| `--extractor-args youtube:player_client=tv` | `'extractor_args': {'youtube': {'player_client': ['tv']}}` |
| `--ignore-no-formats-error` | `'ignore_no_formats_error': True` |
| `--remote-components ejs:github` | `'remote_components': {'ejs:github'}` |
| `-f FORMAT` | `'format': 'FORMAT'` |
| `--cookies-from-browser firefox` | `'cookiesfrombrowser': ['firefox']` |
| `--download-sections *START-END` | `'download_ranges': lambda info, ydl: [{'start_time': S, 'end_time': E}]` |
| `--quiet` | `'quiet': True` |

## Red Flags — STOP and Rethink

- **Modifying format strings when `--list-formats` shows zero formats** — extraction is broken, not format selection
- **Adding fallback chains without CLI reproduction** — you're guessing
- **Blaming extractor_args/player_client when yt-dlp version is old** — update first
- **Iterating on code changes without CLI verification** — the Python API hides the real error
- **"Cached segments work but new ones fail"** — this is almost always extraction-level (n-sig, rate limit, SABR), not format-level

## Common Mistakes

| Mistake | Fix |
|---------|-----|
| Debugging Python API with `quiet: True` | Reproduce at CLI first — always |
| Assuming "No formats" = wrong format string | Check extraction layer first |
| Multiple code changes without CLI test | One CLI test would have found it |
| Skipping `yt-dlp --version` check | Version is cause >50% of the time |
| Not checking EJS solver version | `--remote-components ejs:github` |
| Blaming cookies when n-sig is broken | n-sig failure = no valid format URLs regardless of auth |
