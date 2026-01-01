# Claude Code Project Guide

This file helps Claude Code sessions understand project conventions and avoid common mistakes.

## Critical Rules

### 1. Config Synchronization (MOST IMPORTANT!)
When adding/modifying configuration options:
- **Always update BOTH files:**
  - `src/config.py` - Python dataclass with type hints and defaults
  - `config.yaml` - User-facing YAML config with comments
- They must stay in sync or users won't see new options
- Run `python -m py_compile src/config.py` to verify syntax

### 2. Future Annotations & Type Hints
Both `main.py` and `src/config.py` use:
```python
from __future__ import annotations
```
This makes type hints strings, which can cause issues:
- **Problem:** Nested dataclass fields may load as `dict` instead of the dataclass
- **Solution:** Always add `__post_init__` to convert dicts:
```python
def __post_init__(self):
    if isinstance(self.nested_field, dict):
        self.nested_field = NestedConfig(**self.nested_field)
```

### 3. Config Access Pattern
```python
# Safe config access with fallback:
value = getattr(self.download_config, 'field_name', default_value)

# For nested configs that might be dicts:
if isinstance(config_section, dict):
    value = config_section.get('field_name', default)
else:
    value = getattr(config_section, 'field_name', default)

# For nested configs (e.g., dicts):
tier_timeouts = getattr(self.download_config, 'download_timeouts', {})
timeout = tier_timeouts.get('medium', 120)
```

### 4. Dataclass Field Names
`DownloadedVideo` in `src/downloader.py` has these EXACT field names:
```python
@dataclass
class DownloadedVideo:
    file: str           # NOT 'path'
    url: str
    title: str
    channel: str
    upload_date: str
    duration: float
    duration_tier: str  # NOT 'tier'
    keyword: str
    download_date: str
    license: str = "Unknown"
```
**Never use:** `path=`, `tier=`, `video_id=` - these will cause errors.

### 5. Regex Lookbehind Limitations
Python regex requires **fixed-width lookbehinds**. This will ERROR:
```python
# BAD - variable width lookbehind
r'(?<=[A-Z][a-z]+,\s[A-Z][a-z]+,)\s+'
```
Use capture groups instead:
```python
# GOOD - capture and replace
re.sub(r'([A-Z][a-z]+,\s+[A-Z][a-z]+,)\s+', r'\1|||SPLIT|||', text)
```

## File Structure

| File | Purpose |
|------|---------|
| `main.py` | Entry point, pipeline orchestration |
| `config.yaml` | User-editable settings (YAML format) |
| `src/config.py` | Dataclasses defining config schema with defaults |
| `src/downloader.py` | Video download logic, uses `self.download_config` |
| `src/matching.py` | Video-to-voiceover matching engine |
| `src/transcription.py` | Audio transcription (faster-whisper) |
| `src/checkpoint.py` | Resume/checkpoint management |
| `src/keyword_extractor.py` | LLM-based keyword extraction |

## Common Bugs & Fixes

### 1. `'dict' object has no attribute 'X'`
**Cause:** Nested config loaded as dict due to `from __future__ import annotations`
**Fix:** Add `__post_init__` to parent dataclass to convert dict to proper type.
Example in `OutputConfig`:
```python
def __post_init__(self):
    if isinstance(self.variety, dict):
        self.variety = VarietyConfig(**self.variety)
```

### 2. `DownloadedVideo.__init__() got unexpected keyword argument`
**Cause:** Using wrong field names when creating DownloadedVideo
**Fix:** Use exact field names: `file=`, `duration_tier=`, NOT `path=`, `tier=`

### 3. Live streams hanging forever
**Cause:** yt-dlp tries to download infinite live streams
**Fix:** Add `!is_live` to match-filter:
```python
'--match-filter', f"duration>{min_dur} & duration<{max_dur} & !is_live"
```
Also add to title_blacklist: `live stream`, `livestream`, `webcam`, `24/7`

### 4. Pylance "Module cannot be used as a type"
**Cause:** Type checker can't resolve dataclass types
**Fix:** Add `from __future__ import annotations` at top of file (after docstring)

### 5. Regex lookbehind error
**Cause:** Variable-width patterns in lookbehind
**Fix:** Use capture groups with replacement instead

## Key Features

### Pause-Split Segments
Location: `main.py` → `_pause_split_segments()`
Config:
```yaml
transcription:
  pause_split:
    enabled: true
    split_at_sentences: true      # Every sentence = own segment
    split_at_list_markers: true   # "Number 10, City" isolation
    split_at_locations: true      # "City, State," splits
    min_phrase_words: 2
    min_segment_duration: 0.5
```

### LLM List Detection
Location: `main.py` → `_detect_list_items()`
Uses Gemini to detect "Top 10" style countdowns and extract list items.
Config:
```yaml
keyword:
  list_detection:
    enabled: true
    download_first: true
    skip_if_entity_covered: true
    keyword_suffix: "footage"
```

### Crash Resilience
Location: `src/downloader.py`
- File-based skip logic (checks existing videos by ID)
- Auto-cleanup of `.part` and `.ytdl` files
- `--no-continue` flag to prevent resume hangs

### Variety Config
Location: `src/matching.py` → `StrategyMatcher`
Must handle as both dict and object:
```python
if isinstance(vc, dict):
    value = vc.get('require_different_source', True)
else:
    value = getattr(vc, 'require_different_source', True)
```

## Git Conventions

- **Commit prefixes:** `feat:`, `fix:`, `docs:`, `refactor:`
- **Branch naming:** Branches from Claude Code start with `claude/`
- Always push to the assigned feature branch, not main
- **Push branch to remote:** `git push -u origin branch-name`

## Adding New Config Options

1. Add field to appropriate dataclass in `src/config.py`:
   ```python
   new_option: int = 100  # Description
   ```

2. Add `__post_init__` if it's a nested dataclass:
   ```python
   def __post_init__(self):
       if self.nested is None:
           self.nested = NestedConfig()
   ```

3. Add to `config.yaml` with comment:
   ```yaml
   new_option: 100  # Description
   ```

4. Access in code with fallback:
   ```python
   value = getattr(self.config_section, 'new_option', 100)
   ```

5. Verify syntax:
   ```bash
   python -m py_compile src/config.py
   python -m py_compile main.py
   ```

## Project-Specific Overrides

Users can create `project_config.yaml` in project folders to override settings:
```yaml
keyword:
  max_keywords: 5  # Only 5 keywords for this project

pipeline:
  skip_download: false
  skip_image_search: true
```

## Testing Checklist

Before committing changes:
- [ ] `python -m py_compile main.py`
- [ ] `python -m py_compile src/config.py`
- [ ] `python -m py_compile src/downloader.py`
- [ ] `python -m py_compile src/matching.py`
- [ ] Config options in both `config.py` AND `config.yaml`
- [ ] Nested configs have `__post_init__` for dict conversion
- [ ] DownloadedVideo uses correct field names

## Session History

### 2025-12-30: Comprehensive fixes session
- **Pause-split:** Made aggressive (sentence-level, list isolation)
- **List detection:** Replaced regex with LLM-based detection
- **Live streams:** Added `!is_live` filter and title blacklist
- **Variety config:** Fixed dict vs object handling in matching.py
- **Pylance errors:** Added `from __future__ import annotations`
- **DownloadedVideo:** Fixed field names (file, duration_tier)
- **Config loading:** Enhanced `_build_dataclass` for string annotations

### 2025-12-31: CLAUDE.md expansion
- **Expanded documentation:** Added Common Bugs & Fixes, Key Features, Testing Checklist
- **Future annotations warning:** Documented `__post_init__` requirement for nested dataclasses
- **Regex gotcha:** Added lookbehind limitation and workaround
- **Project overrides:** Documented `project_config.yaml` feature

### 2026-01-01: Tier-specific download timeouts
- **Problem:** 120s timeout too short for medium/long videos
- **Solution:** Added `download_timeouts` dict with per-tier values
- **Files changed:** `src/config.py`, `src/downloader.py`, `config.yaml`
- **Timeouts:** short=120s, medium=300s, long=600s, longer=900s