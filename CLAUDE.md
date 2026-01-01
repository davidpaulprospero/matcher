# Claude Code Project Guide

This file helps Claude Code sessions understand project conventions and avoid common mistakes.

## Critical Rules

### 1. Config Synchronization (IMPORTANT!)
When adding/modifying configuration options:
- **Always update BOTH files:**
  - `src/config.py` - Python dataclass with type hints and defaults
  - `config.yaml` - User-facing YAML config with comments
- They must stay in sync or users won't see new options

### 2. Config Access Pattern
```python
# Safe config access with fallback:
value = getattr(self.download_config, 'field_name', default_value)

# For nested configs (e.g., dicts):
tier_timeouts = getattr(self.download_config, 'download_timeouts', {})
timeout = tier_timeouts.get('medium', 120)
```

## File Structure

| File | Purpose |
|------|---------|
| `config.yaml` | User-editable settings (YAML format) |
| `src/config.py` | Dataclasses defining config schema with defaults |
| `src/downloader.py` | Video download logic, uses `self.download_config` |
| `src/matching.py` | Video-to-voiceover matching engine |
| `src/transcription.py` | Audio transcription (faster-whisper) |

## Git Conventions

- **Commit prefixes:** `feat:`, `fix:`, `docs:`, `refactor:`
- **Branch naming:** Branches from Claude Code start with `claude/`
- Always push to the assigned feature branch, not main

## Adding New Config Options

1. Add field to appropriate dataclass in `src/config.py`:
   ```python
   new_option: int = 100  # Description of what it does
   ```

2. Add to `config.yaml` with comment:
   ```yaml
   new_option: 100  # Description of what it does
   ```

3. Access in code:
   ```python
   value = getattr(self.config_section, 'new_option', 100)
   ```

## Session History

### 2026-01-01: Tier-specific download timeouts
- **Problem:** 120s timeout too short for medium/long videos
- **Solution:** Added `download_timeouts` dict with per-tier values
- **Files changed:** `src/config.py`, `src/downloader.py`, `config.yaml`
- **Timeouts:** short=120s, medium=300s, long=600s, longer=900s
