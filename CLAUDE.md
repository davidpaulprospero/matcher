# CLAUDE.md

This file is the quick working guide for coding agents in this repository.
For complete project context, use `AGENTS.md` as the source of truth.

## Project Snapshot

Matcher Pipeline Stages is a Python pipeline that aligns stock footage to voiceover scripts using embedding + LLM matching, then exports OTIO/EDL/XML timelines.

Core flow:

`ANALYZE -> CAPTION -> MATCH -> DOWNLOAD_SEGMENTS -> OUTPUT`

## Quick Commands

### Setup

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### Pipeline

```bash
# Standard run
python main.py --voiceover script.srt --project "E:\Projects\MyDoc"
python main.py --match-only                    # Re-run matching only
python main.py --output-only                   # Regenerate OTIO only (fastest, needs checkpoint)
python main.py --keyword-list "sunset,ocean" --keyword-mode montage --duration 60
python main.py --project "E:\Edit Job\theresa\Project" --client theresa  # Cross-project learning
python main.py --evolve-preset --client theresa  # Generate evolved preset from history
python main.py --voiceover script.srt --high-matches  # Iterate until 90%+ confidence

# Resume from checkpoint
python main.py --resume

# Re-run matching only
python main.py --match-only

# Regenerate output only
python main.py --output-only

# Fresh run (ignore checkpoint)
python main.py --fresh

# Non-interactive run for automation/tests
python main.py --voiceover script.srt --project "E:\Projects\MyDoc" --non-interactive
```

### OTIO Regeneration

```bash
python scripts/regenerate_otio.py "E:\Edit Job\client\project"
```

## Key CLI Flags

- `--voiceover`, `-v`: Voiceover input (SRT, MP3, WAV, MP4)
- `--project`, `-p`: Project directory
- `--config`, `-c`: Config file path
- `--match-only`: Skip download/transcribe stages
- `--output-only`: Regenerate OTIO/EDL/XML only
- `--resume` / `--fresh`: Continue from checkpoint or reset run
- `--non-interactive`: Required for deterministic test automation
- `--use-keywords [PRESET]`: Reuse saved keyword preset
- `--save-keywords [NAME]`: Save extracted keywords

## Architecture Overview

### Main Packages

- `src/stages/`: Stage entrypoints
- `src/config/`: Typed config loader + section dataclasses
- `src/llm_client/`: Provider abstraction (Gemini/Anthropic/Ollama)
- `src/matching/`: Match scoring and tiered matching
- `src/downloader/`: yt-dlp download logic and audio-first strategy
- `src/transcription/`: faster-whisper integration
- `src/otio/`: Timeline and export logic
- `src/agents/`: Self-healing orchestration
- `src/pipeline.py`: Pipeline orchestrator

### OTIO Track Intent

- `V1`: Primary match track (enabled)
- `V2-V3`: Alternatives
- `V4-V6`: Diversity tracks
- `V7`: Embedding-diversity strategy
- `V8`: B-roll only
- `V9`: Entity images
- `V10`: Stock videos
- `A9`: Voiceover

## Configuration Rules

### Source of Truth

- Global config: `config.yaml`
- Project override: `project_config.yaml` in the project folder (auto-loaded with `--project`)
- Do not pass `project_config.yaml` via `--config`

### Safe Access Pattern

```python
value = getattr(self.download_config, 'field_name', default_value)

if isinstance(config_section, dict):
    value = config_section.get('field_name', default)
else:
    value = getattr(config_section, 'field_name', default)
```

### When Adding New Config

1. Update dataclass in `src/config/sections/*.py`
2. Update default in `config.yaml`
3. Add `__post_init__` conversion when nested dataclasses can load as dict
4. Use backward-safe `getattr(..., default)` access in call sites

## Development Rules (Critical)

- Use `DownloadedVideo(file=..., duration_tier=...)` (not legacy arg names).
- Use `is_embeddings_empty()` for embedding checks (avoid numpy truthiness bugs).
- Route all LLM calls through `src/llm_client/`.
- Import core dataclasses from `src/state.py` or `src/config/`.
- OTIO media URLs must use `file:///E:/...` format with forward slashes.
- In caption-first mode, `source_file` may be a video ID (not a filesystem path).
- Ensure deep merge behavior for project config overrides.
- For subprocess calls with `text=True`, always include:
  - `encoding='utf-8'`
  - `errors='replace'`
- In downloader `_download_by_ids`, download one video at a time (no batch call).
- `WhisperClient.transcribe()` returns a **tuple** `(segments_list, info)`, not just a list. Always unpack: `segments, info = client.transcribe(...)` or `segments = result[0]`.
- Ollama provider (`src/llm_client/providers/ollama.py`) does NOT pass `system_prompt` to the API — concatenate system prompt into the `prompt` field.
- `clients/stu/pipeline_queue_state.json` pipeline keys are **lowercase** (e.g., `kxq0sogh`) but `card_id` values are mixed-case (`KxQ0SoGh`). Use `card_id.lower()` for lookup.
- For Windows paths with special chars (em dashes, `$`, apostrophes), use `glob.glob()` in Python rather than literal bash paths.

## Testing

### Fast Local Tests

```bash
python -m pytest tests/test_keyword_extractor/ tests/test_llm_client/ tests/test_otio/ -v
```

### Full Suite

```bash
python -m pytest tests/ -v
python -m pytest tests/ --cov=src --cov-report=html --cov-report=term
```

### Single Test

```bash
python -m pytest tests/test_otio_timeline.py::test_creates_basic_timeline -v
```

### Unified Autorun Smoke Tests

```bash
python -m pytest tests/test_unified_autorun.py -v --tb=short --no-cov
```

### Pre-commit Safety Check

```bash
python -m py_compile main.py
python -m py_compile src/config/base.py
pytest tests/ -v --tb=short -x
```

## Checkpoint and Cache

- Project checkpoint: `<project>/checkpoint.json`
- Backup checkpoint: `<project>/checkpoint.backup.json`
- Local cache root: `.cache/`
- Global video cache: `~/.matcher_global_cache/`

To force a later stage to re-run, set `last_completed_stage` in `checkpoint.json` to the stage before the target and run with `--resume`.

## Common Failure Patterns

- `403 Forbidden` on downloads: retry later, refresh cookies, or use VPN.
- `AttributeError: 'dict' object has no attribute ...`: missing dict-to-dataclass handling.
- `--output-only` behaves unexpectedly: verify checkpoint integrity.
- Windows subprocess encoding crashes: missing `encoding='utf-8', errors='replace'`.
- `[RATE-LIMIT:impersonation_rotation]` warnings are informational, not errors - they log after every successful request. True rate limiting shows HTTP 429/403 errors or explicit tier advancement logs.

## Google Drive (gws_drive)

- Use `scripts/gws_drive.py` for all Drive operations (NOT gdown — fails on Windows with long filenames/special chars).
- Token: `GOOGLE_WORKSPACE_CLI_TOKEN` from `clients/degold/accounts/david.env`
- Key functions (all require `context=GwsDriveContext(token=...)`):
  - `list_drive_folder_files(folder_id, *, context)` → `list[dict]` with `id`, `name`, `mimeType`, `modifiedTime`, `size`
  - `download_drive_file(file_id, destination: Path, *, context)` → `Path`
- Channel Drive folder IDs are in `clients/shared/channels.py`

## Useful References

- `AGENTS.md`: full project guide
- `README.md`: usage and setup
- `TESTING.md`: test organization and strategy
- `REFACTORING.md`: architecture roadmap
- `CHANGELOG.md`: version and migration notes
- `scripts/ralph/docs/AGENTS.md`: Ralph subsystem guide
