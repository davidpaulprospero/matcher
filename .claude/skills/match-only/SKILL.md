---
name: match-only
description: Re-run matching and output generation for a project without re-downloading or transcribing. Use after config tweaks or to regenerate timeline.
allowed-tools:
  - Read
  - Bash(python:*)
---

# Match Only

Re-run the matching and output stages for a project using cached data. Skips download and transcription.

## When to Use

- After tweaking `config.yaml` or `project_config.yaml` matching settings
- To regenerate OTIO/XML/EDL with different output settings
- After fixing a bug in matching or output code
- To test different matching strategies without full pipeline run

## Instructions

When this skill is invoked:

1. **Parse the argument**: Extract the project path from the user's request. This should be a project directory (e.g., `E:/Edit Job/ProjectName__2026-01-13`).

2. **Validate the project**:
   - Check that the path exists
   - Check for `.cache/` folder (indicates prior run with cached data)
   - Check for `checkpoint.json` (ensures resumable state)

3. **Run match-only pipeline**:
```bash
cd "D:/_Projects/voiceover-matcher" && python main.py --project "<project_path>" --match-only --voiceover "<project_path>/voiceover/<srt_file>"
```

**Note:** Must specify `--voiceover` to avoid interactive prompt. Look for `.srt` file in voiceover folder.

4. **Report results**:
   - Show pipeline completion status
   - Note the output directory created
   - Suggest running `/validate-output` on the new output

## Usage Examples

```
/match-only E:/Edit Job/Stu/January/24__2026-01-13
/match-only "E:/path/with spaces/ProjectName__2026-01-10"
```

## What Gets Re-Run

| Stage | Runs? | Notes |
|-------|-------|-------|
| ANALYZE | Skip | Uses cached keywords/entities |
| ENTITY_IMAGES | Skip | Uses cached images |
| ENTITY_VIDEOS | Skip | Uses cached videos |
| DOWNLOAD | Skip | Uses cached videos |
| STOCK | Skip | Uses cached stock footage |
| BROLL_DOWNLOAD | Skip | Uses cached B-roll |
| REMIX | Skip | Uses cached filtered list |
| TRANSCRIBE | Skip | Uses cached transcriptions |
| SCENE_DETECTION | Skip | Uses cached scene data |
| MATCH | **Run** | Re-matches voiceover to videos |
| BROLL_MATCH | **Run** | Re-matches B-roll scenes |
| OUTPUT | **Run** | Regenerates OTIO/XML/EDL |

## Prerequisites

The project must have been run at least once (with download/transcribe) to have cached data:
- `.cache/transcriptions/` - Video transcriptions
- `.cache/embeddings/` - Vector embeddings
- `checkpoint.json` - Pipeline state

## Common Follow-up

After match-only completes, validate the output:
```
/validate-output <project_path>
```
