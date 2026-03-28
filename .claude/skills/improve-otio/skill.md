---
name: improve-otio
description: Improve OTIO timing by re-transcribing voiceover with start-anchored word-level accuracy, preserving all clip assignments from the previous generation
allowed-tools:
  - Read
  - Write
  - Glob
  - Grep
  - Bash(python:*)
  - Bash(python -m py_compile:*)
  - Agent
---

# Improve OTIO Timing Skill

Rebuild a previous OTIO generation with improved SRT timing while keeping all existing clip assignments (V1-V8) intact.

## When to Use

- After a pipeline run produced an OTIO with drifted timing
- When you want to re-transcribe voiceover with better word-onset accuracy
- To upgrade an old OTIO to use start-anchored contiguous timing

## Invocation

```
/improve-otio <project_path>
/improve-otio E:\Edit Job\Stu\3. THE $36 TRILLION BOMB...
/improve-otio E:\Edit Job\Degold\channel\project_dir
/improve-otio <project_path> --dry-run
```

## Arguments

- `<project_path>` (required): Path to the project directory
- `--dry-run`: Only re-transcribe and compare timing quality, don't rebuild OTIO
- `--output-folder <name>`: Use a specific source output folder instead of latest
- `--model <name>`: Whisper model (default: large-v3)

## Workflow

### Phase 1: Locate Source Data

1. Find the latest `output/<timestamp>/timeline_segments.json` in the project
2. Find the voiceover audio file (`voiceover/*.mp3` or `voiceover/*.wav`)
3. If either is missing, report error and stop

### Phase 2: Discover Segment Files

Before the expensive re-transcription, the script discovers where downloaded video segments live. This is critical because `timeline_segments.json` stores YouTube video IDs (e.g., `qtXrs0ZqX6w`), not file paths. Resolution chain:

1. **OTIO scan** (primary): Reads existing OTIOs (oldest first) looking for V1 clips with `target_url` pointing to actual segment files like `E:/v/matcher-alt/qtXrs0ZqX6w_139_154.mp4`. Extracts the parent directory.
2. **Config fallback**: Uses `config.downloaded_videos_dir` if it exists and contains segment files.

From the discovered directory, builds `DownloadedSegment` objects by parsing filenames (`{video_id}_{start}_{end}.mp4`). These are passed to `create_timeline()` which handles the video_id + source_time → segment_file + adjusted_offset mapping.

### Phase 3: Re-transcribe with Improved Timing

Run the re-transcription using the start-anchored pipeline. For paths with special characters ($, em dash, apostrophes), use `glob.glob()`:

```bash
python scripts/improve_otio_timing.py "<project_path>"
```

Or via Python for safe Windows path handling:

```bash
python -c "
import subprocess, sys, glob
dirs = glob.glob('E:/Edit Job/Stu/3. THE *BOMB*')
result = subprocess.run([sys.executable, 'scripts/improve_otio_timing.py', dirs[0]])
sys.exit(result.returncode)
"
```

The script:
1. Re-transcribes voiceover audio with `large-v3` + word-level timestamps
2. Applies `anchor_segment_starts_to_words()` for speech-onset accuracy
3. Applies `normalize_segments_start_anchored()` for contiguous timing (no gaps)
4. Writes `voiceover_improved.srt`

### Phase 4: Map Clips to New Segments

The script maps old clip assignments to new SRT segments by **word overlap**:
- For each new segment, finds the old segment with the most shared words
- Transfers V1 clip, alternatives (V2-V3), and secondaries (V4-V6)
- Falls back to temporal proximity for unmatched segments

### Phase 5: Rebuild OTIO

Generates a new output folder with:
- `timeline_FULL.otio` — Full OTIO timeline with improved timing
- `timeline_segments.json` — Updated segment map
- `timeline.edl` — EDL export
- `timeline_sequence.xml` — DaVinci Resolve XML
- `voiceover_improved.srt` — The improved SRT file

### Phase 6: Verify

After generation, verify:
1. V1 track has clips (not all gaps) — check the OTIO statistics output
2. Timing accuracy:

```
/verify-timing <project_path>
```

Expected: 0-2 frame drift throughout the entire timeline.

## What Gets Preserved

| Component | Preserved? |
|-----------|-----------|
| V1 primary clips | Yes - same source file and source timing |
| V2-V3 alternatives | Yes |
| V4-V6 secondaries | Yes |
| Match confidence/strategy | Yes |
| Voiceover segment TEXT | Re-transcribed (may differ slightly) |
| Voiceover segment TIMING | Improved (start-anchored) |

## What Changes

- **SRT timing**: Start times anchored to word-level speech onset (typically <250ms of CapCut quality)
- **Contiguity**: Zero gaps between segments (end[i] = start[i+1])
- **No cumulative drift**: Old approach drifted up to 16+ seconds by end of long voiceovers

## How It Works (Design)

The old normalization (`normalize_segments_contiguous`) chained segments by **duration**, shifting all start times. Over a 20-minute voiceover, gaps accumulate and cause multi-second drift.

The new approach (`normalize_segments_start_anchored`) preserves each segment's **original start time** from Whisper's word-level detection and makes contiguous by setting each segment's end to the next segment's start. Zero drift regardless of voiceover length.

## Troubleshooting

### "No output folder found"
The project needs at least one previous pipeline run with a `timeline_segments.json`.

### "No voiceover audio file found"
The voiceover audio (mp3/wav/m4a) must be in the project's `voiceover/` directory.

### V1 track 0% coverage / "Unresolved video ID" spam
The script couldn't map YouTube video IDs to local segment files. This happens when:
- No existing OTIO has populated V1 clips (all previous outputs were also broken)
- The segment files directory moved or `download.root_dir` changed since the original pipeline run
- The `.cache/transcriptions/` is empty (no hash-to-path mapping)

**Fix**: Ensure at least one `output/<timestamp>/timeline_FULL.otio` exists with real V1 clips pointing to the segment directory (e.g., `E:/v/matcher-alt/`). The script scans OTIOs oldest-first to discover the segment root. If the original pipeline output was deleted, you may need to re-run the full pipeline first.

### "No segment directory found" warning
The segment discovery couldn't find where `{video_id}_{start}_{end}.mp4` files are stored. The OTIO will be generated but with all gaps on video tracks. Check:
1. That a previous good OTIO exists in `output/`
2. That the segment directory it references still exists on disk
3. That `config.yaml` `download.root_dir` points to a valid location

### Segment count mismatch
If re-transcription produces a different number of segments, the word-overlap mapping handles this gracefully. Some clips may map to the nearest match.

### GPU not available
The script will auto-fallback to CPU if no GPU is available. Use `--model base` for faster CPU transcription (lower accuracy).

### Windows paths with special characters
Project paths containing `$`, em dashes (`—`), or apostrophes can break bash. Use `glob.glob()` in Python to resolve the path safely (see Phase 3 example).
