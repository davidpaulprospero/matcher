# Fix Trimmed OTIO

Fix OTIO timing for older pipeline runs where the timeline was built against the untrimmed voiceover but the editor uses the trimmed version. Produces a new output with all tracks (V1-V12) aligned to the trimmed audio.

## When to Use

- After importing an OTIO and clips don't align with the voiceover
- Previous pipeline run used original (untrimmed) voiceover timing
- Voiceover in DaVinci is shorter than the timeline
- User says "fix timing", "fix otio", "clips don't align", "timeline is longer than voiceover"

## Arguments

- `<project_path>` (required): Project directory path or Trello card ID
- `--dry-run`: Preview what would change without writing
- `--skip-output`: Fix checkpoint/SRT only, print the command to regenerate later

## What It Does

1. Ensures `voiceover_trimmed.mp3` exists (runs silence removal if not)
2. Detects whether checkpoint segments are in original or trimmed time
3. Remaps segments to trimmed time if needed (same count, different timestamps)
4. Writes corrected `voiceover_trimmed.srt`
5. Updates checkpoint with trimmed timing + `trimmed_segments` field
6. Runs `--output-only` to regenerate all tracks (V1-V12)

## Commands

```bash
# Standard fix
python scripts/fix_trimmed_otio.py "<project_path>"

# Dry run — preview only
python scripts/fix_trimmed_otio.py "<project_path>" --dry-run

# Fix checkpoint/SRT only, skip OTIO regen (useful on machines without GPU)
python scripts/fix_trimmed_otio.py "<project_path>" --skip-output
```

For paths with special characters, use glob:
```bash
python -c "
import subprocess, sys, glob
dirs = glob.glob('E:/Edit Job/Degold/channel/iSxux*')
result = subprocess.run([sys.executable, 'scripts/fix_trimmed_otio.py', dirs[0]])
sys.exit(result.returncode)
"
```

## Default Behavior

When the user runs `/fix-trimmed-otio <arg>`:

### If arg is a project directory path:

1. Run dry-run first to preview:
   ```bash
   python scripts/fix_trimmed_otio.py "PROJECT_PATH" --dry-run
   ```
2. On confirmation (or if timing is clearly wrong), run the actual fix:
   ```bash
   python scripts/fix_trimmed_otio.py "PROJECT_PATH"
   ```
3. Report the output folder and verify timing

### If arg is a Trello card ID:

1. Resolve to project directory (check Degold/Stu queue state, scan filesystem)
2. Proceed as with a project directory

## How It Works

The pipeline stores checkpoint segments in **original-audio time** (e.g., 1191s for a 20-min voiceover). But the editor imports `voiceover_trimmed.mp3` which has silence removed (e.g., 1015s). This mismatch causes clips to extend past the voiceover.

The fix:
- Loads speech regions from `voiceover_trimmed_regions.json`
- Maps each segment's start/end from original time to trimmed time
- The segment COUNT stays the same (match indices preserved)
- Only the timestamps change

## Output

A new `output/<timestamp>/` folder containing:
- `*_FULL.otio` — Full timeline with all tracks, trimmed timing
- `*_V1_*.otio` through `*_V12_*.otio` — Per-track OTIOs
- `*_A8_voiceover.otio` — Voiceover-only OTIO
- `*.edl` — EDL markers
- `*_sequence.xml`, `*_project.xml`, `*_media_part*.xml` — DaVinci XML
- `*_segments.json` — Segment mapping
- `voiceover_improved.srt` — Updated SRT

## Troubleshooting

| Issue | Fix |
|-------|-----|
| "No voiceover.mp3 found" | Voiceover file missing or in wrong location |
| "voiceover file is zero-filled" | Corrupted file from bad transfer. Re-download from Drive/Trello |
| "Segments already in trimmed time" | No fix needed, checkpoint is already correct |
| "Could not determine timing domain" | Script proceeds anyway — check output manually |
| Pipeline fails during regen | Check logs in `logs/` for the actual error |

## Works On

- Windows (primary development)
- Linux (autorun machine) — all paths use forward slashes, Python-only
