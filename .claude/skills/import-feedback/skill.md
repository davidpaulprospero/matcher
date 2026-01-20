---
name: import-feedback
description: Import editorial feedback from DaVinci Resolve marker exports. Supports rejection/approval markers and channel scoring.
allowed-tools:
  - Read
  - Glob
  - Grep
  - Bash(python:*)
  - Write
---

# Import Feedback

Imports editorial feedback from DaVinci Resolve marker exports (CSV or EDL) into the rejection database.

## What It Does

| Marker Color | Name Prefix | Action |
|--------------|-------------|--------|
| Red | `REJECT:`, `BAD:`, `REMOVE:` | Add to rejection database |
| Green | `GOOD:`, `KEEP:`, `APPROVED:` | Boost channel score |
| Yellow | `REPLACE:`, `ALT:` | Flag for re-matching |
| Orange | `WARNING:`, `REVIEW:` | Flag for manual review |
| Blue | `NOTE:`, `INFO:` | Log only (informational) |

## DaVinci Resolve Workflow

1. Review timeline in DaVinci Resolve
2. Add markers to clips with feedback:
   - Red marker: "REJECT: trainer content"
   - Green marker: "GOOD: emotional moment"
   - Yellow marker: "REPLACE: better option available"
3. Export markers: Timeline > Export > Markers (CSV format)
4. Run `/import-feedback <project_path> <csv_path>`

## Instructions

When this skill is invoked:

### 1. Parse arguments

Extract from the user's request:
- **project_path**: Project directory (required)
- **csv_path**: Path to marker CSV file (optional - auto-detects if not provided)

Examples:
```
/import-feedback E:\Edit Job\Stu\January\6__2026-01-15
/import-feedback E:\Edit Job\Stu\January\6__2026-01-15 markers.csv
/import-feedback "E:\path\with spaces\Project" "C:\exports\davinci_markers.csv"
```

### 2. Find the marker file

If marker_path is not provided, search for it in order:
1. `project_path/markers.edl`
2. `project_path/markers.csv`
3. `project_path/davinci_markers.csv`
4. `project_path/feedback.csv`
5. `project_path/output/*/markers.*` (latest output first)

```bash
# Check common locations
ls "<project_path>/markers.edl" 2>/dev/null || \
ls "<project_path>/markers.csv" 2>/dev/null || \
ls "<project_path>/davinci_markers.csv" 2>/dev/null || \
ls "<project_path>/feedback.csv" 2>/dev/null || \
ls -t "<project_path>/output/"*/markers.* 2>/dev/null | head -1
```

### 3. Import the markers

Run the import command:

```bash
cd "D:/_Projects/voiceover-matcher-subtitle" && python -c "
from pathlib import Path
from src.feedback import (
    load_rejection_database,
    import_davinci_markers_full,
    apply_approvals_to_database,
    generate_feedback_report,
)

project_dir = Path('<project_path>')
csv_path = Path('<csv_path>')

# Load database
db = load_rejection_database(project_dir=project_dir, include_global=True)

# Import markers
result = import_davinci_markers_full(csv_path, project_dir)

# Apply rejections
if result.rejections:
    count = db.add_rejections(result.rejections)
    print(f'Added {count} new rejections')

# Apply approvals
if result.approvals:
    count = apply_approvals_to_database(result.approvals, db)
    print(f'Recorded {count} approvals')

# Save database
db.save()

# Generate report
report_path = project_dir / 'feedback_report.md'
generate_feedback_report(result, report_path)
print(f'Report: {report_path}')

# Print summary
print(f'\\nSummary: {result.summary()}')

if result.replacements:
    print(f'\\n{len(result.replacements)} segments need replacement:')
    for r in result.replacements:
        print(f'  - {r.timecode_str}: {r.title[:40]}... ({r.reason})')

if result.warnings:
    print(f'\\n{len(result.warnings)} warnings to review:')
    for w in result.warnings:
        print(f'  - {w.timecode_str}: {w.title[:40]}... ({w.reason})')

if result.unmapped:
    print(f'\\n{len(result.unmapped)} markers could not be mapped to videos')
"
```

### 4. Report results

After import, report:

```
## Feedback Import: <project_name>

### Source
- CSV: <csv_path>
- Markers found: X

### Actions Taken
- Rejections added: X
- Approvals recorded: X

### Follow-up Required
- **Replacements (Yellow)**: X segments need better alternatives
- **Warnings (Orange)**: X items need manual review
- **Unmapped**: X markers couldn't be mapped to videos

### Generated Files
- Feedback report: `feedback_report.md`

### Next Steps
1. Review `feedback_report.md` for details
2. Re-run matching: `python main.py --match-only --project "<project_path>"`
3. Replacement segments will use updated rejection database
```

### 5. If replacements found

If yellow (replace) markers were found, suggest re-running matching:

```bash
cd "D:/_Projects/voiceover-matcher-subtitle" && python main.py --match-only --project "<project_path>" --voiceover "<voiceover_path>"
```

## Usage Examples

```
# Auto-detect marker file in project (CSV or EDL)
/import-feedback E:\Edit Job\theresa\January\PaidTest__2026-01-19

# Specify EDL path
/import-feedback E:\Edit Job\Stu\January\22__2026-01-17 markers.edl

# Specify CSV path
/import-feedback E:\Edit Job\Stu\January\6__2026-01-15 E:\exports\markers.csv

# With spaces in paths
/import-feedback "E:\Edit Job\Client Name\Project" "C:\My Exports\feedback.csv"
```

## Supported Formats

| Format | Extension | Export From |
|--------|-----------|-------------|
| CSV | `.csv` | Timeline > Export > Markers > CSV |
| EDL | `.edl` | Timeline > Export > Markers > EDL |

Both formats are auto-detected. EDL is commonly used when CSV export is unavailable.

## Marker Naming Conventions

**Recommended format**: `PREFIX: description`

| Prefix | Meaning | Example |
|--------|---------|---------|
| `REJECT:` | Don't use this video again | `REJECT: trainer content` |
| `BAD:` | Low quality, wrong content | `BAD: blurry footage` |
| `GOOD:` | Use more from this channel | `GOOD: emotional moment` |
| `KEEP:` | This is perfect | `KEEP: perfect timing` |
| `REPLACE:` | Find alternative | `REPLACE: need happier shot` |
| `ALT:` | Use alternative | `ALT: V2 is better` |
| `REVIEW:` | Manual review needed | `REVIEW: copyright check` |
| `NOTE:` | Information only | `NOTE: audio sync issue` |

**Color-only markers**: If no prefix is used, the marker color determines the action:
- Red marker with any name = REJECT
- Green marker with any name = APPROVE
- Yellow marker with any name = REPLACE

## Conflict Resolution

When multiple markers are on the same video segment, the highest priority action wins:

| Priority | Action | Color | Effect |
|----------|--------|-------|--------|
| 1 (highest) | REJECT | Red | Don't use this video again |
| 2 | REPLACE | Yellow | Find alternative clip |
| 3 | WARNING | Orange | Needs manual review |
| 4 | APPROVE | Green | Use more from this channel |
| 5 (lowest) | NOTE | Blue | Informational only |

Example conflict warning:
```
⚠️ Conflicting markers on video 'abc123' at 01:00:03:06:
   REJECT: trainer content (Red)
   APPROVE: actually nice shot (Green)
   → Using REJECT (highest priority)
```

## Viewing Rejection Database Stats

```bash
cd "D:/_Projects/voiceover-matcher-subtitle" && python main.py --rejection-stats
```

## Troubleshooting

**"No marker file found"**
- Ensure CSV file exists in expected location
- Check file extension is `.csv`
- Use explicit path: `/import-feedback <project> <csv_path>`

**"X markers could not be mapped"**
- Markers at timecodes without video segments
- Ensure `timeline_segments.json` exists in output/
- Re-run pipeline if segments are missing
- Note: Markers within 2 seconds of a segment will snap to it

**"No rejections added"**
- Check marker naming (needs `REJECT:` prefix or red color)
- Verify CSV format (DaVinci Resolve export)

**"Timeline may have changed" warning**
- Marker CSV is newer than timeline_segments.json
- Re-run pipeline with `--match-only` to refresh segments
- Timecode mapping may be inaccurate if timeline was modified

**Timeline offset handling**
- DaVinci timelines typically start at `01:00:00:00` (1 hour offset)
- The importer auto-detects this from `timeline_start_tc` in segments
- Marker timecodes have this offset subtracted automatically

**Title fallback lookup**
- If timecode doesn't match, the importer looks for video titles in marker notes
- Include the video title in the Notes field to help mapping
- Useful for markers at gap positions or after timeline edits
