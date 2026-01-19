# DaVinci Resolve Scripts

Python scripts for automating common DaVinci Resolve tasks, specifically designed to work with our pipeline output.

## Prerequisites

1. **DaVinci Resolve Studio** must be running (scripting requires Studio version)
2. **Python 3.6+** installed
3. Enable scripting in DaVinci Resolve:
   - Preferences > System > General
   - Set "External scripting using" to "Local"

## Installation

### Option 1: Run from command line
```bash
cd scripts/davinci
python quick_setup.py /path/to/output
```

### Option 2: Add to DaVinci Resolve Scripts menu
Copy these files to DaVinci Resolve's scripts folder:

**Windows:**
```
%APPDATA%\Roaming\Blackmagic Design\DaVinci Resolve\Support\Fusion\Scripts\Edit\
```

Then access via: Workspace > Scripts > Edit > [script_name]

## Quick Start

The recommended workflow for importing pipeline output:

```bash
# One-click setup: imports XML bins, OTIO timeline, configures tracks
python quick_setup.py "E:/Edit Job/Project/MyDoc__2026-01-15"

# Or auto-detect most recent project
python quick_setup.py --auto
```

## Available Scripts

### `quick_setup.py` - One-Click Import
The recommended way to import pipeline output. Does everything in order:
1. Imports XML media bins (populates Media Pool)
2. Imports OTIO timeline (auto-links to imported media)
3. Runs diagnostics
4. Configures track visibility

```bash
python quick_setup.py /path/to/output          # Specific output folder
python quick_setup.py /path/to/project         # Auto-finds latest output
python quick_setup.py --auto                   # Most recent E:/v project
python quick_setup.py --tracks broll           # Enable V1+V8 tracks
```

---

### `track_manager.py` - Track Visibility
Enable/disable video tracks according to our V1-V10 layout.

```bash
python track_manager.py                    # Show current status
python track_manager.py --primary          # Enable V1-V3 only
python track_manager.py --all              # Enable all tracks
python track_manager.py --broll            # V1 + V8 (B-roll)
python track_manager.py --entities         # V1 + V9-V10 (entity media)
python track_manager.py --enable 1,2,8     # Enable specific tracks
python track_manager.py --disable 4,5,6    # Disable specific tracks
python track_manager.py --toggle 8         # Toggle track 8
```

**Track Layout:**
| Track | Purpose |
|-------|---------|
| V1 | Primary video |
| V2 | Alternative 1 |
| V3 | Alternative 2 |
| V4-V6 | Secondary (diversity) |
| V7 | Embedding-Diversity |
| V8 | B-roll Only |
| V9 | Entity Images |
| V10 | Stock Videos |

---

### `diagnose_timeline.py` - Find Issues
Check timeline for offline media, problematic clips, and other issues.

```bash
python diagnose_timeline.py              # Full diagnosis
python diagnose_timeline.py --offline    # Only offline media
python diagnose_timeline.py --summary    # Brief summary
python diagnose_timeline.py --json       # JSON output for automation
```

**Detects:**
- Offline/missing media
- Unicode characters in paths
- Path length issues (>260 chars)
- Backslashes in paths (OTIO issue)
- Audio-only files (can cause issues)

---

### `relink_media.py` - Fix Offline Media
Automatically find and relink offline media from known locations.

```bash
python relink_media.py                    # Auto-relink all
python relink_media.py --dry-run          # Preview changes
python relink_media.py --interactive      # Confirm each relink
python relink_media.py --folder "D:/media" # Add custom search path
```

**Default search locations:**
- E:/v (short video path)
- E:/i (short image path)
- ~/.matcher_global_cache
- ~/.matcher_entity_cache

---

### `import_workflow.py` - Manual Import
Import XML bins and OTIO separately (for advanced use).

```bash
python import_workflow.py /path/to/output
python import_workflow.py --xml-only /path/to/media_bins.xml
python import_workflow.py --otio-only /path/to/timeline.otio
```

---

### `import_media_folder.py` - Bulk Import Media
Import media from folders into Media Pool.

```bash
python import_media_folder.py "E:/v/MyProject"  # Specific folder
python import_media_folder.py --all-v           # All from E:/v
python import_media_folder.py --recent 5        # 5 most recent
python import_media_folder.py --no-audio        # Skip audio files
python import_media_folder.py --list-only       # Preview only
```

---

### `export_all_formats.py` - Multi-Format Export
Export timeline to multiple formats at once.

```bash
python export_all_formats.py                    # Export to ~/exports/
python export_all_formats.py /path/to/output    # Specific folder
python export_all_formats.py --formats edl,xml  # Only specific formats
```

**Available formats:** aaf, edl, xml, fcpxml, otio, csv, tab

---

## Common Issues & Solutions

### "Media Offline" after OTIO import
OTIO files only contain filenames, not full paths. DaVinci searches configured directories but often fails.

**Solution:** Use `quick_setup.py` which imports XML bins first (they have full paths).

Or manually:
1. Run `import_workflow.py --xml-only` first
2. Then `import_workflow.py --otio-only`
3. Then `relink_media.py` if needed

### DaVinci hangs on large timeline
Timelines with 10k+ items can crash DaVinci.

**Solution:** Pipeline auto-splits large timelines into parts (PART1-PART4). Import parts separately.

### Unicode in filenames
Files with special characters (ñ, ü, etc.) may fail to link.

**Solution:** Run `diagnose_timeline.py` to identify, then manually relink.

### Tracks all disabled
By default, only V1-V3 are enabled.

**Solution:** Run `track_manager.py --all` or specific preset.

---

## Troubleshooting

### "Could not connect to DaVinci Resolve"
- Make sure DaVinci Resolve is running
- Check scripting is enabled (Preferences > System > General)
- Try restarting DaVinci Resolve

### "No project is open"
- Create or open a project in DaVinci Resolve before running scripts

### Scripts not appearing in menu
- Copy to correct location (see Installation)
- Restart DaVinci Resolve
- Check file has .py extension

### Permission errors
- Run DaVinci Resolve as administrator
- Check folder permissions on E:/v, E:/i
