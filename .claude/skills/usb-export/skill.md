# USB Export

Safely copy project files to a USB drive for the Linux autorun machine, with post-copy integrity verification.

## When to Use

- Transferring projects to the USB drive for the Linux overnight autorun
- User says "copy to USB", "export to USB", "put on the drive", "send to Linux"

## Arguments

- `<project_path>` (required): Project directory path, Trello card ID, or "all" for ready queue
- `<drive_letter>` (optional): USB drive letter (e.g., G:). Auto-detected if only one removable drive.

## Workflow

### Step 1: Detect USB Drive

```bash
wmic logicaldisk where drivetype=2 get name,volumename
```

If multiple removable drives, ask the user which one. If one drive, use it automatically.

### Step 2: Resolve Project Path

If arg is a project directory, use it directly.

If arg is a Trello card ID (8-char alphanumeric):
```bash
python -c "
import json, glob
# Check Degold queue
state = json.load(open('Degold/pipeline_queue_state.json'))
card = state.get('pipelines', {}).get('CARD_ID_LOWER', {})
print(card.get('local_project_dir', ''))
print(card.get('name', ''))
"
```

Also scan filesystem: `ls -d E:/Edit\ Job/Degold/*/CARD_ID-* 2>/dev/null`

### Step 3: Preview Copy

Show what will be copied and what will be excluded:

Use this `should_include` helper throughout preview, copy, and verify steps:

```python
def should_include(rel_path):
    """Include everything except .cache subdirs that aren't .cache/v/.

    Returns True (include), False (exclude), or 'traverse' (.cache root
    — recurse into it but only keep the 'v' subdirectory).
    """
    parts = rel_path.replace(os.sep, '/').split('/')
    if parts[0] == '.cache':
        if len(parts) == 1:
            return 'traverse'  # at .cache itself — filter subdirs
        if parts[1] == 'v':
            return True  # .cache/v/ = yt-dlp segments — keep
        return False  # .cache/embeddings, .cache/audio, etc — skip
    return True
```

Use in `os.walk` loops:

```python
for root, dirs, files in os.walk(proj):
    rel = os.path.relpath(root, proj)
    result = should_include(rel)

    if result == False:
        dirs.clear()
        continue
    if result == 'traverse':
        dirs[:] = [d for d in dirs if d == 'v']
        continue

    # process files here
```

### Step 4: Copy Files

Use Python's shutil for safe Windows path handling (special chars, em dashes, etc.):

```python
import shutil, os, glob, sys, time

proj = 'PROJECT_PATH'
proj_name = os.path.basename(proj)
usb = 'G:'  # detected drive
dest = os.path.join(usb, os.sep, proj_name)

# Remove existing destination if present
if os.path.exists(dest):
    shutil.rmtree(dest)

copied = 0
total_bytes = 0

for root, dirs, files in os.walk(proj):
    rel_root = os.path.relpath(root, proj)
    result = should_include(rel_root)

    if result == False:
        dirs.clear()
        continue
    if result == 'traverse':
        dirs[:] = [d for d in dirs if d == 'v']
        continue

    dest_dir = os.path.join(dest, rel_root) if rel_root != '.' else dest
    os.makedirs(dest_dir, exist_ok=True)

    for f in files:
        src_file = os.path.join(root, f)
        dst_file = os.path.join(dest_dir, f)
        size = os.path.getsize(src_file)
        shutil.copy2(src_file, dst_file)
        copied += 1
        total_bytes += size

print(f'Copied {copied} files ({total_bytes/1024/1024:.1f} MB)')
```

### Step 5: Verify Integrity

**Critical** - Verify copied files are not zero-filled (this has happened before with USB transfers):

```python
import os

dest = 'USB_DEST_PATH'
bad = []
checked = 0

for root, dirs, files in os.walk(dest):
    for f in files:
        full = os.path.join(root, f)
        size = os.path.getsize(full)
        if size == 0:
            continue  # Empty files are OK (lock files, etc.)
        with open(full, 'rb') as fh:
            header = fh.read(min(64, size))
        if not any(b != 0 for b in header):
            bad.append((os.path.relpath(full, dest), size))
        checked += 1

if bad:
    print(f'WARNING: {len(bad)} zero-filled files detected!')
    for path, size in bad[:10]:
        print(f'  {path} ({size:,} bytes)')
else:
    print(f'All {checked} files verified OK')
```

### Step 6: Size Comparison

Compare source vs destination to catch truncation:

```python
import os

src_sizes = {}
for root, dirs, files in os.walk(src_proj):
    rel = os.path.relpath(root, src_proj)
    if not should_include(rel):
        dirs.clear()
        continue
    for f in files:
        rel_path = os.path.join(rel, f) if rel != '.' else f
        src_sizes[rel_path] = os.path.getsize(os.path.join(root, f))

mismatches = []
for rel_path, src_size in src_sizes.items():
    dst_path = os.path.join(dest, rel_path)
    if not os.path.exists(dst_path):
        mismatches.append(f'MISSING: {rel_path}')
    else:
        dst_size = os.path.getsize(dst_path)
        if dst_size != src_size:
            mismatches.append(f'SIZE MISMATCH: {rel_path} (src={src_size}, dst={dst_size})')

if mismatches:
    print(f'ERRORS: {len(mismatches)} issues found')
    for m in mismatches[:10]:
        print(f'  {m}')
else:
    print(f'Size verification passed: {len(src_sizes)} files match')
```

## What Gets Copied

| Directory | Included? | Why |
|-----------|-----------|-----|
| `voiceover/` | Yes | Audio + SRT needed for pipeline |
| `.cache/v/` | Yes | Downloaded yt-dlp segments (saves re-downloading) |
| `stock/` | Yes | Stock footage (Pexels, Shutterstock) |
| `dvids/` | Yes | DVIDS Hub clips |
| `output/` | Yes | Pipeline outputs (OTIO, EDL, XML) |
| `logs/` | Yes | Run logs for reference |
| `checkpoint.json` | Yes | Pipeline state for resume |
| `*.sh`, `*.json` | Yes | Config and run scripts |
| `.cache/embeddings/` | No | Regenerated by pipeline |
| `.cache/audio/` | No | Temporary processing cache |
| `.cache/transcriptions/` | No | Temporary transcription cache |
| `.cache/keyframes/` | No | Temporary keyframe cache |
| `.cache/scenes/` | No | Temporary scene cache |
| `.cache/index/` | No | Temporary index cache |
| `.cache/llm_responses/` | No | Temporary LLM cache |

## Safety

- Uses `shutil.copy2` (preserves metadata, no sparse file issues)
- Post-copy verification: checks every file for zero-fill corruption
- Size comparison: detects truncation or missing files
- Never deletes source files
- Uses Python `os.walk` + `glob.glob` for Windows special character safety

## Example

```
/usb-export E:\Edit Job\Degold\DeepSeaReports\iSxuxqjE-Why USS Gerald R. Ford Chose Greek__2026-03-28
/usb-export iSxuxqjE
/usb-export E:\Edit Job\Degold\DeepSeaReports\iSxuxqjE-Why USS Gerald R. Ford Chose Greek__2026-03-28 G:
```
