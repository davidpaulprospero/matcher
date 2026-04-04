# Generate Proxies

Generate Premiere-optimized proxy media for top-notch scrubbing performance.

## When to Use

- User says "generate proxies", "make proxies", "create proxies"
- User wants faster scrubbing/playback in Premiere
- User says "proxy" in the context of a project's media
- After a pipeline run when media will be edited in Premiere

## Arguments

- `<path>` (optional): Project path or segment directory. Defaults to `E:\Edit Job\Stu\all_segments`.
- `--preset <name>` (optional): `mjpeg` (default), `scrub`, `compact`, or `prores`
- `--workers <n>` (optional): Parallel encode workers (default: 4)
- `--scale <W:H>` (optional): Output resolution (default: 960:540)

## Examples

```
/generate-proxies
/generate-proxies "E:\Edit Job\Stu\all_segments"
/generate-proxies "E:\Edit Job\Stu\all_segments" --preset scrub
/generate-proxies "E:\Edit Job\Stu\all_segments" --scale 640:360
```

## Presets (benchmarked on RTX 3060 Ti, 90s 1080p clip)

| Preset | Codec | Decode Speed | Size (90s) | Projected 30GB src | Scrubbing |
|--------|-------|-------------|------------|-------------------|-----------|
| `mjpeg` | MJPEG Q12 | **859 fps** | 107 MB | ~150 GB | **Fastest** — each frame is a JPEG |
| `scrub` | H.264 All-Intra NVENC | 482 fps | 23 MB | ~33 GB | Great — all keyframes, smallest |
| `compact` | H.264 Short-GOP NVENC | ~400 fps | 15 MB | ~18 GB | Good — max 15-frame decode |
| `prores` | ProRes Proxy (CPU) | 465 fps | 174 MB | ~245 GB | Great — industry standard |

All presets output at 960x540 and force CFR to fix VFR YouTube scrubbing jank.

## Workflow

### Step 1: Determine Source Path

If user provides a specific project path, use that project's segment directory.
If no path given, default to `E:\Edit Job\Stu\all_segments`.

Resolve the source to an absolute path and verify it exists.

### Step 2: Run the Proxy Script

```bash
cd D:/_Projects/voiceover-matcher-dev
python scripts/generate_proxies.py "<source_path>" [--preset <preset>] [--workers <n>]
```

The script:
- Scans the source directory for video files (.mp4, .mov, .mkv, .webm, etc.)
- Skips files already in proxy directories
- Output goes to `F:\Premiere Cache\<source_name>_proxy\` by default
- Skips already-generated proxies (idempotent -- safe to re-run)
- Shows progress with percentage and per-file timing

### Step 3: Report Results

After completion, report:
- Total files encoded
- Total proxy size
- Output directory path
- Premiere attach instructions

### Step 4: Premiere Attachment Instructions

Tell the user:

1. In Premiere, select all clips that need proxies
2. Right-click > Proxy > Attach Proxies
3. Point to the proxy output directory
4. Premiere matches files by name (extension can differ)
5. Toggle proxies on/off with the "Toggle Proxies" button in the Program Monitor

## Output Location

Default output is `F:\Premiere Cache\<source_name>_proxy\` (if `F:\Premiere Cache` exists). Falls back to a `_proxy` sibling directory next to the source. Override with `--output <path>`.

## Script Location

`scripts/generate_proxies.py` -- standalone, no pipeline dependencies.
