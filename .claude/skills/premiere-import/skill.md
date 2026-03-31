# Premiere Pro OTIO Import

Import pipeline-generated OTIO timelines into Adobe Premiere Pro via the pymiere bridge and QE API.

## Usage

```
/premiere-import <project_path>    # Import from a specific project's output
/premiere-import                   # Auto-detect from most recent pipeline run
```

## Prerequisites

- Adobe Premiere Pro 2026+ must be running (installed at `C:\Program Files\Adobe\Adobe Premiere Pro 2026\`)
- pymiere Python package installed (`pip install pymiere`)
- Pymiere Link CEP extension installed at `%APPDATA%/Adobe/CEP/extensions/com.qmasingarbe.PymiereLink/`
- CEP debug mode enabled: `HKCU\SOFTWARE\Adobe\CSXS.12\PlayerDebugMode = "1"`
- comtypes package for dialog handling (`pip install comtypes`)
- win32gui/pywin32 for screenshots (`pip install pywin32`)

## Connection Architecture

pymiere uses an HTTP bridge — a CEP panel inside Premiere runs a Node.js server on `localhost:3000`. Python sends ExtendScript code via POST, the panel evaluates it, and returns JSON results.

```
Python (pymiere) --> HTTP POST localhost:3000 --> CEP Panel (Node.js) --> ExtendScript eval --> Premiere Pro DOM
```

The panel process is separate from Premiere's main thread — `GET localhost:3000` responds even when Premiere is blocked by a dialog, but `POST eval` will hang until the dialog is dismissed.

## Import Workflow

### Step 1: Verify Premiere is alive and responsive

```python
import requests, json

PANEL_URL = "http://127.0.0.1:3000"

def check_premiere():
    """Check panel alive + eval responsive (no blocking dialog)."""
    try:
        requests.get(PANEL_URL, timeout=2)
    except:
        return "not_running"
    try:
        r = requests.post(PANEL_URL, json={"to_eval": "app.project.name;"}, timeout=3)
        return f"ready:{r.text}"
    except requests.exceptions.Timeout:
        return "blocked_by_dialog"
```

### Step 2: Convert OTIO from Clip.2 to Clip.1

Premiere Pro 26's OTIOProjectConverter expects `Clip.1` schema with singular `media_reference`. Our pipeline outputs `Clip.2` with plural `media_references`. Must downgrade before import.

Use `scripts/convert_otio_for_premiere.py`:

```python
from convert_otio_for_premiere import convert_otio_for_premiere
convert_otio_for_premiere(input_otio, output_otio)
```

What it does:
- Changes `Clip.2` -> `Clip.1` schema
- Moves `media_references.DEFAULT_MEDIA` -> `media_reference` (singular)
- Removes `active_media_reference_key` field
- Strips `Resolve_OTIO` metadata (DaVinci-specific, confuses Premiere)

**Why this is lossless**: Our pipeline only ever sets one reference per clip (`DEFAULT_MEDIA`). Clip.2 added multi-reference support (full-res, proxy, audio-only) but we don't use it. Everything else — `source_range`, `effects`, `markers`, `name`, `enabled`, clip metadata (confidence, segment_id, etc.) — is identical between Clip.1 and Clip.2. Track structure, gaps, and timeline metadata are unchanged.

### Step 3: Copy to clean path

Premiere chokes on special characters in file paths (em dashes `—`, `$`, apostrophes). Always copy the converted OTIO to a clean ASCII path before import:

```python
import shutil
clean_path = r"C:\Users\daves\Documents\Adobe\Premiere Pro\26.0\timeline_import.otio"
shutil.copy2(converted_otio, clean_path)
```

### Step 4: Import via QE API

**CRITICAL**: Use `qe.project.importFiles()`, NOT `app.project.importFiles()`. Only the QE version routes through the OTIOProjectConverter.

```python
import json, requests

path_json = json.dumps(clean_path.replace("\\", "/"))
r = requests.post(PANEL_URL, json={"to_eval": f"""
app.enableQE();
try {{
    var result = qe.project.importFiles([{path_json}]);
    JSON.stringify({{success: true, result: result}});
}} catch(e) {{
    JSON.stringify({{success: false, error: e.toString()}});
}}
"""}, timeout=120)
```

### Step 5: Handle dialogs

Import operations can trigger modal dialogs that block ExtendScript eval. Run a dialog watcher in a background thread.

Use `scripts/premiere_dialog_handler.py`:
- Detects blocking state: panel alive but eval times out
- Finds dialog windows by title (File Import Failure, EDL Information, Import Files, etc.)
- Auto-handles: clicks OK on EDL Information (NTSC default), dismisses failures, closes unknown dialogs
- Uses Win32 `EnumWindows` + `SendMessage`/`PostMessage`

**ALWAYS check for dialogs before and during any Premiere API call.** Dialogs block the entire ExtendScript thread and cause Python timeouts.

```python
from premiere_dialog_handler import find_premiere_dialogs, handle_dialog, check_premiere_blocked

blocked, msg = check_premiere_blocked()
if blocked:
    for d in find_premiere_dialogs():
        handle_dialog(d)
```

For persistent blocks with no visible dialog, take a screenshot to see what Premiere is showing:
```python
from PIL import ImageGrab
import win32gui
rect = win32gui.GetWindowRect(premiere_hwnd)
img = ImageGrab.grab(bbox=rect)
img.save("premiere_screenshot.png")
```

### Step 6: Verify import

```python
r = requests.post(PANEL_URL, json={"to_eval": """
var seqs = [];
for (var i = 0; i < app.project.sequences.numSequences; i++) {
    var s = app.project.sequences[i];
    var vClips = 0, aClips = 0;
    for (var t = 0; t < s.videoTracks.numTracks; t++) vClips += s.videoTracks[t].clips.numItems;
    for (var t = 0; t < s.audioTracks.numTracks; t++) aClips += s.audioTracks[t].clips.numItems;
    seqs.push({name: s.name, videoTracks: s.videoTracks.numTracks, audioTracks: s.audioTracks.numTracks, videoClips: vClips, audioClips: aClips});
}
JSON.stringify(seqs);
"""}, timeout=10)
```

## Verified Import Results

Tested with a real 17-track pipeline output (5,703 clips). Import completed in 18.3 seconds:

| Track | Clips | Muted | Purpose |
|-------|-------|-------|---------|
| V1 - Primary | 547 | no | Main edit track |
| V2 - Alternative 1 | 488 | yes | Alt options |
| V3 - Alternative 2 | 467 | yes | Alt options |
| V4 - Secondary Primary | 516 | yes | Secondary tier |
| V5 - Secondary Alt 1 | 451 | yes | Secondary tier |
| V6 - Secondary Alt 2 | 382 | yes | Secondary tier |
| V9-V12 (Entity/Stock/Generated) | 0 | yes | Empty tracks preserved |
| A1 - Video Audio | 547 | no | Paired with V1 |
| A2-A6 | 382-516 | yes | Paired with V2-V6 |
| A7 - Voiceover | 1 | no | Main voiceover |

Track names, mute states, clip positions, in/out points, and media references all transfer correctly.

## Key Gotchas

### Import methods — what works vs what doesn't

| Method | OTIO? | EDL? | XML? | Notes |
|--------|-------|------|------|-------|
| `qe.project.importFiles()` | YES | YES | YES | Routes through project converters — the correct method |
| `app.project.importFiles()` | NO | dialog | dialog | Treats OTIO as unknown media; EDL/XML open config dialogs |
| `app.openDocument()` | NO | - | - | Tries to replace current project, gets stuck on save prompt |
| `app.openFCPXML()` | - | - | maybe | Exists but untested, may work for FCP XML |

### Premiere OTIO support details

- `OTIOProjectConverter.prm` plugin at `C:\Program Files\Adobe\Adobe Premiere Pro 2026\PlugIns\Common\`
- Other converters in same directory: `AAFProjectConverter.prm`, `ALEProjectConverter.prm`, `FCPXProjectConverter.prm`, `IMFProjectConverter.prm`
- Loaded by `ProjectConverterHost.dll`
- Beta feature flag `B_ProjectConverters.OTIO: true` (enabled by default in v26, visible in Sentry crash data)
- Only reads `Clip.1` schema — `Clip.2` with `media_references` (plural) silently produces empty import (no error, no clips)

### EDL import works too

EDL import via `qe.project.importFiles()` works but always shows an "EDL Information" dialog asking for video standard (NTSC/PAL/24P). The dialog handler auto-clicks OK with NTSC selected. EDL import created a sequence with all 10 video tracks and 2,851 clips.

### Dialog handling rules

- `GET localhost:3000` responds even during dialog — panel runs in separate CEP process
- `POST eval` hangs when Premiere has a modal dialog — use timeout (3-5s) to detect
- EDL imports always show "EDL Information" dialog (NTSC/PAL/24P) — auto-click OK
- "File Import Failure" dialogs — dismiss with `WM_CLOSE`
- "Import Files" (file picker) — dismiss with Enter/WM_CLOSE
- `suppressUI=true` does NOT suppress all dialogs
- Some embedded dialogs (Save Changes) don't appear as separate windows — send keystrokes to main Premiere window (`VK_ESCAPE` then `VK_RETURN`)

### Media references

- OTIO `target_url` must point to real files on disk (verified: all 547 clips resolved)
- Premiere resolves paths relative to OTIO file location first, then absolute
- Use forward slashes in URLs: `E:/v/matcher-alt/clip.mp4`

## Crash Report Reading

Premiere stores crash data via Sentry at:
```
C:\Users\daves\AppData\Local\Temp\Adobe\Premiere Pro\26.0\SentryIO-db\
```

Key files:
- `last_crash` — timestamp of most recent crash
- `<run-id>.run/__sentry-event` — crash event (msgpack-ish format, readable as text): level, release, tags, GPU info, app state
- `<run-id>.run/__sentry-breadcrumb1` — event trail leading to crash: thread hangs, exceptions, state transitions
- `<run-id>.run/session.json` — session metadata (start time, duration, status)

Useful tags in crash events: `secondsRunning`, `lastAppState`, `B_ProjectConverters.OTIO`, `gpu-renderer`, `ram_in_megs`, `nvidia_drv_0`

## Scripts

| Script | Purpose |
|--------|---------|
| `scripts/convert_otio_for_premiere.py` | Clip.2 -> Clip.1 downgrade + metadata cleanup |
| `scripts/premiere_dialog_handler.py` | Win32 dialog detection, reading, and auto-dismiss |
| `scripts/dismiss_dialog.py` | Simple Win32 dialog finder + WM_CLOSE sender |
| `scripts/read_dialog.py` | Full UI Automation dialog reader (verbose, for debugging; requires `comtypes`) |

## Premiere Pro API Reference (Verified Working)

### Connecting
```python
import requests, json
PANEL_URL = "http://127.0.0.1:3000"
r = requests.post(PANEL_URL, json={"to_eval": "app.project.name;"}, timeout=10)
```

### Discovering hidden methods
ExtendScript `for (var k in obj)` misses many methods. Use `reflect` instead:
```javascript
var r = app.reflect;
for (var i = 0; i < r.methods.length; i++) methods.push(r.methods[i].name);
```

Key hidden methods discovered: `app.openDocument()`, `app.openFCPXML()`, `app.enableQE()`, `app.setWorkspace()`, `app.isDocumentOpen()`

### Project & sequence
```javascript
app.project.name                          // project name
app.project.path                          // project file path
app.project.sequences.numSequences        // sequence count
app.project.sequences[i]                  // sequence by index
app.project.activeSequence                // current timeline
app.project.rootItem.children             // media pool items
app.project.importFiles([paths], true)    // import media (not OTIO)
app.project.createNewSequence(name)       // new empty sequence
```

### Reading the timeline
```javascript
var seq = app.project.activeSequence;
seq.name                                  // "Matched Footage"
seq.frameSizeHorizontal                   // 1920
seq.frameSizeVertical                     // 1080
seq.videoTracks.numTracks                 // 10
seq.audioTracks.numTracks                 // 7
seq.videoTracks[0].name                   // "V1 - Primary"
seq.videoTracks[0].clips.numItems         // 547
seq.videoTracks[0].isMuted()              // false
seq.videoTracks[0].isLocked()             // false
```

### Clip manipulation (TrackItem) — all verified
```javascript
var clip = seq.videoTracks[0].clips[0];
clip.name                                 // "[S000] matcher-alt_..."
clip.start.seconds                        // 0 (timeline position, R/W)
clip.end.seconds                          // 1.167 (R/W)
clip.duration.seconds                     // 1.167 (read-only)
clip.inPoint.seconds                      // 5.833 (source in, R/W)
clip.outPoint.seconds                     // 7.0 (source out, R/W)
clip.getSpeed()                           // 1.0 (read-only via DOM)
clip.isSpeedReversed()                    // 0
clip.move(newInPoint)                     // reposition clip (PPro 15.4+)
clip.remove(inRipple, inAlignToVideo)     // delete clip
clip.disabled                             // true/false (R/W, mute clip)
clip.setSelected(state, updateUI)         // select in timeline
clip.getLinkedItems()                     // linked audio/video pair
```

### Effects & keyframes — all verified
```javascript
// Built-in components on every clip:
clip.components[0]                        // Opacity
clip.components[0].properties[0]          // Opacity value
clip.components[1]                        // Motion
clip.components[1].properties[0]          // Position [0.5, 0.5]
clip.components[1].properties[1]          // Scale (100)
clip.components[1].properties[4]          // Rotation

// Read/write values:
opacity.getValue()                        // 100
opacity.setValue(50, true)                // set to 50%

// Keyframes:
opacity.setTimeVarying(true)             // enable keyframes
opacity.addKey(timeObj)                   // add keyframe
opacity.setValueAtKey(timeObj, 0, true)  // set value at key
opacity.getKeys()                         // get all keyframes
opacity.areKeyframesSupported()          // true
```

### Markers — verified
```javascript
var marker = seq.markers.createMarker(5.0);
marker.name = "Low Confidence";
marker.comments = "62% match - review this segment";
marker.setColorByIndex(1);               // 0=green 1=red 2=purple 3=orange 4=yellow 5=white 6=blue 7=cyan
seq.markers.deleteMarker(marker);        // cleanup
```

### Export capabilities — verified available
```javascript
seq.exportAsMediaDirect(path, preset, workArea)  // render directly
seq.exportAsFinalCutProXML(path, suppressUI)     // export FCP XML
seq.exportAsProject(path)                         // export .prproj
seq.insertClip(item, time)                        // add clip to timeline
seq.overwriteClip(item, time)                     // overwrite at position
seq.clone()                                        // duplicate sequence
```

### QE API (undocumented but powerful)
```javascript
app.enableQE();
qe.project.importFiles([path]);                   // Import with project converter (OTIO/EDL/XML/AAF)
qe.project.getVideoEffectByName("Gaussian Blur"); // Find effects by name
qe.project.getVideoEffectByName("Cross Dissolve");// Find transitions
qe.project.numSequences                           // sequence count
qe.project.getActiveSequence()                    // active sequence (QE version)
qe.project.importFailures                         // failed import list
var track = seq.getVideoTrackAt(0);
var item = track.getItemAt(0);
item.addVideoEffect(effect);                      // Apply effect to clip
```

### QE project methods (full list)
`close`, `deletePreviewFiles`, `findItemByID`, `flushCache`, `getActiveSequence`, `getAudioEffectByName`, `getAudioEffectList`, `getAudioTransitionByName`, `getAudioTransitionList`, `getBinAt`, `getItemAt`, `getRemainingMetadataCacheIndexCount`, `getRendererNames`, `getSequenceAt`, `getSequenceItemAt`, `getVideoEffectByName`, `getVideoEffectList`, `getVideoTransitionByName`, `getVideoTransitionList`, `import`, `importAEComps`, `importAllAEComps`, `importFiles`, `importPSD`, `importProject`, `init`, `newBarsAndTone`, `newBin`, `newBlackVideo`, `newColorMatte`, `newSequence`, `newSmartBin`, `newTransparentVideo`, `newUniversalCountingLeader`, `redo`, `resetNumFilesCounter`, `save`, `saveAs`, `setRenderer`, `sizeOnDisk`, `undo`, `undoStackIndex`

## Premiere vs DaVinci — API Comparison

| Capability | Premiere (pymiere) | DaVinci (Resolve API) |
|---|---|---|
| Move clips on timeline | YES (`clip.move()`, `clip.start=`) | NO (must delete + re-append) |
| Set opacity | YES (with keyframes) | Limited (DRX grades only) |
| Set position/scale/rotation | YES (Motion component) | NO |
| Apply effects by name | YES (QE API) | Limited |
| Create keyframes | YES (full keyframe API) | NO |
| Import OTIO | YES (via QE + Clip.1 conversion) | YES (native) |
| Headless rendering | NO (GUI required) | YES (`-nogui` flag) |
| Clip speed changes | Read-only (DOM), write via QE | YES (`SetProperty("Speed")`) |
| Native markers | YES (colors + comments) | YES (colors + notes) |
| Timeline export | FCP XML, project, direct render | OTIO, EDL, XML, AAF |
