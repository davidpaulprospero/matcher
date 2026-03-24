# Lipsync Submit Skill

## Primary Workflow: Playwright MCP Browser

**ALWAYS use Playwright MCP browser for lipsync submission.** Do NOT use the `auto_lipsync.py` API approach.

The Playwright workflow:
1. Prep files (detect channel, trim audio, select avatar)
2. Navigate to the Degold form in the MCP browser
3. Fill fields, upload files via `browser_run_code` + `setInputFiles`
4. Click Submit, keep tab open for processing

### Prep Helper (for file preparation only)

Use `auto_lipsync.py` helper functions or manual steps for prep work:
- Detect channel from folder name
- Trim voiceover to 59 seconds
- Select/rotate avatar
- Copy files to `Degold/temp_upload/` for browser access

Do NOT call `auto_lipsync.py` to submit — it uses the API, not the browser.

### Fallback: auto_lipsync.py API (only if Playwright unavailable)

```bash
python Degold/auto_lipsync.py "E:\Edit Job\Degold\DeepSeaReports\3dWWwtJc-Project..."
```

Options:
- `--title "Custom Title"` - Override auto-detected title
- `--channel DSR` - Override auto-detected channel
- `--no-trim` - Skip 1-minute trimming

## Trigger Phrases

- "submit lipsync"
- "lipsync job"
- "submit to degold"
- "generate lipsync"
- "lipsync auto"
- "auto lipsync"

## Auto-Detection

### Channel Detection from Folder Name

The auto_lipsync.py script detects channel from project path:

| Folder Pattern | Channel |
|---------------|---------|
| DeepSeaReports | DSR |
| RennReports | RRU |
| JournalOfDrunkPeople | JDRP |
| Contains "DSR" | DSR |
| Contains "RRU" | RRU |

### Voiceover Detection

Looks in order:
1. `<project>/voiceover/` subfolder
2. `<project>/` root

File extensions: .mp3, .wav, .m4a, .mp4

**ALWAYS prefer `voiceover_trimmed.mp3` over `voiceover.mp3`** when it exists. The trimmed version has silence/dead air removed by the pipeline and produces better lipsyncs. Fall back to `voiceover.mp3` only if no trimmed version is available.

## Parameters

- `$VIDEO_TITLE` - Video title (optional, will prompt if not provided)
- `$CHANNEL_CODE` - Channel code like RRU, JDRP, DSR (optional)
- `$AVATAR_PATH` - Path to avatar image (optional)
- `$AUDIO_FILES` - Audio segment files (optional)
- `$TRELLO_URL` - Trello card URL (optional, will extract title automatically)

## Parsing Titles from Trello URLs

If a Trello URL is provided (e.g., `https://trello.com/c/fr9huYEa/103-1-minute-ago-underwater-drone...`), fetch the card title using the Trello API.

### Method 1: Trello API (Preferred)

Use the Trello API to get the actual card title:

1. **Extract card ID** from URL: `https://trello.com/c/fr9huYEa/...` → card ID is `fr9huYEa`
2. **Load Trello credentials** - Try accounts in order: `stuart.env`, `david.env`
3. **Call API**:
   ```
   GET https://api.trello.com/1/cards/{card_id}?fields=name&key={api_key}&token={token}
   ```
4. **Response** contains `name` field with the exact card title

**Important:** Handle encoding - the API may return garbled em dashes. Fix with:
```python
card_name = response.json()["name"]
card_name = card_name.replace('\ufffd', '\u2014')  # Replace replacement char with em dash
```

Example using Python:
```python
import requests
import os
from dotenv import load_dotenv

# Try accounts in order (stuart usually has access)
for account in ["stuart", "david"]:
    try:
        load_dotenv(f"Degold/accounts/{account}.env")
        api_key = os.getenv("TRELLO_API_KEY")
        token = os.getenv("TRELLO_TOKEN")
        if not api_key or not token:
            continue

        response = requests.get(
            f"https://api.trello.com/1/cards/{card_id}",
            params={"fields": "name", "key": api_key, "token": token}
        )
        if response.status_code == 200:
            card_name = response.json()["name"]
            card_name = card_name.replace('\ufffd', '\u2014')  # Fix encoding
            break
    except:
        continue
```

### Method 2: URL Slug Fallback

If API fails, parse from URL slug:
1. Extract slug from URL path after `/c/[card-id]/`
2. Replace hyphens with spaces
3. Capitalize appropriately
4. Convert double-hyphens to em dashes (`—`)

## Workflow

### Step 0: Check Drive for Existing Lipsync (ALWAYS DO FIRST)

Before gathering info or submitting, check if a lipsync video already exists in the channel's Drive folder. This avoids redundant processing.

1. **Detect channel** from folder name (DeepSeaReports → DSR, RennReports → RRU)
2. **Look up `drive_folder`** from `Degold/channels.py`
3. **List Drive folder** and search for a matching video by title keywords

```python
import sys, os
from pathlib import Path
sys.path.insert(0, 'scripts')
from dotenv import load_dotenv
from gws_drive import GwsDriveContext, list_drive_folder_files

load_dotenv('Degold/accounts/david.env')
ctx = GwsDriveContext(token=os.getenv('GOOGLE_WORKSPACE_CLI_TOKEN', ''))

# List channel's Drive folder
files = list_drive_folder_files(drive_folder_id, context=ctx)

# Search for matching lipsync by title keywords from project folder name
# Extract 2-3 distinctive words from the title to match against filenames
for f in files:
    print(f'{f["modifiedTime"]}  {f["name"]}  ({f["id"]})')
```

**Matching strategy:** Extract distinctive keywords from the project folder name (e.g., "USS Charlotte TORPEDOED" → search for "charlotte" and "torpedoed" in Drive filenames). Match is case-insensitive.

4. **If a match is found:**
   - Report the existing file to the user with name, date, and size
   - Ask: "Lipsync already exists in Drive. Download it instead of resubmitting? (yes/no)"
   - If yes → skip to Step 6 (download) using the matched file's ID
   - If no → continue with normal submission workflow

5. **If no match found:** Continue to Step 1 as normal.

### Step 1: Gather Information

If not all parameters provided, ask user for:

1. **Video Title** - Title for the generated video (e.g., "EP42 - Trabant Secrets")
   - If user provides a Trello URL, extract the title from the API using the rules in "Parsing Titles from Trello URLs"
2. **Channel Code** - Channel to use (RRU, JDRP, DSR, etc.)
3. **Avatar Path** - Path to avatar image file (optional - will download from Google Drive if not provided)
4. **Audio Files** - List of audio segment files in order

**Trello URL Detection:** If the user provides a Trello URL (contains `trello.com/c/`), automatically:
1. Extract the card ID from the URL
2. Fetch the title using the Trello API
3. Present the parsed title to the user for confirmation

**Auto-Download Avatar:** If no avatar path is provided:
1. Get the `avatar_folder` ID from `Degold/channels.py` for the channel
2. Download the avatar from Google Drive using `gws_drive` (see Step 2b)
3. Save to a local path preserving the original filename (e.g., `Degold/avatars/DSR/Harold_V1.jpg`)
4. Use the local path for the form submission

### Step 2: Cross-Reference with CHANNELS.md or channels.py

Read `Degold/channels.py` and verify:
- Channel code exists in the config
- Get the drive_folder ID for the output
- Get the avatar_folder ID for downloading the avatar

### Step 2b: Download Avatar from Google Drive (if needed)

If no avatar path is provided, download the avatar from Google Drive using `gws_drive`:

```python
import sys, os
from pathlib import Path
sys.path.insert(0, 'scripts')
from dotenv import load_dotenv
from gws_drive import GwsDriveContext, list_drive_folder_files, download_drive_file

load_dotenv('Degold/accounts/david.env')
ctx = GwsDriveContext(token=os.getenv('GOOGLE_WORKSPACE_CLI_TOKEN', ''))

# List avatar folder contents
avatar_folder_id = "16cHw8fgefC89zelexv_OSQNoqzhKekwO"  # RRU example
files = list_drive_folder_files(avatar_folder_id, context=ctx)

# Download avatar images
avatars_dir = Path("Degold/avatars/RRU")
avatars_dir.mkdir(parents=True, exist_ok=True)
for f in files:
    if f['mimeType'].startswith('image/'):
        download_drive_file(f['id'], avatars_dir / f['name'], context=ctx)
```

**Do NOT use gdown** — it fails on Windows with long filenames containing special characters.
**Note:** The Google Drive folder may contain multiple avatars.

#### Avatar Rotation Tracking

For channels with multiple avatars (like DSR with V1-V6), use the avatar tracker to cycle through them randomly while ensuring each is used once before repeating.

**Python helper:**
```python
import json
import random
from pathlib import Path

TRACKER_FILE = Path("Degold/avatars/avatar_usage.json")

def get_next_avatar(channel: str) -> str:
    """Get next avatar for channel, cycling randomly but using all once before repeating."""
    with open(TRACKER_FILE, 'r') as f:
        tracker = json.load(f)

    if channel not in tracker:
        raise ValueError(f"Channel {channel} not in tracker")

    data = tracker[channel]
    avatars = data['avatars']
    used_order = data.get('used_order', [])
    current_index = data.get('current_index', 0)

    # If we've used all avatars, shuffle and reset
    if current_index >= len(avatars):
        random.shuffle(avatars)
        used_order = []
        current_index = 0
        data['avatars'] = avatars

    # Get next avatar
    avatar = avatars[current_index]
    used_order.append(avatar)
    current_index += 1

    # Update tracker
    data['used_order'] = used_order
    data['current_index'] = current_index

    with open(TRACKER_FILE, 'w') as f:
        json.dump(tracker, f, indent=2)

    return avatar
```

**Usage:**
```python
avatar_file = get_next_avatar("DSR")
# Returns: "Harold_V5.jpg" (random, then cycles through all 6 before repeating)

avatar_path = f"Degold/avatars/DSR/{avatar_file}"
```

**Tracker file:** `Degold/avatars/avatar_usage.json`

### Step 3: Confirm with User

Show the user a summary:

```
Lipsync Job Submission (Playwright)
==================================
Video Title: [title]
Channel: [channel code]
  - Drive Folder: [folder_id from channels.py]
  - Avatar Folder: [avatar_folder from channels.py]
Avatar: [path]
Audio Files: [list]
  -> Each file will be trimmed to 1 minute before upload

Type "yes" or "y" to confirm and submit.
```

Wait for user confirmation before proceeding.

### Step 3b: Auto-Trim Audio to 1 Minute (ALWAYS REQUIRED!)

**The Degold form only accepts audio up to 1 minute!** This is NOT optional - ALL audio files must be trimmed to **59 seconds** (not 60) before upload to avoid edge-case rejections.

For each audio file:
1. Check if a `_1min` version already exists
2. If not, run ffmpeg to trim to 60 seconds
3. Use the trimmed file for upload

```python
from pathlib import Path

def trim_audio_to_1min(audio_path: str) -> str:
    """Trim audio to first 60 seconds using ffmpeg."""
    audio_path_obj = Path(audio_path)
    suffix = audio_path_obj.suffix
    trimmed_path = audio_path_obj.parent / f"{audio_path_obj.stem}_1min{suffix}"

    # Skip if already trimmed
    if trimmed_path.exists():
        print(f"  Using existing trimmed file: {trimmed_path}")
        return str(trimmed_path)

    print(f"  Trimming {audio_path} to 1 minute...")

    result = subprocess.run(
        ["ffmpeg", "-i", audio_path, "-t", "59", "-c", "copy", str(trimmed_path), "-y"],
        capture_output=True,
        text=True,
        timeout=60
    )

    if result.returncode == 0 and trimmed_path.exists():
        print(f"  [OK] Saved trimmed audio: {trimmed_path}")
        return str(trimmed_path)
    else:
        print(f"  [WARN] Trim failed, using original: {result.stderr[:200] if result.stderr else ''}")
        return audio_path
```

Or via command line:
```bash
ffmpeg -i "input.mp3" -t 59 -c copy "input_1min.mp3" -y
```

**Do NOT ask the user about trimming - just do it automatically.**

### Step 4: Run Playwright Automation

**IMPORTANT: Use Playwright MCP tools (not Python playwright)!** The MCP browser stays open in the Claude Code session, allowing processing to complete. Python playwright scripts close when the script ends, which aborts processing.

Use the Playwright MCP tools to automate the form:

**Form URL:** https://degoldmedia.duckdns.org/form/a82405ee-4d7f-4ff6-9142-cd97c909897c

**Step 4a: Navigate to the form**
```
mcp__plugin_playwright_playwright__browser_navigate
url: "https://degoldmedia.duckdns.org/form/a82405ee-4d7f-4ff6-9142-cd97c909897c"
```

**Step 4b: Wait for form to load**
```
mcp__plugin_playwright_playwright__browser_wait_for
text: "Video Title"
time: 5
```

**Step 4c: Fill in the form fields**
```
mcp__plugin_playwright_playwright__browser_fill_form
fields: [
  {"name": "Video Title", "type": "textbox", "ref": "<field-0 ref>", "value": video_title},
  {"name": "Channel Code", "type": "combobox", "ref": "<field-1 ref>", "value": channel_code},
  {"name": "Drive Folder ID", "type": "textbox", "ref": "<field-4 ref>", "value": drive_folder_id}
]
```

**Step 4d: Upload avatar image**

Use `browser_run_code` with `setInputFiles()` (browser_file_upload has a bug):

```
mcp__plugin_playwright_playwright__browser_run_code
code: |
  async (page) => {
    await page.setInputFiles('input[name="field-2"]', 'D:/_Projects/.../avatar.jpg');
    return 'Avatar uploaded';
  }
```

**Step 4e: Upload audio files**

Same approach - use `browser_run_code`:

```
mcp__plugin_playwright_playwright__browser_run_code
code: |
  async (page) => {
    await page.setInputFiles('input[name="field-3"]', 'D:/_Projects/.../audio.mp3');
    return 'Audio uploaded';
  }
```

**Step 4f: Take screenshot to verify**
```
mcp__plugin_playwright_playwright__browser_take_screenshot
filename: "lipsync-form-filled.png"
type: "png"
```

**Step 4g: Click submit button**

Find and click the submit button.

**Step 4h: Wait for submission confirmation**

The form does NOT show a "success" message. Instead, submission is confirmed when:
- The Submit button becomes [disabled]
- Processing happens in the background

**Do NOT refresh or navigate away after clicking submit!** Just leave the browser open and inform the user the job has been submitted.

```
# After clicking submit, check if button is disabled
# If disabled, submission was successful
mcp__plugin_playwright_playwright__browser_snapshot
# Look for: button "Submit" [disabled]
```

If the button is disabled, the job has been submitted. Take a final screenshot and report success.

### Step 5: Report Result & Monitor Completion

**CRITICAL: The browser tab MUST remain open for processing to complete!**

The n8n backend requires the browser tab to remain open during processing. If you close the tab before processing finishes, the job will NOT be processed.

After submission:
1. Confirm the Submit button is disabled (submission accepted)
2. Tell the user: "Job submitted. KEEP THE BROWSER OPEN - processing is happening in the background"
3. The URL will change to `form-waiting/XXX` — this is the processing page

**Checking completion status:**
- Use `browser_tabs(list)` to see all open tabs
- Select a tab with `browser_tabs(select, index)` then `browser_snapshot`
- **Completed:** Page text shows `"AI Lipsync for [TITLE] is complete"`
- **Still processing:** Page shows form-waiting without "complete" text
- Always check tab status before reporting to the user — don't assume jobs are still running
- Once complete, the tab can safely be closed

### Step 6: Download Lipsync Videos to Project Directory

Once the browser tab confirms "is complete" (Step 5), download the generated lipsync videos from the channel's Drive folder into each project's `lipsync/` subfolder.

**Use gws_drive (NOT gdown)** — gdown fails on Windows with long filenames containing special characters.

```python
import sys, os
from pathlib import Path
sys.path.insert(0, 'scripts')
from dotenv import load_dotenv
from gws_drive import GwsDriveContext, list_drive_folder_files, download_drive_file

load_dotenv('Degold/accounts/david.env')
token = os.getenv('GOOGLE_WORKSPACE_CLI_TOKEN', '')
ctx = GwsDriveContext(token=token)

# 1. List the channel's Drive folder to find new lipsync videos
drive_folder_id = "1XJY8HUEWvFH0cI68tvyrPEyU7bTpeNLQ"  # RRU example
files = list_drive_folder_files(drive_folder_id, context=ctx)

# 2. Find the target file by matching name keywords
for f in files:
    if 'f35' in f['name'].lower():
        file_id = f['id']
        break

# 3. Download to project lipsync/ subfolder
dest = Path("E:/Edit Job/Degold/RennReports/ProjectName__2026-03-16/lipsync")
dest.mkdir(parents=True, exist_ok=True)
out_path = dest / "lipsync_video.mp4"
download_drive_file(file_id, out_path, context=ctx)
```

**Key gws_drive API signatures:**
- `list_drive_folder_files(folder_id: str, *, context: GwsDriveContext, page_size: int = 200) -> list[dict]`
- `download_drive_file(file_id: str, destination: Path, *, context: GwsDriveContext, timeout_seconds: int = 300) -> Path`
- `GwsDriveContext(token: str = '', credentials_file: str = '', impersonated_user: str = '')`

**Matching the right files:** The Drive folder accumulates all past lipsync videos. Match by:
- Audio filename prefix in the video name (e.g., `f35_voiceover_1min` → `f35_voiceover_1min - Title.mp4`)
- Most recent `modifiedTime` for duplicate names
- Each result dict has: `id`, `name`, `mimeType`, `modifiedTime`, `size`

## Error Handling

- If channel code not found in channels.py, warn the user
- If avatar or audio files don't exist, report the error
- If form doesn't load, try refreshing and wait longer
- If submission fails, show screenshot of error and suggest checking manually

## Key Learnings

### Auto-Detection Improvements (2025-03)
- Channel auto-detected from folder name (DeepSeaReports → DSR)
- Avatar automatically rotated through V1-V6 using `avatar_usage.json` tracker
- Audio auto-trimmed to 1 minute (required by Degold limit)
- Title extracted from SRT or folder name if not provided
- Files copied to project directory for API/MCP access — **always preserve original filenames** (e.g., `Harold_V1.jpg`, not `{shortId}_avatar.jpg`)

### Files Must Be in Project Directory
- Playwright MCP can only access files within the project directory
- Copy files to `Degold/temp_upload/` before uploading via Playwright
- Copy from external drives (E:\) to project dir before upload

### Trello API Credentials
- The `stuart.env` account may not have access to all boards (returns "unauthorized card permission requested")
- The `david.env` account has broader access - try it if stuart fails
- Card IDs are the short code in the URL: `trello.com/c/Fr9huYEa/...` → `Fr9huYEa`

### RRU Avatar: RennActor.jpg
- RRU uses `RennActor.jpg` (single avatar, no rotation needed)
- Located at `Degold/avatars/RRU/RennActor.jpg`
- Download from Drive avatar folder `16cHw8fgefC89zelexv_OSQNoqzhKekwO` if missing
- The `auto_lipsync.py` script and `avatar_usage.json` tracker must both use `RennActor.jpg` (not `rru_avatar.jpg`)

### Playwright is Primary, API is Fallback
- **ALWAYS use Playwright MCP browser** for lipsync submission
- Do NOT use `auto_lipsync.py` to submit — it uses the API which bypasses browser processing
- Use `auto_lipsync.py` only for prep work (trim audio, detect channel)
- Playwright workflow: navigate to form URL, fill fields, upload via `browser_run_code` + `setInputFiles`, click Submit
- The browser tab MUST stay open during processing

### Audio File Naming Convention
- The trimmed 59s audio file MUST be named after the video title, not the source voiceover filename
- This is because the audio filename appears in the Drive output (e.g., `{audio_name} - {title}.mp4`)
- Use `trim_audio_to_1min(voiceover_path, title=video_title)` which sanitizes the title for filename safety
- Example: title "US Uses Secret Kamikaze Drones" → `US_Uses_Secret_Kamikaze_Drones.mp3`
- The filename should be ONLY the title — no `_1min`, no card ID, no prefixes
- For Playwright workflow, trim manually with ffmpeg:
  ```bash
  ffmpeg -i "voiceover.mp3" -t 59 -c copy "Degold/temp_upload/US_Uses_Secret_Kamikaze_Drones.mp3" -y
  ```

### Playwright Submission Quick Reference
```
1. Prep:
   a. Get title (from Trello card, SRT, or folder name)
   b. Trim audio to 59s, named after title: {sanitized_title}_1min.mp3
   c. Copy avatar + titled audio to Degold/temp_upload/
2. Navigate: browser_navigate to form URL
3. Fill: browser_fill_form for title, channel (combobox), drive folder ID
4. Upload avatar: browser_run_code → page.setInputFiles('input[name="field-2"]', avatar_path)
5. Upload audio: browser_run_code → page.setInputFiles('input[name="field-3"]', audio_path)
6. Verify: browser_snapshot — check all fields filled, × buttons on uploads
7. Submit: browser_click Submit button
8. Confirm: snapshot shows Submit [disabled] with spinner = processing active
9. DO NOT close tab until "AI Lipsync for ... is complete" appears
```

### DSR Avatar: Use Harold_V1 Only
- DSR lipsync jobs fail with Harold_V2–V6 avatars — the n8n backend returns "Form Submitted / Your response has been recorded" (failure) instead of generating the lipsync video
- **Always use `Harold_V1.jpg`** for DSR submissions until this is resolved
- RRU avatars (RennActor.jpg) work fine — this issue is DSR-specific
- The avatar rotation tracker can still be used for other channels, but DSR should be hardcoded to V1

### File Upload Path Limitation
- Playwright MCP can only access files within the project directory (`D:\_Projects\voiceover-matcher-stable`)
- If audio/avatar files are on external drives (e.g., `E:\...`), copy them to the project directory first
- **IMPORTANT: Always preserve the original filename** when copying files — do NOT rename to generic names like `{shortId}_avatar.jpg` or `{shortId}_audio_1min.mp3`. The filename is visible in the form and should be descriptive (e.g., `Harold_V1.jpg`, `3dWWwtJc-How-USS-Charlotte-SANK_1min.mp3`).
  ```bash
  # Avatar: keep original name
  cp "Degold/avatars/DSR/Harold_V1.jpg" "Degold/avatars/Harold_V1.jpg"
  # Audio: trim in-place with _1min suffix on original name
  ffmpeg -i "E:/.../voiceover/3dWWwtJc-How-USS-Charlotte-SANK.mp3" -t 59 -c copy "Degold/avatars/3dWWwtJc-How-USS-Charlotte-SANK_1min.mp3" -y
  ```
- Use forward slashes for paths in `browser_file_upload`: `D:/_Projects/...`

### Submission & Processing Confirmation
- **CRITICAL: The browser tab MUST remain open for processing to complete!**
- If you close the tab before processing finishes, the job will NOT be processed
- **Processing states:**
  1. **In-progress:** Form stays visible with filled data, Submit button shows [disabled] with a loading spinner. The form URL stays the same. This is the active processing state — DO NOT close or navigate away.
  2. **Success:** Page shows `"AI Lipsync for [TITLE] is complete"`. The lipsync video will appear in the Drive folder.
  3. **Failure:** Page shows `"Form Submitted — Your response has been recorded"` — this means the n8n backend FAILED to generate the lipsync. The video will NOT appear in Drive. Resubmission may be needed.
- **Checking completion:** Use `browser_tabs` to list tabs, then `browser_tabs(select, index)` + `browser_snapshot` to inspect each
  - If snapshot shows the form with disabled Submit + spinner → still processing
  - If snapshot shows "AI Lipsync for [TITLE] is complete" → **success**, tab can safely be closed
  - If snapshot shows "Form Submitted / Your response has been recorded" → **failure**, needs investigation/resubmission
- **Always check tab status before reporting progress** to the user — don't assume jobs are still processing
- **Always verify in Drive** that the lipsync video actually appeared, even if the tab shows success

### Python Playwright vs MCP Browser
- **Python Playwright script**: Browser closes when script ends (even with `input()` - doesn't work in Claude Code)
- **Playwright MCP browser**: Stays open as part of Claude Code session until you explicitly close it or session ends
- **Always use MCP browser** for lipsync submission to ensure browser stays open for processing
- The MCP browser is the recommended approach - it stays alive for the duration of the conversation

### Multiple Jobs: Use Separate Tabs (CRITICAL)
- **NEVER navigate away from a submitted form** — navigating to a new URL kills the n8n processing connection
- When submitting multiple lipsync jobs, open each in a **separate browser tab** using `browser_tabs` with `action: "new"`
- Submit Job 1 in Tab 0, then open Tab 1 for Job 2, etc.
- Each tab keeps its own connection alive independently
- Workflow: `browser_tabs(new)` → `browser_navigate(form URL)` → fill & submit → repeat in next tab

### Empty/Blank Tabs (about:blank)
- `browser_tabs` with `action: "new"` always opens `about:blank` — it requires explicit `browser_navigate` after
- If a tab shows `about:blank` or an empty URL after navigation, **refresh by re-navigating** to the form URL
- Empty tabs can happen from: navigation timeout, network blip, DNS failure, or MCP connection hiccup
- Always check the snapshot after `browser_navigate` — if the page title is empty or URL is `about:blank`, navigate again

### Downloading Results: Use gws_drive (NOT gdown)
- **gdown fails on Windows** with long filenames containing special characters (em dashes, quotes) — `OSError: [Errno 22] Invalid argument`
- **gdown downloads the ENTIRE folder** including all past lipsync videos — very slow and wasteful
- **Always use `gws_drive`** from `scripts/gws_drive.py` — it downloads individual files by ID
- Load credentials from `Degold/accounts/david.env` → `GOOGLE_WORKSPACE_CLI_TOKEN`
- Download to `<project_dir>/lipsync/<descriptive_name>.mp4`
- Clean up any temp gdown downloads if they were attempted: `rm -rf Degold/temp_*`

### File Upload in MCP Browser
- The `browser_file_upload` tool has a bug that rejects valid JSON arrays
- Use `browser_run_code` with `page.setInputFiles()` instead:
  ```javascript
  await page.setInputFiles('input[name="field-2"]', 'D:/_Projects/.../avatar.jpg');
  await page.setInputFiles('input[name="field-3"]', 'D:/_Projects/.../audio.mp3');
  ```

## Playwright MCP Tool Reference

| Tool | Purpose |
|------|---------|
| `browser_tabs` | List/create/select tabs — use `action: "new"` for multi-job submissions |
| `browser_navigate` | Go to form URL (required after opening new tab) |
| `browser_snapshot` | Get current page state — check for `about:blank` after navigation |
| `browser_fill_form` | Fill text fields and dropdowns |
| `browser_run_code` | Upload files using `page.setInputFiles()` (preferred) |
| `browser_click` | Click buttons |
| `browser_take_screenshot` | Capture current view |
| `browser_wait_for` | Wait for text/element |
| `browser_evaluate` | Run JS if needed |

## Example Sessions

### Example 0: Lipsync Already Exists in Drive

```
User: submit lipsync for E:\Edit Job\Degold\DeepSeaReports\USS Charlotte TORPEDOED...

Assistant: [Checks Drive for existing lipsync...]

Found existing lipsync in Drive:
  Name: 3dWWwtJc_audio_1min - USS Charlotte TORPEDOED Iran's Warship...
  Date: 2026-03-17 05:53 UTC
  Size: 9.0 MB

Lipsync already exists in Drive. Download it instead of resubmitting? (yes/no)

User: yes

Assistant: [Downloads to project lipsync/ folder]
Downloaded to: E:\Edit Job\...\lipsync\USS Charlotte TORPEDOED - lipsync.mp4 (9.0 MB)
```

### Example 1: Fully Automated (No Existing Lipsync)

```
User: submit lipsync for E:\Edit Job\Degold\DeepSeaReports\3dWWwtJc-Project

Assistant: [Checks Drive - no existing lipsync found]

Running auto lipsync submitter...

[INFO] Project: 3dWWwtJc-Project
[INFO] Detected channel: DSR (from folder 'DeepSeaReports')
[INFO] Found voiceover: .../voiceover/3dWWwtJc-Voiceover.mp3
[INFO] Trimming audio to 1 minute...
[OK] Created trimmed audio
[INFO] Avatar: DSR/Harold_V1.jpg
[INFO] Title: How USS Charlotte Sank an Iranian Warship (from SRT)

[INFO] Submitting job...
[OK] Job submitted successfully!

Result: Done! Check Google Drive folder for output.
```

### Example 2: With Trello URL

```
User: submit lipsync for this trello card: https://trello.com/c/fr9huYEa/103-1-minute-ago-underwater-drone-entered-the-last-known-position-of-flight-mh370-the-structure-found-on-the-ocean-floor-is-not-natu

Assistant: I'll extract the title from that Trello URL and prepare the lipsync job.

Parsed Title: 1 MINUTE AGO: Underwater Drone Entered the Last Known Position of Flight MH370 — The Structure Found on the Ocean Floor Is Not Natural

Now I need:
1. Channel code (RRU, JDRP, DSR, etc.)
2. Avatar path
3. Audio files

[User provides: RRU, Degold/avatars/face.png, audio/ep103_01.mp3, audio/ep103_02.mp3]

Lipsync Job Submission (Playwright)
==================================
Video Title: 1 MINUTE AGO: Underwater Drone Entered the Last Known Position of Flight MH370 — The Structure Found on the Ocean Floor Is Not Natural
Channel: RRU
  - Drive Folder: 1XJY8HUEWvFH0cI68tvyrPEyU7bTpeNLQ
  - Avatar Folder: 16cHw8fgefC89zelexv_OSQNoqzhKekwO
Avatar: Degold/avatars/face.png
Audio Files: audio/ep103_01.mp3, audio/ep103_02.mp3
  -> Each file will be trimmed to 1 minute before upload

Type "yes" or "y" to confirm and submit.

User: yes

[Assistant runs Playwright automation...]

Result: Job submitted successfully! KEEP THE BROWSER OPEN - processing is happening in the background.
```

### Example 3: Manual Title

```
User: submit lipsync job for EP42 with RRU channel

Assistant: I'll help you submit a lipsync job. I need a few details:

1. Avatar image path (e.g., "Degold/avatars/face.png"):
   - Provide path

2. Audio files (list in order):
   - Provide paths

[User provides: Degold/avatars/face.png, audio/01.mp3, audio/02.mp3]

Assistant: Let me verify the channel and then confirm:

Lipsync Job Submission (Playwright)
==================================
Video Title: EP42
Channel: RRU
  - Drive Folder: 1XJY8HUEWvFH0cI68tvyrPEyU7bTpeNLQ
  - Avatar Folder: 16cHw8fgefC89zelexv_OSQNoqzhKekwO
Avatar: Degold/avatars/face.png
Audio: audio/01.mp3, audio/02.mp3

Type "yes" to confirm and submit.

User: yes

[Assistant runs Playwright automation...]

Result: Job submitted successfully! Screenshot shows confirmation page.
```
