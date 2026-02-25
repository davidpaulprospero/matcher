# Lipsync Submit Skill (Playwright)

## Description

Submit an AI Lipsync job to the Degold form using Playwright browser automation. More reliable than API calls which can be finicky.

## Trigger Phrases

- "submit lipsync"
- "lipsync job"
- "submit to degold"
- "generate lipsync"
- "lipsync with playwright"

## Parameters

- `$VIDEO_TITLE` - Video title (optional, will prompt if not provided)
- `$CHANNEL_CODE` - Channel code like RRU, JDRP, DSR (optional)
- `$AVATAR_PATH` - Path to avatar image (optional)
- `$AUDIO_FILES` - Audio segment files (optional)

## Workflow

### Step 1: Gather Information

If not all parameters provided, ask user for:

1. **Video Title** - Title for the generated video (e.g., "EP42 - Trabant Secrets")
2. **Channel Code** - Channel to use (RRU, JDRP, DSR, etc.)
3. **Avatar Path** - Path to avatar image file
4. **Audio Files** - List of audio segment files in order

### Step 2: Cross-Reference with CHANNELS.md or channels.py

Read `Degold/channels.py` and verify:
- Channel code exists in the config
- Get the drive_folder ID for the output

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

Type "yes" or "y" to confirm and submit.
```

Wait for user confirmation before proceeding.

### Step 4: Run Playwright Automation

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

CRITICAL: Must click the upload button FIRST to trigger file chooser modal, THEN use file upload.

```
# Click the avatar upload button - this triggers the file chooser modal
mcp__plugin_playwright_playwright__browser_click
ref: "e18"  # Avatar Image button

# Now upload the file
mcp__plugin_playwright_playwright__browser_file_upload
paths: ["C:/path/to/avatar.png"]
```

**Step 4e: Upload audio files**

Same pattern - click the button first, then upload:

```
# Click the audio upload button
mcp__plugin_playwright_playwright__browser_click
ref: "e21"  # Audio Segments button

# Upload multiple audio files
mcp__plugin_playwright_playwright__browser_file_upload
paths: ["C:/path/to/audio1.mp3", "C:/path/to/audio2.mp3"]
```

**Step 4f: Take screenshot to verify**
```
mcp__plugin_playwright_playwright__browser_take_screenshot
filename: "lipsync-form-filled.png"
type: "png"
```

**Step 4g: Click submit button**

Find and click the submit button.

**Step 4h: Wait for success and take screenshot**
```
mcp__plugin_playwright_playwright__browser_wait_for
text: "success"
time: 10
```

Take final screenshot to confirm submission.

### Step 5: Report Result

**IMPORTANT: DO NOT close the browser after submission!**

Leave the browser open so the user can see:
- The result page (success or error)
- Any error messages displayed on the form
- The user may want to manually retry or check the form

Show the user:
- Screenshot of filled form (before submit)
- Screenshot of result (after submit) - but KEEP BROWSER OPEN
- Any error messages if submission failed

**Wait for user to confirm they're done before closing.**

## Error Handling

- If channel code not found in channels.py, warn the user
- If avatar or audio files don't exist, report the error
- If form doesn't load, try refreshing and wait longer
- If submission fails, show screenshot of error and suggest checking manually

## Playwright MCP Tool Reference

| Tool | Purpose |
|------|---------|
| `browser_navigate` | Go to form URL |
| `browser_snapshot` | Get current page state |
| `browser_fill_form` | Fill text fields and dropdowns |
| `browser_file_upload` | Upload files to file inputs |
| `browser_click` | Click buttons |
| `browser_take_screenshot` | Capture current view |
| `browser_wait_for` | Wait for text/element |
| `browser_evaluate` | Run JS if needed |

## Example Session

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
