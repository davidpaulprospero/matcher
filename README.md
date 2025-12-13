# Voiceover-Matcher v2.2

End-to-end documentary footage pipeline that matches voiceover narration to stock footage.

## Features

- **Keyword Extraction**: AI extracts search keywords from your voiceover (entity-aware: names, places, dates)
- **Video Downloading**: Automatically downloads relevant footage from YouTube
- **GPU-Accelerated**: CUDA support for transcription, transcoding, and embeddings
- **Smart Matching**: Semantic matching with 8 strategy tracks for editor choice
- **Multi-Project**: Central install with project-specific folders

## Quick Start

### 1. Install (One Time)

Extract to a central location like `C:\Tools\voiceover-matcher\`:

```cmd
cd C:\Tools\voiceover-matcher
install.bat
```

Edit `.env` and add your API keys:
```
GEMINI_API_KEY=your_gemini_key
ANTHROPIC_API_KEY=your_anthropic_key
```

### 2. Create a Project

```cmd
python setup_project.py "E:\Edit Job\Gold Media\Earth Unleashed-\JAPAN Earthquake"
```

This creates:
```
JAPAN Earthquake/
├── run.bat              ← Double-click to run
├── voiceover/           ← Put your .srt or .mp3 here
├── downloaded_videos/   ← Downloaded footage
├── otio_output/         ← Generated timelines
├── .cache/              ← Transcription cache
├── logs/                ← Run logs
└── project_config.yaml  ← Optional overrides
```

### 3. Run

Put your voiceover file in the `voiceover/` folder, then:

**Option A**: Double-click `run.bat`

**Option B**: Command line
```cmd
cd "E:\Edit Job\Gold Media\Earth Unleashed-\JAPAN Earthquake"
run.bat -v voiceover/script.srt
```

**Option C**: From central install
```cmd
cd C:\Tools\voiceover-matcher
python main.py --project "E:\Edit Job\...\JAPAN Earthquake" -v voiceover/script.srt
```

## Supported Voiceover Formats

The voiceover file can be:
- **SRT subtitle file** (`.srt`) - Used directly
- **Audio file** - Transcribed to SRT automatically
  - `.mp3`, `.wav`, `.m4a`, `.flac`, `.ogg`, `.wma`, `.aac`, `.opus`
- **Video file** - Audio extracted and transcribed
  - `.mp4`, `.mov`, `.avi`, `.mkv`, `.webm`, `.wmv`, `.flv`, `.m4v`

If an SRT file already exists next to the audio/video, it will be used instead of transcribing.

## Command Line Options

```
python main.py [OPTIONS]

Project Mode:
  --project, -p PATH     Project directory (outputs go here)

Input:
  --voiceover, -v FILE   Voiceover file (.srt, audio, or video)
  --keywords, -k NUM     Number of keywords to extract (default: 50)

Stage Control:
  --match-only           Skip download, match existing footage
  --download-only        Only download footage, stop before matching
  --skip-transcribe      Use cached transcriptions
  --skip-scenes          Skip scene detection
  --resume               Resume interrupted download

Other:
  --yes, -y              Skip confirmation prompts
  --config, -c FILE      Custom config file
```

## Examples

```cmd
# Full pipeline with voiceover
python main.py -p "E:\Projects\Tsunami" -v voiceover/script.srt -k 50

# Match only (already have footage)
python main.py -p "E:\Projects\Tsunami" --match-only -v voiceover/script.srt

# Download footage only
python main.py -p "E:\Projects\Tsunami" --download-only -v voiceover/script.srt

# Skip prompts (for automation)
python main.py -p "E:\Projects\Tsunami" -v script.srt -k 50 --yes
```

## Project Config Overrides

Edit `project_config.yaml` in your project folder to override global settings:

```yaml
# Use different embedding provider for this project
embedding:
  provider: "voyage"

# Lower confidence threshold
matching:
  confidence_threshold: 0.4

# More alternatives per segment
output:
  num_alternatives: 4
```

## Output

The pipeline generates:
- **OTIO Timeline**: Multi-track timeline for DaVinci Resolve
  - V1: Primary matches
  - V2-V3: Alternatives (different sources)
  - V4-V8: Strategy variations (visual-first, keyword, diversity, etc.)
- **Logs**: Detailed match decisions and reasoning

## GPU Acceleration

With NVIDIA GPU + CUDA:
- **Transcription**: 10-50x faster with faster-whisper
- **Transcoding**: NVENC hardware encoding (when needed)
- **Embeddings**: GPU-accelerated sentence transformers

Ensure `force_cuda: true` in config.yaml.

## Smart Transcoding

The downloader intelligently skips transcoding when possible:

| Format | Action |
|--------|--------|
| H.264/AVC (MP4) | ✅ Use directly - no transcode |
| H.265/HEVC (MP4) | ✅ Use directly - no transcode |
| ProRes (MOV) | ✅ Use directly - no transcode |
| VP9 (WebM) | 🔄 Transcode to H.264 |
| AV1 | 🔄 Transcode to H.264 |

yt-dlp is configured to prefer H.264 downloads when in DaVinci mode, reducing transcode time by ~70%.

## Troubleshooting

**"Voiceover file not found"**
- Use full path or path relative to project folder
- For project mode, paths are relative to project directory

**"API key not found"**  
- Edit `.env` in the install directory
- Required: `GEMINI_API_KEY` and `ANTHROPIC_API_KEY`

**Slow transcription**
- Install `faster-whisper` for 4x speedup
- Ensure CUDA is available: `python -c "import torch; print(torch.cuda.is_available())"`

## File Structure

```
C:\Tools\voiceover-matcher\        ← Central Install
├── main.py                        ← Main program
├── setup_project.py               ← Create new projects
├── install.bat                    ← Installation helper
├── config.yaml                    ← Global defaults
├── .env                           ← API keys (shared)
├── src/                           ← Source code
└── requirements.txt

E:\Projects\MyDocumentary\         ← Project Folder
├── run.bat                        ← Launch script
├── voiceover/                     ← Input files
├── downloaded_videos/             ← Downloaded footage
├── otio_output/                   ← Generated timelines
├── .cache/                        ← Local cache
├── logs/                          ← Run logs
└── project_config.yaml            ← Override settings
```

## Version History

### v2.2
- Project-based folder structure
- GPU acceleration for all FFmpeg operations
- Video/audio voiceover support (auto-transcription)
- Smart transcoding (skip H.264/H.265, only transcode VP9/AV1)
- yt-dlp prefers H.264 to minimize transcoding
- Entity-aware keyword extraction (names, places, dates)
- Different sources enforced across all tracks
- `--match-only` skips keyword extraction API calls

### v2.1
- Source rotation strategy track
- Scene detection with audio analysis
- Comprehensive logging

### v2.0
- FAISS indexing for fast search
- Multi-provider embeddings
- OTIO timeline output
