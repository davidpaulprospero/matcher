---
name: research
description: Research a topic using Perplexity AI. Supports general research, API documentation lookup, and comparisons.
allowed-tools:
  - Read
  - Bash(python:*)
  - Write
---

# Research with Perplexity, Gemini, and Grok

Performs real-time web research using multiple AI backends:

| Backend | Model | Search | Best For |
|---------|-------|--------|----------|
| **Perplexity** | sonar-pro | Web citations | General research, documentation |
| **Gemini** | gemini-2.0-flash | Google Search | Technical docs, comparisons |
| **Grok** | grok-3-latest | X/Twitter + Web | Current events, real-time info |

## Requirements

At least one API key must be set in `.env`:

| Backend | Environment Variable | Get Key |
|---------|---------------------|---------|
| Perplexity | `PERPLEXITY_API_KEY` | https://www.perplexity.ai/settings/api |
| Gemini | `GEMINI_API_KEY` | https://aistudio.google.com/app/apikey |
| Grok | `XAI_API_KEY` | https://console.x.ai |

## Research Modes

| Mode | Syntax | Description |
|------|--------|-------------|
| General | `/research <topic>` | Comprehensive research (default: perplexity) |
| Quick | `/research --quick <topic>` | Fast 500-800 word summary |
| API Docs | `/research --api <api_name>` | API documentation and code examples |
| Compare | `/research --compare <item1>, <item2>, ...` | Compare technologies/approaches |
| Specific backend | `/research --backend gemini <topic>` | Use specific backend |

## Instructions

When this skill is invoked:

### 1. Parse the request

Extract the research query, mode, and backend from the user's request:

**General research (auto-selects available backend):**
```
/research DaVinci Resolve Python API marker limitations
/research PySceneDetect vs FFmpeg scene detection accuracy
```

**Specific backend:**
```
/research --backend gemini WhisperX speaker diarization
/research --backend grok latest AI video generation news
/research --backend perplexity DaVinci Resolve scripting
```

**Quick research:**
```
/research --quick WhisperX vs faster-whisper
```

**API documentation:**
```
/research --api DaVinci Resolve Timeline API
/research --api Perplexity Sonar models
```

**Comparison:**
```
/research --compare Whisper, faster-whisper, WhisperX
/research --compare Shotstack, Creatomate, Remotion
```

### 2. Check available backends

```bash
cd "D:/_Projects/voiceover-matcher-subtitle" && python -m src.research.cli --list-backends
```

This shows which backends have API keys configured.

### 3. Run the research

Invoke the Bash tool to run the research CLI.

**General research (auto backend):**
```bash
cd "D:/_Projects/voiceover-matcher-subtitle" && python -m src.research.cli "<query>"
```

**Specific backend:**
```bash
cd "D:/_Projects/voiceover-matcher-subtitle" && python -m src.research.cli "<query>" --backend gemini
```

**Quick mode:**
```bash
cd "D:/_Projects/voiceover-matcher-subtitle" && python -m src.research.cli "<query>" --quick
```

**API documentation:**
```bash
cd "D:/_Projects/voiceover-matcher-subtitle" && python -m src.research.cli "<api_name>" --api
```

**Comparison:**
```bash
cd "D:/_Projects/voiceover-matcher-subtitle" && python -m src.research.cli "<item1>, <item2>, <item3>" --compare
```

### 4. Save output (optional)

If the user wants to save the research:
```bash
cd "D:/_Projects/voiceover-matcher-subtitle" && python -m src.research.cli "<query>" -o "docs/research/<filename>.md"
```

### 5. Format and present results

Present the research output to the user. The output includes:
- Research content organized by sections
- Source citations from the backend
- Backend and model used

## Usage Examples

```bash
# Auto-select backend (uses first available: perplexity > gemini > grok)
/research DaVinci Resolve Fusion template automation limitations

# Use Gemini for Google Search grounding
/research --backend gemini PySceneDetect AdaptiveDetector accuracy benchmarks

# Use Grok for real-time X/Twitter insights
/research --backend grok latest developments in AI video generation

# Quick summary
/research --quick best Python video editing libraries 2026

# API documentation
/research --api Frame.io webhooks

# Comparison
/research --compare PySceneDetect, FFmpeg scene detection, DaVinci built-in

# Save to file
/research --api AppendToTimeline DaVinci -o docs/research/davinci_timeline_api.md
```

## Troubleshooting

**"No API keys configured"**
- Add at least one API key to `.env`:
  - `PERPLEXITY_API_KEY=pplx-xxxxx`
  - `GEMINI_API_KEY=AIza...`
  - `XAI_API_KEY=xai-...`

**Timeout errors**
- Research can take up to 15 minutes for comprehensive queries
- Try `--quick` for faster results

**Rate limits (429)**
- Wait a few minutes and retry
- Try a different backend

**"google-genai package not installed"**
- Run: `pip install google-genai`

## Integration with Roadmap

Use this skill to research topics for the DaVinci integration roadmap:

```
/research --api DaVinci Resolve SetRenderSettings parameters
/research --backend gemini WhisperX vs AssemblyAI for speaker diarization
/research PySceneDetect AdaptiveDetector vs ContentDetector accuracy
```

Then integrate findings into `DAVINCI_INTEGRATION_ROADMAP.md`.
