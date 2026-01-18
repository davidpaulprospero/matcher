---
name: ollama-check
description: Diagnose Ollama setup, check models, start server with correct path. Use for keyword mode script generation issues.
allowed-tools:
  - Bash
  - Read
  - Edit
---

# Ollama Check & Setup

Diagnoses Ollama installation, verifies models are accessible, and starts the server with the correct models path.

## Quick Reference

| Check | Command |
|-------|---------|
| Ollama installed | `where ollama` or `which ollama` |
| Server running | `curl -s http://localhost:11434/api/tags` |
| List models | `OLLAMA_MODELS="D:/ollama/models" ollama list` |
| Test generation | `OLLAMA_MODELS="D:/ollama/models" ollama run mistral:7b "Hello"` |
| Start server | `OLLAMA_MODELS="D:/ollama/models" ollama serve` |

## Instructions

When this skill is invoked:

### 1. Check Ollama Installation

```bash
# Check if ollama is in PATH
where ollama 2>/dev/null || which ollama 2>/dev/null || echo "NOT_FOUND"
```

If not found:
```
## Ollama Check

❌ **Ollama not installed**

Install from: https://ollama.com/download
Or: `winget install Ollama.Ollama`
```

### 2. Check if Server is Running

```bash
curl -s http://localhost:11434/api/tags 2>/dev/null || echo "NOT_RUNNING"
```

### 3. Check Models Path

The models are stored in `D:\ollama\models`. The `OLLAMA_MODELS` environment variable must be set.

```bash
# Check env var (in cmd context)
cmd //c "echo %OLLAMA_MODELS%"

# Check if models directory exists
ls -la "D:/ollama/models/manifests/registry.ollama.ai/library/" 2>/dev/null
```

### 4. List Available Models

```bash
OLLAMA_MODELS="D:/ollama/models" ollama list
```

**Required models for keyword mode script generation:**

| Model | Size | Priority |
|-------|------|----------|
| `mistral:7b` | 4.4 GB | Primary (recommended) |
| `llama3.2:latest` | 2.0 GB | Fallback |
| `qwen2.5:7b` | 4.7 GB | Alternative |

### 5. Start Server if Not Running

If server is not running, start it with the correct models path:

```bash
# Kill any existing instances first
taskkill //IM "ollama app.exe" //F 2>/dev/null
taskkill //IM "ollama.exe" //F 2>/dev/null
sleep 2

# Start with correct models path
OLLAMA_MODELS="D:/ollama/models" ollama serve &
sleep 3

# Verify
OLLAMA_MODELS="D:/ollama/models" ollama list
```

### 6. Test Generation (Optional)

If user requests a test or if models are present:

```bash
OLLAMA_MODELS="D:/ollama/models" ollama run mistral:7b "Say hello in exactly 5 words." 2>/dev/null
```

Or with fallback:
```bash
OLLAMA_MODELS="D:/ollama/models" ollama run llama3.2:latest "Say hello in exactly 5 words." 2>/dev/null
```

### 7. Report Results

```
## Ollama Check

### Status
- Installation: ✓ Found at <path>
- Server: ✓ Running / ✗ Not running (started)
- Models path: D:/ollama/models

### Available Models
| Model | Size | Script Gen |
|-------|------|------------|
| mistral:7b | 4.4 GB | ✓ Primary |
| llama3.2:latest | 2.0 GB | ✓ Fallback |
| ... | ... | ... |

### Missing Recommended Models
To install: `OLLAMA_MODELS="D:/ollama/models" ollama pull mistral:7b`

### Test Result
✓ Generation working: "Hello there my friendly user!"
```

## Common Issues & Fixes

### Models Not Visible

**Cause:** `OLLAMA_MODELS` not set or pointing to wrong path.

**Fix:**
```bash
# Set permanently (requires new terminal)
setx OLLAMA_MODELS "D:/ollama/models"

# Or use prefix for current session
OLLAMA_MODELS="D:/ollama/models" ollama list
```

### Server Won't Start

**Cause:** Another instance running or port conflict.

**Fix:**
```bash
# Kill all Ollama processes
taskkill //IM "ollama app.exe" //F
taskkill //IM "ollama.exe" //F

# Wait and restart
sleep 2
OLLAMA_MODELS="D:/ollama/models" ollama serve
```

### Model Pull Fails

**Cause:** Network issues or disk space.

**Fix:**
```bash
# Check disk space
df -h D:/

# Retry pull
OLLAMA_MODELS="D:/ollama/models" ollama pull mistral:7b
```

### Generation Timeout/Slow

**Cause:** Model too large for available VRAM, or CPU-only mode.

**Fix:**
- Use smaller model: `llama3.2:latest` (2GB) instead of larger models
- Check GPU detection: Look for "CUDA" in ollama serve output

## Usage Examples

```
/ollama-check
/ollama-check --test
/ollama-check --start
```

## Models Path Reference

```
D:\ollama\models\
├── blobs\           # Model weights (large files)
└── manifests\       # Model metadata
    └── registry.ollama.ai\
        └── library\
            ├── mistral\
            ├── llama3.2\
            └── ...
```

## Related Files

- `start_ollama.bat` - Windows batch file to start Ollama with correct path
- `docs/plans/PLAN_KEYWORD_MODE.md` - Keyword mode implementation plan
