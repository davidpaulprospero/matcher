# Add MiniMax M2.5 Model to Ralph

**Date:** 2026-02-13
**Status:** Plan Document (pending implementation)

## Context

MiniMax M2.5 was released on Feb 12, 2026. It is a high-performance coding model with:
- **SWE-Bench Verified:** 80.2% (vs 74% on M2.1)
- **Context:** 204.8K tokens
- **Speed:** 37% faster than M2.1, matches Claude Opus 4.6
- **Pricing:** $0.30/M input, $1.20/M output (10% of Claude Sonnet 4.5)

It provides an Anthropic-compatible API that can be used with Claude Code CLI, offering a cost-effective alternative to Anthropic models.

Ralph currently supports Claude and Codex providers. Adding MiniMax support allows switching between providers for different cost/quality tradeoffs.

---

## Implementation Plan

### Phase 1: Configure MiniMax API (User Setup)

**File:** `~/.claude/settings.json`

```json
{
  "env": {
    "ANTHROPIC_BASE_URL": "https://api.minimax.io/anthropic",
    "ANTHROPIC_AUTH_TOKEN": "YOUR_MINIMAX_API_KEY",
    "API_TIMEOUT_MS": "3000000",
    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
    "ANTHROPIC_MODEL": "MiniMax-M2.5",
    "ANTHROPIC_SMALL_FAST_MODEL": "MiniMax-M2.5",
    "ANTHROPIC_DEFAULT_SONNET_MODEL": "MiniMax-M2.5",
    "ANTHROPIC_DEFAULT_OPUS_MODEL": "MiniMax-M2.5",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL": "MiniMax-M2.5"
  }
}
```

**Steps for user:**
1. Get API key from https://platform.minimax.io
2. Update `~/.claude/settings.json` with the configuration

---

### Phase 2: Add MiniMax Model to Ralph Config

**File:** `scripts/ralph/config/ralph-config.json`

Add MiniMax model to existing Claude provider models:

```json
{
  "agent": {
    "providers": {
      "claude": {
        "executable": "claude",
        "models": {
          "high": "MiniMax-M2.5",
          "fast": "MiniMax-M2.5", 
          "default": "MiniMax-M2.5"
        }
      }
    }
  }
}
```

**Or** create a dedicated minimax provider:

```json
{
  "agent": {
    "provider": "minimax",
    "providers": {
      "minimax": {
        "executable": "claude",
        "models": {
          "high": "MiniMax-M2.5",
          "fast": "MiniMax-M2.5",
          "default": "MiniMax-M2.5"
        }
      }
    }
  }
}
```

---

### Phase 3: Test the Integration

1. Set `ANTHROPIC_AUTH_TOKEN` environment variable with MiniMax API key
2. Run Ralph with `--model` flag or update config to use MiniMax model
3. Verify the model is being called by checking logs

---

## Alternative: Add as New Provider (Recommended)

For cleaner separation, add a dedicated `minimax` provider:

### Benefits
- Clear distinction between Anthropic and MiniMax backends
- Easy to switch between providers in config
- Future-proof for potential MiniMax-specific optimizations

### Changes Required

1. **No new provider file needed** - Uses existing `claude` executable with different model
2. **Update config** - Add minimax provider entry
3. **Update agent.ps1** - Add minimax to provider switch (optional, falls back to claude)

---

## Files to Modify

| File | Change |
|------|--------|
| `~/.claude/settings.json` | Add MiniMax API configuration (user setup) |
| `scripts/ralph/config/ralph-config.json` | Add minimax model mappings |
| `scripts/ralph/lib/agent.ps1` | (Optional) Add minimax case for explicit handling |

---

## Rollback Plan

If MiniMax doesn't work well:
1. Revert `ralph-config.json` to use `claude` provider with opus/sonnet/haiku
2. Restore original `~/.claude/settings.json` (remove MiniMax env vars)

---

## References

- [MiniMax M2.5 Announcement](https://www.minimax.io/news/minimax-m25)
- [MiniMax M2.5 on OpenRouter](https://openrouter.ai/minimax/minimax-m2.5)
- [MiniMax API Docs](https://platform.minimax.io/docs/guides/models-intro)

---

## Decision Made

| Option | Decision |
|--------|----------|
| Model version | MiniMax-M2.5 (latest, 80.2% SWE-bench) |
| Provider approach | Add to existing `claude` provider |
| Tier mapping | All tiers use MiniMax-M2.5 |
