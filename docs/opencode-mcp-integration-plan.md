# OpenCode MCP Integration Plan

## Summary

Integrate 6 existing MCP servers from VS Code configuration into OpenCode's global config at `~/.config/opencode/opencode.json`.

## MCP Servers to Integrate

| MCP Server | Type | Command/URL | Auth Required |
|------------|------|-------------|--------------|
| `markitdown` | local | `["uvx", "markitdown-mcp==0.0.1a4"]` | No |
| `context7` | local | `["npx", "@upstash/context7-mcp@1.0.31"]` | Yes (CONTEXT7_API_KEY) |
| `github-mcp-server` | local | `["docker", "run", "-i", "--rm", "-e", "GITHUB_PERSONAL_ACCESS_TOKEN=...", "ghcr.io/github/github-mcp-server:0.29.0"]` | Yes (GITHUB_TOKEN) |
| `desktop-commander` | local | `["npx", "@wonderwhy-er/desktop-commander@0.2.29"]` | No |
| `apify-mcp-server` | remote | URL: `https://mcp.apify.com/` | Yes (Bearer token) |
| `remotion-documentation` | local | `["npx", "@remotion/mcp@latest"]` | No |

## OpenCode Config Format

The configuration should be saved to `C:/Users/daves/.config/opencode/opencode.json`:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "mcp": {
    "markitdown": {
      "type": "local",
      "command": ["uvx", "markitdown-mcp==0.0.1a4"],
      "enabled": true
    },
    "context7": {
      "type": "local",
      "command": ["npx", "-y", "@upstash/context7-mcp@1.0.31"],
      "enabled": true,
      "environment": {
        "CONTEXT7_API_KEY": "{env:CONTEXT7_API_KEY}"
      }
    },
    "github-mcp-server": {
      "type": "local",
      "command": ["docker", "run", "-i", "--rm", "-e", "GITHUB_PERSONAL_ACCESS_TOKEN={env:GITHUB_PERSONAL_ACCESS_TOKEN}", "ghcr.io/github/github-mcp-server:0.29.0"],
      "enabled": true
    },
    "desktop-commander": {
      "type": "local",
      "command": ["npx", "-y", "@wonderwhy-er/desktop-commander@0.2.29"],
      "enabled": true
    },
    "apify-mcp-server": {
      "type": "remote",
      "url": "https://mcp.apify.com/",
      "enabled": true,
      "headers": {
        "Authorization": "{env:APIFY_API_KEY}"
      },
      "oauth": false
    },
    "remotion-documentation": {
      "type": "local",
      "command": ["npx", "-y", "@remotion/mcp@latest"],
      "enabled": true
    }
  }
}
```

## Environment Variables Required

Set the following environment variables before using MCP servers:

- `CONTEXT7_API_KEY` - For Context7 documentation search
- `GITHUB_PERSONAL_ACCESS_TOKEN` - For GitHub MCP server
- `APIFY_API_KEY` - For Apify MCP server

## Prerequisites

- Docker must be running for `github-mcp-server`
- Node.js/npm for npx-based MCPs
- Python with uv for markitdown MCP

## Usage

After configuration, MCP tools are automatically available alongside built-in OpenCode tools. To use them in prompts:

```
Search the docs for Python asyncio using context7
List my GitHub repositories using github-mcp-server
Scrape a website using apify-mcp-server
```

## Source Configuration

These MCPs were originally configured in VS Code at:
`C:/Users/daves/AppData/Roaming/Code/User/mcp.json`

---

*Plan created: 2026-02-13*
