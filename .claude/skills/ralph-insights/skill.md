---
name: ralph-insights
description: Analyze Claude Code sessions split by Ralph vs interactive. Shows true success rates, sprint velocity, and usage patterns.
allowed-tools:
  - Bash(python*)
  - Read
---

# Ralph-Aware Session Insights

Runs `scripts/insights/analyze_sessions.py` to classify all Claude Code sessions as Ralph (autonomous) or Interactive (human), then generates a split analytics report.

## Usage

```
/ralph-insights
```

## What It Does

1. Scans all session JSONL files in `~/.claude/projects/` for this project
2. Classifies each session:
   - **Ralph**: First user message contains "Ralph Agent Instructions"
   - **Interactive**: Everything else
3. Cross-references with Ralph's own metrics (metrics.csv, prd.json, sprint_history.json)
4. Generates a split report with accurate stats for each category

## Steps

1. Run the analysis script:

```bash
python scripts/insights/analyze_sessions.py
```

2. Present the full output to the user as-is — it's already formatted as markdown.

3. If the user asks follow-up questions, the script output contains all the data needed. For deeper dives into specific sessions, read individual session files from `~/.claude/projects/D---Projects-voiceover-matcher-subtitle/<session-id>.jsonl`.
