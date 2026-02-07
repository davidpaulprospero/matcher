#!/usr/bin/env python3
"""
Session analytics for Claude Code usage — splits Ralph vs Interactive sessions.

Reads:
  - ~/.claude/projects/<project>/*.jsonl  (session conversations)
  - ~/.claude/history.jsonl               (per-prompt history)
  - ~/.claude/stats-cache.json            (daily aggregates)
  - ~/.claude/telemetry/*.json            (tengu_init print flag)
  - scripts/ralph/session/metrics.csv     (Ralph per-story metrics)
  - scripts/ralph/state/prd.json          (current sprint state)
  - scripts/ralph/state/sprint_history.json (sprint velocity)

Classification:
  1. Session JSONL first user message contains "Ralph Agent Instructions" → Ralph
  2. Telemetry tengu_init print=true → Ralph
  3. history.jsonl prompt matches US-XX-XXX or prd.json references → Ralph
  4. Everything else → Interactive
"""

import csv
import io
import json
import os
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

# Windows non-locking file read to avoid blocking Ralph's writes
if sys.platform == "win32":
    import msvcrt

    def _read_file_nonlocking(path: Path, encoding: str = "utf-8") -> str:
        """Read file without holding an exclusive lock (Windows).
        Uses FILE_SHARE_READ | FILE_SHARE_WRITE so other processes can write."""
        fd = os.open(str(path), os.O_RDONLY | os.O_BINARY)
        try:
            msvcrt.setmode(fd, os.O_BINARY)
            data = b""
            while True:
                chunk = os.read(fd, 65536)
                if not chunk:
                    break
                data += chunk
        finally:
            os.close(fd)
        # Handle BOM
        if data.startswith(b"\xef\xbb\xbf"):
            data = data[3:]
        return data.decode(encoding, errors="replace")
else:
    def _read_file_nonlocking(path: Path, encoding: str = "utf-8") -> str:
        """On non-Windows, regular read is fine."""
        with open(path, "r", encoding=encoding, errors="replace") as f:
            return f.read()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_datetime(s: str) -> datetime:
    """Parse ISO datetime, handling Windows-style fractional seconds (7 digits)."""
    # Python 3.9 only supports up to 6 fractional digits
    s = re.sub(r"(\.\d{6})\d+", r"\1", s)
    return datetime.fromisoformat(s)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
CLAUDE_DIR = Path.home() / ".claude"
HISTORY_PATH = CLAUDE_DIR / "history.jsonl"
STATS_CACHE_PATH = CLAUDE_DIR / "stats-cache.json"
TELEMETRY_DIR = CLAUDE_DIR / "telemetry"
FACETS_DIR = CLAUDE_DIR / "usage-data" / "facets"

# Project-specific sessions
PROJECT_KEY = "D---Projects-voiceover-matcher-subtitle"
SESSIONS_DIR = CLAUDE_DIR / "projects" / PROJECT_KEY

# Ralph data (relative to repo root)
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RALPH_METRICS_CSV = REPO_ROOT / "scripts" / "ralph" / "session" / "metrics.csv"
RALPH_PRD_JSON = REPO_ROOT / "scripts" / "ralph" / "state" / "prd.json"
RALPH_SPRINT_HISTORY = REPO_ROOT / "scripts" / "ralph" / "state" / "sprint_history.json"

# Classification patterns
RALPH_PROMPT_MARKERS = [
    "Ralph Agent Instructions",
    "RETRY CONTEXT (attempt",
]
RALPH_BRANCH_RE = re.compile(r"ralph/sprint-\d+")
STORY_ID_RE = re.compile(r"US-\d{1,3}-\d{3}")
PRD_REF_RE = re.compile(r"prd\.json")


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def load_telemetry_print_flags() -> dict[str, bool]:
    """Parse telemetry dir for tengu_init events → {session_id: is_print_mode}."""
    flags: dict[str, bool] = {}
    if not TELEMETRY_DIR.exists():
        return flags
    for fp in TELEMETRY_DIR.glob("*.json"):
        try:
            with open(fp, "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    evt = json.loads(line)
                    ed = evt.get("event_data", {})
                    if ed.get("event_name") != "tengu_init":
                        continue
                    sid = ed.get("session_id")
                    meta_raw = ed.get("additional_metadata", "{}")
                    meta = json.loads(meta_raw) if isinstance(meta_raw, str) else meta_raw
                    flags[sid] = meta.get("print", False)
        except (json.JSONDecodeError, KeyError, TypeError):
            continue
    return flags


def classify_session_from_jsonl(session_path: Path) -> str | None:
    """Read first user message and branch from a session JSONL file.
    Returns 'ralph' if it matches Ralph markers or is on a ralph/sprint branch,
    'interactive' otherwise, None on error.
    """
    try:
        git_branch = None
        with open(session_path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue

                # Capture git branch from any entry
                if not git_branch and entry.get("gitBranch"):
                    git_branch = entry["gitBranch"]

                # Look for first user message
                if entry.get("type") != "user":
                    continue
                msg = entry.get("message", {})
                content = msg.get("content", "")
                if isinstance(content, list):
                    content = " ".join(
                        b.get("text", "") if isinstance(b, dict) else str(b)
                        for b in content
                    )
                # Check prompt markers
                for marker in RALPH_PROMPT_MARKERS:
                    if marker in content:
                        return "ralph"
                # Check if on a ralph branch (strong signal)
                if git_branch and RALPH_BRANCH_RE.match(git_branch):
                    return "ralph"
                return "interactive"

        # No user message found — check branch only
        if git_branch and RALPH_BRANCH_RE.match(git_branch):
            return "ralph"
    except (OSError, UnicodeDecodeError):
        pass
    return None


def load_session_metadata(session_path: Path) -> dict[str, Any] | None:
    """Extract lightweight metadata from a session JSONL without loading everything."""
    meta: dict[str, Any] = {
        "session_id": session_path.stem,
        "message_count": 0,
        "tool_calls": 0,
        "first_timestamp": None,
        "last_timestamp": None,
        "git_branch": None,
        "version": None,
    }
    try:
        with open(session_path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue

                ts = entry.get("timestamp")
                if ts and (meta["first_timestamp"] is None or ts < meta["first_timestamp"]):
                    meta["first_timestamp"] = ts
                if ts and (meta["last_timestamp"] is None or ts > meta["last_timestamp"]):
                    meta["last_timestamp"] = ts

                entry_type = entry.get("type")
                if entry_type in ("user", "assistant"):
                    meta["message_count"] += 1
                    # Count tool_use blocks within assistant messages
                    if entry_type == "assistant":
                        msg = entry.get("message", {})
                        content = msg.get("content", [])
                        if isinstance(content, list):
                            meta["tool_calls"] += sum(
                                1 for b in content
                                if isinstance(b, dict) and b.get("type") == "tool_use"
                            )

                if not meta["git_branch"] and entry.get("gitBranch"):
                    meta["git_branch"] = entry["gitBranch"]
                if not meta["version"] and entry.get("version"):
                    meta["version"] = entry["version"]
    except (OSError, UnicodeDecodeError):
        return None
    return meta


def load_history() -> list[dict]:
    """Load history.jsonl entries."""
    entries = []
    if not HISTORY_PATH.exists():
        return entries
    with open(HISTORY_PATH, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return entries


def load_stats_cache() -> dict:
    """Load stats-cache.json."""
    if not STATS_CACHE_PATH.exists():
        return {}
    with open(STATS_CACHE_PATH, "r", encoding="utf-8", errors="replace") as f:
        return json.load(f)


def load_ralph_metrics() -> list[dict]:
    """Load Ralph session/metrics.csv using non-locking read."""
    rows = []
    if not RALPH_METRICS_CSV.exists():
        return rows
    content = _read_file_nonlocking(RALPH_METRICS_CSV)
    reader = csv.DictReader(io.StringIO(content))
    for row in reader:
        rows.append(row)
    return rows


def load_ralph_prd() -> dict:
    """Load current prd.json using non-locking read."""
    if not RALPH_PRD_JSON.exists():
        return {}
    content = _read_file_nonlocking(RALPH_PRD_JSON)
    return json.loads(content)


def load_sprint_history() -> dict:
    """Load sprint_history.json using non-locking read."""
    if not RALPH_SPRINT_HISTORY.exists():
        return {}
    content = _read_file_nonlocking(RALPH_SPRINT_HISTORY)
    return json.loads(content)


def load_facets() -> dict[str, dict]:
    """Load all facet files → {session_id: facet_data}."""
    facets = {}
    if not FACETS_DIR.exists():
        return facets
    for fp in FACETS_DIR.glob("*.json"):
        try:
            with open(fp, "r", encoding="utf-8", errors="replace") as f:
                data = json.load(f)
                sid = data.get("session_id", fp.stem)
                facets[sid] = data
        except (json.JSONDecodeError, OSError):
            continue
    return facets


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def classify_all_sessions(
    print_flags: dict[str, bool],
    history: list[dict],
) -> tuple[list[str], list[str]]:
    """Classify all sessions in the project directory.
    Returns (ralph_session_ids, interactive_session_ids).
    """
    ralph_ids: list[str] = []
    interactive_ids: list[str] = []

    # Build history-based hints: session_id → has ralph indicators
    history_hints: dict[str, bool] = {}
    for entry in history:
        sid = entry.get("sessionId", "")
        if not sid:
            continue
        display = entry.get("display", "")
        if STORY_ID_RE.search(display) or PRD_REF_RE.search(display):
            history_hints[sid] = True

    if not SESSIONS_DIR.exists():
        return ralph_ids, interactive_ids

    session_files = sorted(SESSIONS_DIR.glob("*.jsonl"))
    total = len(session_files)

    for i, fp in enumerate(session_files):
        sid = fp.stem
        if (i + 1) % 200 == 0:
            print(f"  Classifying {i + 1}/{total}...", file=sys.stderr)

        # Tier 1: Session JSONL content (most reliable)
        classification = classify_session_from_jsonl(fp)
        if classification:
            if classification == "ralph":
                ralph_ids.append(sid)
            else:
                interactive_ids.append(sid)
            continue

        # Tier 2: Telemetry print flag
        if sid in print_flags:
            if print_flags[sid]:
                ralph_ids.append(sid)
            else:
                interactive_ids.append(sid)
            continue

        # Tier 3: History hints
        if history_hints.get(sid, False):
            ralph_ids.append(sid)
        else:
            interactive_ids.append(sid)

    return ralph_ids, interactive_ids


# ---------------------------------------------------------------------------
# Analytics
# ---------------------------------------------------------------------------

def compute_session_stats(
    session_ids: list[str],
    label: str,
) -> dict[str, Any]:
    """Compute aggregate stats for a set of sessions."""
    total_messages = 0
    total_tools = 0
    durations: list[float] = []  # seconds
    branches: Counter = Counter()
    versions: Counter = Counter()

    sample_size = min(len(session_ids), 500)  # Cap for performance
    sampled = session_ids[:sample_size]

    for sid in sampled:
        fp = SESSIONS_DIR / f"{sid}.jsonl"
        if not fp.exists():
            continue
        meta = load_session_metadata(fp)
        if not meta:
            continue
        total_messages += meta["message_count"]
        total_tools += meta["tool_calls"]
        if meta["git_branch"]:
            branches[meta["git_branch"]] += 1
        if meta["version"]:
            versions[meta["version"]] += 1
        if meta["first_timestamp"] and meta["last_timestamp"]:
            try:
                t0 = parse_datetime(meta["first_timestamp"].replace("Z", "+00:00"))
                t1 = parse_datetime(meta["last_timestamp"].replace("Z", "+00:00"))
                dur = (t1 - t0).total_seconds()
                if dur >= 0:
                    durations.append(dur)
            except (ValueError, TypeError):
                pass

    scale = len(session_ids) / sample_size if sample_size > 0 else 1

    return {
        "label": label,
        "total_sessions": len(session_ids),
        "sampled": sample_size,
        "est_total_messages": int(total_messages * scale),
        "est_total_tool_calls": int(total_tools * scale),
        "avg_messages_per_session": round(total_messages / sample_size, 1) if sample_size else 0,
        "avg_tools_per_session": round(total_tools / sample_size, 1) if sample_size else 0,
        "avg_duration_min": round(sum(durations) / len(durations) / 60, 1) if durations else 0,
        "median_duration_min": round(sorted(durations)[len(durations) // 2] / 60, 1) if durations else 0,
        "top_branches": branches.most_common(5),
        "top_versions": versions.most_common(3),
    }


def compute_ralph_analytics(
    metrics: list[dict],
    prd: dict,
    sprint_hist: dict,
) -> dict[str, Any]:
    """Compute Ralph-specific analytics from Ralph's own data."""
    # Metrics analysis
    total_stories = len(metrics)
    successes = sum(1 for r in metrics if r.get("success", "").lower() == "true")
    timeouts = sum(1 for r in metrics if r.get("timeout", "").lower() == "true")
    focus_counts: Counter = Counter()
    durations: list[float] = []

    for r in metrics:
        fa = r.get("focus_area", "unknown")
        if fa:
            focus_counts[fa] += 1
        try:
            dur = float(r.get("duration_min", 0))
            if dur > 0:
                durations.append(dur)
        except (ValueError, TypeError):
            pass

    # Current sprint from prd.json
    current_sprint = prd.get("sprintNumber", "?")
    current_focus = prd.get("focusArea", "?")
    stories = prd.get("userStories", [])
    current_passed = sum(1 for s in stories if s.get("passes"))
    current_total = len(stories)

    # Sprint history
    hist_sprints = sprint_hist.get("sprints", [])
    total_sprints_completed = sprint_hist.get("totalSprintsCompleted", 0)
    total_stories_completed = sprint_hist.get("totalStoriesCompleted", 0)
    focus_breakdown = sprint_hist.get("focusAreaBreakdown", {})

    # Sprint velocity (stories/sprint)
    velocities: list[float] = []
    sprint_durations: list[float] = []
    for sp in hist_sprints:
        comp = sp.get("storiesCompleted", 0)
        tot = sp.get("storiesTotal", 0)
        velocities.append(comp)
        try:
            t0 = parse_datetime(sp["startedAt"])
            t1 = parse_datetime(sp["completedAt"])
            sprint_durations.append((t1 - t0).total_seconds() / 3600)
        except (ValueError, KeyError, TypeError):
            pass

    return {
        "metrics_total_stories": total_stories,
        "metrics_success_count": successes,
        "metrics_success_rate": round(successes / total_stories * 100, 1) if total_stories else 0,
        "metrics_timeout_count": timeouts,
        "metrics_timeout_rate": round(timeouts / total_stories * 100, 1) if total_stories else 0,
        "metrics_avg_duration_min": round(sum(durations) / len(durations), 1) if durations else 0,
        "metrics_median_duration_min": round(sorted(durations)[len(durations) // 2], 1) if durations else 0,
        "metrics_focus_area_dist": focus_counts.most_common(),
        "current_sprint": current_sprint,
        "current_focus": current_focus,
        "current_stories_passed": current_passed,
        "current_stories_total": current_total,
        "total_sprints_completed": total_sprints_completed,
        "total_stories_completed": total_stories_completed,
        "focus_area_breakdown": focus_breakdown,
        "avg_velocity": round(sum(velocities) / len(velocities), 1) if velocities else 0,
        "avg_sprint_hours": round(sum(sprint_durations) / len(sprint_durations), 1) if sprint_durations else 0,
    }


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def format_report(
    ralph_stats: dict,
    interactive_stats: dict,
    ralph_analytics: dict,
    stats_cache: dict,
    facets: dict[str, dict],
) -> str:
    """Generate the final markdown report."""
    total = ralph_stats["total_sessions"] + interactive_stats["total_sessions"]
    ralph_pct = round(ralph_stats["total_sessions"] / total * 100, 1) if total else 0
    interactive_pct = round(100 - ralph_pct, 1)

    # Aggregate stats from stats-cache
    agg_messages = stats_cache.get("totalMessages", 0)
    agg_sessions = stats_cache.get("totalSessions", 0)
    first_date = stats_cache.get("firstSessionDate", "unknown")

    lines = []
    lines.append("# Claude Code Session Insights (Ralph-Aware)")
    lines.append("")
    lines.append("## Session Breakdown")
    lines.append("")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Total sessions (this project) | {total:,} |")
    lines.append(f"| Ralph (autonomous) | {ralph_stats['total_sessions']:,} ({ralph_pct}%) |")
    lines.append(f"| Interactive (you) | {interactive_stats['total_sessions']:,} ({interactive_pct}%) |")
    lines.append(f"| Total messages (all projects) | {agg_messages:,} |")
    lines.append(f"| Total sessions (all projects) | {agg_sessions:,} |")
    lines.append(f"| First session | {first_date[:10] if first_date != 'unknown' else 'unknown'} |")
    lines.append("")

    # ── Interactive ──
    lines.append("## Interactive Sessions (Your Actual Usage)")
    lines.append("")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Sessions | {interactive_stats['total_sessions']:,} |")
    lines.append(f"| Est. total messages | {interactive_stats['est_total_messages']:,} |")
    lines.append(f"| Est. total tool calls | {interactive_stats['est_total_tool_calls']:,} |")
    lines.append(f"| Avg messages/session | {interactive_stats['avg_messages_per_session']} |")
    lines.append(f"| Avg tool calls/session | {interactive_stats['avg_tools_per_session']} |")
    lines.append(f"| Avg duration | {interactive_stats['avg_duration_min']} min |")
    lines.append(f"| Median duration | {interactive_stats['median_duration_min']} min |")
    lines.append("")

    if interactive_stats["top_branches"]:
        lines.append("**Top branches:**")
        for branch, count in interactive_stats["top_branches"]:
            lines.append(f"- `{branch}` ({count})")
        lines.append("")

    # Facet analysis (only for interactive sessions if available)
    if facets:
        outcomes = Counter()
        helpfulness = Counter()
        goals = Counter()
        for sid, fdata in facets.items():
            outcomes[fdata.get("outcome", "unknown")] += 1
            helpfulness[fdata.get("claude_helpfulness", "unknown")] += 1
            for cat, val in fdata.get("goal_categories", {}).items():
                if val:
                    goals[cat] += val
        lines.append("**Facet analysis** (from Claude's self-assessment):")
        if outcomes:
            lines.append(f"- Outcomes: {dict(outcomes)}")
        if helpfulness:
            lines.append(f"- Helpfulness: {dict(helpfulness)}")
        if goals:
            lines.append(f"- Goal categories: {dict(goals.most_common(5))}")
        lines.append("")

    # ── Ralph ──
    lines.append("## Ralph Sessions (Autonomous Development)")
    lines.append("")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Sessions (Claude Code) | {ralph_stats['total_sessions']:,} |")
    lines.append(f"| Est. total messages | {ralph_stats['est_total_messages']:,} |")
    lines.append(f"| Est. total tool calls | {ralph_stats['est_total_tool_calls']:,} |")
    lines.append(f"| Avg messages/session | {ralph_stats['avg_messages_per_session']} |")
    lines.append(f"| Avg tool calls/session | {ralph_stats['avg_tools_per_session']} |")
    lines.append(f"| Avg duration | {ralph_stats['avg_duration_min']} min |")
    lines.append(f"| Median duration | {ralph_stats['median_duration_min']} min |")
    lines.append("")

    # Ralph's own metrics (ground truth)
    ra = ralph_analytics
    lines.append("### Story Execution (from Ralph metrics.csv)")
    lines.append("")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Total stories attempted | {ra['metrics_total_stories']:,} |")
    lines.append(f"| Stories passed | {ra['metrics_success_count']:,} ({ra['metrics_success_rate']}%) |")
    lines.append(f"| Stories timed out | {ra['metrics_timeout_count']:,} ({ra['metrics_timeout_rate']}%) |")
    lines.append(f"| Avg story duration | {ra['metrics_avg_duration_min']} min |")
    lines.append(f"| Median story duration | {ra['metrics_median_duration_min']} min |")
    lines.append("")

    lines.append("### Sprint Velocity (from sprint_history.json)")
    lines.append("")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Sprints completed | {ra['total_sprints_completed']} |")
    lines.append(f"| Total stories completed | {ra['total_stories_completed']:,} |")
    lines.append(f"| Avg stories/sprint | {ra['avg_velocity']} |")
    lines.append(f"| Avg sprint duration | {ra['avg_sprint_hours']} hrs |")
    lines.append("")

    lines.append(f"### Current Sprint #{ra['current_sprint']} — {ra['current_focus']}")
    lines.append("")
    lines.append(f"- Stories: {ra['current_stories_passed']}/{ra['current_stories_total']} passed")
    lines.append("")

    # Focus area breakdown
    lines.append("### Focus Area Breakdown")
    lines.append("")
    lines.append("| Focus Area | Sprints | Stories |")
    lines.append("|------------|---------|---------|")
    fab = ra["focus_area_breakdown"]
    for area in sorted(fab.keys(), key=lambda a: fab[a].get("stories", 0), reverse=True):
        info = fab[area]
        lines.append(f"| {area} | {info.get('sprints', 0)} | {info.get('stories', 0)} |")
    lines.append("")

    # Story-level focus distribution from metrics
    lines.append("### Story Attempts by Focus Area (metrics.csv)")
    lines.append("")
    lines.append("| Focus Area | Attempts |")
    lines.append("|------------|----------|")
    for area, count in ra["metrics_focus_area_dist"]:
        lines.append(f"| {area} | {count} |")
    lines.append("")

    # ── Combined ──
    lines.append("## Combined Insights")
    lines.append("")

    # Daily activity from stats-cache
    daily = stats_cache.get("dailyActivity", [])
    if daily:
        lines.append("### Daily Activity (all projects)")
        lines.append("")
        lines.append("| Date | Messages | Sessions | Tool Calls |")
        lines.append("|------|----------|----------|------------|")
        # Show last 10 days
        for d in daily[-10:]:
            lines.append(
                f"| {d['date']} | {d['messageCount']:,} | {d['sessionCount']} | {d['toolCallCount']:,} |"
            )
        lines.append("")

    # Model usage
    model_usage = stats_cache.get("modelUsage", {})
    if model_usage:
        lines.append("### Model Usage (all-time)")
        lines.append("")
        lines.append("| Model | Cost (USD) | Input Tokens | Output Tokens |")
        lines.append("|-------|-----------|-------------|--------------|")
        for model, usage in sorted(model_usage.items(), key=lambda x: x[1].get("costUSD", 0), reverse=True):
            short_name = model.split("-20")[0]  # Trim date suffix
            cost = usage.get("costUSD", 0)
            inp = usage.get("inputTokens", 0)
            out = usage.get("outputTokens", 0)
            lines.append(f"| {short_name} | ${cost:,.2f} | {inp:,} | {out:,} |")
        lines.append("")

    lines.append("---")
    lines.append(f"*Generated {datetime.now().strftime('%Y-%m-%d %H:%M')} by `scripts/insights/analyze_sessions.py`*")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Loading data sources...", file=sys.stderr)

    print("  Telemetry...", file=sys.stderr)
    print_flags = load_telemetry_print_flags()
    print(f"  → {len(print_flags)} telemetry sessions", file=sys.stderr)

    print("  History...", file=sys.stderr)
    history = load_history()
    print(f"  → {len(history)} history entries", file=sys.stderr)

    print("  Stats cache...", file=sys.stderr)
    stats_cache = load_stats_cache()

    print("  Facets...", file=sys.stderr)
    facets = load_facets()
    print(f"  → {len(facets)} facet files", file=sys.stderr)

    print("  Ralph metrics...", file=sys.stderr)
    ralph_metrics = load_ralph_metrics()
    print(f"  → {len(ralph_metrics)} story records", file=sys.stderr)

    print("  Ralph PRD...", file=sys.stderr)
    ralph_prd = load_ralph_prd()

    print("  Sprint history...", file=sys.stderr)
    sprint_hist = load_sprint_history()

    print("Classifying sessions...", file=sys.stderr)
    ralph_ids, interactive_ids = classify_all_sessions(print_flags, history)
    print(
        f"  → Ralph: {len(ralph_ids)}, Interactive: {len(interactive_ids)}",
        file=sys.stderr,
    )

    print("Computing stats (sampling up to 500 per category)...", file=sys.stderr)
    ralph_stats = compute_session_stats(ralph_ids, "Ralph")
    interactive_stats = compute_session_stats(interactive_ids, "Interactive")

    print("Computing Ralph analytics...", file=sys.stderr)
    ralph_analytics = compute_ralph_analytics(ralph_metrics, ralph_prd, sprint_hist)

    print("Generating report...", file=sys.stderr)
    report = format_report(
        ralph_stats, interactive_stats, ralph_analytics, stats_cache, facets
    )

    print(report)
    print("\nDone.", file=sys.stderr)


if __name__ == "__main__":
    main()
