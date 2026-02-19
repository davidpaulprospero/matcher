#!/usr/bin/env python3
"""
Diagnostic Log Viewer

Tool for viewing and filtering pipeline logs. Supports both plain text (.log)
and JSON (.json) log formats from the logger.py system.

Usage:
    python scripts/diagnostic_viewer.py --logs-dir ./logs
    python scripts/diagnostic_viewer.py --logs-dir ./logs --level ERROR
    python scripts/diagnostic_viewer.py --logs-dir ./logs --level ERROR --category rate_limit
    python scripts/diagnostic_viewer.py --logs-dir ./logs --from 2026-02-17T10:00:00 --to 2026-02-17T12:00:00
    python scripts/diagnostic_viewer.py --logs-dir ./logs --json-only
    python scripts/diagnostic_viewer.py --logs-dir ./logs --recent 3
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

# Add project root and scripts directory to path for imports
_script_path = os.path.abspath(__file__)
project_root = Path(_script_path).parent.parent
scripts_dir = Path(_script_path).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(scripts_dir))

# Change to project root so relative paths work correctly
os.chdir(project_root)

# Try to import tqdm for progress bars (optional)
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    # Simple fallback tqdm that does nothing
    def tqdm(*args, **kwargs):
        return args[0] if args else iter([])

# Import standardized output functions
from script_utils import set_verbosity, get_verbosity


# Log level colors for terminal output
LEVEL_COLORS = {
    'DEBUG': '\033[36m',     # Cyan
    'INFO': '\033[32m',      # Green
    'WARNING': '\033[33m',   # Yellow
    'ERROR': '\033[31m',     # Red
    'CRITICAL': '\033[35m',  # Magenta
}
RESET = '\033[0m'


def parse_timestamp(ts_str: str) -> Optional[datetime]:
    """Parse various timestamp formats."""
    formats = [
        '%Y-%m-%d %H:%M:%S',
        '%Y-%m-%dT%H:%M:%S',
        '%Y-%m-%dT%H:%M:%S.%f',
        '%H:%M:%S',
    ]
    for fmt in formats:
        try:
            return datetime.strptime(ts_str, fmt)
        except ValueError:
            continue
    return None


def parse_log_line(line: str) -> Optional[Dict[str, Any]]:
    """Parse a single line from a plain text .log file."""
    # Pattern: 2026-02-17 10:30:45 | ERROR | message here
    pattern = r'^(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})\s*\|\s*(\w+)\s*\|\s*(.+)$'
    match = re.match(pattern, line.strip())
    if match:
        return {
            'timestamp': match.group(1),
            'level': match.group(2).upper(),
            'message': match.group(3).strip(),
            'source': 'log'
        }
    return None


def load_json_log(path: Path) -> List[Dict[str, Any]]:
    """Load JSON log file from logger.py system."""
    entries: List[Dict[str, Any]] = []
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data: Dict[str, Any] = json.load(f)

        # RunLogger JSON format
        if 'api_calls' in data or 'errors' in data or 'warnings' in data:
            # Convert to unified entry format
            run_id: str = data.get('run_id', 'unknown')
            start_time: str = data.get('start_time', '')

            # Add errors
            for err in data.get('errors', []):
                entries.append({
                    'timestamp': start_time,
                    'level': 'ERROR',
                    'message': str(err),
                    'source': 'run_log',
                    'run_id': run_id
                })

            # Add warnings
            for warn in data.get('warnings', []):
                entries.append({
                    'timestamp': start_time,
                    'level': 'WARNING',
                    'message': str(warn),
                    'source': 'run_log',
                    'run_id': run_id
                })

            # Add API calls (errors only)
            for call in data.get('api_calls', []):
                call_dict: Dict[str, Any] = call
                if not call_dict.get('success', True):
                    entries.append({
                        'timestamp': call_dict.get('timestamp', start_time),
                        'level': 'ERROR',
                        'message': f"API {call_dict.get('provider')}/{call_dict.get('endpoint')}: {call_dict.get('error', 'Unknown error')}",
                        'source': 'api_call',
                        'run_id': run_id
                    })

            # Add stage timings
            stage_timings: Dict[str, Any] = data.get('stage_timings', {})
            for stage, duration in stage_timings.items():
                entries.append({
                    'timestamp': start_time,
                    'level': 'INFO',
                    'message': f"Stage {stage}: {duration:.1f}s",
                    'source': 'stage_timing',
                    'run_id': run_id
                })

            # Add summary info
            if data.get('summary'):
                summary: Dict[str, Any] = data['summary']
                entries.append({
                    'timestamp': start_time,
                    'level': 'INFO',
                    'message': f"Run completed: {summary.get('total_matches', 0)}/{summary.get('total_segments', 0)} matches, "
                               f"{summary.get('total_api_calls', 0)} API calls, "
                               f"${summary.get('total_api_cost_est_usd', 0):.4f}",
                    'source': 'summary',
                    'run_id': run_id
                })

    except json.JSONDecodeError:
        print(f"Warning: Could not parse JSON from {path}", file=sys.stderr)
    except Exception as e:
        print(f"Warning: Error reading {path}: {e}", file=sys.stderr)

    return entries


def load_plain_log(path: Path) -> List[Dict[str, Any]]:
    """Load plain text .log file."""
    entries: List[Dict[str, Any]] = []
    try:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line_str: str = line
                parsed: Optional[Dict[str, Any]] = parse_log_line(line_str)
                if parsed:
                    entries.append(parsed)
    except Exception as e:
        print(f"Warning: Error reading {path}: {e}", file=sys.stderr)

    return entries


def load_logs(logs_dir: Path, json_only: bool = False, show_progress: bool = True) -> List[Dict[str, Any]]:
    """Load all log files from the logs directory.

    Args:
        logs_dir: Path to logs directory
        json_only: Only load JSON files
        show_progress: Whether to show progress bar (False in quiet mode)
    """
    all_entries = []

    if not logs_dir.exists():
        print(f"Error: Logs directory not found: {logs_dir}", file=sys.stderr)
        return all_entries

    # Find all log files
    log_files = []
    if json_only:
        log_files = list(logs_dir.glob("*.json"))
    else:
        log_files = list(logs_dir.glob("*.log")) + list(logs_dir.glob("*.json"))

    # Sort by modification time (newest first)
    log_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)

    if not log_files:
        return all_entries

    # Show progress when loading large number of files (>1000 entries expected)
    total_expected = len(log_files)
    use_progress = show_progress and HAS_TQDM and total_expected > 1

    progress_desc = "Loading log files"
    for log_file in tqdm(log_files, desc=progress_desc, unit="files", leave=False, disable=not use_progress):
        if log_file.suffix == '.json':
            entries = load_json_log(log_file)
        else:
            entries = load_plain_log(log_file)

        all_entries.extend(entries)

    return all_entries


def filter_entries(
    entries: List[Dict[str, Any]],
    level: Optional[str] = None,
    category: Optional[str] = None,
    from_time: Optional[datetime] = None,
    to_time: Optional[datetime] = None,
    search: Optional[str] = None,
    stage: Optional[str] = None,
    min_duration: Optional[float] = None,
    max_duration: Optional[float] = None,
    error_pattern: Optional[str] = None,
    show_progress: bool = True,
) -> List[Dict[str, Any]]:
    """Filter log entries by various criteria.

    Args:
        entries: List of log entries to filter
        level: Filter by log level
        category: Filter by error category
        from_time: Filter entries from this time
        to_time: Filter entries to this time
        search: Search for text in messages
        stage: Filter by pipeline stage name
        min_duration: Minimum duration in seconds (for stage timings)
        max_duration: Maximum duration in seconds (for stage timings)
        error_pattern: Regex pattern to match against error messages
        show_progress: Whether to show progress bar (False in quiet mode)
    """
    filtered: List[Dict[str, Any]] = []
    compiled_pattern: Optional[re.Pattern] = None

    if error_pattern:
        try:
            compiled_pattern = re.compile(error_pattern, re.IGNORECASE)
        except re.error as e:
            print(f"Warning: Invalid regex pattern '{error_pattern}': {e}", file=sys.stderr)

    # Error categories mapping
    error_categories: Dict[str, List[str]] = {
        'rate_limit': ['rate limit', '429', 'too many requests', 'quota'],
        'network': ['network', 'connection', 'timeout', 'dns', 'socket'],
        'auth': ['auth', 'unauthorized', 'forbidden', '401', '403', 'api key'],
        'download': ['download', 'yt-dlp', 'youtube', 'video not found', '404'],
        'transcription': ['whisper', 'transcription', 'audio'],
        'matching': ['match', 'embedding', 'similarity'],
        'config': ['config', 'configuration', 'yaml'],
        'unknown': ['unknown', 'unclassified', 'unexpected'],
    }

    # Show progress when filtering large number of entries (>1000)
    use_progress = show_progress and HAS_TQDM and len(entries) > 1000

    for entry in tqdm(entries, desc="Filtering entries", unit="entries", leave=False, disable=not use_progress):
        # Filter by level
        if level:
            entry_level: str = entry.get('level', '').upper()
            if level.upper() not in ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']:
                # Treat as category filter if not a valid level
                pass
            elif entry_level != level.upper():
                continue

        # Filter by category (search in message)
        if category:
            msg: str = entry.get('message', '').lower()
            cat_lower: str = category.lower()

            # Check explicit category
            if cat_lower in error_categories:
                found: bool = False
                for keyword in error_categories[cat_lower]:
                    keyword_str: str = keyword
                    if keyword_str in msg:
                        found = True
                        break
                if not found:
                    continue
            else:
                # Search for category term directly
                if cat_lower not in msg:
                    continue

        # Filter by time range
        ts_str: str = entry.get('timestamp', '')
        if from_time or to_time:
            ts: Optional[datetime] = parse_timestamp(ts_str)
            if ts:
                if from_time and ts < from_time:
                    continue
                if to_time and ts > to_time:
                    continue
            else:
                # If we can't parse timestamp, include entry (conservative)
                pass

        # Filter by search term
        if search:
            msg = entry.get('message', '').lower()
            search_lower: str = search.lower()
            if search_lower not in msg:
                continue

        # Filter by stage (look for "Stage X:" pattern)
        if stage:
            msg = entry.get('message', '')
            stage_pattern = f'Stage {stage}:'
            if stage_pattern not in msg:
                # Also check for stage in run_id or source
                if stage.lower() not in entry.get('run_id', '').lower() and stage.lower() not in entry.get('source', '').lower():
                    continue

        # Filter by duration (extract from stage timing messages)
        if min_duration is not None or max_duration is not None:
            msg = entry.get('message', '')
            # Look for "Stage X: Y.Ys" pattern
            duration_match = re.search(r'Stage \w+: ([\d.]+)s', msg)
            if duration_match:
                duration = float(duration_match.group(1))
                if min_duration is not None and duration < min_duration:
                    continue
                if max_duration is not None and duration > max_duration:
                    continue
            else:
                # If no duration found and we have duration filters, skip entry
                if min_duration is not None or max_duration is not None:
                    continue

        # Filter by error pattern (regex)
        if compiled_pattern:
            msg = entry.get('message', '')
            if not compiled_pattern.search(msg):
                continue

        filtered.append(entry)

    return filtered


def display_entries(
    entries: List[Dict[str, Any]],
    count: Optional[int] = None,
    color: bool = True,
    verbose: bool = False,
    quiet: bool = False,
    group_by: Optional[str] = None,
    tail: Optional[int] = None,
    output_format: str = 'plain',
) -> None:
    """Display log entries with formatting.

    Args:
        entries: List of log entries to display
        count: Maximum number of entries to show
        color: Whether to use colored output
        verbose: Show full JSON payload, detailed category info, full timestamps
        quiet: Only show errors (suppress warnings, info, etc.)
        group_by: Group entries by 'stage' or 'level'
        tail: Show last N entries instead of first N
        output_format: Output format: 'plain', 'json', or 'csv'
    """
    # Handle tail (show last N entries)
    if tail:
        entries = entries[-tail:]

    if count:
        entries = entries[:count]

    if not entries:
        if not quiet:
            print("No log entries found matching the criteria.")
        return

    # Handle CSV output
    if output_format == 'csv':
        import csv
        import io
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=['timestamp', 'level', 'message', 'source', 'run_id'])
        writer.writeheader()
        for entry in entries:
            row = {
                'timestamp': entry.get('timestamp', ''),
                'level': entry.get('level', ''),
                'message': entry.get('message', ''),
                'source': entry.get('source', ''),
                'run_id': entry.get('run_id', ''),
            }
            writer.writerow(row)
        print(output.getvalue())
        return

    # Handle JSON output (machine-readable)
    if output_format == 'json':
        print(json.dumps(entries, indent=2, ensure_ascii=False))
        return

    # Handle grouping
    if group_by:
        if group_by == 'stage':
            # Group by extracted stage name
            from collections import defaultdict
            groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
            for entry in entries:
                msg = entry.get('message', '')
                # Extract stage name from "Stage X:" pattern
                stage_match = re.search(r'Stage (\w+):', msg)
                if stage_match:
                    stage_name = stage_match.group(1)
                else:
                    stage_name = 'Other'
                groups[stage_name].append(entry)

            # Display grouped entries
            for stage_name in sorted(groups.keys()):
                print(f"\n{'=' * 60}")
                print(f"Stage: {stage_name} ({len(groups[stage_name])} entries)")
                print(f"{'=' * 60}")
                group_entries = groups[stage_name]
                # Display entries for this group (without re-grouping by run_id)
                for entry in group_entries:
                    _display_single_entry(entry, color, verbose)
            return
        elif group_by == 'level':
            # Group by log level
            from collections import defaultdict
            groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
            for entry in entries:
                level = entry.get('level', 'UNKNOWN')
                groups[level].append(entry)

            # Display grouped entries
            for level_name in ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']:
                if level_name not in groups:
                    continue
                print(f"\n{'=' * 60}")
                print(f"Level: {level_name} ({len(groups[level_name])} entries)")
                print(f"{'=' * 60}")
                group_entries = groups[level_name]
                for entry in group_entries:
                    _display_single_entry(entry, color, verbose)
            return

    # Group by run_id if available
    current_run: Optional[str] = None

    # Error categories for verbose mode
    error_categories: Dict[str, List[str]] = {
        'rate_limit': ['rate limit', '429', 'too many requests', 'quota'],
        'network': ['network', 'connection', 'timeout', 'dns', 'socket'],
        'auth': ['auth', 'unauthorized', 'forbidden', '401', '403', 'api key'],
        'download': ['download', 'yt-dlp', 'youtube', 'video not found', '404'],
        'transcription': ['whisper', 'transcription', 'audio'],
        'matching': ['match', 'embedding', 'similarity'],
        'config': ['config', 'configuration', 'yaml'],
    }

    def get_category(message: str) -> Optional[str]:
        """Determine error category from message."""
        msg_lower = message.lower()
        for cat, keywords in error_categories.items():
            for kw in keywords:
                if kw in msg_lower:
                    return cat
        return None

    def _display_single_entry(entry_dict: Dict[str, Any], use_color: bool, use_verbose: bool) -> None:
        """Display a single log entry."""
        nonlocal current_run
        level: str = entry_dict.get('level', 'INFO')
        ts: str = entry_dict.get('timestamp', '')
        msg: str = entry_dict.get('message', '')

        if use_color and level in LEVEL_COLORS:
            level_str: str = f"{LEVEL_COLORS[level]}{level}{RESET}"
        else:
            level_str = level

        if use_verbose:
            category: Optional[str] = get_category(msg)
            if category:
                print(f"[Category: {category}]")
            print(f"{ts} | {level_str} | {msg}")
            if 'payload' in entry_dict:
                print(f"  Payload: {json.dumps(entry_dict['payload'], indent=2)}")
            if 'extra' in entry_dict:
                print(f"  Extra: {json.dumps(entry_dict['extra'], indent=2)}")
        else:
            if len(msg) > 120:
                msg = msg[:117] + "..."
            print(f"{ts:<22} | {level_str:<8} | {msg}")

    for entry in entries:
        entry_dict: Dict[str, Any] = entry
        level: str = entry_dict.get('level', 'INFO')

        # Quiet mode: only show errors and critical
        if quiet and level not in ('ERROR', 'CRITICAL'):
            continue

        run_id: Optional[str] = entry_dict.get('run_id')

        # Show run header if changed
        if run_id and run_id != current_run:
            if not quiet or verbose:
                print(f"\n{'=' * 60}")
                print(f"Run: {run_id}")
                print(f"{'=' * 60}")
            current_run = run_id

        ts: str = entry_dict.get('timestamp', '')
        msg: str = entry_dict.get('message', '')

        if color and level in LEVEL_COLORS:
            level_str: str = f"{LEVEL_COLORS[level]}{level}{RESET}"
        else:
            level_str = level

        if verbose:
            # Verbose: show full JSON payload, detailed category info
            category: Optional[str] = get_category(msg)
            if category:
                print(f"[Category: {category}]")

            # Show full timestamp without truncation
            print(f"{ts} | {level_str} | {msg}")

            # Show full JSON payload if available
            if 'payload' in entry_dict:
                print(f"  Payload: {json.dumps(entry_dict['payload'], indent=2)}")
            if 'extra' in entry_dict:
                print(f"  Extra: {json.dumps(entry_dict['extra'], indent=2)}")
        else:
            # Normal mode: truncate long messages
            if len(msg) > 120:
                msg = msg[:117] + "..."

            print(f"{ts:<22} | {level_str:<8} | {msg}")


def save_entries(
    entries: List[Dict[str, Any]],
    output_path: str,
    verbose: bool = False,
) -> bool:
    """Save log entries to a file.

    Args:
        entries: List of log entries to save
        output_path: Path to save the file (.json or .txt)
        verbose: Whether to show verbose output

    Returns:
        True if save was successful, False otherwise
    """
    if not entries:
        print("No entries to save.", file=sys.stderr)
        return False

    path = Path(output_path)
    suffix = path.suffix.lower()

    try:
        if suffix == '.json':
            # JSON output: include all matching log entries with metadata
            output_data = {
                'entries': entries,
                'metadata': {
                    'total_entries': len(entries),
                    'saved_at': datetime.now().isoformat(),
                    'format_version': '1.0'
                }
            }
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            if verbose:
                print(f"Saved {len(entries)} entries to {path}")
        elif suffix == '.txt':
            # Text output: preserve formatted output
            with open(path, 'w', encoding='utf-8') as f:
                for entry in entries:
                    entry_dict: Dict[str, Any] = entry
                    ts: str = entry_dict.get('timestamp', '')
                    level: str = entry_dict.get('level', 'INFO')
                    msg: str = entry_dict.get('message', '')
                    run_id: Optional[str] = entry_dict.get('run_id')

                    line = f"{ts:<22} | {level:<8} | {msg}"
                    if run_id:
                        line += f" (run: {run_id})"
                    f.write(line + '\n')
            if verbose:
                print(f"Saved {len(entries)} entries to {path}")
        else:
            print(f"Error: Unsupported output format '{suffix}'. Use .json or .txt", file=sys.stderr)
            return False

        return True

    except Exception as e:
        print(f"Error saving to {output_path}: {e}", file=sys.stderr)
        return False
    """Print summary statistics for log entries."""
    if not entries:
        return

    # Count by level
    level_counts: Dict[str, int] = {}
    for entry in entries:
        entry_dict: Dict[str, Any] = entry
        level: str = entry_dict.get('level', 'UNKNOWN')
        level_counts[level] = level_counts.get(level, 0) + 1

    print("\nSummary:")
    print("-" * 40)

    total: int = len(entries)
    for level in ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']:
        count: int = level_counts.get(level, 0)
        if count > 0:
            pct: float = (count / total) * 100
            print(f"  {level:<10}: {count:>4} ({pct:>5.1f}%)")

    print(f"  {'TOTAL':<10}: {total:>4}")


def main():
    parser = argparse.ArgumentParser(
        description='Diagnostic Log Viewer - View and filter pipeline logs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        '--logs-dir', '-d',
        type=str,
        default='./logs',
        help='Directory containing log files (default: ./logs)'
    )

    parser.add_argument(
        '--level', '-l',
        type=str,
        help='Filter by log level: DEBUG, INFO, WARNING, ERROR, CRITICAL'
    )

    parser.add_argument(
        '--category', '-c',
        type=str,
        help='Filter by error category: rate_limit, network, auth, download, transcription, matching, config, unknown'
    )

    parser.add_argument(
        '--from',
        dest='from_time',
        type=str,
        help='Filter entries from this time (ISO format: 2026-02-17T10:00:00)'
    )

    parser.add_argument(
        '--to',
        dest='to_time',
        type=str,
        help='Filter entries to this time (ISO format: 2026-02-17T12:00:00)'
    )

    parser.add_argument(
        '--search', '-s',
        type=str,
        help='Search for text in log messages'
    )

    parser.add_argument(
        '--json-only',
        action='store_true',
        help='Only load JSON log files (skip .log files)'
    )

    parser.add_argument(
        '--recent', '-r',
        type=int,
        help='Show only the N most recent entries'
    )

    parser.add_argument(
        '--no-color',
        action='store_true',
        help='Disable colored output'
    )

    parser.add_argument(
        '--summary', '-S',
        action='store_true',
        help='Show summary statistics'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output: show timestamps, full JSON payloads, detailed category info'
    )

    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Quiet mode: only show errors matching --level filter'
    )

    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Save filtered logs to file. Use .json for JSON output (includes metadata), .txt for formatted text output'
    )

    # New arguments for US-131-006
    parser.add_argument(
        '--stage', '-g',
        type=str,
        help='Filter by pipeline stage name (e.g., VIDEO_SEARCH, CAPTION, MATCH)'
    )

    parser.add_argument(
        '--min-duration',
        type=float,
        help='Minimum duration in seconds (for stage timings)'
    )

    parser.add_argument(
        '--max-duration',
        type=float,
        help='Maximum duration in seconds (for stage timings)'
    )

    parser.add_argument(
        '--error-pattern', '-e',
        type=str,
        help='Filter by regex pattern matching error messages'
    )

    parser.add_argument(
        '--output-format', '-f',
        type=str,
        choices=['plain', 'json', 'csv'],
        default='plain',
        help='Output format: plain (default), json, or csv'
    )

    parser.add_argument(
        '--group-by',
        type=str,
        choices=['stage', 'level'],
        help='Group output by stage or level'
    )

    parser.add_argument(
        '--tail', '-t',
        type=int,
        help='Show last N entries instead of first N'
    )

    args = parser.parse_args()

    # Set verbosity level
    if args.quiet:
        set_verbosity(0)
    elif args.verbose:
        set_verbosity(2)
    else:
        set_verbosity(1)

    # Convert time filters
    from_time = None
    if args.from_time:
        try:
            from_time = datetime.fromisoformat(args.from_time)
        except ValueError:
            print(f"Error: Invalid from time format: {args.from_time}", file=sys.stderr)
            print("Use ISO format: 2026-02-17T10:00:00", file=sys.stderr)
            sys.exit(1)

    to_time = None
    if args.to_time:
        try:
            to_time = datetime.fromisoformat(args.to_time)
        except ValueError:
            print(f"Error: Invalid to time format: {args.to_time}", file=sys.stderr)
            print("Use ISO format: 2026-02-17T12:00:00", file=sys.stderr)
            sys.exit(1)

    # Load logs
    logs_dir = Path(args.logs_dir)

    verbosity = get_verbosity()
    show_progress = not args.quiet  # Suppress progress in quiet mode

    if verbosity >= 1:
        print(f"Loading logs from: {logs_dir}")

    entries = load_logs(logs_dir, json_only=args.json_only, show_progress=show_progress)

    if verbosity >= 1:
        print(f"Found {len(entries)} log entries")

    # Filter entries
    entries = filter_entries(
        entries,
        level=args.level,
        category=args.category,
        from_time=from_time,
        to_time=to_time,
        search=args.search,
        stage=args.stage,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
        error_pattern=args.error_pattern,
        show_progress=show_progress,
    )

    if verbosity >= 1 and len(entries) != len(load_logs(logs_dir, json_only=args.json_only, show_progress=False)):
        print(f"Filtered to {len(entries)} entries")

    # Display
    display_entries(
        entries,
        count=args.recent,
        color=not args.no_color,
        verbose=args.verbose,
        quiet=args.quiet,
        group_by=args.group_by,
        tail=args.tail,
        output_format=args.output_format,
    )

    # Summary if requested
    if args.summary:
        print_summary(entries)

    # Save to file if requested
    if args.output:
        verbosity = get_verbosity()
        success = save_entries(entries, args.output, verbose=(verbosity >= 1))
        if not success:
            sys.exit(1)


if __name__ == '__main__':
    main()
