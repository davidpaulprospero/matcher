#!/usr/bin/env python3
"""
Batch Project Operations Script for Voiceover-Matcher

Provides batch operations across multiple projects: resume, cleanup, status, export, and import.

Usage:
    python batch_operations.py --path "E:/Edit Job/*" --resume
    python batch_operations.py --path "E:/Edit Job/*" --cleanup --level cache
    python batch_operations.py --path "E:/Edit Job/*" --status
    python batch_operations.py --path "E:/Edit Job/*" --resume --parallel
    python batch_operations.py --path "E:/Edit Job/*" --dry-run --status
    python batch_operations.py --path "E:/Edit Job/*" --export --output checkpoints.json
    python batch_operations.py --path "E:/Edit Job/*" --export --output checkpoints.json --stages video_search,caption
    python batch_operations.py --path "E:/Edit Job/*" --import --input checkpoints.json --dry-run
"""

import os
import sys
import json
import glob
import argparse
import subprocess
import concurrent.futures
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

# Add project root and scripts directory to path for imports
_script_path = os.path.abspath(__file__)
project_root = Path(_script_path).parent.parent
scripts_dir = Path(_script_path).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(scripts_dir))

# Change to project root so relative paths (config.yaml) work correctly
os.chdir(project_root)

# Import standardized output functions
from script_utils import print_error, print_warn, print_ok, print_info
from script_utils import progress_bar, track_progress, set_verbosity, get_verbosity

# Import CLI helpers
from utils.cli_helpers import format_size


# Cleanup levels (same as cleanup_project.py)
LEVEL_CACHE = "cache"
LEVEL_MEDIA = "media"
LEVEL_FULL = "full"

# Stage order for resume logic (mirrors STAGE_ORDER from checkpoint.py)
STAGE_ORDER = [
    "ANALYZE",
    "VIDEO_SEARCH",
    "CAPTION",
    "MATCH",
    "ITERATIVE_MATCH",
    "DOWNLOAD_SEGMENTS",
    "OUTPUT"
]


def get_last_completed_stage(project_dir: Path) -> Optional[str]:
    """
    Detect the last completed stage for a project.

    Args:
        project_dir: Path to project directory

    Returns:
        Stage name (e.g., 'MATCH') or None if no checkpoint/unknown
    """
    checkpoint = load_checkpoint(project_dir)
    return checkpoint.get('last_completed_stage')


def is_stage_completed(project_dir: Path, stage: str) -> bool:
    """
    Check if a specific stage has been completed.

    Args:
        project_dir: Path to project directory
        stage: Stage name (e.g., 'MATCH', 'OUTPUT')

    Returns:
        True if stage is completed, False otherwise
    """
    last_stage = get_last_completed_stage(project_dir)
    if last_stage is None:
        return False

    if last_stage not in STAGE_ORDER:
        return False

    last_idx = STAGE_ORDER.index(last_stage)
    stage_idx = STAGE_ORDER.index(stage) if stage in STAGE_ORDER else -1

    return stage_idx >= 0 and last_idx >= stage_idx


def should_skip_for_resume(project_dir: Path) -> Tuple[bool, str]:
    """
    Determine if a project should be skipped during resume.

    Args:
        project_dir: Path to project directory

    Returns:
        Tuple of (should_skip, reason)
    """
    last_stage = get_last_completed_stage(project_dir)

    # No checkpoint - shouldn't happen if project was found, but handle gracefully
    if last_stage is None:
        return False, "No checkpoint found"

    # Check if already at OUTPUT stage (pipeline complete)
    if last_stage == "OUTPUT":
        return True, f"Already at OUTPUT stage (pipeline complete)"

    # Check if beyond all known stages (unknown/future stage)
    if last_stage not in STAGE_ORDER:
        return False, f"Unknown stage: {last_stage}"

    return False, ""


def resume_project_from_stage(project_dir: Path, stage: Optional[str] = None,
                             dry_run: bool = False) -> Tuple[bool, str]:
    """
    Resume a single project from a specific stage using main.py.

    Args:
        project_dir: Path to project directory
        stage: Stage to resume from (None = resume from checkpoint)
        dry_run: If True, only preview without executing

    Returns:
        Tuple of (success, message)
    """
    if dry_run:
        if stage:
            return True, f"[DRY-RUN] Would resume: {project_dir.name} from {stage}"
        return True, f"[DRY-RUN] Would resume: {project_dir.name}"

    cmd = [sys.executable, "main.py", "--project", str(project_dir), "--resume"]

    if stage:
        # Modify checkpoint to resume from specific stage
        checkpoint = load_checkpoint(project_dir)
        if checkpoint:
            # Set last_completed_stage to the stage before the target
            if stage in STAGE_ORDER:
                stage_idx = STAGE_ORDER.index(stage)
                if stage_idx > 0:
                    checkpoint['last_completed_stage'] = STAGE_ORDER[stage_idx - 1]
                    # Write modified checkpoint
                    checkpoint_path = project_dir / "checkpoint.json"
                    try:
                        with open(checkpoint_path, 'w', encoding='utf-8') as f:
                            json.dump(checkpoint, f, indent=2, default=str)
                    except Exception as e:
                        return False, f"Error modifying checkpoint: {e}"

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',
            timeout=3600  # 1 hour timeout
        )

        if result.returncode == 0:
            return True, f"Resumed: {project_dir.name}"
        else:
            return False, f"Failed: {project_dir.name} (exit code {result.returncode})"

    except subprocess.TimeoutExpired:
        return False, f"Timeout: {project_dir.name}"
    except Exception as e:
        return False, f"Error: {project_dir.name} - {str(e)}"


def find_projects(path_pattern: str) -> List[Path]:
    """
    Find all project directories matching the given path pattern.

    Args:
        path_pattern: Glob pattern or direct path (e.g., 'E:/Edit Job/*')

    Returns:
        List of project directory paths
    """
    # Expand the glob pattern
    expanded = glob.glob(str(path_pattern))
    projects = []

    for path_str in expanded:
        path = Path(path_str)
        if not path.exists():
            continue

        # If it's a directory, check if it's a valid project
        if path.is_dir():
            if _is_valid_project(path):
                projects.append(path)
        elif path.is_file():
            # If it's a file, check if its parent is a valid project
            if _is_valid_project(path.parent):
                projects.append(path.parent)

    return sorted(projects, key=lambda p: p.name)


def _is_valid_project(path: Path) -> bool:
    """Check if directory is a valid project (has run.bat, run.sh, or checkpoint)."""
    return (
        (path / "run.bat").exists() or
        (path / "run.sh").exists() or
        (path / "checkpoint.json").exists()
    )


def load_checkpoint(project_dir: Path) -> Dict[str, Any]:
    """Load checkpoint.json for a project."""
    checkpoint_path = project_dir / "checkpoint.json"
    if not checkpoint_path.exists():
        return {}

    try:
        with open(checkpoint_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return {}


def get_project_status(project_dir: Path) -> Dict[str, Any]:
    """
    Get status information for a project.

    Returns dict with:
        - path: Project path
        - name: Project name
        - last_stage: Last completed stage
        - video_count: Number of videos
        - match_count: Number of matches
        - avg_confidence: Average confidence score
        - completed: Whether pipeline is complete
        - has_checkpoint: Whether checkpoint exists
    """
    checkpoint = load_checkpoint(project_dir)

    status: Dict[str, Any] = {
        'path': str(project_dir),
        'name': project_dir.name,
        'last_stage': checkpoint.get('last_completed_stage'),
        'completed': checkpoint.get('last_completed_stage') == 'OUTPUT',
        'has_checkpoint': (project_dir / "checkpoint.json").exists(),
        'video_count': 0,
        'match_count': 0,
        'avg_confidence': 0,
        'created_at': checkpoint.get('created_at'),
        'updated_at': checkpoint.get('updated_at'),
    }

    # Extract stats from checkpoint
    if checkpoint:
        download = checkpoint.get('download', {})
        status['video_count'] = len(download.get('video_paths', []))

        match = checkpoint.get('match', {})
        status['match_count'] = match.get('match_count', 0)
        status['avg_confidence'] = match.get('avg_confidence', 0)

    return status


def print_status_table(projects: List[Dict[str, Any]], verbose: bool = False) -> None:
    """Print status table for all projects."""
    if not projects:
        print("  No projects found.")
        return

    # Calculate column widths
    name_width = max(len(p['name']) for p in projects) + 2
    stage_width = max(len(p.get('last_stage', 'N/A') or 'N/A') for p in projects) + 2

    # Print header
    print(f"\n  {'Project Name':<{name_width}} {'Last Stage':<{stage_width}} {'Videos':>7} {'Matches':>8} {'Conf':>6} {'Status':<10}")
    print("  " + "-" * (name_width + stage_width + 40))

    for p in projects:
        name = p['name'][:name_width-2]
        stage = p.get('last_stage', 'N/A') or 'N/A'
        videos = p.get('video_count', 0)
        matches = p.get('match_count', 0)
        conf = p.get('avg_confidence', 0)
        status = "COMPLETE" if p.get('completed') else "IN PROGRESS"

        print(f"  {name:<{name_width}} {stage:<{stage_width}} {videos:>7} {matches:>8} {conf:>6.0%} {status:<10}")

    # Summary
    total = len(projects)
    completed = sum(1 for p in projects if p.get('completed'))
    in_progress = total - completed
    print(f"\n  Total: {total} | Completed: {completed} | In Progress: {in_progress}")


def resume_project(project_dir: Path, dry_run: bool = False) -> Tuple[bool, str]:
    """
    Resume a single project using main.py.

    Returns:
        Tuple of (success, message)
    """
    if dry_run:
        return True, f"[DRY-RUN] Would resume: {project_dir.name}"

    try:
        # Run main.py with --resume flag
        result = subprocess.run(
            [sys.executable, "main.py", "--project", str(project_dir), "--resume"],
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',
            timeout=3600  # 1 hour timeout
        )

        if result.returncode == 0:
            return True, f"Resumed: {project_dir.name}"
        else:
            return False, f"Failed: {project_dir.name} (exit code {result.returncode})"

    except subprocess.TimeoutExpired:
        return False, f"Timeout: {project_dir.name}"
    except Exception as e:
        return False, f"Error: {project_dir.name} - {str(e)}"


def cleanup_project(project_dir: Path, level: str = LEVEL_CACHE,
                    dry_run: bool = False) -> Tuple[bool, str]:
    """
    Cleanup a single project using cleanup_project.py.

    Returns:
        Tuple of (success, message)
    """
    cmd = [sys.executable, "scripts/cleanup_project.py",
           "--project", str(project_dir),
           "--level", level]

    if dry_run:
        cmd.append("--dry-run")

    cmd.extend(["--yes"])  # Skip confirmation for batch operations

    if dry_run:
        return True, f"[DRY-RUN] Would cleanup: {project_dir.name} (level: {level})"

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',
            timeout=300  # 5 minute timeout
        )

        if result.returncode == 0:
            return True, f"Cleaned: {project_dir.name}"
        else:
            return False, f"Failed: {project_dir.name} (exit code {result.returncode})"

    except subprocess.TimeoutExpired:
        return False, f"Timeout: {project_dir.name}"
    except Exception as e:
        return False, f"Error: {project_dir.name} - {str(e)}"


def batch_resume(projects: List[Path], parallel: bool = False,
                dry_run: bool = False, force: bool = False,
                resume_from_stage: Optional[str] = None) -> Dict[str, Any]:
    """
    Run batch resume on multiple projects with checkpoint-aware logic.

    Args:
        projects: List of project directories
        parallel: Run in parallel
        dry_run: Preview without executing
        force: Force resume even for completed projects
        resume_from_stage: Resume from specific stage

    Returns:
        Results dict with summary info
    """
    results: Dict[str, Any] = {
        'total': len(projects),
        'succeeded': 0,
        'failed': 0,
        'skipped': 0,
        'projects': [],
        'skipped_projects': []
    }

    print(f"\n  Batch resuming {len(projects)} project(s)...")

    # Get verbosity for progress bar control
    verbose = get_verbosity() >= 1

    # Filter projects based on checkpoint status (unless --force)
    projects_to_resume = []
    for project in projects:
        should_skip, reason = should_skip_for_resume(project)
        if should_skip and not force:
            results['skipped'] += 1
            last_stage = get_last_completed_stage(project)
            results['skipped_projects'].append({
                'path': str(project),
                'name': project.name,
                'last_stage': last_stage,
                'reason': reason
            })
            print_warn(f"Skipped: {project.name} - {reason}")
        else:
            projects_to_resume.append(project)

    if not projects_to_resume:
        print_warn("  No projects to resume.")
        return results

    print(f"  Resuming {len(projects_to_resume)} project(s)...")

    if parallel:
        # Run in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(resume_project_from_stage, p, resume_from_stage, dry_run): p
                      for p in projects_to_resume}
            for future in concurrent.futures.as_completed(futures):
                project = futures[future]
                success, message = future.result()
                results['projects'].append({'path': str(project), 'success': success, 'message': message})
                if success:
                    results['succeeded'] += 1
                    print_ok(message)
                else:
                    results['failed'] += 1
                    print_error(message)
    else:
        # Run sequentially with progress bar
        for project in progress_bar(projects_to_resume, desc="Resuming projects", disable=not verbose):
            success, message = resume_project_from_stage(project, resume_from_stage, dry_run)
            results['projects'].append({'path': str(project), 'success': success, 'message': message})
            if success:
                results['succeeded'] += 1
                print_ok(message)
            else:
                results['failed'] += 1
                print_error(message)

    return results


def batch_cleanup(projects: List[Path], level: str = LEVEL_CACHE,
                 parallel: bool = False, dry_run: bool = False) -> Dict[str, Any]:
    """Run batch cleanup on multiple projects."""
    results: Dict[str, Any] = {
        'total': len(projects),
        'succeeded': 0,
        'failed': 0,
        'cleanup_level': level,
        'projects': []
    }

    print(f"\n  Batch cleaning up {len(projects)} project(s) (level: {level})...")

    # Get verbosity for progress bar control
    verbose = get_verbosity() >= 1

    if parallel:
        # Run in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(cleanup_project, p, level, dry_run): p for p in projects}
            for future in concurrent.futures.as_completed(futures):
                project = futures[future]
                success, message = future.result()
                results['projects'].append({'path': str(project), 'success': success, 'message': message})
                if success:
                    results['succeeded'] += 1
                    print_ok(message)
                else:
                    results['failed'] += 1
                    print_error(message)
    else:
        # Run sequentially with progress bar
        for project in progress_bar(projects, desc="Cleaning projects", disable=not verbose):
            success, message = cleanup_project(project, level, dry_run)
            results['projects'].append({'path': str(project), 'success': success, 'message': message})
            if success:
                results['succeeded'] += 1
                print_ok(message)
            else:
                results['failed'] += 1
                print_error(message)

    return results


def batch_status(projects: List[Path], verbose: bool = False,
                dry_run: bool = False) -> Dict[str, Any]:
    """Get batch status for multiple projects."""
    print(f"\n  Checking status of {len(projects)} project(s)...")

    statuses = []
    for project in projects:
        status = get_project_status(project)
        statuses.append(status)

    # Print status table
    print_status_table(statuses, verbose)

    # Summary stats
    total = len(statuses)
    completed = sum(1 for s in statuses if s.get('completed'))
    in_progress = total - completed
    total_videos = sum(s.get('video_count', 0) for s in statuses)
    total_matches = sum(s.get('match_count', 0) for s in statuses)

    return {
        'total': total,
        'completed': completed,
        'in_progress': in_progress,
        'total_videos': total_videos,
        'total_matches': total_matches,
        'projects': statuses
    }


# =============================================================================
# EXPORT FUNCTIONS
# =============================================================================

def export_checkpoint(project_dir: Path, include_stages: Optional[List[str]] = None,
                       since_date: Optional[str] = None) -> Dict[str, Any]:
    """
    Export checkpoint data from a project.

    Args:
        project_dir: Path to project directory
        include_stages: List of stages to include (None = all stages)
        since_date: ISO date string to filter by date range

    Returns:
        Dict with export data or error info
    """
    checkpoint_path = project_dir / "checkpoint.json"
    if not checkpoint_path.exists():
        return {"exists": False, "error": "checkpoint.json not found"}

    try:
        with open(checkpoint_path, 'r', encoding='utf-8') as f:
            checkpoint = json.load(f)
    except json.JSONDecodeError as e:
        return {"exists": False, "error": f"Invalid JSON: {e}"}

    export_data = {
        "version": checkpoint.get("version", "unknown"),
        "last_completed_stage": checkpoint.get("last_completed_stage"),
        "created_at": checkpoint.get("created_at"),
        "updated_at": checkpoint.get("updated_at"),
    }

    # Include stages
    stages_to_export = include_stages or list(checkpoint.keys())
    valid_stages = {'analyze', 'video_search', 'caption', 'match',
                    'iterative_match', 'download_segments', 'output'}

    included_stages = []
    for stage in stages_to_export:
        if stage in checkpoint and stage in valid_stages:
            export_data[stage] = checkpoint[stage]
            included_stages.append(stage)

    # Filter by date if specified
    if since_date:
        try:
            since_dt = datetime.fromisoformat(since_date.replace('Z', '+00:00'))
            export_data["_filter"] = {"since_date": since_date}

            # Check if checkpoint is newer than since_date
            updated_at = checkpoint.get("updated_at")
            if updated_at:
                updated_dt = datetime.fromisoformat(updated_at.replace('Z', '+00:00'))
                if updated_dt < since_dt:
                    return {"exists": True, "filtered": True, "reason": "Checkpoint older than since_date"}
        except ValueError:
            pass  # Invalid date format, skip filter

    export_data["_export_info"] = {
        "exported_at": datetime.now().isoformat(),
        "included_stages": included_stages if include_stages else "all",
        "source_project": str(project_dir),
    }

    return {"exists": True, "data": export_data}


def validate_import_data(data: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate imported checkpoint data before applying.

    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []

    # Check required fields
    if not isinstance(data, dict):
        errors.append("Data must be a dictionary")
        return False, errors

    # Check version
    version = data.get("version")
    if not version:
        errors.append("Missing version field")

    # Check last_completed_stage is valid
    stage = data.get("last_completed_stage")
    valid_stages = {'ANALYZE', 'VIDEO_SEARCH', 'CAPTION', 'MATCH',
                     'ITERATIVE_MATCH', 'DOWNLOAD_SEGMENTS', 'OUTPUT', None}
    if stage and stage not in valid_stages:
        errors.append(f"Invalid stage: {stage}")

    # Check data integrity - stages should be dicts
    stage_keys = {'analyze', 'video_search', 'caption', 'match',
                  'iterative_match', 'download_segments', 'output'}
    for key in data.keys():
        if key in stage_keys and not isinstance(data[key], dict):
            errors.append(f"Stage '{key}' data must be a dictionary")

    return len(errors) == 0, errors


def import_checkpoint(project_dir: Path, import_data: Dict[str, Any],
                     dry_run: bool = False, validate: bool = True) -> Tuple[bool, str]:
    """
    Import checkpoint data into a project.

    Args:
        project_dir: Path to project directory
        import_data: The checkpoint data to import
        dry_run: If True, only validate without applying
        validate: If True, validate data before importing

    Returns:
        Tuple of (success, message)
    """
    # Validate first
    if validate:
        is_valid, errors = validate_import_data(import_data)
        if not is_valid:
            return False, f"Validation failed: {'; '.join(errors)}"

    checkpoint_path = project_dir / "checkpoint.json"
    existing_data = {}

    # Load existing checkpoint if present
    if checkpoint_path.exists():
        try:
            with open(checkpoint_path, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
        except json.JSONDecodeError as e:
            print_warn(f"Could not read existing checkpoint: {e}")

    if dry_run:
        return True, f"[DRY-RUN] Would import checkpoint to {project_dir.name}"

    # Merge import data with existing (import takes precedence)
    merged = {**existing_data, **import_data}

    # Preserve some metadata from existing checkpoint
    if existing_data:
        for key in ['created_at']:
            if key in existing_data and key not in import_data:
                merged[key] = existing_data[key]

    # Add import metadata
    merged['_imported_at'] = datetime.now().isoformat()
    merged['_imported_from'] = import_data.get('_export_info', {}).get('source_project', 'unknown')

    # Write checkpoint
    try:
        with open(checkpoint_path, 'w', encoding='utf-8') as f:
            json.dump(merged, f, indent=2, default=str)
        return True, f"Imported checkpoint to {project_dir.name}"
    except Exception as e:
        return False, f"Failed to write checkpoint: {e}"


def batch_export(projects: List[Path], output_path: Path,
                include_stages: Optional[List[str]] = None,
                since_date: Optional[str] = None) -> Dict[str, Any]:
    """Export checkpoints from multiple projects to a single JSON file."""
    print(f"\n  Exporting {len(projects)} project(s)...")

    results: Dict[str, Any] = {
        'total': len(projects),
        'succeeded': 0,
        'failed': 0,
        'projects': []
    }

    verbose = get_verbosity() >= 1

    for project in progress_bar(projects, desc="Exporting projects", disable=not verbose):
        export_result = export_checkpoint(project, include_stages, since_date)

        if export_result.get("exists") and not export_result.get("filtered"):
            results['succeeded'] += 1
            results['projects'].append({
                'path': str(project),
                'name': project.name,
                'success': True,
                'last_stage': export_result['data'].get('last_completed_stage')
            })
            print_ok(f"Exported: {project.name}")
        else:
            results['failed'] += 1
            reason = export_result.get("error") or export_result.get("reason", "unknown")
            results['projects'].append({
                'path': str(project),
                'name': project.name,
                'success': False,
                'reason': reason
            })
            print_error(f"Failed: {project.name} - {reason}")

    # Write combined export file
    if results['succeeded'] > 0:
        export_output = {
            "version": "1.0",
            "exported_at": datetime.now().isoformat(),
            "include_stages": include_stages if include_stages else "all",
            "since_date": since_date,
            "projects": results['projects']
        }

        # Include actual checkpoint data
        export_output['data'] = {}
        for project in projects:
            result = export_checkpoint(project, include_stages, since_date)
            if result.get("exists") and not result.get("filtered"):
                export_output['data'][project.name] = result['data']

        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(export_output, f, indent=2, default=str)
            print_ok(f"Written to: {output_path}")
        except Exception as e:
            print_error(f"Failed to write export file: {e}")

    return results


def batch_import(projects: List[Path], input_path: Path,
                dry_run: bool = False, validate: bool = True,
                allow_missing: bool = False) -> Dict[str, Any]:
    """Import checkpoints from a JSON file to multiple projects."""
    print(f"\n  Importing to {len(projects)} project(s)...")

    # Load import file
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            import_file = json.load(f)
    except Exception as e:
        print_error(f"Failed to read import file: {e}")
        return {'total': 0, 'succeeded': 0, 'failed': 0, 'projects': []}

    # Get import data (can be combined export or individual)
    import_data = import_file.get('data', import_file)

    results: Dict[str, Any] = {
        'total': len(projects),
        'succeeded': 0,
        'failed': 0,
        'projects': []
    }

    verbose = get_verbosity() >= 1

    for project in progress_bar(projects, desc="Importing projects", disable=not verbose):
        project_name = project.name

        # Find matching data for this project
        if project_name in import_data:
            data_to_import = import_data[project_name]
        elif allow_missing:
            print_warn(f"No data for {project_name}, skipping")
            continue
        else:
            results['failed'] += 1
            results['projects'].append({
                'path': str(project),
                'name': project_name,
                'success': False,
                'reason': 'No matching data in import file'
            })
            print_error(f"No data for: {project_name}")
            continue

        # Import
        success, message = import_checkpoint(project, data_to_import, dry_run, validate)

        results['projects'].append({
            'path': str(project),
            'name': project_name,
            'success': success,
            'message': message
        })

        if success:
            results['succeeded'] += 1
            print_ok(message)
        else:
            results['failed'] += 1
            print_error(f"Failed: {project_name} - {message}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Batch operations on multiple Voiceover-Matcher projects',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Check status of all projects in a directory
  python batch_operations.py --path "E:/Edit Job/*" --status

  # Resume all projects in a directory
  python batch_operations.py --path "E:/Edit Job/*" --resume

  # Resume with parallel execution
  python batch_operations.py --path "E:/Edit Job/*" --resume --parallel

  # Dry-run to preview operations
  python batch_operations.py --path "E:/Edit Job/*" --resume --dry-run

  # Cleanup all projects (cache level)
  python batch_operations.py --path "E:/Edit Job/*" --cleanup

  # Cleanup with different level
  python batch_operations.py --path "E:/Edit Job/*" --cleanup --level media

  # Verbose status output
  python batch_operations.py --path "E:/Edit Job/*" --status --verbose

  # JSON output for programmatic use
  python batch_operations.py --path "E:/Edit Job/*" --status --json
        '''
    )

    parser.add_argument(
        '--path', '-p',
        required=True,
        help='Path pattern to match projects (glob or direct path)'
    )

    # Operation flags (mutually exclusive, but we'll handle that)
    parser.add_argument(
        '--status', '-s',
        action='store_true',
        help='Show status of all matching projects'
    )

    parser.add_argument(
        '--resume', '-r',
        action='store_true',
        help='Resume all matching projects'
    )

    parser.add_argument(
        '--cleanup', '-c',
        action='store_true',
        help='Cleanup all matching projects'
    )

    parser.add_argument(
        '--export',
        action='store_true',
        help='Export project checkpoints to JSON'
    )

    parser.add_argument(
        '--import', dest='import_data',
        action='store_true',
        help='Import project checkpoints from JSON'
    )

    # Options
    parser.add_argument(
        '--level', '-l',
        choices=[LEVEL_CACHE, LEVEL_MEDIA, LEVEL_FULL],
        default=LEVEL_CACHE,
        help='Cleanup level (default: cache)'
    )

    parser.add_argument(
        '--parallel',
        action='store_true',
        help='Run operations in parallel (for resume/cleanup)'
    )

    parser.add_argument(
        '--force', '-f',
        action='store_true',
        help='Force resume even if project is already complete (for --resume)'
    )

    parser.add_argument(
        '--stage',
        type=str,
        choices=STAGE_ORDER,
        help='Resume from specific stage (for --resume). Example: --stage MATCH'
    )

    parser.add_argument(
        '--dry-run', '-n',
        action='store_true',
        help='Preview operations without executing'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output'
    )

    parser.add_argument(
        '--json', '-j',
        action='store_true',
        help='Output results as JSON'
    )

    # Export/Import specific options
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Output file path (for --export)'
    )

    parser.add_argument(
        '--input', '-i',
        type=str,
        help='Input file path (for --import)'
    )

    parser.add_argument(
        '--stages',
        type=str,
        help='Comma-separated list of stages to export (e.g., "video_search,caption,match")'
    )

    parser.add_argument(
        '--since-date',
        type=str,
        help='Export only checkpoints modified since this date (ISO format, e.g., "2026-01-01")'
    )

    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='Skip validation during import'
    )

    parser.add_argument(
        '--allow-missing',
        action='store_true',
        help='Allow importing even if project has no matching data in import file'
    )

    args = parser.parse_args()

    # Set verbosity level based on flags
    if hasattr(args, 'quiet') and args.quiet:
        set_verbosity(0)
    elif args.verbose:
        set_verbosity(2)
    else:
        set_verbosity(1)

    # Validate that an operation is specified
    if not (args.status or args.resume or args.cleanup or args.export or args.import_data):
        print_error("Must specify at least one operation: --status, --resume, --cleanup, --export, or --import")
        sys.exit(1)

    # Validate export/import arguments
    if args.export and not args.output:
        print_error("--output required for --export")
        sys.exit(1)

    if args.import_data and not args.input:
        print_error("--input required for --import")
        sys.exit(1)

    # Find matching projects
    print(f"  Searching for projects matching: {args.path}")
    projects = find_projects(args.path)

    if not projects:
        print_warn(f"No projects found matching: {args.path}")
        sys.exit(0)

    print(f"  Found {len(projects)} project(s):")
    for p in projects:
        print(f"    - {p.name}")
    print()

    # Execute the operation
    results: Dict[str, Any] = {}

    # Parse stages list
    include_stages = None
    if args.stages:
        include_stages = [s.strip() for s in args.stages.split(',')]

    if args.status:
        results = batch_status(projects, args.verbose, args.dry_run)
    elif args.resume:
        results = batch_resume(projects, args.parallel, args.dry_run, args.force, args.stage)
    elif args.cleanup:
        results = batch_cleanup(projects, args.level, args.parallel, args.dry_run)
    elif args.export:
        results = batch_export(projects, Path(args.output), include_stages, args.since_date)
    elif args.import_data:
        results = batch_import(projects, Path(args.input), args.dry_run,
                              not args.no_validate, args.allow_missing)

    # Print summary
    if args.status:
        # Status already prints table
        pass
    else:
        print(f"\n  Summary:")
        print(f"    Total: {results.get('total', 0)}")
        if args.resume or args.cleanup or args.export or args.import_data:
            print(f"    Succeeded: {results.get('succeeded', 0)}")
            print(f"    Failed: {results.get('failed', 0)}")
            if args.resume:
                print(f"    Skipped: {results.get('skipped', 0)}")

    # JSON output
    if args.json:
        print()
        print(json.dumps(results, indent=2))

    sys.exit(0)


if __name__ == '__main__':
    main()
