#!/usr/bin/env python3
"""
Project Health Report Generator

Generates comprehensive health reports for matcher projects including:
- Checkpoint validity and completeness
- Cache sizes and age analysis
- Missing referenced files detection
- Config validation against schema
- Health score (0-100) with recommendations

Usage:
    python scripts/project_health.py --project "E:/Edit Job/client/project"
    python scripts/project_health.py --project "E:/Edit Job/client/project" --output health.json
    python scripts/project_health.py --project "E:/Edit Job/client/project" --verbose
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add project root and scripts directory to path for imports
_script_path = os.path.abspath(__file__)
project_root = Path(_script_path).parent.parent
scripts_dir = Path(_script_path).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(scripts_dir))

# Change to project root so relative paths work correctly
os.chdir(project_root)

# Import standardized output functions
from script_utils import set_verbosity, get_verbosity, print_header, print_ok, print_warn, print_error, print_info


def get_directory_size(path: Path) -> int:
    """Get total size of directory in bytes."""
    total = 0
    try:
        for entry in path.rglob('*'):
            if entry.is_file():
                try:
                    total += entry.stat().st_size
                except (OSError, PermissionError):
                    pass
    except (OSError, PermissionError):
        pass
    return total


def format_bytes(size: int) -> str:
    """Format bytes as human-readable string."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size < 1024.0:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} PB"


def get_cache_age(path: Path) -> Optional[float]:
    """Get age of cache directory in hours."""
    try:
        if path.exists():
            # Get most recent file modification time
            latest_mtime = 0
            for entry in path.rglob('*'):
                if entry.is_file():
                    try:
                        latest_mtime = max(latest_mtime, entry.stat().st_mtime)
                    except (OSError, PermissionError):
                        pass
            if latest_mtime > 0:
                age_seconds = datetime.now().timestamp() - latest_mtime
                return age_seconds / 3600  # Convert to hours
    except (OSError, PermissionError):
        pass
    return None


def check_checkpoint(project_path: Path) -> Dict[str, Any]:
    """Check checkpoint validity and completeness."""
    from src.checkpoint import validate_checkpoint, STAGE_ORDER
    from src.config import load_config

    checkpoint_path = project_path / "checkpoint.json"
    result = {
        'exists': checkpoint_path.exists(),
        'path': str(checkpoint_path),
        'valid': False,
        'issues': [],
        'warnings': [],
        'age_hours': 0.0,
        'is_stale': False,
        'last_completed_stage': '',
        'completed_stages': [],
        'remaining_stages': [],
    }

    if not checkpoint_path.exists():
        result['issues'].append("Checkpoint file does not exist")
        return result

    # Get checkpoint age
    try:
        mtime = checkpoint_path.stat().st_mtime
        age_seconds = datetime.now().timestamp() - mtime
        result['age_hours'] = age_seconds / 3600
        result['is_stale'] = result['age_hours'] > 24  # 24 hours stale threshold
    except OSError:
        pass

    # Validate checkpoint
    validation = validate_checkpoint(str(checkpoint_path))
    result['valid'] = validation['is_valid']
    result['issues'].extend(validation.get('issues', []))
    result['warnings'].extend(validation.get('warnings', []))

    if validation.get('last_stage'):
        result['last_completed_stage'] = validation['last_stage']

    # Load full checkpoint data for more details
    try:
        import gzip
        with open(checkpoint_path, 'rb') as f:
            content = f.read()
        if len(content) >= 2 and content[0] == 0x1f and content[1] == 0x8b:
            content = gzip.decompress(content)
        data = json.loads(content.decode('utf-8'))

        result['completed_stages'] = data.get('completed_stages', [])

        # Calculate remaining stages
        last_stage = data.get('last_completed_stage', '')
        if last_stage in STAGE_ORDER:
            last_idx = STAGE_ORDER.index(last_stage)
            result['remaining_stages'] = STAGE_ORDER[last_idx + 1:]
        elif not last_stage:
            result['remaining_stages'] = STAGE_ORDER
    except Exception as e:
        result['warnings'].append(f"Could not load checkpoint details: {e}")

    return result


def check_cache(project_path: Path) -> Dict[str, Any]:
    """Report cache sizes and age."""
    result = {
        'caches': [],
        'total_size': 0,
    }

    # Cache locations to check
    cache_dirs = {
        '.cache': 'Main cache',
        '.cache/transcriptions': 'Transcriptions',
        '.cache/embeddings': 'Embeddings',
        '.cache/llm_responses': 'LLM Responses',
    }

    for cache_name, description in cache_dirs.items():
        cache_path = project_path / cache_name
        if cache_path.exists() and cache_path.is_dir():
            size = get_directory_size(cache_path)
            age_hours = get_cache_age(cache_path)
            result['caches'].append({
                'name': cache_name,
                'description': description,
                'path': str(cache_path),
                'size_bytes': size,
                'size_formatted': format_bytes(size),
                'age_hours': age_hours,
                'is_stale': age_hours > 24 if age_hours else False,
            })
            result['total_size'] += size

    # Also check global caches
    global_cache_paths = [
        Path.home() / ".matcher_global_cache",
    ]

    for global_path in global_cache_paths:
        if global_path.exists() and global_path.is_dir():
            size = get_directory_size(global_path)
            age_hours = get_cache_age(global_path)
            result['caches'].append({
                'name': str(global_path.name),
                'description': 'Global cache',
                'path': str(global_path),
                'size_bytes': size,
                'size_formatted': format_bytes(size),
                'age_hours': age_hours,
                'is_stale': age_hours > 168 if age_hours else False,  # 1 week for global
            })
            result['total_size'] += size

    result['total_size_formatted'] = format_bytes(result['total_size'])

    return result


def check_missing_files(project_path: Path) -> Dict[str, Any]:
    """Check for missing referenced files."""
    result = {
        'missing_voiceover': False,
        'missing_downloaded_videos': False,
        'missing_outputs': False,
        'issues': [],
    }

    # Check voiceover file
    voiceover_path = project_path / "voiceover"
    if not voiceover_path.exists():
        result['missing_voiceover'] = True
        result['issues'].append("Voiceover directory does not exist")

    # Check downloaded videos directory
    config = None
    try:
        config = load_config(str(project_path / "config.yaml") if (project_path / "config.yaml").exists() else None)
        downloaded_videos_dir = config.downloaded_videos_dir if config else None
    except Exception:
        downloaded_videos_dir = None

    if downloaded_videos_dir:
        # downloaded_videos_dir is usually relative to project
        if not os.path.isabs(downloaded_videos_dir):
            videos_path = project_path / downloaded_videos_dir
        else:
            videos_path = Path(downloaded_videos_dir)

        if videos_path.exists():
            video_files = list(videos_path.glob("*.mp4")) + list(videos_path.glob("*.mkv"))
            if not video_files:
                result['missing_downloaded_videos'] = True
                result['issues'].append(f"No video files found in {videos_path}")
        else:
            result['missing_downloaded_videos'] = True
            result['issues'].append(f"Downloaded videos directory does not exist: {videos_path}")

    # Check output directory
    output_path = project_path / "output"
    if not output_path.exists():
        result['missing_outputs'] = True
        result['issues'].append("Output directory does not exist")

    return result


def check_config(project_path: Path) -> Dict[str, Any]:
    """Validate config against schema."""
    from src.config import load_config, validate_config_schema

    config_path = project_path / "config.yaml"
    result = {
        'exists': config_path.exists(),
        'path': str(config_path),
        'valid': False,
        'issues': [],
        'warnings': [],
    }

    if not config_path.exists():
        result['issues'].append("Config file does not exist")
        return result

    # Load and validate config
    try:
        config = load_config(str(config_path))
        result['valid'] = True

        # Get config summary
        result['project_name'] = getattr(config, 'project', None)
        if config and hasattr(config, 'project') and config.project:
            result['project_name'] = getattr(config.project, 'name', None)
            result['project_version'] = getattr(config.project, 'version', None)

    except Exception as e:
        result['issues'].append(f"Config validation failed: {e}")

    # Additional config checks
    try:
        import yaml
        with open(config_path, 'r', encoding='utf-8') as f:
            raw_config = yaml.safe_load(f)

        # Check for deprecated options
        deprecated_options = ['project_config', 'checkpoint_auto_save']
        for deprecated in deprecated_options:
            if deprecated in raw_config:
                result['warnings'].append(f"Deprecated option detected: {deprecated}")

        # Check required sections
        required_sections = ['project', 'matching', 'download']
        for section in required_sections:
            if section not in raw_config:
                result['warnings'].append(f"Missing recommended config section: {section}")

    except Exception as e:
        result['warnings'].append(f"Could not perform additional config checks: {e}")

    return result


def check_pipeline_readiness(project_path: Path) -> Dict[str, Any]:
    """Check if project is ready to run."""
    result = {
        'ready': False,
        'missing_requirements': [],
        'can_resume': False,
        'can_run_fresh': False,
    }

    # Check if checkpoint exists and is valid
    checkpoint_path = project_path / "checkpoint.json"
    if checkpoint_path.exists():
        result['can_resume'] = True
    else:
        result['can_resume'] = False

    # Check voiceover
    voiceover_path = project_path / "voiceover"
    voiceover_files = []
    if voiceover_path.exists():
        voiceover_files = list(voiceover_path.glob("*.srt")) + list(voiceover_path.glob("*.mp3")) + \
                        list(voiceover_path.glob("*.wav")) + list(voiceover_path.glob("*.mp4"))

    if voiceover_files:
        result['can_run_fresh'] = True
    else:
        result['missing_requirements'].append("Voiceover file (SRT, MP3, WAV, or MP4)")

    # Project is ready if it can run fresh or resume
    result['ready'] = result['can_resume'] or result['can_run_fresh']

    return result


def calculate_health_score(
    checkpoint: Dict[str, Any],
    cache: Dict[str, Any],
    missing_files: Dict[str, Any],
    config: Dict[str, Any],
    readiness: Dict[str, Any],
) -> Dict[str, Any]:
    """Calculate overall health score (0-100) and generate recommendations."""

    score = 100
    max_deduct = 100

    # Deduct for checkpoint issues (max 30 points)
    if not checkpoint['valid']:
        score -= 20
        max_deduct -= 20
    if checkpoint.get('is_stale'):
        score -= 5
    if checkpoint.get('issues'):
        score -= len(checkpoint['issues']) * 3
    if checkpoint.get('warnings'):
        score -= len(checkpoint['warnings']) * 1

    # Deduct for cache issues (max 10 points)
    has_old_cache = any(c.get('is_stale', False) for c in cache.get('caches', []))
    if has_old_cache:
        score -= 5
    if cache.get('total_size', 0) > 50 * 1024 * 1024 * 1024:  # > 50GB
        score -= 5

    # Deduct for missing files (max 30 points)
    if missing_files.get('missing_voiceover'):
        score -= 15
    if missing_files.get('missing_downloaded_videos'):
        score -= 10
    if missing_files.get('missing_outputs'):
        score -= 5

    # Deduct for config issues (max 20 points)
    if not config.get('valid'):
        score -= 15
    if config.get('issues'):
        score -= len(config['issues']) * 2

    # Deduct for readiness issues (max 10 points)
    if not readiness.get('ready'):
        score -= 10

    # Ensure score is in valid range
    score = max(0, min(100, score))

    # Determine health level
    if score >= 90:
        level = "Excellent"
    elif score >= 70:
        level = "Good"
    elif score >= 50:
        level = "Fair"
    elif score >= 30:
        level = "Poor"
    else:
        level = "Critical"

    # Generate recommendations
    recommendations = []

    if not checkpoint['valid']:
        recommendations.append({
            'severity': 'critical',
            'category': 'checkpoint',
            'message': 'Checkpoint is invalid or corrupted',
            'action': 'Run with --fresh to start fresh or restore from backup',
        })

    if checkpoint.get('is_stale'):
        recommendations.append({
            'severity': 'warning',
            'category': 'checkpoint',
            'message': f"Checkpoint is stale ({checkpoint['age_hours']:.1f} hours old)",
            'action': 'Resume from checkpoint or run fresh if needed',
        })

    if missing_files.get('missing_voiceover'):
        recommendations.append({
            'severity': 'critical',
            'category': 'files',
            'message': 'Voiceover file is missing',
            'action': 'Add voiceover file (SRT, MP3, WAV, or MP4) to voiceover/ directory',
        })

    if not config.get('valid'):
        recommendations.append({
            'severity': 'critical',
            'category': 'config',
            'message': 'Config validation failed',
            'action': 'Fix config.yaml errors',
        })

    if not readiness.get('ready'):
        recommendations.append({
            'severity': 'warning',
            'category': 'readiness',
            'message': 'Project is not ready to run',
            'action': 'Add required files or restore checkpoint',
        })

    if has_old_cache:
        recommendations.append({
            'severity': 'info',
            'category': 'cache',
            'message': 'Old cache files detected',
            'action': 'Consider clearing cache with --fresh to reclaim space',
        })

    if config.get('warnings'):
        for warning in config['warnings']:
            recommendations.append({
                'severity': 'info',
                'category': 'config',
                'message': warning,
                'action': 'Review config.yaml for recommended changes',
            })

    return {
        'score': score,
        'level': level,
        'recommendations': recommendations,
    }


def generate_report(project_path: str, verbose: bool = False) -> Dict[str, Any]:
    """Generate comprehensive health report for a project."""

    project = Path(project_path)
    verbosity = get_verbosity()

    if verbosity >= 1:
        print_header(f"Project Health Report: {project.name}")

    # Run all checks
    if verbosity >= 1:
        print_info("Checking checkpoint...")

    checkpoint_result = check_checkpoint(project)

    if verbosity >= 1:
        print_info("Analyzing cache...")

    cache_result = check_cache(project)

    if verbosity >= 1:
        print_info("Checking for missing files...")

    missing_files_result = check_missing_files(project)

    if verbosity >= 1:
        print_info("Validating config...")

    config_result = check_config(project)

    if verbosity >= 1:
        print_info("Assessing pipeline readiness...")

    readiness_result = check_pipeline_readiness(project)

    # Calculate health score
    health_score = calculate_health_score(
        checkpoint_result,
        cache_result,
        missing_files_result,
        config_result,
        readiness_result,
    )

    # Compile full report
    report = {
        'project': {
            'name': project.name,
            'path': str(project),
            'generated_at': datetime.now().isoformat(),
        },
        'checkpoint': checkpoint_result,
        'cache': cache_result,
        'missing_files': missing_files_result,
        'config': config_result,
        'readiness': readiness_result,
        'health': health_score,
    }

    return report


def display_report(report: Dict[str, Any], verbose: bool = False) -> None:
    """Display health report in human-readable format."""

    verbosity = get_verbosity()

    # Health Score Summary
    health = report['health']
    print(f"\n  Health Score: {health['score']}/100 ({health['level']})")

    # Checkpoint Status
    cp = report['checkpoint']
    print(f"\n  Checkpoint:")
    if cp['valid']:
        print_ok(f"  Valid - Last stage: {cp['last_completed_stage'] or 'None'}")
    else:
        print_error(f"  Invalid - {cp['issues'][0] if cp['issues'] else 'Unknown error'}")

    if cp.get('age_hours', 0) > 0:
        stale_msg = " (STALE)" if cp.get('is_stale') else ""
        print(f"    Age: {cp['age_hours']:.1f} hours{stale_msg}")

    if cp.get('completed_stages'):
        print(f"    Completed: {', '.join(cp['completed_stages'])}")
    if cp.get('remaining_stages'):
        print(f"    Remaining: {', '.join(cp['remaining_stages'])}")

    # Cache Status
    cache = report['cache']
    print(f"\n  Cache:")
    print(f"    Total size: {cache['total_size_formatted']}")
    for cache_info in cache.get('caches', []):
        stale_msg = " (STALE)" if cache_info.get('is_stale') else ""
        age_str = f" - {cache_info['age_hours']:.1f}h old" if cache_info.get('age_hours') else ""
        print(f"    {cache_info['name']}: {cache_info['size_formatted']}{age_str}{stale_msg}")

    # Missing Files
    mf = report['missing_files']
    print(f"\n  Files:")
    if mf['issues']:
        for issue in mf['issues']:
            print_warn(f"  {issue}")
    else:
        print_ok("  All expected files present")

    # Config Status
    cfg = report['config']
    print(f"\n  Config:")
    if cfg['valid']:
        print_ok("  Valid")
        if cfg.get('project_name'):
            print(f"    Project: {cfg['project_name']}")
    else:
        print_error(f"  Invalid - {cfg['issues'][0] if cfg['issues'] else 'Unknown error'}")

    if cfg.get('warnings'):
        for warning in cfg['warnings']:
            print_warn(f"  {warning}")

    # Readiness
    rd = report['readiness']
    print(f"\n  Pipeline Ready: {'Yes' if rd['ready'] else 'No'}")
    if rd.get('can_resume'):
        print("    Can resume from checkpoint")
    if rd.get('can_run_fresh'):
        print("    Can run fresh")

    # Recommendations
    recommendations = health.get('recommendations', [])
    if recommendations:
        print(f"\n  Recommendations:")
        for rec in recommendations:
            severity_tag = rec.get('severity', 'info').upper()
            print(f"    [{severity_tag}] {rec.get('message', '')}")
            if verbosity >= 2:
                print(f"      Action: {rec.get('action', '')}")


def main():
    parser = argparse.ArgumentParser(
        description='Project Health Report Generator',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        '--project', '-p',
        type=str,
        required=True,
        help='Path to project directory'
    )

    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Output report to JSON file'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )

    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Quiet mode: only show errors'
    )

    args = parser.parse_args()

    # Set verbosity
    if args.quiet:
        set_verbosity(0)
    elif args.verbose:
        set_verbosity(2)
    else:
        set_verbosity(1)

    # Generate report
    report = generate_report(args.project, verbose=args.verbose)

    # Display report
    display_report(report, verbose=args.verbose)

    # Save to file if requested
    if args.output:
        output_path = Path(args.output)
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            print(f"\n  Report saved to: {output_path}")
        except Exception as e:
            print_error(f"Failed to save report: {e}")

    # Exit with appropriate code
    health = report['health']
    if health['score'] < 30:
        sys.exit(2)  # Critical
    elif health['score'] < 50:
        sys.exit(1)  # Poor
    else:
        sys.exit(0)  # Good or better


if __name__ == '__main__':
    main()
