#!/usr/bin/env python3
"""
Unified Health Check Script

Central entry point for all health checks in the project. Combines:
- Environment health (Python, packages, tools, network, disk)
- Config validation (structure, API keys, paths, consistency)
- Script health (coding standards compliance)

Usage:
    python scripts/unified_health_check.py                 # Run all checks
    python scripts/unified_health_check.py --json         # JSON output for CI
    python scripts/unified_health_check.py --category env # Environment only
    python scripts/unified_health_check.py --category config  # Config only
    python scripts/unified_health_check.py --category scripts  # Script health only
    python scripts/unified_health_check.py -v             # Verbose output
"""

import argparse
import json
import os
import re
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add project root and scripts directory to path
_script_path = os.path.abspath(__file__)
project_root = Path(_script_path).parent.parent
scripts_dir = Path(_script_path).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(scripts_dir))
os.chdir(project_root)

# Import standardized output functions
from script_utils import print_header, print_ok, print_warn, print_error, print_info, set_verbosity, get_verbosity

# Constants
MIN_PYTHON_VERSION = (3, 9)
DEFAULT_TIMEOUT = 10


# =============================================================================
# CATEGORY: Environment Health Checks
# =============================================================================

def check_python_version() -> Dict[str, Any]:
    """Check if Python version meets minimum requirement."""
    current = sys.version_info
    version_str = f"{current.major}.{current.minor}.{current.micro}"
    meets_min = current >= MIN_PYTHON_VERSION

    return {
        "name": "Python Version",
        "status": "ok" if meets_min else "failed",
        "message": f"Python {version_str}" + ("" if meets_min else f" (minimum: {'.'.join(map(str, MIN_PYTHON_VERSION))})"),
        "details": {
            "current": version_str,
            "minimum": ".".join(map(str, MIN_PYTHON_VERSION)),
            "meets_requirement": meets_min
        }
    }


def check_required_packages() -> Dict[str, Any]:
    """Check if all required Python packages are installed."""
    required = {
        'yaml': 'PyYAML',
        'torch': 'torch',
        'sentence_transformers': 'sentence-transformers',
        'faster_whisper': 'faster-whisper',
        'anthropic': 'anthropic',
        'yt_dlp': 'yt-dlp',
        'cv2': 'opencv-python',
        'librosa': 'librosa',
        'imagehash': 'imagehash',
        'PIL': 'Pillow',
        'requests': 'requests',
        'dotenv': 'python-dotenv',
    }

    installed = []
    missing = []
    versions = {}

    for module_name, display_name in required.items():
        try:
            if module_name == 'yaml':
                import yaml
                versions[display_name] = yaml.__version__
            elif module_name == 'PIL':
                from PIL import Image
                versions[display_name] = Image.__version__
            elif module_name == 'yt_dlp':
                import yt_dlp
                versions[display_name] = yt_dlp.__version__
            elif module_name == 'dotenv':
                import dotenv
                versions[display_name] = dotenv.__version__
            else:
                module = __import__(module_name)
                versions[display_name] = getattr(module, '__version__', 'unknown')
            installed.append(display_name)
        except ImportError:
            missing.append(display_name)
        except Exception as e:
            versions[display_name] = f"error: {e}"

    all_installed = len(missing) == 0
    return {
        "name": "Required Packages",
        "status": "ok" if all_installed else "warning",
        "message": f"{len(installed)}/{len(required)} packages installed",
        "details": {
            "installed": installed,
            "missing": missing,
            "versions": versions,
            "total_installed": len(installed),
            "total_required": len(required)
        }
    }


def check_external_tools() -> Dict[str, Any]:
    """Check if external tools (ffmpeg, yt-dlp) are available."""
    tools = {
        'ffmpeg': ['ffmpeg', '-version'],
        'ffprobe': ['ffprobe', '-version'],
        'yt-dlp': ['yt-dlp', '--version'],
    }

    results = {}
    all_available = True

    for tool_name, cmd in tools.items():
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=DEFAULT_TIMEOUT)
            if result.returncode == 0:
                first_line = result.stdout.split('\n')[0]
                if tool_name == 'ffmpeg':
                    match = re.search(r'ffmpeg version ([^ ]+)', first_line)
                    version = match.group(1) if match else "found"
                elif tool_name == 'ffprobe':
                    match = re.search(r'ffprobe version ([^ ]+)', first_line)
                    version = match.group(1) if match else "found"
                else:
                    version = result.stdout.strip()
                results[tool_name] = {"status": "ok", "version": version}
            else:
                results[tool_name] = {"status": "failed", "error": "command failed"}
                all_available = False
        except FileNotFoundError:
            results[tool_name] = {"status": "missing", "error": "not found"}
            all_available = False
        except subprocess.TimeoutExpired:
            results[tool_name] = {"status": "failed", "error": "timeout"}
            all_available = False

    # ffprobe is optional
    if results.get('ffprobe', {}).get('status') == 'missing':
        all_available = results.get('ffmpeg', {}).get('status') == 'ok'

    return {
        "name": "External Tools",
        "status": "ok" if all_available else "failed",
        "message": "All required tools available" if all_available else "Some tools missing",
        "details": results
    }


def check_network_connectivity(quick: bool = False) -> Dict[str, Any]:
    """Check network connectivity to key endpoints."""
    endpoints = [("8.8.8.8", "Google DNS")]

    if not quick:
        endpoints.extend([
            ("pypi.org", "PyPI"),
            ("api.anthropic.com", "Anthropic API"),
        ])

    results = {}
    all_reachable = True

    for host, name in endpoints:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(DEFAULT_TIMEOUT)
            s.connect((host, 53))
            s.close()
            results[host] = {"status": "ok", "name": name, "method": "dns_tcp"}
        except socket.timeout:
            results[host] = {"status": "failed", "name": name, "error": "timeout"}
            all_reachable = False
        except Exception as e:
            results[host] = {"status": "failed", "name": name, "error": str(e)}
            all_reachable = False

    # HTTP checks for web endpoints
    if not quick:
        web_endpoints = [
            ("pypi.org", "PyPI"),
            ("api.anthropic.com", "Anthropic API"),
        ]
        for host, name in web_endpoints:
            try:
                result = subprocess.run(
                    ['curl', '-s', '-o', '/dev/null', '-w', '%{http_code}',
                     f'https://{host}', '--max-time', str(DEFAULT_TIMEOUT)],
                    capture_output=True, text=True, timeout=DEFAULT_TIMEOUT + 5
                )
                if result.returncode == 0 and result.stdout.strip() in ['200', '301', '302']:
                    results[host] = {"status": "ok", "name": name, "http_code": result.stdout.strip()}
                else:
                    results[host] = {"status": "failed", "name": name, "error": "unreachable"}
                    all_reachable = False
            except Exception as e:
                results[host] = {"status": "failed", "name": name, "error": str(e)}
                all_reachable = False

    if quick:
        all_reachable = all(r.get("status") == "ok" for r in results.values())

    return {
        "name": "Network Connectivity",
        "status": "ok" if all_reachable else "warning",
        "message": "Network reachable" if all_reachable else "Some endpoints unreachable",
        "details": results
    }


def check_disk_space() -> Dict[str, Any]:
    """Check available disk space."""
    try:
        import shutil
        total, used, free = shutil.disk_usage(project_root)

        free_gb = free / (1024 ** 3)
        total_gb = total / (1024 ** 3)
        used_percent = (used / total) * 100
        is_ok = free_gb >= 10

        return {
            "name": "Disk Space",
            "status": "ok" if is_ok else "warning",
            "message": f"{free_gb:.1f} GB free ({used_percent:.1f}% used)",
            "details": {
                "total_gb": round(total_gb, 1),
                "used_gb": round(used / (1024 ** 3), 1),
                "free_gb": round(free_gb, 1),
                "used_percent": round(used_percent, 1)
            }
        }
    except Exception as e:
        return {
            "name": "Disk Space",
            "status": "warning",
            "message": f"Could not check disk space: {e}",
            "details": {"error": str(e)}
        }


def run_environment_checks(quick: bool = False) -> List[Dict[str, Any]]:
    """Run all environment health checks."""
    results = []
    results.append(check_python_version())
    results.append(check_required_packages())
    results.append(check_external_tools())
    results.append(check_disk_space())
    results.append(check_network_connectivity(quick))
    return results


# =============================================================================
# CATEGORY: Config Validation
# =============================================================================

def check_config_structure(config: Any) -> Dict[str, Any]:
    """Validate config structure has all required sections."""
    required_sections = [
        ('project', 'ProjectConfig'),
        ('transcription', 'TranscriptionConfig'),
        ('embedding', 'EmbeddingConfig'),
        ('matching', 'MatchingConfig'),
        ('download', 'DownloadingConfig'),
        ('pipeline', 'PipelineConfig'),
    ]

    sections = []
    all_valid = True

    for section_name, _ in required_sections:
        if hasattr(config, section_name):
            sections.append({"name": section_name, "valid": True})
        else:
            sections.append({"name": section_name, "valid": False})
            all_valid = False

    return {
        "name": "Config Structure",
        "status": "ok" if all_valid else "failed",
        "message": f"{sum(1 for s in sections if s['valid'])}/{len(sections)} sections valid",
        "details": {"sections": sections}
    }


def check_config_validation(config: Any) -> Dict[str, Any]:
    """Run config's built-in validation."""
    errors = config.validate() if hasattr(config, 'validate') else []

    return {
        "name": "Config Validation",
        "status": "ok" if len(errors) == 0 else "warning",
        "message": "All checks passed" if len(errors) == 0 else f"{len(errors)} warnings",
        "details": {"errors": errors}
    }


def check_config_paths(config: Any) -> Dict[str, Any]:
    """Validate configured paths."""
    paths_to_check = [
        ('project_dir', getattr(config, 'project_dir', None)),
        ('cache', getattr(config, 'cache', None)),
    ]

    path_results = []
    all_exist = True

    for name, path in paths_to_check:
        if path is None:
            continue
        path_str = str(path) if not hasattr(path, 'cache_dir') else str(path.cache_dir)
        exists = Path(path_str).exists() if path_str else False
        path_results.append({"name": name, "path": path_str, "exists": exists})
        if not exists:
            all_exist = False

    return {
        "name": "Config Paths",
        "status": "ok" if all_exist else "warning",
        "message": "All paths valid" if all_exist else "Some paths missing",
        "details": {"paths": path_results}
    }


def run_config_checks(config_path: str = "config.yaml") -> List[Dict[str, Any]]:
    """Run all config validation checks."""
    results = []

    try:
        from src.config import load_config
        # Skip final validation for health check to allow running without API keys
        config = load_config(config_path, skip_final_validation=True)
    except Exception as e:
        return [{
            "name": "Config Loading",
            "status": "failed",
            "message": f"Failed to load config: {e}",
            "details": {"error": str(e)}
        }]

    results.append(check_config_structure(config))
    results.append(check_config_validation(config))
    results.append(check_config_paths(config))

    return results


# =============================================================================
# CATEGORY: Script Health
# =============================================================================

def check_shebang(content: str) -> Tuple[bool, str]:
    """Check if script has proper shebang."""
    lines = content.split('\n')
    if not lines:
        return False, "Empty file"
    first_line = lines[0].strip()
    if first_line == '#!/usr/bin/env python3':
        return True, ""
    elif first_line.startswith('#!'):
        return False, f"Invalid shebang: {first_line}"
    else:
        return False, "Missing shebang"


def check_docstring(content: str) -> Tuple[bool, bool, str]:
    """Check if script has docstring with usage instructions."""
    pattern = r'(?:^#!.*\n)?\s*("""|\'\'\')[\s\S]*?\1'
    match = re.search(pattern, content, re.MULTILINE)

    if not match:
        return False, False, "Missing module-level docstring"

    docstring = match.group(0)
    has_usage = 'Usage:' in docstring or 'usage:' in docstring

    return True, has_usage, ""


def check_script_utils_usage(content: str) -> Tuple[bool, str]:
    """Check if script uses script_utils for output."""
    import_pattern = r'from\s+script_utils\s+import\s+'
    if re.search(import_pattern, content):
        return True, ""
    return False, "Does not import from script_utils"


def check_argparse_usage(content: str) -> Tuple[bool, str]:
    """Check if script has argparse with --help support."""
    has_argparse_import = re.search(r'import\s+argparse', content)
    if not has_argparse_import:
        return False, "Missing argparse import"
    has_parser = re.search(r'ArgumentParser', content)
    if not has_parser:
        return False, "Missing ArgumentParser"
    has_args = re.search(r'add_argument', content)
    if not has_args:
        return False, "Missing add_argument calls"
    return True, ""


def check_single_script(script_path: Path) -> Dict[str, Any]:
    """Check a single script's health."""
    try:
        with open(script_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        return {
            "name": script_path.name,
            "status": "error",
            "message": f"Failed to read: {e}",
            "details": {"error": str(e)}
        }

    has_shebang, _ = check_shebang(content)
    has_docstring, has_usage, _ = check_docstring(content)
    uses_utils, _ = check_script_utils_usage(content)
    has_argparse, _ = check_argparse_usage(content)

    # Calculate score
    score = 0
    if has_shebang:
        score += 20
    if has_docstring:
        score += 20
    if has_usage:
        score += 20
    if uses_utils:
        score += 25
    if has_argparse:
        score += 15

    is_healthy = has_shebang and has_docstring and has_usage and uses_utils and has_argparse

    return {
        "name": script_path.name,
        "status": "ok" if is_healthy else "warning",
        "message": f"Score: {score}/100",
        "details": {
            "has_shebang": has_shebang,
            "has_docstring": has_docstring,
            "has_usage": has_usage,
            "uses_script_utils": uses_utils,
            "has_argparse": has_argparse,
            "score": score
        }
    }


def run_script_health_checks() -> List[Dict[str, Any]]:
    """Run all script health checks."""
    results = []

    # Get list of scripts to check
    script_files = []
    for f in scripts_dir.glob('*.py'):
        if f.name in ['__init__.py', 'script_utils.py']:
            continue
        if f.name.startswith('utils_'):
            continue
        # Skip this script and its variants
        if 'unified_health_check' in f.name:
            continue
        script_files.append(f)

    if not script_files:
        return [{
            "name": "Script Health",
            "status": "warning",
            "message": "No scripts found to check",
            "details": {}
        }]

    for script in sorted(script_files):
        result = check_single_script(script)
        results.append(result)

    # Summary
    healthy_count = sum(1 for r in results if r['status'] == 'ok')
    total_score = sum(r['details'].get('score', 0) for r in results)
    avg_score = int(total_score / len(results)) if results else 0

    summary = {
        "name": "Script Health Summary",
        "status": "ok" if healthy_count == len(results) else "warning",
        "message": f"{healthy_count}/{len(results)} scripts healthy (avg score: {avg_score})",
        "details": {
            "total_scripts": len(results),
            "healthy_scripts": healthy_count,
            "average_score": avg_score
        }
    }

    return [summary] + results


# =============================================================================
# Main Orchestration
# =============================================================================

def run_all_checks(category: Optional[str] = None, quick: bool = False) -> Dict[str, Any]:
    """Run all health checks based on category."""
    all_results = []
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    # Environment checks
    if category is None or category == 'env':
        env_results = run_environment_checks(quick)
        all_results.extend(env_results)

    # Config checks
    if category is None or category == 'config':
        try:
            config_results = run_config_checks()
            all_results.extend(config_results)
        except Exception as e:
            all_results.append({
                "name": "Config Validation",
                "status": "error",
                "message": str(e),
                "details": {"error": str(e)}
            })

    # Script health checks
    if category is None or category == 'scripts':
        script_results = run_script_health_checks()
        all_results.extend(script_results)

    # Calculate summary
    ok_count = sum(1 for r in all_results if r["status"] == "ok")
    warning_count = sum(1 for r in all_results if r["status"] == "warning")
    error_count = sum(1 for r in all_results if r["status"] in ("error", "failed"))

    overall_status = "ok" if error_count == 0 else ("warning" if warning_count > 0 else "failed")

    return {
        "status": overall_status,
        "summary": {
            "total": len(all_results),
            "ok": ok_count,
            "warning": warning_count,
            "failed": error_count
        },
        "checks": all_results,
        "timestamp": timestamp,
        "categories_run": category or "all"
    }


def print_human_readable(results: Dict[str, Any]) -> None:
    """Print results in human-readable format."""
    print_header("HEALTH CHECK SUMMARY")

    # Group by category
    current_category = None
    for check in results["checks"]:
        # Detect category from check name patterns
        if "Python" in check["name"] or "Packages" in check["name"] or "Tools" in check["name"] or "Network" in check["name"] or "Disk" in check["name"]:
            cat = "ENVIRONMENT"
        elif "Config" in check["name"] or "Script Health Summary" in check["name"]:
            cat = check["name"]
        else:
            cat = "OTHER"

        if cat != current_category:
            print(f"\n  [{cat}]")
            current_category = cat

        status_icon = {
            "ok": "[OK]",
            "warning": "[WARN]",
            "error": "[ERROR]",
            "failed": "[ERROR]"
        }.get(check["status"], "[???]")

        print(f"    {status_icon} {check['name']}: {check['message']}")

    print()
    summary = results["summary"]
    print(f"  Results: {summary['ok']} ok, {summary['warning']} warning, {summary['failed']} failed")
    print(f"  Overall: {results['status'].upper()}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Unified health check script - all diagnostics in one place",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/unified_health_check.py              # Run all checks
  python scripts/unified_health_check.py --json        # JSON output for CI
  python scripts/unified_health_check.py --category env  # Environment only
  python scripts/unified_health_check.py --category config  # Config only
  python scripts/unified_health_check.py --category scripts  # Scripts only
  python scripts/unified_health_check.py -v            # Verbose output
  python scripts/unified_health_check.py -o results.json  # Save to file
        """
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress non-essential output'
    )
    parser.add_argument(
        '--json', '-j',
        action='store_true',
        help='Output results as JSON'
    )
    parser.add_argument(
        '--quick', '-k',
        action='store_true',
        help='Skip network checks for faster execution'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        metavar='FILE',
        help='Save results to JSON file'
    )
    parser.add_argument(
        '--category', '-c',
        type=str,
        choices=['env', 'config', 'scripts'],
        help='Run specific category only: env (environment), config, scripts'
    )

    args = parser.parse_args()

    # Set verbosity
    if args.quiet:
        set_verbosity(0)
    elif args.verbose:
        set_verbosity(2)
    else:
        set_verbosity(1)

    # Run checks
    results = run_all_checks(category=args.category, quick=args.quick)

    # Save to file if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        if not args.json and not args.quiet:
            print_ok(f"Results saved to {args.output}")

    # Output
    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print_human_readable(results)

    # Exit code
    if results["status"] == "failed":
        sys.exit(1)
    elif results["summary"]["warning"] > 0 and not args.json:
        sys.exit(0)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
