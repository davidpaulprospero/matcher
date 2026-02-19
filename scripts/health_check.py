#!/usr/bin/env python3
"""
Health Check Script for Development Environment

Validates the development environment is properly configured for the matcher pipeline.
Checks Python version, required packages, external tools, and network connectivity.

Usage:
    python scripts/health_check.py                    # Run all checks
    python scripts/health_check.py --json              # Output as JSON
    python scripts/health_check.py --verbose           # Detailed output
    python scripts/health_check.py --quick            # Skip network checks
    python scripts/health_check.py --output results.json  # Save to file
"""

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Import standardized output functions
from script_utils import print_ok, print_warn, print_error, print_info, print_header

# Add project root and scripts directory to path for imports
_script_path = os.path.abspath(__file__)
project_root = Path(_script_path).parent.parent
scripts_dir = Path(_script_path).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(scripts_dir))

# Change to project root so relative paths work correctly
os.chdir(project_root)

# Constants
MIN_PYTHON_VERSION = (3, 9)
DEFAULT_TIMEOUT = 10


def check_python_version(verbose: bool = False) -> Dict[str, Any]:
    """Check if Python version meets minimum requirement."""
    current = sys.version_info
    version_str = f"{current.major}.{current.minor}.{current.micro}"
    meets_min = current >= MIN_PYTHON_VERSION

    result = {
        "name": "Python Version",
        "status": "ok" if meets_min else "failed",
        "message": f"Python {version_str}" + ("" if meets_min else f" (minimum: {'.'.join(map(str, MIN_PYTHON_VERSION))})"),
        "details": {
            "current": version_str,
            "minimum": ".".join(map(str, MIN_PYTHON_VERSION)),
            "meets_requirement": meets_min
        }
    }

    if verbose:
        if meets_min:
            print_ok(f"Python: {version_str} (meets minimum {'.'.join(map(str, MIN_PYTHON_VERSION))})")
        else:
            print_error(f"Python: {version_str} (minimum: {'.'.join(map(str, MIN_PYTHON_VERSION))})")

    return result


def check_required_packages(verbose: bool = False) -> Dict[str, Any]:
    """Check if all required Python packages are installed."""
    # Required packages with display names
    required = {
        'yaml': 'PyYAML',
        'yaml': 'pyyaml',
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
    result = {
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

    if verbose:
        print_header("REQUIRED PACKAGES")
        for name in installed:
            version = versions.get(name, "unknown")
            print_ok(f"  {name}: {version}")
        for name in missing:
            print_error(f"  {name}: NOT INSTALLED")

    return result


def check_external_tools(verbose: bool = False) -> Dict[str, Any]:
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
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=DEFAULT_TIMEOUT
            )
            if result.returncode == 0:
                # Extract version
                first_line = result.stdout.split('\n')[0]
                if tool_name == 'ffmpeg':
                    match = __import__('re').search(r'ffmpeg version ([^ ]+)', first_line)
                    version = match.group(1) if match else "found"
                elif tool_name == 'ffprobe':
                    match = __import__('re').search(r'ffprobe version ([^ ]+)', first_line)
                    version = match.group(1) if match else "found"
                else:
                    version = result.stdout.strip()

                results[tool_name] = {"status": "ok", "version": version}
                if verbose:
                    print_ok(f"  {tool_name}: {version}")
            else:
                results[tool_name] = {"status": "failed", "error": "command failed"}
                all_available = False
                if verbose:
                    print_error(f"  {tool_name}: command failed")
        except FileNotFoundError:
            results[tool_name] = {"status": "missing", "error": "not found"}
            all_available = False
            if verbose:
                print_error(f"  {tool_name}: NOT FOUND")
        except subprocess.TimeoutExpired:
            results[tool_name] = {"status": "failed", "error": "timeout"}
            all_available = False
            if verbose:
                print_error(f"  {tool_name}: timeout")

    # ffprobe is optional
    if results.get('ffprobe', {}).get('status') == 'missing':
        all_available = results.get('ffmpeg', {}).get('status') == 'ok'

    result = {
        "name": "External Tools",
        "status": "ok" if all_available else "failed",
        "message": "All required tools available" if all_available else "Some tools missing",
        "details": results
    }

    return result


def check_network_connectivity(verbose: bool = False, quick: bool = False) -> Dict[str, Any]:
    """Check network connectivity to key endpoints."""
    # Use HTTP checks for all endpoints
    endpoints = [
        ("8.8.8.8", "Google DNS", "http"),
    ]

    if not quick:
        endpoints.extend([
            ("pypi.org", "PyPI", "http"),
            ("api.anthropic.com", "Anthropic API", "http"),
        ])

    results = {}
    all_reachable = True

    for host, name, check_type in endpoints:
        try:
            # Use socket connection to port 53 (DNS) for basic connectivity
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(DEFAULT_TIMEOUT)
            s.connect((host, 53))
            s.close()
            results[host] = {"status": "ok", "name": name, "method": "dns_tcp"}
            if verbose:
                print_ok(f"  {name} ({host}): reachable")
        except socket.timeout:
            results[host] = {"status": "failed", "name": name, "error": "timeout"}
            all_reachable = False
            if verbose:
                print_warn(f"  {name} ({host}): timeout")
        except Exception as e:
            results[host] = {"status": "failed", "name": name, "error": str(e)}
            all_reachable = False
            if verbose:
                print_warn(f"  {name} ({host}): {e}")

    # HTTP checks for web endpoints (non-quick mode)
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
                    capture_output=True,
                    text=True,
                    timeout=DEFAULT_TIMEOUT + 5
                )
                if result.returncode == 0 and result.stdout.strip() in ['200', '301', '302']:
                    results[host] = {"status": "ok", "name": name, "http_code": result.stdout.strip()}
                    if verbose:
                        print_ok(f"  {name} ({host}): {result.stdout.strip()}")
                else:
                    results[host] = {"status": "failed", "name": name, "error": "unreachable"}
                    all_reachable = False
                    if verbose:
                        print_warn(f"  {name} ({host}): unreachable")
            except Exception as e:
                results[host] = {"status": "failed", "name": name, "error": str(e)}
                all_reachable = False
                if verbose:
                    print_warn(f"  {name} ({host}): {e}")

    # For quick mode, we only check DNS which is always expected to work
    if quick:
        all_reachable = all(r.get("status") == "ok" for r in results.values())

    result = {
        "name": "Network Connectivity",
        "status": "ok" if all_reachable else "warning",
        "message": "Network reachable" if all_reachable else "Some endpoints unreachable",
        "details": results
    }

    return result


def check_disk_space(verbose: bool = False) -> Dict[str, Any]:
    """Check available disk space."""
    try:
        import shutil
        total, used, free = shutil.disk_usage(project_root)

        free_gb = free / (1024 ** 3)
        total_gb = total / (1024 ** 3)
        used_percent = (used / total) * 100

        is_ok = free_gb >= 10  # At least 10GB free

        result = {
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

        if verbose:
            if is_ok:
                print_ok(f"  {free_gb:.1f} GB free ({used_percent:.1f}% used)")
            else:
                print_warn(f"  Only {free_gb:.1f} GB free ({used_percent:.1f}% used) - consider freeing space")

        return result
    except Exception as e:
        return {
            "name": "Disk Space",
            "status": "warning",
            "message": f"Could not check disk space: {e}",
            "details": {"error": str(e)}
        }


def run_health_checks(verbose: bool = False, quick: bool = False) -> Dict[str, Any]:
    """Run all health checks and return results."""
    results: List[Dict[str, Any]] = []

    if verbose:
        print_header("HEALTH CHECKS")

    # Run checks
    results.append(check_python_version(verbose))
    results.append(check_required_packages(verbose))
    results.append(check_external_tools(verbose))
    results.append(check_disk_space(verbose))
    results.append(check_network_connectivity(verbose, quick))

    # Summary
    ok_count = sum(1 for r in results if r["status"] == "ok")
    warning_count = sum(1 for r in results if r["status"] == "warning")
    failed_count = sum(1 for r in results if r["status"] == "failed")

    overall_status = "ok" if failed_count == 0 else "failed"

    return {
        "status": overall_status,
        "summary": {
            "total": len(results),
            "ok": ok_count,
            "warning": warning_count,
            "failed": failed_count
        },
        "checks": results,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    }


def main():
    parser = argparse.ArgumentParser(
        description="Health check script for development environment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/health_check.py
  python scripts/health_check.py --verbose
  python scripts/health_check.py --json
  python scripts/health_check.py --quick --output health.json
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

    args = parser.parse_args()

    # Run health checks
    results = run_health_checks(verbose=args.verbose, quick=args.quiet)

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
        # Human-readable summary
        if not args.quiet:
            print_header("HEALTH CHECK SUMMARY")
            for check in results["checks"]:
                status_icon = {
                    "ok": "[OK]",
                    "warning": "[WARN]",
                    "failed": "[ERROR]"
                }.get(check["status"], "[???]")
                print(f"  {status_icon} {check['name']}: {check['message']}")

            print()
            summary = results["summary"]
            print(f"  Results: {summary['ok']} ok, {summary['warning']} warning, {summary['failed']} failed")
            print(f"  Overall: {results['status'].upper()}")

        # Exit code
        if results["status"] == "failed":
            sys.exit(1)
        elif results["summary"]["warning"] > 0 and not args.json:
            sys.exit(0)  # Warnings don't fail
        else:
            sys.exit(0)


if __name__ == "__main__":
    main()
