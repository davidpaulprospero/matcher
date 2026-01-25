#!/usr/bin/env python3
"""
Caption Test Fixtures Health Check Script (US-010)

Validates that YouTube video fixtures used in integration tests still have
the expected caption states. Run this periodically or before releases to
ensure tests won't fail due to video/caption changes on YouTube.

Usage:
    python scripts/check_caption_fixtures.py
    python scripts/check_caption_fixtures.py --verbose
    python scripts/check_caption_fixtures.py --output report.json

Exit codes:
    0 - All fixtures valid
    1 - One or more fixtures need attention

Can also be run in CI via GitHub Actions (see .github/workflows/caption-fixtures-check.yml).
"""

import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to path
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Fix Windows console encoding
import io
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')


# Import VIDEO_FIXTURES from test file
def load_video_fixtures() -> Dict[str, Any]:
    """Load VIDEO_FIXTURES from the integration test file."""
    test_file = PROJECT_ROOT / "tests" / "test_caption_integration.py"

    if not test_file.exists():
        raise FileNotFoundError(f"Test file not found: {test_file}")

    # Read and execute just the fixture definition
    # This is safer than importing the whole test module
    content = test_file.read_text(encoding='utf-8')

    # Find VIDEO_FIXTURES definition
    import re
    match = re.search(
        r'^VIDEO_FIXTURES\s*=\s*\{',
        content,
        re.MULTILINE
    )
    if not match:
        raise ValueError("Could not find VIDEO_FIXTURES in test file")

    # Extract the dict - find matching brace
    start = match.start()
    brace_count = 0
    end = start
    in_string = False
    escape_next = False

    for i, char in enumerate(content[start:], start):
        if escape_next:
            escape_next = False
            continue
        if char == '\\':
            escape_next = True
            continue
        if char == '"' or char == "'":
            # Simple string detection (doesn't handle all edge cases)
            if not in_string:
                in_string = char
            elif in_string == char:
                in_string = False
            continue
        if in_string:
            continue
        if char == '{':
            brace_count += 1
        elif char == '}':
            brace_count -= 1
            if brace_count == 0:
                end = i + 1
                break

    fixture_code = content[start:end]

    # Execute in isolated namespace
    namespace: Dict[str, Any] = {}
    exec(fixture_code, namespace)

    return namespace['VIDEO_FIXTURES']


@dataclass
class FixtureCheckResult:
    """Result of checking a single fixture."""
    fixture_key: str
    video_id: str
    expected_state: str  # 'has_captions', 'no_captions', etc.
    actual_state: str
    is_valid: bool
    details: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    suggested_action: Optional[str] = None


def check_fixture(
    fetcher: Any,
    fixture_key: str,
    fixture_data: Dict[str, Any],
    verbose: bool = False
) -> FixtureCheckResult:
    """Check a single fixture video's caption state.

    Args:
        fetcher: CaptionFetcher instance
        fixture_key: Key in VIDEO_FIXTURES dict
        fixture_data: Fixture configuration dict
        verbose: Whether to print verbose output

    Returns:
        FixtureCheckResult with validation results
    """
    video_id = fixture_data["id"]
    notes = fixture_data.get("notes", "")
    expected_languages = fixture_data.get("expected_languages", [])

    # Determine expected state from fixture key
    if "no_captions" in fixture_key:
        expected_state = "no_captions"
    elif "with_captions" in fixture_key or "backup" in fixture_key or "auto" in fixture_key:
        expected_state = "has_captions"
    else:
        expected_state = "unknown"

    result = FixtureCheckResult(
        fixture_key=fixture_key,
        video_id=video_id,
        expected_state=expected_state,
        actual_state="unknown",
        is_valid=False,
        details={"notes": notes}
    )

    try:
        # Check if video has captions
        languages = fetcher.list_available_languages(video_id)

        has_captions = len(languages) > 0
        result.actual_state = "has_captions" if has_captions else "no_captions"

        # Store language details
        result.details["available_languages"] = [
            {
                "code": lang.code,
                "name": lang.name,
                "is_auto_generated": lang.is_auto_generated
            }
            for lang in languages
        ]
        result.details["language_count"] = len(languages)

        # Check if expected languages are available
        if expected_languages:
            available_codes = [lang.code for lang in languages]
            missing = [lang for lang in expected_languages if lang not in available_codes]
            if missing:
                result.details["missing_expected_languages"] = missing

        # Determine validity
        if expected_state == "has_captions":
            result.is_valid = has_captions
            if not has_captions:
                result.suggested_action = (
                    f"Video {video_id} no longer has captions. "
                    f"Find a replacement video with captions and update "
                    f"VIDEO_FIXTURES['{fixture_key}']['id']."
                )
        elif expected_state == "no_captions":
            result.is_valid = not has_captions
            if has_captions:
                result.suggested_action = (
                    f"Video {video_id} now has captions ({len(languages)} languages). "
                    f"Find a replacement video without captions and update "
                    f"VIDEO_FIXTURES['{fixture_key}']['id']."
                )
        else:
            # Unknown expected state - just report what we found
            result.is_valid = True

    except Exception as e:
        error_str = str(e).lower()

        # Check for video unavailability
        if any(term in error_str for term in ["unavailable", "private", "deleted", "not found", "sign in"]):
            result.actual_state = "video_unavailable"
            result.error = str(e)
            result.is_valid = False
            result.suggested_action = (
                f"Video {video_id} is no longer available on YouTube. "
                f"Find a replacement video and update VIDEO_FIXTURES['{fixture_key}']['id']. "
                f"Notes: {notes}"
            )
        else:
            result.actual_state = "error"
            result.error = str(e)
            result.is_valid = False
            result.suggested_action = f"Check if video {video_id} is accessible: {str(e)[:100]}"

    return result


def run_health_check(verbose: bool = False) -> List[FixtureCheckResult]:
    """Run health check on all video fixtures.

    Args:
        verbose: Whether to print verbose output

    Returns:
        List of FixtureCheckResult for each fixture
    """
    # Import here to avoid import errors if dependencies missing
    from src.caption_fetcher import CaptionFetcher

    # Load fixtures
    fixtures = load_video_fixtures()

    if verbose:
        print(f"Loaded {len(fixtures)} video fixtures from tests/test_caption_integration.py")
        print()

    # Create fetcher
    fetcher = CaptionFetcher()

    results = []
    for fixture_key, fixture_data in fixtures.items():
        if verbose:
            print(f"Checking {fixture_key} ({fixture_data['id']})...", end=" ", flush=True)

        result = check_fixture(fetcher, fixture_key, fixture_data, verbose)
        results.append(result)

        if verbose:
            status = "OK" if result.is_valid else "NEEDS ATTENTION"
            print(f"{status}")
            if not result.is_valid and result.suggested_action:
                print(f"  -> {result.suggested_action}")
            print()

    return results


def print_summary(results: List[FixtureCheckResult]) -> None:
    """Print a summary of check results."""
    valid_count = sum(1 for r in results if r.is_valid)
    total_count = len(results)

    print("=" * 60)
    print("CAPTION FIXTURES HEALTH CHECK SUMMARY")
    print("=" * 60)
    print()
    print(f"Total fixtures: {total_count}")
    print(f"Valid:          {valid_count}")
    print(f"Need attention: {total_count - valid_count}")
    print()

    if all(r.is_valid for r in results):
        print("All fixtures are valid!")
    else:
        print("FIXTURES NEEDING ATTENTION:")
        print("-" * 40)
        for result in results:
            if not result.is_valid:
                print(f"\n{result.fixture_key} ({result.video_id}):")
                print(f"  Expected: {result.expected_state}")
                print(f"  Actual:   {result.actual_state}")
                if result.error:
                    print(f"  Error:    {result.error[:100]}")
                if result.suggested_action:
                    print(f"  Action:   {result.suggested_action}")


def export_report(results: List[FixtureCheckResult], output_path: Path) -> None:
    """Export check results to a JSON file."""
    report = {
        "timestamp": datetime.now().isoformat(),
        "summary": {
            "total": len(results),
            "valid": sum(1 for r in results if r.is_valid),
            "needs_attention": sum(1 for r in results if not r.is_valid),
        },
        "results": [
            {
                "fixture_key": r.fixture_key,
                "video_id": r.video_id,
                "expected_state": r.expected_state,
                "actual_state": r.actual_state,
                "is_valid": r.is_valid,
                "details": r.details,
                "error": r.error,
                "suggested_action": r.suggested_action,
            }
            for r in results
        ]
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)

    print(f"Report saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Check health of caption test fixtures",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/check_caption_fixtures.py
    python scripts/check_caption_fixtures.py --verbose
    python scripts/check_caption_fixtures.py --output fixture_report.json
        """
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print verbose output during checks"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        help="Export results to JSON file"
    )

    args = parser.parse_args()

    try:
        results = run_health_check(verbose=args.verbose)

        if not args.verbose:
            print_summary(results)

        if args.output:
            export_report(results, args.output)

        # Exit code based on results
        all_valid = all(r.is_valid for r in results)
        sys.exit(0 if all_valid else 1)

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error running health check: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
