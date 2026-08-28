"""Append the 5 lower-thirds tracks from lowerthirds.otio onto fullhere.otio.

The lower-thir lowerthirds.otio is a standalone timeline with 5 tracks
(V1 Names, V2 Dates, V3 Key Points, V4 Places, V5 Chapters). All clips are
positioned in absolute timeline time. We append those tracks to an existing
master timeline (e.g. fullhere.otio) so DaVinci Resolve sees the lower-thirds
on top of the existing video stack.

Usage:
    python scripts/merge_lowerthirds.py <master.otio> [--lowerthirds <path>] [--output <path>]

Default paths assume the project layout at E:/Edit Job/Primise/21-8/2-china-driverless-cars/.
"""

import argparse
import logging
import sys
from copy import deepcopy
from pathlib import Path

import opentimelineio as otio

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

logger = logging.getLogger("merge_lowerthirds")

PROJECT_DIR = Path("E:/Edit Job/Primise/21-8/2-china-driverless-cars")
DEFAULT_MASTER = PROJECT_DIR / "fullhere.otio"
DEFAULT_LOWERTHIRDS = PROJECT_DIR / "lowerthirds.otio"


def merge(master_path: Path, lt_path: Path, output_path: Path) -> None:
    master = otio.schema.Timeline.from_json_file(str(master_path))
    lts = otio.schema.Timeline.from_json_file(str(lt_path))

    # Track names already start with V1-V5 in lowerthirds.otio; the master has
    # V1-V7 + A17. To avoid track-name collisions in DaVinci we rename the
    # incoming lower-thirds tracks to V8-V12.
    name_renames = {
        "V1 - Lower Thirds - Names": "V8 - Lower Thirds - Names",
        "V2 - Lower Thirds - Dates": "V9 - Lower Thirds - Dates",
        "V3 - Lower Thirds - Key Points": "V10 - Lower Thirds - Key Points",
        "V4 - Lower Thirds - Places": "V11 - Lower Thirds - Places",
        "V5 - Lower Thirds - Chapters": "V12 - Lower Thirds - Chapters",
    }
    # Strip any non-ASCII characters (the source file uses the UTF-8 middle
    # dot · which can break DaVinci import) and match by prefix instead.
    prefix_to_new = [
        ("V1 - Lower Thirds", "V8 - Lower Thirds"),
        ("V2 - Lower Thirds", "V9 - Lower Thirds"),
        ("V3 - Lower Thirds", "V10 - Lower Thirds"),
        ("V4 - Lower Thirds", "V11 - Lower Thirds"),
        ("V5 - Lower Thirds", "V12 - Lower Thirds"),
    ]

    added = 0
    for lt_track in lts.tracks:
        # Deep-copy so the track can be re-parented into master without
        # tripping OTIO's "child already has a parent" guard.
        track_copy = otio.schema.Track()
        # Replace non-ASCII chars (e.g. middle dot) with hyphens so DaVinci
        # imports the track name cleanly. DaVinci has historically choked
        # on UTF-8 special chars in OTIO track names.
        clean_name = lt_track.name.encode("ascii", "replace").decode("ascii").replace("?", "-")
        track_copy.name = clean_name
        track_copy.source_range = lt_track.source_range
        for prefix, new_prefix in prefix_to_new:
            if track_copy.name.startswith(prefix):
                track_copy.name = track_copy.name.replace(prefix, new_prefix, 1)
                break
        for child in lt_track:
            track_copy.append(deepcopy(child))
        master.tracks.append(track_copy)
        added += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    otio.adapters.write_to_file(master, str(output_path))

    logger.info(f"Merged {added} lower-third tracks from {lt_path} onto {master_path}")
    logger.info(f"Output: {output_path}")
    logger.info(f"Master now has {len(master.tracks)} tracks:")
    for i, t in enumerate(master.tracks):
        clip_count = sum(1 for c in t if isinstance(c, otio.schema.Clip))
        logger.info(f"  [{i}] {t.name} - {clip_count} clips")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Append lower-thirds tracks onto an existing OTIO master timeline.",
    )
    parser.add_argument(
        "master",
        nargs="?",
        default=str(DEFAULT_MASTER),
        help=f"Master OTIO timeline path (default: {DEFAULT_MASTER}).",
    )
    parser.add_argument(
        "--lowerthirds",
        default=str(DEFAULT_LOWERTHIRDS),
        help=f"Lower-thirds OTIO timeline path (default: {DEFAULT_LOWERTHIRDS}).",
    )
    parser.add_argument(
        "--output",
        help="Output path. Default: <master_stem>_with_lowerthirds.otio alongside master.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    master_path = Path(args.master)
    lt_path = Path(args.lowerthirds)
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = master_path.with_name(f"{master_path.stem}_with_lowerthirds.otio")

    if not master_path.exists():
        logger.error(f"Master OTIO not found: {master_path}")
        return 1
    if not lt_path.exists():
        logger.error(f"Lower-thirds OTIO not found: {lt_path}")
        return 1

    merge(master_path, lt_path, output_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())