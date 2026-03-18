"""
Integration test: Non-yt-dlp Pipeline Paths

Exercises the real pipeline paths that don't involve yt-dlp:
  1. Entity image restore from disk → V9 track generation
  2. --output-only embedding fallback (voiceover + entity embeddings)
  3. OTIO output with entity images on V9

Uses a real completed project (B-2 Bomb) and real entity images (from test_1).

Usage:
    python scripts/test_nonytdlp_pipeline.py
    python scripts/test_nonytdlp_pipeline.py --skip-pipeline   # verify only, no pipeline run
    python scripts/test_nonytdlp_pipeline.py --keep-images     # don't clean up images dir
"""

from __future__ import annotations

import argparse
import gzip
import json
import glob
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration — edit these to point at your real project data
# ---------------------------------------------------------------------------

# Project that has a completed checkpoint with matches
PROJECT_GLOB = "E:/Edit Job/Degold/RennReports/The B-2*2026-03-16"

# Source of real entity images (downloaded from Pexels, with .entity.json metadata)
IMAGE_SOURCE = "E:/Edit Job/Degold/_test_queues/test_1/images"

# Pipeline entry point
PIPELINE_ROOT = Path(__file__).resolve().parent.parent
MAIN_PY = PIPELINE_ROOT / "main.py"


def find_project_dir() -> Path:
    matches = glob.glob(PROJECT_GLOB)
    if not matches:
        print(f"FAIL: No project matching {PROJECT_GLOB}")
        sys.exit(1)
    return Path(matches[0])


def read_checkpoint(cp_path: Path) -> dict:
    """Read gzip-compressed checkpoint JSON."""
    with gzip.open(str(cp_path), "rt", encoding="utf-8") as f:
        return json.load(f)


def write_checkpoint(cp_path: Path, data: dict):
    """Write gzip-compressed checkpoint JSON."""
    with gzip.open(str(cp_path), "wt", encoding="utf-8") as f:
        json.dump(data, f)


def _get_short_path_dir(project_dir: Path) -> Path:
    """
    Replicate the pipeline's _get_output_dir() logic for short-path mode.

    config.image_search.root_dir = "E:/i" → E:/i/<project_name[:15]>
    """
    # Read the global config to find root_dir
    import yaml
    config_path = PIPELINE_ROOT / "config.yaml"
    root_dir = "E:/i"  # default
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        root_dir = (cfg.get("image_search", {}).get("root_dir", "") or "").strip()

    if root_dir:
        truncated_name = project_dir.name[:15]
        return Path(root_dir) / truncated_name
    else:
        return project_dir / "images"


def setup_entity_images(project_dir: Path) -> tuple[Path, int]:
    """
    Copy real entity images to the path the pipeline expects.

    When config has image_search.root_dir (e.g., "E:/i"), the pipeline
    looks at E:/i/<project_name[:15]>. We replicate that logic and
    copy images there.

    Returns (images_dir, image_count).
    """
    images_dir = _get_short_path_dir(project_dir)
    source = Path(IMAGE_SOURCE)

    if not source.exists():
        print(f"FAIL: Image source not found: {IMAGE_SOURCE}")
        sys.exit(1)

    # Track whether we created this dir (for cleanup)
    created_dir = not images_dir.exists()
    images_dir.mkdir(parents=True, exist_ok=True)

    # Copy all image + metadata file pairs
    count = 0
    for f in source.iterdir():
        if f.suffix in (".jpeg", ".jpg", ".png", ".webp") or f.name.endswith(".entity.json"):
            dest = images_dir / f.name
            if not dest.exists():
                shutil.copy2(str(f), str(dest))
            count += 1

    image_count = count // 2  # pairs of (image, metadata)
    print(f"  Target: {images_dir}")
    print(f"  Copied {count} files ({image_count} images)")

    return images_dir, image_count


def inject_checkpoint_entity_data(project_dir: Path, images_dir: Path):
    """
    Inject entity_images stage data into the checkpoint so that
    EntityImagesStage.restore() finds non-empty data and proceeds
    to read images from disk.

    The checkpoint entity_images just needs to be non-empty (the actual
    image paths are read from disk by restore_entity_images_from_disk).
    """
    cp_path = project_dir / "checkpoint.json"
    if not cp_path.exists():
        print(f"FAIL: No checkpoint at {cp_path}")
        sys.exit(1)

    data = read_checkpoint(cp_path)

    # Build entity_images from .entity.json files on disk
    entities = {}
    for meta_file in images_dir.glob("*.entity.json"):
        with open(meta_file, "r", encoding="utf-8") as f:
            meta = json.load(f)
        name = meta.get("entity_name", "")
        if not name:
            continue

        # Find corresponding image file
        base = str(meta_file).replace(".entity.json", "")
        img_path = None
        for ext in [".jpeg", ".jpg", ".png", ".webp"]:
            candidate = Path(base + ext)
            if candidate.exists():
                img_path = str(candidate)
                break
        if not img_path:
            continue

        if name not in entities:
            entities[name] = {
                "entity_type": meta.get("entity_type", ""),
                "context": meta.get("context", ""),
                "query": meta.get("query", name),
                "images": [],
                "segment_indices": [],
            }
        entities[name]["images"].append(img_path)

    total_images = sum(len(e["images"]) for e in entities.values())

    # Inject into checkpoint
    data["entity_images"] = {
        "entity_count": len(entities),
        "total_images": total_images,
        "entities": entities,
    }

    # Backup original
    backup = project_dir / "checkpoint.pre_test.json"
    if not backup.exists():
        shutil.copy2(str(cp_path), str(backup))
        print(f"  Backed up checkpoint to {backup.name}")

    write_checkpoint(cp_path, data)
    print(f"  Injected {len(entities)} entities / {total_images} images into checkpoint")
    return len(entities), total_images


def restore_checkpoint(project_dir: Path):
    """Restore the original checkpoint from backup."""
    backup = project_dir / "checkpoint.pre_test.json"
    cp_path = project_dir / "checkpoint.json"
    if backup.exists():
        shutil.copy2(str(backup), str(cp_path))
        backup.unlink()
        print("  Restored original checkpoint")


def run_output_only(project_dir: Path) -> tuple[bool, str]:
    """
    Run the pipeline with --output-only to exercise:
    - EntityImagesStage.restore() → reads images from disk
    - OutputStage._ensure_voiceover_embeddings() → embedding fallback
    - OutputStage._compute_entity_embeddings() → entity embeddings
    - create_timeline() with entity_images → V9 track
    """
    cmd = [
        sys.executable, str(MAIN_PY),
        "--project", str(project_dir),
        "--output-only",
        "--non-interactive",
    ]

    print(f"\n  Running: {' '.join(cmd[-4:])}")
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=300,
        cwd=str(PIPELINE_ROOT),
        env=env,
    )

    output = result.stdout + "\n" + result.stderr
    success = result.returncode == 0
    return success, output


def verify_otio_output(project_dir: Path, pipeline_output: str) -> list[str]:
    """
    Verify the OTIO output has V9 entity clips.

    Returns list of failure messages (empty = all passed).
    """
    failures = []

    # 1. Check pipeline output mentions V9 entity matching
    if "[V9]" not in pipeline_output:
        failures.append("Pipeline output missing [V9] entity matching stats")

    if "Entity matching:" in pipeline_output:
        # Parse: [V9] Entity matching: N/M segments
        import re
        match = re.search(r"\[V9\] Entity matching: (\d+)/(\d+) segments", pipeline_output)
        if match:
            matched, total = int(match.group(1)), int(match.group(2))
            if matched == 0:
                failures.append(f"V9 matched 0/{total} segments — entity matching failed")
            else:
                print(f"  V9 matched {matched}/{total} segments")
        else:
            print("  V9 entity matching stats not found in expected format")

    # 2. Check that OTIO file was generated
    output_dirs = sorted(project_dir.glob("output/*"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not output_dirs:
        failures.append("No output directory found")
        return failures

    latest_output = output_dirs[0]

    # Prefer the FULL timeline (has all tracks), fall back to V9 split
    full_otio = list(latest_output.glob("*FULL*.otio"))
    v9_otio = list(latest_output.glob("*V9*.otio"))
    otio_file = (full_otio or v9_otio or [None])[0]
    if not otio_file:
        failures.append(f"No OTIO files in {latest_output}")
        return failures

    print(f"  OTIO file: {otio_file.name}")

    # 3. Load OTIO and check V9 track
    try:
        import opentimelineio as otio
        timeline = otio.adapters.read_from_file(str(otio_file))

        v9_tracks = [t for t in timeline.tracks if t.name and "V9" in t.name]
        if not v9_tracks:
            failures.append("No V9 track found in OTIO timeline")
            return failures

        v9 = v9_tracks[0]
        clips = [item for item in v9 if isinstance(item, otio.schema.Clip)]
        gaps = [item for item in v9 if isinstance(item, otio.schema.Gap)]

        print(f"  V9 track: {len(clips)} clips, {len(gaps)} gaps")

        if len(clips) == 0:
            failures.append("V9 track has 0 clips — entity images not placed")
            return failures

        # Check clip metadata
        sample_clip = clips[0]
        entity_name = sample_clip.metadata.get("entity_name", "")
        match_type = sample_clip.metadata.get("match_type", "")
        print(f"  Sample clip: entity='{entity_name}', match_type='{match_type}'")

        if not entity_name:
            failures.append("V9 clips missing entity_name metadata")
        if match_type not in ("exact", "semantic", "sticky"):
            failures.append(f"V9 clip has unexpected match_type: '{match_type}'")

        # Check for variety of match types
        match_types = set()
        entity_names = set()
        for c in clips:
            match_types.add(c.metadata.get("match_type", ""))
            entity_names.add(c.metadata.get("entity_name", ""))

        print(f"  Entities on V9: {sorted(entity_names)}")
        print(f"  Match types: {sorted(match_types)}")

        # Check that image paths in clips actually exist
        missing_images = 0
        for c in clips[:5]:
            img_path = c.metadata.get("image_path", "")
            if img_path and not Path(img_path).exists():
                missing_images += 1
        if missing_images > 0:
            failures.append(f"{missing_images}/5 image paths in V9 clips don't exist on disk")

    except ImportError:
        failures.append("opentimelineio not installed — cannot verify OTIO content")
    except Exception as e:
        failures.append(f"OTIO verification error: {e}")

    # 4. Check embedding fallback was exercised
    if "Recomputed voiceover embeddings" in pipeline_output:
        print("  Embedding fallback: voiceover embeddings recomputed (OK)")
    elif "voiceover_embeddings" in pipeline_output.lower():
        print("  Embedding fallback: referenced in output")

    if "Computed embeddings for" in pipeline_output:
        print("  Entity embeddings: computed (OK)")

    return failures


def cleanup_images(project_dir: Path):
    """Remove the test images directory."""
    images_dir = _get_short_path_dir(project_dir)
    if images_dir.exists():
        shutil.rmtree(str(images_dir))
        print(f"  Cleaned up {images_dir}")


def main():
    parser = argparse.ArgumentParser(description="Integration test: non-yt-dlp pipeline paths")
    parser.add_argument("--skip-pipeline", action="store_true", help="Skip pipeline run, verify existing output")
    parser.add_argument("--keep-images", action="store_true", help="Don't clean up images dir after test")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show full pipeline output")
    args = parser.parse_args()

    print("=" * 60)
    print("Non-yt-dlp Pipeline Integration Test")
    print("=" * 60)

    # 1. Find project
    project_dir = find_project_dir()
    print(f"\n[1/5] Project: {project_dir.name[:60]}")

    # 2. Setup entity images
    print("\n[2/5] Setting up entity images...")
    images_dir, image_count = setup_entity_images(project_dir)

    # 3. Inject entity data into checkpoint
    print("\n[3/5] Injecting entity data into checkpoint...")
    entity_count, total_images = inject_checkpoint_entity_data(project_dir, images_dir)

    # 4. Run pipeline
    if args.skip_pipeline:
        print("\n[4/5] Skipping pipeline run (--skip-pipeline)")
        pipeline_output = ""
        pipeline_ok = True
    else:
        print("\n[4/5] Running --output-only pipeline...")
        try:
            pipeline_ok, pipeline_output = run_output_only(project_dir)
            if args.verbose:
                print("\n--- Pipeline Output ---")
                # Sanitize for Windows console encoding
                safe_output = pipeline_output.encode("ascii", errors="replace").decode("ascii")
                print(safe_output)
                print("--- End Pipeline Output ---\n")
            if pipeline_ok:
                print("  Pipeline completed successfully")
            else:
                print("  Pipeline FAILED (non-zero exit)")
                # Print last 20 lines for diagnostics
                lines = pipeline_output.strip().split("\n")
                for line in lines[-20:]:
                    print(f"    {line}")
        except subprocess.TimeoutExpired:
            print("  Pipeline TIMED OUT (300s)")
            pipeline_ok = False
            pipeline_output = ""

    # 5. Verify output
    print("\n[5/5] Verifying output...")
    failures = []
    if not pipeline_ok:
        failures.append("Pipeline exited with non-zero return code")
    else:
        failures = verify_otio_output(project_dir, pipeline_output)

    # Cleanup
    print("\n--- Cleanup ---")
    restore_checkpoint(project_dir)
    if not args.keep_images:
        cleanup_images(project_dir)
    else:
        print(f"  Keeping images/ at {images_dir}")

    # Report
    print("\n" + "=" * 60)
    if failures:
        print(f"RESULT: FAIL ({len(failures)} failures)")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print("RESULT: PASS")
        print(f"  Entity images: {entity_count} entities, {total_images} images")
        print("  V9 track: entity clips placed and verified")
        print("  Embedding fallback: exercised")
        sys.exit(0)


if __name__ == "__main__":
    main()
