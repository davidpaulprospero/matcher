"""Rename generated images with unique project prefix and update OTIO references."""
import opentimelineio as otio
from opentimelineio.opentime import RationalTime, TimeRange
from pathlib import Path
import shutil

PROJECTS = [
    {
        "path": Path(r"E:/Edit Job/Stu/WCYZKSaq-2. Smart Money Move The Complete Playbook For"),
        "prefix": "wcyzksaq",
    },
    {
        "path": Path(r"E:/Edit Job/Stu/eqIeCV9k-4. The Great Reset How The Worlds Most"),
        "prefix": "eqiecv9k",
    },
    {
        "path": Path(r"E:/Edit Job/Stu/ujKR12Dg-5. THE ESCAPE PLAN — HOW ORDINARY"),
        "prefix": "ujkr12dg",
    },
]

RATE = 30.0


def rename_project(project_path: Path, prefix: str):
    img_dir = project_path / "generated_images"
    otio_path = project_path / "generated_images.otio"

    if not img_dir.is_dir():
        print(f"SKIP: {img_dir} not found")
        return

    if not otio_path.exists():
        print(f"SKIP: {otio_path} not found")
        return

    # Collect existing PNGs
    pngs = {p: p for p in img_dir.glob("generated_*.png")}
    if not pngs:
        print(f"SKIP: no generated_*.png in {img_dir}")
        return

    print(f"\n{'='*60}")
    print(f"Project: {project_path.name}")
    print(f"Prefix:  {prefix}")
    print(f"Images:  {len(pngs)}")

    # Build old→new name map
    rename_map = {}
    for old_path in sorted(pngs.keys()):
        num = old_path.stem.removeprefix("generated_")  # "000", "001", ...
        new_name = f"{prefix}_{num}.png"
        new_path = old_path.parent / new_name
        rename_map[old_path] = new_path

    # Rename files on disk
    for old_path, new_path in rename_map.items():
        if old_path.exists():
            shutil.move(str(old_path), str(new_path))
            print(f"  RENAME: {old_path.name} -> {new_path.name}")

    # Read OTIO, update references, write back
    timeline = otio.adapters.read_from_file(str(otio_path))
    v12 = timeline.tracks[0]

    updated = 0
    for clip in v12:
        if clip.__class__.__name__ != "Clip":
            continue
        url = clip.media_reference.target_url
        old_name = Path(url).name
        if old_name.startswith("generated_"):
            num = old_name.removeprefix("generated_").removesuffix(".png")
            new_name = f"{prefix}_{num}.png"
            # Update target_url
            new_url = str(Path(url).parent / new_name).replace("\\", "/")
            clip.media_reference.target_url = new_url
            # Update media_ref name too
            clip.media_reference.name = new_name
            # Update media_references dict
            refs = clip.media_references()
            if "DEFAULT_MEDIA" in refs:
                refs["DEFAULT_MEDIA"].target_url = new_url
                refs["DEFAULT_MEDIA"].name = new_name
            # Update clip name to match
            clip.name = f"IMG:{prefix}_{num}"
            updated += 1

    # Write updated OTIO
    otio.adapters.write_to_file(timeline, str(otio_path))
    print(f"  OTIO updated: {updated} clip references")
    print(f"  Written: {otio_path}")


for proj in PROJECTS:
    rename_project(proj["path"], proj["prefix"])

print(f"\n{'='*60}")
print("Done.")
