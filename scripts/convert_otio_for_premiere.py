"""Convert OTIO from Clip.2 (multi-ref) to Clip.1 (single-ref) for Premiere Pro."""
import json
import copy
import sys


def downgrade_clip(item):
    """Convert Clip.2 -> Clip.1 by flattening media_references to media_reference."""
    if item.get("OTIO_SCHEMA") == "Clip.2":
        item["OTIO_SCHEMA"] = "Clip.1"

        # Move active media reference from plural to singular
        refs = item.pop("media_references", {})
        active_key = item.pop("active_media_reference_key", "DEFAULT_MEDIA")

        if refs and active_key in refs:
            item["media_reference"] = refs[active_key]
        elif refs:
            # Fallback to first available ref
            item["media_reference"] = next(iter(refs.values()))
        else:
            item["media_reference"] = {
                "OTIO_SCHEMA": "MissingReference.1",
                "metadata": {},
                "name": "",
                "available_range": None,
                "available_image_bounds": None,
            }

    return item


def clean_resolve_metadata(obj):
    """Remove DaVinci-specific metadata that might confuse Premiere."""
    if isinstance(obj, dict):
        # Remove Resolve_OTIO metadata
        meta = obj.get("metadata", {})
        if isinstance(meta, dict):
            meta.pop("Resolve_OTIO", None)

        # Recurse into children
        for key in ("children", "tracks", "effects", "markers"):
            if key in obj:
                val = obj[key]
                if isinstance(val, list):
                    for child in val:
                        clean_resolve_metadata(child)
                elif isinstance(val, dict):
                    clean_resolve_metadata(val)

        # Handle media_reference
        if "media_reference" in obj:
            clean_resolve_metadata(obj["media_reference"])

    return obj


def convert_otio_for_premiere(input_path, output_path):
    """Convert an OTIO file for Premiere Pro compatibility."""
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Downgrade all clips from Clip.2 to Clip.1
    tracks = data.get("tracks", {}).get("children", [])
    clip_count = 0
    ref_count = 0

    for track in tracks:
        for i, item in enumerate(track.get("children", [])):
            if item.get("OTIO_SCHEMA", "").startswith("Clip"):
                downgrade_clip(item)
                clip_count += 1
                if item.get("media_reference", {}).get("OTIO_SCHEMA") == "ExternalReference.1":
                    ref_count += 1

    # Clean DaVinci-specific metadata
    clean_resolve_metadata(data)

    # Write converted file
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print(f"Converted {clip_count} clips ({ref_count} with media refs)")
    print(f"Output: {output_path}")
    return output_path


if __name__ == "__main__":
    if len(sys.argv) >= 3:
        convert_otio_for_premiere(sys.argv[1], sys.argv[2])
    else:
        print("Usage: convert_otio_for_premiere.py <input.otio> <output.otio>")
