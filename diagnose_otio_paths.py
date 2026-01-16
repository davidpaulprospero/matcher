#!/usr/bin/env python3
"""
Diagnose OTIO path issues - check what source_files are pointing to.
"""
import json
from pathlib import Path

# Read the full OTIO and extract all target_urls
otio_path = Path("E:/Edit Job/test/jan_test/audio__2026-01-01/output/20260117_020532/timeline_FULL.otio")

with open(otio_path) as f:
    otio_data = json.load(f)

# Collect all media references
media_paths = set()

def extract_media_refs(obj):
    if isinstance(obj, dict):
        if "target_url" in obj:
            media_paths.add(obj["target_url"])
        for v in obj.values():
            extract_media_refs(v)
    elif isinstance(obj, list):
        for item in obj:
            extract_media_refs(item)

extract_media_refs(otio_data)

print("OTIO Target URLs Found:")
print("=" * 80)
problem_paths = []
valid_paths = []

for path in sorted(media_paths):
    if path.startswith(('E:', 'D:', 'C:')):
        exists = Path(path).exists()
        status = "OK" if exists else "MISSING"
    else:
        status = "Unknown"
    
    if not path.startswith(('E:', 'D:', 'C:')) or not Path(path).exists():
        problem_paths.append((status, path))
    else:
        valid_paths.append((status, path))

# Print problem paths first
print("\nPROBLEM PATHS (will cause DaVinci import to fail):")
for status, path in problem_paths:
    print(f"  [{status}] {path}")

print("\nVALID PATHS:")
for status, path in valid_paths:
    print(f"  [{status}] {path}")

print("\n" + "=" * 80)
print("\nROOT CAUSE:")
print("  Video IDs being used as file paths:")
for path in problem_paths:
    if "/_Projects/" in path[1]:
        print(f"    - {path[1]}")

print("\nFIX NEEDED:")
print("  Map caption files to actual video files in TRANSCRIBE stage")
