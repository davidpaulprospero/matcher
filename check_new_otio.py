#!/usr/bin/env python3
"""
Check if new OTIO has correct file paths
"""
import json
from pathlib import Path

# Read the new FULL OTIO
otio_path = Path("E:/Edit Job/test/jan_test/audio__2026-01-01/output/20260117_025007/timeline_FULL.otio")

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

print("NEW OTIO Target URLs Found:")
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
if problem_paths:
    print("\nPROBLEM PATHS (will cause DaVinci import to fail):")
    for status, path in problem_paths:
        print(f"  [{status}] {path}")
else:
    print("\n[SUCCESS] NO PROBLEM PATHS FOUND!")

print("\nVALID PATHS:")
for status, path in valid_paths:
    print(f"  [{status}] {path}")

print("\n" + "=" * 80)
if not problem_paths:
    print("\nVICTORY! All OTIO paths are valid and point to real files!")
    print("DaVinci Resolve should be able to import this timeline.")
