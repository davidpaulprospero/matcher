---
name: validate-output
description: Validate pipeline output structure against baseline. Use when checking OTIO, XML, EDL files are correctly built, or verifying output after a pipeline run.
allowed-tools:
  - Read
  - Bash(python:*)
  - Bash(pytest:*)
---

# Validate Output

Validate pipeline output **structure** against the baseline - ensures OTIO, XML, EDL, and JSON files are correctly built.

## What This Validates

| Check | What It Verifies |
|-------|------------------|
| OTIO files | Parse without errors, correct track names |
| XML files | Well-formed xmeml, proper element structure |
| XML media bins | `media_part*.xml` have `<bin>` + `<sequence>` siblings (required for DaVinci) |
| EDL file | Has TITLE and FCM headers |
| Segments JSON | Has required keys (generated_at, frame_rate, segments) |
| Track layout | 19 tracks present (V1-V10, A1-A9) |
| Clip integrity | No zero-duration clips |
| Coverage | V1 >= 90%, V2 >= 50%, V8 >= 20% |
| Duplicate paths | Same file from multiple paths (causes DaVinci hang) |

## Structure Report (NLE Import)

The validation now includes a **Structure Report** with import-critical info:

| Section | Details |
|---------|---------|
| Timeline | Duration, frame rate |
| Files | OTIO/XML counts, EDL entries, segments |
| Tracks | Per-track clip count and coverage % |
| Media References | Files found on disk (%) |
| Import Readiness | Path issues (>200 chars, unicode), DaVinci XML compatibility |

This helps ensure clean imports into DaVinci Resolve or other NLEs.

## Instructions

When this skill is invoked:

1. **Parse the argument**: Extract the path from the user's request. Accepts:
   - **Output path**: Direct path to output folder (e.g., `.../output/20260115_064843`)
   - **Project path**: Path to project folder - auto-finds latest output (e.g., `E:/Edit Job/ProjectName__2026-01-13`)
   - **No path**: Uses default baseline

2. **Run structural validation**:
```bash
cd "D:/_Projects/voiceover-matcher" && python tests/test_output_comparison.py "<path>" -v
```

3. **Present results clearly**:
   - Show pass/fail/warning counts
   - List any failures
   - Show key metrics (track count, coverage percentages)
   - Indicate overall pass/fail status

4. **If validation fails**, suggest running the detailed test suite:
```bash
OUTPUT_DIR="<output_path>" pytest tests/test_output_baseline.py -v
```

## Usage Examples

```
/validate-output
/validate-output E:/Edit Job/Stu/January/24__2026-01-13/output/20260115_064843
/validate-output E:/Edit Job/Stu/January/24__2026-01-13
/validate-output "E:/path/with spaces/ProjectName__2026-01-10"
```

## Path Resolution

| Input | Resolution |
|-------|------------|
| Output path (has `timeline_FULL.otio`) | Used directly |
| Project path (has `output/` folder) | Finds latest timestamped output |
| No path provided | Uses default baseline |

## Baseline Reference

Current baseline: `E:/Edit Job/Stu/January/24__2026-01-13/output/20260115_064843`

| Metric | Expected |
|--------|----------|
| OTIO files | 12 |
| XML files | 7 |
| Track count | 19 |
| V1 clips | 768 (99.1% coverage) |
| V9 Entity Images | 52 clips (7.1% coverage) |
