name: fix-otio-paths
description: Fix OTIO file paths to use Windows-compatible paths (converts E:/v/... or /home/hpmint/... paths to project/segments_archives/...)
trigger: /fix-otio-paths

instructions: |
  Use this skill when the user asks to "fix OTIO paths" or "fix the OTIO files".

  This skill converts Linux-style paths in OTIO files to correct Windows paths:
  - E:/v/Degold/DISNEY/... style paths -> project/segments_archives/filename.mp4
  - /home/hpmint/... Linux paths -> project/segments_archives/filename.mp4

  The script finds all .otio files in a directory or processes a single file,
  converts all target_url entries, and creates backups before modifying.

  Usage:
    /fix-otio-paths <project_output_folder>
    /fix-otio-paths <otio_file> <project_base>

  Examples:
    /fix-otio-paths "E:/Edit Job/Client/Project/output/20260506_004005"
    /fix-otio-paths "timeline_FULL.otio" "E:/Edit Job/Client/Project"

  The script is at: .claude/skills/fix-otio-paths/script.py

  Runs: python script.py <path> [project_base]