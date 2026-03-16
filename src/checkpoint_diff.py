"""
Checkpoint Diff Tool - Compare two checkpoint files and show staged changes.

This tool enables debugging checkpoint state changes by showing:
- Stage-level diff: added/removed/modified stage data
- Field-level diff for modified stages: before/after values

Purpose
-------
The checkpoint diff tool helps developers understand what changed between two
checkpoint states during pipeline execution. This is particularly useful for:
- Debugging why a pipeline chose different videos
- Understanding what happened during iterative matching
- Tracking progress across pipeline restarts
- Identifying where rate limits or errors occurred

How Diff is Computed
--------------------
The tool compares two checkpoints at three levels:

1. Metadata Level: Tracks changes to structural fields:
   - version, created_at, updated_at
   - last_completed_stage
   - config_hash, voiceover_path, voiceover_hash
   - transcription_metrics

2. Stage Level: Compares each pipeline stage:
   - analyze, video_search, caption
   - match, iterative_match, download_segments

3. Field Level: For modified stages, recursively compares all nested fields:
   - Simple values (strings, numbers, booleans) - direct comparison
   - Lists - element-by-element comparison
   - Dicts - recursive key-by-key comparison

Fields Compared
--------------
- METADATA_FIELDS: Structural/checkpoint metadata
- STAGE_FIELDS: Pipeline stage names
- OTHER_FIELDS: chapter_data, stage_metrics, keyword_presets

Usage Examples
--------------
1. CLI usage with two checkpoint files:
    $ python -m src.checkpoint_diff checkpoint1.json checkpoint2.json
    $ python main.py --checkpoint-diff checkpoint1.json checkpoint2.json

2. JSON output for programmatic use:
    $ python main.py --checkpoint-diff checkpoint1.json checkpoint2.json --json

3. Programmatic usage:
    >>> from src.checkpoint_diff import compute_checkpoint_diff, format_diff_json
    >>> diff = compute_checkpoint_diff("cp1.json", "cp2.json")
    >>> print(format_diff_json(diff))

4. Check for changes only:
    >>> diff = compute_checkpoint_diff("cp1.json", "cp2.json")
    >>> if diff.has_changes():
    ...     print("Checkpoint has changes!")

JSON Output Format
-----------------
When using --json flag, the output is structured as:
{
    "checkpoint1": "path/to/checkpoint1.json",
    "checkpoint2": "path/to/checkpoint2.json",
    "has_changes": true,
    "metadata_diffs": [
        {
            "field": "last_completed_stage",
            "change_type": "modified",
            "old_value": "MATCH",
            "new_value": "DOWNLOAD_SEGMENTS"
        }
    ],
    "stage_diffs": [
        {
            "stage": "match",
            "change_type": "modified",
            "field_diffs": [
                {
                    "field": "matches[0].video_id",
                    "change_type": "modified",
                    "old_value": "abc123",
                    "new_value": "def456"
                }
            ]
        }
    ]
}

Change Types
------------
- "added": Field exists in new checkpoint but not old
- "removed": Field exists in old checkpoint but not new
- "modified": Field exists in both but value changed
- "unchanged": No difference detected
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass


# Fields that are metadata/structural rather than stage data
METADATA_FIELDS = {
    'version', 'created_at', 'updated_at', 'last_completed_stage',
    'config_hash', 'voiceover_path', 'voiceover_hash', 'transcription_metrics'
}

# Stage fields in checkpoint
STAGE_FIELDS = {
    'analyze', 'video_search', 'caption', 'match',
    'iterative_match', 'download_segments'
}

# Other top-level fields
OTHER_FIELDS = {'chapter_data', 'stage_metrics', 'keyword_presets'}


@dataclass
class FieldDiff:
    """Represents a single field difference between two checkpoint values.

    Attributes:
        field: The dot-notation path to the field (e.g., "matches[0].video_id")
        change_type: Type of change - "added", "removed", or "modified"
        old_value: Value in the first checkpoint (None for added fields)
        new_value: Value in the second checkpoint (None for removed fields)
    """
    field: str
    change_type: str  # 'added', 'removed', 'modified'
    old_value: Any
    new_value: Any

    def __repr__(self):
        if self.change_type == 'added':
            return f"+ {self.field}: {self._truncate(self.new_value)}"
        elif self.change_type == 'removed':
            return f"- {self.field}: {self._truncate(self.old_value)}"
        else:
            return f"~ {self.field}: {self._truncate(self.old_value)} -> {self._truncate(self.new_value)}"

    def _truncate(self, value: Any, max_len: int = 80) -> str:
        s = str(value)
        if len(s) > max_len:
            return s[:max_len-3] + '...'
        return s


@dataclass
class StageDiff:
    """Represents differences for a single pipeline stage.

    Attributes:
        stage_name: Name of the pipeline stage (e.g., "match", "video_search")
        change_type: Overall change type - "added", "removed", "modified", "unchanged"
        field_diffs: List of FieldDiff objects showing specific field changes
    """
    stage_name: str
    change_type: str  # 'added', 'removed', 'modified', 'unchanged'
    field_diffs: List[FieldDiff]

    def has_changes(self) -> bool:
        return self.change_type != 'unchanged'


@dataclass
class CheckpointDiff:
    """Complete diff between two checkpoint files.

    This is the main result object returned by compute_checkpoint_diff().
    It contains all differences found between two checkpoint states.

    Attributes:
        checkpoint1_path: Path to the first (older) checkpoint file
        checkpoint2_path: Path to the second (newer) checkpoint file
        stage_diffs: List of StageDiff for each pipeline stage
        metadata_diffs: List of FieldDiff for metadata fields
        stage_order: Ordered list of stage names (defaults to standard pipeline order)

    Example:
        >>> diff = compute_checkpoint_diff("old.json", "new.json")
        >>> diff.has_changes()
        True
        >>> for sd in diff.stage_diffs:
        ...     if sd.has_changes():
        ...         print(f"{sd.stage_name}: {sd.change_type}")
    """
    checkpoint1_path: str
    checkpoint2_path: str
    stage_diffs: List[StageDiff]
    metadata_diffs: List[FieldDiff]
    stage_order: List[str] = None

    def __post_init__(self):
        if self.stage_order is None:
            self.stage_order = ['analyze', 'video_search', 'caption', 'match',
                               'iterative_match', 'download_segments']

    def has_changes(self) -> bool:
        return any(sd.has_changes() for sd in self.stage_diffs) or len(self.metadata_diffs) > 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict"""
        return {
            'checkpoint1': self.checkpoint1_path,
            'checkpoint2': self.checkpoint2_path,
            'has_changes': self.has_changes(),
            'metadata_diffs': [
                {
                    'field': d.field,
                    'change_type': d.change_type,
                    'old_value': d.old_value,
                    'new_value': d.new_value
                }
                for d in self.metadata_diffs
            ],
            'stage_diffs': [
                {
                    'stage': sd.stage_name,
                    'change_type': sd.change_type,
                    'field_diffs': [
                        {
                            'field': fd.field,
                            'change_type': fd.change_type,
                            'old_value': fd.old_value,
                            'new_value': fd.new_value
                        }
                        for fd in sd.field_diffs
                    ]
                }
                for sd in self.stage_diffs
            ]
        }


def load_checkpoint(path: str) -> Dict[str, Any]:
    """Load checkpoint from file, handling compression.

    Args:
        path: Path to checkpoint file (.json or .json.gz)

    Returns:
        Dictionary containing checkpoint data

    Raises:
        FileNotFoundError: If checkpoint file doesn't exist
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    if p.suffix == '.gz':
        import gzip
        with gzip.open(p, 'rt', encoding='utf-8') as f:
            return json.load(f)
    else:
        with open(p, 'r', encoding='utf-8') as f:
            return json.load(f)


def compute_field_diff(old: Any, new: Any, path: str = "") -> List[FieldDiff]:
    """Compute diff between two values recursively.

    Performs deep comparison of values, handling:
    - None values (treated as absent)
    - Type changes
    - Dictionaries (key-by-key comparison)
    - Lists (element-by-element if same length, otherwise as whole)
    - Primitives (direct comparison)

    Args:
        old: Value from first checkpoint
        new: Value from second checkpoint
        path: Current path in the object tree (for nested field naming)

    Returns:
        List of FieldDiff objects representing all differences found
    """
    diffs = []

    # Handle None values
    if old is None and new is not None:
        return [FieldDiff(path, 'added', None, new)]
    if new is None and old is not None:
        return [FieldDiff(path, 'removed', old, None)]

    # Handle type changes
    if type(old) != type(new):
        return [FieldDiff(path, 'modified', old, new)]

    # Handle dicts
    if isinstance(old, dict):
        all_keys = set(old.keys()) | set(new.keys())
        for key in sorted(all_keys):
            new_path = f"{path}.{key}" if path else key
            if key not in new:
                diffs.append(FieldDiff(new_path, 'removed', old[key], None))
            elif key not in old:
                diffs.append(FieldDiff(new_path, 'added', None, new[key]))
            elif old[key] != new[key]:
                diffs.extend(compute_field_diff(old[key], new[key], new_path))
        return diffs

    # Handle lists - compare element by element if same length
    if isinstance(old, list):
        if len(old) != len(new):
            diffs.append(FieldDiff(path, 'modified', old, new))
        else:
            for i, (o, n) in enumerate(zip(old, new)):
                if o != n:
                    diffs.extend(compute_field_diff(o, n, f"{path}[{i}]"))
        return diffs

    # Handle primitives
    if old != new:
        return [FieldDiff(path, 'modified', old, new)]

    return diffs


def compute_stage_diff(stage_name: str, old_data: Dict, new_data: Dict) -> StageDiff:
    """Compute diff for a single pipeline stage.

    Compares stage data between two checkpoints and determines:
    - If stage was added (exists only in new)
    - If stage was removed (exists only in old)
    - If stage was modified (exists in both with changes)
    - If stage is unchanged (exists in both with same data)

    Args:
        stage_name: Name of the pipeline stage
        old_data: Stage data from first checkpoint (can be empty dict)
        new_data: Stage data from second checkpoint (can be empty dict)

    Returns:
        StageDiff object containing all field-level differences
    """
    if not old_data and new_data:
        field_diffs = []
        for key, value in new_data.items():
            field_diffs.append(FieldDiff(key, 'added', None, value))
        return StageDiff(stage_name, 'added', field_diffs)

    if old_data and not new_data:
        field_diffs = []
        for key, value in old_data.items():
            field_diffs.append(FieldDiff(key, 'removed', value, None))
        return StageDiff(stage_name, 'removed', field_diffs)

    if not old_data and not new_data:
        return StageDiff(stage_name, 'unchanged', [])

    # Both have data - compute field-level diff
    field_diffs = compute_field_diff(old_data, new_data)
    if not field_diffs:
        return StageDiff(stage_name, 'unchanged', [])

    return StageDiff(stage_name, 'modified', field_diffs)


def compute_checkpoint_diff(checkpoint1_path: str, checkpoint2_path: str) -> CheckpointDiff:
    """Compute diff between two checkpoint files.

    This is the main function for comparing checkpoints. It:
    1. Loads both checkpoint files (handles .json.gz compression)
    2. Computes metadata-level differences
    3. Computes stage-level differences for each pipeline stage

    Args:
        checkpoint1_path: Path to the first (typically older) checkpoint
        checkpoint2_path: Path to the second (typically newer) checkpoint

    Returns:
        CheckpointDiff object containing all differences

    Raises:
        FileNotFoundError: If either checkpoint file doesn't exist
    """
    cp1 = load_checkpoint(checkpoint1_path)
    cp2 = load_checkpoint(checkpoint2_path)

    # Compute metadata diffs
    metadata_diffs = []
    for field in METADATA_FIELDS:
        old_val = cp1.get(field)
        new_val = cp2.get(field)
        if old_val != new_val:
            if old_val is None and new_val is not None:
                metadata_diffs.append(FieldDiff(field, 'added', None, new_val))
            elif new_val is None and old_val is not None:
                metadata_diffs.append(FieldDiff(field, 'removed', old_val, None))
            else:
                metadata_diffs.append(FieldDiff(field, 'modified', old_val, new_val))

    # Compute stage diffs
    stage_diffs = []
    for stage in STAGE_FIELDS:
        old_data = cp1.get(stage, {})
        new_data = cp2.get(stage, {})
        stage_diffs.append(compute_stage_diff(stage, old_data, new_data))

    return CheckpointDiff(
        checkpoint1_path=checkpoint1_path,
        checkpoint2_path=checkpoint2_path,
        stage_diffs=stage_diffs,
        metadata_diffs=metadata_diffs
    )


def format_diff_human(diff: CheckpointDiff) -> str:
    """Format diff for human-readable output.

    Creates a readable text representation of the checkpoint differences,
    with sections for metadata changes and per-stage changes.

    Args:
        diff: CheckpointDiff object from compute_checkpoint_diff()

    Returns:
        Formatted string suitable for console output

    Example output:
        ============================================================
        Checkpoint Diff: old.json -> new.json
        ============================================================

        Summary: 1 stage(s) changed, 2 metadata field(s) changed

        --- Metadata Changes ---
          ~ last_completed_stage: MATCH -> DOWNLOAD_SEGMENTS
          + voiceover_hash: None -> abc123

        --- Stage Changes ---
        [match] (modified)
          ~ matches[0].video_id: abc123 -> def456
    """
    lines = []

    # Header
    lines.append("=" * 60)
    lines.append(f"Checkpoint Diff: {diff.checkpoint1_path} -> {diff.checkpoint2_path}")
    lines.append("=" * 60)

    if not diff.has_changes():
        lines.append("\nNo differences found.")
        return "\n".join(lines)

    # Summary
    changed_stages = [sd for sd in diff.stage_diffs if sd.has_changes()]
    lines.append(f"\nSummary: {len(changed_stages)} stage(s) changed, "
                 f"{len(diff.metadata_diffs)} metadata field(s) changed")

    # Metadata changes
    if diff.metadata_diffs:
        lines.append("\n--- Metadata Changes ---")
        for md in diff.metadata_diffs:
            lines.append(f"  {md}")

    # Stage changes
    lines.append("\n--- Stage Changes ---")
    for sd in diff.stage_diffs:
        if not sd.has_changes():
            continue

        lines.append(f"\n[{sd.stage_name}] ({sd.change_type})")
        for fd in sd.field_diffs:
            # Show more detail for modified fields
            if fd.change_type == 'modified' and isinstance(fd.old_value, (dict, list)):
                lines.append(f"  {fd.field}:")
                lines.append(f"    - {str(fd.old_value)[:100]}")
                lines.append(f"    + {str(fd.new_value)[:100]}")
            else:
                lines.append(f"  {fd}")

    return "\n".join(lines)


def format_diff_json(diff: CheckpointDiff) -> str:
    """Format diff as JSON for programmatic consumption.

    Args:
        diff: CheckpointDiff object from compute_checkpoint_diff()

    Returns:
        JSON string representation of the diff
    """
    return json.dumps(diff.to_dict(), indent=2)


def main():
    """CLI entry point"""
    if len(sys.argv) < 3:
        print("Usage: python -m src.checkpoint_diff <checkpoint1> <checkpoint2> [--json]")
        print("       python main.py --checkpoint-diff <checkpoint1> <checkpoint2> [--json]")
        sys.exit(1)

    checkpoint1 = sys.argv[1]
    checkpoint2 = sys.argv[2]
    json_output = '--json' in sys.argv

    try:
        diff = compute_checkpoint_diff(checkpoint1, checkpoint2)

        if json_output:
            print(format_diff_json(diff))
        else:
            print(format_diff_human(diff))

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in checkpoint: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
