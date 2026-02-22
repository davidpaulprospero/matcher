"""
Simple snapshot testing utility for OTIO output validation.

Provides basic snapshot functionality without requiring pytest-snapshot.
"""

import json
import hashlib
from pathlib import Path
from typing import Any, Optional


class SnapshotManager:
    """Manages snapshot files for regression testing."""

    def __init__(self, snapshot_dir: Path):
        self.snapshot_dir = Path(snapshot_dir)
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)

    def _get_snapshot_path(self, name: str) -> Path:
        """Get path for a snapshot file."""
        return self.snapshot_dir / f"{name}.snap.json"

    def save_snapshot(self, name: str, data: Any) -> None:
        """Save data as a snapshot."""
        snapshot_path = self._get_snapshot_path(name)

        # Convert to JSON-serializable format
        if hasattr(data, 'to_json'):
            serializable = data.to_json()
        elif hasattr(data, '__dict__'):
            serializable = self._make_serializable(data.__dict__)
        else:
            serializable = self._make_serializable(data)

        with open(snapshot_path, 'w', encoding='utf-8') as f:
            json.dump(serializable, f, indent=2, ensure_ascii=False)

    def _make_serializable(self, obj: Any) -> Any:
        """Convert object to JSON-serializable format."""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif hasattr(obj, '__dict__'):
            return self._make_serializable(obj.__dict__)
        elif isinstance(obj, (str, int, float, bool, type(None))):
            return obj
        else:
            return str(obj)

    def load_snapshot(self, name: str) -> Optional[Any]:
        """Load a snapshot by name."""
        snapshot_path = self._get_snapshot_path(name)
        if not snapshot_path.exists():
            return None

        with open(snapshot_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def compare(self, name: str, data: Any) -> tuple[bool, Optional[str]]:
        """
        Compare current data against stored snapshot.

        Returns:
            (matches: bool, diff_message: Optional[str])
        """
        snapshot = self.load_snapshot(name)
        if snapshot is None:
            return False, f"Snapshot '{name}' does not exist"

        # Convert data to comparable format
        if hasattr(data, 'to_json'):
            current = data.to_json()
        elif hasattr(data, '__dict__'):
            current = self._make_serializable(data.__dict__)
        else:
            current = self._make_serializable(data)

        if current == snapshot:
            return True, None

        # Generate simple diff
        return False, self._generate_diff(snapshot, current)

    def _generate_diff(self, expected: Any, actual: Any, path: str = "") -> str:
        """Generate a simple diff message."""
        if isinstance(expected, dict) and isinstance(actual, dict):
            all_keys = set(expected.keys()) | set(actual.keys())
            diffs = []
            for key in sorted(all_keys):
                exp_val = expected.get(key)
                act_val = actual.get(key)
                if exp_val != act_val:
                    new_path = f"{path}.{key}" if path else key
                    diffs.append(f"  {new_path}: expected {exp_val}, got {act_val}")
            return "\n".join(diffs) if diffs else "Objects differ"
        elif expected != actual:
            return f"  {path}: expected {expected}, got {actual}"
        return "Objects differ"

    def update_snapshot(self, name: str, data: Any) -> None:
        """Update snapshot with new data (for intentional changes)."""
        self.save_snapshot(name, data)

    def get_snapshot_hash(self, name: str) -> Optional[str]:
        """Get hash of a snapshot for change detection."""
        snapshot = self.load_snapshot(name)
        if snapshot is None:
            return None
        content = json.dumps(snapshot, sort_keys=True, ensure_ascii=False)
        return hashlib.md5(content.encode('utf-8')).hexdigest()


def snapshot_test(name: str, data: Any, snapshot_dir: Path) -> bool:
    """
    Simple snapshot comparison function.

    Returns True if data matches snapshot, False otherwise.
    """
    manager = SnapshotManager(snapshot_dir)
    matches, _ = manager.compare(name, data)
    return matches
