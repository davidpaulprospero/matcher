"""
Batch checkpoint for caption fetching with partial result recovery.

Provides checkpoint support for batch caption fetching operations.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

if TYPE_CHECKING:
    from .models import ErrorPatternResult

logger = logging.getLogger(__name__)


@dataclass
class CaptionBatchCheckpoint:
    """Checkpoint for batch caption fetching with partial result recovery (US-005 Sprint 8).

    Stores successfully fetched captions during batch operation, enabling:
    1. Periodic saves every N successful fetches
    2. Recovery from ErrorPatternAbortError with partial results
    3. Resume capability that skips already-fetched videos

    Checkpoint is saved to: <project>/.cache/caption_checkpoint.json

    Attributes:
        results: Dict mapping video_id to serialized CaptionResult or error dict.
        total_requested: Total number of videos requested in the batch.
        success_count: Number of successfully fetched captions.
        error_count: Number of failed fetches (errors/unavailable).
        aborted: Whether batch was aborted due to error pattern.
        abort_reason: Human-readable abort reason if aborted.
        abort_pattern_info: Dict with ErrorPatternResult data if aborted by pattern.
        created_at: Unix timestamp when checkpoint was created.
        updated_at: Unix timestamp when checkpoint was last updated.
        remaining_video_ids: List of video IDs not yet processed.
    """
    results: Dict[str, Any] = field(default_factory=dict)
    total_requested: int = 0
    success_count: int = 0
    error_count: int = 0
    aborted: bool = False
    abort_reason: str = ""
    abort_pattern_info: Optional[Dict[str, Any]] = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    remaining_video_ids: List[str] = field(default_factory=list)

    def update(self, video_id: str, result: Any) -> None:
        """Add or update a result in the checkpoint.

        Args:
            video_id: Video ID that was processed.
            result: CaptionResult on success, or error dict on failure.
        """
        # Check if result is a CaptionResult object
        # We avoid importing to prevent circular imports
        if hasattr(result, 'video_id') and hasattr(result, 'segments'):
            # Serialize CaptionResult to dict
            self.results[video_id] = {
                'video_id': result.video_id,
                'segments': [seg.to_dict() for seg in result.segments],
                'language': result.language,
                'is_auto_generated': result.is_auto_generated,
                'format_source': result.format_source,
                'segment_count': len(result.segments),
                'caption_quality': result.caption_quality,
            }
            self.success_count += 1
        else:
            # Already a dict (error result)
            self.results[video_id] = result
            if result.get('error') or result.get('unavailable'):
                self.error_count += 1
            else:
                self.success_count += 1

        # Remove from remaining if present
        if video_id in self.remaining_video_ids:
            self.remaining_video_ids.remove(video_id)

        self.updated_at = time.time()

    def mark_aborted(
        self,
        reason: str,
        pattern_result: Optional['ErrorPatternResult'] = None,
        remaining_ids: Optional[List[str]] = None
    ) -> None:
        """Mark the checkpoint as aborted due to error pattern.

        Args:
            reason: Human-readable abort reason.
            pattern_result: ErrorPatternResult with detection details.
            remaining_ids: List of video IDs not yet processed.
        """
        self.aborted = True
        self.abort_reason = reason
        if pattern_result:
            self.abort_pattern_info = {
                'detected': pattern_result.detected,
                'error_signature': pattern_result.error_signature,
                'affected_video_ids': pattern_result.affected_video_ids,
                'sample_size': pattern_result.sample_size,
                'ratio': pattern_result.ratio,
                'likely_cause': pattern_result.likely_cause,
            }
        if remaining_ids:
            self.remaining_video_ids = remaining_ids
        self.updated_at = time.time()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'results': self.results,
            'total_requested': self.total_requested,
            'success_count': self.success_count,
            'error_count': self.error_count,
            'aborted': self.aborted,
            'abort_reason': self.abort_reason,
            'abort_pattern_info': self.abort_pattern_info,
            'created_at': self.created_at,
            'updated_at': self.updated_at,
            'remaining_video_ids': self.remaining_video_ids,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CaptionBatchCheckpoint':
        """Create from dictionary loaded from JSON."""
        return cls(
            results=data.get('results', {}),
            total_requested=data.get('total_requested', 0),
            success_count=data.get('success_count', 0),
            error_count=data.get('error_count', 0),
            aborted=data.get('aborted', False),
            abort_reason=data.get('abort_reason', ''),
            abort_pattern_info=data.get('abort_pattern_info'),
            created_at=data.get('created_at', time.time()),
            updated_at=data.get('updated_at', time.time()),
            remaining_video_ids=data.get('remaining_video_ids', []),
        )

    def save(self, path: Union[str, Path]) -> bool:
        """Save checkpoint to JSON file.

        Args:
            path: Path to checkpoint file.

        Returns:
            True if save succeeded, False otherwise.
        """
        try:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)

            # Write atomically (write to temp, then rename)
            temp_path = path.with_suffix('.tmp')
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(self.to_dict(), f, indent=2)
            temp_path.replace(path)

            logger.debug(f"Saved caption checkpoint: {self.success_count} successes, {self.error_count} errors")
            return True
        except Exception as e:
            logger.error(f"Failed to save caption checkpoint: {e}")
            return False

    @classmethod
    def load(cls, path: Union[str, Path]) -> Optional['CaptionBatchCheckpoint']:
        """Load checkpoint from JSON file.

        Args:
            path: Path to checkpoint file.

        Returns:
            CaptionBatchCheckpoint if loaded successfully, None otherwise.
        """
        try:
            path = Path(path)
            if not path.exists():
                return None

            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            checkpoint = cls.from_dict(data)
            logger.info(
                f"Loaded caption checkpoint: {checkpoint.success_count} successes, "
                f"{checkpoint.error_count} errors, aborted={checkpoint.aborted}"
            )
            return checkpoint
        except json.JSONDecodeError as e:
            logger.warning(f"Corrupt caption checkpoint (JSON error): {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to load caption checkpoint: {e}")
            return None

    @staticmethod
    def get_checkpoint_path(project_dir: Union[str, Path]) -> Path:
        """Get the standard checkpoint file path for a project.

        Args:
            project_dir: Project directory path.

        Returns:
            Path to .cache/caption_checkpoint.json in the project.
        """
        return Path(project_dir) / '.cache' / 'caption_checkpoint.json'

    def get_remaining_ids(self, all_video_ids: List[str]) -> List[str]:
        """Get video IDs that haven't been processed yet.

        Args:
            all_video_ids: Complete list of video IDs for the batch.

        Returns:
            List of video IDs not in results.
        """
        processed = set(self.results.keys())
        return [vid for vid in all_video_ids if vid not in processed]

    def has_result(self, video_id: str) -> bool:
        """Check if a video ID has already been processed."""
        return video_id in self.results

    def get_successful_results(self) -> Dict[str, Any]:
        """Get only successful results (no errors/unavailable)."""
        return {
            vid: result for vid, result in self.results.items()
            if not result.get('error') and not result.get('unavailable')
        }
