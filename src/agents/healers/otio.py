"""
OTIO Healer - Comprehensive timeline generation error recovery.

Handles all OTIO-related failures including:
- Media reference errors (missing files, invalid paths, encoding)
- Clip timing issues (duration, overlap, gaps, speed adjustment)
- Track structure problems (misalignment, too many tracks)
- Export failures (EDL, XML, adapter errors)
- Frame rate / timecode issues
- Metadata serialization errors
- Memory / performance issues with large timelines
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple

from ..base import Healer, HealerResult, HealerAction

if TYPE_CHECKING:
    from ...config import Config
    from ...state import PipelineState

logger = logging.getLogger(__name__)


class OTIOHealer(Healer):
    """
    Comprehensive OTIO timeline healer.

    Handles errors in:
    - Timeline creation (create_timeline)
    - Clip creation (create_clip_with_timewarp)
    - Timeline export (OTIO, EDL, XML)
    - Media references
    - Metadata serialization
    """

    name = "otio-healer"
    description = "Fix timeline generation errors"

    # Comprehensive error patterns
    error_patterns = [
        # OTIO core errors
        "opentimelineio", "otio", "timeline", "clip", "track",
        # Media reference errors
        "media reference", "external reference", "target_url",
        "file not found", "filenotfound", "no such file",
        "media_reference", "available_range",
        # Duration/timing errors
        "duration", "negative duration", "zero duration",
        "source_range", "time_range", "rationaltime",
        "frame", "timecode", "rate",
        # Gap/overlap errors
        "gap", "overflow", "overlap", "collision",
        # Track errors
        "track", "tracks", "video track", "audio track",
        # Export errors
        "edl", "xml", "adapter", "write_to_file", "export",
        "resolve", "davinci", "premiere", "fcpx",
        # Path errors
        "path", "url", "encoding", "unicode", "utf",
        "backslash", "forward slash",
        # Metadata errors
        "metadata", "serialize", "json", "numpy",
        # Memory errors
        "memory", "too large", "too many clips",
    ]

    exception_types = [
        ValueError,
        TypeError,
        FileNotFoundError,
        OSError,
    ]

    # Constants
    MIN_CLIP_DURATION = 0.04  # ~1 frame at 24fps
    MAX_CLIP_DURATION = 86400.0  # 24 hours
    MIN_SPEED = 0.1  # 10% speed minimum
    MAX_SPEED = 10.0  # 1000% speed maximum
    GAP_MODES = ["scale", "proportional", "none"]
    MAX_TRACKS = 99  # Most NLEs support this

    def __init__(self, config, project_dir):
        super().__init__(config, project_dir)
        self.fixed_paths: Set[str] = set()
        self.fixed_durations: int = 0
        self.gap_mode_changes: int = 0

    def fix(
        self,
        error: Exception,
        state: 'PipelineState',
        stage_name: str
    ) -> HealerResult:
        """
        Attempt to fix OTIO-related errors.

        Tries fixes in order from most specific to most general.
        """
        error_str = str(error).lower()
        error_type = type(error).__name__

        self.log_attempt(f"Analyzing error: {error_type}")

        # === Media Reference Errors ===
        if self._is_media_error(error_str):
            result = self._fix_media_references(error, state)
            if result.success:
                return result

        # === Duration/Timing Errors ===
        if self._is_duration_error(error_str):
            result = self._fix_durations(error, state)
            if result.success:
                return result

        # === Gap/Overflow Errors ===
        if self._is_gap_error(error_str):
            result = self._fix_gaps(error, state)
            if result.success:
                return result

        # === Overlap Errors ===
        if self._is_overlap_error(error_str):
            result = self._fix_overlaps(error, state)
            if result.success:
                return result

        # === Track Structure Errors ===
        if self._is_track_error(error_str):
            result = self._fix_tracks(error, state)
            if result.success:
                return result

        # === Export Format Errors ===
        if self._is_export_error(error_str):
            result = self._fix_export(error, state)
            if result.success:
                return result

        # === Metadata Errors ===
        if self._is_metadata_error(error_str):
            result = self._fix_metadata(error, state)
            if result.success:
                return result

        # === Path Encoding Errors ===
        if self._is_path_error(error_str):
            result = self._fix_paths(error, state)
            if result.success:
                return result

        # === Frame Rate Errors ===
        if self._is_framerate_error(error_str):
            result = self._fix_framerate(error, state)
            if result.success:
                return result

        # === Memory/Size Errors ===
        if self._is_memory_error(error_str):
            result = self._fix_memory(error, state)
            if result.success:
                return result

        # === Generic Fallback ===
        return self._apply_safe_mode(error, state)

    # =========================================================================
    # ERROR DETECTION METHODS
    # =========================================================================

    def _is_media_error(self, error_str: str) -> bool:
        patterns = ["file not found", "filenotfound", "no such file",
                   "media reference", "target_url", "external reference",
                   "media_reference", "available_range", "cannot open",
                   "missing media"]  # From preflight_check messages
        return any(p in error_str for p in patterns)

    def _is_duration_error(self, error_str: str) -> bool:
        patterns = ["duration", "negative", "zero", "source_range",
                   "time_range", "rationaltime", "invalid time"]
        return any(p in error_str for p in patterns)

    def _is_gap_error(self, error_str: str) -> bool:
        patterns = ["gap", "overflow", "exceed", "total duration",
                   "timeline duration", "too long", "doesn't fit"]
        return any(p in error_str for p in patterns)

    def _is_overlap_error(self, error_str: str) -> bool:
        patterns = ["overlap", "collision", "intersect", "conflict"]
        return any(p in error_str for p in patterns)

    def _is_track_error(self, error_str: str) -> bool:
        patterns = ["track", "tracks", "video track", "audio track",
                   "track mismatch", "track count"]
        return any(p in error_str for p in patterns)

    def _is_export_error(self, error_str: str) -> bool:
        patterns = ["edl", "xml", "adapter", "write_to_file", "export",
                   "resolve", "davinci", "premiere", "fcpx", "fcp"]
        return any(p in error_str for p in patterns)

    def _is_metadata_error(self, error_str: str) -> bool:
        patterns = ["metadata", "serialize", "json", "numpy",
                   "not serializable", "encode"]
        return any(p in error_str for p in patterns)

    def _is_path_error(self, error_str: str) -> bool:
        patterns = ["path", "url", "encoding", "unicode", "utf",
                   "backslash", "invalid character", "illegal"]
        return any(p in error_str for p in patterns)

    def _is_framerate_error(self, error_str: str) -> bool:
        patterns = ["frame rate", "framerate", "fps", "rate mismatch",
                   "timecode", "drop frame"]
        return any(p in error_str for p in patterns)

    def _is_memory_error(self, error_str: str) -> bool:
        patterns = ["memory", "too large", "too many", "out of memory",
                   "allocation", "exceeded"]
        return any(p in error_str for p in patterns)

    # =========================================================================
    # FIX METHODS
    # =========================================================================

    def _fix_media_references(self, error: Exception, state: 'PipelineState') -> HealerResult:
        """Fix missing or invalid media file references."""
        self.log_attempt("Resolving media reference issues...")

        if not hasattr(state, 'matches') or not state.matches:
            return HealerResult.failed("No matches to fix media references for")

        fixed_count = 0
        missing_count = 0
        cache_dirs = self._get_cache_dirs()

        for match in state.matches:
            video_path = self._get_video_path(match)
            if not video_path:
                continue

            path = Path(video_path)

            # Check if file exists
            if path.exists():
                continue

            missing_count += 1

            # Try to resolve from various locations
            resolved = self._resolve_missing_file(video_path, cache_dirs, match)
            if resolved:
                self._set_video_path(match, resolved)
                fixed_count += 1
                self.fixed_paths.add(video_path)
                self.log_attempt(f"Resolved: {path.name} -> {Path(resolved).name}")

        if fixed_count > 0:
            self.log_success(f"Resolved {fixed_count}/{missing_count} missing media files")
            return HealerResult.fixed(
                f"Resolved {fixed_count} missing media files",
                action=HealerAction.RETRY,
                fixed_count=fixed_count,
                missing_count=missing_count
            )

        if missing_count > 0:
            return HealerResult.failed(f"{missing_count} media files still missing")

        return HealerResult.failed("No media reference issues found")

    def _fix_durations(self, error: Exception, state: 'PipelineState') -> HealerResult:
        """Fix invalid clip durations."""
        self.log_attempt("Fixing clip durations...")

        if not hasattr(state, 'matches') or not state.matches:
            return HealerResult.failed("No matches to fix durations for")

        fixed_count = 0

        for match in state.matches:
            # Fix segment durations
            segment = getattr(match, 'segment', None)
            if segment:
                fixed_count += self._fix_segment_duration(segment)

            # Fix video clip times
            fixed_count += self._fix_match_times(match)

            # Fix speed/time_scalar issues
            fixed_count += self._fix_speed_adjustment(match)

        self.fixed_durations += fixed_count

        if fixed_count > 0:
            self.log_success(f"Fixed {fixed_count} duration issues")
            return HealerResult.fixed(
                f"Fixed {fixed_count} clip duration issues",
                action=HealerAction.RETRY,
                fixed_count=fixed_count
            )

        return HealerResult.failed("No duration issues found to fix")

    def _fix_gaps(self, error: Exception, state: 'PipelineState') -> HealerResult:
        """Fix gap calculation/overflow issues."""
        self.log_attempt("Fixing gap distribution...")

        output_config = getattr(self.config, 'output', None)
        if not output_config:
            return HealerResult.failed("No output config available")

        current_mode = getattr(output_config, 'gap_mode', 'scale')

        try:
            current_idx = self.GAP_MODES.index(current_mode)
        except ValueError:
            current_idx = -1

        next_idx = current_idx + 1
        if next_idx >= len(self.GAP_MODES):
            # Already on simplest mode, try disabling gaps entirely
            self.log_attempt("Disabling voiceover alignment (no gaps)")
            if hasattr(output_config, 'align_to_voiceover'):
                output_config.align_to_voiceover = False
            return HealerResult.config_changed(
                "Disabled voiceover alignment to avoid gap issues",
                gap_mode="disabled"
            )

        new_mode = self.GAP_MODES[next_idx]
        self._set_config_attr(output_config, 'gap_mode', new_mode)
        self.gap_mode_changes += 1

        self.log_success(f"Switched gap_mode: {current_mode} -> {new_mode}")
        return HealerResult.config_changed(
            f"Switched gap_mode from '{current_mode}' to '{new_mode}'",
            old_mode=current_mode,
            new_mode=new_mode
        )

    def _fix_overlaps(self, error: Exception, state: 'PipelineState') -> HealerResult:
        """Fix overlapping clips on timeline."""
        self.log_attempt("Resolving clip overlaps...")

        if not hasattr(state, 'matches') or not state.matches:
            return HealerResult.failed("No matches to fix overlaps for")

        # Sort by segment start time
        sorted_matches = sorted(
            state.matches,
            key=lambda m: self._get_segment_start(m)
        )

        fixed_count = 0
        GAP_BUFFER = 0.02  # 20ms minimum gap

        for i in range(len(sorted_matches) - 1):
            current = sorted_matches[i]
            next_match = sorted_matches[i + 1]

            current_end = self._get_segment_end(current)
            next_start = self._get_segment_start(next_match)

            if current_end is None or next_start is None:
                continue

            if current_end > next_start - GAP_BUFFER:
                # Trim current clip
                new_end = next_start - GAP_BUFFER
                if new_end > self._get_segment_start(current) + self.MIN_CLIP_DURATION:
                    self._set_segment_end(current, new_end)
                    fixed_count += 1
                    self.log_attempt(f"Trimmed clip: {current_end:.2f}s -> {new_end:.2f}s")

        if fixed_count > 0:
            self.log_success(f"Fixed {fixed_count} overlapping clips")
            return HealerResult.fixed(
                f"Trimmed {fixed_count} overlapping clips",
                action=HealerAction.RETRY,
                fixed_count=fixed_count
            )

        return HealerResult.failed("No overlaps found to fix")

    def _fix_tracks(self, error: Exception, state: 'PipelineState') -> HealerResult:
        """Fix track structure issues."""
        self.log_attempt("Fixing track structure...")

        output_config = getattr(self.config, 'output', None)
        if not output_config:
            return HealerResult.failed("No output config available")

        changes = []

        # Reduce alternative tracks if too many
        num_alts = getattr(output_config, 'num_alternatives', 2)
        if num_alts > 5:
            self._set_config_attr(output_config, 'num_alternatives', 2)
            changes.append("Reduced alternatives to 2")

        # Disable strategy tracks if problematic
        if getattr(output_config, 'include_strategy_tracks', True):
            self._set_config_attr(output_config, 'include_strategy_tracks', False)
            changes.append("Disabled strategy tracks")

        if changes:
            self.log_success(f"Track fixes: {', '.join(changes)}")
            return HealerResult.config_changed(
                f"Fixed track structure: {', '.join(changes)}",
                changes=changes
            )

        return HealerResult.failed("No track issues found to fix")

    def _fix_export(self, error: Exception, state: 'PipelineState') -> HealerResult:
        """Fix export format issues."""
        self.log_attempt("Handling export issues...")

        error_str = str(error).lower()
        output_config = getattr(self.config, 'output', None)

        # EDL-specific issues
        if "edl" in error_str:
            if output_config and hasattr(output_config, 'export_edl'):
                self._set_config_attr(output_config, 'export_edl', False)
            self.log_success("Disabled EDL export (OTIO export will continue)")
            return HealerResult.config_changed(
                "Disabled EDL export due to adapter issues",
                skip_edl=True
            )

        # XML-specific issues
        if "xml" in error_str or "resolve" in error_str:
            if output_config and hasattr(output_config, 'export_xml'):
                self._set_config_attr(output_config, 'export_xml', False)
            self.log_success("Disabled XML export")
            return HealerResult.config_changed(
                "Disabled XML export due to format issues",
                skip_xml=True
            )

        # General adapter issues - try OTIO only
        self.log_attempt("Falling back to OTIO-only export")
        if output_config:
            self._set_config_attr(output_config, 'export_edl', False)
            self._set_config_attr(output_config, 'export_xml', False)

        return HealerResult.config_changed(
            "Using OTIO-only export (disabled EDL/XML)",
            otio_only=True
        )

    def _fix_metadata(self, error: Exception, state: 'PipelineState') -> HealerResult:
        """Fix metadata serialization issues."""
        self.log_attempt("Sanitizing clip metadata...")

        if not hasattr(state, 'matches') or not state.matches:
            return HealerResult.failed("No matches to fix metadata for")

        fixed_count = 0

        for match in state.matches:
            # Sanitize any metadata attached to matches
            if hasattr(match, 'metadata') and match.metadata:
                sanitized = self._sanitize_metadata(match.metadata)
                if sanitized != match.metadata:
                    match.metadata = sanitized
                    fixed_count += 1

            # Sanitize segment metadata
            segment = getattr(match, 'segment', None)
            if segment and hasattr(segment, 'metadata') and segment.metadata:
                sanitized = self._sanitize_metadata(segment.metadata)
                if sanitized != segment.metadata:
                    segment.metadata = sanitized
                    fixed_count += 1

        if fixed_count > 0:
            self.log_success(f"Sanitized {fixed_count} metadata entries")
            return HealerResult.fixed(
                f"Sanitized {fixed_count} metadata entries",
                action=HealerAction.RETRY,
                fixed_count=fixed_count
            )

        return HealerResult.failed("No metadata issues found")

    def _fix_paths(self, error: Exception, state: 'PipelineState') -> HealerResult:
        """Fix path encoding and format issues."""
        self.log_attempt("Fixing path encoding issues...")

        if not hasattr(state, 'matches') or not state.matches:
            return HealerResult.failed("No matches to fix paths for")

        fixed_count = 0

        for match in state.matches:
            video_path = self._get_video_path(match)
            if not video_path:
                continue

            sanitized = self._sanitize_path(video_path)
            if sanitized != video_path:
                # Check if sanitized path exists
                if Path(sanitized).exists():
                    self._set_video_path(match, sanitized)
                    fixed_count += 1
                elif Path(video_path).exists():
                    # Try to rename the file
                    try:
                        shutil.move(video_path, sanitized)
                        self._set_video_path(match, sanitized)
                        fixed_count += 1
                        self.log_attempt(f"Renamed file to sanitized path")
                    except Exception:
                        pass

        if fixed_count > 0:
            self.log_success(f"Fixed {fixed_count} path encoding issues")
            return HealerResult.fixed(
                f"Sanitized {fixed_count} file paths",
                action=HealerAction.RETRY,
                fixed_count=fixed_count
            )

        return HealerResult.failed("No path issues found to fix")

    def _fix_framerate(self, error: Exception, state: 'PipelineState') -> HealerResult:
        """Fix frame rate mismatches."""
        self.log_attempt("Checking frame rate settings...")

        output_config = getattr(self.config, 'output', None)
        if not output_config:
            return HealerResult.failed("No output config available")

        # Try common frame rates
        current_rate = getattr(output_config, 'frame_rate', 30.0)
        fallback_rates = [30.0, 24.0, 25.0, 29.97, 23.976]

        # Remove current rate and try next
        remaining = [r for r in fallback_rates if abs(r - current_rate) > 0.1]
        if remaining:
            new_rate = remaining[0]
            self._set_config_attr(output_config, 'frame_rate', new_rate)
            self.log_success(f"Changed frame rate: {current_rate} -> {new_rate}")
            return HealerResult.config_changed(
                f"Changed frame rate from {current_rate} to {new_rate}",
                old_rate=current_rate,
                new_rate=new_rate
            )

        return HealerResult.failed("Cannot fix frame rate issues")

    def _fix_memory(self, error: Exception, state: 'PipelineState') -> HealerResult:
        """Fix memory/performance issues with large timelines."""
        self.log_attempt("Reducing timeline complexity...")

        output_config = getattr(self.config, 'output', None)
        if not output_config:
            return HealerResult.failed("No output config available")

        changes = []

        # Disable all optional tracks
        if getattr(output_config, 'include_alternatives', True):
            self._set_config_attr(output_config, 'include_alternatives', False)
            changes.append("Disabled alternative tracks")

        if getattr(output_config, 'include_strategy_tracks', True):
            self._set_config_attr(output_config, 'include_strategy_tracks', False)
            changes.append("Disabled strategy tracks")

        # Disable entity tracks
        if getattr(output_config, 'include_entity_images', True):
            self._set_config_attr(output_config, 'include_entity_images', False)
            changes.append("Disabled entity image track")

        if getattr(output_config, 'include_entity_videos', True):
            self._set_config_attr(output_config, 'include_entity_videos', False)
            changes.append("Disabled entity video track")

        if changes:
            self.log_success(f"Simplified timeline: {len(changes)} changes")
            return HealerResult.config_changed(
                f"Simplified timeline: {', '.join(changes)}",
                changes=changes,
                simplified=True
            )

        return HealerResult.failed("Timeline already at minimum complexity")

    def _apply_safe_mode(self, error: Exception, state: 'PipelineState') -> HealerResult:
        """Apply safe mode settings as last resort."""
        self.log_attempt("Applying safe mode settings...")

        output_config = getattr(self.config, 'output', None)
        if not output_config:
            return HealerResult.failed("No output config available")

        # Apply minimal, safe settings
        safe_settings = {
            'include_alternatives': False,
            'include_strategy_tracks': False,
            'include_entity_images': False,
            'include_entity_videos': False,
            'export_edl': False,
            'export_xml': False,
            'gap_mode': 'none',
            'frame_rate': 30.0,
        }

        changes = []
        for key, value in safe_settings.items():
            if hasattr(output_config, key):
                current = getattr(output_config, key)
                if current != value:
                    self._set_config_attr(output_config, key, value)
                    changes.append(f"{key}={value}")

        if changes:
            self.log_success(f"Applied safe mode: {len(changes)} settings changed")
            return HealerResult.config_changed(
                f"Applied OTIO safe mode: {', '.join(changes[:5])}{'...' if len(changes) > 5 else ''}",
                safe_mode=True,
                changes=changes
            )

        return HealerResult.failed("Cannot apply safe mode (already minimal)")

    # =========================================================================
    # HELPER METHODS
    # =========================================================================

    def _get_cache_dirs(self) -> List[Path]:
        """Get all potential cache directories."""
        dirs = []

        if self.project_dir:
            project = Path(self.project_dir)
            for subdir in [".cache", "videos", "downloads", "media"]:
                d = project / subdir
                if d.exists():
                    dirs.append(d)

        # Global cache
        global_cache = Path.home() / ".matcher_global_cache"
        if global_cache.exists():
            dirs.append(global_cache)

        # Short path roots
        for root in ["E:/v", "D:/v", "C:/v"]:
            p = Path(root)
            if p.exists():
                dirs.append(p)

        return dirs

    def _resolve_missing_file(
        self,
        original_path: str,
        cache_dirs: List[Path],
        match: Any
    ) -> Optional[str]:
        """Try to find a missing file in various locations."""
        original = Path(original_path)
        filename = original.name
        stem = original.stem

        # Get video ID if available
        video_id = getattr(match, 'video_id', None) or getattr(match, 'source_id', None)

        extensions = ['.mp4', '.webm', '.mkv', '.mov', '.avi', '.mp3', '.wav', '.m4a']

        for cache_dir in cache_dirs:
            # Direct match
            direct = cache_dir / filename
            if direct.exists():
                return str(direct)

            # Match by video ID
            if video_id:
                for ext in extensions:
                    by_id = cache_dir / f"{video_id}{ext}"
                    if by_id.exists():
                        return str(by_id)

            # Search subdirectories (1 level)
            try:
                for subdir in cache_dir.iterdir():
                    if subdir.is_dir():
                        direct = subdir / filename
                        if direct.exists():
                            return str(direct)

                        if video_id:
                            for ext in extensions:
                                by_id = subdir / f"{video_id}{ext}"
                                if by_id.exists():
                                    return str(by_id)
            except PermissionError:
                continue

        return None

    def _get_video_path(self, match: Any) -> Optional[str]:
        """Get video path from match object."""
        for attr in ['video_path', 'source_file', 'file_path', 'path', 'file']:
            if hasattr(match, attr):
                return getattr(match, attr)
        return None

    def _set_video_path(self, match: Any, path: str):
        """Set video path on match object."""
        for attr in ['video_path', 'source_file', 'file_path', 'path', 'file']:
            if hasattr(match, attr):
                setattr(match, attr, path)
                return

    def _get_segment_start(self, match: Any) -> float:
        """Get segment start time."""
        segment = getattr(match, 'segment', None)
        if segment:
            return getattr(segment, 'start', 0) or 0
        return 0

    def _get_segment_end(self, match: Any) -> Optional[float]:
        """Get segment end time."""
        segment = getattr(match, 'segment', None)
        if segment:
            return getattr(segment, 'end', None)
        return None

    def _set_segment_end(self, match: Any, end: float):
        """Set segment end time."""
        segment = getattr(match, 'segment', None)
        if segment and hasattr(segment, 'end'):
            segment.end = end

    def _fix_segment_duration(self, segment: Any) -> int:
        """Fix segment duration, return count of fixes."""
        fixed = 0

        start = getattr(segment, 'start', 0) or 0
        end = getattr(segment, 'end', None)

        if end is not None:
            duration = end - start

            if duration <= 0:
                segment.end = start + self.MIN_CLIP_DURATION
                fixed += 1
            elif duration > self.MAX_CLIP_DURATION:
                segment.end = start + self.MAX_CLIP_DURATION
                fixed += 1

        if hasattr(segment, 'duration') and segment.duration is not None:
            if segment.duration <= 0:
                segment.duration = self.MIN_CLIP_DURATION
                fixed += 1
            elif segment.duration > self.MAX_CLIP_DURATION:
                segment.duration = self.MAX_CLIP_DURATION
                fixed += 1

        return fixed

    def _fix_match_times(self, match: Any) -> int:
        """Fix match start/end times."""
        fixed = 0

        start = getattr(match, 'start_time', None)
        end = getattr(match, 'end_time', None)

        if start is not None and end is not None:
            if end <= start:
                match.end_time = start + self.MIN_CLIP_DURATION
                fixed += 1
            elif (end - start) > self.MAX_CLIP_DURATION:
                match.end_time = start + self.MAX_CLIP_DURATION
                fixed += 1

        if start is not None and start < 0:
            match.start_time = 0
            fixed += 1

        return fixed

    def _fix_speed_adjustment(self, match: Any) -> int:
        """Fix speed/time_scalar values."""
        fixed = 0

        for attr in ['time_scalar', 'speed', 'speed_factor']:
            if hasattr(match, attr):
                value = getattr(match, attr)
                if value is not None:
                    if value < self.MIN_SPEED:
                        setattr(match, attr, self.MIN_SPEED)
                        fixed += 1
                    elif value > self.MAX_SPEED:
                        setattr(match, attr, self.MAX_SPEED)
                        fixed += 1

        return fixed

    def _sanitize_metadata(self, metadata: Dict) -> Dict:
        """Sanitize metadata for JSON serialization."""
        result = {}

        for key, value in metadata.items():
            # Skip non-serializable types
            if value is None:
                result[key] = None
                continue

            try:
                # Try JSON serialization
                json.dumps(value)
                result[key] = value
            except (TypeError, ValueError):
                # Convert numpy types
                if hasattr(value, 'tolist'):
                    result[key] = value.tolist()
                elif hasattr(value, 'item'):
                    result[key] = value.item()
                elif hasattr(value, '__float__'):
                    result[key] = float(value)
                elif hasattr(value, '__int__'):
                    result[key] = int(value)
                else:
                    # Convert to string as last resort
                    result[key] = str(value)

        return result

    def _sanitize_path(self, path: str) -> str:
        """Sanitize path for OTIO compatibility."""
        if not path:
            return path

        # Unicode replacements
        replacements = {
            '\u2019': "'", '\u2018': "'",
            '\u201c': '"', '\u201d': '"',
            '\u2013': '-', '\u2014': '-',
            '\u2026': '...', '\u00a0': ' ',
        }

        result = path
        for char, replacement in replacements.items():
            result = result.replace(char, replacement)

        # Get filename
        p = Path(result)
        clean_name = p.name

        # Remove non-ASCII from filename
        clean_name = re.sub(r'[^\x00-\x7F]+', '_', clean_name)

        # Remove Windows-invalid characters
        for char in '<>:"|?*':
            clean_name = clean_name.replace(char, '_')

        # Remove multiple underscores
        while '__' in clean_name:
            clean_name = clean_name.replace('__', '_')

        return str(p.parent / clean_name)

    def _set_config_attr(self, config: Any, attr: str, value: Any):
        """Safely set config attribute."""
        if isinstance(config, dict):
            config[attr] = value
        elif hasattr(config, attr):
            setattr(config, attr, value)

    # =========================================================================
    # PREFLIGHT CHECK
    # =========================================================================

    def preflight_check(self, state: 'PipelineState') -> List[str]:
        """
        Run preflight checks before timeline generation.

        Returns list of issues found.
        """
        issues = []

        # If no matches yet, nothing to check (not an error, just early in pipeline)
        if not hasattr(state, 'matches') or not state.matches:
            return issues  # Empty list - nothing to fix

        missing_media = 0
        invalid_duration = 0
        overlaps = 0

        sorted_matches = sorted(
            state.matches,
            key=lambda m: self._get_segment_start(m)
        )

        prev_end = 0.0
        for match in sorted_matches:
            # Check media
            video_path = self._get_video_path(match)
            if video_path and not Path(video_path).exists():
                missing_media += 1

            # Check duration
            segment = getattr(match, 'segment', None)
            if segment:
                start = getattr(segment, 'start', 0) or 0
                end = getattr(segment, 'end', None)
                if end is not None and end <= start:
                    invalid_duration += 1
                if start < prev_end:
                    overlaps += 1
                if end:
                    prev_end = end

        if missing_media > 0:
            issues.append(f"{missing_media} missing media files")
        if invalid_duration > 0:
            issues.append(f"{invalid_duration} clips with invalid duration")
        if overlaps > 0:
            issues.append(f"{overlaps} overlapping clips")

        return issues
