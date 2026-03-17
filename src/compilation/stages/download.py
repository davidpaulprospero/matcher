"""
Download Stage

Downloads filtered candidates using VideoDownloader with full rate limiting stack.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Any, List

from ..state import CompilationState, DownloadedClip

if TYPE_CHECKING:
    from src.config import Config

logger = logging.getLogger(__name__)


class DownloadStage:
    """
    Download videos for compilation.

    Wraps VideoDownloader to inherit all rate limiting features:
    - 7-tier bypass system
    - Cookie rotation
    - Proxy/VPN/Tor support
    - Exponential backoff
    """

    name = "DOWNLOAD"
    description = "Download videos with rate limiting"

    def __init__(self, config: 'Config'):
        self.config = config
        self._downloader = None

    @property
    def downloader(self):
        """Lazy-load VideoDownloader."""
        if self._downloader is None:
            from src.downloader.core import VideoDownloader
            self._downloader = VideoDownloader(self.config)
        return self._downloader

    def run(self, state: CompilationState, compilation_config: Dict[str, Any]) -> bool:
        """
        Download filtered candidates.

        Args:
            state: Compilation state (modified in place)
            compilation_config: Compilation config dict

        Returns:
            True if downloads succeeded
        """
        download_config = compilation_config.get('download', {})
        output_subdir = download_config.get('output_subdir', 'videos')

        project_dir = state.get_project_path()
        output_dir = project_dir / output_subdir
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"[DOWNLOAD] Downloading {len(state.filtered_candidates)} videos")
        logger.info(f"  Output: {output_dir}")

        # Group candidates by keyword for organized storage
        by_keyword: Dict[str, List] = {}
        for candidate in state.filtered_candidates:
            keyword = candidate.keyword or "general"
            if keyword not in by_keyword:
                by_keyword[keyword] = []
            by_keyword[keyword].append(candidate)

        downloaded_clips = []
        failed_count = 0

        for keyword, candidates in by_keyword.items():
            # Sanitize keyword for directory name
            safe_keyword = self._sanitize_dirname(keyword)
            keyword_dir = output_dir / safe_keyword
            keyword_dir.mkdir(parents=True, exist_ok=True)

            logger.info(f"  Downloading {len(candidates)} videos for: {keyword}")

            video_ids = [c.video_id for c in candidates]

            try:
                # Use VideoDownloader which has all rate limiting built in
                results = self.downloader._download_by_ids(
                    video_ids=video_ids,
                    keyword_dir=keyword_dir,
                    output_dir=output_dir,
                    keyword=keyword,
                    tier="short"  # We've already filtered by duration
                )

                # Convert DownloadedVideo to DownloadedClip
                for result in results:
                    # Find the matching candidate
                    vid_id = self._extract_video_id(result.file)
                    candidate = next(
                        (c for c in candidates if c.video_id == vid_id),
                        None
                    )

                    # Get actual file path
                    file_path = output_dir / result.file
                    if not file_path.exists():
                        file_path = keyword_dir / Path(result.file).name
                        if not file_path.exists():
                            logger.warning(f"    File not found: {result.file}")
                            failed_count += 1
                            continue

                    # Get actual duration from file
                    actual_duration = self._get_video_duration(file_path)
                    if actual_duration <= 0:
                        actual_duration = candidate.duration if candidate else 30.0

                    clip = DownloadedClip(
                        file=str(file_path),
                        video_id=vid_id or result.url.split("v=")[-1][:11] if result.url else "",
                        actual_duration=actual_duration,
                        keyword=keyword,
                        title=candidate.title if candidate else result.title
                    )
                    downloaded_clips.append(clip)

                logger.info(f"    Downloaded {len(results)} videos")

            except Exception as e:
                logger.error(f"    Download error for {keyword}: {e}")
                failed_count += len(candidates)

        # Update state
        state.downloaded_clips.extend(downloaded_clips)

        total_duration = sum(c.actual_duration for c in downloaded_clips)
        logger.info(f"[DOWNLOAD] Downloaded {len(downloaded_clips)} clips")
        logger.info(f"  Total duration: {total_duration:.0f}s ({total_duration/60:.1f} min)")
        logger.info(f"  Failed: {failed_count}")

        return len(downloaded_clips) > 0

    def _sanitize_dirname(self, name: str) -> str:
        """Sanitize string for use as directory name."""
        import re
        # Replace problematic characters
        safe = re.sub(r'[<>:"/\\|?*]', '_', name)
        # Limit length
        safe = safe[:50]
        return safe.strip() or "general"

    def _extract_video_id(self, filename: str) -> str:
        """Extract YouTube video ID from filename."""
        import re
        # Common pattern: title_VIDEOID.ext
        match = re.search(r'_([a-zA-Z0-9_-]{11})\.[^.]+$', filename)
        if match:
            return match.group(1)
        # Try just finding an 11-char alphanumeric sequence
        match = re.search(r'([a-zA-Z0-9_-]{11})', filename)
        if match:
            return match.group(1)
        return ""

    def _get_video_duration(self, file_path: Path) -> float:
        """Get video duration using ffprobe."""
        import subprocess
        from ...downloader.utils import SUBPROCESS_FLAGS

        try:
            cmd = [
                'ffprobe',
                '-v', 'error',
                '-show_entries', 'format=duration',
                '-of', 'default=noprint_wrappers=1:nokey=1',
                str(file_path)
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=10,
                encoding='utf-8',
                errors='replace',
                **SUBPROCESS_FLAGS
            )

            if result.returncode == 0 and result.stdout.strip():
                return float(result.stdout.strip())

        except Exception as e:
            logger.debug(f"Could not get duration for {file_path}: {e}")

        return 0.0
