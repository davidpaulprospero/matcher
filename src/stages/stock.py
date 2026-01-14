"""
Stock Video Stage - Download Generic Stock Footage

Stage 2c of the video matching pipeline:
- Downloads generic stock footage (B-roll) from Pexels and Pixabay based on keywords
- Different from EntityVideosStage: downloads general B-roll, not entity-specific content
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

from . import Stage, StageResult, register_stage
from ..state import DownloadedVideo

if TYPE_CHECKING:
    from ..config import Config
    from ..checkpoint import CheckpointManager
    from ..state import PipelineState

logger = logging.getLogger(__name__)


@register_stage
class StockVideoStage(Stage):
    """
    Downloads generic stock footage for keywords.

    Inputs:
        - state.keywords: Search keywords

    Outputs:
        - Appends to state.downloaded_videos with source='stock'

    Configuration:
        - config.enhanced.enabled: Master on/off switch (legacy)
        - config.enhanced.enable_pexels: Enable Pexels API
        - config.enhanced.enable_pixabay: Enable Pixabay API
        - config.enhanced.stock_per_keyword: Videos per keyword (default: 3)
        - config.downloaded_videos_dir: Base directory for downloads
    """

    name = "STOCK"
    description = "Download generic stock footage from Pexels/Pixabay"

    def run(
        self,
        state: 'PipelineState',
        config: 'Config',
        checkpoint: 'CheckpointManager'
    ) -> StageResult:
        """Execute the stock video stage"""
        warnings = []

        # Check if skip_download is enabled
        if getattr(config.pipeline, 'skip_download', False):
            logger.debug("Download skipped via pipeline.skip_download")
            return StageResult.ok({
                'skipped': True,
                'reason': 'skip_download_enabled'
            })

        # Check if enhanced features are enabled (legacy check)
        if not getattr(config, 'enhanced', None) or not config.enhanced.enabled:
            logger.debug("Enhanced features disabled in config")
            return StageResult.ok({
                'skipped': True,
                'reason': 'enhanced_disabled'
            })

        # Check if either Pexels or Pixabay is enabled
        enable_pexels = getattr(config.enhanced, 'enable_pexels', False)
        enable_pixabay = getattr(config.enhanced, 'enable_pixabay', False)

        if not enable_pexels and not enable_pixabay:
            logger.debug("Neither Pexels nor Pixabay enabled")
            return StageResult.ok({
                'skipped': True,
                'reason': 'no_stock_sources_enabled'
            })

        # Check if keywords exist
        if not state.keywords:
            logger.debug("No keywords available for stock footage search")
            return StageResult.ok({
                'skipped': True,
                'reason': 'no_keywords'
            })

        print(f"\n  ─── Stage 2c: DOWNLOAD STOCK FOOTAGE ───")

        try:
            # Create stock directory
            stock_dir = Path(config.downloaded_videos_dir) / "stock"
            stock_dir.mkdir(parents=True, exist_ok=True)

            # Limit keywords for API rate limits
            stock_keywords = state.keywords[:15]
            per_keyword = getattr(config.enhanced, 'stock_per_keyword', 3)

            print(f"  Searching {len(stock_keywords)} keywords for stock footage")
            print(f"  Videos per keyword: {per_keyword}")

            total_paths = []
            sources = {'pexels': 0, 'pixabay': 0}

            # Download from Pexels
            if enable_pexels:
                pexels_paths, pexels_counts = self._download_pexels(
                    stock_keywords,
                    str(stock_dir),
                    per_keyword,
                    config
                )
                total_paths.extend(pexels_paths)
                sources['pexels'] = len(pexels_paths)
                print(f"  ✓ Pexels: {len(pexels_paths)} videos")

                # Track failed keywords
                for kw, count in pexels_counts.items():
                    if count == 0 and kw not in state.failed_keywords:
                        state.failed_keywords.append(kw)

            # Download from Pixabay
            if enable_pixabay:
                pixabay_paths, pixabay_counts = self._download_pixabay(
                    stock_keywords,
                    str(stock_dir),
                    per_keyword,
                    config
                )
                total_paths.extend(pixabay_paths)
                sources['pixabay'] = len(pixabay_paths)
                print(f"  ✓ Pixabay: {len(pixabay_paths)} videos")

                # Track failed keywords
                for kw, count in pixabay_counts.items():
                    if count == 0 and kw not in state.failed_keywords:
                        state.failed_keywords.append(kw)

            print(f"  ✓ Total stock footage: {len(total_paths)} videos")

            # Add to state.downloaded_videos
            for path in total_paths:
                # Check if already in downloaded_videos
                path_str = str(path)
                if not any(v.file == path_str for v in state.downloaded_videos):
                    state.downloaded_videos.append(DownloadedVideo(
                        file=path_str,
                        source='stock',
                        keyword='stock_footage',
                        url='',
                        title=Path(path).stem,
                        duration=0.0
                    ))

            # Build checkpoint data
            checkpoint_data = {
                'stock_count': len(total_paths),
                'stock_paths': [str(p) for p in total_paths],
                'sources': sources
            }

            return StageResult.ok(checkpoint_data, warnings=warnings)

        except Exception as e:
            logger.error(f"Stock footage download failed: {e}", exc_info=True)
            print(f"  ⚠ Stock footage download failed: {e}")
            return StageResult.fail(str(e))

    def can_skip(
        self,
        state: 'PipelineState',
        checkpoint: 'CheckpointManager'
    ) -> bool:
        """Check if this stage can be skipped (already completed)"""
        return checkpoint.should_skip_stage(self.name)

    def restore(
        self,
        state: 'PipelineState',
        checkpoint: 'CheckpointManager',
        config: 'Config' = None
    ) -> bool:
        """Restore stock videos from checkpoint data"""
        try:
            data = checkpoint.get_stage_data(self.name)
            if not data:
                logger.warning(f"No checkpoint data for {self.name}")
                return False

            # Re-add stock videos to downloaded_videos
            stock_paths = data.get('stock_paths', [])
            existing_paths = [p for p in stock_paths if Path(p).exists()]

            for path_str in existing_paths:
                # Check if already in downloaded_videos
                if not any(v.file == path_str for v in state.downloaded_videos):
                    state.downloaded_videos.append(DownloadedVideo(
                        file=path_str,
                        source='stock',
                        keyword='stock_footage',
                        url='',
                        title=Path(path_str).stem,
                        duration=0.0
                    ))

            logger.info(f"Restored {self.name}: {len(existing_paths)} stock videos")

            if len(existing_paths) == 0:
                logger.warning("No stock videos found on disk")
                return False

            return True

        except Exception as e:
            logger.error(f"Failed to restore {self.name}: {e}", exc_info=True)
            return False

    def validate_inputs(
        self,
        state: 'PipelineState',
        config: 'Config'
    ) -> Optional[str]:
        """
        Validate inputs before running.

        This is an optional stage, so we don't enforce hard requirements.
        """
        # Optional stage - no hard requirements
        return None

    def _download_pexels(
        self,
        keywords: List[str],
        output_dir: str,
        per_keyword: int,
        config: 'Config'
    ) -> Tuple[List[str], Dict[str, int]]:
        """
        Download Pexels footage.

        Returns:
            Tuple of (downloaded_paths, keyword_counts)
        """
        try:
            from ..pexels import download_pexels_footage

            print(f"  Pexels: Searching {len(keywords)} keywords...")

            paths, counts = download_pexels_footage(
                keywords,
                output_dir,
                per_keyword=per_keyword
            )

            return paths, counts

        except ImportError:
            logger.warning("Pexels module not available")
            return [], {}
        except Exception as e:
            logger.error(f"Pexels download error: {e}")
            print(f"  ⚠ Pexels error: {e}")
            return [], {}

    def _download_pixabay(
        self,
        keywords: List[str],
        output_dir: str,
        per_keyword: int,
        config: 'Config'
    ) -> Tuple[List[str], Dict[str, int]]:
        """
        Download Pixabay footage.

        Returns:
            Tuple of (downloaded_paths, keyword_counts)
        """
        try:
            from ..pixabay import download_pixabay_footage

            print(f"  Pixabay: Searching {len(keywords)} keywords...")

            paths, counts = download_pixabay_footage(
                keywords,
                output_dir,
                per_keyword=per_keyword
            )

            return paths, counts

        except ImportError:
            logger.warning("Pixabay module not available")
            return [], {}
        except Exception as e:
            logger.error(f"Pixabay download error: {e}")
            print(f"  ⚠ Pixabay error: {e}")
            return [], {}
