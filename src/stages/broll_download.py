"""
B-roll Download Stage - Download B-roll Footage

Stage 2d of the video matching pipeline:
- Downloads videos specifically searched as B-roll
- Search terms generated from keywords/entities with B-roll suffixes
- Sources: YouTube first, then Pexels/Pixabay stock

Created during B-roll improvement refactoring (Jan 2026).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple

from . import Stage, StageResult, register_stage

if TYPE_CHECKING:
    from ..config import Config
    from ..checkpoint import CheckpointManager
    from ..state import PipelineState

logger = logging.getLogger(__name__)


@register_stage
class BrollDownloadStage(Stage):
    """
    Downloads B-roll footage using keyword-derived search terms.

    Inputs:
        - state.keywords: Search keywords
        - state.extracted_entities: Entities (people, places)

    Outputs:
        - state.broll_downloads: List of BrollDownload
        - Appends to state.downloaded_videos with source='broll'

    Configuration (config.broll):
        - enabled: Master on/off switch
        - download_enabled: Enable download phase
        - downloads_per_term: Videos per search term (default: 3)
        - max_total_downloads: Cap on total B-roll downloads (default: 30)
        - search_suffixes: ["b-roll", "footage", "cinematic"]
        - include_generic_searches: Include generic B-roll searches
    """

    name = "BROLL_DOWNLOAD"
    description = "Download B-roll footage based on keywords"

    def run(
        self,
        state: 'PipelineState',
        config: 'Config',
        checkpoint: 'CheckpointManager'
    ) -> StageResult:
        """Execute the B-roll download stage"""
        warnings = []

        # Check master enable
        broll_config = getattr(config, 'broll', None)
        if not broll_config or not getattr(broll_config, 'enabled', True):
            logger.debug("B-roll disabled in config")
            return StageResult.ok({
                'skipped': True,
                'reason': 'broll_disabled'
            })

        # Check download enable
        if not getattr(broll_config, 'download_enabled', True):
            logger.debug("B-roll download disabled in config")
            return StageResult.ok({
                'skipped': True,
                'reason': 'download_disabled'
            })

        # Check if skip_download is enabled
        if getattr(config.pipeline, 'skip_download', False):
            logger.debug("Download skipped via pipeline.skip_download")
            return StageResult.ok({
                'skipped': True,
                'reason': 'skip_download_enabled'
            })

        # Check if keywords exist
        if not state.keywords:
            logger.debug("No keywords available for B-roll search")
            return StageResult.ok({
                'skipped': True,
                'reason': 'no_keywords'
            })

        print(f"\n  ─── Stage 2d: DOWNLOAD B-ROLL FOOTAGE ───")

        try:
            # Generate search terms
            search_terms = self._generate_search_terms(state, broll_config)

            if not search_terms:
                logger.info("No B-roll search terms generated")
                return StageResult.ok({
                    'skipped': True,
                    'reason': 'no_search_terms'
                })

            print(f"  Generated {len(search_terms)} B-roll search terms")

            # Get download limits
            per_term = getattr(broll_config, 'downloads_per_term', 3)
            max_total = getattr(broll_config, 'max_total_downloads', 30)
            duration_tier = getattr(broll_config, 'duration_tier', 'short')

            print(f"  Downloads per term: {per_term}, Max total: {max_total}")

            # Create B-roll directory
            broll_dir = Path(config.downloaded_videos_dir) / "broll"
            broll_dir.mkdir(parents=True, exist_ok=True)

            # Initialize state.broll_downloads if needed
            if not hasattr(state, 'broll_downloads'):
                state.broll_downloads = []

            # Download from YouTube first
            youtube_paths = self._download_youtube_broll(
                search_terms,
                str(broll_dir),
                per_term,
                max_total,
                duration_tier,
                config
            )

            downloaded_count = len(youtube_paths)
            print(f"  ✓ YouTube B-roll: {downloaded_count} videos")

            # Download from stock APIs if under limit
            stock_paths = []
            remaining = max_total - downloaded_count
            if remaining > 0:
                stock_paths = self._download_stock_broll(
                    search_terms,
                    str(broll_dir),
                    per_term,
                    remaining,
                    config
                )
                print(f"  ✓ Stock B-roll: {len(stock_paths)} videos")

            total_paths = youtube_paths + stock_paths
            print(f"  ✓ Total B-roll downloads: {len(total_paths)} videos")

            # Add to state
            from ..state import DownloadedVideo
            for path, source, term in total_paths:
                # Check if already in downloaded_videos
                path_str = str(path)
                if not any(v.file == path_str for v in state.downloaded_videos):
                    state.downloaded_videos.append(DownloadedVideo(
                        file=path_str,
                        source='broll',
                        keyword=term,
                        url='',
                        title=Path(path).stem,
                        duration=0.0
                    ))

                # Store in broll_downloads for tracking
                state.broll_downloads.append({
                    'file': path_str,
                    'source': source,
                    'search_term': term
                })

            # Build checkpoint data
            checkpoint_data = {
                'broll_count': len(total_paths),
                'broll_paths': [str(p[0]) for p in total_paths],
                'search_terms': search_terms,
                'youtube_count': len(youtube_paths),
                'stock_count': len(stock_paths)
            }

            return StageResult.ok(checkpoint_data, warnings=warnings)

        except Exception as e:
            logger.error(f"B-roll download failed: {e}", exc_info=True)
            print(f"  ⚠ B-roll download failed: {e}")
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
        """Restore B-roll downloads from checkpoint data"""
        try:
            data = checkpoint.get_stage_data(self.name)
            if not data:
                logger.warning(f"No checkpoint data for {self.name}")
                return False

            # Re-add B-roll videos to downloaded_videos
            broll_paths = data.get('broll_paths', [])
            existing_paths = [p for p in broll_paths if Path(p).exists()]

            from ..state import DownloadedVideo
            for path_str in existing_paths:
                # Check if already in downloaded_videos
                if not any(v.file == path_str for v in state.downloaded_videos):
                    state.downloaded_videos.append(DownloadedVideo(
                        file=path_str,
                        source='broll',
                        keyword='broll',
                        url='',
                        title=Path(path_str).stem,
                        duration=0.0
                    ))

            # Initialize broll_downloads if needed
            if not hasattr(state, 'broll_downloads'):
                state.broll_downloads = []

            for path_str in existing_paths:
                state.broll_downloads.append({
                    'file': path_str,
                    'source': 'restored',
                    'search_term': ''
                })

            logger.info(f"Restored {self.name}: {len(existing_paths)} B-roll videos")

            if len(existing_paths) == 0:
                logger.warning("No B-roll videos found on disk")
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
        return None

    def _generate_search_terms(
        self,
        state: 'PipelineState',
        broll_config
    ) -> List[str]:
        """
        Generate B-roll search terms from keywords and entities.

        Returns:
            List of search terms like "[keyword] b-roll", "[entity] footage"
        """
        search_terms = []
        seen_terms: Set[str] = set()

        # Get config
        suffixes = getattr(broll_config, 'search_suffixes', ['b-roll', 'footage', 'cinematic'])
        include_generic = getattr(broll_config, 'include_generic_searches', True)

        # Generate terms from keywords (limit to top 10)
        for keyword in state.keywords[:10]:
            for suffix in suffixes:
                term = f"{keyword} {suffix}"
                term_lower = term.lower()
                if term_lower not in seen_terms:
                    search_terms.append(term)
                    seen_terms.add(term_lower)

        # Generate terms from entities
        for entity in state.extracted_entities[:5]:
            entity_name = entity.get('name', '') if isinstance(entity, dict) else str(entity)
            entity_type = entity.get('type', '') if isinstance(entity, dict) else ''

            if not entity_name:
                continue

            # Add entity-specific suffixes based on type
            entity_suffixes = suffixes.copy()
            if entity_type == 'LOCATION':
                entity_suffixes.extend(['aerial', 'drone', 'timelapse'])
            elif entity_type == 'PERSON':
                # Skip person entities for B-roll (they need specific footage)
                continue

            for suffix in entity_suffixes[:3]:  # Limit suffixes per entity
                term = f"{entity_name} {suffix}"
                term_lower = term.lower()
                if term_lower not in seen_terms:
                    search_terms.append(term)
                    seen_terms.add(term_lower)

        # Add generic B-roll searches
        if include_generic:
            generic_terms = [
                "stock footage compilation",
                "cinematic footage 4k",
                "aerial drone footage",
                "nature b-roll",
                "urban timelapse"
            ]
            for term in generic_terms:
                if term.lower() not in seen_terms:
                    search_terms.append(term)
                    seen_terms.add(term.lower())

        return search_terms

    def _download_youtube_broll(
        self,
        search_terms: List[str],
        output_dir: str,
        per_term: int,
        max_total: int,
        duration_tier: str,
        config: 'Config'
    ) -> List[Tuple[str, str, str]]:
        """
        Download B-roll from YouTube.

        Returns:
            List of (path, source, search_term) tuples
        """
        try:
            from ..downloader import VideoDownloader

            downloader = VideoDownloader(config=config)

            downloaded = []
            for term in search_terms:
                if len(downloaded) >= max_total:
                    break

                try:
                    # Create term-specific subfolder
                    term_slug = term.lower().replace(' ', '_')[:30]
                    term_dir = Path(output_dir) / f"yt_{term_slug}"
                    term_dir.mkdir(parents=True, exist_ok=True)

                    # Download videos for this term
                    remaining = min(per_term, max_total - len(downloaded))
                    videos, _ = downloader.download_all(
                        keywords=[term],
                        output_dir=term_dir,
                        resume=False,
                        topic=f"B-roll footage for: {term}"
                    )

                    for vid in videos[:remaining]:
                        path = vid.file if hasattr(vid, 'file') else vid.get('file', vid.get('path', ''))
                        if path and Path(path).exists():
                            downloaded.append((path, 'youtube', term))

                except Exception as e:
                    logger.warning(f"YouTube B-roll search failed for '{term}': {e}")
                    continue

            return downloaded

        except ImportError:
            logger.warning("VideoDownloader not available")
            return []
        except Exception as e:
            logger.error(f"YouTube B-roll download error: {e}")
            return []

    def _download_stock_broll(
        self,
        search_terms: List[str],
        output_dir: str,
        per_term: int,
        max_total: int,
        config: 'Config'
    ) -> List[Tuple[str, str, str]]:
        """
        Download B-roll from stock APIs (Pexels, Pixabay).

        Returns:
            List of (path, source, search_term) tuples
        """
        downloaded = []

        # Try Pexels
        try:
            from ..pexels import download_pexels_footage

            pexels_terms = search_terms[:10]  # Limit API calls
            pexels_dir = Path(output_dir) / "pexels"
            pexels_dir.mkdir(parents=True, exist_ok=True)

            paths, _ = download_pexels_footage(
                pexels_terms,
                str(pexels_dir),
                per_keyword=min(per_term, 2)
            )

            for path in paths[:max_total // 2]:
                if len(downloaded) >= max_total:
                    break
                downloaded.append((path, 'pexels', 'stock'))

        except ImportError:
            logger.debug("Pexels module not available")
        except Exception as e:
            logger.warning(f"Pexels B-roll error: {e}")

        # Try Pixabay
        remaining = max_total - len(downloaded)
        if remaining > 0:
            try:
                from ..pixabay import download_pixabay_footage

                pixabay_terms = search_terms[:10]
                pixabay_dir = Path(output_dir) / "pixabay"
                pixabay_dir.mkdir(parents=True, exist_ok=True)

                paths, _ = download_pixabay_footage(
                    pixabay_terms,
                    str(pixabay_dir),
                    per_keyword=min(per_term, 2)
                )

                for path in paths[:remaining]:
                    downloaded.append((path, 'pixabay', 'stock'))

            except ImportError:
                logger.debug("Pixabay module not available")
            except Exception as e:
                logger.warning(f"Pixabay B-roll error: {e}")

        return downloaded
