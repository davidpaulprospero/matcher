"""
Stock Footage Stage - Download generic stock videos for timeline cadence.

This stage is intentionally separate from entity tracks:
- Entity tracks are driven by extracted named entities.
- Stock footage track (V10) is driven by generic segment/topic queries.
"""

from __future__ import annotations

import logging
import os
import re
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from . import Stage, StageResult
from ..logging_templates import log_error_with_context, log_stage_complete

if TYPE_CHECKING:
    from ..config import Config
    from ..checkpoint import CheckpointManager
    from ..state import PipelineState

logger = logging.getLogger(__name__)

_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in", "is",
    "it", "its", "of", "on", "or", "that", "the", "to", "was", "were", "with",
}


class StockFootageStage(Stage):
    """Download generic (non-entity) stock clips for V10 cadence placement."""

    name = "STOCK_FOOTAGE"
    description = "Download generic stock videos from Pexels/Pixabay"
    DEPENDS_ON = ["ANALYZE"]

    def run(
        self,
        state: "PipelineState",
        config: "Config",
        checkpoint: "CheckpointManager",
    ) -> StageResult:
        stage_start_time = time.time()
        warnings: List[str] = []

        stock_cfg = getattr(config, "stock_footage", None)
        if not stock_cfg or not getattr(stock_cfg, "enabled", True):
            logger.info("Skipping STOCK_FOOTAGE stage (config: stock_footage.enabled=false)")
            return StageResult.ok({"skipped": True, "reason": "disabled"})

        if not state.voiceover_segments:
            logger.info("Skipping STOCK_FOOTAGE stage (no voiceover segments)")
            return StageResult.ok({"skipped": True, "reason": "no_segments"})

        segment_interval = max(1, int(getattr(stock_cfg, "segment_interval", 3)))
        selection_mode = str(getattr(stock_cfg, "selection_mode", "best_in_block"))
        clips_per_segment = max(1, int(getattr(stock_cfg, "max_clips_per_selected_segment", 1)))

        try:
            selected = self._select_segments(
                voiceover_segments=state.voiceover_segments,
                keywords=state.keywords or [],
                interval=segment_interval,
                mode=selection_mode,
            )

            if not selected:
                logger.info("No segments selected for stock footage")
                state.stock_videos = {}
                return StageResult.ok({"segment_count": 0, "segments": {}})

            segment_queries: Dict[int, str] = {}
            for seg_idx, seg in selected:
                query = self._build_query(
                    segment_text=getattr(seg, "text", "") or "",
                    keywords=state.keywords or [],
                    topic_context=state.topic_context or "",
                )
                if query:
                    segment_queries[seg_idx] = query

            if not segment_queries:
                logger.info("No valid stock footage queries built")
                state.stock_videos = {}
                return StageResult.ok({"segment_count": 0, "segments": {}})

            from ..media_sources import download_stock_videos

            output_dir = self._get_output_dir(checkpoint)
            output_dir.mkdir(parents=True, exist_ok=True)

            min_duration = float(getattr(stock_cfg, "min_duration", 5))
            max_duration = float(getattr(stock_cfg, "max_duration", 60))
            prefer_landscape = bool(getattr(stock_cfg, "prefer_landscape", True))

            pexels_key = None
            pixabay_key = None
            if getattr(stock_cfg, "pexels_enabled", True):
                pexels_key = getattr(config.api_keys, "pexels_api_key", "") or os.getenv("PEXELS_API_KEY")
            if getattr(stock_cfg, "pixabay_enabled", True):
                pixabay_key = getattr(config.api_keys, "pixabay_api_key", "") or os.getenv("PIXABAY_API_KEY")

            stock_results = download_stock_videos(
                segment_queries=segment_queries,
                output_dir=str(output_dir),
                clips_per_segment=clips_per_segment,
                min_duration=min_duration,
                max_duration=max_duration,
                prefer_hd=prefer_landscape,
                pexels_key=pexels_key,
                pixabay_key=pixabay_key,
                config=config,
            )

            state.stock_videos = stock_results
            checkpoint_data = self._build_checkpoint_data(stock_results)

            elapsed = time.time() - stage_start_time
            total_videos = sum(len(item.get("videos", [])) for item in stock_results.values())
            log_stage_complete(
                logger,
                "STOCK_FOOTAGE",
                elapsed_seconds=elapsed,
                segments_selected=len(segment_queries),
                videos_downloaded=total_videos,
            )
            return StageResult.ok(checkpoint_data, warnings=warnings)

        except ImportError as e:
            err = f"Could not import stock footage downloader: {e}"
            log_error_with_context(logger, "PIPE-001", err)
            return StageResult.fail(err)
        except Exception as e:
            log_error_with_context(logger, "SEARCH-001", f"Stock footage stage failed: {e}")
            return StageResult.fail(str(e))

    def can_skip(self, state: "PipelineState", checkpoint: "CheckpointManager") -> bool:
        return checkpoint.should_skip_stage(self.name)

    def restore(
        self,
        state: "PipelineState",
        checkpoint: "CheckpointManager",
        config: "Config" = None,
    ) -> bool:
        try:
            data = checkpoint.get_stage_data(self.name)
            if not data:
                logger.warning(f"No checkpoint data for {self.name}")
                return False

            raw_segments = data.get("segments", {})
            if not isinstance(raw_segments, dict):
                logger.warning(f"Invalid STOCK_FOOTAGE checkpoint payload type: {type(raw_segments).__name__}")
                return False

            restored: Dict[int, Dict[str, Any]] = {}
            for seg_key, seg_info in raw_segments.items():
                try:
                    seg_idx = int(seg_key)
                except (TypeError, ValueError):
                    continue
                videos = seg_info.get("videos", []) if isinstance(seg_info, dict) else []
                existing = [p for p in videos if Path(p).exists()]
                if not existing:
                    continue
                restored[seg_idx] = {
                    "segment_index": seg_idx,
                    "query": seg_info.get("query", "") if isinstance(seg_info, dict) else "",
                    "videos": existing,
                    "sources": seg_info.get("sources", []) if isinstance(seg_info, dict) else [],
                }

            state.stock_videos = restored
            logger.info(f"Restored {self.name}: {len(restored)} segment entries")
            return len(restored) > 0
        except Exception as e:
            log_error_with_context(logger, "PIPE-002", f"Failed to restore {self.name}: {e}")
            return False

    def validate_inputs(self, state: "PipelineState", config: "Config") -> Optional[str]:
        # Optional stage with graceful skips.
        return None

    def get_input_output_info(self, state: "PipelineState", config: "Config") -> Dict[str, Any]:
        return {
            "inputs": "voiceover segments, keywords, topic context",
            "outputs": "stock videos by segment",
            "input_count": len(state.voiceover_segments) if state.voiceover_segments else 0,
            "output_count": len(state.stock_videos) if state.stock_videos else 0,
        }

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return [w for w in re.findall(r"[A-Za-z0-9']+", text.lower()) if w and w not in _STOPWORDS]

    def _score_segment(self, text: str, keywords: List[str]) -> float:
        words = self._tokenize(text)
        if not words:
            return 0.0
        seg_terms = set(words)
        keyword_terms = set(self._tokenize(" ".join(keywords)))
        overlap = len(seg_terms & keyword_terms)
        return (2.0 * overlap) + min(len(words), 15) / 15.0

    def _select_segments(
        self,
        voiceover_segments: List[Any],
        keywords: List[str],
        interval: int,
        mode: str,
    ) -> List[Tuple[int, Any]]:
        selected: List[Tuple[int, Any]] = []
        if not voiceover_segments:
            return selected

        rotate_offset = 0
        for block_start in range(0, len(voiceover_segments), interval):
            block = list(enumerate(voiceover_segments[block_start:block_start + interval], start=block_start))
            if not block:
                continue

            if mode == "first_in_block":
                selected.append(block[0])
                continue

            if mode == "rotate_in_block":
                pick_idx = min(rotate_offset % len(block), len(block) - 1)
                selected.append(block[pick_idx])
                rotate_offset += 1
                continue

            # Default: best_in_block
            best = max(
                block,
                key=lambda pair: (
                    self._score_segment(getattr(pair[1], "text", "") or "", keywords),
                    -pair[0],  # tie-breaker: earlier segment
                ),
            )
            selected.append(best)

        return selected

    def _build_query(self, segment_text: str, keywords: List[str], topic_context: str) -> str:
        seg_terms = self._tokenize(segment_text)
        kw_terms = self._tokenize(" ".join(keywords))
        topic_terms = self._tokenize(topic_context)

        parts: List[str] = []
        for term in seg_terms[:5]:
            if term not in parts:
                parts.append(term)
        for term in kw_terms[:3]:
            if term not in parts:
                parts.append(term)
        for term in topic_terms[:2]:
            if term not in parts:
                parts.append(term)

        if not parts:
            return "documentary b-roll"
        return " ".join(parts)

    def _get_output_dir(self, checkpoint: "CheckpointManager") -> Path:
        return checkpoint.project_dir / "stock"

    def _build_checkpoint_data(self, results: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
        if not results:
            return {"segment_count": 0, "total_videos": 0, "segments": {}}

        segments: Dict[str, Dict[str, Any]] = {}
        total_videos = 0
        for seg_idx, info in results.items():
            videos = list(info.get("videos", []))
            total_videos += len(videos)
            segments[str(seg_idx)] = {
                "segment_index": seg_idx,
                "query": info.get("query", ""),
                "videos": videos,
                "sources": list(info.get("sources", [])),
            }

        return {
            "segment_count": len(segments),
            "total_videos": total_videos,
            "segments": segments,
        }

