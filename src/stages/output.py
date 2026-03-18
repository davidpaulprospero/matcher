"""
Output Stage - Timeline and Report Generation

Stage 7 of the simplified 7-stage pipeline:
- Generates OTIO timeline (split or single)
- Creates segment map for post-edit analysis
- Generates EDL export
- Generates DaVinci Resolve XML with bins
- Creates match report
"""

from __future__ import annotations

import logging
import re
import time
from datetime import datetime
from pathlib import Path
import json
from typing import TYPE_CHECKING, Any, Dict, List, Optional, NamedTuple

from . import Stage, StageResult, register_stage, validate_required_state_attrs
from ..matching.match_normalizer import MatchNormalizer
from ..logging_templates import log_stage_start, log_stage_complete, log_error_with_context


class SegmentInfo(NamedTuple):
    """Info about a downloaded video segment for path resolution."""
    video_id: str
    file: str
    original_start: float
    original_end: float


class SegmentPathIndex:
    """
    O(1) lookup index for video segment files.

    Maps (video_id, start, end) tuples to file paths for fast segment resolution.
    Per CLAUDE.md rule 38, checks BOTH legacy *_segments dirs AND flat download dir.
    Flat dir takes precedence when segment exists in both.
    """

    def __init__(self, segments: List[SegmentInfo]):
        """
        Build index from segment list.

        Args:
            segments: List of SegmentInfo objects from _scan_video_segments
        """
        self._segments = segments
        self._index: Dict[tuple, str] = {}
        self._video_ids: set = set()
        self._build_index(segments)

    def _build_index(self, segments: List[SegmentInfo]):
        """Build the lookup index from segments. Flat dir takes precedence."""
        # First pass: add legacy segments
        for seg in segments:
            key = (seg.video_id, int(seg.original_start), int(seg.original_end))
            if key not in self._index:
                self._index[key] = seg.file
                self._video_ids.add(seg.video_id)

        # Second pass: add flat dir segments (they take precedence)
        # Flat dir segments have more precise end times, so they override legacy
        for seg in segments:
            key = (seg.video_id, int(seg.original_start), int(seg.original_end))
            self._index[key] = seg.file  # Overwrites legacy if present
            self._video_ids.add(seg.video_id)

    @property
    def segments(self) -> List[SegmentInfo]:
        """Return the original segment list for backward compatibility."""
        return self._segments

    def lookup(self, video_id: str, start: float, end: float) -> Optional[str]:
        """
        Look up file path for a segment.

        Args:
            video_id: YouTube video ID
            start: Start time in seconds
            end: End time in seconds

        Returns:
            File path if found, None otherwise
        """
        key = (video_id, int(start), int(end))
        return self._index.get(key)

    def get_by_video_id(self, video_id: str) -> List[tuple]:
        """Get all (start, end, file) tuples for a video ID."""
        return [
            (start, end, path)
            for (vid, start, end), path in self._index.items()
            if vid == video_id
        ]

    def has_video(self, video_id: str) -> bool:
        """Check if any segments exist for a video ID."""
        return video_id in self._video_ids

    def __len__(self) -> int:
        """Return number of indexed segments."""
        return len(self._index)

    def __bool__(self) -> bool:
        """Return True if index has any segments."""
        return bool(self._index)

    def __iter__(self):
        """Iterate over segments for backward compatibility."""
        return iter(self._segments)


class HashToFileMapping:
    """Maps hash IDs to actual source file paths from transcription cache."""

    def __init__(self, cache_dir: Path):
        self._mapping: Dict[str, str] = {}
        self._build_mapping(cache_dir)

    def _build_mapping(self, cache_dir: Path):
        """Build hash-to-file mapping by reading transcription cache files."""
        transcriptions_dir = cache_dir / "transcriptions"
        if not transcriptions_dir.exists():
            logger.warning(f"Transcriptions cache not found: {transcriptions_dir}")
            return

        for cache_file in transcriptions_dir.glob("*.json"):
            hash_id = cache_file.stem  # filename without extension is the hash
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # Extract source_file from cache data
                source_file = None
                if isinstance(data, list) and len(data) > 0:
                    source_file = data[0].get('source_file', '')
                elif isinstance(data, dict):
                    source_file = data.get('source_file', '')
                    if not source_file and 'segments' in data:
                        segs = data.get('segments', [])
                        if segs:
                            source_file = segs[0].get('source_file', '')

                if source_file and Path(source_file).exists():
                    self._mapping[hash_id] = source_file
            except Exception as e:
                logger.debug(f"Could not read cache file {cache_file.name}: {e}")

        logger.info(f"Built hash-to-file mapping with {len(self._mapping)} entries")

    def resolve(self, hash_or_path: str) -> str:
        """Resolve a hash ID to file path, or return original if not a hash."""
        # If it's already a valid file path, return as-is
        if Path(hash_or_path).exists():
            return hash_or_path

        # If it looks like a path (has path separators), return as-is
        if '/' in hash_or_path or '\\' in hash_or_path:
            return hash_or_path

        # Try to resolve as hash ID
        return self._mapping.get(hash_or_path, hash_or_path)

if TYPE_CHECKING:
    from ..config import Config
    from ..checkpoint import CheckpointManager
    from ..state import PipelineState

logger = logging.getLogger(__name__)


@register_stage
class OutputStage(Stage):
    """
    Generates output files from matches.

    Inputs:
        - state.matches: List of Match objects
        - state.voiceover_path: Path to voiceover file
        - state.entity_images: Optional entity images for V9
        - state.stock_videos: Optional generic stock videos for V10
        - state.entity_videos: Optional entity videos for V11

    Outputs:
        - state.output_files: List of generated output paths
        - state.otio_files: List of OTIO file paths
    """

    name = "OUTPUT"
    description = "Generate timeline and output files"
    DEPENDS_ON = ['MATCH', 'DOWNLOAD_SEGMENTS']
    PRODUCES = ['otio_files', 'output_files']

    def _scan_video_segments(self, config: 'Config') -> SegmentPathIndex:
        """
        Scan disk for downloaded video segment files and build segment index.

        Video segments may be stored in two formats:
        1. Legacy: *_segments directories with {video_id}_{start_4digit}.mp4
        2. Current: flat in downloaded_videos_dir with {video_id}_{start}_{end}.mp4

        Returns a SegmentPathIndex for O(1) lookup. Per CLAUDE.md rule 38,
        flat dir takes precedence when segment exists in both locations.
        """
        segments = []
        seen_files = set()

        # Try to get videos root from config
        videos_root = None

        # 1. Try config.download.root_dir (short path mode like E:/v)
        if hasattr(config, 'download') and hasattr(config.download, 'root_dir'):
            root_dir = getattr(config.download, 'root_dir', '')
            if root_dir:
                videos_root = Path(root_dir)

        # 2. Try downloaded_videos_dir (computed path)
        if not videos_root or not videos_root.exists():
            if hasattr(config, 'downloaded_videos_dir'):
                videos_root = Path(config.downloaded_videos_dir).parent  # Go to parent (E:/v) not project folder

        # 3. Fall back to project directory
        if not videos_root or not videos_root.exists():
            if hasattr(config, 'otio_output_dir'):
                project_root = Path(config.otio_output_dir).parent
                videos_root = project_root / "videos"

        if not videos_root or not videos_root.exists():
            logger.debug(f"Videos root not found, skipping segment scan")
            return SegmentPathIndex([])

        logger.info(f"Scanning for video segments in: {videos_root}")

        # Find all *_segments directories (legacy format)
        segment_dirs = list(videos_root.glob("*_segments"))

        # Also check date-prefixed subdirectories (e.g., E:/v/24__2026-01-13/*_segments)
        for subdir in videos_root.iterdir():
            if subdir.is_dir():
                segment_dirs.extend(subdir.glob("*_segments"))

        # Pattern for legacy format: {video_id}_{start_4digit}.mp4
        legacy_pattern = re.compile(r'^(.+)_(\d{4})\.mp4$')

        for seg_dir in segment_dirs:
            if not seg_dir.is_dir():
                continue

            for mp4_file in seg_dir.glob("*.mp4"):
                match = legacy_pattern.match(mp4_file.name)
                if match:
                    video_id = match.group(1)
                    start_seconds = int(match.group(2))
                    end_seconds = start_seconds + 120  # Conservative estimate

                    file_str = str(mp4_file)
                    seen_files.add(file_str)
                    segments.append(SegmentInfo(
                        video_id=video_id,
                        file=file_str,
                        original_start=float(start_seconds),
                        original_end=float(end_seconds)
                    ))

        # Scan downloaded_videos_dir for flat segment files: {video_id}_{start}_{end}.mp4
        download_dir = getattr(config, 'downloaded_videos_dir', '')
        if download_dir:
            download_path = Path(download_dir)
            if download_path.exists() and download_path.is_dir():
                # Pattern: {video_id}_{start}_{end}.mp4 (variable-length numbers)
                flat_pattern = re.compile(r'^(.+?)_(\d+)_(\d+)\.mp4$')
                flat_count = 0
                for mp4_file in download_path.glob("*.mp4"):
                    file_str = str(mp4_file)
                    if file_str in seen_files:
                        continue
                    match = flat_pattern.match(mp4_file.name)
                    if match:
                        video_id = match.group(1)
                        start_seconds = int(match.group(2))
                        end_seconds = int(match.group(3))

                        seen_files.add(file_str)
                        segments.append(SegmentInfo(
                            video_id=video_id,
                            file=file_str,
                            original_start=float(start_seconds),
                            original_end=float(end_seconds)
                        ))
                        flat_count += 1
                if flat_count:
                    logger.info(f"Found {flat_count} segments in download dir: {download_path}")

        if segments:
            logger.info(f"Found {len(segments)} video segments on disk for path resolution")

        # Return SegmentPathIndex for O(1) lookups
        return SegmentPathIndex(segments)

    def _synthesize_entity_only_matches(self, state: 'PipelineState') -> list:
        """Synthesize placeholder Match objects from voiceover segments for entity-only mode.

        Entity track functions only read voiceover_segment timing/text from matches.
        These synthetic matches have empty video_file so V1-V8 clips are skipped.
        """
        from ..state import Match as StateMatch

        if not state.voiceover_segments:
            logger.warning("Entity-only mode: no voiceover segments to synthesize matches from")
            return []

        matches = []
        for seg in state.voiceover_segments:
            matches.append(StateMatch(
                segment_index=seg.index,
                video_file="",
                video_start=seg.start,
                video_end=seg.end,
                confidence=0.0,
                strategy="entity_only",
                reason="synthetic-entity-only",
            ))
        logger.info(f"Synthesized {len(matches)} placeholder matches from voiceover segments")
        return matches

    def _resolve_match_paths(self, state: 'PipelineState', config: 'Config') -> int:
        """
        Resolve hash IDs to actual file paths in match data.

        Returns the number of paths resolved.
        """
        # Try multiple locations for cache directory
        cache_dir = None

        # 1. Try config.cache.cache_dir (most common)
        if hasattr(config, 'cache') and hasattr(config.cache, 'cache_dir'):
            cache_dir = Path(config.cache.cache_dir)

        # 2. Try relative .cache from output directory
        if not cache_dir or not cache_dir.exists():
            project_root = Path(config.otio_output_dir).parent if hasattr(config, 'otio_output_dir') else Path(".")
            cache_dir = project_root / ".cache"

        # 3. Try current working directory
        if not cache_dir.exists():
            cache_dir = Path(".cache")

        if not cache_dir.exists():
            logger.warning(f"Cache directory not found: {cache_dir}")
            return 0

        # Build hash-to-file mapping
        hash_mapping = HashToFileMapping(cache_dir)

        resolved_count = 0
        unresolved = []

        for match_result in state.matches:
            if not match_result:
                continue

            # Resolve primary match
            if hasattr(match_result, 'primary_match') and match_result.primary_match:
                pm = match_result.primary_match
                if hasattr(pm, 'video_segment') and pm.video_segment:
                    old_path = pm.video_segment.source_file
                    new_path = hash_mapping.resolve(old_path)
                    if new_path != old_path:
                        pm.video_segment.source_file = new_path
                        resolved_count += 1
                    elif not Path(old_path).exists() and '/' not in old_path and '\\' not in old_path:
                        unresolved.append(old_path[:16] + '...')

            # Resolve alternatives
            if hasattr(match_result, 'alternatives'):
                for alt in match_result.alternatives or []:
                    if hasattr(alt, 'video_segment') and alt.video_segment:
                        old_path = alt.video_segment.source_file
                        new_path = hash_mapping.resolve(old_path)
                        if new_path != old_path:
                            alt.video_segment.source_file = new_path
                            resolved_count += 1

            # Resolve secondary matches
            if hasattr(match_result, 'secondary_matches'):
                for sec in match_result.secondary_matches or []:
                    if hasattr(sec, 'video_segment') and sec.video_segment:
                        old_path = sec.video_segment.source_file
                        new_path = hash_mapping.resolve(old_path)
                        if new_path != old_path:
                            sec.video_segment.source_file = new_path
                            resolved_count += 1

            # Resolve strategy matches
            if hasattr(match_result, 'strategy_matches'):
                for strat in match_result.strategy_matches or []:
                    if hasattr(strat, 'video_segment') and strat.video_segment:
                        old_path = strat.video_segment.source_file
                        new_path = hash_mapping.resolve(old_path)
                        if new_path != old_path:
                            strat.video_segment.source_file = new_path
                            resolved_count += 1

        if unresolved:
            unique_unresolved = list(set(unresolved))[:5]
            logger.warning(f"Could not resolve {len(unresolved)} hash IDs: {unique_unresolved}")

        return resolved_count

    def run(
        self,
        state: 'PipelineState',
        config: 'Config',
        checkpoint: 'CheckpointManager'
    ) -> StageResult:
        """Execute the output stage.

        US-44-002: Validates required state attributes exist.
        """
        # US-167-009: Track stage timing
        stage_start_time = time.time()

        # US-44-002: Validate required attributes exist
        validate_required_state_attrs(
            state, ['matches', 'voiceover_segments'], self.name
        )

        warnings = []

        # Determine output formats to be generated
        output_formats = []
        if config.output.generate_otio:
            output_formats.append("OTIO")
        if config.output.generate_edl:
            output_formats.append("EDL")
        if getattr(config.output, 'generate_xml', True):
            output_formats.append("XML")
        if config.output.generate_report:
            output_formats.append("report")
        if getattr(config.output, 'quality_report_enabled', True):
            output_formats.append("quality_report")

        # Track count: 8 standard tracks (V1-V8)
        track_count = 8

        try:
            log_stage_start(
                logger, "OUTPUT",
                total_matches=len(state.matches) if state.matches else 0,
                output_formats=output_formats,
                track_count=track_count
            )

            # Generate timestamp for this run's outputs
            run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Create timestamped subdirectory
            base_output_dir = Path(config.otio_output_dir)
            output_dir = base_output_dir / run_timestamp
            output_dir.mkdir(parents=True, exist_ok=True)

            logger.info(f"Output directory: {output_dir}")

            # Determine output file basename from Trello card title
            output_basename = self._get_output_basename(config)
            if output_basename != "timeline":
                logger.info(f"Output basename: {output_basename}")

            outputs: Dict[str, Any] = {}

            if not state.matches:
                entity_only = getattr(config.pipeline, 'mode', 'full') == 'entity_only'
                has_entity_data = bool(state.entity_images or state.stock_videos or state.entity_videos)
                if entity_only and has_entity_data:
                    logger.info("Entity-only mode: synthesizing segment timing from voiceover")
                    state.matches = self._synthesize_entity_only_matches(state)
                else:
                    logger.warning("No matches to export")
                    warnings.append("No matches to export")
                    return StageResult.ok({'outputs': outputs, 'output_dir': str(output_dir)}, warnings)

            # Import OTIO package (modular refactored version)
            try:
                from ..otio import (
                    create_timeline,
                    save_timeline,
                    save_timeline_split,
                    save_timeline_as_edl,
                    generate_segment_map,
                    generate_resolve_xml_with_bins
                )
                from ..otio.xml_export import generate_davinci_sequence_xml
            except ImportError as e:
                return StageResult.fail(f"Could not import OTIO modules: {e}", warnings)

            # Generate timeline
            logger.info("Creating timeline...")

            # Debug: Check media availability for V9/V10/V11 tracks
            if state.entity_images:
                total_images = sum(len(getattr(r, 'images', [])) for r in state.entity_images.values())
                logger.info(f"[V9] Entity images available: {len(state.entity_images)} entities, {total_images} images")
                for name, result in list(state.entity_images.items())[:3]:
                    img_count = len(getattr(result, 'images', []))
                    logger.debug(f"  {name}: {img_count} images")
                if len(state.entity_images) > 3:
                    logger.info(f"  ... and {len(state.entity_images) - 3} more entities")
            else:
                logger.warning("[V9] [WARN] No entity images available for V9 track")

            if state.stock_videos:
                total_stock = sum(len((item or {}).get('videos', [])) for item in state.stock_videos.values())
                logger.info(f"[V10] Generic stock videos available: {len(state.stock_videos)} segments, {total_stock} videos")
                for seg_idx, result in list(state.stock_videos.items())[:3]:
                    vid_count = len((result or {}).get('videos', []))
                    logger.debug(f"  segment {seg_idx}: {vid_count} videos")
                if len(state.stock_videos) > 3:
                    logger.info(f"  ... and {len(state.stock_videos) - 3} more segments")
            else:
                logger.warning("[V10] [WARN] No generic stock videos available for V10 track")

            if state.entity_videos:
                total_videos = sum(len(getattr(r, 'videos', [])) for r in state.entity_videos.values())
                logger.info(f"[V11] Entity videos available: {len(state.entity_videos)} entities, {total_videos} videos")
                for name, result in list(state.entity_videos.items())[:3]:
                    vid_count = len(getattr(result, 'videos', []))
                    logger.debug(f"  {name}: {vid_count} videos")
                if len(state.entity_videos) > 3:
                    logger.info(f"  ... and {len(state.entity_videos) - 3} more entities")
            else:
                logger.warning("[V11] [WARN] No entity videos available for V11 track")

            # Resolve hash IDs to actual file paths in match data
            # Skip in entity-only mode (no real video files to resolve)
            entity_only_mode = getattr(config.pipeline, 'mode', 'full') == 'entity_only'
            downloaded_segments = None
            if not entity_only_mode:
                logger.info("Resolving video paths...")
                resolved_count = self._resolve_match_paths(state, config)
                if resolved_count > 0:
                    logger.info(f"[OK] Resolved {resolved_count} hash IDs to file paths")

                # Scan for downloaded video segments (for audio-first mode resolution)
                downloaded_segments = self._scan_video_segments(config)

            # Normalize matches to MatchResult objects if needed
            # Checkpoint restore creates simple Match objects, but create_timeline needs MatchResult
            normalizer = MatchNormalizer()
            normalized_matches = normalizer.normalize(state)
            state.matches = normalized_matches  # Update state so all methods use normalized matches

            # Calculate quality metrics for OTIO metadata and quality report
            quality_metrics = self._calculate_quality_metrics(state.matches)
            quality_metrics_dict = quality_metrics.to_dict() if quality_metrics else None

            # Compute embeddings for V9 semantic entity matching
            # Voiceover embeddings are normally persisted from MATCH stage;
            # recompute here for --output-only runs (cache makes this fast)
            self._ensure_voiceover_embeddings(state, config)
            self._compute_entity_embeddings(state, config)

            timeline = create_timeline(
                matches=state.matches,
                config=config,
                voiceover_path=state.voiceover_path or None,
                frame_rate=getattr(config.output, 'frame_rate', 30.0),
                entity_images=state.entity_images or None,
                stock_videos=state.stock_videos or None,
                entity_videos=state.entity_videos or None,
                downloaded_segments=downloaded_segments,
                quality_metrics=quality_metrics_dict,
                voiceover_embeddings=state.voiceover_embeddings,
                entity_embeddings=state.entity_embeddings or None,
                generated_images=state.generated_images or None,
            )

            # Generate OTIO
            if config.output.generate_otio:
                try:
                    otio_paths = self._generate_otio(
                        timeline, output_dir, config, state, outputs,
                        output_basename=output_basename
                    )
                    state.otio_files = [Path(p) for p in otio_paths] if otio_paths else []

                    # Generate segment map
                    segment_map_path = generate_segment_map(
                        matches=state.matches,
                        output_path=str(output_dir / output_basename),
                        frame_rate=getattr(config.output, 'frame_rate', 30.0),
                        source_srt=state.voiceover_path or '',
                        timeline_start_tc=getattr(config.output, 'timeline_start_tc', "00:00:00:00"),
                        entity_images=state.entity_images or None,
                        stock_videos=state.stock_videos or None,
                        entity_videos=state.entity_videos or None
                    )
                    outputs['segment_map'] = segment_map_path
                    logger.info(f"+ Segment map: {Path(segment_map_path).name}")
                except Exception as e:
                    log_error_with_context(logger, "OUTPUT-001", f"OTIO generation failed: {e}")

            # Generate EDL
            if config.output.generate_edl:
                try:
                    edl_path = self._generate_edl(
                        state, output_dir, config, save_timeline_as_edl,
                        output_basename=output_basename
                    )
                    outputs['edl'] = str(edl_path)
                except Exception as e:
                    log_error_with_context(logger, "OUTPUT-002", f"EDL generation failed: {e}")

            # Generate DaVinci Resolve XML
            if getattr(config.output, 'generate_xml', True):
                try:
                    xml_paths = self._generate_xml(
                        state, output_dir, config, generate_resolve_xml_with_bins,
                        downloaded_segments=downloaded_segments,
                        output_basename=output_basename
                    )
                    outputs['xml'] = xml_paths

                    # Also generate DaVinci-native sequence XML format
                    try:
                        sequence_xml_path = generate_davinci_sequence_xml(
                            matches=state.matches,
                            output_path=str(output_dir / output_basename),
                            frame_rate=getattr(config.output, 'frame_rate', 30.0),
                            downloaded_segments=downloaded_segments,
                            timeline_start_tc=getattr(config.output, 'timeline_start_tc', '01:00:00:00')
                        )
                        outputs['sequence_xml'] = sequence_xml_path
                        logger.info(f"+ XML (DaVinci): {Path(sequence_xml_path).name}")
                    except Exception as e:
                        log_error_with_context(logger, "OUTPUT-003", f"DaVinci sequence XML generation failed: {e}")
                except Exception as e:
                    log_error_with_context(logger, "OUTPUT-003", f"XML generation failed: {e}")

            # Generate report
            if config.output.generate_report:
                try:
                    report_path = self._generate_report(state, output_dir)
                    outputs['report'] = str(report_path)
                except Exception as e:
                    log_error_with_context(logger, "OUTPUT-004", f"Report generation failed: {e}")

            # Generate quality report JSON
            if getattr(config.output, 'quality_report_enabled', True):
                try:
                    quality_report_path = self._generate_quality_report(
                        state, output_dir, quality_metrics
                    )
                    outputs['quality_report'] = str(quality_report_path)
                except Exception as e:
                    log_error_with_context(logger, "OUTPUT-004", f"Quality report generation failed: {e}")

            # Track all output files
            state.output_files = [Path(p) for p in self._collect_output_paths(outputs)]

            checkpoint_data = {
                'outputs': outputs,
                'output_dir': str(output_dir),
                'timestamp': run_timestamp,
                'match_count': len(state.matches),
            }

            # US-167-009: Log stage completion with timing
            elapsed = time.time() - stage_start_time

            # Calculate timeline duration from voiceover segments
            timeline_duration = 0.0
            if state.voiceover_segments:
                end_times = [seg.end_time for seg in state.voiceover_segments if hasattr(seg, 'end_time')]
                timeline_duration = max(end_times) if end_times else 0.0

            log_stage_complete(
                logger, "OUTPUT",
                elapsed_seconds=elapsed,
                files_written=len(state.output_files),
                timeline_duration=timeline_duration
            )

            return StageResult.ok(checkpoint_data, warnings)

        except Exception as e:
            log_error_with_context(logger, "OUTPUT-001", f"Output stage failed: {e}")
            return StageResult.fail(str(e), warnings)

    def can_skip(
        self,
        state: 'PipelineState',
        checkpoint: 'CheckpointManager'
    ) -> bool:
        """Output stage should not be skipped - always regenerate"""
        # Output stage typically should NOT be skipped since user may want
        # fresh outputs. But we check checkpoint for consistency.
        return False

    def restore(
        self,
        state: 'PipelineState',
        checkpoint: 'CheckpointManager',
        config: 'Config' = None
    ) -> bool:
        """
        Restore output stage from checkpoint.

        Validates that output paths exist before restoring.
        Logs specific errors for missing paths and invalid data.

        Returns False on validation failure (not exception).
        """
        try:
            data = checkpoint.get_stage_data(self.name)
            if not data:
                logger.warning(f"No checkpoint data for {self.name}: checkpoint returned None")
                return False

            # Validate data is a dict
            if not isinstance(data, dict):
                logger.warning(f"Invalid checkpoint data for {self.name}: expected dict, got {type(data).__name__}")
                return False

            outputs = data.get('outputs', {})

            # Validate outputs is a dict
            if not isinstance(outputs, dict):
                logger.warning(f"Invalid checkpoint data for {self.name}: 'outputs' is not a dict (got {type(outputs).__name__})")
                return False

            # Collect and validate output paths
            all_paths = self._collect_output_paths(outputs)
            existing_paths = []
            missing_paths = []

            for p in all_paths:
                if not isinstance(p, str):
                    logger.debug(f"Skipping non-string path in checkpoint: {repr(p)}")
                    continue
                if Path(p).exists():
                    existing_paths.append(p)
                else:
                    missing_paths.append(p)

            # Log missing paths but don't fail - files may be in different location
            if missing_paths:
                logger.warning(f"Some output files from checkpoint no longer exist: {missing_paths[:3]}")
                if len(missing_paths) > 3:
                    logger.warning(f"... and {len(missing_paths) - 3} more missing files")

            # Restore output file paths (include all, not just existing)
            state.output_files = [Path(p) for p in all_paths if isinstance(p, str)]

            # Restore OTIO paths with validation
            otio_data = outputs.get('otio', [])
            if isinstance(otio_data, list):
                state.otio_files = [Path(p) for p in otio_data if isinstance(p, str)]
            elif isinstance(otio_data, str):
                state.otio_files = [Path(otio_data)]
            else:
                logger.debug(f"Unexpected otio type in checkpoint: {type(otio_data).__name__}")
                state.otio_files = []

            # Validate OTIO paths exist
            otio_missing = [str(p) for p in state.otio_files if not p.exists()]
            if otio_missing:
                logger.warning(f"OTIO files from checkpoint no longer exist: {otio_missing[:3]}")

            logger.info(f"Restored OUTPUT: {len(state.output_files)} files ({len(existing_paths)} exist, {len(missing_paths)} missing)")
            return True

        except Exception as e:
            logger.warning(f"Failed to restore OUTPUT: {e}")
            return False

    def validate_inputs(
        self,
        state: 'PipelineState',
        config: 'Config'
    ) -> Optional[str]:
        """
        Validate inputs before running.

        Returns specific missing field names and suggestions for which stage to run.
        """
        missing_fields = []

        # In entity-only mode, matches are synthesized from voiceover segments in run()
        entity_only = getattr(config.pipeline, 'mode', 'full') == 'entity_only'
        if not state.matches and not entity_only:
            missing_fields.append("matches")

        # Check optional but recommended fields and add warnings
        if not hasattr(config, 'otio_output_dir') or not config.otio_output_dir:
            missing_fields.append("config.otio_output_dir")

        if missing_fields:
            fields_str = ", ".join(missing_fields)
            suggestions = []

            if "matches" in missing_fields:
                suggestions.append("run MATCH stage first to generate matches")

            if "config.otio_output_dir" in missing_fields:
                suggestions.append("set otio_output_dir in config")

            suggestion_str = "; ".join(suggestions) if suggestions else "check pipeline configuration"
            return f"Missing required fields: {fields_str}. Suggestion: {suggestion_str}"

        return None

    def get_input_output_info(
        self,
        state: 'PipelineState',
        config: 'Config'
    ) -> Dict[str, Any]:
        """Get input/output info for dry-run preview"""
        # Count inputs (matches)
        input_count = len(state.matches) if state.matches else 0

        # Count outputs (timeline files)
        output_count = None

        return {
            'inputs': 'matches',
            'outputs': 'OTIO/EDL/XML',
            'input_count': input_count,
            'output_count': output_count,
        }

    # === Helper Methods ===

    def _get_output_basename(self, config: 'Config') -> str:
        """
        Get the base name for output files from Trello card title.

        Reads trello_card.json from the project directory and returns
        a sanitized card title. Falls back to "timeline" if unavailable.
        """
        from ..utils.project_metadata import get_card_title

        project_dir = Path(config.otio_output_dir).parent if hasattr(config, 'otio_output_dir') else None
        if not project_dir:
            return "timeline"

        title = get_card_title(project_dir)
        return title or "timeline"

    def _generate_otio(
        self,
        timeline: Any,
        output_dir: Path,
        config: 'Config',
        state: 'PipelineState',
        outputs: Dict[str, Any],
        output_basename: str = "timeline"
    ) -> List[str]:
        """Generate OTIO timeline files"""
        from ..otio_builder import save_timeline, save_timeline_split

        otio_base_path = output_dir / output_basename
        split_otio = getattr(config.output, 'split_otio', True)

        if split_otio:
            otio_paths = save_timeline_split(timeline, str(otio_base_path))
            outputs['otio'] = otio_paths

            # Categorize for display
            full = [p for p in otio_paths if '_FULL' in p]
            tracks = [p for p in otio_paths if '_V' in Path(p).name and '_FULL' not in p]
            audio = [p for p in otio_paths if '_A8_' in p]

            logger.info(f"+ OTIO files generated ({len(otio_paths)} total):")

            if tracks:
                logger.info(f"  Individual tracks:")
                for p in tracks:
                    logger.debug(f"    - {Path(p).name}")

            if audio:
                for p in audio:
                    logger.debug(f"    - {Path(p).name}")

            if full:
                logger.info(f"  Full timeline:")
                for p in full:
                    logger.debug(f"    - {Path(p).name}")

            return otio_paths
        else:
            otio_path = str(otio_base_path) + ".otio"
            save_timeline(timeline, otio_path)
            outputs['otio'] = otio_path
            logger.info(f"+ OTIO: {otio_path}")
            return [otio_path]

    def _generate_edl(
        self,
        state: 'PipelineState',
        output_dir: Path,
        config: 'Config',
        save_edl_func: callable,
        output_basename: str = "timeline"
    ) -> Path:
        """Generate EDL file"""
        from src.otio.xml_export import _is_ntsc_rate

        edl_path = output_dir / f"{output_basename}.edl"
        frame_rate = getattr(config.output, 'frame_rate', 30.0)
        # Auto-detect drop-frame for NTSC rates (29.97, 59.94)
        # 23.976 is NTSC but uses non-drop-frame timecode (24fps timebase)
        drop_frame = _is_ntsc_rate(frame_rate) and round(frame_rate) in (30, 60)
        save_edl_func(
            state.matches,
            str(edl_path),
            frame_rate=frame_rate,
            timeline_start_tc=getattr(config.output, 'timeline_start_tc', "00:00:00:00"),
            entities=state.extracted_entities or [],
            drop_frame=drop_frame
        )
        logger.info(f"+ EDL: {edl_path}")
        return edl_path

    def _generate_xml(
        self,
        state: 'PipelineState',
        output_dir: Path,
        config: 'Config',
        generate_xml_func: callable,
        downloaded_segments: List = None,
        output_basename: str = "timeline"
    ) -> List[str]:
        """Generate DaVinci Resolve XML"""
        xml_base_path = output_dir / output_basename
        num_parts = getattr(config.output, 'xml_parts', 2)

        xml_paths = generate_xml_func(
            matches=state.matches,
            output_path=str(xml_base_path),
            voiceover_path=state.voiceover_path or None,
            frame_rate=getattr(config.output, 'frame_rate', 30.0),
            entity_images=state.entity_images or None,
            stock_videos=state.stock_videos or None,
            entity_videos=state.entity_videos or None,
            config=config,
            num_parts=num_parts,
            downloaded_segments=downloaded_segments,
            timeline_start_tc=getattr(config.output, 'timeline_start_tc', '01:00:00:00')
        )
        logger.info(f"+ XML (fallback): {Path(xml_paths[0]).name}")
        return xml_paths

    def _generate_report(
        self,
        state: 'PipelineState',
        output_dir: Path
    ) -> Path:
        """Generate a markdown report of matches"""
        report_path = output_dir / "match_report.md"

        lines = [
            "# Match Report",
            "",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Total segments: {len(state.matches)}",
            "",
            "## Matches",
            ""
        ]

        for i, match_result in enumerate(state.matches, 1):
            if not match_result:
                continue

            # Handle both Match objects and MatchResult objects
            if hasattr(match_result, 'primary_match') and match_result.primary_match:
                m = match_result.primary_match
                text = m.voiceover_segment.text[:100] if hasattr(m, 'voiceover_segment') else ''
                video = Path(m.video_segment.source_file).name if hasattr(m, 'video_segment') else ''
                confidence = m.confidence if hasattr(m, 'confidence') else 0.0
                reasoning = m.reasoning if hasattr(m, 'reasoning') else ''
            else:
                # Direct Match object
                text = getattr(match_result, 'text', '')[:100] if hasattr(match_result, 'text') else ''
                video = Path(getattr(match_result, 'video_file', '')).name
                confidence = getattr(match_result, 'confidence', 0.0)
                reasoning = getattr(match_result, 'reason', '')

            lines.append(f"### Segment {i}")
            if text:
                lines.append(f"**Voiceover:** {text}...")
            if video:
                lines.append(f"**Video:** {video}")
            lines.append(f"**Confidence:** {confidence:.1%}")
            if reasoning:
                lines.append(f"**Reasoning:** {reasoning}")
            lines.append("")

        report_path.write_text("\n".join(lines), encoding='utf-8')
        logger.info(f"+ Report: {report_path}")
        return report_path

    def _collect_output_paths(self, outputs: Dict[str, Any]) -> List[str]:
        """Collect all output file paths from outputs dict"""
        paths = []
        for key, value in outputs.items():
            if isinstance(value, list):
                paths.extend(value)
            elif isinstance(value, str):
                paths.append(value)
        return paths

    def _ensure_voiceover_embeddings(
        self,
        state: 'PipelineState',
        config: 'Config'
    ) -> None:
        """Ensure voiceover embeddings exist (recompute for --output-only runs)."""
        if state.voiceover_embeddings is not None:
            return
        if not state.voiceover_segments:
            return

        try:
            from ..embeddings import compute_embeddings, get_embedding_provider
            from ..utils import CacheManager

            provider = get_embedding_provider(config)
            cache_dir = getattr(config.cache, 'cache_dir', '.cache')
            cache = CacheManager(cache_dir)

            vo_texts = [seg.text for seg in state.voiceover_segments]
            vo_embeddings = compute_embeddings(
                texts=vo_texts,
                provider=provider,
                cache=cache,
                cache_key="voiceover",
                embed_mode="query",
                config=config
            )
            if vo_embeddings is not None and len(vo_embeddings) > 0:
                state.voiceover_embeddings = vo_embeddings
                logger.info(f"[OUTPUT] Recomputed voiceover embeddings for {len(vo_texts)} segments")
        except Exception as e:
            logger.warning(f"[OUTPUT] Could not compute voiceover embeddings: {e}")

    def _compute_entity_embeddings(
        self,
        state: 'PipelineState',
        config: 'Config'
    ) -> None:
        """Compute embeddings for entity query texts for V9 semantic matching."""
        if not state.entity_images:
            return

        try:
            from ..embeddings import get_embedding_provider

            provider = get_embedding_provider(config)

            # Collect entity query texts
            entity_texts = {}
            for entity_name, entity_result in state.entity_images.items():
                query = getattr(entity_result, 'query', entity_name)
                entity_texts[entity_name] = query or entity_name

            if not entity_texts:
                return

            # Embed all entity texts
            texts_list = list(entity_texts.values())
            names_list = list(entity_texts.keys())
            embeddings = provider.embed(texts_list, embed_mode='document')

            # Store in state
            for name, embedding in zip(names_list, embeddings):
                if embedding is not None:
                    if hasattr(embedding, 'tolist'):
                        state.entity_embeddings[name] = embedding.tolist()
                    else:
                        state.entity_embeddings[name] = list(embedding)

            logger.info(f"[OUTPUT] Computed embeddings for {len(state.entity_embeddings)} entities")

        except Exception as e:
            logger.warning(f"[OUTPUT] Entity embedding computation failed, falling back to word overlap: {e}")
            state.entity_embeddings = {}

    def _calculate_quality_metrics(
        self,
        matches: List[Any]
    ) -> Optional[Any]:
        """
        Calculate quality metrics for matches.

        Returns MatchQualityMetrics with calculated values including
        avg_confidence, min_confidence, max_confidence, gap_count,
        match_rate, and source_variety.
        """
        try:
            from ..matching.metrics import (
                MatchQualityMetrics,
                calculate_match_quality_metrics
            )
        except ImportError:
            logger.warning("Could not import matching.metrics, skipping quality metrics")
            return None

        if not matches:
            return MatchQualityMetrics(total_segments=0)

        # Calculate metrics using the existing function
        metrics = calculate_match_quality_metrics(matches, len(matches))

        # Also calculate source_variety (unique source files / total matches)
        source_files = set()
        for m in matches:
            if hasattr(m, 'primary_match') and m.primary_match:
                pm = m.primary_match
                if hasattr(pm, 'video_segment') and pm.video_segment:
                    source_file = pm.video_segment.source_file
                    if source_file:
                        source_files.add(source_file)

        # Store source_variety in a custom attribute
        # We'll include it when generating the report
        metrics._source_variety = len(source_files) / len(matches) if matches else 0.0
        metrics._unique_sources = len(source_files)

        return metrics

    def _generate_quality_report(
        self,
        state: 'PipelineState',
        output_dir: Path,
        metrics: Optional[Any]
    ) -> Path:
        """
        Generate quality_report.json with match quality metrics.

        The report includes:
        - total_segments: Total number of voiceover segments
        - matched_segments: Number of segments with matches
        - avg_confidence: Average confidence score
        - gaps_count: Number of gaps (unmatched segments)
        - source_variety: Ratio of unique sources to total matches
        """
        report_path = output_dir / "quality_report.json"

        report_data = {
            'total_segments': len(state.matches) if state.matches else 0,
            'matched_segments': 0,
            'avg_confidence': 0.0,
            'min_confidence': 0.0,
            'max_confidence': 0.0,
            'confidence_std': 0.0,
            'gaps_count': 0,
            'match_rate': 0.0,
            'source_variety': 0.0,
            'unique_sources': 0,
        }

        if metrics:
            report_data['matched_segments'] = metrics.matched_segments
            report_data['avg_confidence'] = round(metrics.avg_confidence, 4)
            report_data['min_confidence'] = round(metrics.min_confidence, 4)
            report_data['max_confidence'] = round(metrics.max_confidence, 4)
            report_data['confidence_std'] = round(metrics.confidence_std, 4)
            report_data['gaps_count'] = metrics.gap_count
            report_data['match_rate'] = round(metrics.match_rate, 4)

            # Include source variety from custom attribute
            if hasattr(metrics, '_source_variety'):
                report_data['source_variety'] = round(metrics._source_variety, 4)
            if hasattr(metrics, '_unique_sources'):
                report_data['unique_sources'] = metrics._unique_sources

        # Write JSON report
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2)

        logger.info(f"+ Quality report: {report_path.name}")

        return report_path
