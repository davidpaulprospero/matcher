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
from datetime import datetime
from pathlib import Path
import json
from typing import TYPE_CHECKING, Any, Dict, List, Optional, NamedTuple

from . import Stage, StageResult, register_stage, validate_required_state_attrs


class SegmentInfo(NamedTuple):
    """Info about a downloaded video segment for path resolution."""
    video_id: str
    file: str
    original_start: float
    original_end: float


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
        - state.entity_videos: Optional stock videos for V10

    Outputs:
        - state.output_files: List of generated output paths
        - state.otio_files: List of OTIO file paths
    """

    name = "OUTPUT"
    description = "Generate timeline and output files"

    def _scan_video_segments(self, config: 'Config') -> List[SegmentInfo]:
        """
        Scan disk for downloaded video segment files and build segment info list.

        Video segments are stored in *_segments directories with filenames like:
        {video_id}_{start_seconds}.mp4

        This allows create_timeline to resolve audio files to video segment paths.
        """
        segments = []

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
            return segments

        logger.info(f"Scanning for video segments in: {videos_root}")

        # Find all *_segments directories
        segment_dirs = list(videos_root.glob("*_segments"))

        # Also check date-prefixed subdirectories (e.g., E:/v/24__2026-01-13/*_segments)
        for subdir in videos_root.iterdir():
            if subdir.is_dir():
                segment_dirs.extend(subdir.glob("*_segments"))

        # Pattern to parse segment filenames: {video_id}_{start_seconds}.mp4
        # video_id can contain letters, numbers, hyphens, underscores
        # start_seconds is always 4 digits (e.g., 0000, 0123)
        segment_pattern = re.compile(r'^(.+)_(\d{4})\.mp4$')

        for seg_dir in segment_dirs:
            if not seg_dir.is_dir():
                continue

            for mp4_file in seg_dir.glob("*.mp4"):
                match = segment_pattern.match(mp4_file.name)
                if match:
                    video_id = match.group(1)
                    start_seconds = int(match.group(2))

                    # Estimate segment duration (assume 60s segments, will be overridden if actual duration known)
                    # The exact end time isn't critical - it's used for range matching
                    end_seconds = start_seconds + 120  # Conservative estimate

                    segments.append(SegmentInfo(
                        video_id=video_id,
                        file=str(mp4_file),
                        original_start=float(start_seconds),
                        original_end=float(end_seconds)
                    ))

        if segments:
            logger.info(f"Found {len(segments)} video segments on disk for path resolution")

        return segments

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

            # Resolve strategy alternatives
            if hasattr(match_result, 'strategy_alternatives'):
                for strat in match_result.strategy_alternatives or []:
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
        # US-44-002: Validate required attributes exist
        validate_required_state_attrs(
            state, ['matches', 'voiceover_segments'], self.name
        )

        warnings = []

        try:
            print(f"\n  --- Stage 7: GENERATE OUTPUT ---")

            # Generate timestamp for this run's outputs
            run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Create timestamped subdirectory
            base_output_dir = Path(config.otio_output_dir)
            output_dir = base_output_dir / run_timestamp
            output_dir.mkdir(parents=True, exist_ok=True)

            print(f"  Output directory: {output_dir}")

            outputs: Dict[str, Any] = {}

            if not state.matches:
                print("  ! No matches to export")
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
            print(f"  Creating timeline...")

            # Debug: Check entity data availability for V9/V10 tracks
            if state.entity_images:
                total_images = sum(len(getattr(r, 'images', [])) for r in state.entity_images.values())
                print(f"  [V9] Entity images available: {len(state.entity_images)} entities, {total_images} images")
                for name, result in list(state.entity_images.items())[:3]:
                    img_count = len(getattr(result, 'images', []))
                    print(f"    • {name}: {img_count} images")
                if len(state.entity_images) > 3:
                    print(f"    ... and {len(state.entity_images) - 3} more entities")
            else:
                print(f"  [V9] [WARN] No entity images available for V9 track")

            if state.entity_videos:
                total_videos = sum(len(getattr(r, 'videos', [])) for r in state.entity_videos.values())
                print(f"  [V10] Stock videos available: {len(state.entity_videos)} entities, {total_videos} videos")
                for name, result in list(state.entity_videos.items())[:3]:
                    vid_count = len(getattr(result, 'videos', []))
                    print(f"    • {name}: {vid_count} videos")
                if len(state.entity_videos) > 3:
                    print(f"    ... and {len(state.entity_videos) - 3} more entities")
            else:
                print(f"  [V10] [WARN] No stock videos available for V10 track")

            # Resolve hash IDs to actual file paths in match data
            print(f"  Resolving video paths...")
            resolved_count = self._resolve_match_paths(state, config)
            if resolved_count > 0:
                print(f"  [OK] Resolved {resolved_count} hash IDs to file paths")

            # Scan for downloaded video segments (for audio-first mode resolution)
            downloaded_segments = self._scan_video_segments(config)

            # Normalize matches to MatchResult objects if needed
            # Checkpoint restore creates simple Match objects, but create_timeline needs MatchResult
            normalized_matches = self._normalize_matches(state)
            state.matches = normalized_matches  # Update state so all methods use normalized matches

            # Calculate quality metrics for OTIO metadata and quality report
            quality_metrics = self._calculate_quality_metrics(state.matches)
            quality_metrics_dict = quality_metrics.to_dict() if quality_metrics else None

            timeline = create_timeline(
                matches=state.matches,
                config=config,
                voiceover_path=state.voiceover_path or None,
                frame_rate=getattr(config.output, 'frame_rate', 30.0),
                entity_images=state.entity_images or None,
                entity_videos=state.entity_videos or None,
                downloaded_segments=downloaded_segments,
                quality_metrics=quality_metrics_dict
            )

            # Generate OTIO
            if config.output.generate_otio:
                otio_paths = self._generate_otio(
                    timeline, output_dir, config, state, outputs
                )
                state.otio_files = [Path(p) for p in otio_paths] if otio_paths else []

                # Generate segment map
                segment_map_path = generate_segment_map(
                    matches=state.matches,
                    output_path=str(output_dir / "timeline"),
                    frame_rate=getattr(config.output, 'frame_rate', 30.0),
                    source_srt=state.voiceover_path or '',
                    timeline_start_tc=getattr(config.output, 'timeline_start_tc', "01:00:00:00")
                )
                outputs['segment_map'] = segment_map_path
                print(f"  + Segment map: {Path(segment_map_path).name}")

            # Generate EDL
            if config.output.generate_edl:
                edl_path = self._generate_edl(
                    state, output_dir, config, save_timeline_as_edl
                )
                outputs['edl'] = str(edl_path)

            # Generate DaVinci Resolve XML
            if getattr(config.output, 'generate_xml', True):
                xml_paths = self._generate_xml(
                    state, output_dir, config, generate_resolve_xml_with_bins,
                    downloaded_segments=downloaded_segments
                )
                outputs['xml'] = xml_paths

                # Also generate DaVinci-native sequence XML format
                try:
                    sequence_xml_path = generate_davinci_sequence_xml(
                        matches=state.matches,
                        output_path=str(output_dir / "timeline"),
                        frame_rate=getattr(config.output, 'frame_rate', 30.0),
                        downloaded_segments=downloaded_segments,
                        timeline_start_tc=getattr(config.output, 'timeline_start_tc', '01:00:00:00')
                    )
                    outputs['sequence_xml'] = sequence_xml_path
                    print(f"  + XML (DaVinci): {Path(sequence_xml_path).name}")
                except Exception as e:
                    logger.warning(f"Failed to generate DaVinci sequence XML: {e}")

            # Generate report
            if config.output.generate_report:
                report_path = self._generate_report(state, output_dir)
                outputs['report'] = str(report_path)

            # Generate quality report JSON
            if getattr(config.output, 'quality_report_enabled', True):
                quality_report_path = self._generate_quality_report(
                    state, output_dir, quality_metrics
                )
                outputs['quality_report'] = str(quality_report_path)

            # Track all output files
            state.output_files = [Path(p) for p in self._collect_output_paths(outputs)]

            checkpoint_data = {
                'outputs': outputs,
                'output_dir': str(output_dir),
                'timestamp': run_timestamp,
                'match_count': len(state.matches),
            }

            return StageResult.ok(checkpoint_data, warnings)

        except Exception as e:
            logger.exception(f"Output stage failed: {e}")
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

        if not state.matches:
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

    # === Helper Methods ===

    def _generate_otio(
        self,
        timeline: Any,
        output_dir: Path,
        config: 'Config',
        state: 'PipelineState',
        outputs: Dict[str, Any]
    ) -> List[str]:
        """Generate OTIO timeline files"""
        from ..otio_builder import save_timeline, save_timeline_split

        otio_base_path = output_dir / "timeline"
        split_otio = getattr(config.output, 'split_otio', True)

        if split_otio:
            otio_paths = save_timeline_split(timeline, str(otio_base_path))
            outputs['otio'] = otio_paths

            # Categorize for display
            full = [p for p in otio_paths if '_FULL' in p]
            tracks = [p for p in otio_paths if '_V' in Path(p).name and '_FULL' not in p]
            audio = [p for p in otio_paths if '_A8_' in p]

            print(f"  + OTIO files generated ({len(otio_paths)} total):")

            if tracks:
                print(f"    Individual tracks:")
                for p in tracks:
                    print(f"      - {Path(p).name}")

            if audio:
                for p in audio:
                    print(f"      - {Path(p).name}")

            if full:
                print(f"    Full timeline:")
                for p in full:
                    print(f"      - {Path(p).name}")

            return otio_paths
        else:
            otio_path = str(otio_base_path) + ".otio"
            save_timeline(timeline, otio_path)
            outputs['otio'] = otio_path
            print(f"  + OTIO: {otio_path}")
            return [otio_path]

    def _generate_edl(
        self,
        state: 'PipelineState',
        output_dir: Path,
        config: 'Config',
        save_edl_func: callable
    ) -> Path:
        """Generate EDL file"""
        edl_path = output_dir / "timeline.edl"
        save_edl_func(
            state.matches,
            str(edl_path),
            frame_rate=getattr(config.output, 'frame_rate', 30.0),
            timeline_start_tc=getattr(config.output, 'timeline_start_tc', "01:00:00:00"),
            entities=state.extracted_entities or []
        )
        print(f"  + EDL: {edl_path}")
        return edl_path

    def _generate_xml(
        self,
        state: 'PipelineState',
        output_dir: Path,
        config: 'Config',
        generate_xml_func: callable,
        downloaded_segments: List = None
    ) -> List[str]:
        """Generate DaVinci Resolve XML"""
        xml_base_path = output_dir / "timeline"
        num_parts = getattr(config.output, 'xml_parts', 2)

        xml_paths = generate_xml_func(
            matches=state.matches,
            output_path=str(xml_base_path),
            voiceover_path=state.voiceover_path or None,
            frame_rate=getattr(config.output, 'frame_rate', 30.0),
            entity_images=state.entity_images or None,
            entity_videos=state.entity_videos or None,
            config=config,
            num_parts=num_parts,
            downloaded_segments=downloaded_segments,
            timeline_start_tc=getattr(config.output, 'timeline_start_tc', '01:00:00:00')
        )
        print(f"  + XML (fallback): {Path(xml_paths[0]).name}")
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
        print(f"  + Report: {report_path}")
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

    def _normalize_matches(self, state: 'PipelineState') -> List[Any]:
        """
        Normalize matches to MatchResult format.

        When matches are restored from checkpoint, they're simple Match objects
        from state.py with fields: segment_index, video_file, video_start, etc.

        create_timeline expects MatchResult objects from utils.py with fields:
        primary_match (containing voiceover_segment, video_segment), alternatives, etc.

        This method converts simple Match objects to MatchResult objects.
        """
        from ..utils import Match as UtilsMatch, MatchResult, SRTSegment

        if not state.matches:
            return []

        # Check if matches are already MatchResult objects
        first_match = state.matches[0]
        if hasattr(first_match, 'primary_match'):
            # Already MatchResult format
            return state.matches

        # Need to convert simple Match objects to MatchResult
        logger.info("Converting checkpoint matches to MatchResult format")
        normalized = []

        for match in state.matches:
            if not match:
                continue

            # Get segment_index - simple Match uses segment_index field
            segment_index = getattr(match, 'segment_index', 0)

            # Get voiceover segment from state
            if segment_index < len(state.voiceover_segments):
                vo_seg = state.voiceover_segments[segment_index]
            else:
                # Create minimal voiceover segment
                vo_seg = SRTSegment(
                    index=segment_index,
                    start_time=0.0,
                    end_time=1.0,
                    text="",
                    source_file=""
                )

            # Create video segment from simple Match fields
            video_file = getattr(match, 'video_file', '')
            video_start = getattr(match, 'video_start', 0.0)
            video_end = getattr(match, 'video_end', video_start + 1.0)

            video_seg = SRTSegment(
                index=segment_index,
                start_time=video_start,
                end_time=video_end,
                text="",  # Not preserved in checkpoint
                source_file=video_file
            )

            # Create Match (from utils.py) with the segments
            confidence = getattr(match, 'confidence', 0.5)
            reasoning = getattr(match, 'reason', '')

            utils_match = UtilsMatch(
                voiceover_segment=vo_seg,
                video_segment=video_seg,
                video_scene=None,
                confidence=confidence,
                reasoning=reasoning
            )

            # Wrap in MatchResult
            match_result = MatchResult(
                primary_match=utils_match,
                alternatives=[],
                secondary_matches=[],
                strategy_matches=[]
            )

            normalized.append(match_result)

        logger.info(f"Converted {len(normalized)} matches to MatchResult format")
        return normalized

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

        print(f"  + Quality report: {report_path.name}")
        logger.info(f"Generated quality report: {report_path}")

        return report_path
