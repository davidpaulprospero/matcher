"""
Output Stage - Timeline and Report Generation

Stage 5 of the video matching pipeline:
- Generates OTIO timeline (split or single)
- Creates segment map for post-edit analysis
- Generates EDL export
- Generates DaVinci Resolve XML with bins
- Creates match report
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from . import Stage, StageResult, register_stage

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

    def run(
        self,
        state: 'PipelineState',
        config: 'Config',
        checkpoint: 'CheckpointManager'
    ) -> StageResult:
        """Execute the output stage"""
        warnings = []

        try:
            print(f"\n  --- Stage 5: GENERATE OUTPUT ---")

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
                print(f"  [V9] ⚠ No entity images available for V9 track")

            if state.entity_videos:
                total_videos = sum(len(getattr(r, 'videos', [])) for r in state.entity_videos.values())
                print(f"  [V10] Stock videos available: {len(state.entity_videos)} entities, {total_videos} videos")
                for name, result in list(state.entity_videos.items())[:3]:
                    vid_count = len(getattr(result, 'videos', []))
                    print(f"    • {name}: {vid_count} videos")
                if len(state.entity_videos) > 3:
                    print(f"    ... and {len(state.entity_videos) - 3} more entities")
            else:
                print(f"  [V10] ⚠ No stock videos available for V10 track")

            timeline = create_timeline(
                matches=state.matches,
                config=config,
                voiceover_path=state.voiceover_path or None,
                frame_rate=getattr(config.output, 'frame_rate', 30.0),
                entity_images=state.entity_images or None,
                entity_videos=state.entity_videos or None,
                downloaded_segments=None  # Will be added when DownloadStage extracts
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
                    state, output_dir, config, generate_resolve_xml_with_bins
                )
                outputs['xml'] = xml_paths

            # Generate report
            if config.output.generate_report:
                report_path = self._generate_report(state, output_dir)
                outputs['report'] = str(report_path)

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
        checkpoint: 'CheckpointManager'
    ) -> bool:
        """Restore output stage from checkpoint"""
        try:
            data = checkpoint.get_stage_data(self.name)
            if not data:
                return False

            outputs = data.get('outputs', {})

            # Restore output file paths
            state.output_files = [
                Path(p) for p in self._collect_output_paths(outputs)
            ]

            # Restore OTIO paths
            otio_data = outputs.get('otio', [])
            if isinstance(otio_data, list):
                state.otio_files = [Path(p) for p in otio_data]
            elif isinstance(otio_data, str):
                state.otio_files = [Path(otio_data)]
            else:
                state.otio_files = []

            logger.info(f"Restored OUTPUT: {len(state.output_files)} files")
            return True

        except Exception as e:
            logger.warning(f"Failed to restore OUTPUT: {e}")
            return False

    def validate_inputs(
        self,
        state: 'PipelineState',
        config: 'Config'
    ) -> Optional[str]:
        """Validate inputs before running"""
        # Matches are required for meaningful output
        if not state.matches:
            return "No matches available for output generation"
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
        generate_xml_func: callable
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
            num_parts=num_parts
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
