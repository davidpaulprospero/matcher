#!/usr/bin/env python3
"""
Regenerate OTIO timeline from existing checkpoint and segment data.

This script bypasses all pipeline phases (analyze, download, match) and
regenerates OTIO files directly from:
  - checkpoint.json (voiceover timing)
  - timeline_segments.json (match data)

Useful for:
  - Fixing timing/sync issues without re-processing
  - Changing output settings (frame rate, timecode)
  - Recovery from corrupted OTIO files
  - Manual clip swaps (edit timeline_segments.json, then regenerate)

Usage:
  python regenerate_otio.py                     # Run in project directory
  python regenerate_otio.py /path/to/project    # Specify project directory
  python regenerate_otio.py --output 20260113_203704  # Use specific output folder
  python regenerate_otio.py --fps 24            # Change frame rate
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# Add voiceover-matcher to path
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))


def find_latest_output(project_dir: Path) -> Path:
    """Find the most recent output folder containing timeline_segments.json."""
    output_dir = project_dir / "output"
    if not output_dir.exists():
        raise FileNotFoundError(f"No output directory found at {output_dir}")

    # Find all folders with timeline_segments.json
    candidates = []
    for folder in output_dir.iterdir():
        if folder.is_dir():
            segments_file = folder / "timeline_segments.json"
            if segments_file.exists():
                candidates.append(folder)

    if not candidates:
        raise FileNotFoundError(f"No timeline_segments.json found in any output subfolder")

    # Sort by name (timestamp format) and return latest
    candidates.sort(key=lambda x: x.name, reverse=True)
    return candidates[0]


def find_video_root(project_dir: Path) -> Path:
    """Determine video root from config.yaml or common locations."""
    # Try config.yaml first
    config_path = project_dir / "config.yaml"
    if config_path.exists():
        try:
            import yaml
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            video_root = config.get('paths', {}).get('video_root')
            if video_root and Path(video_root).exists():
                return Path(video_root)
        except Exception:
            pass

    # Try checkpoint for video paths
    checkpoint_path = project_dir / "checkpoint.json"
    if checkpoint_path.exists():
        try:
            with open(checkpoint_path, 'r', encoding='utf-8') as f:
                checkpoint = json.load(f)
            # Look for any video path in download stage
            downloads = checkpoint.get('download', {}).get('downloaded_files', [])
            if downloads:
                first_video = Path(downloads[0])
                if first_video.exists():
                    return first_video.parent.parent  # Go up from keyword folder
        except Exception:
            pass

    # Common fallback locations
    fallbacks = [
        project_dir / "videos",
        project_dir / "broll",
        project_dir.parent / "v" / project_dir.name,
    ]

    for fb in fallbacks:
        if fb.exists():
            return fb

    return project_dir / "videos"  # Default even if doesn't exist


def load_config_defaults(project_dir: Path) -> dict:
    """Load defaults from config.yaml if present."""
    defaults = {
        'frame_rate': 30.0,
        'timeline_start_tc': '01:00:00:00',
        'include_alternatives': True,
        'num_alternatives': 2,
    }

    config_path = project_dir / "config.yaml"
    if config_path.exists():
        try:
            import yaml
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            output_cfg = config.get('output', {})
            defaults['frame_rate'] = output_cfg.get('frame_rate', defaults['frame_rate'])
            defaults['timeline_start_tc'] = output_cfg.get('timeline_start_tc', defaults['timeline_start_tc'])
            defaults['include_alternatives'] = output_cfg.get('include_alternatives', defaults['include_alternatives'])
            defaults['num_alternatives'] = output_cfg.get('num_alternatives', defaults['num_alternatives'])
        except Exception:
            pass

    return defaults


def main():
    parser = argparse.ArgumentParser(
        description='Regenerate OTIO timeline from existing checkpoint/segment data.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        'project_dir',
        nargs='?',
        default='.',
        help='Project directory (default: current directory)'
    )
    parser.add_argument(
        '--output', '-o',
        help='Specific output folder name (e.g., 20260113_203704). Default: latest'
    )
    parser.add_argument(
        '--video-root', '-v',
        help='Video root directory. Default: auto-detect from config/checkpoint'
    )
    parser.add_argument(
        '--fps', '-f',
        type=float,
        help='Frame rate (default: from config or 30.0)'
    )
    parser.add_argument(
        '--timecode', '-tc',
        help='Timeline start timecode (default: from config or 01:00:00:00)'
    )
    parser.add_argument(
        '--no-alternatives',
        action='store_true',
        help='Skip alternative tracks (V2, V3)'
    )

    args = parser.parse_args()

    # Resolve project directory
    project_dir = Path(args.project_dir).resolve()
    if not project_dir.exists():
        print(f"ERROR: Project directory not found: {project_dir}")
        sys.exit(1)

    print(f"Project: {project_dir}")

    # Load config defaults
    defaults = load_config_defaults(project_dir)

    # Find source output folder
    if args.output:
        source_output = project_dir / "output" / args.output
        if not source_output.exists():
            print(f"ERROR: Output folder not found: {source_output}")
            sys.exit(1)
    else:
        try:
            source_output = find_latest_output(project_dir)
        except FileNotFoundError as e:
            print(f"ERROR: {e}")
            sys.exit(1)

    print(f"Source: {source_output.name}")

    # Load segment data
    segments_path = source_output / "timeline_segments.json"
    print(f"Loading segments from: {segments_path}")

    with open(segments_path, 'r', encoding='utf-8') as f:
        segments_data = json.load(f)

    segments = segments_data['segments']
    print(f"  Found {len(segments)} segments")

    # Load voiceover timing from checkpoint
    checkpoint_path = project_dir / "checkpoint.json"
    if not checkpoint_path.exists():
        print(f"ERROR: checkpoint.json not found at {checkpoint_path}")
        sys.exit(1)

    with open(checkpoint_path, 'r', encoding='utf-8') as f:
        checkpoint = json.load(f)

    vo_segments = checkpoint.get('analyze', {}).get('segments', [])
    print(f"  Loaded {len(vo_segments)} voiceover segments from checkpoint")

    if len(vo_segments) != len(segments):
        print(f"  WARNING: Segment count mismatch! VO={len(vo_segments)}, Segments={len(segments)}")

    # Determine video root
    if args.video_root:
        video_root = Path(args.video_root)
    else:
        video_root = find_video_root(project_dir)
    print(f"  Video root: {video_root}")

    # Import pipeline classes
    from src.utils import SRTSegment, Match, MatchResult, AlternativeMatch

    # Build video file cache for faster lookups
    print("  Building video file index...")
    video_cache = {}
    if video_root.exists():
        for p in video_root.rglob("*"):
            if p.is_file() and p.suffix.lower() in ['.mp4', '.mov', '.avi', '.mkv', '.webm']:
                video_cache[p.name] = str(p)
    print(f"  Indexed {len(video_cache)} video files")

    # Reconstruct MatchResult objects
    matches = []
    missing_files = 0

    for i, seg in enumerate(segments):
        # Get voiceover timing from checkpoint (accurate)
        vo_data = vo_segments[i] if i < len(vo_segments) else {}

        vo_seg = SRTSegment(
            index=i,
            start_time=vo_data.get('start', 0),
            end_time=vo_data.get('end', 0),
            text=seg.get('voiceover_text', ''),
            source_file=''
        )

        # Get video clip info
        v1_clip = seg.get('v1_clip', {})
        clip_file = v1_clip.get('file', '')

        # Find video path from cache
        video_path = video_cache.get(clip_file, '')
        if clip_file and not video_path:
            missing_files += 1

        vid_seg = SRTSegment(
            index=i,
            start_time=v1_clip.get('source_start', 0),
            end_time=v1_clip.get('source_end', 0),
            text='',
            source_file=video_path
        )

        match = Match(
            voiceover_segment=vo_seg,
            video_segment=vid_seg,
            video_scene=None,
            confidence=v1_clip.get('confidence', 0.5),
            reasoning='',
            is_keyword_match=False,
            is_visual_match=False,
            embedding_similarity=0.0,
            clip_reuse_count=0
        )

        # Build alternatives
        alternatives = []
        if not args.no_alternatives:
            for alt in seg.get('alternatives', []):
                alt_file = alt.get('file', '')
                alt_path = video_cache.get(alt_file, '')

                alt_vid_seg = SRTSegment(
                    index=i,
                    start_time=0,
                    end_time=0,
                    text='',
                    source_file=alt_path
                )
                alt_match = AlternativeMatch(
                    video_segment=alt_vid_seg,
                    video_scene=None,
                    confidence=alt.get('confidence', 0.5),
                    reasoning='',
                )
                alternatives.append(alt_match)

        match_result = MatchResult(
            primary_match=match,
            alternatives=alternatives
        )
        matches.append(match_result)

    print(f"  Reconstructed {len(matches)} MatchResult objects")
    if missing_files > 0:
        print(f"  WARNING: {missing_files} video files not found in video root")

    # Find voiceover path
    voiceover_path = None
    for name in ['combined_output.mp3', 'combined_output.wav', 'voiceover.mp3', 'voiceover.wav']:
        candidate = project_dir / "voiceover" / name
        if candidate.exists():
            voiceover_path = str(candidate)
            break

    if not voiceover_path:
        voiceover_path = str(project_dir / "voiceover" / "combined_output.mp3")
        print(f"  WARNING: Voiceover file not found, using default path")

    # Create new output directory
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = project_dir / "output" / run_timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput: {output_dir}")

    # Import OTIO functions
    from src.otio_builder import (
        create_timeline,
        save_timeline_split,
        generate_segment_map
    )

    # Build config
    frame_rate = args.fps or defaults['frame_rate']
    timeline_start_tc = args.timecode or defaults['timeline_start_tc']

    class MinimalConfig:
        class Output:
            pass
        output = Output()

    config = MinimalConfig()
    config.output.frame_rate = frame_rate
    config.output.timeline_start_tc = timeline_start_tc
    config.output.split_otio = True
    config.output.include_alternatives = not args.no_alternatives and defaults['include_alternatives']
    config.output.num_alternatives = defaults['num_alternatives']
    config.output.include_strategy_tracks = False
    config.output.strategy_tracks = []

    print(f"\nSettings:")
    print(f"  Frame rate: {frame_rate} fps")
    print(f"  Start TC: {timeline_start_tc}")
    print(f"  Alternatives: {'Yes' if config.output.include_alternatives else 'No'}")

    # Create timeline
    print(f"\nGenerating OTIO timeline...")
    timeline = create_timeline(
        matches=matches,
        config=config,
        voiceover_path=voiceover_path,
        frame_rate=frame_rate,
        entity_images=None,
        entity_videos=None,
    )

    # Save split OTIO files
    print("Saving OTIO files...")
    otio_paths = save_timeline_split(timeline, str(output_dir / "timeline"))

    for p in otio_paths:
        print(f"  + {Path(p).name}")

    # Generate segment map
    print("\nGenerating segment map...")
    segment_map_path = generate_segment_map(
        matches=matches,
        output_path=str(output_dir / "timeline"),
        frame_rate=frame_rate,
        source_srt=voiceover_path,
        timeline_start_tc=timeline_start_tc
    )
    print(f"  + {Path(segment_map_path).name}")

    print(f"\n{'='*60}")
    print(f"SUCCESS! OTIO files regenerated")
    print(f"{'='*60}")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
