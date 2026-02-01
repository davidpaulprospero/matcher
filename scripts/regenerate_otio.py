"""
Standalone OTIO Regeneration Script

Regenerates OTIO/EDL/XML output files directly from existing timeline_segments.json,
bypassing the checkpoint system entirely. Use this when:
- Checkpoint is corrupted but output files exist
- You want to regenerate with different config settings
- --output-only fails due to missing stage data

Usage:
    python scripts/regenerate_otio.py "E:\\Edit Job\\client\\project"
    python scripts/regenerate_otio.py "E:\\Edit Job\\client\\project" --output-folder 20260124_052654
    python scripts/regenerate_otio.py "E:\\Edit Job\\client\\project" --voiceover "path\\to\\audio.mp3"
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to path
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Fix Windows console encoding
import io
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from src.state import Match

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def find_latest_output_folder(project_dir: Path) -> Optional[Path]:
    """Find the most recent output folder with timeline_segments.json."""
    output_dir = project_dir / "output"
    if not output_dir.exists():
        return None

    # Find all timestamped folders
    candidates = []
    for folder in output_dir.iterdir():
        if folder.is_dir() and (folder / "timeline_segments.json").exists():
            candidates.append(folder)

    if not candidates:
        return None

    # Sort by name (timestamp format YYYYMMDD_HHMMSS sorts correctly)
    candidates.sort(key=lambda p: p.name, reverse=True)
    return candidates[0]


def load_segments_json(json_path: Path) -> Dict[str, Any]:
    """Load and parse timeline_segments.json."""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


class VideoPathResolver:
    """
    Resolves video references to actual file paths using multiple strategies:
    1. Transcription cache hash mapping
    2. Search in video root directories
    3. Search in project-specific directories
    """

    def __init__(self, project_dir: Path, video_roots: List[Path] = None):
        self._hash_mapping: Dict[str, str] = {}
        self._file_index: Dict[str, str] = {}  # filename -> full path

        # Build hash mapping from transcription cache
        cache_dir = project_dir / ".cache"
        if cache_dir.exists():
            self._build_hash_mapping(cache_dir)

        # Build file index from video roots
        if video_roots:
            for root in video_roots:
                if root.exists():
                    self._index_video_files(root)

    def _build_hash_mapping(self, cache_dir: Path):
        """Build hash-to-file mapping from transcription cache."""
        transcriptions_dir = cache_dir / "transcriptions"
        if not transcriptions_dir.exists():
            return

        for cache_file in transcriptions_dir.glob("*.json"):
            hash_id = cache_file.stem
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

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
                    self._hash_mapping[hash_id] = source_file
                    # Also index by filename
                    filename = Path(source_file).name
                    self._file_index[filename] = source_file
            except Exception:
                pass

        logger.info(f"Built hash mapping with {len(self._hash_mapping)} entries")

    def _index_video_files(self, root: Path):
        """Index video files from a root directory."""
        for pattern in ['**/*.mp4', '**/*.webm', '**/*.mkv']:
            for f in root.glob(pattern):
                self._file_index[f.name] = str(f)

    def resolve(self, video_ref: str) -> str:
        """Resolve a video reference to actual file path."""
        if not video_ref:
            return ''

        # If it's already a valid path, return as-is
        if Path(video_ref).exists():
            return str(video_ref)

        # Try hash mapping first (for hash IDs)
        if video_ref in self._hash_mapping:
            return self._hash_mapping[video_ref]

        # Try file index (for filenames)
        if video_ref in self._file_index:
            return self._file_index[video_ref]

        # Try with common extensions
        for ext in ['.mp4', '.webm', '.mkv']:
            test_name = video_ref if video_ref.endswith(ext) else video_ref + ext
            if test_name in self._file_index:
                return self._file_index[test_name]

        # Try partial match (video_id prefix)
        for filename, path in self._file_index.items():
            if filename.startswith(video_ref):
                return path

        # Return original if not found
        return video_ref


class TimelineAwareMatchWrapper:
    """
    Custom MatchResultWrapper that uses timeline timing instead of video timing.

    The standard MatchResultWrapper uses video_start/video_end for voiceover_segment,
    but for regeneration we need to use the original timeline frame positions.
    """

    def __init__(self, simple_match: Match, timeline_start: float, timeline_end: float,
                 alternatives=None, secondary_matches=None):
        self._simple_match = simple_match
        self._timeline_start = timeline_start
        self._timeline_end = timeline_end
        self._alternatives = alternatives or []
        self._secondary_matches = secondary_matches or []

        # Create voiceover segment with TIMELINE timing (not video timing)
        self._voiceover_segment = self._TimelineSegment(
            timeline_start, timeline_end, '',
            index=getattr(simple_match, 'segment_index', 0)
        )
        # Video segment uses source video timing
        self._video_segment = self._VideoSegment(simple_match)

    class _TimelineSegment:
        """Segment with timeline positioning."""
        def __init__(self, start: float, end: float, text: str, index: int = 0):
            self.start_time = start
            self.end_time = end
            self.text = text
            self.index = index

    class _VideoSegment:
        """Video segment with source video timing."""
        def __init__(self, m):
            self.source_file = getattr(m, 'video_file', '')
            self.start_time = getattr(m, 'video_start', 0.0)
            self.end_time = getattr(m, 'video_end', 0.0)
            self.text = ''

    class _AltMatch:
        """Alternative match wrapper."""
        def __init__(self, data: dict):
            self.video_segment = TimelineAwareMatchWrapper._AltVideoSegment(data)
            self.video_scene = None
            self.confidence = data.get('confidence', 0.5)
            self.reasoning = data.get('strategy', '')
            self.strategy = data.get('strategy', '')

    class _AltVideoSegment:
        """Video segment from dict."""
        def __init__(self, data: dict):
            self.source_file = data.get('source_file', '')
            self.start_time = data.get('start_time', 0.0)
            self.end_time = data.get('end_time', 0.0)
            self.text = ''

    class _PrimaryMatch:
        """Primary match wrapper."""
        def __init__(self, wrapper):
            self._wrapper = wrapper
            self._simple = wrapper._simple_match

        @property
        def voiceover_segment(self):
            return self._wrapper._voiceover_segment

        @property
        def video_segment(self):
            return self._wrapper._video_segment

        @property
        def video_scene(self):
            return None

        @property
        def confidence(self):
            return getattr(self._simple, 'confidence', 0.5)

        @property
        def reasoning(self):
            return getattr(self._simple, 'reason', '')

        @property
        def is_keyword_match(self):
            return False

        @property
        def is_visual_match(self):
            return False

        @property
        def embedding_similarity(self):
            return 0.0

        @property
        def clip_reuse_count(self):
            return 0

    @property
    def primary_match(self):
        return self._PrimaryMatch(self)

    @property
    def alternatives(self):
        return [self._AltMatch(a) for a in self._alternatives]

    @property
    def secondary_matches(self):
        return [self._AltMatch(s) for s in self._secondary_matches]

    @property
    def strategy_matches(self):
        return []

    @property
    def has_gap(self):
        return False

    @property
    def gap_reason(self):
        return ''


def reconstruct_matches(segments_data: Dict[str, Any], resolver: 'VideoPathResolver') -> List[TimelineAwareMatchWrapper]:
    """
    Reconstruct match objects from timeline_segments.json data.

    Uses TimelineAwareMatchWrapper to preserve original timeline positioning.

    Args:
        segments_data: Parsed JSON from timeline_segments.json
        resolver: VideoPathResolver for resolving file paths
    """
    matches = []
    frame_rate = segments_data.get('frame_rate', 30.0)

    for i, segment in enumerate(segments_data.get('segments', [])):
        # Get timeline positioning (converted from frames to seconds)
        start_frame = segment.get('start_frame', 0)
        end_frame = segment.get('end_frame', start_frame + 30)
        timeline_start = start_frame / frame_rate
        timeline_end = end_frame / frame_rate

        # Get V1 clip info
        v1_clip = segment.get('v1_clip', {})
        video_file = v1_clip.get('file', '')

        # Resolve video file path
        video_file = resolver.resolve(video_file)

        # Create simple Match object (with VIDEO timing for source clip)
        simple_match = Match(
            segment_index=i,
            video_file=video_file,
            video_start=v1_clip.get('source_start', 0.0),
            video_end=v1_clip.get('source_end', 0.0),
            confidence=v1_clip.get('confidence', 0.0),
            strategy=v1_clip.get('strategy', 'restored'),
            reason=v1_clip.get('reason', ''),
            face_score=v1_clip.get('face_score', 0.5)
        )

        # Extract alternatives (V2-V3)
        alternatives = []
        for alt in segment.get('alternatives', []):
            alt_file = resolver.resolve(alt.get('file', ''))
            alternatives.append({
                'source_file': alt_file,
                'start_time': alt.get('source_start', 0.0),
                'end_time': alt.get('source_end', 0.0),
                'confidence': alt.get('confidence', 0.5),
                'strategy': alt.get('strategy', '')
            })

        # Extract secondary matches (V4-V6)
        secondary_matches = []
        for sec in segment.get('secondary', []):
            sec_file = resolver.resolve(sec.get('file', ''))
            secondary_matches.append({
                'source_file': sec_file,
                'start_time': sec.get('source_start', 0.0),
                'end_time': sec.get('source_end', 0.0),
                'confidence': sec.get('confidence', 0.5),
                'strategy': sec.get('strategy', '')
            })

        # Wrap with TIMELINE timing (not video timing)
        wrapped = TimelineAwareMatchWrapper(
            simple_match, timeline_start, timeline_end,
            alternatives, secondary_matches
        )
        matches.append(wrapped)

    return matches


def load_config(project_dir: Path) -> 'Config':
    """Load configuration from config files."""
    print(f"  Loading config...")
    from src.cli.config_utils import load_project_config

    # Load config and set project directory
    global_config_path = PROJECT_ROOT / "config.yaml"
    config = load_project_config(project_dir, global_config_path)
    print(f"  Config loaded")

    # Set output directory
    config.otio_output_dir = str(project_dir / "output")

    return config


def regenerate_otio(
    project_dir: Path,
    output_folder: Optional[str] = None,
    voiceover_path: Optional[str] = None
) -> Path:
    """
    Regenerate OTIO files from existing timeline_segments.json.

    Args:
        project_dir: Project directory path
        output_folder: Specific output folder name (e.g., "20260124_052654")
        voiceover_path: Optional path to voiceover file

    Returns:
        Path to the new output directory
    """
    print(f"\n=== OTIO Regeneration ===")
    print(f"Project: {project_dir}")

    # Find source data
    if output_folder:
        source_dir = project_dir / "output" / output_folder
    else:
        source_dir = find_latest_output_folder(project_dir)

    if not source_dir or not source_dir.exists():
        raise FileNotFoundError(f"No output folder found in {project_dir / 'output'}")

    segments_json = source_dir / "timeline_segments.json"
    if not segments_json.exists():
        raise FileNotFoundError(f"timeline_segments.json not found in {source_dir}")

    print(f"Source: {segments_json}")

    # Load segment data
    segments_data = load_segments_json(segments_json)
    print(f"Loaded {segments_data.get('total_segments', 0)} segments")

    # Build video path resolver
    print(f"  Building video path resolver...")

    # Collect video root directories to search
    video_roots = []
    config = load_config(project_dir)

    # Add config root dir (E:/v)
    if hasattr(config, 'download') and hasattr(config.download, 'root_dir'):
        root_dir = config.download.root_dir
        if root_dir and Path(root_dir).exists():
            video_roots.append(Path(root_dir))

    # Add project-specific video dir
    project_video_dir = project_dir / "videos"
    if project_video_dir.exists():
        video_roots.append(project_video_dir)

    # Create resolver with transcription cache + video roots
    resolver = VideoPathResolver(project_dir, video_roots)
    print(f"  Indexed {len(resolver._file_index)} video files, {len(resolver._hash_mapping)} hash mappings")

    # Reconstruct matches
    matches = reconstruct_matches(segments_data, resolver)
    print(f"Reconstructed {len(matches)} matches")

    # Load config
    config = load_config(project_dir)

    # Get voiceover path
    vo_path = voiceover_path or segments_data.get('source_srt', '')
    if vo_path and not Path(vo_path).exists():
        # Try to find voiceover in project dir
        for ext in ['*.mp3', '*.wav', '*.srt', '*.mp4']:
            candidates = list(project_dir.glob(ext))
            if candidates:
                vo_path = str(candidates[0])
                break

    print(f"Voiceover: {vo_path or 'None'}")

    # Import OTIO modules
    print(f"  Importing OTIO modules...")
    from src.otio import (
        create_timeline,
        save_timeline_as_edl,
        generate_segment_map,
        generate_resolve_xml_with_bins
    )
    from src.otio.export import save_timeline_split_with_config
    print(f"  OTIO modules imported")

    # Create new output directory
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = project_dir / "output" / run_timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput: {output_dir}")

    # Get frame rate from source data or config
    frame_rate = segments_data.get('frame_rate', getattr(config.output, 'frame_rate', 30.0))

    # Create timeline
    print(f"\nGenerating timeline...")
    timeline = create_timeline(
        matches=matches,
        config=config,
        voiceover_path=vo_path if vo_path else None,
        frame_rate=frame_rate,
        entity_images=None,
        entity_videos=None,
        downloaded_segments=None
    )

    # Generate OTIO files
    print(f"Saving OTIO files...")
    otio_results = save_timeline_split_with_config(
        timeline=timeline,
        output_path=str(output_dir / "timeline"),
        config=config.output
    )
    otio_paths = list(otio_results.values()) if isinstance(otio_results, dict) else [otio_results]
    for path in otio_paths:
        print(f"  + {Path(path).name}")

    # Generate segment map
    segment_map_path = generate_segment_map(
        matches=matches,
        output_path=str(output_dir / "timeline"),
        frame_rate=frame_rate,
        source_srt=vo_path or '',
        timeline_start_tc=segments_data.get('timeline_start_tc', "01:00:00:00")
    )
    print(f"  + {Path(segment_map_path).name}")

    # Generate EDL
    print(f"Generating EDL...")
    edl_path = save_timeline_as_edl(
        matches=matches,
        output_path=str(output_dir / "timeline.edl"),
        frame_rate=frame_rate
    )
    if edl_path:
        print(f"  + {Path(edl_path).name}")

    # Generate XML
    print(f"Generating DaVinci XML...")
    xml_paths = generate_resolve_xml_with_bins(
        matches=matches,
        output_path=str(output_dir / "timeline"),
        voiceover_path=vo_path,
        frame_rate=frame_rate,
        config=config
    )
    for path in xml_paths if isinstance(xml_paths, list) else [xml_paths]:
        if path:
            print(f"  + {Path(path).name}")

    print(f"\n[OK] Regeneration complete: {output_dir}")
    return output_dir


def main():
    parser = argparse.ArgumentParser(
        description="Regenerate OTIO files from existing timeline_segments.json",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/regenerate_otio.py "E:\\Edit Job\\theresa\\PaidTest"
    python scripts/regenerate_otio.py "E:\\Edit Job\\Stu\\January\\22__2026-01-17" --output-folder 20260124_052654
    python scripts/regenerate_otio.py "E:\\Edit Job\\client\\project" --voiceover "audio.mp3"
        """
    )

    parser.add_argument(
        'project',
        type=str,
        help='Project directory path'
    )

    parser.add_argument(
        '--output-folder', '-o',
        type=str,
        default=None,
        help='Specific output folder name (default: latest)'
    )

    parser.add_argument(
        '--voiceover', '-v',
        type=str,
        default=None,
        help='Path to voiceover file (default: from segments.json)'
    )

    parser.add_argument(
        '--validate',
        action='store_true',
        help='Validate generated OTIO against media files on disk'
    )

    args = parser.parse_args()

    project_dir = Path(args.project)
    if not project_dir.exists():
        print(f"Error: Project directory not found: {project_dir}")
        sys.exit(1)

    try:
        output_dir = regenerate_otio(
            project_dir=project_dir,
            output_folder=args.output_folder,
            voiceover_path=args.voiceover
        )

        # Validate if requested
        if args.validate:
            print("\nValidating media references...")
            from validate_otio_media import validate_otio, print_report
            otio_path = output_dir / "timeline_FULL.otio"
            if otio_path.exists():
                result = validate_otio(str(otio_path))
                print_report(result, verbose=True)
                if result['missing_files']:
                    print("\n[WARNING] Some media files are missing!")
                    sys.exit(1)
            else:
                print(f"  Could not find {otio_path} for validation")

        sys.exit(0)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
