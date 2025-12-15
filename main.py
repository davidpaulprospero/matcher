#!/usr/bin/env python3
"""
Voiceover-Matcher: End-to-End Documentary Footage Pipeline

One command to go from voiceover → matched timeline:
1. Extract keywords from voiceover (LLM)
2. Download footage from YouTube (all 3 duration tiers)
3. Transcribe & index videos
4. Match to voiceover segments
5. Output DaVinci Resolve timeline

Usage:
    python main.py --voiceover script.srt
    python main.py --voiceover script.srt --keywords 30
    python main.py --project "E:\\Projects\\MyDoc" --voiceover voiceover.srt
    python main.py --resume  # Resume interrupted run
    python main.py --match-only  # Skip download, just match existing footage

Project Mode:
    When --project is specified, all paths become project-relative:
    - downloaded_videos/ is in project folder
    - .cache/ is in project folder
    - logs/ is in project folder
    - otio_output/ is in project folder
    
    Global .env and config.yaml are loaded from install directory.
    Project-specific overrides can be placed in project_config.yaml
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Optional

# Determine install directory (where this script lives)
INSTALL_DIR = Path(__file__).parent.resolve()

# Project directory (set later via --project argument)
PROJECT_DIR = None


def strip_extended_path_prefix(path: Path) -> Path:
    r"""
    Strip Windows extended-length path prefix (\\?\) from a Path.
    This prefix can cause issues with some applications.
    """
    path_str = str(path)
    
    # Remove Windows extended-length path prefix
    prefixes = ['\\\\?\\', '\\\\.\\', '//?/', '//./']
    for prefix in prefixes:
        if path_str.startswith(prefix):
            path_str = path_str[len(prefix):]
            break
    
    return Path(path_str)


def load_environment(project_dir: Path = None):
    """Load .env file from install dir, then optionally from project dir"""
    try:
        from dotenv import load_dotenv
        
        # Load global .env from install directory
        global_env = INSTALL_DIR / '.env'
        if global_env.exists():
            load_dotenv(global_env)
            print(f"  ✓ Loaded global environment from {global_env}")
        
        # Load project-specific .env (overrides global)
        if project_dir:
            project_env = project_dir / '.env'
            if project_env.exists():
                load_dotenv(project_env, override=True)
                print(f"  ✓ Loaded project environment from {project_env}")
        
        # Fallback: try current working directory
        if not global_env.exists() and Path('.env').exists():
            load_dotenv()
            print(f"  ✓ Loaded environment from .env")
            
    except ImportError:
        pass  # dotenv not installed, rely on system env vars


def make_paths_project_relative(config, project_dir: Path):
    """Update config paths to be relative to project directory"""
    project_dir = project_dir.resolve()
    
    # These paths should be in the project folder
    project_relative_paths = [
        'downloaded_videos_dir',
        'otio_output_dir',
        'cache_dir',
    ]
    
    for attr in project_relative_paths:
        if hasattr(config, attr):
            current = getattr(config, attr)
            # If it's a relative path, make it relative to project
            if current and not Path(current).is_absolute():
                new_path = str(project_dir / current)
                setattr(config, attr, new_path)
    
    # Handle logging separately
    if hasattr(config, 'logging') and hasattr(config.logging, 'log_dir'):
        log_dir = config.logging.log_dir
        if log_dir and not Path(log_dir).is_absolute():
            config.logging.log_dir = str(project_dir / log_dir)
    
    # Voiceover path - if relative, make project-relative
    if hasattr(config, 'voiceover_path') and config.voiceover_path:
        vo_path = Path(config.voiceover_path)
        if not vo_path.is_absolute():
            config.voiceover_path = str(project_dir / vo_path)
    
    return config


def load_project_config(project_dir: Path, global_config_path: Path = None):
    """
    Load configuration with project overrides.
    
    1. Load global config from install dir
    2. Load project_config.yaml if it exists
    3. Merge (project overrides global)
    4. Make paths project-relative
    """
    from config import load_config
    import yaml
    
    # Load global config
    if global_config_path is None:
        global_config_path = INSTALL_DIR / 'config.yaml'
    
    config = load_config(str(global_config_path))
    
    # Load and merge project config if it exists
    project_config_path = project_dir / 'project_config.yaml'
    if project_config_path.exists():
        print(f"  ✓ Loading project config: {project_config_path}")
        try:
            with open(project_config_path, 'r', encoding='utf-8') as f:
                project_overrides = yaml.safe_load(f)
            
            if project_overrides:
                # Deep merge project overrides into config
                config = merge_config(config, project_overrides)
        except Exception as e:
            print(f"  ⚠ Could not load project config: {e}")
    
    # Make paths project-relative
    config = make_paths_project_relative(config, project_dir)
    
    return config


def merge_config(config, overrides: dict):
    """Deep merge overrides into config object"""
    if not overrides:
        return config
    
    for key, value in overrides.items():
        if hasattr(config, key):
            current = getattr(config, key)
            
            # If both are objects with attributes, recurse
            if isinstance(value, dict) and hasattr(current, '__dict__'):
                merge_config(current, value)
            else:
                # Direct override
                setattr(config, key, value)
        else:
            # New attribute
            setattr(config, key, value)
    
    return config


# Add src to path
sys.path.insert(0, str(INSTALL_DIR / 'src'))

from config import Config, load_config
from downloader import VideoDownloader, DownloadCheckpoint
from keyword_extractor import LLMKeywordExtractor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


class Pipeline:
    """End-to-end documentary footage pipeline"""
    
    def __init__(self, config: Config):
        self.config = config
        self.downloader = None
        self.keyword_extractor = None
        self.scene_detector = None
        
        # State tracking
        self.keywords: List[str] = []
        self.downloaded_videos: List[dict] = []
        self.voiceover_segments: List[dict] = []
        
        # Processing state
        self.transcripts = {}
        self.embeddings = []
        self.text_metadata = []
        self.embedding_index = None
        self.matches = []
        self.cache = None
        
        # Auto-logging
        self.run_logger = None
        if getattr(config, 'logging', None) and getattr(config.logging, 'enabled', False):
            try:
                from src.logger import RunLogger
                log_dir = getattr(config.logging, 'log_dir', './logs')
                self.run_logger = RunLogger(log_dir=log_dir)
                self.run_logger.log_config(config)
                print(f"  ✓ Logging enabled: {self.run_logger.log_file}")
            except Exception as e:
                print(f"  ⚠ Could not initialize logger: {e}")
    
    def _print_banner(self):
        """Print startup banner"""
        print("=" * 70)
        print("  VOICEOVER-MATCHER: End-to-End Documentary Pipeline v2.2")
        print("=" * 70)
        print(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        if PROJECT_DIR and PROJECT_DIR != Path.cwd():
            print(f"  Project: {PROJECT_DIR.name}")
            print(f"  Location: {PROJECT_DIR}")
        print(f"  Videos: {self.config.downloaded_videos_dir}")
        print(f"  Output: {self.config.otio_output_dir}")
        if self.run_logger:
            print(f"  Log: {self.run_logger.log_file}")
        print("=" * 70)
    
    def _print_stage(self, stage: int, name: str):
        """Print stage header and log it"""
        print(f"\n{'─' * 70}")
        print(f"  STAGE {stage}: {name}")
        print(f"{'─' * 70}")
        
        # Log to file
        if self.run_logger:
            self.run_logger.file_logger.info(f"STAGE {stage}: {name}")
    
    def _parse_srt(self, srt_path: str) -> List[dict]:
        """Parse SRT file into segments"""
        try:
            import srt
            with open(srt_path, 'r', encoding='utf-8') as f:
                subtitles = list(srt.parse(f.read()))
            
            segments = []
            for sub in subtitles:
                segments.append({
                    'index': sub.index,
                    'start_time': sub.start.total_seconds(),
                    'end_time': sub.end.total_seconds(),
                    'text': sub.content.strip(),
                    'duration': (sub.end - sub.start).total_seconds()
                })
            return segments
        except ImportError:
            logger.error("srt package not installed. Install with: pip install srt")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Failed to parse SRT: {e}")
            sys.exit(1)
    
    def _get_user_confirmation(self, prompt: str, default: bool = True) -> bool:
        """Get Y/N confirmation from user"""
        suffix = "[Y/n]" if default else "[y/N]"
        try:
            response = input(f"{prompt} {suffix}: ").strip().lower()
            if not response:
                return default
            return response in ('y', 'yes')
        except (EOFError, KeyboardInterrupt):
            print("\nAborted.")
            sys.exit(0)
    
    def _get_user_input(self, prompt: str, default: str = "") -> str:
        """Get text input from user"""
        suffix = f"[{default}]" if default else ""
        try:
            response = input(f"{prompt} {suffix}: ").strip()
            return response if response else default
        except (EOFError, KeyboardInterrupt):
            print("\nAborted.")
            sys.exit(0)
    
    def stage_analyze_voiceover(self, voiceover_path: str) -> List[dict]:
        """Stage 1: Parse and analyze voiceover (supports SRT, audio, and video files)"""
        self._print_stage(1, "ANALYZE VOICEOVER")
        
        voiceover_path = Path(voiceover_path)
        
        if not voiceover_path.exists():
            logger.error(f"Voiceover file not found: {voiceover_path}")
            sys.exit(1)
        
        # Supported formats
        audio_extensions = {'.mp3', '.wav', '.m4a', '.flac', '.ogg', '.wma', '.aac', '.opus'}
        video_extensions = {'.mp4', '.mov', '.avi', '.mkv', '.webm', '.wmv', '.flv', '.m4v'}
        
        suffix = voiceover_path.suffix.lower()
        needs_transcription = suffix in audio_extensions or suffix in video_extensions
        
        if needs_transcription:
            # Audio or video file - transcribe to SRT first
            file_type = "Video" if suffix in video_extensions else "Audio"
            print(f"  {file_type} file detected: {voiceover_path.name}")
            print(f"  Transcribing with faster-whisper (GPU)...")
            
            try:
                from src.transcription import transcribe_voiceover_media
                
                # Get transcription settings from config
                model_name = getattr(self.config.transcription, 'model', 'base')
                language = getattr(self.config.transcription, 'language', 'en')
                compute_type = getattr(self.config.transcription, 'compute_type', 'auto')
                
                # Output SRT next to the source file
                srt_path = voiceover_path.with_suffix('.srt')
                
                # Check if SRT already exists
                if srt_path.exists():
                    print(f"  ✓ Using existing SRT: {srt_path.name}")
                else:
                    srt_path = transcribe_voiceover_media(
                        media_path=str(voiceover_path),
                        output_srt_path=str(srt_path),
                        model_name=model_name,
                        language=language,
                        compute_type=compute_type
                    )
                    print(f"  ✓ Transcribed to: {srt_path}")
                
                voiceover_path = Path(srt_path)
                
            except ImportError as e:
                logger.error(f"Cannot transcribe - missing dependency: {e}")
                logger.error("Install with: pip install faster-whisper srt")
                sys.exit(1)
            except Exception as e:
                logger.error(f"Failed to transcribe voiceover: {e}")
                sys.exit(1)
        
        # Parse SRT
        logger.info(f"Parsing: {voiceover_path}")
        segments = self._parse_srt(str(voiceover_path))
        
        # Calculate stats
        total_duration = sum(s['duration'] for s in segments)
        avg_duration = total_duration / len(segments) if segments else 0
        
        print(f"  ✓ Parsed {len(segments)} segments")
        print(f"  ✓ Total duration: {total_duration / 60:.1f} minutes")
        print(f"  ✓ Avg segment: {avg_duration:.1f} seconds")
        
        self.voiceover_segments = segments
        return segments
    
    def stage_extract_keywords(
        self,
        segments: List[dict],
        max_keywords: int = 50,
        review_enabled: bool = True
    ) -> List[str]:
        """Stage 1b: Extract search keywords and entities from voiceover"""
        print(f"\n  Extracting keywords (max {max_keywords})...")
        
        # Initialize extractor
        self.keyword_extractor = LLMKeywordExtractor(self.config)
        
        # Extract keywords
        result = self.keyword_extractor.extract_keywords(
            segments,
            max_keywords=max_keywords,
            expand=True
        )
        
        keywords = result.keywords
        entities = getattr(result, 'entities', [])
        topic = getattr(result, 'topic', '')
        
        print(f"  ✓ Extracted {len(keywords)} keywords ({result.extraction_method})")
        if entities:
            print(f"  ✓ Found {len(entities)} named entities (people, places, dates)")
        
        # Store entities for later use
        self.entities = entities
        self.topic = topic
        
        # Show sample
        if keywords:
            print(f"\n  Sample keywords:")
            for kw in keywords[:10]:
                print(f"    • {kw}")
            if len(keywords) > 10:
                print(f"    ... and {len(keywords) - 10} more")
        
        if entities:
            print(f"\n  Named entities:")
            for ent in entities[:5]:
                ent_text = ent.get('text', ent.get('name', str(ent)))
                ent_type = ent.get('type', ent.get('label', ''))
                print(f"    • {ent_text} ({ent_type})")
            if len(entities) > 5:
                print(f"    ... and {len(entities) - 5} more")
        
        # Interactive keyword review (if enabled)
        if review_enabled and not getattr(self.config.pipeline, 'skip_keyword_review', False):
            try:
                from src.interactive import review_keywords
                
                # Determine project dir for saving removed keywords
                project_dir = None
                if hasattr(self.config, 'cache_dir'):
                    project_dir = Path(self.config.cache_dir).parent
                
                print(f"\n  Launching keyword review...")
                keywords = review_keywords(
                    keywords=keywords,
                    entities=entities,
                    topic=topic,
                    project_dir=project_dir
                )
                print(f"  ✓ {len(keywords)} keywords after review")
            except ImportError:
                pass  # Interactive module not available
            except Exception as e:
                logger.warning(f"Keyword review failed: {e}")
        
        self.keywords = keywords
        return keywords
    
    def stage_pre_run_summary(
        self,
        keywords: List[str],
        segments: List[dict]
    ) -> bool:
        """Show pre-run summary and get confirmation"""
        print(f"\n{'─' * 70}")
        print("  PRE-RUN SUMMARY")
        print(f"{'─' * 70}\n")
        
        # Initialize downloader for estimates
        self.downloader = VideoDownloader(self.config)
        
        # Check dependencies
        ok, msg = self.downloader.check_dependencies()
        print(msg)
        if not ok:
            return False
        
        # Get estimates
        estimate = self.downloader.get_download_estimate(len(keywords))
        
        # Check existing footage
        output_dir = Path(self.config.downloaded_videos_dir)
        inventory = self.downloader.get_inventory_report(output_dir)
        
        print(f"\n  Voiceover:")
        print(f"    • {len(segments)} segments")
        print(f"    • {sum(s['duration'] for s in segments) / 60:.1f} minutes total")
        
        print(f"\n  Keywords:")
        print(f"    • {len(keywords)} search terms")
        
        print(f"\n  Download Plan:")
        for tier, desc in estimate['tiers'].items():
            print(f"    • {tier}: {desc}")
        print(f"    • Total: ~{estimate['total_videos']} videos")
        print(f"    • Est. storage: {estimate['est_storage_gb']} GB")
        print(f"    • Est. time: {estimate['est_time_minutes']:.0f} minutes")
        
        if inventory['total_videos'] > 0:
            print(f"\n  Existing Footage:")
            print(f"    • {inventory['total_videos']} videos ({inventory['total_duration_hours']:.1f} hours)")
            print(f"    • {inventory['num_keywords_covered']} keywords covered")
        
        print()
        
        # Confirm
        if self.config.pipeline.confirm_before_download:
            return self._get_user_confirmation("Proceed with download?", default=True)
        return True
    
    def stage_download(
        self,
        keywords: List[str],
        resume: bool = False
    ) -> List[dict]:
        """Stage 2: Download footage"""
        self._print_stage(2, "DOWNLOAD FOOTAGE")
        
        if not self.downloader:
            self.downloader = VideoDownloader(self.config)
        
        output_dir = Path(self.config.downloaded_videos_dir)
        
        # Download
        downloaded, failed = self.downloader.download_all(
            keywords=keywords,
            output_dir=output_dir,
            max_concurrent=self.config.download.max_concurrent,
            resume=resume
        )
        
        print(f"\n  ✓ Downloaded {len(downloaded)} videos")
        if failed:
            print(f"  ⚠ {len(failed)} keywords with no results")
            
            # Save failed keywords
            failed_file = output_dir / "no_results_keywords.txt"
            with open(failed_file, 'w') as f:
                f.write(f"# Keywords with 0 results - {datetime.now().isoformat()}\n")
                for kw in failed:
                    f.write(f"{kw}\n")
            print(f"  ⚠ Saved to: {failed_file}")
        
        self.downloaded_videos = [d.to_dict() for d in downloaded]
        return self.downloaded_videos
    
    def stage_deduplicate(self) -> dict:
        """Stage 2b: Deduplicate downloaded footage"""
        self._print_stage("2b", "DEDUPLICATE FOOTAGE")
        
        try:
            from src.deduplication import VideoDeduplicator
        except ImportError as e:
            logger.warning(f"Could not import deduplication: {e}")
            print("  ⚠ Deduplication not available")
            print("    Install with: pip install imagehash Pillow")
            return {'duplicates_deleted': 0}
        
        video_dir = Path(self.config.downloaded_videos_dir)
        
        deduplicator = VideoDeduplicator(self.config)
        
        if not deduplicator.is_available():
            print("  ⚠ imagehash not installed, skipping deduplication")
            return {'duplicates_deleted': 0}
        
        # Run deduplication
        report_path = video_dir / "deduplication_report.json"
        report = deduplicator.deduplicate(
            video_dir=str(video_dir),
            auto_delete=True,
            report_path=str(report_path)
        )
        
        print(f"\n  Scanned: {report.total_videos} videos")
        print(f"  ✓ Unique: {report.unique_videos}")
        
        if report.duplicates_deleted > 0:
            print(f"  ✓ Deleted: {report.duplicates_deleted} duplicates")
            print(f"  ✓ Saved: {report.space_saved_mb:.1f} MB")
            print(f"  ✓ Report: {report_path}")
        else:
            print(f"  ✓ No duplicates found")
        
        return {
            'total_videos': report.total_videos,
            'duplicates_deleted': report.duplicates_deleted,
            'space_saved_mb': report.space_saved_mb
        }
    
    def stage_transcribe_index(self) -> dict:
        """Stage 3: Transcribe and index videos"""
        self._print_stage(3, "TRANSCRIBE & INDEX")
        
        # Import the existing transcription/indexing modules
        try:
            from src.transcription import transcribe_videos_parallel
            from src.embeddings import compute_embeddings, build_embedding_index, get_embedding_provider
            from src.vision import process_video_vision
            from src.utils import CacheManager, SRTSegment
        except ImportError as e:
            logger.warning(f"Could not import modules: {e}")
            logger.info("Using placeholder - integrate with existing modules")
            return {'videos_indexed': 0}
        
        video_dir = Path(self.config.downloaded_videos_dir)
        
        # Find all video files
        video_extensions = {'.mp4', '.mkv', '.webm', '.avi', '.mov', '.mxf'}
        videos = []
        for ext in video_extensions:
            videos.extend(video_dir.rglob(f'*{ext}'))
        
        print(f"  Found {len(videos)} videos to process")
        
        # Initialize cache
        cache = CacheManager(self.config.cache_dir)
        
        # Transcribe
        provider = getattr(self.config.transcription, 'provider', 'faster-whisper')
        print(f"  Transcribing with {provider} ({self.config.transcription.model})...")
        transcripts = transcribe_videos_parallel([str(v) for v in videos], cache, self.config)
        print(f"  ✓ Transcribed {len(transcripts)} videos")
        
        # Vision descriptions (if enabled)
        if self.config.vision.enabled:
            print(f"  Describing scenes with vision API...")
            all_scenes = {}
            for video_path in videos:
                video_str = str(video_path)
                # Get transcript segments for this video
                video_transcripts = transcripts.get(video_str, [])
                # Convert to SRTSegment format if needed
                segments = []
                for t in video_transcripts:
                    if hasattr(t, 'start_time'):
                        segments.append(t)
                    elif isinstance(t, dict):
                        segments.append(SRTSegment(
                            index=t.get('index', 0),
                            start_time=t.get('start_time', 0),
                            end_time=t.get('end_time', 0),
                            text=t.get('text', '')
                        ))
                scenes = process_video_vision(video_str, segments, cache, self.config)
                all_scenes[video_str] = scenes
            total_scenes = sum(len(s) for s in all_scenes.values())
            print(f"  ✓ Described {total_scenes} scenes across {len(all_scenes)} videos")
        
        # Extract text strings from transcripts for embedding
        all_texts = []
        text_metadata = []  # Track which video/segment each text belongs to
        for video_path, segments in transcripts.items():
            for seg in segments:
                if isinstance(seg, dict):
                    text = seg.get('text', '')
                else:
                    text = getattr(seg, 'text', '')
                if text and text.strip():
                    all_texts.append(text.strip())
                    text_metadata.append({'video': video_path, 'segment': seg})
        
        # Compute embeddings
        print(f"  Computing embeddings ({self.config.embedding.provider})...")
        if all_texts:
            embedding_provider = get_embedding_provider(self.config)
            embeddings = compute_embeddings(
                texts=all_texts,
                provider=embedding_provider,
                cache=cache,
                cache_key="video_transcripts"
            )
            print(f"  ✓ Computed embeddings for {len(embeddings)} clips")
        else:
            embeddings = []
            print(f"  ⚠ No text to embed")
        
        # Build index
        if self.config.indexing.use_faiss and embeddings:
            print(f"  Building FAISS index...")
            index = build_embedding_index(embeddings, self.config)
            print(f"  ✓ Built index")
        else:
            index = None
        
        # Store for later stages
        self.transcripts = transcripts
        self.embeddings = embeddings
        self.text_metadata = text_metadata
        self.embedding_index = index
        self.cache = cache
        
        return {
            'videos_indexed': len(videos),
            'clips_embedded': len(embeddings)
        }
    
    def stage_scene_detection(self) -> dict:
        """Stage 3b: Detect scene boundaries in videos"""
        self._print_stage("3b", "SCENE DETECTION")
        
        if not self.config.scene_detection.enabled:
            print("  ⚠ Scene detection disabled in config")
            return {'scenes': 0}
        
        try:
            from src.scene_detection import SceneDetector
        except ImportError as e:
            logger.warning(f"Could not import scene_detection: {e}")
            print("  ⚠ Scene detection module not available")
            print("    Install with: pip install scenedetect opencv-python")
            return {'scenes': 0}
        
        video_dir = Path(self.config.downloaded_videos_dir)
        
        # Initialize detector
        print(f"  Preset: {self.config.scene_detection.preset}")
        force_cuda = getattr(self.config.scene_detection, 'force_cuda', False)
        print(f"  CUDA: {'Forced' if force_cuda else 'Auto'}")
        print(f"  Timecode: {self.config.scene_detection.start_timecode}")
        
        detector = SceneDetector(self.config)
        
        # Process all videos
        min_duration = self.config.scene_detection.min_video_duration
        if min_duration > 0:
            print(f"  Min duration: {min_duration}s")
        
        results = detector.process_all_videos(video_dir, min_duration=min_duration)
        
        # Get stats
        stats = detector.get_stats()
        
        print(f"\n  ✓ Processed {stats['videos_processed']} videos")
        print(f"  ✓ Detected {stats['total_scenes']} total scenes")
        print(f"  ✓ Average: {stats['avg_scenes_per_video']} scenes/video")
        
        # Audio stats
        if stats.get('audio_analysis_enabled'):
            print(f"  ✓ Audio analysis: {stats['videos_with_speech']} videos with speech")
            print(f"  ✓ Audio cut points: {stats['total_audio_cut_points']}")
        
        if self.config.scene_detection.output_otio:
            print(f"  ✓ OTIO files: {detector.otio_output_dir}")
        if self.config.scene_detection.output_scene_index:
            print(f"  ✓ Scene index: {detector.scene_index_path}")
        
        # Store detector for use in matching
        self.scene_detector = detector
        
        return {
            'videos_processed': stats['videos_processed'],
            'total_scenes': stats['total_scenes'],
            'scene_clips': len(detector.get_scene_clips())
        }
    
    def stage_match(self) -> dict:
        """Stage 4: Match footage to voiceover"""
        self._print_stage(4, "MATCH FOOTAGE")
        
        try:
            from src.matching import match_all_segments
            from src.embeddings import get_embedding_provider, compute_embeddings
            from src.utils import SRTSegment
        except ImportError as e:
            logger.warning(f"Could not import modules: {e}")
            logger.info("Using placeholder - integrate with existing modules")
            return {'matches': 0}
        
        # Check prerequisites
        if not self.voiceover_segments:
            print("  ⚠ No voiceover segments (run stage 1 first)")
            return {'matches': 0}
        
        if not self.embeddings:
            print("  ⚠ No video embeddings (run stage 3 first)")
            return {'matches': 0}
        
        # Convert voiceover segments to SRTSegment format and filter empty
        vo_segments = []
        for seg in self.voiceover_segments:
            if isinstance(seg, dict):
                text = seg.get('text', '').strip()
                if text:  # Only include segments with text
                    vo_segments.append(SRTSegment(
                        index=seg.get('index', 0),
                        start_time=seg.get('start_time', 0),
                        end_time=seg.get('end_time', 0),
                        text=text
                    ))
            else:
                if seg.text and seg.text.strip():
                    vo_segments.append(seg)
        
        print(f"  Matching {len(vo_segments)} voiceover segments...")
        
        # Compute voiceover embeddings
        print(f"  Computing voiceover embeddings...")
        vo_texts = [seg.text for seg in vo_segments]
        embedding_provider = get_embedding_provider(self.config)
        vo_embeddings = compute_embeddings(
            texts=vo_texts,
            provider=embedding_provider,
            cache=self.cache,
            cache_key="voiceover_segments"
        )
        print(f"  ✓ {len(vo_embeddings)} voiceover embeddings")
        
        # Convert video text_metadata to SRTSegment format
        video_segments = []
        for meta in self.text_metadata:
            seg = meta['segment']
            if isinstance(seg, dict):
                video_segments.append(SRTSegment(
                    index=seg.get('index', 0),
                    start_time=seg.get('start_time', 0),
                    end_time=seg.get('end_time', 0),
                    text=seg.get('text', ''),
                    source_file=meta.get('video', '')
                ))
            else:
                # Already SRTSegment, just add source_file
                if not hasattr(seg, 'source_file') or not seg.source_file:
                    seg.source_file = meta.get('video', '')
                video_segments.append(seg)
        
        # Get scenes if available
        scenes = None
        if hasattr(self, 'scene_detector') and self.scene_detector:
            try:
                scenes = self.scene_detector.get_all_scenes()
            except:
                pass
        
        # Run matching
        print(f"  Running two-stage matching...")
        matches = match_all_segments(
            voiceover_segments=vo_segments,
            video_segments=video_segments,
            voiceover_embeddings=vo_embeddings,
            video_embeddings=self.embeddings,
            scenes=scenes,
            config=self.config,
            cache=self.cache,
            embedding_index=self.embedding_index
        )
        
        # Count stats
        matched = sum(1 for m in matches if hasattr(m, 'primary_match') and m.primary_match.confidence > 0.5)
        print(f"  ✓ Matched {matched}/{len(vo_segments)} segments (>50% confidence)")
        
        # Store matches for output stage
        self.matches = matches
        
        return {
            'matches': len(matches),
            'high_confidence': matched
        }
    
    def stage_output(self) -> dict:
        """Stage 5: Generate output files"""
        self._print_stage(5, "GENERATE OUTPUT")
        
        try:
            from src.otio_builder import (
                create_timeline, 
                save_timeline, 
                save_timeline_as_edl, 
                save_timeline_as_resolve_xml,
                generate_match_report
            )
        except ImportError as e:
            logger.warning(f"Could not import modules: {e}")
            return {'files': []}
        
        # Check if we have matches
        if not hasattr(self, 'matches') or not self.matches:
            print("  ⚠ No matches to export (run stage 4 first)")
            return {'files': []}
        
        output_dir = Path(self.config.otio_output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        files = []
        
        # Build OTIO timeline from matches
        voiceover_path = getattr(self.config, 'voiceover_path', None)
        timeline = create_timeline(self.matches, self.config, voiceover_path)
        
        # Save OTIO
        otio_path = output_dir / f"timeline_{timestamp}.otio"
        save_timeline(timeline, str(otio_path))
        print(f"  ✓ OTIO: {otio_path}")
        files.append(str(otio_path))
        
        # Export EDL
        edl_path = output_dir / f"timeline_{timestamp}.edl"
        save_timeline_as_edl(self.matches, str(edl_path))
        print(f"  ✓ EDL: {edl_path}")
        files.append(str(edl_path))
        
        # Export DaVinci XML
        xml_path = output_dir / f"timeline_{timestamp}.xml"
        save_timeline_as_resolve_xml(self.matches, str(xml_path), voiceover_path=voiceover_path)
        print(f"  ✓ DaVinci XML: {xml_path}")
        files.append(str(xml_path))
        
        # Generate match report
        report_path = output_dir / f"match_report_{timestamp}.md"
        generate_match_report(self.matches, str(report_path), self.config)
        print(f"  ✓ Match Report: {report_path}")
        files.append(str(report_path))
        
        return {'files': files}
    
    def run(
        self,
        voiceover_path: str,
        max_keywords: int = 50,
        resume: bool = False,
        skip_download: bool = False,
        skip_transcribe: bool = False,
        skip_scenes: bool = False,
        skip_match: bool = False,
        download_only: bool = False,
        match_only: bool = False
    ):
        """Run the full pipeline"""
        self._print_banner()
        
        # Stage 1: Analyze voiceover
        segments = self.stage_analyze_voiceover(voiceover_path)
        
        # Stage 1b: Extract keywords (skip if match-only since we already have videos)
        if not match_only:
            keywords = self.stage_extract_keywords(segments, max_keywords)
        else:
            keywords = []
            print("\n  Skipping keyword extraction (--match-only)")
        
        if match_only:
            skip_download = True
        
        # Pre-run summary and confirmation
        if not skip_download:
            if not self.stage_pre_run_summary(keywords, segments):
                print("\nAborted by user.")
                return
        
        # Stage 2: Download
        if not skip_download:
            self.stage_download(keywords, resume=resume)
            
            # Stage 2b: Deduplicate
            self.stage_deduplicate()
            
            if download_only:
                print("\n" + "=" * 70)
                print("  Download complete. Run with --match-only to continue.")
                print("=" * 70)
                return
        
        # Stage 3: Transcribe & Index
        if not skip_transcribe:
            self.stage_transcribe_index()
        
        # Stage 3b: Scene Detection
        if not skip_transcribe and not skip_scenes and self.config.scene_detection.enabled:
            self.stage_scene_detection()
        
        # Stage 4: Match
        if not skip_match:
            self.stage_match()
        
        # Stage 5: Output
        self.stage_output()
        
        # Stage 6: Clip Grading (optional)
        if not skip_match and self.matches:
            grading_config = getattr(self.config, 'clip_grading', None)
            if (grading_config and 
                getattr(grading_config, 'enabled', False) and
                getattr(grading_config, 'prompt_after_match', True)):
                
                try:
                    from src.interactive import grade_clips_after_match
                    
                    # Determine project dir
                    project_dir = None
                    if hasattr(self.config, 'cache_dir'):
                        project_dir = Path(self.config.cache_dir).parent
                    
                    print("\n" + "─" * 70)
                    print("  CLIP GRADING (Optional)")
                    print("─" * 70)
                    
                    allow_skip = getattr(grading_config, 'allow_skip', True)
                    grader = grade_clips_after_match(
                        self.matches,
                        project_dir=project_dir,
                        allow_skip=allow_skip
                    )
                    
                    # Merge into global cache if enabled
                    cross_cache = getattr(self.config, 'cross_project_cache', None)
                    if (cross_cache and 
                        getattr(cross_cache, 'enabled', False) and
                        getattr(cross_cache, 'share_clip_grades', True)):
                        try:
                            from src.interactive import GlobalCache
                            global_cache = GlobalCache(
                                global_cache_dir=getattr(cross_cache, 'global_cache_dir', None),
                                install_dir=str(Path(__file__).parent)
                            )
                            global_cache.merge_project_grades(str(grader.grades_file))
                        except Exception as e:
                            logger.warning(f"Could not sync grades to global cache: {e}")
                            
                except ImportError:
                    pass
                except Exception as e:
                    logger.warning(f"Clip grading failed: {e}")
        
        # Finalize logging
        if self.run_logger:
            try:
                # Log summary stats
                self.run_logger.run_log.voiceover_segments = len(self.voiceover_segments)
                self.run_logger.run_log.video_files = len(self.transcripts)
                self.run_logger.run_log.video_segments = len(self.embeddings)
                
                # Log match decisions
                for i, match in enumerate(self.matches):
                    if hasattr(match, 'primary_match'):
                        pm = match.primary_match
                        self.run_logger.log_match_decision(
                            segment_index=i,
                            voiceover_text=pm.voiceover_segment.text[:100] if pm.voiceover_segment else "",
                            selected_clip=Path(pm.video_segment.source_file).stem if pm.video_segment else "",
                            selected_source=pm.video_segment.source_file if pm.video_segment else "",
                            confidence=pm.confidence,
                            reasoning=pm.reasoning,
                            embedding_similarity=pm.embedding_similarity,
                            is_hybrid_match=pm.is_visual_match
                        )
                
                # Save the log
                self.run_logger.finalize()
                print(f"  Log saved: {self.run_logger.log_file}")
            except Exception as e:
                print(f"  ⚠ Could not save log: {e}")
        
        # Final summary
        print("\n" + "=" * 70)
        print("  ✓ PIPELINE COMPLETE")
        print("=" * 70)
        print(f"  Output: {self.config.otio_output_dir}")
        print(f"  Videos: {self.config.downloaded_videos_dir}")
        if self.downloader:
            print(f"  Sources: {self.downloader.sources_file}")
        if self.scene_detector:
            stats = self.scene_detector.get_stats()
            print(f"  Scenes: {stats['total_scenes']} across {stats['videos_processed']} videos")
        if self.run_logger:
            print(f"  Log: {self.run_logger.log_file}")
        print("=" * 70)


def main():
    global PROJECT_DIR
    
    parser = argparse.ArgumentParser(
        description='Voiceover-Matcher: End-to-End Documentary Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --voiceover script.srt
  python main.py --voiceover script.srt --keywords 30
  python main.py --project "E:\\Projects\\MyDoc" -v voiceover/script.srt
  python main.py --resume
  python main.py --match-only
  python main.py --download-only

Project Mode:
  Use --project to specify a project directory. All output files will be
  saved relative to that directory, while using global config and API keys
  from the install location.
        """
    )
    
    # Project mode
    parser.add_argument(
        '--project', '-p',
        help='Project directory (all paths become project-relative)',
        default=None
    )
    
    # Input
    parser.add_argument(
        '--voiceover', '-v',
        help='Path to voiceover SRT/audio file',
        default=None
    )
    parser.add_argument(
        '--config', '-c',
        help='Path to config.yaml (default: install_dir/config.yaml)',
        default=None
    )
    
    # Keywords
    parser.add_argument(
        '--keywords', '-k',
        type=int,
        help='Maximum keywords to extract (default: prompt user)',
        default=None
    )
    
    # Stage control
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume interrupted download'
    )
    parser.add_argument(
        '--skip-download',
        action='store_true',
        help='Skip download stage (use existing footage)'
    )
    parser.add_argument(
        '--skip-transcribe',
        action='store_true',
        help='Skip transcription stage (use cached)'
    )
    parser.add_argument(
        '--skip-match',
        action='store_true',
        help='Skip matching stage'
    )
    parser.add_argument(
        '--skip-scenes',
        action='store_true',
        help='Skip scene detection stage'
    )
    parser.add_argument(
        '--download-only',
        action='store_true',
        help='Only download footage, then stop'
    )
    parser.add_argument(
        '--match-only',
        action='store_true',
        help='Skip download, match existing footage'
    )
    
    # Confirmation
    parser.add_argument(
        '--yes', '-y',
        action='store_true',
        help='Skip confirmation prompts'
    )
    
    args = parser.parse_args()
    
    # Determine project directory
    if args.project:
        PROJECT_DIR = strip_extended_path_prefix(Path(args.project).resolve())
        # Strip trailing backslash/slash that Windows might add
        if str(PROJECT_DIR).endswith(('\\', '/')):
            PROJECT_DIR = PROJECT_DIR.parent / PROJECT_DIR.name
        
        print(f"\n📁 Project Mode")
        print(f"   Project: {PROJECT_DIR}")
        print(f"   Install: {INSTALL_DIR}")
        
        # Ensure project directory exists
        if not PROJECT_DIR.exists():
            print(f"\n⚠ Project directory does not exist: {PROJECT_DIR}")
            print(f"   Run: python setup_project.py \"{PROJECT_DIR}\"")
            sys.exit(1)
    else:
        PROJECT_DIR = Path.cwd()
    
    # Load environment (.env files)
    load_environment(PROJECT_DIR if args.project else None)
    
    # Load config (project mode vs standard mode)
    if args.project:
        # Project mode: load global config, merge project overrides, make paths relative
        global_config_path = Path(args.config) if args.config else INSTALL_DIR / 'config.yaml'
        config = load_project_config(PROJECT_DIR, global_config_path)
    else:
        # Standard mode: load config from specified path or default
        config_path = args.config if args.config else 'config.yaml'
        config = load_config(config_path)
    
    # Override with command line
    if args.yes:
        config.pipeline.confirm_before_download = False
    
    # Import interactive module
    try:
        from src.interactive import (
            select_voiceover_file, 
            review_keywords, 
            prompt_face_preference,
            grade_clips_after_match
        )
        has_interactive = True
    except ImportError:
        has_interactive = False
    
    # Get voiceover path
    voiceover_path = args.voiceover or getattr(config, 'voiceover_path', None)
    
    # In project mode, resolve voiceover path relative to project
    if args.project and voiceover_path and not Path(voiceover_path).is_absolute():
        voiceover_path = str(PROJECT_DIR / voiceover_path)
    
    if not voiceover_path or not Path(voiceover_path).exists():
        if has_interactive and not args.yes:
            # Interactive file selection
            voiceover_path = select_voiceover_file(
                project_dir=PROJECT_DIR if args.project else None
            )
            
            if not voiceover_path:
                print("No voiceover file selected. Exiting.")
                sys.exit(0)
        else:
            # Fallback to text prompt
            print("\nVoiceover file not specified.")
            if args.project:
                print(f"  Looking in: {PROJECT_DIR / 'voiceover'}/")
            voiceover_path = input("Enter path to voiceover file: ").strip()
            
            # Resolve relative to project if in project mode
            if args.project and voiceover_path and not Path(voiceover_path).is_absolute():
                voiceover_path = str(PROJECT_DIR / voiceover_path)
        
        if not voiceover_path or not Path(voiceover_path).exists():
            print(f"Error: Voiceover file not found: {voiceover_path}")
            sys.exit(1)
    
    # Face detection preference (if enabled and not --yes)
    face_preference = getattr(config, 'face_detection', None)
    if (face_preference and 
        getattr(face_preference, 'enabled', False) and 
        getattr(face_preference, 'prompt_per_run', False) and
        has_interactive and not args.yes):
        
        face_pref = prompt_face_preference()
        # Store in config for pipeline to use
        config.face_detection.current_preference = face_pref
    
    # Get max keywords
    max_keywords = args.keywords
    if max_keywords is None and not args.match_only:
        # Prompt user (skip if --match-only since keywords aren't used)
        print(f"\nKeyword extraction (default: {config.pipeline.max_keywords})")
        user_input = input(f"How many keywords to extract? [{config.pipeline.max_keywords}]: ").strip()
        if user_input:
            try:
                max_keywords = int(user_input)
            except ValueError:
                max_keywords = config.pipeline.max_keywords
        else:
            max_keywords = config.pipeline.max_keywords
    elif max_keywords is None:
        max_keywords = config.pipeline.max_keywords
    
    # Run pipeline
    pipeline = Pipeline(config)
    pipeline.run(
        voiceover_path=voiceover_path,
        max_keywords=max_keywords,
        resume=args.resume,
        skip_download=args.skip_download,
        skip_transcribe=args.skip_transcribe,
        skip_scenes=args.skip_scenes,
        skip_match=args.skip_match,
        download_only=args.download_only,
        match_only=args.match_only
    )


if __name__ == '__main__':
    main()
