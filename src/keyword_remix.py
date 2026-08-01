"""
Zero-Download Keyword Remix Module

Enables keyword-based video remixing on already downloaded content 
without requiring additional downloads. Triggered after initial batch 
downloads complete, before transcription.

Features:
- Score videos against keywords using filename/metadata
- Filter low-relevance videos
- Prioritize high-relevance content
- Interactive curation mode
- Comprehensive logging with structured events

Author: Voiceover-Matcher Team
Version: 1.0.0
"""

import os
import json
import re
import logging
import time
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class VideoScore:
    """Relevance score for a downloaded video"""
    file_path: str
    filename: str
    keyword_matches: List[str]
    match_score: float  # 0.0 - 1.0
    file_size_mb: float
    duration_estimate: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass 
class RemixResult:
    """Result of remix operation"""
    total_files: int
    included_files: int
    excluded_files: int
    included_videos: List[VideoScore]
    excluded_videos: List[VideoScore]
    processing_time_seconds: float
    keywords_used: List[str]
    avg_match_score: float
    
    def to_dict(self) -> dict:
        result = asdict(self)
        result['included_videos'] = [v.to_dict() for v in self.included_videos]
        result['excluded_videos'] = [v.to_dict() for v in self.excluded_videos]
        return result


@dataclass
class RemixConfig:
    """Configuration for keyword remix functionality"""
    enabled: bool = True
    trigger_after_download: bool = True
    
    # Scoring thresholds
    min_relevance_score: float = 0.1  # Minimum score to include
    high_relevance_threshold: float = 0.5  # Score for "high relevance" label
    
    # Processing limits
    max_files_to_process: int = 500
    max_files_to_include: int = 100  # Maximum videos to pass to transcription
    
    # Matching settings
    fuzzy_match: bool = True  # Allow partial keyword matches
    case_sensitive: bool = False
    
    # Interactive mode
    interactive_curation: bool = True  # Ask user to confirm remix
    show_excluded: bool = True  # Show which files were excluded
    
    # Auto-accept filter (set upfront to skip mid-pipeline prompt)
    # Options: "filtered" (use filtered), "all" (use all videos), "prompt" (ask)
    auto_accept_filter: str = "prompt"
    
    # Logging
    log_level: str = "INFO"
    log_file_processing: bool = True  # Log each file as it's processed
    
    # Performance
    parallel_scoring: bool = True
    max_workers: int = 4


# =============================================================================
# KEYWORD REMIX PROCESSOR
# =============================================================================

class KeywordRemixProcessor:
    """
    Zero-download keyword remix processor.
    
    Analyzes already-downloaded videos against keywords to score relevance
    and filter the video pool before transcription.
    """
    
    def __init__(self, config: RemixConfig, keywords: List[str]):
        """
        Initialize remix processor.
        
        Args:
            config: RemixConfig with processing settings
            keywords: List of keywords to match against
        """
        self.config = config
        self.keywords = [k.lower() if not config.case_sensitive else k for k in keywords]
        self.keyword_patterns = self._compile_keyword_patterns()
        
        # Metrics
        self.metrics = {
            'files_scanned': 0,
            'files_scored': 0,
            'scoring_errors': 0,
            'start_time': None,
            'end_time': None
        }
        
        logger.info(json.dumps({
            'event': 'remix_processor_init',
            'keywords_count': len(keywords),
            'config': {
                'min_relevance': config.min_relevance_score,
                'max_files': config.max_files_to_include,
                'fuzzy_match': config.fuzzy_match
            }
        }))
    
    def _compile_keyword_patterns(self) -> List[re.Pattern]:
        """Compile regex patterns for keyword matching"""
        patterns = []
        for kw in self.keywords:
            # Escape special regex characters
            escaped = re.escape(kw)
            if self.config.fuzzy_match:
                # Allow word boundaries and partial matches
                # Split multi-word keywords
                words = escaped.split(r'\ ')
                if len(words) > 1:
                    # For multi-word: match any word
                    pattern = '|'.join(words)
                else:
                    pattern = escaped
            else:
                # Exact match with word boundaries
                pattern = r'\b' + escaped + r'\b'
            
            flags = 0 if self.config.case_sensitive else re.IGNORECASE
            try:
                patterns.append(re.compile(pattern, flags))
            except re.error as e:
                logger.warning(f"Invalid pattern for keyword '{kw}': {e}")
        
        return patterns
    
    def _score_filename(self, filename: str) -> Tuple[float, List[str]]:
        """
        Score a filename against keywords.
        
        Returns:
            Tuple of (score, list of matched keywords)
        """
        # Normalize filename for matching
        name = filename.lower() if not self.config.case_sensitive else filename
        # Remove extension and clean up
        name_clean = Path(filename).stem
        name_clean = re.sub(r'[-_]', ' ', name_clean)
        
        matches = []
        for i, pattern in enumerate(self.keyword_patterns):
            if pattern.search(name_clean):
                matches.append(self.keywords[i])
        
        # Calculate score: ratio of matched keywords
        if not self.keywords:
            return 0.0, []
        
        score = len(matches) / len(self.keywords)
        
        # Boost score for exact matches
        for kw in self.keywords:
            if kw.lower() in name_clean.lower():
                score = min(1.0, score + 0.1)
        
        return min(1.0, score), matches
    
    def _score_metadata(self, info_json_path: Path) -> Tuple[float, List[str], Dict]:
        """
        Score based on video metadata from info.json
        
        Returns:
            Tuple of (score, matched keywords, metadata dict)
        """
        if not info_json_path.exists():
            return 0.0, [], {}
        
        try:
            with open(info_json_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.debug(f"Could not read metadata: {info_json_path}: {e}")
            return 0.0, [], {}
        
        # Fields to search for keywords
        searchable_fields = ['title', 'description', 'tags', 'categories']
        text_to_search = []
        
        for field in searchable_fields:
            value = metadata.get(field, '')
            if isinstance(value, list):
                text_to_search.extend(str(v) for v in value)
            elif value:
                text_to_search.append(str(value))
        
        combined_text = ' '.join(text_to_search)
        if not self.config.case_sensitive:
            combined_text = combined_text.lower()
        
        matches = []
        for i, pattern in enumerate(self.keyword_patterns):
            if pattern.search(combined_text):
                matches.append(self.keywords[i])
        
        score = len(matches) / len(self.keywords) if self.keywords else 0.0
        
        # Extract useful metadata
        useful_meta = {
            'title': metadata.get('title', ''),
            'duration': metadata.get('duration'),
            'view_count': metadata.get('view_count'),
            'upload_date': metadata.get('upload_date')
        }
        
        return min(1.0, score), matches, useful_meta
    
    def score_video(self, video_path: Path) -> VideoScore:
        """
        Score a single video file against keywords.
        
        Args:
            video_path: Path to video file
            
        Returns:
            VideoScore with relevance information
        """
        filename = video_path.name
        
        # Score from filename
        filename_score, filename_matches = self._score_filename(filename)
        
        # Score from metadata
        info_json = video_path.with_suffix('.info.json')
        metadata_score, metadata_matches, metadata = self._score_metadata(info_json)
        
        # Combine scores (weighted average)
        combined_score = (filename_score * 0.3) + (metadata_score * 0.7)
        if not info_json.exists():
            combined_score = filename_score  # Fall back to filename only
        
        # Combine matches (deduplicate)
        all_matches = list(set(filename_matches + metadata_matches))
        
        # Get file size
        try:
            file_size_mb = video_path.stat().st_size / (1024 * 1024)
        except OSError:
            file_size_mb = 0.0
        
        # Get duration from metadata
        duration = metadata.get('duration') if metadata else None
        
        return VideoScore(
            file_path=str(video_path),
            filename=filename,
            keyword_matches=all_matches,
            match_score=combined_score,
            file_size_mb=file_size_mb,
            duration_estimate=duration,
            metadata=metadata
        )
    
    def scan_directory(self, video_dir: Path, include_audio: bool = False) -> List[Path]:
        """
        Scan directory for video/audio files.

        Args:
            video_dir: Directory to scan
            include_audio: Whether to include audio files (for audio-first mode)

        Returns:
            List of video/audio file paths
        """
        video_extensions = {'.mp4', '.mkv', '.webm', '.avi', '.mov', '.m4v'}
        audio_extensions = {'.mp3', '.m4a', '.opus', '.ogg', '.wav', '.flac'}

        allowed_extensions = video_extensions
        if include_audio:
            allowed_extensions = video_extensions | audio_extensions

        video_files = []

        for root, dirs, files in os.walk(video_dir):
            for file in files:
                if Path(file).suffix.lower() in allowed_extensions:
                    video_files.append(Path(root) / file)

        self.metrics['files_scanned'] = len(video_files)

        logger.info(json.dumps({
            'event': 'directory_scan_complete',
            'directory': str(video_dir),
            'video_files_found': len(video_files),
            'include_audio': include_audio
        }))

        return video_files[:self.config.max_files_to_process]

    def process_file_list(
        self,
        file_paths: List[Path],
        show_progress: bool = True
    ) -> RemixResult:
        """
        Process a list of files and score against keywords.

        Args:
            file_paths: List of file paths to score
            show_progress: Whether to show progress output

        Returns:
            RemixResult with scoring results
        """
        self.metrics['start_time'] = time.time()
        self.metrics['files_scanned'] = len(file_paths)

        logger.info(json.dumps({
            'event': 'remix_processing_start',
            'file_count': len(file_paths),
            'keywords': self.keywords[:10],
            'keywords_total': len(self.keywords)
        }))

        if not file_paths:
            return RemixResult(
                total_files=0,
                included_files=0,
                excluded_files=0,
                included_videos=[],
                excluded_videos=[],
                processing_time_seconds=0,
                keywords_used=self.keywords,
                avg_match_score=0.0
            )

        if show_progress:
            logger.info(f"Scoring {len(file_paths)} files against {len(self.keywords)} keywords")

        # Score all files
        scored_videos: List[VideoScore] = []

        if self.config.parallel_scoring and len(file_paths) > 10:
            with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                futures = {executor.submit(self.score_video, Path(fp)): fp for fp in file_paths}

                for i, future in enumerate(as_completed(futures)):
                    try:
                        score = future.result()
                        scored_videos.append(score)
                        self.metrics['files_scored'] += 1

                        if show_progress and (i + 1) % 20 == 0:
                            logger.info(f"Scored {i + 1}/{len(file_paths)} files")
                    except Exception as e:
                        self.metrics['scoring_errors'] += 1
                        logger.error(f"Scoring error: {e}")
        else:
            for i, fp in enumerate(file_paths):
                try:
                    score = self.score_video(Path(fp))
                    scored_videos.append(score)
                    self.metrics['files_scored'] += 1

                    if show_progress and (i + 1) % 20 == 0:
                        logger.info(f"Scored {i + 1}/{len(file_paths)} files")
                except Exception as e:
                    self.metrics['scoring_errors'] += 1
                    logger.error(f"Scoring error for {fp}: {e}")

        # Sort by score (highest first)
        scored_videos.sort(key=lambda x: x.match_score, reverse=True)

        # Apply filtering
        included = []
        excluded = []

        for video in scored_videos:
            if video.match_score >= self.config.min_relevance_score:
                if len(included) < self.config.max_files_to_include:
                    included.append(video)
                else:
                    excluded.append(video)
            else:
                excluded.append(video)

        # Calculate metrics
        self.metrics['end_time'] = time.time()
        processing_time = self.metrics['end_time'] - self.metrics['start_time']
        avg_score = sum(v.match_score for v in included) / len(included) if included else 0.0

        return RemixResult(
            total_files=len(file_paths),
            included_files=len(included),
            excluded_files=len(excluded),
            included_videos=included,
            excluded_videos=excluded,
            processing_time_seconds=processing_time,
            keywords_used=self.keywords,
            avg_match_score=avg_score
        )
    
    def process_videos(
        self,
        video_dir: Path,
        show_progress: bool = True
    ) -> RemixResult:
        """
        Process all videos in directory and score against keywords.
        
        Args:
            video_dir: Directory containing downloaded videos
            show_progress: Whether to show progress output
            
        Returns:
            RemixResult with scoring results
        """
        self.metrics['start_time'] = time.time()
        
        logger.info(json.dumps({
            'event': 'remix_processing_start',
            'directory': str(video_dir),
            'keywords': self.keywords[:10],  # Log first 10
            'keywords_total': len(self.keywords)
        }))
        
        # Scan for videos
        video_files = self.scan_directory(video_dir)
        
        if not video_files:
            logger.warning(json.dumps({
                'event': 'no_videos_found',
                'directory': str(video_dir)
            }))
            return RemixResult(
                total_files=0,
                included_files=0,
                excluded_files=0,
                included_videos=[],
                excluded_videos=[],
                processing_time_seconds=0,
                keywords_used=self.keywords,
                avg_match_score=0.0
            )
        
        if show_progress:
            logger.info(f"Scoring {len(video_files)} videos against {len(self.keywords)} keywords")
        
        # Score all videos
        scored_videos: List[VideoScore] = []
        
        if self.config.parallel_scoring and len(video_files) > 10:
            # Parallel scoring
            with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                futures = {executor.submit(self.score_video, vf): vf for vf in video_files}
                
                for i, future in enumerate(as_completed(futures)):
                    try:
                        score = future.result()
                        scored_videos.append(score)
                        self.metrics['files_scored'] += 1
                        
                        if self.config.log_file_processing:
                            logger.debug(json.dumps({
                                'event': 'file_scored',
                                'file': score.filename[:50],
                                'score': round(score.match_score, 3),
                                'matches': score.keyword_matches
                            }))
                        
                        if show_progress and (i + 1) % 20 == 0:
                            logger.info(f"Scored {i + 1}/{len(video_files)} videos")
                            
                    except Exception as e:
                        self.metrics['scoring_errors'] += 1
                        logger.error(json.dumps({
                            'event': 'scoring_error',
                            'file': str(futures[future]),
                            'error': str(e)
                        }))
        else:
            # Sequential scoring
            for i, vf in enumerate(video_files):
                try:
                    score = self.score_video(vf)
                    scored_videos.append(score)
                    self.metrics['files_scored'] += 1
                    
                    if show_progress and (i + 1) % 20 == 0:
                        logger.info(f"Scored {i + 1}/{len(video_files)} videos")
                        
                except Exception as e:
                    self.metrics['scoring_errors'] += 1
                    logger.error(json.dumps({
                        'event': 'scoring_error',
                        'file': str(vf),
                        'error': str(e)
                    }))
        
        # Sort by score (highest first)
        scored_videos.sort(key=lambda x: x.match_score, reverse=True)
        
        # Apply filtering
        included = []
        excluded = []
        
        for video in scored_videos:
            if video.match_score >= self.config.min_relevance_score:
                if len(included) < self.config.max_files_to_include:
                    included.append(video)
                else:
                    excluded.append(video)
            else:
                excluded.append(video)
        
        # Calculate metrics
        self.metrics['end_time'] = time.time()
        processing_time = self.metrics['end_time'] - self.metrics['start_time']
        avg_score = sum(v.match_score for v in included) / len(included) if included else 0.0
        
        result = RemixResult(
            total_files=len(video_files),
            included_files=len(included),
            excluded_files=len(excluded),
            included_videos=included,
            excluded_videos=excluded,
            processing_time_seconds=processing_time,
            keywords_used=self.keywords,
            avg_match_score=avg_score
        )
        
        logger.info(json.dumps({
            'event': 'remix_processing_complete',
            'total_files': result.total_files,
            'included': result.included_files,
            'excluded': result.excluded_files,
            'avg_score': round(result.avg_match_score, 3),
            'processing_time_sec': round(processing_time, 2),
            'errors': self.metrics['scoring_errors']
        }))
        
        return result


# =============================================================================
# MAIN REMIX FUNCTION
# =============================================================================

def remix_downloaded_videos(
    video_dir: Path,
    keywords: List[str],
    config: Optional[RemixConfig] = None,
    interactive: bool = True,
    show_progress: bool = True
) -> Tuple[List[str], RemixResult]:
    """
    Main entry point for zero-download keyword remix.
    
    Args:
        video_dir: Directory containing downloaded videos
        keywords: Keywords to match against
        config: RemixConfig (uses defaults if None)
        interactive: Whether to prompt user for confirmation
        show_progress: Whether to show progress output
        
    Returns:
        Tuple of (list of included video paths, RemixResult)
    """
    if config is None:
        config = RemixConfig()
    
    if not config.enabled:
        logger.info("Remix disabled in config, returning all videos")
        # Return all videos without filtering
        video_files = []
        for ext in ['.mp4', '.mkv', '.webm', '.avi', '.mov']:
            video_files.extend(video_dir.rglob(f'*{ext}'))
        return [str(v) for v in video_files], None
    
    # Initialize processor
    processor = KeywordRemixProcessor(config, keywords)
    
    # Process videos
    result = processor.process_videos(video_dir, show_progress)
    
    if show_progress:
        print(f"\n  Remix Results:")
        print(f"    • Total videos scanned: {result.total_files}")
        print(f"    • Included (score ≥{config.min_relevance_score}): {result.included_files}")
        print(f"    • Excluded: {result.excluded_files}")
        print(f"    • Average relevance score: {result.avg_match_score:.1%}")
        print(f"    • Processing time: {result.processing_time_seconds:.1f}s")
        
        if result.included_videos:
            print(f"\n  Top 5 matches:")
            for i, video in enumerate(result.included_videos[:5]):
                relevance = "HIGH" if video.match_score >= config.high_relevance_threshold else "MED"
                print(f"    {i+1}. [{relevance}] {video.filename[:50]}... (score: {video.match_score:.1%})")
                if video.keyword_matches:
                    print(f"       Keywords: {', '.join(video.keyword_matches[:3])}")
        
        if config.show_excluded and result.excluded_videos:
            print(f"\n  Excluded ({len(result.excluded_videos)} videos with score <{config.min_relevance_score}):")
            for video in result.excluded_videos[:3]:
                print(f"    • {video.filename[:50]}... (score: {video.match_score:.1%})")
            if len(result.excluded_videos) > 3:
                print(f"    ... and {len(result.excluded_videos) - 3} more")
    
    # Interactive confirmation
    if interactive and config.interactive_curation:
        # Check if auto_accept_filter is set (from upfront config)
        auto_accept = getattr(config, 'auto_accept_filter', 'prompt')
        
        if auto_accept == 'filtered':
            # Use filtered videos without prompting
            logger.info(f"Auto-accepting filtered videos ({result.included_files})")
            print(f"\n  ✓ Using {result.included_files} filtered videos (auto-accept)")
            return [v.file_path for v in result.included_videos], result
        elif auto_accept == 'all':
            # Use all videos without prompting
            logger.info("Auto-accepting all videos (skip filtering)")
            print(f"\n  ✓ Using ALL {result.total_files} videos (auto-accept)")
            all_videos = result.included_videos + result.excluded_videos
            return [v.file_path for v in all_videos], result
        else:
            # Prompt user (auto_accept == 'prompt' or any other value)
            print(f"\n  Proceed with {result.included_files} videos?")
            print(f"    [Y] Yes, use these {result.included_files} videos")
            print(f"    [A] Use ALL videos (skip filtering)")
            print(f"    [N] Cancel")
            
            choice = input("  Select [Y/A/N]: ").strip().upper()
            
            if choice == 'A':
                logger.info("User selected all videos, bypassing filter")
                all_videos = result.included_videos + result.excluded_videos
                return [v.file_path for v in all_videos], result
            elif choice != 'Y':
                logger.info("User cancelled remix")
                return [], result
    
    # Return included video paths
    return [v.file_path for v in result.included_videos], result


def remix_audio_files(
    audio_files: List[str],
    keywords: List[str],
    config: Optional[RemixConfig] = None,
    interactive: bool = True,
    show_progress: bool = True
) -> Tuple[List[str], RemixResult]:
    """
    Remix audio files for audio-first mode.

    Args:
        audio_files: List of audio file paths to score
        keywords: Keywords to match against
        config: RemixConfig (uses defaults if None)
        interactive: Whether to prompt user for confirmation
        show_progress: Whether to show progress output

    Returns:
        Tuple of (list of included audio paths, RemixResult)
    """
    if config is None:
        config = RemixConfig()

    if not config.enabled:
        logger.info("Remix disabled in config, returning all audio files")
        return audio_files, None

    if not audio_files:
        logger.warning("No audio files provided for remix")
        return [], None

    # Initialize processor
    processor = KeywordRemixProcessor(config, keywords)

    # Process audio files directly
    result = processor.process_file_list([Path(f) for f in audio_files], show_progress)

    if show_progress:
        print(f"\n  Remix Results (Audio-First Mode):")
        print(f"    • Total audio files scored: {result.total_files}")
        print(f"    • Included (score ≥{config.min_relevance_score}): {result.included_files}")
        print(f"    • Excluded: {result.excluded_files}")
        print(f"    • Average relevance score: {result.avg_match_score:.1%}")
        print(f"    • Processing time: {result.processing_time_seconds:.1f}s")

        if result.included_videos:
            print(f"\n  Top 5 matches:")
            for i, video in enumerate(result.included_videos[:5]):
                relevance = "HIGH" if video.match_score >= config.high_relevance_threshold else "MED"
                print(f"    {i+1}. [{relevance}] {video.filename[:50]}... (score: {video.match_score:.1%})")
                if video.keyword_matches:
                    print(f"       Keywords: {', '.join(video.keyword_matches[:3])}")

        if config.show_excluded and result.excluded_videos:
            print(f"\n  Excluded ({len(result.excluded_videos)} files with score <{config.min_relevance_score}):")
            for video in result.excluded_videos[:3]:
                print(f"    • {video.filename[:50]}... (score: {video.match_score:.1%})")
            if len(result.excluded_videos) > 3:
                print(f"    ... and {len(result.excluded_videos) - 3} more")

    # Interactive confirmation
    if interactive and config.interactive_curation:
        auto_accept = getattr(config, 'auto_accept_filter', 'prompt')

        if auto_accept == 'filtered':
            logger.info(f"Auto-accepting filtered audio files ({result.included_files})")
            print(f"\n  ✓ Using {result.included_files} filtered audio files (auto-accept)")
            return [v.file_path for v in result.included_videos], result
        elif auto_accept == 'all':
            logger.info("Auto-accepting all audio files (skip filtering)")
            print(f"\n  ✓ Using ALL {result.total_files} audio files (auto-accept)")
            all_files = result.included_videos + result.excluded_videos
            return [v.file_path for v in all_files], result
        else:
            print(f"\n  Proceed with {result.included_files} audio files?")
            print(f"    [Y] Yes, use these {result.included_files} files")
            print(f"    [A] Use ALL files (skip filtering)")
            print(f"    [N] Cancel")

            choice = input("  Select [Y/A/N]: ").strip().upper()

            if choice == 'A':
                logger.info("User selected all audio files, bypassing filter")
                all_files = result.included_videos + result.excluded_videos
                return [v.file_path for v in all_files], result
            elif choice != 'Y':
                logger.info("User cancelled remix")
                return [], result

    return [v.file_path for v in result.included_videos], result


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_remix_summary(result: RemixResult) -> str:
    """Generate human-readable summary of remix results"""
    if result is None:
        return "Remix not performed"
    
    lines = [
        f"Remix Summary:",
        f"  Videos processed: {result.total_files}",
        f"  Included: {result.included_files}",
        f"  Excluded: {result.excluded_files}",
        f"  Avg score: {result.avg_match_score:.1%}",
        f"  Time: {result.processing_time_seconds:.1f}s"
    ]
    return '\n'.join(lines)


def save_remix_report(result: RemixResult, output_path: Path):
    """Save remix results to JSON file"""
    with open(output_path, 'w') as f:
        json.dump(result.to_dict(), f, indent=2)
    logger.info(f"Remix report saved to {output_path}")


# =============================================================================
# LLM-BASED KEYWORD REMIXER (Zero-Download Recovery)
# =============================================================================

@dataclass
class KeywordRemixResult:
    """Result of LLM keyword remix operation"""
    original_keyword: str
    remixed_keywords: List[str]
    reasoning: str
    success: bool
    provider: str
    attempt: int
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass 
class KeywordRemixBatchResult:
    """Result of batch keyword remix operation"""
    total_original: int
    total_remixed: int
    successful_remixes: int
    failed_remixes: int
    results: List[KeywordRemixResult]
    processing_time_seconds: float
    provider: str
    
    def to_dict(self) -> dict:
        data = asdict(self)
        data['results'] = [r.to_dict() for r in self.results]
        return data


class KeywordRemixer:
    """
    LLM-based keyword remixer for zero-download scenarios.
    
    When a keyword returns 0 downloads from YouTube, this class uses
    an LLM to generate alternative search terms that are more likely
    to find relevant video content.
    
    Strategies:
    1. Broaden specific terms → generic alternatives
    2. Add/remove qualifiers (location, date, type)
    3. Use synonyms and related concepts
    4. Split compound keywords
    5. Focus on visual content availability
    """
    
    REMIX_PROMPT = '''You are helping find stock footage on YouTube. 
The following search keyword returned ZERO results: "{keyword}"

Generate 3-5 alternative search keywords that are MORE LIKELY to find relevant video footage.

Guidelines:
1. Remove overly specific terms (dates, exact names, niche terminology)
2. Use broader, more visual search terms
3. Focus on what would actually appear in video footage
4. Keep the core concept but make it more searchable
5. Consider YouTube's content library (news clips, documentaries, b-roll)

Original keyword: "{keyword}"
Topic context: {topic_context}

Respond ONLY with a JSON array of alternative keywords, nothing else:
["keyword1", "keyword2", "keyword3"]'''

    BATCH_REMIX_PROMPT = '''You are helping find stock footage on YouTube.
The following search keywords returned ZERO results and need alternatives.

For EACH failed keyword, generate 2-3 alternative search terms that are MORE LIKELY to find relevant video footage.

Guidelines:
1. Remove overly specific terms (dates, exact names, niche terminology)  
2. Use broader, more visual search terms
3. Focus on what would actually appear in video footage
4. Keep the core concept but make it more searchable
5. Consider YouTube's content library (news clips, documentaries, b-roll)

Failed keywords:
{keywords_list}

Topic context: {topic_context}

Respond with a JSON object mapping each original keyword to its alternatives:
{{
  "original keyword 1": ["alt1", "alt2"],
  "original keyword 2": ["alt1", "alt2", "alt3"]
}}'''

    def __init__(
        self,
        config: Any = None,
        topic_context: str = "",
        cache_dir: Optional[str] = None
    ):
        """
        Initialize keyword remixer.
        
        Args:
            config: Config object with API keys
            topic_context: Context about the video topic
            cache_dir: Directory for caching remix results
        """
        self.config = config
        self.topic_context = topic_context or "general video content"
        self.cache_dir = Path(cache_dir) if cache_dir else Path(".cache/keyword_remix")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Get API keys
        self.gemini_api_key = None
        self.anthropic_api_key = None
        
        if config:
            self.gemini_api_key = getattr(config, 'gemini_api_key', None) or os.getenv('GEMINI_API_KEY')
            self.anthropic_api_key = getattr(config, 'anthropic_api_key', None) or os.getenv('ANTHROPIC_API_KEY')
        else:
            self.gemini_api_key = os.getenv('GEMINI_API_KEY')
            self.anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')
        
        # Stats
        self.stats = {
            'total_remixes': 0,
            'successful_remixes': 0,
            'cache_hits': 0,
            'api_calls': 0
        }
        
        logger.info(f"KeywordRemixer initialized (topic: {self.topic_context[:50]}...)")
    
    def _get_cache_key(self, keyword: str, attempt: int) -> str:
        """Generate cache key for remix result"""
        import hashlib
        content = f"{keyword}|{self.topic_context}|{attempt}"
        return hashlib.md5(content.encode()).hexdigest()[:16]
    
    def _get_cached_remix(self, keyword: str, attempt: int) -> Optional[List[str]]:
        """Get cached remix result"""
        cache_key = self._get_cache_key(keyword, attempt)
        cache_file = self.cache_dir / f"remix_{cache_key}.json"
        
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                    self.stats['cache_hits'] += 1
                    logger.debug(f"Cache hit for '{keyword}'")
                    return data.get('remixed_keywords', [])
            except:
                pass
        return None
    
    def _cache_remix(self, keyword: str, attempt: int, remixed: List[str], reasoning: str = ""):
        """Cache remix result"""
        cache_key = self._get_cache_key(keyword, attempt)
        cache_file = self.cache_dir / f"remix_{cache_key}.json"
        
        try:
            with open(cache_file, 'w') as f:
                json.dump({
                    'original': keyword,
                    'remixed_keywords': remixed,
                    'reasoning': reasoning,
                    'attempt': attempt,
                    'timestamp': datetime.now().isoformat()
                }, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to cache remix: {e}")
    
    def remix_keyword_gemini(self, keyword: str, attempt: int = 1) -> Tuple[List[str], str]:
        """Remix a single keyword using Gemini.

        Delegates to unified KeywordAlternativeGenerator.
        """
        if not self.gemini_api_key:
            raise ValueError("Gemini API key not available")

        from src.keyword_alternatives import KeywordAlternativeGenerator

        generator = KeywordAlternativeGenerator(self.config)
        remixed, reasoning = generator.generate_multiple_alternatives(
            keyword=keyword,
            topic_context=self.topic_context,
            provider="gemini",
            api_key=self.gemini_api_key
        )

        if remixed:
            self.stats['api_calls'] += 1

        return remixed, reasoning

    def remix_keyword_anthropic(self, keyword: str, attempt: int = 1) -> Tuple[List[str], str]:
        """Remix a single keyword using Claude.

        Delegates to unified KeywordAlternativeGenerator.
        """
        if not self.anthropic_api_key:
            raise ValueError("Anthropic API key not available")

        from src.keyword_alternatives import KeywordAlternativeGenerator

        # Get max_tokens from config
        max_tokens = 500
        if self.config:
            if hasattr(self.config, 'llm'):
                max_tokens = getattr(self.config.llm, 'max_tokens', max_tokens)

        generator = KeywordAlternativeGenerator(self.config)
        remixed, reasoning = generator.generate_multiple_alternatives(
            keyword=keyword,
            topic_context=self.topic_context,
            provider="anthropic",
            api_key=self.anthropic_api_key,
            max_tokens=max_tokens
        )

        if remixed:
            self.stats['api_calls'] += 1

        return remixed, reasoning
    
    def _fallback_remix(self, keyword: str) -> List[str]:
        """Rule-based fallback when LLM is unavailable.

        Delegates to unified KeywordAlternativeGenerator.
        """
        from src.keyword_alternatives import KeywordAlternativeGenerator

        generator = KeywordAlternativeGenerator(self.config)
        return generator.fallback_remix(keyword)
    
    def remix_keyword(self, keyword: str, attempt: int = 1) -> KeywordRemixResult:
        """
        Remix a single keyword using available LLM.
        
        Args:
            keyword: Original keyword that had 0 downloads
            attempt: Attempt number (for progressive broadening)
            
        Returns:
            KeywordRemixResult with alternative keywords
        """
        self.stats['total_remixes'] += 1
        
        # Check cache first
        cached = self._get_cached_remix(keyword, attempt)
        if cached:
            return KeywordRemixResult(
                original_keyword=keyword,
                remixed_keywords=cached,
                reasoning="Loaded from cache",
                success=True,
                provider="cache",
                attempt=attempt
            )
        
        # Try Gemini first
        remixed = []
        reasoning = ""
        provider = "none"
        
        if self.gemini_api_key:
            remixed, reasoning = self.remix_keyword_gemini(keyword, attempt)
            provider = "gemini"
        
        # Fallback to Claude
        if not remixed and self.anthropic_api_key:
            remixed, reasoning = self.remix_keyword_anthropic(keyword, attempt)
            provider = "anthropic"
        
        # Fallback to rule-based
        if not remixed:
            remixed = self._fallback_remix(keyword)
            reasoning = "Rule-based fallback"
            provider = "fallback"
        
        # Cache result
        if remixed:
            self._cache_remix(keyword, attempt, remixed, reasoning)
            self.stats['successful_remixes'] += 1
        
        return KeywordRemixResult(
            original_keyword=keyword,
            remixed_keywords=remixed,
            reasoning=reasoning,
            success=len(remixed) > 0,
            provider=provider,
            attempt=attempt
        )
    
    def remix_keywords_batch(
        self,
        keywords: List[str],
        attempt: int = 1
    ) -> KeywordRemixBatchResult:
        """
        Remix multiple keywords.
        
        Args:
            keywords: List of keywords that had 0 downloads
            attempt: Attempt number
            
        Returns:
            KeywordRemixBatchResult with all remix results
        """
        start_time = time.time()
        
        logger.info(f"Remixing {len(keywords)} zero-download keywords (attempt {attempt})")
        
        results = []
        all_remixed = []
        
        # Try batch processing with Gemini
        if self.gemini_api_key and len(keywords) > 1:
            batch_result = self._batch_remix_gemini(keywords, attempt)
            if batch_result:
                results.extend(batch_result)
                for r in batch_result:
                    all_remixed.extend(r.remixed_keywords)
        
        # Process remaining keywords individually
        processed = {r.original_keyword for r in results}
        remaining = [k for k in keywords if k not in processed]
        
        for kw in remaining:
            result = self.remix_keyword(kw, attempt)
            results.append(result)
            all_remixed.extend(result.remixed_keywords)
        
        unique_remixed = list(dict.fromkeys(all_remixed))
        processing_time = time.time() - start_time
        
        batch_result = KeywordRemixBatchResult(
            total_original=len(keywords),
            total_remixed=len(unique_remixed),
            successful_remixes=sum(1 for r in results if r.success),
            failed_remixes=sum(1 for r in results if not r.success),
            results=results,
            processing_time_seconds=processing_time,
            provider="mixed"
        )
        
        logger.info(json.dumps({
            'event': 'keyword_remix_complete',
            'original_count': len(keywords),
            'remixed_count': len(unique_remixed),
            'success_rate': f"{batch_result.successful_remixes}/{len(keywords)}",
            'processing_time': round(processing_time, 2)
        }))
        
        return batch_result
    
    def _batch_remix_gemini(self, keywords: List[str], attempt: int) -> List[KeywordRemixResult]:
        """Batch remix using Gemini"""
        if not self.gemini_api_key:
            return []

        try:
            from src.llm_client import create_client, LLMRequest, ResponseFormat

            # Get model from config
            gemini_model = 'gemini-2.5-flash'
            if self.config:
                if hasattr(self.config, 'matching'):
                    gemini_model = getattr(self.config.matching, 'gemini_model', gemini_model)
                elif hasattr(self.config, 'llm'):
                    gemini_model = getattr(self.config.llm, 'model', gemini_model)

            client = create_client("gemini", api_key=self.gemini_api_key, model=gemini_model)

            keywords_list = "\n".join([f"- {kw}" for kw in keywords])
            prompt = self.BATCH_REMIX_PROMPT.format(
                keywords_list=keywords_list,
                topic_context=self.topic_context
            )

            request = LLMRequest(
                prompt=prompt,
                response_format=ResponseFormat.JSON,
                cache_key_prefix="keyword_remix_batch"
            )
            response = client.generate(request)

            if response.parsed_data and isinstance(response.parsed_data, dict):
                mapping = response.parsed_data

                results = []
                for original, remixed in mapping.items():
                    if isinstance(remixed, list) and remixed:
                        self._cache_remix(original, attempt, remixed, "Gemini batch")
                        results.append(KeywordRemixResult(
                            original_keyword=original,
                            remixed_keywords=remixed,
                            reasoning="Gemini batch remix",
                            success=True,
                            provider="gemini",
                            attempt=attempt
                        ))
                        self.stats['successful_remixes'] += 1

                self.stats['api_calls'] += 1
                return results

        except Exception as e:
            logger.warning(f"Batch remix failed: {e}")
            return []


def remix_zero_download_keywords(
    failed_keywords: List[str],
    config: Any = None,
    topic_context: str = "",
    cache_dir: Optional[str] = None
) -> Tuple[List[str], KeywordRemixBatchResult]:
    """
    Convenience function to remix failed keywords.
    
    Args:
        failed_keywords: Keywords that had 0 downloads
        config: Config object
        topic_context: Topic context for better remixing
        cache_dir: Cache directory
        
    Returns:
        Tuple of (remixed_keywords, batch_result)
    """
    if not failed_keywords:
        return [], None
    
    remixer = KeywordRemixer(
        config=config,
        topic_context=topic_context,
        cache_dir=cache_dir
    )
    
    result = remixer.remix_keywords_batch(failed_keywords, attempt=1)
    
    all_remixed = []
    for r in result.results:
        all_remixed.extend(r.remixed_keywords)
    
    unique = list(dict.fromkeys(all_remixed))
    
    return unique, result
