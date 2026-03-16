"""
Video Deduplication Module

Uses perceptual hashing (imagehash) to detect and remove duplicate videos.
Runs after download, before transcription to save processing time.

Features:
- First frame comparison (fast)
- Auto-delete duplicates
- Generates deduplication report
"""

import os
import json
import logging
import subprocess
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

from .downloader.utils import SUBPROCESS_FLAGS
from collections import defaultdict

logger = logging.getLogger(__name__)

# Supported video extensions
VIDEO_EXTENSIONS = {'.mp4', '.mkv', '.avi', '.mov', '.wmv', '.flv', '.webm', '.m4v', '.mpeg', '.mpg', '.mxf'}


@dataclass
class DuplicateGroup:
    """A group of duplicate videos"""
    keep: str  # Path to the video we're keeping
    keep_size: int  # File size of kept video
    duplicates: List[str]  # Paths to duplicate videos
    similarity: float  # Hash similarity (0-1)
    hash_distance: int  # Hamming distance between hashes


@dataclass
class DeduplicationReport:
    """Summary of deduplication results"""
    total_videos: int
    unique_videos: int
    duplicates_found: int
    duplicates_deleted: int
    space_saved_mb: float
    duplicate_groups: List[DuplicateGroup]
    timestamp: str
    
    def to_dict(self) -> dict:
        return {
            'total_videos': self.total_videos,
            'unique_videos': self.unique_videos,
            'duplicates_found': self.duplicates_found,
            'duplicates_deleted': self.duplicates_deleted,
            'space_saved_mb': round(self.space_saved_mb, 2),
            'duplicate_groups': [asdict(g) for g in self.duplicate_groups],
            'timestamp': self.timestamp
        }


class VideoDeduplicator:
    """
    Detect and remove duplicate videos using perceptual hashing.
    """
    
    # Hash distance threshold (lower = stricter)
    # 0 = identical, <5 = same clip, <10 = very similar, <15 = somewhat similar
    DEFAULT_THRESHOLD = 10
    DEFAULT_FRAME_TIMEOUT = 30  # Seconds for FFmpeg frame extraction
    
    def __init__(self, config=None, threshold: int = None):
        """
        Initialize deduplicator.
        
        Args:
            config: Pipeline config (optional)
            threshold: Hash distance threshold (default: 10)
        """
        self.config = config
        
        # Get threshold from config or use default
        if threshold is not None:
            self.threshold = threshold
        elif config and hasattr(config, 'deduplication'):
            self.threshold = getattr(config.deduplication, 'hash_threshold', self.DEFAULT_THRESHOLD)
        else:
            self.threshold = self.DEFAULT_THRESHOLD
        
        # Get frame extraction timeout from config
        self.frame_timeout = self.DEFAULT_FRAME_TIMEOUT
        if config and hasattr(config, 'deduplication'):
            self.frame_timeout = getattr(config.deduplication, 'frame_timeout', self.DEFAULT_FRAME_TIMEOUT)
        
        # Get auto_delete setting from config
        self.auto_delete = True
        if config and hasattr(config, 'deduplication'):
            self.auto_delete = getattr(config.deduplication, 'auto_delete', True)
        
        # Try to import imagehash
        try:
            import imagehash
            from PIL import Image
            self.imagehash = imagehash
            self.Image = Image
            self._available = True
        except ImportError:
            logger.warning("imagehash not installed. Install with: pip install imagehash Pillow")
            self._available = False
        
        # Cache for computed hashes
        self._hash_cache: Dict[str, str] = {}
    
    def is_available(self) -> bool:
        """Check if deduplication is available"""
        return self._available
    
    def _extract_first_frame(self, video_path: str) -> Optional[str]:
        """
        Extract first frame from video using FFmpeg.
        Returns path to temporary frame image.
        """
        # Create temp frame path
        frame_path = Path(video_path).with_suffix('.thumb.jpg')
        
        # FFmpeg command to extract first frame
        # Try GPU-accelerated decoding first
        cmd = [
            'ffmpeg',
            '-loglevel', 'error',
            '-y',
            '-hwaccel', 'cuda',  # GPU decoding for NVIDIA
            '-i', video_path,
            '-vframes', '1',
            '-q:v', '2',
            str(frame_path)
        ]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.frame_timeout,
                encoding='utf-8',
                errors='replace',
                **SUBPROCESS_FLAGS
            )
            
            if frame_path.exists() and frame_path.stat().st_size > 0:
                return str(frame_path)
            
            # Fallback to CPU if GPU fails
            cmd_cpu = [
                'ffmpeg',
                '-loglevel', 'error',
                '-y',
                '-i', video_path,
                '-vframes', '1',
                '-q:v', '2',
                str(frame_path)
            ]
            result = subprocess.run(cmd_cpu, capture_output=True, text=True, timeout=self.frame_timeout, encoding='utf-8', errors='replace', **SUBPROCESS_FLAGS)
            
            if frame_path.exists() and frame_path.stat().st_size > 0:
                return str(frame_path)
            else:
                return None
                
        except Exception as e:
            logger.debug(f"Failed to extract frame from {video_path}: {e}")
            return None
    
    def _compute_hash(self, video_path: str) -> Optional[str]:
        """
        Compute perceptual hash of video's first frame.
        Returns hash string or None if failed.
        """
        if not self._available:
            return None
        
        # Check cache
        if video_path in self._hash_cache:
            return self._hash_cache[video_path]
        
        # Extract first frame
        frame_path = self._extract_first_frame(video_path)
        if not frame_path:
            return None
        
        try:
            # Load image and compute hash
            img = self.Image.open(frame_path)
            
            # Use perceptual hash (robust to minor changes)
            phash = self.imagehash.phash(img)
            hash_str = str(phash)
            
            # Cache result
            self._hash_cache[video_path] = hash_str
            
            return hash_str
            
        except Exception as e:
            logger.debug(f"Failed to compute hash for {video_path}: {e}")
            return None
            
        finally:
            # Clean up temp frame
            try:
                Path(frame_path).unlink()
            except (OSError, FileNotFoundError) as e:
                # Temp file cleanup is non-critical - file may already be gone
                logger.debug(f"Could not remove temp frame {frame_path}: {e}")
    
    def _hash_distance(self, hash1: str, hash2: str) -> int:
        """
        Compute Hamming distance between two hash strings.
        Lower = more similar.
        """
        if not self._available:
            return 999
        
        try:
            h1 = self.imagehash.hex_to_hash(hash1)
            h2 = self.imagehash.hex_to_hash(hash2)
            return h1 - h2
        except (ValueError, TypeError) as e:
            # Invalid hash string format or type - return max distance
            logger.debug(f"Hash distance calculation failed: {e}")
            return 999
    
    def find_duplicates(
        self,
        video_dir: str,
        threshold: int = None
    ) -> List[DuplicateGroup]:
        """
        Find duplicate videos in directory.
        
        Args:
            video_dir: Directory containing videos
            threshold: Hash distance threshold (default: self.threshold)
        
        Returns:
            List of DuplicateGroup objects
        """
        if not self._available:
            logger.warning("Deduplication not available (imagehash not installed)")
            return []
        
        threshold = threshold or self.threshold
        video_dir = Path(video_dir)
        
        # Find all video files
        videos = []
        for ext in VIDEO_EXTENSIONS:
            videos.extend(video_dir.rglob(f'*{ext}'))
        
        if not videos:
            return []
        
        logger.info(f"Computing hashes for {len(videos)} videos...")
        
        # Compute hashes for all videos
        video_hashes: Dict[str, str] = {}
        for video in videos:
            video_path = str(video)
            hash_val = self._compute_hash(video_path)
            if hash_val:
                video_hashes[video_path] = hash_val
        
        logger.info(f"Computed {len(video_hashes)} hashes, finding duplicates...")
        
        # Group by similar hashes
        # Using Union-Find approach for efficiency
        parent = {v: v for v in video_hashes}
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py
        
        # Compare all pairs (O(n^2) but necessary for hash comparison)
        video_list = list(video_hashes.keys())
        for i in range(len(video_list)):
            for j in range(i + 1, len(video_list)):
                v1, v2 = video_list[i], video_list[j]
                dist = self._hash_distance(video_hashes[v1], video_hashes[v2])
                if dist <= threshold:
                    union(v1, v2)
        
        # Group videos by their root parent
        groups: Dict[str, List[str]] = defaultdict(list)
        for video in video_hashes:
            root = find(video)
            groups[root].append(video)
        
        # Convert to DuplicateGroup objects
        duplicate_groups = []
        for root, members in groups.items():
            if len(members) > 1:
                # Sort by file size (keep largest)
                members_with_size = [(m, Path(m).stat().st_size) for m in members]
                members_with_size.sort(key=lambda x: x[1], reverse=True)
                
                keep_path, keep_size = members_with_size[0]
                duplicates = [m[0] for m in members_with_size[1:]]
                
                # Calculate average similarity
                keep_hash = video_hashes[keep_path]
                distances = [self._hash_distance(keep_hash, video_hashes[d]) for d in duplicates]
                avg_distance = sum(distances) / len(distances) if distances else 0
                similarity = 1.0 - (avg_distance / 64.0)  # 64 bits in phash
                
                duplicate_groups.append(DuplicateGroup(
                    keep=keep_path,
                    keep_size=keep_size,
                    duplicates=duplicates,
                    similarity=round(similarity, 3),
                    hash_distance=int(avg_distance)
                ))
        
        return duplicate_groups
    
    def deduplicate(
        self,
        video_dir: str,
        auto_delete: bool = None,
        report_path: str = None
    ) -> DeduplicationReport:
        """
        Find and remove duplicate videos.
        
        Args:
            video_dir: Directory containing videos
            auto_delete: Whether to automatically delete duplicates (default: from config)
            report_path: Path to save JSON report (optional)
        
        Returns:
            DeduplicationReport with summary
        """
        # Use config value if not explicitly provided
        if auto_delete is None:
            auto_delete = self.auto_delete
        
        video_dir = Path(video_dir)
        
        # Count total videos before
        all_videos = []
        for ext in VIDEO_EXTENSIONS:
            all_videos.extend(video_dir.rglob(f'*{ext}'))
        total_videos = len(all_videos)
        
        # Find duplicates
        duplicate_groups = self.find_duplicates(str(video_dir))
        
        # Calculate stats
        duplicates_found = sum(len(g.duplicates) for g in duplicate_groups)
        space_saved = 0
        deleted_count = 0
        
        # Delete duplicates if requested
        if auto_delete and duplicate_groups:
            logger.info(f"Deleting {duplicates_found} duplicate videos...")
            
            for group in duplicate_groups:
                for dup_path in group.duplicates:
                    try:
                        dup_size = Path(dup_path).stat().st_size
                        Path(dup_path).unlink()
                        space_saved += dup_size
                        deleted_count += 1
                        logger.debug(f"Deleted duplicate: {Path(dup_path).name}")
                    except Exception as e:
                        logger.warning(f"Failed to delete {dup_path}: {e}")
        
        # Create report
        report = DeduplicationReport(
            total_videos=total_videos,
            unique_videos=total_videos - duplicates_found,
            duplicates_found=duplicates_found,
            duplicates_deleted=deleted_count,
            space_saved_mb=space_saved / (1024 * 1024),
            duplicate_groups=duplicate_groups,
            timestamp=datetime.now().isoformat()
        )
        
        # Save report if path provided
        if report_path:
            report_path = Path(report_path)
            report_path.parent.mkdir(parents=True, exist_ok=True)
            with open(report_path, 'w') as f:
                json.dump(report.to_dict(), f, indent=2)
            logger.info(f"Saved deduplication report: {report_path}")
        
        return report
    
    def get_stats_preview(self, video_dir: str) -> Dict:
        """
        Quick preview of potential duplicates without deleting.
        """
        groups = self.find_duplicates(str(video_dir))
        
        total_dupes = sum(len(g.duplicates) for g in groups)
        potential_savings = sum(
            sum(Path(d).stat().st_size for d in g.duplicates)
            for g in groups
        )
        
        return {
            'duplicate_groups': len(groups),
            'total_duplicates': total_dupes,
            'potential_savings_mb': round(potential_savings / (1024 * 1024), 2)
        }


def deduplicate_videos(
    video_dir: str,
    auto_delete: bool = True,
    threshold: int = 10,
    report_path: str = None
) -> DeduplicationReport:
    """
    Convenience function to deduplicate videos in a directory.
    
    Args:
        video_dir: Directory containing videos
        auto_delete: Whether to automatically delete duplicates
        threshold: Hash distance threshold (0-15, lower=stricter)
        report_path: Path to save JSON report
    
    Returns:
        DeduplicationReport
    """
    deduplicator = VideoDeduplicator(threshold=threshold)
    
    if not deduplicator.is_available():
        logger.error("Cannot deduplicate: imagehash not installed")
        return DeduplicationReport(
            total_videos=0,
            unique_videos=0,
            duplicates_found=0,
            duplicates_deleted=0,
            space_saved_mb=0,
            duplicate_groups=[],
            timestamp=datetime.now().isoformat()
        )
    
    return deduplicator.deduplicate(
        video_dir=video_dir,
        auto_delete=auto_delete,
        report_path=report_path
    )
