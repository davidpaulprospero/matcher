"""
File watcher for pre-compute on import
Watches the downloaded_videos directory and automatically indexes new videos
"""

import os
import time
import logging
import threading
from pathlib import Path
from typing import Set, Callable, Optional
from datetime import datetime

from .config import Config
from .utils import CacheManager, VideoIndex

logger = logging.getLogger(__name__)

VIDEO_EXTENSIONS = {'.mp4', '.mkv', '.avi', '.mov', '.webm', '.flv', '.wmv'}


class VideoWatcher:
    """Watches for new videos and triggers indexing"""
    
    def __init__(
        self,
        config: Config,
        cache: CacheManager,
        on_new_video: Optional[Callable[[str], None]] = None
    ):
        self.config = config
        self.cache = cache
        self.on_new_video = on_new_video
        
        self.watch_dir = Path(config.downloaded_videos_dir)
        self.interval = config.watch_interval
        
        self._known_files: Set[str] = set()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        
        # Initialize known files from cache
        self._load_known_files()
    
    def _load_known_files(self):
        """Load list of already-indexed files from cache"""
        master_index = self.cache.get_master_index()
        self._known_files = set(master_index.keys())
        logger.info(f"Loaded {len(self._known_files)} known videos from index")
    
    def _save_known_files(self):
        """Save known files to cache"""
        # Get hashes for all known files
        master_index = {}
        for file_path in self._known_files:
            if Path(file_path).exists():
                try:
                    file_hash = self.cache.get_file_hash(file_path)
                    master_index[file_path] = file_hash
                except:
                    pass
        self.cache.save_master_index(master_index)
    
    def _scan_for_videos(self) -> Set[str]:
        """Scan directory for video files"""
        videos = set()
        
        if not self.watch_dir.exists():
            return videos
        
        for ext in VIDEO_EXTENSIONS:
            for video_path in self.watch_dir.glob(f"*{ext}"):
                videos.add(str(video_path))
            for video_path in self.watch_dir.glob(f"*{ext.upper()}"):
                videos.add(str(video_path))
        
        return videos
    
    def _check_for_new_videos(self) -> Set[str]:
        """Check for new videos that haven't been indexed"""
        current_videos = self._scan_for_videos()
        new_videos = current_videos - self._known_files
        return new_videos
    
    def get_videos_to_index(self) -> list:
        """Get list of videos that need indexing (incremental)"""
        if self.config.incremental:
            return list(self._check_for_new_videos())
        else:
            return list(self._scan_for_videos())
    
    def mark_indexed(self, video_path: str):
        """Mark a video as indexed"""
        self._known_files.add(video_path)
        self._save_known_files()
    
    def _watch_loop(self):
        """Main watch loop"""
        logger.info(f"Starting video watcher on {self.watch_dir}")
        logger.info(f"Checking every {self.interval} seconds")
        
        while self._running:
            try:
                new_videos = self._check_for_new_videos()
                
                if new_videos:
                    logger.info(f"Found {len(new_videos)} new video(s)")
                    
                    for video_path in new_videos:
                        logger.info(f"  New: {Path(video_path).name}")
                        
                        if self.on_new_video:
                            try:
                                self.on_new_video(video_path)
                            except Exception as e:
                                logger.error(f"Failed to index {video_path}: {e}")
                        
                        self._known_files.add(video_path)
                    
                    self._save_known_files()
            
            except Exception as e:
                logger.error(f"Watch loop error: {e}")
            
            # Sleep in intervals to allow quick shutdown
            for _ in range(self.interval):
                if not self._running:
                    break
                time.sleep(1)
        
        logger.info("Video watcher stopped")
    
    def start(self):
        """Start watching in background thread"""
        if self._running:
            return
        
        self._running = True
        self._thread = threading.Thread(target=self._watch_loop, daemon=True)
        self._thread.start()
    
    def stop(self):
        """Stop watching"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None
    
    def watch_once(self):
        """Check for new videos once (non-blocking)"""
        new_videos = self._check_for_new_videos()
        
        if new_videos and self.on_new_video:
            for video_path in new_videos:
                try:
                    self.on_new_video(video_path)
                    self._known_files.add(video_path)
                except Exception as e:
                    logger.error(f"Failed to index {video_path}: {e}")
            
            self._save_known_files()
        
        return new_videos


def get_incremental_videos(config: Config, cache: CacheManager) -> list:
    """
    Get list of videos that need processing.
    If incremental=True, only returns new videos.
    """
    watcher = VideoWatcher(config, cache)
    return watcher.get_videos_to_index()


def watch_and_index(
    config: Config,
    cache: CacheManager,
    index_function: Callable[[str], None]
):
    """
    Start watching for new videos and index them.
    
    Args:
        config: Configuration
        cache: Cache manager
        index_function: Function to call when new video is found
    """
    watcher = VideoWatcher(config, cache, on_new_video=index_function)
    
    # Initial scan
    logger.info("Performing initial scan...")
    watcher.watch_once()
    
    if config.watch_mode:
        logger.info("Starting continuous watch mode...")
        watcher.start()
        
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Stopping watcher...")
            watcher.stop()
    else:
        logger.info("Watch mode disabled, single scan completed")
