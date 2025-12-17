"""
Pexels API Integration for Stock Footage

Downloads royalty-free stock footage from Pexels.
All downloads are tagged as "stock_footage" for separate OTIO track.

API Key: Get free at https://www.pexels.com/api/
"""

import os
import re
import json
import logging
import requests
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)

PEXELS_API_URL = "https://api.pexels.com/videos/search"


@dataclass
class PexelsVideo:
    """Represents a Pexels video result"""
    id: int
    url: str
    duration: int  # seconds
    width: int
    height: int
    download_url: str
    photographer: str
    tags: List[str]
    
    @property
    def quality(self) -> str:
        if self.height >= 2160:
            return "4K"
        elif self.height >= 1080:
            return "HD"
        elif self.height >= 720:
            return "720p"
        else:
            return "SD"


class PexelsDownloader:
    """
    Download stock footage from Pexels API.
    
    Features:
    - Search by keyword
    - Filter by duration, resolution
    - Rate limiting (200 requests/hour free tier)
    - Tags downloads as stock footage
    """
    
    def __init__(self, api_key: str = None, output_dir: str = "./downloaded_videos/stock"):
        self.api_key = api_key or os.getenv("PEXELS_API_KEY")
        if not self.api_key:
            raise ValueError("PEXELS_API_KEY not set. Get one at https://www.pexels.com/api/")
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": self.api_key,
            "User-Agent": "Voiceover-Matcher/2.3"
        })
        
        # Rate limiting
        self._request_count = 0
        self._last_request_time = 0
        self._min_request_interval = 0.5  # seconds between requests
        
        # Download tracking
        self.downloaded_files: List[Dict] = []
    
    def _rate_limit(self):
        """Enforce rate limiting"""
        now = time.time()
        elapsed = now - self._last_request_time
        if elapsed < self._min_request_interval:
            time.sleep(self._min_request_interval - elapsed)
        self._last_request_time = time.time()
        self._request_count += 1
    
    def search(
        self,
        query: str,
        per_page: int = 15,
        page: int = 1,
        min_duration: int = 5,
        max_duration: int = 60,
        min_height: int = 720,
        orientation: str = "landscape"  # landscape, portrait, square
    ) -> List[PexelsVideo]:
        """
        Search for videos on Pexels.
        
        Args:
            query: Search keywords
            per_page: Results per page (max 80)
            page: Page number
            min_duration: Minimum video duration in seconds
            max_duration: Maximum video duration in seconds
            min_height: Minimum video height (720, 1080, 2160)
            orientation: Video orientation
        
        Returns:
            List of PexelsVideo objects
        """
        self._rate_limit()
        
        params = {
            "query": query,
            "per_page": min(per_page, 80),
            "page": page,
            "orientation": orientation
        }
        
        try:
            response = self.session.get(PEXELS_API_URL, params=params)
            response.raise_for_status()
            data = response.json()
            
            videos = []
            for item in data.get("videos", []):
                # Filter by duration
                duration = item.get("duration", 0)
                if duration < min_duration or duration > max_duration:
                    continue
                
                # Get best quality video file
                video_files = item.get("video_files", [])
                best_file = self._get_best_video_file(video_files, min_height)
                
                if not best_file:
                    continue
                
                videos.append(PexelsVideo(
                    id=item["id"],
                    url=item["url"],
                    duration=duration,
                    width=best_file.get("width", 0),
                    height=best_file.get("height", 0),
                    download_url=best_file.get("link", ""),
                    photographer=item.get("user", {}).get("name", "Unknown"),
                    tags=self._extract_tags(item)
                ))
            
            logger.info(f"Pexels search '{query}': {len(videos)} results (filtered)")
            return videos
            
        except requests.RequestException as e:
            logger.error(f"Pexels API error: {e}")
            return []
    
    def _get_best_video_file(self, video_files: List[Dict], min_height: int) -> Optional[Dict]:
        """Get the best quality video file meeting minimum height"""
        # Sort by height descending
        sorted_files = sorted(
            [f for f in video_files if f.get("height", 0) >= min_height],
            key=lambda x: x.get("height", 0),
            reverse=True
        )
        
        # Prefer HD (1080p) over 4K to save bandwidth, unless min_height requires 4K
        for f in sorted_files:
            height = f.get("height", 0)
            if 1080 <= height <= 1440:
                return f
        
        # Return highest available if no HD
        return sorted_files[0] if sorted_files else None
    
    def _extract_tags(self, item: Dict) -> List[str]:
        """Extract tags from video item"""
        tags = ["stock_footage", "pexels"]  # Always tag as stock
        
        # Add video pictures as implicit tags (scene descriptions)
        for pic in item.get("video_pictures", []):
            if "picture" in pic:
                tags.append("has_thumbnail")
                break
        
        return tags
    
    def download(
        self,
        video: PexelsVideo,
        filename: str = None
    ) -> Optional[str]:
        """
        Download a video from Pexels.
        
        Args:
            video: PexelsVideo object
            filename: Custom filename (auto-generated if None)
        
        Returns:
            Path to downloaded file, or None if failed
        """
        if not video.download_url:
            logger.warning(f"No download URL for video {video.id}")
            return None
        
        # Generate filename
        if not filename:
            safe_photographer = re.sub(r'[^\w\-]', '_', video.photographer)[:20]
            filename = f"pexels_{video.id}_{safe_photographer}_{video.quality}.mp4"
        
        output_path = self.output_dir / filename
        
        # Skip if already exists
        if output_path.exists():
            logger.info(f"Already downloaded: {filename}")
            return str(output_path)
        
        self._rate_limit()
        
        try:
            response = self.session.get(video.download_url, stream=True)
            response.raise_for_status()
            
            # Download with progress
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    downloaded += len(chunk)
            
            # Create metadata file
            metadata = {
                "source": "pexels",
                "id": video.id,
                "url": video.url,
                "photographer": video.photographer,
                "duration": video.duration,
                "resolution": f"{video.width}x{video.height}",
                "tags": video.tags,
                "is_stock_footage": True,
                "downloaded_at": datetime.now().isoformat()
            }
            
            meta_path = output_path.with_suffix('.meta.json')
            with open(meta_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            self.downloaded_files.append({
                "path": str(output_path),
                "metadata": metadata
            })
            
            logger.info(f"Downloaded: {filename} ({video.duration}s, {video.quality})")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Download failed for {video.id}: {e}")
            if output_path.exists():
                output_path.unlink()
            return None
    
    def search_and_download(
        self,
        query: str,
        max_videos: int = 5,
        min_duration: int = 5,
        max_duration: int = 60,
        min_height: int = 720
    ) -> List[str]:
        """
        Search and download videos in one call.
        
        Args:
            query: Search keywords
            max_videos: Maximum videos to download
            min_duration: Minimum duration in seconds
            max_duration: Maximum duration in seconds
            min_height: Minimum video height
        
        Returns:
            List of downloaded file paths
        """
        videos = self.search(
            query=query,
            per_page=max_videos * 2,  # Get extra in case some fail
            min_duration=min_duration,
            max_duration=max_duration,
            min_height=min_height
        )
        
        downloaded = []
        for video in videos[:max_videos]:
            path = self.download(video)
            if path:
                downloaded.append(path)
        
        return downloaded


def download_pexels_footage(
    keywords: List[str],
    output_dir: str,
    per_keyword: int = 3,
    min_duration: int = 5,
    max_duration: int = 60,
    api_key: str = None
) -> Tuple[List[str], Dict[str, int]]:
    """
    Convenience function to download footage for multiple keywords.
    
    Args:
        keywords: List of search keywords
        output_dir: Directory to save videos
        per_keyword: Videos to download per keyword
        min_duration: Minimum video duration
        max_duration: Maximum video duration
        api_key: Pexels API key (uses env var if not provided)
    
    Returns:
        Tuple of (list of downloaded paths, dict of keyword -> count)
    """
    try:
        downloader = PexelsDownloader(api_key=api_key, output_dir=output_dir)
    except ValueError as e:
        logger.warning(f"Pexels not available: {e}")
        return [], {}
    
    all_downloaded = []
    keyword_counts = {}
    
    for keyword in keywords:
        logger.info(f"Pexels: Searching '{keyword}'...")
        
        paths = downloader.search_and_download(
            query=keyword,
            max_videos=per_keyword,
            min_duration=min_duration,
            max_duration=max_duration
        )
        
        keyword_counts[keyword] = len(paths)
        all_downloaded.extend(paths)
        
        # Small delay between keywords
        time.sleep(0.5)
    
    logger.info(f"Pexels: Downloaded {len(all_downloaded)} videos total")
    return all_downloaded, keyword_counts
