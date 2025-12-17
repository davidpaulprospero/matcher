"""
Pixabay API Integration for Stock Footage

Downloads royalty-free stock footage from Pixabay.
All downloads are tagged as "stock_footage" for separate OTIO track.

API Key: Get free at https://pixabay.com/api/docs/
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

PIXABAY_VIDEO_API_URL = "https://pixabay.com/api/videos/"


@dataclass
class PixabayVideo:
    """Represents a Pixabay video result"""
    id: int
    page_url: str
    duration: int  # seconds
    width: int
    height: int
    download_url: str
    user: str
    tags: List[str]
    views: int
    downloads: int
    
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


class PixabayDownloader:
    """
    Download stock footage from Pixabay API.
    
    Features:
    - Search by keyword with filters
    - Multiple quality options (4K, HD, SD)
    - Rate limiting (100 requests/minute)
    - Tags downloads as stock footage
    """
    
    def __init__(self, api_key: str = None, output_dir: str = "./downloaded_videos/stock"):
        self.api_key = api_key or os.getenv("PIXABAY_API_KEY")
        if not self.api_key:
            raise ValueError("PIXABAY_API_KEY not set. Get one at https://pixabay.com/api/docs/")
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Voiceover-Matcher/2.3"
        })
        
        # Rate limiting
        self._request_count = 0
        self._last_request_time = 0
        self._min_request_interval = 0.6  # seconds between requests
        
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
        per_page: int = 20,
        page: int = 1,
        min_duration: int = 5,
        max_duration: int = 60,
        min_height: int = 720,
        video_type: str = "all",  # all, film, animation
        category: str = None,  # nature, science, etc.
        order: str = "popular"  # popular, latest
    ) -> List[PixabayVideo]:
        """
        Search for videos on Pixabay.
        
        Args:
            query: Search keywords
            per_page: Results per page (max 200)
            page: Page number
            min_duration: Minimum video duration in seconds
            max_duration: Maximum video duration in seconds
            min_height: Minimum video height
            video_type: Type of video content
            category: Content category filter
            order: Sort order
        
        Returns:
            List of PixabayVideo objects
        """
        self._rate_limit()
        
        params = {
            "key": self.api_key,
            "q": query,
            "per_page": min(per_page, 200),
            "page": page,
            "video_type": video_type,
            "order": order,
            "safesearch": "true"
        }
        
        if category:
            params["category"] = category
        
        try:
            response = self.session.get(PIXABAY_VIDEO_API_URL, params=params)
            response.raise_for_status()
            data = response.json()
            
            videos = []
            for item in data.get("hits", []):
                # Filter by duration
                duration = item.get("duration", 0)
                if duration < min_duration or duration > max_duration:
                    continue
                
                # Get video files
                video_data = item.get("videos", {})
                best_file = self._get_best_video_file(video_data, min_height)
                
                if not best_file:
                    continue
                
                # Extract tags from comma-separated string
                tags_str = item.get("tags", "")
                tags = [t.strip() for t in tags_str.split(",") if t.strip()]
                tags.extend(["stock_footage", "pixabay"])
                
                videos.append(PixabayVideo(
                    id=item["id"],
                    page_url=item.get("pageURL", ""),
                    duration=duration,
                    width=best_file.get("width", 0),
                    height=best_file.get("height", 0),
                    download_url=best_file.get("url", ""),
                    user=item.get("user", "Unknown"),
                    tags=tags,
                    views=item.get("views", 0),
                    downloads=item.get("downloads", 0)
                ))
            
            logger.info(f"Pixabay search '{query}': {len(videos)} results (filtered)")
            return videos
            
        except requests.RequestException as e:
            logger.error(f"Pixabay API error: {e}")
            return []
    
    def _get_best_video_file(self, video_data: Dict, min_height: int) -> Optional[Dict]:
        """Get the best quality video file meeting minimum height"""
        # Quality order: large (1920), medium (1280), small (960), tiny (640)
        quality_order = ["large", "medium", "small", "tiny"]
        height_map = {"large": 1080, "medium": 720, "small": 540, "tiny": 360}
        
        for quality in quality_order:
            if quality in video_data:
                file_info = video_data[quality]
                height = file_info.get("height", height_map.get(quality, 0))
                if height >= min_height:
                    return file_info
        
        # If nothing meets min_height, return best available
        for quality in quality_order:
            if quality in video_data:
                return video_data[quality]
        
        return None
    
    def download(
        self,
        video: PixabayVideo,
        filename: str = None
    ) -> Optional[str]:
        """
        Download a video from Pixabay.
        
        Args:
            video: PixabayVideo object
            filename: Custom filename (auto-generated if None)
        
        Returns:
            Path to downloaded file, or None if failed
        """
        if not video.download_url:
            logger.warning(f"No download URL for video {video.id}")
            return None
        
        # Generate filename
        if not filename:
            safe_user = re.sub(r'[^\w\-]', '_', video.user)[:20]
            filename = f"pixabay_{video.id}_{safe_user}_{video.quality}.mp4"
        
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
                "source": "pixabay",
                "id": video.id,
                "url": video.page_url,
                "user": video.user,
                "duration": video.duration,
                "resolution": f"{video.width}x{video.height}",
                "tags": video.tags,
                "is_stock_footage": True,
                "views": video.views,
                "downloads": video.downloads,
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
        
        Returns:
            List of downloaded file paths
        """
        videos = self.search(
            query=query,
            per_page=max_videos * 2,
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


def download_pixabay_footage(
    keywords: List[str],
    output_dir: str,
    per_keyword: int = 3,
    min_duration: int = 5,
    max_duration: int = 60,
    api_key: str = None
) -> Tuple[List[str], Dict[str, int]]:
    """
    Convenience function to download footage for multiple keywords.
    
    Returns:
        Tuple of (list of downloaded paths, dict of keyword -> count)
    """
    try:
        downloader = PixabayDownloader(api_key=api_key, output_dir=output_dir)
    except ValueError as e:
        logger.warning(f"Pixabay not available: {e}")
        return [], {}
    
    all_downloaded = []
    keyword_counts = {}
    
    for keyword in keywords:
        logger.info(f"Pixabay: Searching '{keyword}'...")
        
        paths = downloader.search_and_download(
            query=keyword,
            max_videos=per_keyword,
            min_duration=min_duration,
            max_duration=max_duration
        )
        
        keyword_counts[keyword] = len(paths)
        all_downloaded.extend(paths)
        
        time.sleep(0.5)
    
    logger.info(f"Pixabay: Downloaded {len(all_downloaded)} videos total")
    return all_downloaded, keyword_counts
