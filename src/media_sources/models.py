"""
Data models for media downloads.

Extracted from entity_images.py lines 42-985 (Jan 7, 2026).
Contains dataclasses for image and video search results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class ImageResult:
    """Represents a downloadable image"""
    id: str
    source: str  # pexels, pixabay, unsplash, google, bing
    url: str  # Web page URL
    download_url: str
    width: int
    height: int
    photographer: str
    description: str
    tags: List[str]
    estimated_size: int = 0  # bytes, if known
    file_path: str = ""  # Path after download
    entity_name: str = ""  # Entity this image represents
    entity_type: str = ""  # PERSON, GPE, ORG, etc.
    query: str = ""  # Search query used


@dataclass
class EntityImageResult:
    """Result of image search for an entity"""
    entity_name: str
    entity_type: str
    context: str
    query: str
    images: List[str] = field(default_factory=list)  # Downloaded file paths
    segment_indices: List[int] = field(default_factory=list)  # Segments mentioning this entity


@dataclass
class VideoResult:
    """Result from video search"""
    id: str
    source: str  # pexels, pixabay
    url: str  # Web page URL
    download_url: str
    width: int
    height: int
    duration: float  # seconds
    quality: str  # hd, sd, etc
    file_type: str  # mp4, etc
    file_path: str = ""  # Path after download


@dataclass
class EntityVideoResult:
    """Result of video search for an entity"""
    entity_name: str
    entity_type: str
    context: str
    query: str
    videos: List[str] = field(default_factory=list)  # Downloaded file paths
    segment_indices: List[int] = field(default_factory=list)
