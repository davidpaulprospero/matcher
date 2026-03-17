"""Shared fixtures for benchmark tests."""

import pytest
import tempfile
from pathlib import Path
from dataclasses import dataclass
from typing import List
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


@dataclass
class MockSegment:
    """Mock voiceover segment for benchmarking."""
    index: int
    start_time: float
    end_time: float
    text: str


@dataclass
class MockVideo:
    """Mock video metadata for benchmarking."""
    file: str
    duration: float
    title: str
    video_id: str


@pytest.fixture
def sample_segments() -> List[MockSegment]:
    """Create sample voiceover segments for benchmarking."""
    segments = []
    for i in range(100):
        segments.append(MockSegment(
            index=i,
            start_time=i * 5.0,
            end_time=(i + 1) * 5.0,
            text=f"This is test segment {i} with some sample text about various topics."
        ))
    return segments


@pytest.fixture
def sample_videos() -> List[MockVideo]:
    """Create sample video metadata for benchmarking."""
    videos = []
    for i in range(1000):
        videos.append(MockVideo(
            file=f"/fake/path/video_{i}.mp4",
            duration=60.0 + (i % 120),
            title=f"Test video {i} about technology and innovation",
            video_id=f"video_{i:04d}"
        ))
    return videos


@pytest.fixture
def temp_benchmark_dir():
    """Create temporary directory for benchmark files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_text_corpus() -> str:
    """Create sample text corpus for keyword extraction."""
    return """
    Artificial intelligence and machine learning are transforming the technology industry.
    Deep learning neural networks enable computers to recognize patterns in vast amounts of data.
    Natural language processing allows machines to understand and generate human language.
    Computer vision systems can identify objects, faces, and scenes in images and videos.
    Robotics combines AI with physical systems to create autonomous machines.
    The future of technology lies in the intersection of AI, quantum computing, and biotechnology.
    """ * 10  # Repeat to create substantial corpus


@pytest.fixture
def sample_keywords() -> List[str]:
    """Create sample keywords for matching."""
    return [
        "artificial intelligence",
        "machine learning",
        "deep learning",
        "neural networks",
        "computer vision",
        "natural language processing",
        "robotics",
        "quantum computing",
        "biotechnology",
        "technology innovation"
    ]
