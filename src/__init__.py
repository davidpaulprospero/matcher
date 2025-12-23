"""
Voiceover-to-Footage Matcher v2.5

A comprehensive video matching system with:
- FAISS indexing for fast similarity search
- Hybrid text + visual embeddings
- TF-IDF auto keyword weight detection
- Two-stage matching optimization
- Duration-aware scoring
- Source rotation strategy track
- Comprehensive logging
- GPU-accelerated transcription and transcoding
- MP3/video voiceover transcription support
- GPU lock fix for parallel transcription
"""

__version__ = "2.5.0"

# Convenience exports
from .transcription import (
    transcribe_voiceover_audio, 
    transcribe_voiceover_media,
    transcribe_videos_parallel,
    DeltaAwareIndex
)
