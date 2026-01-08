"""Keyword extraction configuration: Keyword and entity extraction settings.

Extracted from monolithic config.py during refactoring (Jan 7, 2026).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

__all__ = [
    'ListDetectionConfig',
    'KeywordConfig',
]


@dataclass
class ListDetectionConfig:
    """List-based keyword detection settings

    Detects numbered lists in voiceover (e.g., "Number 10 Austin, Texas")
    and ensures each list item gets a guaranteed download keyword.
    """
    enabled: bool = True
    download_first: bool = True  # Download list keywords before general keywords
    skip_if_entity_covered: bool = True  # Skip if already in entity extraction
    keyword_suffix: str = "footage"  # Suffix for generated keywords


@dataclass
class KeywordConfig:
    """Keyword extraction settings"""
    provider: str = "gemini"  # gemini, anthropic, tfidf

    # Extraction settings
    max_keywords: int = 30
    min_keyword_length: int = 3

    # List detection
    list_detection: ListDetectionConfig = None

    # Entity extraction
    extract_entities: bool = True
    entity_types: List[str] = field(default_factory=lambda: [
        "PERSON", "GPE", "ORG", "DATE", "EVENT"
    ])

    # TF-IDF fallback
    use_tfidf_weights: bool = True
    tfidf_max_features: int = 100

    # Footage suffixes
    add_footage_suffixes: bool = True
    footage_suffixes: List[str] = field(default_factory=lambda: [
        "4K footage", "news footage", "drone footage",
        "aerial footage", "stock footage", "documentary footage"
    ])

    # Batch processing
    batch_size: int = 50

    def __post_init__(self):
        if self.list_detection is None:
            self.list_detection = ListDetectionConfig()
