"""
Data models for keyword extraction results.
"""

from dataclasses import dataclass, field
from typing import List, Dict


@dataclass
class PrioritizedKeyword:
    """A keyword with priority score for download ordering"""
    keyword: str
    priority: float  # 0.0-1.0, higher = more important
    source: str  # 'entity', 'topic', 'general', 'list_item'
    mention_count: int = 1  # How many times mentioned in script


@dataclass
class KeywordResult:
    """Result from keyword extraction"""
    keywords: List[str]
    segments_analyzed: int
    extraction_method: str
    entities: List[Dict] = field(default_factory=list)  # Named entities with type info
    topic: str = ""  # Detected main topic
    prioritized_keywords: List[PrioritizedKeyword] = field(default_factory=list)  # Keywords with priority

    def get_sorted_keywords(self) -> List[str]:
        """Get keywords sorted by priority (highest first)"""
        if self.prioritized_keywords:
            return [pk.keyword for pk in sorted(self.prioritized_keywords, key=lambda x: x.priority, reverse=True)]
        return self.keywords
