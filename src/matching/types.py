from dataclasses import dataclass, asdict, field
from typing import List, Optional
from ..utils import StrategyMatch

@dataclass
class VideoScene:
    """
    Lightweight scene information for strategy matching.
    Used to hold scene metadata retrieved from global cache or scene detection.
    """
    start_time: float
    end_time: float
    description: str = ""
    visual_keywords: List[str] = field(default_factory=list)
    is_broll: bool = False
    face_score: float = 0.5

    def to_dict(self) -> dict:
        return asdict(self)
