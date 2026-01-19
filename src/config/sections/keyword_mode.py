"""Keyword Mode configuration: Pipeline without voiceover input.

Supports three modes:
- montage: Equal-time segments from keywords (simplest)
- script: LLM-generated narration script -> SRT (primary)
- collection: Download/organize videos by keyword (no timeline)

Created: 2026-01-19
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

__all__ = [
    'MontageConfig',
    'ScriptLLMConfig',
    'ScriptConfig',
    'CollectionConfig',
    'KeywordModeConfig',
]


@dataclass
class MontageConfig:
    """Settings for montage mode - equal-time segments from keywords.

    Each keyword becomes a segment with configurable duration.
    """
    segment_duration: float = 10.0  # Seconds per keyword (if no total_duration)
    total_duration: Optional[float] = None  # If set, divides evenly among keywords
    transition_gap: float = 0.0  # Gap between segments in seconds


@dataclass
class ScriptLLMConfig:
    """LLM settings for script generation."""
    provider: str = "ollama"  # ollama, gemini, anthropic
    model: str = "mistral:7b"  # Default model
    fallback_models: List[str] = field(default_factory=lambda: [
        "qwen2.5:7b",
        "llama3.2:3b",
    ])
    temperature: float = 0.7
    max_tokens: int = 1000
    ollama_host: str = "http://localhost:11434"
    timeout: int = 60  # seconds


@dataclass
class ScriptConfig:
    """Settings for script mode - LLM generates narration script.

    Pipeline: Keywords -> LLM Script -> Timed SRT -> Normal matching
    The SRT is the artifact - user can optionally do TTS later.
    """
    # Script generation style
    style: str = "documentary"  # documentary | promotional | narrative | listicle | poetic | minimal
    tone: str = "inspiring"  # inspiring | serious | playful | urgent | contemplative

    # Timing
    target_duration: float = 120.0  # Target duration in seconds
    words_per_minute: int = 150  # 120=slow, 150=normal, 180=fast
    min_segment_duration: float = 2.0  # Minimum segment duration in seconds
    segment_gap: float = 0.3  # Gap between segments in seconds

    # Output
    save_script_txt: bool = True  # Save raw script as .txt
    save_srt: bool = True  # Save generated .srt

    # LLM settings
    llm: ScriptLLMConfig = None

    # Validation
    require_all_keywords: bool = True  # Ensure all keywords appear in script

    # Segment splitting
    split_mode: str = "paragraph"  # paragraph | sentence | hybrid

    def __post_init__(self):
        if self.llm is None:
            self.llm = ScriptLLMConfig()
        elif isinstance(self.llm, dict):
            self.llm = ScriptLLMConfig(**self.llm)


@dataclass
class CollectionConfig:
    """Settings for collection mode - download/organize without timeline.

    Downloads videos for each keyword and organizes into folders.
    """
    clips_per_keyword: int = 10
    output_format: str = "folders"  # folders | flat_with_manifest
    include_metadata: bool = True  # Save transcripts, scores in manifest


@dataclass
class KeywordModeConfig:
    """Master config for keyword mode - pipeline without voiceover.

    Usage in config.yaml:
    ```yaml
    keyword_mode:
      enabled: true
      mode: "script"  # montage | script | collection
      keywords: ["sunset", "ocean", "beach"]  # Optional CLI override

      montage:
        segment_duration: 10.0

      script:
        style: "documentary"
        target_duration: 120

      collection:
        clips_per_keyword: 10
    ```
    """
    enabled: bool = False  # Must be explicitly enabled
    mode: str = "script"  # montage | script | collection
    keywords: List[str] = field(default_factory=list)  # Keywords from config (CLI can override)

    # Mode-specific configs
    montage: MontageConfig = None
    script: ScriptConfig = None
    collection: CollectionConfig = None

    def __post_init__(self):
        if self.montage is None:
            self.montage = MontageConfig()
        elif isinstance(self.montage, dict):
            self.montage = MontageConfig(**self.montage)

        if self.script is None:
            self.script = ScriptConfig()
        elif isinstance(self.script, dict):
            self.script = ScriptConfig(**self.script)

        if self.collection is None:
            self.collection = CollectionConfig()
        elif isinstance(self.collection, dict):
            self.collection = CollectionConfig(**self.collection)
