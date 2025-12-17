"""
Multi-Style OTIO Generation

Generates multiple OTIO timelines with different matching styles:
- Style 1: Default settings (current behavior)
- Style 2: User-defined alternative settings

Also handles:
- Stock footage on separate track
- Image sequences on separate track
"""

import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from pathlib import Path
from copy import deepcopy

logger = logging.getLogger(__name__)


@dataclass
class OTIOStyle:
    """Configuration for an OTIO output style"""
    name: str
    description: str
    
    # Matching settings
    confidence_threshold: float = 0.5
    prefer_longer_clips: bool = False
    prefer_shorter_clips: bool = False
    ideal_speed_range: tuple = (0.85, 1.15)
    
    # Track settings
    num_alternatives: int = 2
    include_strategy_tracks: bool = True
    strategy_tracks: List[str] = field(default_factory=lambda: [
        "visual_first", "different_source", "keyword_only", 
        "embedding_diversity", "source_rotation"
    ])
    
    # Source preferences
    prefer_stock_footage: bool = False
    prefer_youtube: bool = True
    mix_sources: bool = True
    
    # Visual preferences  
    prefer_no_faces: bool = True
    prefer_landscape: bool = True
    
    # Pacing
    match_voiceover_pace: bool = True
    allow_speed_adjust: bool = False
    max_speed_adjust: float = 1.2
    
    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "description": self.description,
            "confidence_threshold": self.confidence_threshold,
            "prefer_longer_clips": self.prefer_longer_clips,
            "prefer_shorter_clips": self.prefer_shorter_clips,
            "ideal_speed_range": self.ideal_speed_range,
            "num_alternatives": self.num_alternatives,
            "include_strategy_tracks": self.include_strategy_tracks,
            "strategy_tracks": self.strategy_tracks,
            "prefer_stock_footage": self.prefer_stock_footage,
            "prefer_youtube": self.prefer_youtube,
            "mix_sources": self.mix_sources,
            "prefer_no_faces": self.prefer_no_faces,
            "prefer_landscape": self.prefer_landscape,
            "match_voiceover_pace": self.match_voiceover_pace,
            "allow_speed_adjust": self.allow_speed_adjust,
            "max_speed_adjust": self.max_speed_adjust
        }


# Preset styles
STYLE_DEFAULT = OTIOStyle(
    name="default",
    description="Standard matching with balanced settings",
    confidence_threshold=0.5,
    num_alternatives=2,
    include_strategy_tracks=True,
    prefer_no_faces=True
)

STYLE_STRICT = OTIOStyle(
    name="strict",
    description="High confidence, fewer alternatives",
    confidence_threshold=0.7,
    num_alternatives=1,
    include_strategy_tracks=False,
    prefer_no_faces=True
)

STYLE_STOCK_HEAVY = OTIOStyle(
    name="stock_heavy",
    description="Prefer stock footage (Pexels/Pixabay)",
    confidence_threshold=0.5,
    num_alternatives=3,
    include_strategy_tracks=True,
    prefer_stock_footage=True,
    prefer_youtube=False
)

STYLE_FAST_PACED = OTIOStyle(
    name="fast_paced",
    description="Shorter clips, faster cutting",
    confidence_threshold=0.5,
    prefer_shorter_clips=True,
    ideal_speed_range=(0.7, 1.0),
    num_alternatives=2
)

STYLE_CINEMATIC = OTIOStyle(
    name="cinematic",
    description="Longer clips, slower pace",
    confidence_threshold=0.6,
    prefer_longer_clips=True,
    ideal_speed_range=(1.0, 1.5),
    num_alternatives=2
)


PRESET_STYLES = {
    "default": STYLE_DEFAULT,
    "strict": STYLE_STRICT,
    "stock_heavy": STYLE_STOCK_HEAVY,
    "fast_paced": STYLE_FAST_PACED,
    "cinematic": STYLE_CINEMATIC
}


class MultiStyleOTIOGenerator:
    """
    Generate multiple OTIO timelines with different styles.
    """
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.styles: List[OTIOStyle] = []
        self.generated_files: Dict[str, str] = {}  # style_name -> file_path
    
    def add_style(self, style: OTIOStyle):
        """Add a style to generate"""
        self.styles.append(style)
        logger.info(f"Added OTIO style: {style.name} - {style.description}")
    
    def add_preset(self, preset_name: str):
        """Add a preset style by name"""
        if preset_name in PRESET_STYLES:
            self.add_style(PRESET_STYLES[preset_name])
        else:
            logger.warning(f"Unknown preset: {preset_name}")
    
    def create_custom_style(
        self,
        name: str,
        description: str,
        **kwargs
    ) -> OTIOStyle:
        """Create a custom style with specified settings"""
        style = OTIOStyle(name=name, description=description)
        
        for key, value in kwargs.items():
            if hasattr(style, key):
                setattr(style, key, value)
        
        return style
    
    def get_style_config_for_matching(self, style: OTIOStyle) -> Dict:
        """Convert style to matching config overrides"""
        return {
            "confidence_threshold": style.confidence_threshold,
            "prefer_longer_clips": style.prefer_longer_clips,
            "prefer_shorter_clips": style.prefer_shorter_clips,
            "ideal_speed_range": style.ideal_speed_range,
            "prefer_stock_footage": style.prefer_stock_footage,
            "prefer_no_faces": style.prefer_no_faces
        }
    
    def get_style_config_for_output(self, style: OTIOStyle) -> Dict:
        """Convert style to output config overrides"""
        return {
            "num_alternatives": style.num_alternatives,
            "include_strategy_tracks": style.include_strategy_tracks,
            "strategy_tracks": style.strategy_tracks
        }


def prompt_for_second_style() -> OTIOStyle:
    """
    Interactive prompt to configure the second OTIO style.
    
    Returns:
        Configured OTIOStyle
    """
    print("\n" + "=" * 60)
    print("  SECOND OTIO STYLE CONFIGURATION")
    print("=" * 60)
    print("\n  You'll get TWO timelines with different settings.")
    print("  Style 1: Default (standard matching)")
    print("  Style 2: Configure below\n")
    
    # Preset or custom?
    print("  Available presets:")
    print("    1. strict     - High confidence, fewer alternatives")
    print("    2. stock_heavy - Prefer Pexels/Pixabay footage")
    print("    3. fast_paced  - Shorter clips, faster cutting")
    print("    4. cinematic   - Longer clips, slower pace")
    print("    5. custom      - Configure manually")
    
    try:
        choice = input("\n  Select preset [1-5, default=1]: ").strip() or "1"
        
        preset_map = {
            "1": "strict",
            "2": "stock_heavy", 
            "3": "fast_paced",
            "4": "cinematic",
            "strict": "strict",
            "stock_heavy": "stock_heavy",
            "fast_paced": "fast_paced",
            "cinematic": "cinematic"
        }
        
        if choice in preset_map:
            preset = preset_map[choice]
            style = deepcopy(PRESET_STYLES[preset])
            print(f"\n  ✓ Selected: {style.name} - {style.description}")
            return style
        
        elif choice == "5" or choice.lower() == "custom":
            return _prompt_custom_style()
        
        else:
            # Default to strict
            return deepcopy(STYLE_STRICT)
            
    except (EOFError, KeyboardInterrupt):
        print("\n  Using default: strict")
        return deepcopy(STYLE_STRICT)


def _prompt_custom_style() -> OTIOStyle:
    """Prompt for custom style configuration"""
    print("\n  Custom Style Configuration:")
    
    style = OTIOStyle(name="custom", description="User-defined style")
    
    try:
        # Confidence
        conf = input("  Minimum confidence [0.5]: ").strip()
        if conf:
            style.confidence_threshold = float(conf)
        
        # Alternatives
        alts = input("  Number of alternatives [2]: ").strip()
        if alts:
            style.num_alternatives = int(alts)
        
        # Clip length preference
        print("  Clip length preference:")
        print("    1. Balanced (default)")
        print("    2. Prefer shorter")
        print("    3. Prefer longer")
        length = input("  Select [1-3]: ").strip() or "1"
        if length == "2":
            style.prefer_shorter_clips = True
        elif length == "3":
            style.prefer_longer_clips = True
        
        # Source preference
        print("  Source preference:")
        print("    1. Mix all sources (default)")
        print("    2. Prefer stock footage")
        print("    3. Prefer YouTube")
        source = input("  Select [1-3]: ").strip() or "1"
        if source == "2":
            style.prefer_stock_footage = True
            style.prefer_youtube = False
        elif source == "3":
            style.prefer_stock_footage = False
            style.prefer_youtube = True
        
        # Strategy tracks
        strat = input("  Include strategy tracks (V4-V8)? [Y/n]: ").strip().lower()
        style.include_strategy_tracks = strat != "n"
        
        # Name
        name = input("  Style name [custom]: ").strip() or "custom"
        style.name = name
        
        desc = input("  Description [User-defined]: ").strip() or "User-defined style"
        style.description = desc
        
        print(f"\n  ✓ Created custom style: {style.name}")
        return style
        
    except (EOFError, KeyboardInterrupt, ValueError):
        print("\n  Using defaults for remaining options")
        return style


def prompt_multi_style_enabled() -> bool:
    """Ask if user wants multiple OTIO styles"""
    try:
        choice = input("\n  Generate multiple OTIO styles? [y/N]: ").strip().lower()
        return choice in ('y', 'yes')
    except (EOFError, KeyboardInterrupt):
        return False


# Track definitions for stock footage and images
STOCK_FOOTAGE_TRACK = "V_Stock"  # Separate track for stock footage
STOCK_AUDIO_TRACK = "A_Stock"
IMAGE_TRACK = "V_Images"  # Track for images/stills


def get_track_for_source(source_type: str) -> str:
    """
    Get the appropriate track name for a source type.
    
    Args:
        source_type: "youtube", "pexels", "pixabay", "image"
    
    Returns:
        Track name
    """
    if source_type in ("pexels", "pixabay"):
        return STOCK_FOOTAGE_TRACK
    elif source_type == "image":
        return IMAGE_TRACK
    else:
        return "V1"  # Default primary track


def is_stock_footage(metadata: Dict) -> bool:
    """Check if clip is stock footage based on metadata"""
    if not metadata:
        return False
    
    # Check source field
    source = metadata.get("source", "").lower()
    if source in ("pexels", "pixabay"):
        return True
    
    # Check tags
    tags = metadata.get("tags", [])
    if "stock_footage" in tags or "stock" in tags:
        return True
    
    # Check is_stock_footage flag
    return metadata.get("is_stock_footage", False)
