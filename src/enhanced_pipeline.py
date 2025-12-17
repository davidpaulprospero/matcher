"""
Enhanced Pipeline Integration Module

Integrates all new features:
1. 90% confidence enforcement with keyword remix
2. Zero-download keyword remix
3. Pexels/Pixabay stock footage (separate track)
4. Image downloads (>1MB filter)
5. Multi-style OTIO generation

Usage in main.py:
    from src.enhanced_pipeline import EnhancedPipeline
    
    pipeline = EnhancedPipeline(config, project_dir)
    pipeline.run()
"""

import logging
import json
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class EnhancedPipelineConfig:
    """Configuration for enhanced pipeline features"""
    
    # Confidence enforcement
    min_confidence: float = 0.90  # 90% minimum
    max_remix_retries: int = 3
    remix_low_confidence: bool = True
    remix_zero_downloads: bool = True
    
    # Stock footage
    enable_pexels: bool = True
    enable_pixabay: bool = True
    pexels_per_keyword: int = 2
    pixabay_per_keyword: int = 2
    stock_footage_track: str = "V_Stock"
    
    # Image downloads
    enable_images: bool = True
    image_min_size_mb: float = 1.0
    images_per_keyword: int = 2
    image_track: str = "V_Images"
    
    # Multi-style OTIO
    enable_multi_style: bool = True
    default_style: str = "default"
    second_style: str = "strict"  # Or user-configured
    
    # Topic context for keyword generation
    topic_context: str = ""


class EnhancedDownloader:
    """
    Enhanced downloader with multiple sources and retry logic.
    """
    
    def __init__(self, config: EnhancedPipelineConfig, output_dir: str):
        self.config = config
        self.output_dir = Path(output_dir)
        
        # Track downloads by source
        self.downloads_by_source: Dict[str, List[str]] = {
            "youtube": [],
            "pexels": [],
            "pixabay": [],
            "images": []
        }
        
        self.keyword_counts: Dict[str, Dict[str, int]] = {}  # keyword -> {source: count}
        self.failed_keywords: List[str] = []
        self.remixed_keywords: Dict[str, List[str]] = {}
    
    def download_all_sources(
        self,
        keywords: List[str],
        youtube_downloader=None,
        progress_callback=None
    ) -> Dict[str, int]:
        """
        Download from all enabled sources.
        
        Returns:
            Dict of keyword -> total download count
        """
        total_counts = {k: 0 for k in keywords}
        
        # 1. YouTube (existing downloader)
        if youtube_downloader:
            logger.info("Downloading from YouTube...")
            # This uses the existing downloader
            # Results tracked separately
        
        # 2. Pexels
        if self.config.enable_pexels:
            try:
                from .pexels import download_pexels_footage
                
                stock_dir = self.output_dir / "stock"
                pexels_paths, pexels_counts = download_pexels_footage(
                    keywords=keywords,
                    output_dir=str(stock_dir),
                    per_keyword=self.config.pexels_per_keyword
                )
                
                self.downloads_by_source["pexels"] = pexels_paths
                for k, count in pexels_counts.items():
                    total_counts[k] = total_counts.get(k, 0) + count
                    
                logger.info(f"Pexels: {len(pexels_paths)} videos downloaded")
                
            except Exception as e:
                logger.warning(f"Pexels download failed: {e}")
        
        # 3. Pixabay
        if self.config.enable_pixabay:
            try:
                from .pixabay import download_pixabay_footage
                
                stock_dir = self.output_dir / "stock"
                pixabay_paths, pixabay_counts = download_pixabay_footage(
                    keywords=keywords,
                    output_dir=str(stock_dir),
                    per_keyword=self.config.pixabay_per_keyword
                )
                
                self.downloads_by_source["pixabay"] = pixabay_paths
                for k, count in pixabay_counts.items():
                    total_counts[k] = total_counts.get(k, 0) + count
                    
                logger.info(f"Pixabay: {len(pixabay_paths)} videos downloaded")
                
            except Exception as e:
                logger.warning(f"Pixabay download failed: {e}")
        
        # 4. Images
        if self.config.enable_images:
            try:
                from .imagedl import download_images
                
                images_dir = self.output_dir / "images"
                image_paths, image_counts = download_images(
                    keywords=keywords,
                    output_dir=str(images_dir),
                    per_keyword=self.config.images_per_keyword,
                    min_size_mb=self.config.image_min_size_mb
                )
                
                self.downloads_by_source["images"] = image_paths
                logger.info(f"Images: {len(image_paths)} images downloaded (>={self.config.image_min_size_mb}MB)")
                
            except Exception as e:
                logger.warning(f"Image download failed: {e}")
        
        # Track failed keywords
        self.failed_keywords = [k for k, count in total_counts.items() if count == 0]
        
        return total_counts
    
    def get_stock_footage_paths(self) -> List[str]:
        """Get all stock footage paths (Pexels + Pixabay)"""
        return self.downloads_by_source["pexels"] + self.downloads_by_source["pixabay"]
    
    def get_image_paths(self) -> List[str]:
        """Get all downloaded image paths"""
        return self.downloads_by_source["images"]


class ConfidenceEnforcementLoop:
    """
    Main loop for enforcing confidence and remixing keywords.
    """
    
    def __init__(
        self,
        config: EnhancedPipelineConfig,
        remixer=None
    ):
        self.config = config
        self.remixer = remixer
        
        self.retry_counts: Dict[str, int] = {}
        self.all_remixes: Dict[str, List[str]] = {}
        self.final_matches: List[Dict] = []
    
    def run_with_enforcement(
        self,
        keywords: List[str],
        download_func,
        match_func,
        max_iterations: int = 3
    ) -> Tuple[List[str], List[Dict]]:
        """
        Run download/match loop with confidence enforcement.
        
        Args:
            keywords: Initial keywords
            download_func: Function(keywords) -> download_counts
            match_func: Function() -> matches with confidence
            max_iterations: Maximum retry iterations
        
        Returns:
            Tuple of (final_keywords_used, final_matches)
        """
        current_keywords = keywords.copy()
        all_keywords_tried = set(keywords)
        iteration = 0
        
        while iteration < max_iterations:
            iteration += 1
            logger.info(f"Confidence enforcement iteration {iteration}/{max_iterations}")
            
            # 1. Download
            download_counts = download_func(current_keywords)
            
            # 2. Handle zero downloads
            zero_keywords = [k for k, count in download_counts.items() if count == 0]
            if zero_keywords and self.config.remix_zero_downloads and self.remixer:
                logger.info(f"Remixing {len(zero_keywords)} zero-download keywords...")
                
                new_keywords = []
                for kw in zero_keywords:
                    if self.retry_counts.get(kw, 0) < self.config.max_remix_retries:
                        remixed = self.remixer.remix_for_zero_downloads(
                            kw, 
                            attempt=self.retry_counts.get(kw, 0) + 1
                        )
                        if remixed:
                            new_keywords.extend(remixed)
                            self.all_remixes[kw] = remixed
                        self.retry_counts[kw] = self.retry_counts.get(kw, 0) + 1
                
                # Add new keywords for next iteration
                new_keywords = [k for k in new_keywords if k not in all_keywords_tried]
                if new_keywords:
                    current_keywords = new_keywords
                    all_keywords_tried.update(new_keywords)
                    continue  # Retry download with new keywords
            
            # 3. Match
            matches = match_func()
            
            # 4. Check confidence
            low_conf_matches = [m for m in matches if m.get('confidence', 0) < self.config.min_confidence]
            
            if not low_conf_matches:
                logger.info(f"All matches meet {self.config.min_confidence:.0%} confidence!")
                self.final_matches = matches
                break
            
            # 5. Remix for low confidence
            if self.config.remix_low_confidence and self.remixer:
                logger.warning(f"{len(low_conf_matches)} matches below {self.config.min_confidence:.0%}")
                
                # Get keywords responsible for low matches
                low_keywords = set()
                for m in low_conf_matches:
                    source_kw = m.get('source_keyword', '')
                    if source_kw and self.retry_counts.get(source_kw, 0) < self.config.max_remix_retries:
                        low_keywords.add(source_kw)
                
                if low_keywords:
                    new_keywords = []
                    for kw in low_keywords:
                        # Find a representative low match for context
                        rep_match = next((m for m in low_conf_matches if m.get('source_keyword') == kw), None)
                        
                        remixed = self.remixer.remix_for_low_confidence(
                            keyword=kw,
                            matched_text=rep_match.get('matched_text', '') if rep_match else '',
                            voiceover_text=rep_match.get('voiceover_text', '') if rep_match else '',
                            confidence=rep_match.get('confidence', 0) if rep_match else 0,
                            attempt=self.retry_counts.get(kw, 0) + 1
                        )
                        
                        if remixed:
                            new_keywords.extend(remixed)
                            self.all_remixes[kw] = remixed
                        self.retry_counts[kw] = self.retry_counts.get(kw, 0) + 1
                    
                    new_keywords = [k for k in new_keywords if k not in all_keywords_tried]
                    if new_keywords:
                        current_keywords = new_keywords
                        all_keywords_tried.update(new_keywords)
                        continue
            
            # No more remixes possible
            self.final_matches = matches
            break
        
        return list(all_keywords_tried), self.final_matches


def integrate_enhanced_features(main_pipeline):
    """
    Decorator/mixin to add enhanced features to main pipeline.
    
    Usage:
        pipeline = VoiceoverMatcher(config)
        pipeline = integrate_enhanced_features(pipeline)
        pipeline.run()
    """
    # Add enhanced config
    main_pipeline.enhanced_config = EnhancedPipelineConfig()
    
    # Store original methods
    original_download = getattr(main_pipeline, 'download_footage', None)
    original_match = getattr(main_pipeline, 'match_segments', None)
    
    # Enhanced downloader
    main_pipeline.enhanced_downloader = None
    
    def enhanced_download(keywords):
        """Enhanced download with multiple sources"""
        if main_pipeline.enhanced_downloader is None:
            main_pipeline.enhanced_downloader = EnhancedDownloader(
                main_pipeline.enhanced_config,
                main_pipeline.output_dir
            )
        
        # Call original YouTube download
        if original_download:
            original_download(keywords)
        
        # Add stock footage
        return main_pipeline.enhanced_downloader.download_all_sources(keywords)
    
    main_pipeline.enhanced_download = enhanced_download
    
    return main_pipeline


# Prompt functions for interactive configuration

def prompt_enhanced_settings() -> EnhancedPipelineConfig:
    """Interactive prompt for enhanced pipeline settings"""
    config = EnhancedPipelineConfig()
    
    print("\n" + "=" * 60)
    print("  ENHANCED PIPELINE SETTINGS")
    print("=" * 60)
    
    try:
        # Confidence enforcement
        print("\n  [Confidence Enforcement]")
        conf = input(f"  Minimum confidence % [90]: ").strip()
        if conf:
            config.min_confidence = float(conf) / 100
        
        retries = input(f"  Max remix retries [3]: ").strip()
        if retries:
            config.max_remix_retries = int(retries)
        
        # Stock footage
        print("\n  [Stock Footage Sources]")
        pexels = input("  Enable Pexels? [Y/n]: ").strip().lower()
        config.enable_pexels = pexels != 'n'
        
        pixabay = input("  Enable Pixabay? [Y/n]: ").strip().lower()
        config.enable_pixabay = pixabay != 'n'
        
        # Images
        print("\n  [Image Downloads]")
        images = input("  Enable image downloads? [Y/n]: ").strip().lower()
        config.enable_images = images != 'n'
        
        if config.enable_images:
            min_size = input("  Minimum image size MB [1.0]: ").strip()
            if min_size:
                config.image_min_size_mb = float(min_size)
        
        # Multi-style
        print("\n  [Multi-Style OTIO]")
        multi = input("  Generate multiple OTIO styles? [Y/n]: ").strip().lower()
        config.enable_multi_style = multi != 'n'
        
        if config.enable_multi_style:
            print("  Second style options: strict, stock_heavy, fast_paced, cinematic")
            style = input("  Second style [strict]: ").strip() or "strict"
            config.second_style = style
        
    except (EOFError, KeyboardInterrupt):
        print("\n  Using defaults")
    
    return config


def prompt_topic_context() -> str:
    """Prompt for topic context to improve keyword generation"""
    print("\n" + "-" * 40)
    print("  Topic Context (for better keyword remixing)")
    print("-" * 40)
    print("  Example: 'Hawaii Kilauea volcano eruption 2018'")
    print("  This helps generate contextual keywords, not just dates.")
    
    try:
        context = input("\n  Main topic: ").strip()
        if context:
            print(f"  ✓ Topic set: {context}")
        return context
    except (EOFError, KeyboardInterrupt):
        return ""
