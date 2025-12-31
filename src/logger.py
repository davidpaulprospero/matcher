"""
Enhanced Logging System v3.0

Chain-of-thought: Comprehensive logging for debugging and auditing
Reasoning: Dual output (file + JSON) enables human and machine consumption
Decision: Track API costs, match decisions, and config access patterns

Features:
- Dual output: Human-readable .log + machine-parseable .json
- API cost tracking with built-in pricing table
- Match decision logging with confidence tiers
- Config access logging and hardcoded value warnings
- Performance metrics (stage timing, throughput)
- Hot-reload detection logging

Usage:
    from src.logger import RunLogger, log_config_access
    
    logger = RunLogger(log_dir="logs")
    logger.log_api_call("gemini", "embed", tokens=1000)
    logger.log_match_decision(segment_idx=0, clip="video.mp4", confidence=0.85)
"""

import os
import json
import logging
import time
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any
from contextlib import contextmanager
import threading

# Optional numpy import for JSON encoding
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    np = None
    HAS_NUMPY = False


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles numpy types"""
    def default(self, obj):
        if HAS_NUMPY:
            if isinstance(obj, (np.integer, np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.bool_):
                return bool(obj)
        return super().default(obj)

# =============================================================================
# API COST TRACKING
# =============================================================================

# Pricing per 1K tokens (as of late 2024)
API_PRICING = {
    'gemini-2.0-flash': {'input': 0.000075, 'output': 0.0003},
    'gemini-1.5-flash': {'input': 0.000075, 'output': 0.0003},
    'gemini-1.5-pro': {'input': 0.00125, 'output': 0.005},
    'claude-3-haiku': {'input': 0.00025, 'output': 0.00125},
    'claude-3-sonnet': {'input': 0.003, 'output': 0.015},
    'claude-3-opus': {'input': 0.015, 'output': 0.075},
    'text-embedding-004': {'input': 0.00001, 'output': 0},
    'voyage-2': {'input': 0.0001, 'output': 0},
    'local': {'input': 0, 'output': 0},
}


def estimate_tokens(text: str) -> int:
    """Estimate token count (rough: 4 chars ≈ 1 token)"""
    return len(text) // 4


def get_api_cost(model: str, input_tokens: int, output_tokens: int = 0) -> float:
    """Calculate estimated API cost"""
    pricing = API_PRICING.get(model, API_PRICING.get('local', {'input': 0, 'output': 0}))
    
    input_cost = (input_tokens / 1000) * pricing['input']
    output_cost = (output_tokens / 1000) * pricing['output']
    
    return input_cost + output_cost


# =============================================================================
# LOG DATA STRUCTURES
# =============================================================================

@dataclass
class APICallLog:
    """Log entry for API call"""
    provider: str
    endpoint: str
    model: str
    timestamp: str
    duration_ms: float
    input_tokens_est: int
    output_tokens_est: int
    cost_est_usd: float
    success: bool
    error: Optional[str] = None
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class MatchDecisionLog:
    """Log entry for match decision"""
    segment_index: int
    voiceover_text: str
    selected_clip: str
    confidence: float
    confidence_tier: str  # HIGH, GOOD, MEDIUM, LOW, GAP
    reasoning: str
    embedding_similarity: float
    duration_penalty: float
    keyword_boost: float
    entity_boost: float
    is_keyword_match: bool
    is_visual_match: bool
    is_hybrid_match: bool
    alternatives_considered: int
    llm_reranked: bool
    clip_reuse_count: int
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ConfigAccessLog:
    """Log entry for config access"""
    timestamp: str
    component: str
    config_path: str
    value: Any
    source: str  # 'config', 'default', 'hardcoded'


@dataclass
class PerformanceLog:
    """Log entry for performance metrics"""
    stage: str
    duration_seconds: float
    items_processed: int
    items_per_second: float
    memory_mb: Optional[float] = None


@dataclass
class RunLog:
    """Complete run log"""
    run_id: str
    start_time: str
    end_time: Optional[str] = None
    config_hash: str = ""
    config_path: str = ""
    
    # Summaries
    total_segments: int = 0
    total_matches: int = 0
    avg_confidence: float = 0.0
    total_api_calls: int = 0
    total_api_cost_est_usd: float = 0.0
    
    # Download stats
    videos_downloaded: int = 0
    videos_skipped: int = 0  # Already existed
    videos_failed: int = 0
    
    # Transcription stats
    videos_transcribed: int = 0
    transcription_cache_hits: int = 0
    
    # Embedding stats
    embeddings_computed: int = 0
    embedding_cache_hits: int = 0
    
    # Entity media stats
    entity_images_downloaded: int = 0
    entity_videos_downloaded: int = 0
    
    # Stage timings (stage_name -> seconds)
    stage_timings: Dict[str, float] = field(default_factory=dict)
    
    # Files generated
    files_generated: Dict[str, str] = field(default_factory=dict)  # type -> path
    
    # Detailed logs
    performance: List[PerformanceLog] = field(default_factory=list)
    api_calls: List[APICallLog] = field(default_factory=list)
    match_decisions: List[MatchDecisionLog] = field(default_factory=list)
    config_accesses: List[ConfigAccessLog] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    output_files: List[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            'run_id': self.run_id,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'config_hash': self.config_hash,
            'config_path': self.config_path,
            'summary': {
                'total_segments': self.total_segments,
                'total_matches': self.total_matches,
                'avg_confidence': self.avg_confidence,
                'total_api_calls': self.total_api_calls,
                'total_api_cost_est_usd': self.total_api_cost_est_usd,
                'videos_downloaded': self.videos_downloaded,
                'videos_skipped': self.videos_skipped,
                'videos_failed': self.videos_failed,
                'videos_transcribed': self.videos_transcribed,
                'transcription_cache_hits': self.transcription_cache_hits,
                'embeddings_computed': self.embeddings_computed,
                'embedding_cache_hits': self.embedding_cache_hits,
                'entity_images_downloaded': self.entity_images_downloaded,
                'entity_videos_downloaded': self.entity_videos_downloaded,
            },
            'stage_timings': self.stage_timings,
            'files_generated': self.files_generated,
            'performance': [asdict(p) for p in self.performance],
            'api_calls': [c.to_dict() for c in self.api_calls],
            'match_decisions': [m.to_dict() for m in self.match_decisions],
            'config_accesses': [asdict(c) for c in self.config_accesses],
            'warnings': self.warnings,
            'errors': self.errors,
            'output_files': self.output_files,
        }


# =============================================================================
# CONFIDENCE TIER HELPERS
# =============================================================================

def get_confidence_tier(confidence: float) -> str:
    """Get confidence tier label"""
    if confidence >= 0.80:
        return "HIGH"
    elif confidence >= 0.60:
        return "GOOD"
    elif confidence >= 0.40:
        return "MEDIUM"
    elif confidence >= 0.20:
        return "LOW"
    else:
        return "GAP"


def get_confidence_color(confidence: float) -> str:
    """Get confidence color for display"""
    tier = get_confidence_tier(confidence)
    colors = {
        "HIGH": "GREEN",
        "GOOD": "CYAN",
        "MEDIUM": "YELLOW",
        "LOW": "ORANGE",
        "GAP": "RED"
    }
    return colors.get(tier, "WHITE")


# =============================================================================
# RUN LOGGER
# =============================================================================

class RunLogger:
    """
    Comprehensive run logger with dual output.
    
    Creates:
    - Human-readable .log file
    - Machine-parseable .json file
    """
    
    def __init__(self, log_dir: str = "./logs", run_id: str = None):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate run ID
        if run_id is None:
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self.run_id = run_id
        self.start_time = datetime.now()
        
        # File paths
        self.log_file = self.log_dir / f"run_{run_id}.log"
        self.json_file = self.log_dir / f"run_{run_id}.json"
        
        # Initialize run log
        self.run_log = RunLog(
            run_id=run_id,
            start_time=self.start_time.isoformat()
        )
        
        # Setup file logger
        self.file_logger = self._setup_file_logger()
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Stage timing
        self._stage_start_times = {}
        
        # Config tracking
        self._config_warnings_issued = set()
    
    def _setup_file_logger(self) -> logging.Logger:
        """Setup file logger"""
        logger = logging.getLogger(f"run_{self.run_id}")
        logger.setLevel(logging.DEBUG)
        
        # File handler
        handler = logging.FileHandler(self.log_file, encoding='utf-8')
        handler.setLevel(logging.DEBUG)
        
        # Format
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(message)s',
            datefmt='%H:%M:%S'
        )
        handler.setFormatter(formatter)
        
        logger.addHandler(handler)
        
        return logger
    
    def log_config(self, config: Any):
        """Log configuration snapshot"""
        self.file_logger.info("=" * 60)
        self.file_logger.info("CONFIGURATION")
        self.file_logger.info("=" * 60)
        
        if hasattr(config, '_config_path'):
            self.run_log.config_path = config._config_path
            self.file_logger.info(f"Config path: {config._config_path}")
        
        if hasattr(config, '_config_hash'):
            self.run_log.config_hash = config._config_hash
            self.file_logger.info(f"Config hash: {config._config_hash}")
        
        if hasattr(config, '_load_time_ms'):
            self.file_logger.info(f"Config load time: {config._load_time_ms:.1f}ms")
        
        # Log key settings
        if hasattr(config, 'matching'):
            mc = config.matching
            self.file_logger.info(f"Matching: min_conf={mc.min_confidence}, "
                                  f"high_conf={mc.high_confidence_threshold}, "
                                  f"candidates={mc.embedding_candidates}")
        
        if hasattr(config, 'transcription'):
            tc = config.transcription
            self.file_logger.info(f"Transcription: model={tc.model}, "
                                  f"workers={tc.max_workers}, "
                                  f"gpu={tc.use_gpu}")
        
        if hasattr(config, 'embedding'):
            ec = config.embedding
            self.file_logger.info(f"Embedding: provider={ec.provider}, "
                                  f"batch_size={ec.batch_size}")
    
    def log_api_call(
        self,
        provider: str,
        endpoint: str,
        model: str = None,
        input_text: str = "",
        output_text: str = "",
        duration_ms: float = 0,
        success: bool = True,
        error: str = None
    ):
        """Log an API call with cost estimation"""
        with self._lock:
            # Estimate tokens
            input_tokens = estimate_tokens(input_text)
            output_tokens = estimate_tokens(output_text)
            
            # Calculate cost
            model_name = model or provider
            cost = get_api_cost(model_name, input_tokens, output_tokens)
            
            # Create log entry
            log_entry = APICallLog(
                provider=provider,
                endpoint=endpoint,
                model=model_name,
                timestamp=datetime.now().isoformat(),
                duration_ms=duration_ms,
                input_tokens_est=input_tokens,
                output_tokens_est=output_tokens,
                cost_est_usd=cost,
                success=success,
                error=error
            )
            
            self.run_log.api_calls.append(log_entry)
            self.run_log.total_api_calls += 1
            self.run_log.total_api_cost_est_usd += cost
            
            # File log
            status = "✓" if success else "✗"
            self.file_logger.info(
                f"API {status} | {provider}/{endpoint} | "
                f"{input_tokens}+{output_tokens} tokens | "
                f"${cost:.6f} | {duration_ms:.0f}ms"
            )
            
            if error:
                self.file_logger.error(f"API Error: {error}")
    
    def log_match_decision(
        self,
        segment_index: int,
        voiceover_text: str,
        selected_clip: str,
        confidence: float,
        reasoning: str = "",
        embedding_similarity: float = 0.0,
        duration_penalty: float = 0.0,
        keyword_boost: float = 0.0,
        entity_boost: float = 0.0,
        is_keyword_match: bool = False,
        is_visual_match: bool = False,
        alternatives_considered: int = 0,
        llm_reranked: bool = False,
        clip_reuse_count: int = 0
    ):
        """Log a match decision"""
        with self._lock:
            tier = get_confidence_tier(confidence)
            
            log_entry = MatchDecisionLog(
                segment_index=segment_index,
                voiceover_text=voiceover_text[:100],
                selected_clip=selected_clip,
                confidence=confidence,
                confidence_tier=tier,
                reasoning=reasoning,
                embedding_similarity=embedding_similarity,
                duration_penalty=duration_penalty,
                keyword_boost=keyword_boost,
                entity_boost=entity_boost,
                is_keyword_match=is_keyword_match,
                is_visual_match=is_visual_match,
                is_hybrid_match=is_keyword_match and is_visual_match,
                alternatives_considered=alternatives_considered,
                llm_reranked=llm_reranked,
                clip_reuse_count=clip_reuse_count
            )
            
            self.run_log.match_decisions.append(log_entry)
            self.run_log.total_matches += 1
            
            # File log
            self.file_logger.info(
                f"Match [{segment_index}] | {tier} ({confidence:.2f}) | "
                f"{Path(selected_clip).name} | "
                f"emb={embedding_similarity:.2f} | "
                f"reuse={clip_reuse_count}"
            )
    
    def log_config_access(
        self,
        component: str,
        config_path: str,
        value: Any,
        source: str = "config"
    ):
        """Log config value access"""
        with self._lock:
            log_entry = ConfigAccessLog(
                timestamp=datetime.now().isoformat(),
                component=component,
                config_path=config_path,
                value=str(value)[:100],
                source=source
            )
            
            self.run_log.config_accesses.append(log_entry)
            
            # Log hardcoded warnings once per path
            if source == "hardcoded":
                warning_key = f"{component}:{config_path}"
                if warning_key not in self._config_warnings_issued:
                    self._config_warnings_issued.add(warning_key)
                    self.file_logger.warning(
                        f"HARDCODED VALUE in {component}: {config_path}={value} - "
                        f"Consider adding to config.yaml"
                    )
                    self.run_log.warnings.append(
                        f"Hardcoded value: {component}.{config_path}={value}"
                    )
    
    def log_hardcoded_warning(self, component: str, value_name: str, value: Any):
        """Convenience method for logging hardcoded value warnings"""
        self.log_config_access(component, value_name, value, source="hardcoded")
    
    @contextmanager
    def stage_timer(self, stage_name: str, item_count: int = 0):
        """Context manager for timing pipeline stages"""
        start_time = time.perf_counter()
        self._stage_start_times[stage_name] = start_time
        
        self.file_logger.info(f"STAGE START: {stage_name}")
        
        try:
            yield
        finally:
            duration = time.perf_counter() - start_time
            items_per_sec = item_count / duration if duration > 0 and item_count > 0 else 0
            
            perf_log = PerformanceLog(
                stage=stage_name,
                duration_seconds=duration,
                items_processed=item_count,
                items_per_second=items_per_sec
            )
            
            with self._lock:
                self.run_log.performance.append(perf_log)
            
            self.file_logger.info(
                f"STAGE END: {stage_name} | {duration:.2f}s | "
                f"{item_count} items | {items_per_sec:.1f}/s"
            )
    
    def log_warning(self, message: str):
        """Log a warning"""
        with self._lock:
            self.run_log.warnings.append(message)
            self.file_logger.warning(message)
    
    def log_error(self, message: str):
        """Log an error"""
        with self._lock:
            self.run_log.errors.append(message)
            self.file_logger.error(message)
    
    def log_output_file(self, file_path: str):
        """Log an output file"""
        with self._lock:
            self.run_log.output_files.append(file_path)
            self.file_logger.info(f"OUTPUT: {file_path}")
    
    def log_config_reload(self, old_hash: str, new_hash: str):
        """Log config reload event"""
        self.file_logger.info(
            f"CONFIG RELOAD: {old_hash} → {new_hash}"
        )
        self.run_log.warnings.append(f"Config reloaded: {old_hash} → {new_hash}")
    
    def log_stage_complete(self, stage_name: str, duration_seconds: float, stats: Dict[str, Any] = None):
        """Log stage completion with timing and stats"""
        with self._lock:
            self.run_log.stage_timings[stage_name] = duration_seconds
            
            # Build stats string
            stats_str = ""
            if stats:
                stats_parts = [f"{k}={v}" for k, v in stats.items()]
                stats_str = f" | {' | '.join(stats_parts)}"
            
            self.file_logger.info(f"✓ {stage_name} [{duration_seconds:.1f}s]{stats_str}")
    
    def update_stats(self, **kwargs):
        """Update run statistics"""
        with self._lock:
            for key, value in kwargs.items():
                if hasattr(self.run_log, key):
                    if isinstance(value, int) and key.endswith('_hits') or key.startswith('videos_') or key.startswith('entity_') or key.startswith('embedding'):
                        # Increment counters
                        current = getattr(self.run_log, key)
                        setattr(self.run_log, key, current + value)
                    else:
                        setattr(self.run_log, key, value)
    
    def set_stats(self, **kwargs):
        """Set run statistics directly (not increment)"""
        with self._lock:
            for key, value in kwargs.items():
                if hasattr(self.run_log, key):
                    setattr(self.run_log, key, value)
    
    def log_file_generated(self, file_type: str, file_path: str):
        """Log a generated file"""
        with self._lock:
            self.run_log.files_generated[file_type] = file_path
            self.run_log.output_files.append(file_path)
    
    def finalize(self):
        """Finalize logging, save JSON, and generate summary files"""
        self.run_log.end_time = datetime.now().isoformat()

        # Calculate summary stats from match decisions if available
        if self.run_log.match_decisions:
            confidences = [m.confidence for m in self.run_log.match_decisions]
            self.run_log.avg_confidence = sum(confidences) / len(confidences)
            if self.run_log.total_matches == 0:
                self.run_log.total_matches = len(self.run_log.match_decisions)

        # Calculate total duration
        try:
            start = datetime.fromisoformat(self.run_log.start_time)
            end = datetime.fromisoformat(self.run_log.end_time)
            total_duration = (end - start).total_seconds()
        except:
            total_duration = sum(self.run_log.stage_timings.values())

        # =================================================================
        # CONSOLE SUMMARY (clean, organized format)
        # =================================================================
        print("\n" + "=" * 60)
        print("  RUN SUMMARY")
        print("=" * 60)

        # Matching results
        print("\n  " + "-" * 56)
        print("  MATCHING RESULTS")
        print("  " + "-" * 56)
        match_pct = (self.run_log.total_matches / max(self.run_log.total_segments, 1)) * 100
        print(f"  Segments matched:    {self.run_log.total_matches}/{self.run_log.total_segments} ({match_pct:.0f}%)")
        print(f"  Average confidence:  {self.run_log.avg_confidence:.1%}")

        # Confidence breakdown
        if self.run_log.match_decisions:
            high = sum(1 for m in self.run_log.match_decisions if m.confidence >= 0.80)
            good = sum(1 for m in self.run_log.match_decisions if 0.60 <= m.confidence < 0.80)
            medium = sum(1 for m in self.run_log.match_decisions if 0.40 <= m.confidence < 0.60)
            low = sum(1 for m in self.run_log.match_decisions if m.confidence < 0.40)
            print(f"  Confidence tiers:    HIGH={high} GOOD={good} MED={medium} LOW={low}")

        # Processing stats
        total_videos = self.run_log.videos_downloaded + self.run_log.videos_skipped + self.run_log.videos_failed
        if total_videos > 0 or self.run_log.videos_transcribed > 0:
            print("\n  " + "-" * 56)
            print("  PROCESSING")
            print("  " + "-" * 56)

            if total_videos > 0:
                print(f"  Videos:     {self.run_log.videos_downloaded:>4} downloaded | {self.run_log.videos_skipped:>4} cached | {self.run_log.videos_failed:>4} failed")

            total_transcribed = self.run_log.videos_transcribed + self.run_log.transcription_cache_hits
            if total_transcribed > 0:
                print(f"  Transcribe: {self.run_log.videos_transcribed:>4} new        | {self.run_log.transcription_cache_hits:>4} cached")

            total_embeddings = self.run_log.embeddings_computed + self.run_log.embedding_cache_hits
            if total_embeddings > 0:
                print(f"  Embeddings: {self.run_log.embeddings_computed:>4} computed   | {self.run_log.embedding_cache_hits:>4} cached")

            if self.run_log.entity_images_downloaded > 0 or self.run_log.entity_videos_downloaded > 0:
                print(f"  Entity:     {self.run_log.entity_images_downloaded:>4} images     | {self.run_log.entity_videos_downloaded:>4} videos")

        # Stage timings
        if self.run_log.stage_timings:
            print("\n  " + "-" * 56)
            print("  STAGE TIMINGS")
            print("  " + "-" * 56)
            for stage, duration in self.run_log.stage_timings.items():
                pct = (duration / total_duration * 100) if total_duration > 0 else 0
                bar_len = int(pct / 5)  # 20 char max bar
                bar = "█" * bar_len + "░" * (20 - bar_len)
                print(f"  {stage:<12} {duration:>6.1f}s  {bar} {pct:>5.1f}%")

        # API usage
        if self.run_log.total_api_calls > 0:
            print("\n  " + "-" * 56)
            print("  API USAGE")
            print("  " + "-" * 56)
            print(f"  Total calls:  {self.run_log.total_api_calls}")
            print(f"  Est. cost:    ${self.run_log.total_api_cost_est_usd:.4f}")

        # Files generated
        if self.run_log.files_generated:
            print("\n  " + "-" * 56)
            print("  FILES GENERATED")
            print("  " + "-" * 56)
            for file_type, path in self.run_log.files_generated.items():
                print(f"  {file_type:<12} {Path(path).name}")

        # Warnings/Errors
        if self.run_log.warnings or self.run_log.errors:
            print("\n  " + "-" * 56)
            print("  ISSUES")
            print("  " + "-" * 56)
            if self.run_log.warnings:
                print(f"  Warnings: {len(self.run_log.warnings)}")
                for w in self.run_log.warnings[:3]:
                    print(f"    • {w[:60]}...")
                if len(self.run_log.warnings) > 3:
                    print(f"    ... and {len(self.run_log.warnings) - 3} more")
            if self.run_log.errors:
                print(f"  Errors: {len(self.run_log.errors)}")
                for e in self.run_log.errors[:3]:
                    print(f"    ✗ {e[:60]}...")

        # Total time
        print("\n  " + "-" * 56)
        mins = int(total_duration // 60)
        secs = total_duration % 60
        if mins > 0:
            print(f"  TOTAL TIME: {mins}m {secs:.1f}s")
        else:
            print(f"  TOTAL TIME: {total_duration:.1f}s")
        print("=" * 60 + "\n")

        # =================================================================
        # FILE LOG (detailed)
        # =================================================================
        self.file_logger.info("=" * 60)
        self.file_logger.info("RUN SUMMARY")
        self.file_logger.info("=" * 60)
        self.file_logger.info(f"Segments: {self.run_log.total_segments}")
        self.file_logger.info(f"Matches: {self.run_log.total_matches}/{self.run_log.total_segments}")
        self.file_logger.info(f"Avg confidence: {self.run_log.avg_confidence:.1%}")

        if total_videos > 0:
            self.file_logger.info(f"Videos: {self.run_log.videos_downloaded} new, {self.run_log.videos_skipped} cached, {self.run_log.videos_failed} failed")

        if self.run_log.videos_transcribed + self.run_log.transcription_cache_hits > 0:
            self.file_logger.info(f"Transcription: {self.run_log.videos_transcribed} new, {self.run_log.transcription_cache_hits} cached")

        if self.run_log.embeddings_computed + self.run_log.embedding_cache_hits > 0:
            self.file_logger.info(f"Embeddings: {self.run_log.embeddings_computed} new, {self.run_log.embedding_cache_hits} cached")

        if self.run_log.entity_images_downloaded > 0 or self.run_log.entity_videos_downloaded > 0:
            self.file_logger.info(f"Entity media: {self.run_log.entity_images_downloaded} images, {self.run_log.entity_videos_downloaded} videos")

        for stage, duration in self.run_log.stage_timings.items():
            self.file_logger.info(f"Stage {stage}: {duration:.1f}s")

        self.file_logger.info(f"API calls: {self.run_log.total_api_calls}")
        self.file_logger.info(f"API cost: ${self.run_log.total_api_cost_est_usd:.4f}")

        for file_type, path in self.run_log.files_generated.items():
            self.file_logger.info(f"File {file_type}: {Path(path).name}")

        self.file_logger.info(f"Warnings: {len(self.run_log.warnings)}")
        self.file_logger.info(f"Errors: {len(self.run_log.errors)}")
        self.file_logger.info(f"Total time: {total_duration:.1f}s")

        # Save JSON
        with open(self.json_file, 'w', encoding='utf-8') as f:
            json.dump(self.run_log.to_dict(), f, indent=2, cls=NumpyEncoder)

        self.file_logger.info(f"JSON log saved: {self.json_file}")

        # Generate LLM-friendly summary files
        self._generate_llm_summary(total_duration)
    
    def _generate_llm_summary(self, total_duration: float):
        """Generate token-friendly summary files for LLMs"""
        log_dir = Path(self.json_file).parent
        base_name = Path(self.json_file).stem
        
        # Markdown summary
        md_path = log_dir / f"{base_name}_summary.md"
        self._write_markdown_summary(md_path, total_duration)
        
        # Plaintext summary (minimal tokens)
        txt_path = log_dir / f"{base_name}_summary.txt"
        self._write_plaintext_summary(txt_path, total_duration)
        
        self.file_logger.info(f"LLM summaries: {md_path.name}, {txt_path.name}")
    
    def _write_markdown_summary(self, path: Path, total_duration: float):
        """Write markdown summary"""
        r = self.run_log
        
        lines = [
            f"# Run Summary: {r.run_id}",
            "",
            "## Overview",
            f"- **Start:** {r.start_time}",
            f"- **Duration:** {total_duration:.1f}s",
            f"- **Config:** {Path(r.config_path).name} ({r.config_hash[:8]})",
            "",
            "## Matching",
            f"- Segments: {r.total_segments}",
            f"- Matched: {r.total_matches}/{r.total_segments} ({r.total_matches/max(r.total_segments,1)*100:.0f}%)",
            f"- Avg Confidence: {r.avg_confidence:.1%}",
            "",
            "## Processing",
            f"- Videos Downloaded: {r.videos_downloaded}",
            f"- Videos Cached: {r.videos_skipped}",
            f"- Videos Failed: {r.videos_failed}",
            f"- Transcribed: {r.videos_transcribed} (cached: {r.transcription_cache_hits})",
            f"- Embeddings: {r.embeddings_computed} (cached: {r.embedding_cache_hits})",
            "",
            "## Entity Media",
            f"- Images: {r.entity_images_downloaded}",
            f"- Videos: {r.entity_videos_downloaded}",
            "",
            "## Stage Timings",
        ]
        
        for stage, duration in r.stage_timings.items():
            lines.append(f"- {stage}: {duration:.1f}s")
        
        lines.extend([
            "",
            "## API Usage",
            f"- Calls: {r.total_api_calls}",
            f"- Cost: ${r.total_api_cost_est_usd:.4f}",
            "",
            "## Files Generated",
        ])
        
        for file_type, file_path in r.files_generated.items():
            lines.append(f"- {file_type}: `{Path(file_path).name}`")
        
        if r.warnings:
            lines.extend(["", "## Warnings"])
            for w in r.warnings[:10]:  # Limit to 10
                lines.append(f"- {w[:100]}")
            if len(r.warnings) > 10:
                lines.append(f"- ... and {len(r.warnings) - 10} more")
        
        if r.errors:
            lines.extend(["", "## Errors"])
            for e in r.errors[:10]:
                lines.append(f"- {e[:100]}")
            if len(r.errors) > 10:
                lines.append(f"- ... and {len(r.errors) - 10} more")
        
        with open(path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
    
    def _write_plaintext_summary(self, path: Path, total_duration: float):
        """Write minimal plaintext summary (token-efficient)"""
        r = self.run_log
        
        lines = [
            f"RUN {r.run_id}",
            f"time={total_duration:.0f}s config={r.config_hash[:8]}",
            f"segments={r.total_segments} matches={r.total_matches} conf={r.avg_confidence:.0%}",
            f"videos: dl={r.videos_downloaded} cache={r.videos_skipped} fail={r.videos_failed}",
            f"transcribe: new={r.videos_transcribed} cache={r.transcription_cache_hits}",
            f"embed: new={r.embeddings_computed} cache={r.embedding_cache_hits}",
            f"entity: img={r.entity_images_downloaded} vid={r.entity_videos_downloaded}",
            f"api: calls={r.total_api_calls} cost=${r.total_api_cost_est_usd:.4f}",
            "stages: " + " ".join(f"{k}={v:.0f}s" for k, v in r.stage_timings.items()),
            f"files: {len(r.files_generated)}",
            f"warn={len(r.warnings)} err={len(r.errors)}",
        ]
        
        with open(path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))


# =============================================================================
# GLOBAL CONFIG ACCESS LOGGING
# =============================================================================

_global_logger: Optional[RunLogger] = None


def set_global_logger(logger: RunLogger):
    """Set the global logger instance"""
    global _global_logger
    _global_logger = logger


def get_global_logger() -> Optional[RunLogger]:
    """Get the global logger instance"""
    return _global_logger


def log_config_access(component: str, config_path: str, value: Any, source: str = "config"):
    """Log config access globally if logger is set"""
    if _global_logger:
        _global_logger.log_config_access(component, config_path, value, source)


def log_hardcoded(component: str, value_name: str, value: Any):
    """Log hardcoded value warning globally"""
    if _global_logger:
        _global_logger.log_hardcoded_warning(component, value_name, value)
    else:
        # Fallback to standard logging
        logging.warning(
            f"HARDCODED VALUE in {component}: {value_name}={value} - "
            f"Consider adding to config.yaml"
        )


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'RunLogger',
    'APICallLog',
    'MatchDecisionLog',
    'ConfigAccessLog',
    'PerformanceLog',
    'RunLog',
    'get_confidence_tier',
    'get_confidence_color',
    'estimate_tokens',
    'get_api_cost',
    'set_global_logger',
    'get_global_logger',
    'log_config_access',
    'log_hardcoded',
]
