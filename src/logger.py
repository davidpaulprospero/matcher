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
import re
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


def _sanitize_run_id(run_id):
    """
    Sanitize a run ID for use in filenames.

    Replaces special characters with underscores and collapses multiple underscores.

    Args:
        run_id: Run ID to sanitize

    Returns:
        Sanitized run ID, or None/empty if input was None/empty
    """
    if run_id is None:
        return None
    if run_id == '':
        return ''

    # Replace special characters with underscores
    result = re.sub(r'[/\:*?"<>|\n\t\r]', '_', run_id)

    # Collapse multiple underscores into single
    result = re.sub(r'_+', '_', result)

    # Strip leading/trailing underscores
    result = result.strip('_')

    # If result is empty or only underscores, return default
    if not result or result == '_':
        return 'unnamed_run'

    return result

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
class MatchDetailLog:
    """Detailed match log with alternatives"""
    segment_index: str  # S001, S002, etc.
    voiceover_text: str
    matched_clip: str
    clip_timecode: str  # 00:12-00:18
    confidence: float
    alternatives: List[Dict[str, Any]] = field(default_factory=list)  # [{clip, confidence}, ...]


@dataclass
class TrackVarietyLog:
    """Track variety statistics"""
    track: str
    unique_sources: int
    most_used: str
    most_used_count: int


@dataclass
class StageLog:
    """Log for a pipeline stage"""
    name: str
    start_time: str
    end_time: str
    duration_seconds: float
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RunLog:
    """Complete run log"""
    run_id: str
    start_time: str
    end_time: Optional[str] = None
    config_hash: str = ""
    config_path: str = ""
    project_name: str = ""

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

    # Embedding stats
    embeddings_computed: int = 0
    embedding_cache_hits: int = 0
    embedding_dimensions: int = 0
    embedding_batches: int = 0
    embedding_rate: float = 0.0

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

    # Verbose logging data
    stages: List[StageLog] = field(default_factory=list)
    match_detail_logs: List[MatchDetailLog] = field(default_factory=list)
    track_variety_logs: List[TrackVarietyLog] = field(default_factory=list)

    # Matching config snapshot
    matching_config: Dict[str, Any] = field(default_factory=dict)

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
        self.verbose_md_file = self.log_dir / f"run_{run_id}_verbose.md"
        
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

            # File log (handle None selected_clip)
            clip_name = Path(selected_clip).name if selected_clip else "NO_CLIP"
            self.file_logger.info(
                f"Match [{segment_index}] | {tier} ({confidence:.2f}) | "
                f"{clip_name} | "
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

    # =========================================================================
    # VERBOSE LOGGING METHODS
    # =========================================================================

    def set_project_name(self, name: str):
        """Set the project name for verbose logs"""
        with self._lock:
            self.run_log.project_name = name

    def log_stage_start(self, stage_name: str):
        """Log stage start for verbose markdown"""
        with self._lock:
            self._current_stage = {
                'name': stage_name,
                'start_time': datetime.now().isoformat(),
                'details': {}
            }

    def log_stage_end(self, stage_name: str, **details):
        """Log stage end with details for verbose markdown"""
        with self._lock:
            if hasattr(self, '_current_stage') and self._current_stage:
                end_time = datetime.now()
                start = datetime.fromisoformat(self._current_stage['start_time'])
                duration = (end_time - start).total_seconds()

                stage_log = StageLog(
                    name=stage_name,
                    start_time=self._current_stage['start_time'],
                    end_time=end_time.isoformat(),
                    duration_seconds=duration,
                    details=details
                )
                self.run_log.stages.append(stage_log)
                self._current_stage = None

    def log_match_detail(
        self,
        segment_index: int,
        voiceover_text: str,
        matched_clip: str,
        start_time: float,
        end_time: float,
        confidence: float,
        alternatives: List[Dict[str, Any]] = None
    ):
        """Log detailed match with alternatives for verbose markdown"""
        with self._lock:
            # Format segment ID
            seg_id = f"S{segment_index + 1:03d}"

            # Format timecode
            def fmt_time(s):
                mins = int(s // 60)
                secs = int(s % 60)
                return f"{mins:02d}:{secs:02d}"

            timecode = f"{fmt_time(start_time)}-{fmt_time(end_time)}"

            self.run_log.match_detail_logs.append(MatchDetailLog(
                segment_index=seg_id,
                voiceover_text=voiceover_text[:50] + "..." if len(voiceover_text) > 50 else voiceover_text,
                matched_clip=Path(matched_clip).name if matched_clip else "",
                clip_timecode=timecode,
                confidence=confidence,
                alternatives=alternatives or []
            ))

    def log_track_variety(self, track: str, unique_sources: int, most_used: str, most_used_count: int):
        """Log track variety stats for verbose markdown"""
        with self._lock:
            self.run_log.track_variety_logs.append(TrackVarietyLog(
                track=track,
                unique_sources=unique_sources,
                most_used=most_used,
                most_used_count=most_used_count
            ))

    def log_embedding_stats(self, total: int, dimensions: int, batches: int, rate: float, duration: float):
        """Log embedding computation stats"""
        with self._lock:
            self.run_log.embeddings_computed = total
            self.run_log.embedding_dimensions = dimensions
            self.run_log.embedding_batches = batches
            self.run_log.embedding_rate = rate

    def log_matching_config(self, config_dict: Dict[str, Any]):
        """Log matching configuration for verbose markdown"""
        with self._lock:
            self.run_log.matching_config = config_dict

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

        # Calculate both wall-clock time and stage time
        try:
            start = datetime.fromisoformat(self.run_log.start_time)
            end = datetime.fromisoformat(self.run_log.end_time)
            wall_clock_time = (end - start).total_seconds()
        except:
            wall_clock_time = sum(self.run_log.stage_timings.values())

        # Stage time (actual work)
        stage_total = sum(self.run_log.stage_timings.values())
        overhead = wall_clock_time - stage_total

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
        if total_videos > 0:
            print("\n  " + "-" * 56)
            print("  PROCESSING")
            print("  " + "-" * 56)

            if total_videos > 0:
                print(f"  Videos:     {self.run_log.videos_downloaded:>4} downloaded | {self.run_log.videos_skipped:>4} cached | {self.run_log.videos_failed:>4} failed")

            total_embeddings = self.run_log.embeddings_computed + self.run_log.embedding_cache_hits
            if total_embeddings > 0:
                print(f"  Embeddings: {self.run_log.embeddings_computed:>4} computed   | {self.run_log.embedding_cache_hits:>4} cached")

            if self.run_log.entity_images_downloaded > 0 or self.run_log.entity_videos_downloaded > 0:
                print(f"  Entity:     {self.run_log.entity_images_downloaded:>4} images     | {self.run_log.entity_videos_downloaded:>4} videos")

        # Stage timings (with overhead shown separately)
        if self.run_log.stage_timings:
            print("\n  " + "-" * 56)
            print(f"  STAGE TIMINGS ({stage_total:.1f}s active, {wall_clock_time:.1f}s total)")
            print("  " + "-" * 56)
            for stage, duration in self.run_log.stage_timings.items():
                # Percentages based on stage time (always sum to 100%)
                pct = (duration / stage_total * 100) if stage_total > 0 else 0
                bar_len = int(pct / 5)  # 20 char max bar
                bar = "█" * bar_len + "░" * (20 - bar_len)
                print(f"  {stage:<12} {duration:>6.1f}s  {bar} {pct:>5.1f}%")

            # Show overhead if significant
            if overhead > 0.1:
                overhead_pct = (overhead / wall_clock_time * 100)
                print(f"\n  Overhead: {overhead:.1f}s ({overhead_pct:.1f}% - checkpoints, I/O)")

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
        mins = int(wall_clock_time // 60)
        secs = wall_clock_time % 60
        if mins > 0:
            print(f"  TOTAL TIME: {mins}m {secs:.1f}s")
        else:
            print(f"  TOTAL TIME: {wall_clock_time:.1f}s")
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
        self.file_logger.info(f"Total time: {wall_clock_time:.1f}s")

        # Save JSON
        with open(self.json_file, 'w', encoding='utf-8') as f:
            json.dump(self.run_log.to_dict(), f, indent=2, cls=NumpyEncoder)

        self.file_logger.info(f"JSON log saved: {self.json_file}")

        # Generate LLM-friendly summary files
        self._generate_llm_summary(wall_clock_time)
    
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

        # Verbose markdown (detailed, structured for LLM parsing)
        self._write_verbose_markdown(self.verbose_md_file, total_duration)

        self.file_logger.info(f"LLM summaries: {md_path.name}, {txt_path.name}, {self.verbose_md_file.name}")
    
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
            f"embed: new={r.embeddings_computed} cache={r.embedding_cache_hits}",
            f"entity: img={r.entity_images_downloaded} vid={r.entity_videos_downloaded}",
            f"api: calls={r.total_api_calls} cost=${r.total_api_cost_est_usd:.4f}",
            "stages: " + " ".join(f"{k}={v:.0f}s" for k, v in r.stage_timings.items()),
            f"files: {len(r.files_generated)}",
            f"warn={len(r.warnings)} err={len(r.errors)}",
        ]
        
        with open(path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))

    def _write_verbose_markdown(self, path: Path, total_duration: float):
        """Write detailed verbose markdown log for LLM parsing"""
        r = self.run_log
        lines = []

        # Helper for time formatting
        def fmt_time(seconds):
            mins = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{mins:02d}:{secs:02d}"

        def fmt_duration(seconds):
            if seconds >= 60:
                return f"{seconds / 60:.1f}m"
            return f"{seconds:.1f}s"

        # Header
        start_dt = datetime.fromisoformat(r.start_time)
        lines.append(f"# Pipeline Run: {start_dt.strftime('%Y-%m-%d %H:%M:%S')}")
        project = r.project_name or "Unknown Project"
        config_hash = r.config_hash[:8] if r.config_hash else "N/A"
        lines.append(f"Project: {project} | Config hash: {config_hash}")
        lines.append("")
        lines.append("---")
        lines.append("")

        # Generate stage sections from stored stages or stage_timings
        stages_to_write = []
        if r.stages:
            for stage in r.stages:
                stages_to_write.append({
                    'name': stage.name,
                    'start_time': stage.start_time,
                    'end_time': stage.end_time,
                    'duration': stage.duration_seconds,
                    'details': stage.details
                })
        else:
            # Fallback to stage_timings
            for name, duration in r.stage_timings.items():
                stages_to_write.append({
                    'name': name.upper(),
                    'start_time': '',
                    'end_time': '',
                    'duration': duration,
                    'details': {}
                })

        # Embeddings section
        if r.embeddings_computed > 0 or r.embedding_cache_hits > 0:
            lines.append("## Embeddings")
            total_emb = r.embeddings_computed + r.embedding_cache_hits
            lines.append(f"- Total: {total_emb:,} vectors ({r.embedding_dimensions} dim)")
            if r.embedding_batches > 0:
                lines.append(f"- Batches: {r.embedding_batches} | Rate: {r.embedding_rate:.0f}/sec")
            lines.append("")
            lines.append("---")
            lines.append("")

        # MATCH Stage
        if r.total_matches > 0 or r.match_detail_logs:
            lines.append("## Stage: MATCH")
            duration = r.stage_timings.get('match', r.stage_timings.get('matching', 0))
            lines.append(f"**Duration:** {fmt_duration(duration)}")
            lines.append("")

            # Config section
            if r.matching_config:
                lines.append("### Config")
                mc = r.matching_config
                lines.append(f"- Candidates: {mc.get('embedding_candidates', 'N/A')} → LLM rerank: {mc.get('llm_rerank_count', 'N/A')}")
                lines.append(f"- Max reuse: {mc.get('max_clip_reuse', 'N/A')} | Face pref: {mc.get('face_preference', 'none')}")
                if mc.get('llm_providers'):
                    lines.append(f"- LLMs: {', '.join(mc.get('llm_providers', []))}")
                lines.append("")

            # Matches table
            if r.match_detail_logs:
                lines.append(f"### Matches ({len(r.match_detail_logs)} total)")
                lines.append("| Seg | Text | Matched Clip | Conf | Alternatives |")
                lines.append("|-----|------|--------------|------|--------------|")
                for md in r.match_detail_logs:
                    # Format alternatives
                    alt_str = ""
                    if md.alternatives:
                        alt_parts = []
                        for alt in md.alternatives[:2]:  # Top 2 alternatives
                            alt_clip = Path(alt.get('clip', '')).name if alt.get('clip') else ''
                            alt_conf = alt.get('confidence', 0)
                            if alt_clip:
                                alt_parts.append(f"{alt_clip[:20]} ({alt_conf:.0%})")
                        alt_str = ", ".join(alt_parts)

                    text = md.voiceover_text.replace("|", "\\|").replace("\n", " ")
                    clip = md.matched_clip[:30] if md.matched_clip else ""
                    lines.append(f"| {md.segment_index} | \"{text}\" | {clip} @ {md.clip_timecode} | {md.confidence:.0%} | {alt_str} |")
            else:
                lines.append(f"- Total matches: {r.total_matches}")
                lines.append(f"- Average confidence: {r.avg_confidence:.1%}")

            lines.append("")

            # Track variety
            if r.track_variety_logs:
                lines.append("### Track Variety")
                lines.append("| Track | Unique Sources | Most Used | Count |")
                lines.append("|-------|----------------|-----------|-------|")
                for tv in r.track_variety_logs:
                    lines.append(f"| {tv.track} | {tv.unique_sources} | {tv.most_used[:30]} | {tv.most_used_count} |")
                lines.append("")

            lines.append("---")
            lines.append("")

        # OUTPUT Stage
        if r.files_generated:
            lines.append("## Stage: OUTPUT")
            duration = r.stage_timings.get('output', r.stage_timings.get('otio', 0))
            lines.append(f"**Duration:** {fmt_duration(duration)}")
            lines.append("")

            lines.append("### Files Generated")
            for file_type, file_path in r.files_generated.items():
                lines.append(f"- ✅ {file_type}: `{Path(file_path).name}`")
            lines.append("")

        # Warnings section
        if r.warnings:
            lines.append("### Warnings")
            for w in r.warnings:
                lines.append(f"- ⚠️ {w}")
            lines.append("")

        # Errors section
        if r.errors:
            lines.append("### Errors")
            for e in r.errors:
                lines.append(f"- ❌ {e}")
            lines.append("")

        lines.append("---")
        lines.append("")

        # Final Summary
        lines.append("## Final Summary")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        lines.append(f"| Total Time | {fmt_duration(total_duration)} |")
        lines.append(f"| Segments | {r.total_segments} |")
        match_rate = (r.total_matches / max(r.total_segments, 1)) * 100
        lines.append(f"| Match Rate | {match_rate:.0f}% |")
        lines.append(f"| Avg Confidence | {r.avg_confidence:.1%} |")
        total_videos = r.videos_downloaded + r.videos_skipped
        lines.append(f"| Videos | {total_videos} |")
        total_emb = r.embeddings_computed + r.embedding_cache_hits
        lines.append(f"| Embeddings | {total_emb:,} |")
        lines.append(f"| Errors | {len(r.errors)} |")
        lines.append(f"| Warnings | {len(r.warnings)} |")
        lines.append("")

        # Write file
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
    'MatchDetailLog',
    'TrackVarietyLog',
    'StageLog',
    'get_confidence_tier',
    'get_confidence_color',
    'estimate_tokens',
    'get_api_cost',
    'set_global_logger',
    'get_global_logger',
    'log_config_access',
    'log_hardcoded',
]
