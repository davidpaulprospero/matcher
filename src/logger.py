"""
Comprehensive logging system for documentary purposes.
Outputs both human-readable .log and machine-parseable .json files.
"""

import os
import json
import logging
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field, asdict
from contextlib import contextmanager


@dataclass
class APICallLog:
    """Log entry for an API call"""
    provider: str
    endpoint: str
    timestamp: str
    duration_ms: float
    input_tokens_est: int = 0
    output_tokens_est: int = 0
    cost_est_usd: float = 0.0
    success: bool = True
    error: Optional[str] = None


@dataclass
class MatchDecisionLog:
    """Log entry for a match decision"""
    segment_index: int
    voiceover_text: str
    selected_clip: str
    selected_source: str
    confidence: float
    reasoning: str
    embedding_similarity: float
    duration_penalty: float = 0.0
    keyword_boost: float = 0.0
    is_hybrid_match: bool = False
    alternatives_considered: int = 0
    llm_reranked: bool = False


@dataclass
class PerformanceLog:
    """Log entry for performance timing"""
    stage: str
    duration_seconds: float
    items_processed: int = 0
    items_per_second: float = 0.0


@dataclass
class RunLog:
    """Complete log for a single run"""
    run_id: str
    start_time: str
    end_time: str = ""
    duration_seconds: float = 0.0
    
    # Config snapshot
    config_snapshot: Dict[str, Any] = field(default_factory=dict)
    
    # Input summary
    voiceover_file: str = ""
    voiceover_segments: int = 0
    video_files: int = 0
    video_segments: int = 0
    
    # Processing stats
    performance: List[PerformanceLog] = field(default_factory=list)
    api_calls: List[APICallLog] = field(default_factory=list)
    match_decisions: List[MatchDecisionLog] = field(default_factory=list)
    
    # Summary stats
    total_api_calls: int = 0
    total_api_cost_est_usd: float = 0.0
    avg_confidence: float = 0.0
    high_confidence_matches: int = 0
    low_confidence_matches: int = 0
    gaps_found: int = 0
    
    # Warnings and errors
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    
    # Output files
    output_files: List[str] = field(default_factory=list)


class PipelineLogger:
    """
    Comprehensive logger for the voiceover-matcher pipeline.
    Outputs both .log (human readable) and .json (machine parseable) files.
    """
    
    # Estimated costs per 1K tokens (as of late 2024)
    API_COSTS = {
        'gemini-2.0-flash': {'input': 0.000075, 'output': 0.0003},
        'gemini-1.5-flash': {'input': 0.000075, 'output': 0.0003},
        'claude-3-haiku': {'input': 0.00025, 'output': 0.00125},
        'claude-3-sonnet': {'input': 0.003, 'output': 0.015},
        'text-embedding-004': {'input': 0.00001, 'output': 0},
        'voyage-2': {'input': 0.0001, 'output': 0},
        'local': {'input': 0, 'output': 0},
    }
    
    def __init__(self, log_dir: str = "./logs", project_name: str = "voiceover_match"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate run ID
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.run_id = f"{project_name}_{timestamp}"
        
        # Initialize run log
        self.run_log = RunLog(
            run_id=self.run_id,
            start_time=datetime.now().isoformat()
        )
        
        # Setup file paths
        self.log_file = self.log_dir / f"{self.run_id}.log"
        self.json_file = self.log_dir / f"{self.run_id}.json"
        
        # Setup Python logger for .log file
        self._setup_file_logger()
        
        # Timing trackers
        self._stage_timers: Dict[str, float] = {}
        self._api_call_count = 0
        
    def _setup_file_logger(self):
        """Setup file logger for human-readable output"""
        self.file_logger = logging.getLogger(f"pipeline_{self.run_id}")
        self.file_logger.setLevel(logging.DEBUG)
        
        # Remove any existing handlers
        self.file_logger.handlers = []
        
        # File handler
        fh = logging.FileHandler(self.log_file, encoding='utf-8')
        fh.setLevel(logging.DEBUG)
        
        # Format
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        fh.setFormatter(formatter)
        self.file_logger.addHandler(fh)
        
        # Write header
        self.file_logger.info("=" * 80)
        self.file_logger.info("VOICEOVER-TO-FOOTAGE MATCHER - RUN LOG")
        self.file_logger.info(f"Run ID: {self.run_id}")
        self.file_logger.info(f"Started: {self.run_log.start_time}")
        self.file_logger.info("=" * 80)
    
    def log_config(self, config: Any):
        """Log the configuration snapshot"""
        # Convert config to dict
        if hasattr(config, '_to_dict'):
            config_dict = config._to_dict()
        elif hasattr(config, '__dict__'):
            config_dict = self._serialize_config(config)
        else:
            config_dict = dict(config)
        
        # Remove sensitive data
        safe_config = {k: v for k, v in config_dict.items() 
                       if not k.endswith('_api_key')}
        
        self.run_log.config_snapshot = safe_config
        
        self.file_logger.info("-" * 40)
        self.file_logger.info("CONFIGURATION")
        self.file_logger.info("-" * 40)
        for key, value in safe_config.items():
            if isinstance(value, dict):
                self.file_logger.info(f"  {key}:")
                for k, v in value.items():
                    self.file_logger.info(f"    {k}: {v}")
            else:
                self.file_logger.info(f"  {key}: {value}")
    
    def _serialize_config(self, obj) -> dict:
        """Recursively serialize config object to dict"""
        result = {}
        for key, value in obj.__dict__.items():
            if key.startswith('_'):
                continue
            if hasattr(value, '__dict__') and not isinstance(value, (str, int, float, bool, list, dict)):
                result[key] = self._serialize_config(value)
            else:
                result[key] = value
        return result
    
    def log_input_summary(self, voiceover_file: str, voiceover_segments: int,
                          video_files: int, video_segments: int):
        """Log input file summary"""
        self.run_log.voiceover_file = voiceover_file
        self.run_log.voiceover_segments = voiceover_segments
        self.run_log.video_files = video_files
        self.run_log.video_segments = video_segments
        
        self.file_logger.info("-" * 40)
        self.file_logger.info("INPUT SUMMARY")
        self.file_logger.info("-" * 40)
        self.file_logger.info(f"  Voiceover: {voiceover_file}")
        self.file_logger.info(f"  Voiceover segments: {voiceover_segments}")
        self.file_logger.info(f"  Video files: {video_files}")
        self.file_logger.info(f"  Video segments: {video_segments}")
    
    @contextmanager
    def time_stage(self, stage_name: str, items_count: int = 0):
        """Context manager to time a pipeline stage"""
        start = time.time()
        self.file_logger.info(f"\n[STAGE] {stage_name} started...")
        
        try:
            yield
        finally:
            duration = time.time() - start
            items_per_sec = items_count / duration if duration > 0 and items_count > 0 else 0
            
            perf_log = PerformanceLog(
                stage=stage_name,
                duration_seconds=round(duration, 2),
                items_processed=items_count,
                items_per_second=round(items_per_sec, 2)
            )
            self.run_log.performance.append(perf_log)
            
            self.file_logger.info(
                f"[STAGE] {stage_name} completed in {duration:.2f}s "
                f"({items_count} items, {items_per_sec:.1f}/sec)"
            )
    
    def log_api_call(self, provider: str, endpoint: str, 
                     input_text_len: int = 0, output_text_len: int = 0,
                     duration_ms: float = 0, success: bool = True, error: str = None):
        """Log an API call with estimated cost"""
        
        # Estimate tokens (rough: 4 chars = 1 token)
        input_tokens = input_text_len // 4
        output_tokens = output_text_len // 4
        
        # Estimate cost
        cost_info = self.API_COSTS.get(provider, self.API_COSTS['local'])
        cost = (input_tokens / 1000 * cost_info['input'] + 
                output_tokens / 1000 * cost_info['output'])
        
        api_log = APICallLog(
            provider=provider,
            endpoint=endpoint,
            timestamp=datetime.now().isoformat(),
            duration_ms=duration_ms,
            input_tokens_est=input_tokens,
            output_tokens_est=output_tokens,
            cost_est_usd=round(cost, 6),
            success=success,
            error=error
        )
        self.run_log.api_calls.append(api_log)
        self._api_call_count += 1
        
        # Only log errors/warnings, not every call (too verbose)
        if not success:
            self.file_logger.warning(f"API call failed: {provider}/{endpoint} - {error}")
    
    def log_match_decision(self, segment_index: int, voiceover_text: str,
                           selected_clip: str, selected_source: str,
                           confidence: float, reasoning: str,
                           embedding_similarity: float = 0,
                           duration_penalty: float = 0,
                           keyword_boost: float = 0,
                           is_hybrid_match: bool = False,
                           alternatives_considered: int = 0,
                           llm_reranked: bool = False):
        """Log a match decision"""
        
        match_log = MatchDecisionLog(
            segment_index=segment_index,
            voiceover_text=voiceover_text[:100] + "..." if len(voiceover_text) > 100 else voiceover_text,
            selected_clip=selected_clip,
            selected_source=selected_source,
            confidence=round(confidence, 3),
            reasoning=reasoning,
            embedding_similarity=round(embedding_similarity, 3),
            duration_penalty=round(duration_penalty, 3),
            keyword_boost=round(keyword_boost, 3),
            is_hybrid_match=is_hybrid_match,
            alternatives_considered=alternatives_considered,
            llm_reranked=llm_reranked
        )
        self.run_log.match_decisions.append(match_log)
        
        # Log to file
        confidence_tier = "HIGH" if confidence >= 0.8 else "GOOD" if confidence >= 0.6 else "MED" if confidence >= 0.4 else "LOW"
        self.file_logger.debug(
            f"Match #{segment_index}: [{confidence_tier}] {confidence:.0%} | "
            f"{Path(selected_source).stem} @ {selected_clip} | "
            f"emb={embedding_similarity:.2f} kw_boost={keyword_boost:+.2f} dur_pen={duration_penalty:-.2f}"
        )
    
    def log_warning(self, message: str):
        """Log a warning"""
        self.run_log.warnings.append(message)
        self.file_logger.warning(message)
    
    def log_error(self, message: str):
        """Log an error"""
        self.run_log.errors.append(message)
        self.file_logger.error(message)
    
    def log_info(self, message: str):
        """Log info message"""
        self.file_logger.info(message)
    
    def log_debug(self, message: str):
        """Log debug message"""
        self.file_logger.debug(message)
    
    def log_output_file(self, filepath: str):
        """Log an output file"""
        self.run_log.output_files.append(filepath)
        self.file_logger.info(f"Output: {filepath}")
    
    def finalize(self, match_results: List = None):
        """Finalize the log with summary statistics"""
        
        # Calculate end time and duration
        self.run_log.end_time = datetime.now().isoformat()
        start = datetime.fromisoformat(self.run_log.start_time)
        end = datetime.fromisoformat(self.run_log.end_time)
        self.run_log.duration_seconds = (end - start).total_seconds()
        
        # Calculate API stats
        self.run_log.total_api_calls = len(self.run_log.api_calls)
        self.run_log.total_api_cost_est_usd = sum(c.cost_est_usd for c in self.run_log.api_calls)
        
        # Calculate match stats
        if match_results:
            confidences = [m.primary_match.confidence for m in match_results]
            self.run_log.avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            self.run_log.high_confidence_matches = sum(1 for c in confidences if c >= 0.7)
            self.run_log.low_confidence_matches = sum(1 for c in confidences if c < 0.5)
            self.run_log.gaps_found = sum(1 for m in match_results if m.has_gap)
        
        # Write summary to log file
        self.file_logger.info("\n" + "=" * 80)
        self.file_logger.info("RUN SUMMARY")
        self.file_logger.info("=" * 80)
        self.file_logger.info(f"  Duration: {self.run_log.duration_seconds:.1f} seconds")
        self.file_logger.info(f"  Total API calls: {self.run_log.total_api_calls}")
        self.file_logger.info(f"  Estimated API cost: ${self.run_log.total_api_cost_est_usd:.4f}")
        self.file_logger.info(f"  Average confidence: {self.run_log.avg_confidence:.1%}")
        self.file_logger.info(f"  High confidence (≥70%): {self.run_log.high_confidence_matches}")
        self.file_logger.info(f"  Low confidence (<50%): {self.run_log.low_confidence_matches}")
        self.file_logger.info(f"  Gaps found: {self.run_log.gaps_found}")
        self.file_logger.info(f"  Warnings: {len(self.run_log.warnings)}")
        self.file_logger.info(f"  Errors: {len(self.run_log.errors)}")
        
        # Performance breakdown
        self.file_logger.info("\n" + "-" * 40)
        self.file_logger.info("PERFORMANCE BREAKDOWN")
        self.file_logger.info("-" * 40)
        for perf in self.run_log.performance:
            self.file_logger.info(
                f"  {perf.stage}: {perf.duration_seconds:.1f}s "
                f"({perf.items_processed} items)"
            )
        
        # Output files
        self.file_logger.info("\n" + "-" * 40)
        self.file_logger.info("OUTPUT FILES")
        self.file_logger.info("-" * 40)
        for filepath in self.run_log.output_files:
            self.file_logger.info(f"  {filepath}")
        
        # Add log files themselves
        self.file_logger.info(f"  {self.log_file}")
        self.file_logger.info(f"  {self.json_file}")
        
        self.file_logger.info("\n" + "=" * 80)
        self.file_logger.info("END OF LOG")
        self.file_logger.info("=" * 80)
        
        # Write JSON log
        self._write_json_log()
        
        return str(self.log_file), str(self.json_file)
    
    def _write_json_log(self):
        """Write the complete log as JSON"""
        
        # Convert dataclasses to dicts
        log_dict = {
            'run_id': self.run_log.run_id,
            'start_time': self.run_log.start_time,
            'end_time': self.run_log.end_time,
            'duration_seconds': self.run_log.duration_seconds,
            'config_snapshot': self.run_log.config_snapshot,
            'input': {
                'voiceover_file': self.run_log.voiceover_file,
                'voiceover_segments': self.run_log.voiceover_segments,
                'video_files': self.run_log.video_files,
                'video_segments': self.run_log.video_segments,
            },
            'performance': [asdict(p) for p in self.run_log.performance],
            'api_calls': [asdict(a) for a in self.run_log.api_calls],
            'match_decisions': [asdict(m) for m in self.run_log.match_decisions],
            'summary': {
                'total_api_calls': self.run_log.total_api_calls,
                'total_api_cost_est_usd': self.run_log.total_api_cost_est_usd,
                'avg_confidence': self.run_log.avg_confidence,
                'high_confidence_matches': self.run_log.high_confidence_matches,
                'low_confidence_matches': self.run_log.low_confidence_matches,
                'gaps_found': self.run_log.gaps_found,
            },
            'warnings': self.run_log.warnings,
            'errors': self.run_log.errors,
            'output_files': self.run_log.output_files,
        }
        
        with open(self.json_file, 'w', encoding='utf-8') as f:
            json.dump(log_dict, f, indent=2, ensure_ascii=False)


# Global logger instance (set by main.py)
_pipeline_logger: Optional[PipelineLogger] = None


def get_logger() -> Optional[PipelineLogger]:
    """Get the global pipeline logger"""
    return _pipeline_logger


def set_logger(logger: PipelineLogger):
    """Set the global pipeline logger"""
    global _pipeline_logger
    _pipeline_logger = logger


# Alias for backward compatibility
RunLogger = PipelineLogger
