"""
Caption fetcher timeout and rate limit management.

Provides adaptive timeout escalation, global rate limiting coordination,
and per-format timeout policies to improve caption fetch reliability.

Example:
    # Rate limit tracking across parallel workers
    tracker = RateLimitTracker()
    
    # Check if we should pause before fetching
    if tracker.should_pause():
        time.sleep(tracker.get_recommended_delay())
    
    try:
        result = fetch_captions(video_id)
    except RateLimitError:
        tracker.record_rate_limit()
        
    # Progressive timeout escalation
    timeout_mgr = ProgressiveTimeoutManager(base_timeout=30)
    
    # First attempt: 30s
    # After timeout: escalate to 45s
    # After 2nd timeout: escalate to 60s (max)
    timeout = timeout_mgr.get_timeout_for_attempt(attempt)
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class TimeoutEscalationLevel(Enum):
    """Timeout escalation levels based on consecutive failures."""
    NORMAL = auto()      # Base timeout
    ELEVATED = auto()    # 1.5x base after 1 timeout
    HIGH = auto()        # 2x base after 2 timeouts
    MAXIMUM = auto()     # 3x base (cap) after 3+ timeouts


@dataclass
class RateLimitState:
    """Global rate limit state shared across workers.
    
    Tracks rate limiting signals to coordinate backoff across
    parallel caption fetch workers.
    
    Attributes:
        last_rate_limit_time: Timestamp of most recent rate limit
        consecutive_rate_limits: Count of consecutive rate limit errors
        total_rate_limits: Total rate limit hits in current session
        global_backoff_until: Unix timestamp when global backoff ends
        rate_limit_history: Recent rate limit events (last 10)
    """
    last_rate_limit_time: float = 0.0
    consecutive_rate_limits: int = 0
    total_rate_limits: int = 0
    global_backoff_until: float = 0.0
    rate_limit_history: List[Tuple[float, str]] = field(default_factory=list)
    
    # Configuration
    max_history: int = 10
    base_backoff_seconds: float = 5.0
    max_backoff_seconds: float = 300.0  # 5 minutes max
    
    def record_rate_limit(self, video_id: str = "") -> float:
        """Record a rate limit event and return recommended delay.
        
        Args:
            video_id: Video ID that triggered the rate limit
            
        Returns:
            Recommended delay in seconds before next request
        """
        now = time.time()
        self.last_rate_limit_time = now
        self.consecutive_rate_limits += 1
        self.total_rate_limits += 1
        
        # Add to history
        self.rate_limit_history.append((now, video_id))
        if len(self.rate_limit_history) > self.max_history:
            self.rate_limit_history.pop(0)
        
        # Calculate exponential backoff
        # 1st: 5s, 2nd: 10s, 3rd: 20s, 4th: 40s, max: 300s
        delay = min(
            self.base_backoff_seconds * (2 ** (self.consecutive_rate_limits - 1)),
            self.max_backoff_seconds
        )
        
        self.global_backoff_until = now + delay
        
        logger.warning(
            f"Rate limit recorded for {video_id or 'unknown'}: "
            f"consecutive={self.consecutive_rate_limits}, "
            f"recommended_delay={delay:.1f}s"
        )
        
        return delay
    
    def record_success(self) -> None:
        """Record a successful request to reset consecutive counter."""
        if self.consecutive_rate_limits > 0:
            logger.debug(
                f"Rate limit state reset after success "
                f"(was {self.consecutive_rate_limits} consecutive)"
            )
        self.consecutive_rate_limits = 0
    
    def should_pause(self) -> bool:
        """Check if requests should pause due to global backoff."""
        return time.time() < self.global_backoff_until
    
    def get_recommended_delay(self) -> float:
        """Get recommended delay before next request.
        
        Returns:
            Seconds to wait (0 if no backoff needed)
        """
        remaining = self.global_backoff_until - time.time()
        return max(0.0, remaining)
    
    def get_rate_limit_ratio(self, window_seconds: float = 60.0) -> float:
        """Calculate rate limit ratio in recent time window.
        
        Args:
            window_seconds: Time window to analyze
            
        Returns:
            Ratio of rate-limited requests (0.0-1.0)
        """
        now = time.time()
        cutoff = now - window_seconds
        recent_hits = sum(1 for t, _ in self.rate_limit_history if t > cutoff)
        
        # Assume at least 10 requests in window for ratio calculation
        total_estimated = max(10, len(self.rate_limit_history))
        return recent_hits / total_estimated
    
    def is_rate_limit_critical(self) -> bool:
        """Check if rate limiting has reached critical levels.
        
        Returns:
            True if >50% rate limit ratio or >5 consecutive hits
        """
        if self.consecutive_rate_limits >= 5:
            return True
        
        if self.get_rate_limit_ratio(60.0) > 0.5:
            return True
        
        return False
    
    def to_dict(self) -> dict:
        """Serialize state to dictionary."""
        return {
            'last_rate_limit_time': self.last_rate_limit_time,
            'consecutive_rate_limits': self.consecutive_rate_limits,
            'total_rate_limits': self.total_rate_limits,
            'global_backoff_until': self.global_backoff_until,
            'rate_limit_history': self.rate_limit_history,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'RateLimitState':
        """Restore state from dictionary."""
        state = cls()
        state.last_rate_limit_time = data.get('last_rate_limit_time', 0.0)
        state.consecutive_rate_limits = data.get('consecutive_rate_limits', 0)
        state.total_rate_limits = data.get('total_rate_limits', 0)
        state.global_backoff_until = data.get('global_backoff_until', 0.0)
        state.rate_limit_history = data.get('rate_limit_history', [])
        return state


class RateLimitTracker:
    """Thread-safe global rate limit tracker for parallel workers.
    
    Coordinates rate limiting across multiple parallel caption fetch
    workers to prevent overwhelming YouTube's API.
    
    Example:
        tracker = RateLimitTracker()
        
        # In worker thread
        with tracker.check_rate_limit():
            result = fetch_captions(video_id)
    """
    
    _instance: Optional['RateLimitTracker'] = None
    _lock: threading.Lock = threading.Lock()
    
    def __new__(cls, **kwargs) -> 'RateLimitTracker':
        """Singleton pattern for global state sharing."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(
        self,
        base_backoff_seconds: float = 5.0,
        max_backoff_seconds: float = 300.0,
        max_history: int = 10,
    ):
        if getattr(self, '_initialized', False):
            return

        self._state = RateLimitState(
            base_backoff_seconds=base_backoff_seconds,
            max_backoff_seconds=max_backoff_seconds,
            max_history=max_history,
        )
        self._state_lock = threading.RLock()
        self._initialized = True
    
    def should_pause(self) -> bool:
        """Check if requests should pause."""
        with self._state_lock:
            return self._state.should_pause()
    
    def get_recommended_delay(self) -> float:
        """Get recommended delay before next request."""
        with self._state_lock:
            return self._state.get_recommended_delay()
    
    def record_rate_limit(self, video_id: str = "") -> float:
        """Record a rate limit event.
        
        Returns:
            Recommended delay in seconds
        """
        with self._state_lock:
            return self._state.record_rate_limit(video_id)
    
    def record_success(self) -> None:
        """Record a successful request."""
        with self._state_lock:
            self._state.record_success()
    
    def get_state_summary(self) -> dict:
        """Get current state summary for metrics/logging."""
        with self._state_lock:
            return {
                'consecutive_rate_limits': self._state.consecutive_rate_limits,
                'total_rate_limits': self._state.total_rate_limits,
                'should_pause': self._state.should_pause(),
                'recommended_delay': self._state.get_recommended_delay(),
                'rate_limit_ratio_60s': self._state.get_rate_limit_ratio(60.0),
                'is_critical': self._state.is_rate_limit_critical(),
            }
    
    def is_critical(self) -> bool:
        """Check if rate limiting is at critical levels."""
        with self._state_lock:
            return self._state.is_rate_limit_critical()
    
    def wait_if_needed(self) -> float:
        """Wait if backoff is in effect.
        
        Returns:
            Seconds waited
        """
        delay = self.get_recommended_delay()
        if delay > 0:
            logger.info(f"Rate limit backoff: waiting {delay:.1f}s")
            time.sleep(delay)
        return delay


@dataclass
class TimeoutEscalationPolicy:
    """Policy for escalating timeouts on consecutive failures.
    
    Attributes:
        base_timeout: Starting timeout in seconds
        max_timeout: Maximum allowed timeout
        escalation_multiplier: Multiplier per escalation level
        reset_after_success: Whether to reset on success
    """
    base_timeout: float = 30.0
    max_timeout: float = 120.0
    escalation_multiplier: float = 1.5
    reset_after_success: bool = True
    
    def get_timeout(self, consecutive_failures: int) -> float:
        """Get timeout for given number of consecutive failures.
        
        Args:
            consecutive_failures: Number of consecutive timeout failures
            
        Returns:
            Timeout in seconds
        """
        if consecutive_failures <= 0:
            return self.base_timeout
        
        escalated = self.base_timeout * (self.escalation_multiplier ** consecutive_failures)
        return min(escalated, self.max_timeout)
    
    def get_level(self, consecutive_failures: int) -> TimeoutEscalationLevel:
        """Get escalation level for failure count."""
        if consecutive_failures <= 0:
            return TimeoutEscalationLevel.NORMAL
        elif consecutive_failures == 1:
            return TimeoutEscalationLevel.ELEVATED
        elif consecutive_failures == 2:
            return TimeoutEscalationLevel.HIGH
        else:
            return TimeoutEscalationLevel.MAXIMUM


class ProgressiveTimeoutManager:
    """Manages progressive timeout escalation per video.
    
    Tracks timeout failures per video and escalates timeouts
    adaptively. Resets on success.
    
    Example:
        mgr = ProgressiveTimeoutManager(base_timeout=30)
        
        for attempt in range(max_retries):
            timeout = mgr.get_timeout(video_id, attempt)
            try:
                result = fetch_with_timeout(timeout)
                mgr.record_success(video_id)
                break
            except TimeoutError:
                mgr.record_timeout(video_id)
    """
    
    def __init__(self, policy: Optional[TimeoutEscalationPolicy] = None):
        self.policy = policy or TimeoutEscalationPolicy()
        self._video_failures: Dict[str, int] = {}
        self._lock = threading.Lock()
    
    def get_timeout(self, video_id: str, attempt: int = 0) -> float:
        """Get timeout for video on specific attempt.
        
        Args:
            video_id: Video identifier
            attempt: Current attempt number (0-indexed)
        
        Returns:
            Timeout in seconds
        """
        with self._lock:
            consecutive = self._video_failures.get(video_id, 0)
        
        # Add attempt to consecutive for progressive escalation
        total_escalation = min(consecutive + attempt, 3)  # Cap at MAXIMUM level
        return self.policy.get_timeout(total_escalation)
    
    def record_timeout(self, video_id: str) -> int:
        """Record a timeout failure for video.
        
        Returns:
            New consecutive failure count
        """
        with self._lock:
            self._video_failures[video_id] = self._video_failures.get(video_id, 0) + 1
            return self._video_failures[video_id]
    
    def record_success(self, video_id: str) -> None:
        """Record success to reset escalation."""
        if not self.policy.reset_after_success:
            return
        
        with self._lock:
            if video_id in self._video_failures:
                logger.debug(f"Timeout escalation reset for {video_id}")
                del self._video_failures[video_id]
    
    def get_failure_count(self, video_id: str) -> int:
        """Get current failure count for video."""
        with self._lock:
            return self._video_failures.get(video_id, 0)
    
    def reset(self, video_id: Optional[str] = None) -> None:
        """Reset escalation for specific video or all videos."""
        with self._lock:
            if video_id:
                self._video_failures.pop(video_id, None)
            else:
                self._video_failures.clear()


@dataclass
class FormatTimeoutPolicy:
    """Per-format timeout policy for caption fetching.
    
    Different subtitle formats have different characteristics:
    - json3: Larger, structured, may need more time
    - vtt: Smaller, faster to download
    - srt: Similar to vtt
    
    This policy assigns appropriate timeouts per format to avoid
    wasting time on slow formats when faster alternatives exist.
    
    Attributes:
        timeouts: Dict mapping format name to timeout in seconds
        progressive_fallback: Whether to reduce timeout on format fallback
    """
    timeouts: Dict[str, float] = field(default_factory=lambda: {
        'json3': 45.0,   # Structured format, slightly slower
        'srv3': 30.0,    # YouTube's internal format, fast
        'vtt': 25.0,     # Text format, faster
        'srt': 25.0,     # Text format, faster
    })
    progressive_fallback: bool = True
    fallback_reduction: float = 0.8  # Reduce timeout by 20% on fallback
    
    def get_timeout(self, format_name: str, fallback_level: int = 0) -> float:
        """Get timeout for specific format.
        
        Args:
            format_name: Format (json3, vtt, srt, srv3)
            fallback_level: How many formats have been tried before this
            
        Returns:
            Timeout in seconds
        """
        base = self.timeouts.get(format_name, 30.0)
        
        if self.progressive_fallback and fallback_level > 0:
            # Reduce timeout on fallback - faster formats should be faster
            reduction = self.fallback_reduction ** fallback_level
            return base * reduction
        
        return base


class StalledOperationDetector:
    """Detects stalled operations that aren't making progress.
    
    Unlike simple timeouts, this detects when an operation is
    running but not producing output (stalled).
    
    Example:
        detector = StalledOperationDetector(timeout_seconds=30)
        detector.start()
    
        for line in subprocess_output:
            detector.update_progress()
      
        if detector.is_stalled():
            raise TimeoutError("Operation stalled")
    """
    
    def __init__(
        self,
        timeout_seconds: float = 30.0,
        check_interval: float = 5.0,
        progress_indicator: Optional[str] = None
    ):
        self.timeout_seconds = timeout_seconds
        self.check_interval = check_interval
        self.progress_indicator = progress_indicator or "download"
        
        self._last_progress_time: float = 0.0
        self._start_time: float = 0.0
        self._is_running: bool = False
        self._lock = threading.Lock()
    
    def start(self) -> None:
        """Start monitoring."""
        now = time.time()
        with self._lock:
            self._start_time = now
            self._last_progress_time = now
            self._is_running = True
    
    def update_progress(self) -> None:
        """Call when progress is detected."""
        with self._lock:
            if self._is_running:
                self._last_progress_time = time.time()
    
    def is_stalled(self) -> bool:
        """Check if operation appears stalled.
        
        Returns:
            True if no progress for timeout_seconds
        """
        with self._lock:
            if not self._is_running:
                return False
            
            elapsed_since_progress = time.time() - self._last_progress_time
            return elapsed_since_progress > self.timeout_seconds
    
    def elapsed(self) -> float:
        """Get total elapsed time since start."""
        with self._lock:
            if not self._is_running:
                return 0.0
            return time.time() - self._start_time
    
    def stop(self) -> None:
        """Stop monitoring."""
        with self._lock:
            self._is_running = False
    
    def get_status(self) -> dict:
        """Get current status."""
        with self._lock:
            elapsed = time.time() - self._start_time if self._is_running else 0.0
            since_progress = time.time() - self._last_progress_time if self._is_running else 0.0
            
            return {
                'is_running': self._is_running,
                'elapsed_seconds': elapsed,
                'since_last_progress': since_progress,
                'is_stalled': self.is_stalled() if self._is_running else False,
                'timeout': self.timeout_seconds,
            }
