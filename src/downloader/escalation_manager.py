"""
4-tier escalation manager for yt-dlp bypass orchestration.

Manages per-keyword escalation state across four tiers:
  - Tier 1 (IMPERSONATE_ONLY): --impersonate only (delegated to ImpersonationManager)
  - Tier 2 (EXTRACTOR_ARGS): --impersonate + --extractor-args player_client
  - Tier 3 (FULL_BYPASS): Both + cookie rotation flag
  - Tier 4 (VPN_ROTATION): All above + Mullvad VPN server rotation

On consecutive 403/bot-detection errors, the manager escalates to the next tier.
Escalation is sticky per-session: success resets the 403 counter but does not
de-escalate to a lower tier.

Thread-safe: uses one threading.Lock per keyword for concurrent download access.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from .types import EscalationState, EscalationTier
from .rate_limit_predictor import RateLimitPredictor
from ..logging_templates import log_rate_limit

logger = logging.getLogger(__name__)


# ============== Adaptive Backoff Time-of-Day (US-114-004, US-136-009) ==============

# Time period bins for more granular backoff (US-136-009)
TIME_PERIODS = {
    'night': (0, 6),      # 00:00-05:59 UTC
    'morning': (6, 12),   # 06:00-11:59 UTC
    'afternoon': (12, 18), # 12:00-17:59 UTC
    'evening': (18, 24),  # 18:00-23:59 UTC
}


def get_time_period(hour: int, time_periods: Optional[Dict[str, Tuple[int, int]]] = None) -> str:
    """Get the time period name for a given hour.

    Args:
        hour: Hour of day in UTC (0-23)
        time_periods: Optional custom time periods dict. If None, uses default TIME_PERIODS.

    Returns:
        Time period name: 'night', 'morning', 'afternoon', or 'evening'
    """
    periods = time_periods if time_periods is not None else TIME_PERIODS
    for period, (start, end) in periods.items():
        if start <= hour < end:
            return period
    return 'night'  # Fallback


class AdaptiveBackoffTimeOfDay:
    """Tracks download success rates by hour of day for adaptive backoff.

    This class maintains historical success rate data per hour (0-23 UTC) and
    calculates adaptive backoff multipliers based on time-of-day patterns.

    US-136-009 Enhancements:
    - More granular hour bins (supports 1-hour, 2-hour, 4-hour bins)
    - Weekend vs weekday differentiation
    - Exponential moving average for success rates
    - Configurable time period ranges

    Usage:
        tracker = AdaptiveBackoffTimeOfDay(enabled=True)
        tracker.record_attempt(hour=14, success=True)
        multiplier = tracker.get_time_multiplier(hour=14)

        # With weekend support
        tracker.record_attempt(hour=14, success=True, is_weekend=False)
    """

    def __init__(
        self,
        enabled: bool = True,
        multiplier_range: Tuple[float, float] = (1.0, 2.0),
        low_success_threshold: float = 0.5,
        high_success_threshold: float = 0.8,
        max_data_age_hours: int = 24,
        min_samples_per_hour: int = 5,
        # US-136-009: New parameters
        use_granular_bins: bool = True,
        bin_size_hours: int = 2,
        enable_weekend_diff: bool = True,
        ema_alpha: float = 0.3,
        weekend_multiplier_boost: float = 0.2,
        # Configurable time periods (US-136-009)
        time_periods: Optional[Dict[str, Tuple[int, int]]] = None,
    ):
        # Validate multiplier range
        min_mult, max_mult = multiplier_range
        if min_mult < 1.0 or max_mult < 1.0:
            raise ValueError(
                f"multiplier_range must have values >= 1.0, got ({min_mult}, {max_mult})"
            )
        if min_mult > max_mult:
            raise ValueError(
                f"multiplier_range min ({min_mult}) must be <= max ({max_mult})"
            )

        # Validate bin size
        valid_bin_sizes = {1, 2, 3, 4, 6, 8, 12, 24}
        if bin_size_hours not in valid_bin_sizes:
            raise ValueError(
                f"bin_size_hours must be one of {valid_bin_sizes}, got {bin_size_hours}"
            )

        # Validate EMA alpha
        if not 0 < ema_alpha <= 1:
            raise ValueError(f"ema_alpha must be between 0 and 1, got {ema_alpha}")

        self.enabled = enabled
        self.multiplier_range = multiplier_range
        self.low_success_threshold = low_success_threshold
        self.high_success_threshold = high_success_threshold
        self.max_data_age_hours = max_data_age_hours
        self.min_samples_per_hour = min_samples_per_hour

        # US-136-009: New configuration
        self.use_granular_bins = use_granular_bins
        self.bin_size_hours = bin_size_hours
        self.enable_weekend_diff = enable_weekend_diff
        self.ema_alpha = ema_alpha
        self.weekend_multiplier_boost = weekend_multiplier_boost
        # Configurable time periods - use provided or default
        self.time_periods = time_periods if time_periods is not None else TIME_PERIODS

        # Per-hour tracking: {hour: {'successes': int, 'total': int, 'last_update': float, 'ema': float}}
        self._hourly_data: Dict[int, Dict[str, float]] = {}
        # Weekend-specific data (US-136-009)
        self._weekend_data: Dict[int, Dict[str, float]] = {}
        self._weekday_data: Dict[int, Dict[str, float]] = {}
        self._last_cleanup_time: float = time.time()

    def record_attempt(
        self,
        hour: int,
        success: bool,
        is_weekend: Optional[bool] = None,
    ) -> None:
        """Record a download attempt result for the given hour.

        Args:
            hour: Hour of day in UTC (0-23)
            success: True if the download succeeded, False if it failed
            is_weekend: True if this attempt was on weekend, False for weekday.
                        If None, auto-detects from current time.
        """
        if not self.enabled:
            return

        # Validate hour range
        hour = hour % 24

        # Auto-detect weekend if not specified
        if is_weekend is None:
            now = time.gmtime()
            # Python weekday: Monday=0, Sunday=6
            is_weekend = now.tm_wday >= 5

        # Get the bin key based on granular bins setting
        bin_key = self._get_bin_key(hour)

        # Initialize if needed - main hourly data
        if bin_key not in self._hourly_data:
            self._hourly_data[bin_key] = {
                'successes': 0.0,
                'total': 0.0,
                'last_update': time.time(),
                'ema': 0.5,  # Start with neutral EMA
            }

        data = self._hourly_data[bin_key]
        data['total'] += 1.0
        if success:
            data['successes'] += 1.0

        # Update EMA (exponential moving average)
        old_ema = data.get('ema', 0.5)
        data['ema'] = self._ema_update(old_ema, 1.0 if success else 0.0)

        data['last_update'] = time.time()

        # US-136-009: Track weekend/weekday separately
        if self.enable_weekend_diff:
            if is_weekend:
                data_store = self._weekend_data
            else:
                data_store = self._weekday_data

            if bin_key not in data_store:
                data_store[bin_key] = {
                    'successes': 0.0,
                    'total': 0.0,
                    'last_update': time.time(),
                    'ema': 0.5,
                }

            wd = data_store[bin_key]
            wd['total'] += 1.0
            if success:
                wd['successes'] += 1.0

            # Update EMA
            old_ema = wd.get('ema', 0.5)
            wd['ema'] = self._ema_update(old_ema, 1.0 if success else 0.0)
            wd['last_update'] = time.time()

    def _get_bin_key(self, hour: int) -> int:
        """Get the bin key for a given hour based on bin size.

        Args:
            hour: Hour of day in UTC (0-23)

        Returns:
            Bin key (hour for 1-hour bins, even hour for 2-hour bins, etc.)
        """
        if not self.use_granular_bins or self.bin_size_hours == 1:
            return hour

        # Round down to nearest bin
        return (hour // self.bin_size_hours) * self.bin_size_hours

    def _ema_update(self, old_ema: float, new_value: float) -> float:
        """Update exponential moving average.

        Args:
            old_ema: Previous EMA value
            new_value: New observation (0.0 or 1.0)

        Returns:
            Updated EMA value
        """
        return self.ema_alpha * new_value + (1 - self.ema_alpha) * old_ema

    def get_success_rate(
        self,
        hour: int,
        use_ema: bool = False,
        is_weekend: Optional[bool] = None,
    ) -> Optional[float]:
        """Get the success rate for a specific hour.

        Args:
            hour: Hour of day in UTC (0-23)
            use_ema: If True, return EMA-adjusted success rate instead of raw
            is_weekend: If True/False, use weekend/weekday specific data.
                        If None, uses combined data.

        Returns:
            Success rate (0.0-1.0) if enough samples exist, None otherwise
        """
        if not self.enabled:
            return None

        hour = hour % 24
        bin_key = self._get_bin_key(hour)

        # Determine which data store to use
        if is_weekend is not None and self.enable_weekend_diff:
            data_store = self._weekend_data if is_weekend else self._weekday_data
        else:
            data_store = self._hourly_data

        data = data_store.get(bin_key)

        if data is None or data['total'] < self.min_samples_per_hour:
            # Fall back to combined data if available
            if is_weekend is not None:
                fallback = self._weekend_data if is_weekend else self._weekday_data
                data = fallback.get(bin_key)
            else:
                data = None

            if data is None or data['total'] < self.min_samples_per_hour:
                return None

        # Check if data is stale
        age_hours = (time.time() - data['last_update']) / 3600.0
        if age_hours > self.max_data_age_hours:
            return None

        # Return EMA if requested
        if use_ema:
            return data.get('ema', data['successes'] / data['total'])

        return data['successes'] / data['total']

    def get_time_multiplier(
        self,
        hour: Optional[int] = None,
        use_ema: bool = True,
        is_weekend: Optional[bool] = None,
    ) -> float:
        """Get the adaptive backoff multiplier for the given hour.

        If hour is None, uses current UTC hour.

        Args:
            hour: Hour of day in UTC (0-23), or None for current hour
            use_ema: If True, use exponential moving average for smoother transitions
            is_weekend: If True/False, apply weekend-specific multiplier.
                        If None, auto-detects from current time.

        Returns:
            Multiplier between multiplier_range[0] and multiplier_range[1]
        """
        if not self.enabled:
            return 1.0

        if hour is None:
            hour = int(time.gmtime().tm_hour)

        # Auto-detect weekend if not specified
        if is_weekend is None:
            now = time.gmtime()
            is_weekend = now.tm_wday >= 5

        hour = hour % 24
        success_rate = self.get_success_rate(hour, use_ema=use_ema, is_weekend=is_weekend)

        # If no data, use neutral multiplier
        if success_rate is None:
            return 1.0

        min_mult, max_mult = self.multiplier_range

        # Interpolate multiplier based on success rate
        # Low success rate (< threshold) -> higher multiplier
        # High success rate (> threshold) -> lower multiplier
        if success_rate >= self.high_success_threshold:
            # Peak hours - use minimum multiplier
            base_multiplier = min_mult
        elif success_rate <= self.low_success_threshold:
            # Low success hours - use maximum multiplier
            base_multiplier = max_mult
        else:
            # Linear interpolation between thresholds
            range_size = self.high_success_threshold - self.low_success_threshold
            rate_position = (success_rate - self.low_success_threshold) / range_size
            # Invert: lower success = higher multiplier
            base_multiplier = max_mult - (rate_position * (max_mult - min_mult))
            base_multiplier = max(min_mult, min(max_mult, base_multiplier))

        # US-136-009: Apply weekend boost
        if self.enable_weekend_diff and is_weekend:
            return base_multiplier * (1.0 + self.weekend_multiplier_boost)

        return base_multiplier

    def get_all_rates(
        self,
        use_ema: bool = False,
        is_weekend: Optional[bool] = None,
    ) -> Dict[int, float]:
        """Get success rates for all hours with enough samples.

        Args:
            use_ema: If True, return EMA-adjusted success rates
            is_weekend: If True/False, return weekend/weekday specific rates

        Returns:
            Dict mapping hour (0-23) to success rate
        """
        if not self.enabled:
            return {}

        result = {}
        for hour in range(24):
            rate = self.get_success_rate(hour, use_ema=use_ema, is_weekend=is_weekend)
            if rate is not None:
                result[hour] = rate
        return result

    def get_multiplier_for_current_hour(self) -> float:
        """Get the multiplier for the current UTC hour.

        Convenience method that uses current time.

        Returns:
            Multiplier for current hour
        """
        return self.get_time_multiplier()

    def get_period_multiplier(self, period: str) -> float:
        """Get the average multiplier for a time period.

        Args:
            period: Time period name: 'night', 'morning', 'afternoon', 'evening'

        Returns:
            Average multiplier for the period
        """
        if period not in self.time_periods:
            return 1.0

        start, end = self.time_periods[period]
        multipliers = []

        for hour in range(start, end):
            mult = self.get_time_multiplier(hour)
            if mult != 1.0:  # Only include if we have data
                multipliers.append(mult)

        if not multipliers:
            return 1.0

        return sum(multipliers) / len(multipliers)

    def to_dict(self) -> Dict:
        """Serialize state for checkpoint persistence."""
        return {
            'hourly_data': self._hourly_data,
            'weekend_data': self._weekend_data,
            'weekday_data': self._weekday_data,
            'enabled': self.enabled,
            'multiplier_range': self.multiplier_range,
            'low_success_threshold': self.low_success_threshold,
            'high_success_threshold': self.high_success_threshold,
            'max_data_age_hours': self.max_data_age_hours,
            'min_samples_per_hour': self.min_samples_per_hour,
            # US-136-009 new fields
            'use_granular_bins': self.use_granular_bins,
            'bin_size_hours': self.bin_size_hours,
            'enable_weekend_diff': self.enable_weekend_diff,
            'ema_alpha': self.ema_alpha,
            'weekend_multiplier_boost': self.weekend_multiplier_boost,
            'time_periods': self.time_periods,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "AdaptiveBackoffTimeOfDay":
        """Restore from checkpoint data."""
        if not data:
            return cls()

        # Handle time_periods deserialization (convert tuple values to tuples)
        time_periods = data.get('time_periods')
        if time_periods:
            time_periods = {k: tuple(v) for k, v in time_periods.items()}

        instance = cls(
            enabled=data.get('enabled', True),
            multiplier_range=tuple(data.get('multiplier_range', (1.0, 2.0))),
            low_success_threshold=data.get('low_success_threshold', 0.5),
            high_success_threshold=data.get('high_success_threshold', 0.8),
            max_data_age_hours=data.get('max_data_age_hours', 24),
            min_samples_per_hour=data.get('min_samples_per_hour', 5),
            # US-136-009 new fields
            use_granular_bins=data.get('use_granular_bins', True),
            bin_size_hours=data.get('bin_size_hours', 2),
            enable_weekend_diff=data.get('enable_weekend_diff', True),
            ema_alpha=data.get('ema_alpha', 0.3),
            weekend_multiplier_boost=data.get('weekend_multiplier_boost', 0.2),
            time_periods=time_periods,
        )
        instance._hourly_data = data.get('hourly_data', {})
        instance._weekend_data = data.get('weekend_data', {})
        instance._weekday_data = data.get('weekday_data', {})
        return instance


# ============== End Adaptive Backoff Time-of-Day ==============


# ============== Dynamic Player Client Selection (US-123-003) ==============

class PlayerClientTracker:
    """Tracks download success rates per player_client for dynamic selection.

    This class maintains historical success rate data per player_client variant
    (web, tv, web_safari, etc.) and provides methods to get the best performing
    client based on recent success rates.

    Usage:
        tracker = PlayerClientTracker(enabled=True, window_size=50)
        tracker.record_attempt('web_safari', success=True)
        best_client = tracker.get_best_client(['web_safari', 'tv_downgraded', 'web'])
    """

    def __init__(
        self,
        enabled: bool = True,
        window_size: int = 50,
        min_samples: int = 3,
    ):
        """Initialize the player client tracker.

        Args:
            enabled: Whether tracking is enabled.
            window_size: Number of recent attempts to consider for success rate.
            min_samples: Minimum samples needed before considering a client.
        """
        self.enabled = enabled
        self.window_size = window_size
        self.min_samples = min_samples

        # Per-client tracking: {client: [(timestamp, success), ...]}
        self._client_results: Dict[str, List[Tuple[float, bool]]] = {}

    def record_attempt(self, client: str, success: bool) -> None:
        """Record a download attempt result for a specific client.

        Args:
            client: The player_client value (e.g., 'web_safari', 'tv_downgraded')
            success: True if the download succeeded, False if it failed
        """
        if not self.enabled:
            return

        if client not in self._client_results:
            self._client_results[client] = []

        # Add new result
        self._client_results[client].append((time.time(), success))

        # Trim to window size
        if len(self._client_results[client]) > self.window_size:
            self._client_results[client] = self._client_results[client][-self.window_size:]

    def get_success_rate(self, client: str) -> Optional[float]:
        """Get the success rate for a specific client.

        Args:
            client: The player_client value

        Returns:
            Success rate (0.0-1.0) if enough samples exist, None otherwise
        """
        if not self.enabled:
            return None

        results = self._client_results.get(client)
        if not results or len(results) < self.min_samples:
            return None

        successes = sum(1 for _, success in results if success)
        return successes / len(results)

    def get_best_client(self, clients: List[str]) -> Optional[str]:
        """Get the client with the highest recent success rate.

        Args:
            clients: List of client names to consider

        Returns:
            The best performing client, or None if no client has enough samples
        """
        if not self.enabled or not clients:
            return None

        best_client = None
        best_rate = -1.0

        for client in clients:
            rate = self.get_success_rate(client)
            if rate is not None and rate > best_rate:
                best_rate = rate
                best_client = client

        return best_client

    def get_all_rates(self) -> Dict[str, float]:
        """Get success rates for all clients with enough samples.

        Returns:
            Dict mapping client name to success rate
        """
        if not self.enabled:
            return {}

        result = {}
        for client in self._client_results:
            rate = self.get_success_rate(client)
            if rate is not None:
                result[client] = rate
        return result

    def get_sample_count(self, client: str) -> int:
        """Get the number of samples recorded for a client.

        Args:
            client: The player_client value

        Returns:
            Number of samples recorded
        """
        return len(self._client_results.get(client, []))

    def reset(self) -> None:
        """Reset all tracking data."""
        self._client_results.clear()

    def to_dict(self) -> Dict:
        """Serialize state for checkpoint persistence."""
        return {
            'enabled': self.enabled,
            'window_size': self.window_size,
            'min_samples': self.min_samples,
            'client_results': {
                client: results
                for client, results in self._client_results.items()
            },
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "PlayerClientTracker":
        """Restore from checkpoint data."""
        if not data:
            return cls()

        instance = cls(
            enabled=data.get('enabled', True),
            window_size=data.get('window_size', 50),
            min_samples=data.get('min_samples', 3),
        )
        instance._client_results = data.get('client_results', {})
        return instance


# ============== End Dynamic Player Client Selection ==============


# ============== Graceful Tier Degradation (US-123-009) ==============

class TierFailureTracker:
    """Tracks per-tier failure rates for graceful degradation.

    Maintains a sliding window of recent download outcomes per tier and
    calculates failure rates to detect when a tier is struggling.

    Usage:
        tracker = TierFailureTracker(window_seconds=300.0)
        tracker.record_attempt(tier=EscalationTier.EXTRACTOR_ARGS, success=False)
        is_struggling = tracker.is_tier_struggling(EscalationTier.EXTRACTOR_ARGS, threshold=0.6)
    """

    def __init__(
        self,
        window_seconds: float = 300.0,
        min_samples: int = 3,
    ):
        """Initialize the tier failure tracker.

        Args:
            window_seconds: Time window for tracking failures (default 5 minutes).
            min_samples: Minimum samples needed before considering a tier struggling.
        """
        self.window_seconds = window_seconds
        self.min_samples = min_samples

        # Per-tier tracking: {tier_value: [(timestamp, success), ...]}
        self._tier_results: Dict[int, List[Tuple[float, bool]]] = {}

    def record_attempt(self, tier: EscalationTier, success: bool) -> None:
        """Record a download attempt result for a specific tier.

        Args:
            tier: The escalation tier used for this attempt.
            success: True if the download succeeded, False if it failed.
        """
        tier_value = tier.value

        if tier_value not in self._tier_results:
            self._tier_results[tier_value] = []

        # Add new result
        self._tier_results[tier_value].append((time.time(), success))

        # Prune old results outside the window
        self._prune_old_results(tier_value)

    def _prune_old_results(self, tier_value: int) -> None:
        """Remove results outside the time window for a tier."""
        if tier_value not in self._tier_results:
            return

        window_start = time.time() - self.window_seconds
        self._tier_results[tier_value] = [
            (ts, success) for ts, success in self._tier_results[tier_value]
            if ts > window_start
        ]

    def get_failure_rate(self, tier: EscalationTier) -> Optional[float]:
        """Get the failure rate for a specific tier in the recent window.

        Args:
            tier: The escalation tier to check.

        Returns:
            Failure rate (0.0-1.0) if enough samples exist, None otherwise.
        """
        tier_value = tier.value
        if tier_value not in self._tier_results:
            return None

        results = self._tier_results[tier_value]

        # Prune first to ensure accurate count
        self._prune_old_results(tier_value)
        results = self._tier_results.get(tier_value, [])

        if len(results) < self.min_samples:
            return None

        failures = sum(1 for _, success in results if not success)
        return failures / len(results)

    def is_tier_struggling(self, tier: EscalationTier, threshold: float = 0.6) -> bool:
        """Check if a tier is struggling (failure rate above threshold).

        Args:
            tier: The escalation tier to check.
            threshold: Failure rate threshold (default 0.6 = 60%).

        Returns:
            True if the tier is struggling, False otherwise.
        """
        failure_rate = self.get_failure_rate(tier)

        if failure_rate is None:
            return False

        return failure_rate >= threshold

    def get_struggling_tiers(self, threshold: float = 0.6) -> List[EscalationTier]:
        """Get list of tiers that are struggling.

        Args:
            threshold: Failure rate threshold (default 0.6 = 60%).

        Returns:
            List of EscalationTier that are struggling.
        """
        struggling = []

        for tier_value in self._tier_results:
            tier = EscalationTier(tier_value)
            if self.is_tier_struggling(tier, threshold):
                struggling.append(tier)

        return struggling

    def get_tier_stats(self) -> Dict[str, Dict]:
        """Get statistics for all tracked tiers.

        Returns:
            Dict mapping tier name to stats dict with failure_rate, sample_count.
        """
        stats = {}

        for tier_value in self._tier_results:
            tier = EscalationTier(tier_value)
            failure_rate = self.get_failure_rate(tier)
            results = self._tier_results.get(tier_value, [])

            stats[tier.name] = {
                'failure_rate': failure_rate,
                'sample_count': len(results),
                'is_struggling': self.is_tier_struggling(tier),
            }

        return stats

    def reset(self) -> None:
        """Reset all tracking data."""
        self._tier_results.clear()

    def to_dict(self) -> Dict:
        """Serialize state for checkpoint persistence."""
        return {
            'window_seconds': self.window_seconds,
            'min_samples': self.min_samples,
            'tier_results': {
                str(tier_value): results
                for tier_value, results in self._tier_results.items()
            },
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "TierFailureTracker":
        """Restore from checkpoint data."""
        if not data:
            return cls()

        instance = cls(
            window_seconds=data.get('window_seconds', 300.0),
            min_samples=data.get('min_samples', 3),
        )

        tier_results = data.get('tier_results', {})
        instance._tier_results = {
            int(tier_value): results
            for tier_value, results in tier_results.items()
        }

        return instance


# ============== End Graceful Tier Degradation ==============

# Category-specific patterns for escalation trigger classification.
# This is the SINGLE SOURCE OF TRUTH for all escalation trigger detection.
# The combined regex (_ESCALATION_TRIGGER_RE) is built dynamically from these.
# Order matters: more specific patterns checked first in classify_trigger().
_TRIGGER_CATEGORIES: List[Tuple[str, "re.Pattern[str]"]] = [
    ('429', re.compile(
        r'HTTP Error 429|Too Many Requests|rate.?limit',
        re.IGNORECASE,
    )),
    ('age_gate', re.compile(
        r'age.?gate|age.?restrict|sign.*in.*to.*confirm.*age',
        re.IGNORECASE,
    )),
    ('ip_blocked', re.compile(
        r'IP address|ip.*block|access denied|geo.?block|geo.restricted|not available in your',
        re.IGNORECASE,
    )),
    ('bot_detection', re.compile(
        r'bot|captcha|verify you are human',
        re.IGNORECASE,
    )),
    # US-113-006: Add login_required pattern for escalation
    ('login_required', re.compile(
        r'login.?required|sign.?in.?to.?watch',
        re.IGNORECASE,
    )),
    ('403', re.compile(
        r'HTTP Error 403|Sign in to confirm|blocked',
        re.IGNORECASE,
    )),
]

# Validate all category patterns compile as valid regex at import time.
# This catches typos or invalid patterns immediately rather than at runtime.
for _cat_name, _cat_pattern in _TRIGGER_CATEGORIES:
    assert isinstance(_cat_pattern, re.Pattern), (
        f"Invalid regex pattern in _TRIGGER_CATEGORIES[{_cat_name!r}]: "
        f"expected compiled re.Pattern, got {type(_cat_pattern).__name__}"
    )

# Build combined trigger regex dynamically from _TRIGGER_CATEGORIES.
# Joins all category patterns with '|' so is_escalation_trigger() matches
# any category without needing a separate hand-maintained regex.
_ESCALATION_TRIGGER_RE = re.compile(
    '|'.join(cat_pattern.pattern for _, cat_pattern in _TRIGGER_CATEGORIES),
    re.IGNORECASE,
)


def is_escalation_trigger(stderr_output: str) -> bool:
    """Check if yt-dlp stderr output indicates a 403/bot-detection error.

    This is used by download call sites (core.py, audio_first.py, etc.) to
    decide whether to call ``record_failure()`` on the EscalationManager.

    Note: This is a separate concern from ``ERROR_SEVERITY_PATTERNS`` in
    ``core.py``, which classifies error *severity* for backoff timing.
    ``is_escalation_trigger`` detects whether the error warrants *escalation*
    to a higher bypass tier.

    Args:
        stderr_output: Raw stderr text from a yt-dlp subprocess.

    Returns:
        True if the output matches any 403/bot-detection pattern.
    """
    if not stderr_output:
        return False
    return bool(_ESCALATION_TRIGGER_RE.search(stderr_output))


def classify_trigger(stderr_output: str) -> Optional[str]:
    """Classify the escalation trigger category from yt-dlp stderr output.

    Returns a specific category string for metrics granularity, or None
    if the output does not match any known escalation trigger.

    Categories (checked in order of specificity):
        - ``'429'``: Rate-limit / HTTP 429 / Too Many Requests
        - ``'age_gate'``: Age verification required
        - ``'ip_blocked'``: IP-based blocking / geo-blocking / access denied
        - ``'bot_detection'``: Bot / captcha / human verification
        - ``'403'``: HTTP 403 / sign-in / generic blocking

    Args:
        stderr_output: Raw stderr text from a yt-dlp subprocess.

    Returns:
        Category string or None if no trigger matched.
    """
    if not stderr_output:
        return None
    for category, pattern in _TRIGGER_CATEGORIES:
        if pattern.search(stderr_output):
            return category
    return None


# Type alias for the config - avoid circular import by using duck typing.
# Expects: .enabled, .player_clients, .escalation_threshold, .cooldown_seconds, .max_tier
try:
    from ..config.sections.download import ExtractorArgsConfig
except ImportError:  # pragma: no cover
    ExtractorArgsConfig = None  # type: ignore[misc,assignment]

try:
    from ..config.sections.download import MullvadConfig, _get_region_from_country
except ImportError:  # pragma: no cover
    MullvadConfig = None  # type: ignore[misc,assignment]
    _get_region_from_country = None  # type: ignore[misc,assignment]

try:
    from .impersonation import ImpersonationManager
except ImportError:  # pragma: no cover
    ImpersonationManager = None  # type: ignore[misc,assignment]

try:
    from .escalation_strategy import EscalationStrategy
except ImportError:  # pragma: no cover
    EscalationStrategy = None  # type: ignore[misc,assignment]

try:
    from .rate_limit_budget import RateLimitBudget
except ImportError:  # pragma: no cover
    RateLimitBudget = None  # type: ignore[misc,assignment]

try:
    from .circuit_breaker import CircuitBreaker
except ImportError:  # pragma: no cover
    CircuitBreaker = None  # type: ignore[misc,assignment]

try:
    from .mullvad_vpn import MullvadVPN
except ImportError:  # pragma: no cover
    MullvadVPN = None  # type: ignore[misc,assignment]

try:
    from .escalation_metrics import EscalationMetrics
except ImportError:  # pragma: no cover
    EscalationMetrics = None  # type: ignore[misc,assignment]


@dataclass
class EscalationResult:
    """Result from get_escalation_args() with args and metadata.

    Attributes:
        args: List of yt-dlp CLI arguments (--impersonate, --extractor-args, etc.)
        tier: The escalation tier used to generate these args.
        rotate_cookies: If True, caller should trigger cookie rotation (Tier 3).
        rotate_vpn: If True, caller should trigger VPN server rotation (Tier 4).
        impersonation_target: The browser impersonation target string used, or None.
    """
    args: List[str] = field(default_factory=list)
    tier: EscalationTier = EscalationTier.IMPERSONATE_ONLY
    rotate_cookies: bool = False
    rotate_vpn: bool = False
    impersonation_target: Optional[str] = None


@dataclass
class TierFloorEvent:
    """A single tier floor change event for history tracking."""

    timestamp: float
    from_tier: Optional[EscalationTier]
    to_tier: Optional[EscalationTier]
    reason: str  # 'automatic_elevation', 'automatic_reduction', 'manual_set', 'manual_clear'
    rate_limit_percentage: float = 0.0  # Percentage of keywords that triggered this


class EscalationManager:
    """Orchestrates 4-tier yt-dlp bypass escalation per keyword.

    Per-keyword tracking ensures that one keyword hitting 403 errors does not
    affect the escalation state of other keywords.

    Tier progression:
        - Tier 1 (IMPERSONATE_ONLY): --impersonate only
        - Tier 2 (EXTRACTOR_ARGS): --impersonate + --extractor-args player_client
        - Tier 3 (FULL_BYPASS): Both + cookie rotation flag
        - Tier 4 (VPN_ROTATION): All above + VPN server rotation

    Args:
        impersonation_manager: Provides Tier 1 --impersonate args.
        extractor_args_config: Configuration for Tier 2 player_client rotation.
            If None, a default config is used with escalation disabled.
        budget: Optional RateLimitBudget for budget-aware escalation.
            When provided, escalation decisions consult the budget:
            - record_failure() calls budget.record_rotation() on tier advances
            - If budget is exhausted, skip intermediate tiers to max tier
            - get_escalation_args() calls budget.record_attempt()
        mullvad_vpn: Optional MullvadVPN manager for Tier 4 VPN rotation.
        on_vpn_rotation_needed: Optional callback invoked when escalating to Tier 4.
            Called with (keyword: str) to allow caller to handle VPN rotation.
    """

    # Default thresholds for automatic tier floor management
    DEFAULT_ELEVATION_THRESHOLD = 0.3  # 30% of keywords
    DEFAULT_REDUCTION_TIMEOUT = 600.0  # 10 minutes
    DEFAULT_ELEVATION_WINDOW = 300.0  # 5 minutes
    DEFAULT_DEESCALATION_STEP = 1  # De-escalate 1 tier at a time

    def __init__(
        self,
        impersonation_manager: "ImpersonationManager",
        extractor_args_config: Optional["ExtractorArgsConfig"] = None,
        budget: Optional["RateLimitBudget"] = None,
        strategy: Optional["EscalationStrategy"] = None,
        mullvad_vpn: Optional["MullvadVPN"] = None,
        on_vpn_rotation_needed: Optional[Callable[[str], None]] = None,
        metrics: Optional["EscalationMetrics"] = None,
        # Automatic tier floor management config
        elevation_threshold: float = DEFAULT_ELEVATION_THRESHOLD,
        reduction_timeout: float = DEFAULT_REDUCTION_TIMEOUT,
        # Rate limit predictor for proactive budget adjustment (US-113-003)
        rate_limit_predictor: Optional[RateLimitPredictor] = None,
        predictor_enabled: bool = True,
        # Adaptive backoff time-of-day for US-114-004
        adaptive_backoff_time_of_day: Optional[AdaptiveBackoffTimeOfDay] = None,
    ):
        self._impersonation_manager = impersonation_manager
        self._extractor_config = extractor_args_config
        self._budget = budget
        # Create strategy if not provided (for backwards compatibility)
        self._strategy = strategy or EscalationStrategy(extractor_args_config)
        self._circuit_breaker: Optional["CircuitBreaker"] = None
        self._mullvad_vpn: Optional["MullvadVPN"] = mullvad_vpn
        self._on_vpn_rotation_needed = on_vpn_rotation_needed
        self._keyword_states: Dict[str, EscalationState] = {}
        self._keyword_locks: Dict[str, threading.Lock] = {}
        self._global_lock = threading.RLock()
        self._tier_floor: Optional[EscalationTier] = None
        self._slow_speed_counts: Dict[str, int] = {}  # keyword -> consecutive slow count
        # Delegate all metrics tracking to EscalationMetrics
        self._metrics = metrics if metrics is not None else EscalationMetrics()

        # Automatic tier floor management
        self._elevation_threshold = elevation_threshold
        self._reduction_timeout = reduction_timeout
        self._elevation_window = self.DEFAULT_ELEVATION_WINDOW
        self._deescalation_step = self.DEFAULT_DEESCALATION_STEP
        self._tier_floor_history: List[TierFloorEvent] = []  # History of tier floor changes
        self._recent_rate_limit_events: List[Tuple[float, str]] = []  # (timestamp, keyword) tuples
        self._last_rate_limit_time: Optional[float] = None  # Last time a rate limit was recorded

        # Rate limit predictor for proactive budget adjustment (US-113-003)
        self._rate_limit_predictor = rate_limit_predictor
        self._predictor_enabled = predictor_enabled

        # Adaptive backoff time-of-day (US-114-004)
        self._adaptive_backoff_time_of_day = (
            adaptive_backoff_time_of_day
            if adaptive_backoff_time_of_day is not None
            else AdaptiveBackoffTimeOfDay(enabled=True)
        )

        # Dynamic player_client selection (US-123-003)
        self._player_client_tracker = PlayerClientTracker(enabled=True)

        # Graceful tier degradation (US-123-009)
        # These are set from config in the factory method or manually after construction
        self._tier_graceful_degradation: bool = True
        self._tier_degradation_threshold: float = 0.6
        self._tier_degradation_window: float = 300.0
        self._tier_failure_tracker = TierFailureTracker(
            window_seconds=300.0,
            min_samples=3,
        )

    def set_graceful_degradation_config(
        self,
        enabled: bool = True,
        threshold: float = 0.6,
        window_seconds: float = 300.0,
    ) -> None:
        """Configure graceful tier degradation.

        Args:
            enabled: Whether graceful degradation is enabled.
            threshold: Failure rate threshold (0.0-1.0) for struggling detection.
            window_seconds: Time window for tracking failures.
        """
        self._tier_graceful_degradation = enabled
        self._tier_degradation_threshold = threshold
        self._tier_degradation_window = window_seconds
        self._tier_failure_tracker = TierFailureTracker(
            window_seconds=window_seconds,
            min_samples=3,
        )

    def is_graceful_degradation_enabled(self) -> bool:
        """Check if graceful tier degradation is enabled.

        Returns:
            True if graceful degradation is enabled.
        """
        return self._tier_graceful_degradation

    def get_graceful_degradation_config(self) -> Dict:
        """Get the graceful degradation configuration.

        Returns:
            Dict with enabled, threshold, and window settings.
        """
        return {
            'enabled': self._tier_graceful_degradation,
            'threshold': self._tier_degradation_threshold,
            'window_seconds': self._tier_degradation_window,
        }

    def get_struggling_tiers(self) -> List[EscalationTier]:
        """Get list of tiers that are currently struggling.

        Returns:
            List of EscalationTier that are struggling (>60% failure rate).
        """
        if not self._tier_graceful_degradation:
            return []

        return self._tier_failure_tracker.get_struggling_tiers(self._tier_degradation_threshold)

    def is_tier_struggling(self, tier: EscalationTier) -> bool:
        """Check if a specific tier is currently struggling.

        Args:
            tier: The escalation tier to check.

        Returns:
            True if the tier is struggling.
        """
        if not self._tier_graceful_degradation:
            return False

        return self._tier_failure_tracker.is_tier_struggling(tier, self._tier_degradation_threshold)

    def graceful_degrade(self, keyword: str, already_locked: bool = False) -> bool:
        """Gracefully degrade a keyword's tier rather than full escalation.

        When a tier is struggling (>60% failure rate), this method reduces
        the tier's parameters (e.g., rotates player_client) instead of
        jumping to the next full tier. This is less aggressive than full
        tier escalation.

        Args:
            keyword: The keyword to potentially degrade.
            already_locked: If True, assumes caller already holds the lock
                           (used by get_escalation_args to avoid deadlock).

        Returns:
            True if graceful degradation was applied, False otherwise.
        """
        if not self._tier_graceful_degradation:
            return False

        def _do_degrade():
            state = self._get_state(keyword)
            current_tier = state.current_tier

            # Check if current tier is struggling
            if not self._tier_failure_tracker.is_tier_struggling(
                current_tier, self._tier_degradation_threshold
            ):
                return False

            # Graceful degradation: rotate extractor_args instead of full tier escalation
            # This applies primarily to Tier 2+ where extractor_args are used
            if current_tier >= EscalationTier.EXTRACTOR_ARGS:
                state.extractor_args_index += 1
                logger.info(
                    f"Graceful degradation for {keyword}: rotating extractor_args "
                    f"(tier {current_tier.name} is struggling), index now {state.extractor_args_index}"
                )
                return True

            return False

        if already_locked:
            return _do_degrade()
        else:
            lock = self._get_lock(keyword)
            with lock:
                return _do_degrade()

    def get_tier_degradation_stats(self) -> Dict:
        """Get tier degradation statistics for debugging/monitoring.

        Returns:
            Dict with struggling tiers, per-tier stats, and config.
        """
        return {
            'struggling_tiers': [t.name for t in self.get_struggling_tiers()],
            'tier_stats': self._tier_failure_tracker.get_tier_stats(),
            'config': self.get_graceful_degradation_config(),
        }

    def _get_lock(self, keyword: str) -> threading.Lock:
        """Get or create a per-keyword lock (thread-safe)."""
        with self._global_lock:
            if keyword not in self._keyword_locks:
                self._keyword_locks[keyword] = threading.Lock()
            return self._keyword_locks[keyword]

    def _get_state(self, keyword: str) -> EscalationState:
        """Get or create the escalation state for a keyword.

        When a tier floor is set, newly created states start at the floor tier
        instead of Tier 1. Existing states below the floor are elevated.
        """
        if keyword not in self._keyword_states:
            state = EscalationState()
            if self._tier_floor is not None and state.current_tier < self._tier_floor:
                state.current_tier = self._tier_floor
            self._keyword_states[keyword] = state
        else:
            state = self._keyword_states[keyword]
            if self._tier_floor is not None and state.current_tier < self._tier_floor:
                state.current_tier = self._tier_floor
        return self._keyword_states[keyword]

    def set_tier_floor(self, tier: EscalationTier, reason: str = "manual_set") -> None:
        """Set a global minimum escalation tier for all keywords.

        When set, all new and existing keywords will start at this tier
        instead of Tier 1. Used by download_segments stage to propagate
        broad bot-detection signals across all video IDs.

        Args:
            tier: The minimum escalation tier to enforce.
            reason: Reason for tier floor change (default: 'manual_set').
                Other values: 'automatic_elevation', 'automatic_reduction'.
        """
        with self._global_lock:
            from_tier = self._tier_floor
            self._tier_floor = tier

            # Record history event
            event = TierFloorEvent(
                timestamp=time.time(),
                from_tier=from_tier,
                to_tier=tier,
                reason=reason,
                rate_limit_percentage=self._get_rate_limit_percentage(),
            )
            self._tier_floor_history.append(event)

            logger.info(f"Global tier floor set to {tier.name} (reason: {reason})")

    def clear_tier_floor(self, reason: str = "manual_clear") -> None:
        """Remove the global tier floor, allowing new keywords to start at Tier 1.

        Args:
            reason: Reason for clearing (default: 'manual_clear').
                Other values: 'automatic_reduction'.
        """
        with self._global_lock:
            from_tier = self._tier_floor
            self._tier_floor = None

            # Record history event
            event = TierFloorEvent(
                timestamp=time.time(),
                from_tier=from_tier,
                to_tier=None,
                reason=reason,
                rate_limit_percentage=0.0,
            )
            self._tier_floor_history.append(event)

            logger.info(f"Global tier floor cleared (reason: {reason})")

    def _get_rate_limit_percentage(self) -> float:
        """Calculate the percentage of keywords with rate limits in the elevation window.

        Returns:
            Float between 0.0 and 1.0 representing percentage of keywords
            with recent rate limit events.
        """
        if not self._keyword_states:
            return 0.0

        # Get unique keywords that have had rate limit events in the window
        now = time.time()
        window_start = now - self._elevation_window

        # Filter events in window and extract unique keywords
        unique_keywords = set()
        for timestamp, keyword in self._recent_rate_limit_events:
            if timestamp > window_start:
                unique_keywords.add(keyword)

        return len(unique_keywords) / len(self._keyword_states)

    def _prune_rate_limit_events(self) -> None:
        """Remove rate limit events outside the elevation window."""
        now = time.time()
        window_start = now - self._elevation_window
        self._recent_rate_limit_events = [
            (ts, kw) for ts, kw in self._recent_rate_limit_events if ts > window_start
        ]

    def record_global_rate_limit(self, keyword: str) -> None:
        """Record a rate limit event for automatic tier floor management.

        Call this when any keyword experiences a rate limit. The manager
        tracks these events and automatically elevates the tier floor when
        the threshold (>30% in 5 minutes) is exceeded.

        Args:
            keyword: The keyword that experienced the rate limit.
        """
        now = time.time()

        with self._global_lock:
            self._recent_rate_limit_events.append((now, keyword))
            self._last_rate_limit_time = now

            # Prune old events
            self._prune_rate_limit_events()

            # Check if we need to elevate tier floor
            self._maybe_elevate_tier_floor()

    def _maybe_elevate_tier_floor(self) -> None:
        """Automatically elevate tier floor if rate limit threshold exceeded."""
        if not self._keyword_states:
            return

        percentage = self._get_rate_limit_percentage()

        if percentage >= self._elevation_threshold:
            # Determine new tier floor (escalate by 1 from current)
            current_floor = self._tier_floor or EscalationTier.IMPERSONATE_ONLY

            # Only elevate if not already at max tier
            if current_floor < EscalationTier.FULL_BYPASS:
                # Escalate one tier
                new_tier = EscalationTier(current_floor.value + 1)

                logger.info(
                    f"Automatic tier floor elevation: {percentage:.1%} keywords hit rate limits "
                    f"in {self._elevation_window}s window, elevating from {current_floor.name} "
                    f"to {new_tier.name}"
                )

                self.set_tier_floor(new_tier, reason="automatic_elevation")

    def check_and_reduce_tier_floor(self) -> bool:
        """Check if tier floor should be automatically reduced.

        Called periodically (e.g., every minute) to check if conditions
        are met for automatic de-escalation:
        - No rate limits for reduction_timeout (default 10 minutes)
        - Gradual de-escalation: one tier at a time

        Returns:
            True if tier floor was reduced, False otherwise.
        """
        with self._global_lock:
            if self._tier_floor is None:
                return False

            if self._last_rate_limit_time is None:
                # No rate limits recorded yet, don't reduce
                return False

            now = time.time()
            time_since_last_limit = now - self._last_rate_limit_time

            if time_since_last_limit >= self._reduction_timeout:
                # Time to reduce - de-escalate one tier
                current_floor = self._tier_floor

                if current_floor > EscalationTier.IMPERSONATE_ONLY:
                    new_tier = EscalationTier(current_floor.value - self._deescalation_step)
                    new_tier = max(new_tier, EscalationTier.IMPERSONATE_ONLY)

                    logger.info(
                        f"Automatic tier floor reduction: no rate limits for "
                        f"{time_since_last_limit:.0f}s (threshold: {self._reduction_timeout}s), "
                        f"reducing from {current_floor.name} to {new_tier.name}"
                    )

                    self.set_tier_floor(new_tier, reason="automatic_reduction")
                    return True
                else:
                    # Already at minimum tier, clear the floor
                    logger.info(
                        f"Automatic tier floor reduction: no rate limits for "
                        f"{time_since_last_limit:.0f}s, clearing floor (already at minimum)"
                    )
                    self.clear_tier_floor(reason="automatic_reduction")
                    return True

            return False

    def get_tier_floor_history(self) -> List[Dict]:
        """Get the history of tier floor changes.

        Returns:
            List of dicts with tier floor change events.
        """
        return [
            {
                "timestamp": event.timestamp,
                "from_tier": event.from_tier.name if event.from_tier else None,
                "to_tier": event.to_tier.name if event.to_tier else None,
                "reason": event.reason,
                "rate_limit_percentage": round(event.rate_limit_percentage, 4),
            }
            for event in self._tier_floor_history
        ]

    @property
    def tier_floor(self) -> Optional[EscalationTier]:
        """Get the current tier floor."""
        return self._tier_floor

    @property
    def tier_floor_config(self) -> Dict:
        """Get tier floor management configuration and state."""
        return {
            "elevation_threshold": self._elevation_threshold,
            "reduction_timeout": self._reduction_timeout,
            "elevation_window": self._elevation_window,
            "current_tier_floor": self._tier_floor.name if self._tier_floor else None,
            "recent_rate_limit_count": len(self._recent_rate_limit_events),
            "last_rate_limit_time": self._last_rate_limit_time,
        }

    def set_circuit_breaker(self, circuit_breaker: "CircuitBreaker") -> None:
        """Link a CircuitBreaker for coordinated rate-limiting.

        When linked, get_escalation_args() will return Tier 3 args
        immediately during circuit breaker pause (skip lower tiers).

        Args:
            circuit_breaker: The CircuitBreaker to consult.
        """
        self._circuit_breaker = circuit_breaker

    def set_mullvad_vpn(self, mullvad_vpn: "MullvadVPN") -> None:
        """Link a MullvadVPN manager for Tier 4 VPN rotation.

        When linked, escalation to Tier 4 will trigger VPN server rotation
        for IP-based rate limit bypass.

        Args:
            mullvad_vpn: The MullvadVPN manager to use.
        """
        self._mullvad_vpn = mullvad_vpn

    def get_mullvad_vpn(self) -> Optional["MullvadVPN"]:
        """Get the linked MullvadVPN manager.

        Returns:
            The MullvadVPN manager if linked, None otherwise.
        """
        return self._mullvad_vpn

    def get_current_region(self) -> Optional[str]:
        """Get the current region based on VPN country code.

        Queries the MullvadVPN status to determine the current country,
        then maps it to a region (us, eu, asia, other).

        Returns:
            Region string ('us', 'eu', 'asia', 'other') or None if no VPN.
        """
        if self._mullvad_vpn is None:
            return None
        if _get_region_from_country is None:
            return None
        try:
            status = self._mullvad_vpn.get_status()
            country = status.get("country")
            if country:
                return _get_region_from_country(country)
        except Exception:
            pass
        return None

    # ============== Rate Limit Predictor Integration (US-113-003, US-143-004) ==============

    def get_rate_limit_predictor(self) -> Optional[RateLimitPredictor]:
        """Get the rate limit predictor instance.

        Returns:
            The RateLimitPredictor if configured, None otherwise.
        """
        return self._rate_limit_predictor

    def is_predictor_enabled(self) -> bool:
        """Check if the rate limit predictor is enabled.

        Returns:
            True if predictor is enabled, False otherwise.
        """
        return self._predictor_enabled and self._rate_limit_predictor is not None

    def get_prediction_for_metrics(self) -> Optional[Dict]:
        """Get rate limit prediction data for metrics export.

        US-143-004: New method to expose prediction via download metrics.

        Returns:
            Dict with prediction details, or None if predictor not enabled.
        """
        if not self.is_predictor_enabled() or self._rate_limit_predictor is None:
            return None

        try:
            predictor = self._rate_limit_predictor

            # Get time window prediction
            tw_prediction = predictor.get_time_window_prediction()

            # Get hourly prediction
            hourly_prediction = predictor.get_hourly_prediction()

            # Get weekend stats
            weekend_stats = predictor.get_weekend_stats()

            return {
                "time_window_prediction": tw_prediction,
                "hourly_prediction": hourly_prediction,
                "weekend_stats": weekend_stats,
                "sensitivity": predictor._sensitivity,
            }
        except Exception as e:
            logger.debug(f"Error getting prediction for metrics: {e}")
            return None

    def predict_and_adjust_budget(self) -> float:
        """Predict rate limit likelihood and proactively adjust budget allocation.

        This method should be called before budget allocation to increase
        budgets when rate limit likelihood is high (> 0.6).

        Returns:
            The predicted rate limit likelihood (0.0-1.0).
        """
        if not self.is_predictor_enabled():
            return 0.0

        if self._budget is None:
            return 0.0

        try:
            likelihood = self._rate_limit_predictor.predict_rate_limit_likelihood()
            logger.debug(f"Rate limit likelihood: {likelihood:.2f}")

            # Increase budget allocation if likelihood exceeds threshold
            if likelihood > self._rate_limit_predictor.BUDGET_INCREASE_THRESHOLD:
                self._rate_limit_predictor.increase_budget_allocation(self._budget)
                logger.info(
                    f"Budget proactively increased due to high rate limit likelihood: {likelihood:.2f}"
                )

            return likelihood
        except Exception as e:
            log_rate_limit(logger, "prediction", "rate_limit_predictor", "failed", error=str(e))
            return 0.0

    def record_predictor_attempt(self, keyword: str = None) -> None:
        """Record a download attempt in the predictor for pattern tracking.

        Args:
            keyword: The keyword that attempted download (optional, ignored by predictor).
        """
        if not self.is_predictor_enabled():
            return

        try:
            # Note: RateLimitPredictor.record_attempt() doesn't accept keyword parameter
            self._rate_limit_predictor.record_attempt()
        except Exception as e:
            logger.debug(f"Failed to record predictor attempt: {e}")

    def record_predictor_rate_limit(self, trigger_category: str = "unknown",
                                    tier: str = "tier1", keyword: str = None) -> None:
        """Record a rate limit event in the predictor for pattern tracking.

        Args:
            trigger_category: Category of trigger (e.g., '429', '403').
            tier: Which tier triggered the rate limit.
            keyword: The keyword that triggered the event.
        """
        if not self.is_predictor_enabled():
            return

        try:
            self._rate_limit_predictor.record_rate_limit_event(
                trigger_category=trigger_category,
                tier=tier,
                keyword=keyword
            )
        except Exception as e:
            logger.debug(f"Failed to record predictor rate limit: {e}")

    # ============== End Rate Limit Predictor Integration ==============

    # ============== Adaptive Backoff Time-of-Day (US-114-004) ==============

    def get_adaptive_time_multiplier(self, hour: Optional[int] = None) -> float:
        """Get the adaptive backoff multiplier for the given hour.

        This applies time-of-day based adjustment to backoff durations.
        During historically low-success hours (e.g., US night = 0-6 UTC),
        longer backoff is applied to increase chances of success.

        Args:
            hour: Hour of day in UTC (0-23), or None for current hour

        Returns:
            Multiplier between configured min and max (default: 1.0-2.0)
        """
        return self._adaptive_backoff_time_of_day.get_time_multiplier(hour)

    def record_time_of_day_result(self, success: bool) -> None:
        """Record a download attempt result for time-of-day tracking.

        Args:
            success: True if the download succeeded, False if it failed
        """
        current_hour = int(time.gmtime().tm_hour)
        self._adaptive_backoff_time_of_day.record_attempt(current_hour, success)

    def get_time_of_day_stats(self) -> Dict:
        """Get time-of-day tracking statistics for debugging/monitoring.

        Returns:
            Dict with hourly success rates and current multiplier
        """
        hourly_rates = self._adaptive_backoff_time_of_day.get_all_rates()
        return {
            'hourly_success_rates': hourly_rates,
            'current_hour': int(time.gmtime().tm_hour),
            'current_multiplier': self._adaptive_backoff_time_of_day.get_multiplier_for_current_hour(),
            'enabled': self._adaptive_backoff_time_of_day.enabled,
        }

    def get_time_of_day_tracker(self) -> AdaptiveBackoffTimeOfDay:
        """Get the adaptive backoff time-of-day tracker instance.

        Returns:
            The AdaptiveBackoffTimeOfDay instance
        """
        return self._adaptive_backoff_time_of_day

    # ============== End Adaptive Backoff Time-of-Day ==============

    def set_vpn_rotation_callback(self, callback: Callable[[str], None]) -> None:
        """Set or replace the on_vpn_rotation_needed callback (US-36-003).

        This allows the pipeline to wire up budget reset logic after
        the EscalationManager is instantiated. Called by HealingOrchestrator
        to ensure budget.reset_on_ip_change() is invoked on Tier 4 escalation.

        Args:
            callback: Callable taking keyword (str) to invoke on VPN rotation.
        """
        self._on_vpn_rotation_needed = callback

    @property
    def metrics(self) -> "EscalationMetrics":
        """Access the EscalationMetrics instance."""
        return self._metrics

    @property
    def keyword_states(self) -> Dict[str, EscalationState]:
        """Read-only access to keyword states (for metrics/debugging)."""
        return dict(self._keyword_states)

    @property
    def strategy(self) -> "EscalationStrategy":
        """Access the escalation strategy (for testing/inspection)."""
        return self._strategy

    def get_escalation_args(self, keyword: str) -> EscalationResult:
        """Get yt-dlp arguments for the current escalation tier of a keyword.

        Tier 1: --impersonate <target> only
        Tier 2: --impersonate <target> + --extractor-args "youtube:player_client=X,Y,Z"
        Tier 3: All of Tier 2 + rotate_cookies=True flag
        Tier 4: All of Tier 3 + rotate_vpn=True flag

        Also records the attempt in the budget (if available) to track total
        download attempts across keywords.

        Args:
            keyword: The download keyword or video ID.

        Returns:
            EscalationResult with args list, tier, cookie rotation, and VPN rotation flags.
        """
        # PROACTIVE: Predict rate limit likelihood and adjust budget BEFORE allocation (US-113-003)
        # This increases budget allocation proactively when historical patterns suggest high risk
        self.predict_and_adjust_budget()

        # Track attempt in budget (outside lock - budget has its own thread safety)
        if self._budget is not None:
            self._budget.record_attempt(keyword)

        # Also record in predictor for pattern tracking
        self.record_predictor_attempt(keyword)

        lock = self._get_lock(keyword)
        with lock:
            state = self._get_state(keyword)
            tier = state.current_tier

            # Graceful degradation check (US-123-009): before full escalation,
            # try rotating parameters instead of jumping to next tier
            if self._tier_graceful_degradation and self.is_tier_struggling(tier):
                # Try graceful degradation first - this rotates extractor_args
                # instead of doing a full tier escalation
                degraded = self.graceful_degrade(keyword, already_locked=True)
                if degraded:
                    logger.debug(
                        f"Graceful degradation applied for {keyword}: "
                        f"tier {tier.name} is struggling"
                    )
                    # After degradation, re-read state (extractor_args_index changed)
                    tier = state.current_tier

            # Delegate circuit breaker shortcut decision to strategy
            cb_open = self._circuit_breaker is not None and self._circuit_breaker.is_open
            shortcut_decision = self._strategy.should_shortcut_to_max(tier, cb_open)
            if shortcut_decision.should_escalate:
                logger.info(
                    f"Circuit breaker open: shortcutting keyword={keyword} "
                    f"from {tier.name} to FULL_BYPASS"
                )
                tier = shortcut_decision.target_tier

            # Tier 1: impersonation only
            args, imp_target = self._impersonation_manager.get_impersonate_args_with_target()

            result = EscalationResult(
                args=list(args),
                tier=tier,
                rotate_cookies=False,
                rotate_vpn=False,
                impersonation_target=imp_target,
            )

            # Tier 2+: add extractor-args
            if tier >= EscalationTier.EXTRACTOR_ARGS:
                extractor_args = self._build_extractor_args(state)
                if extractor_args:
                    result.args.extend(extractor_args)

            # Tier 3+: signal cookie rotation
            if tier >= EscalationTier.FULL_BYPASS:
                result.rotate_cookies = True

            # Tier 4: signal VPN rotation (only if Mullvad is configured)
            if tier >= EscalationTier.VPN_ROTATION and self._mullvad_vpn is not None:
                result.rotate_vpn = True

            logger.debug(
                f"Escalation args: keyword={keyword} tier={tier.name} "
                f"args_count={len(result.args)} rotate_cookies={result.rotate_cookies} "
                f"rotate_vpn={result.rotate_vpn}"
            )

            return result

    def _build_extractor_args(self, state: EscalationState) -> List[str]:
        """Build --extractor-args for Tier 2+ from ExtractorArgsConfig.

        Rotates the player_client list starting position using
        state.extractor_args_index so successive escalations try different
        client orderings.

        When dynamic selection is enabled (success_rate_window > 0 and no
        manual override), uses the best performing client based on recent
        success rates.

        Returns:
            ['--extractor-args', 'youtube:player_client=X,Y,Z'] or empty list.
        """
        if self._extractor_config is None:
            return []
        if not getattr(self._extractor_config, 'enabled', True):
            return []

        clients = getattr(self._extractor_config, 'player_clients', [])
        if not clients:
            return []

        # Get the best client if dynamic selection is enabled
        best_client = self.get_best_extractor_args(clients)
        if best_client:
            # Use the best client as primary, followed by others in rotation order
            other_clients = [c for c in clients if c != best_client]
            rotated = [best_client] + other_clients
            client_str = ','.join(rotated)
            # Log dynamic selection success
            log_rate_limit(
                logger, "player_client", "escalation_manager", "dynamic_selection",
                selected_client=best_client, fallback_type="dynamic"
            )
        else:
            # Fallback to rotation-based selection
            idx = state.extractor_args_index % len(clients)
            rotated = clients[idx:] + clients[:idx]
            client_str = ','.join(rotated)
            # Log fallback to rotation
            log_rate_limit(
                logger, "player_client", "escalation_manager", "fallback_to_rotation",
                selected_client=rotated[0], fallback_type="rotation",
                extractor_args_index=idx
            )

        return ['--extractor-args', f'youtube:player_client={client_str}']

    def get_best_extractor_args(self, clients: List[str]) -> Optional[str]:
        """Get the best player_client based on recent success rates.

        Uses dynamic selection if:
        - success_rate_window > 0 (enabled)
        - extractor_args_fallback_order is empty (no manual override)

        Args:
            clients: List of available player_client values

        Returns:
            The best performing client, or None if dynamic selection is disabled
            or not enough data
        """
        if self._extractor_config is None:
            return None

        # Check if manual override is configured
        fallback_order = getattr(self._extractor_config, 'extractor_args_fallback_order', [])
        if fallback_order:
            # Use manual override - return the first client from fallback order
            # that's in the available clients list
            for client in fallback_order:
                if client in clients:
                    # Log fallback order selection
                    log_rate_limit(
                        logger, "player_client", "escalation_manager", "fallback_order",
                        selected_client=client, fallback_order=fallback_order
                    )
                    return client

        # Check if dynamic selection is enabled
        success_rate_window = getattr(self._extractor_config, 'success_rate_window', 50)
        if success_rate_window <= 0:
            return None

        # Update tracker window size if configured
        self._player_client_tracker.window_size = success_rate_window

        # Get best client based on success rates
        return self._player_client_tracker.get_best_client(clients)

    def record_extractor_args_result(self, client: str, success: bool) -> None:
        """Record a download result for a specific player_client.

        Used to track success rates per client for dynamic selection.

        Args:
            client: The player_client value used (e.g., 'web_safari')
            success: True if the download succeeded, False if it failed
        """
        self._player_client_tracker.record_attempt(client, success)

    def get_extractor_args_stats(self) -> Dict:
        """Get player_client tracking statistics for debugging/monitoring.

        Returns:
            Dict with client success rates and sample counts
        """
        return {
            'client_success_rates': self._player_client_tracker.get_all_rates(),
            'client_sample_counts': {
                client: self._player_client_tracker.get_sample_count(client)
                for client in self._player_client_tracker._client_results
            },
            'enabled': self._player_client_tracker.enabled,
        }

    def _record_timeline_event(
        self, keyword: str, from_tier: EscalationTier, to_tier: EscalationTier,
        trigger_category: Optional[str] = None,
    ) -> None:
        """Record an escalation event in the per-keyword timeline.

        Delegates to EscalationMetrics.record_escalation().

        Args:
            keyword: The keyword that escalated.
            from_tier: Tier before escalation.
            to_tier: Tier after escalation.
            trigger_category: Category from classify_trigger() or None.
        """
        # Note: record_escalation also increments _total_escalations and
        # _escalations_per_tier, so callers should NOT duplicate those updates.
        # This is handled by the refactored record_failure/record_slow_speed.

    def record_failure(self, keyword: str, error_output: str = "",
                      player_client: Optional[str] = None) -> None:
        """Record a download failure for a keyword.

        Increments the consecutive 403 counter. If the threshold is reached,
        escalates to the next tier. When a budget is available:
        - Records a rotation on tier advance (Tier 2 or Tier 3)
        - If budget is exhausted (can_rotate() is False), skips intermediate
          tiers and jumps directly to max tier (FULL_BYPASS)

        Also records global rate limit events for automatic tier floor management.

        Args:
            keyword: The download keyword or video ID.
            error_output: stderr output from the failed subprocess.
            player_client: The player_client used (for success rate tracking, US-123-003).
        """
        # Classify trigger category for timeline tracking
        trigger_category = classify_trigger(error_output) if error_output else None

        lock = self._get_lock(keyword)
        # Track if we escalated for later use outside the lock
        did_escalate = False
        # Track tier used for failure tracking (US-123-009)
        tier_used = None
        with lock:
            state = self._get_state(keyword)
            tier_used = state.current_tier  # Track tier before any escalation
            state.consecutive_403s += 1
            state.consecutive_successes = 0  # Reset success streak on any failure
            self._metrics.record_failure(keyword)

            # Delegate escalation decision to strategy
            budget_exhausted = self._budget is not None and not self._budget.can_rotate()
            decision = self._strategy.should_escalate_on_failure(state, budget_exhausted)

            if decision.should_escalate:
                did_escalate = True
                old_tier = state.current_tier
                n_403s = state.consecutive_403s

                if decision.skip_to_max:
                    # Budget exhausted: skip to max tier
                    logger.warning(
                        f"Budget exhausted for keyword={keyword}: "
                        f"skipping to FULL_BYPASS (was {state.current_tier.name})"
                    )
                    # Log tier transition for budget skip
                    from_tier_num = old_tier.value
                    to_tier_num = decision.target_tier.value
                    log_rate_limit(
                        logger, "tier_escalation", "escalation_manager", "budget_skip",
                        keyword=keyword, from_tier=from_tier_num, to_tier=to_tier_num,
                        consecutive_403s=n_403s
                    )
                    state.current_tier = decision.target_tier
                    state.last_escalation_time = time.time()
                    state.escalation_history.append(
                        (state.last_escalation_time, state.current_tier)
                    )
                    state.consecutive_403s = 0
                    state.extractor_args_index += 1
                else:
                    # Normal escalation
                    state.escalate()
                    # Increment extractor_args_index on Tier 2 escalation
                    if state.current_tier >= EscalationTier.EXTRACTOR_ARGS:
                        state.extractor_args_index += 1

                # Delegate escalation tracking to metrics
                self._metrics.record_escalation(
                    keyword, old_tier, state.current_tier, trigger_category
                )

                # Budget tracking: record rotation when advancing to Tier 2 or Tier 3
                if self._budget is not None and state.current_tier > old_tier:
                    self._budget.record_rotation(keyword)

                logger.info(
                    f"Escalation: keyword={keyword} tier {old_tier.name}->{state.current_tier.name} "
                    f"after {n_403s} consecutive 403s"
                )

                # Log tier transition with log_rate_limit
                from_tier_num = old_tier.value
                to_tier_num = state.current_tier.value
                if decision.skip_to_max:
                    action = "budget_skip"
                else:
                    action = f"tier_{from_tier_num}_to_{to_tier_num}"
                log_rate_limit(
                    logger, "tier_escalation", "escalation_manager", action,
                    keyword=keyword, from_tier=from_tier_num, to_tier=to_tier_num,
                    consecutive_403s=n_403s, trigger_category=trigger_category
                )

                if state.current_tier == EscalationTier.FULL_BYPASS:
                    logger.warning(
                        f"Tier 3 reached for keyword={keyword}, engaging full bypass with cookies"
                    )
                    log_rate_limit(
                        logger, "tier_reached", "escalation_manager", "tier_3_full_bypass",
                        keyword=keyword, tier=3
                    )
                elif state.current_tier == EscalationTier.VPN_ROTATION:
                    logger.warning(
                        f"Max escalation (Tier 4) reached for keyword={keyword}, "
                        f"engaging VPN rotation"
                    )
                    log_rate_limit(
                        logger, "tier_reached", "escalation_manager", "tier_4_vpn_rotation",
                        keyword=keyword, tier=4
                    )
                    # Invoke callback for VPN rotation handling
                    if self._on_vpn_rotation_needed is not None:
                        try:
                            self._on_vpn_rotation_needed(keyword)
                        except Exception as e:
                            logger.error(
                                f"VPN rotation callback failed for keyword={keyword}: {e}"
                            )

        # Record global rate limit event for automatic tier floor management
        # (outside per-keyword lock to avoid deadlock with _global_lock)
        # Check if this was a rate limit trigger or if escalation occurred
        is_rate_limit = trigger_category in ('429', 'rate_limit')
        if is_rate_limit or did_escalate:
            self.record_global_rate_limit(keyword)

        # Record rate limit event in predictor for pattern tracking (US-113-003)
        if is_rate_limit or trigger_category:
            tier_name = "tier_unknown"
            if self._keyword_states.get(keyword):
                tier_name = self._keyword_states[keyword].current_tier.name.lower()
            self.record_predictor_rate_limit(
                trigger_category=trigger_category or "unknown",
                tier=tier_name,
                keyword=keyword
            )

        # Track time-of-day failure for adaptive backoff (US-114-004)
        self.record_time_of_day_result(success=False)

        # Track player_client success rate for dynamic selection (US-123-003)
        if player_client:
            self.record_extractor_args_result(player_client, success=False)

        # Track tier failure for graceful degradation (US-123-009)
        # Record the tier that was used (before escalation if escalation happened)
        if self._tier_graceful_degradation and tier_used is not None:
            self._tier_failure_tracker.record_attempt(tier_used, success=False)

    def record_success(self, keyword: str, player_client: Optional[str] = None) -> None:
        """Record a successful download for a keyword.

        Increments consecutive_successes and resets 403 counter. If
        de_escalation is enabled and consecutive_successes reaches the
        threshold, decreases tier by 1.

        Args:
            keyword: The download keyword or video ID.
            player_client: The player_client used (for success rate tracking, US-123-003).
        """
        lock = self._get_lock(keyword)
        with lock:
            state = self._get_state(keyword)
            old_tier = state.current_tier

            # Get de-escalation config from extractor config (with defaults)
            de_escalation_enabled = True
            de_escalation_threshold = 5
            if self._extractor_config is not None:
                de_escalation_enabled = getattr(
                    self._extractor_config, 'de_escalation_enabled', True
                )
                de_escalation_threshold = getattr(
                    self._extractor_config, 'de_escalation_threshold', 5
                )

            de_escalated = state.record_success(
                de_escalation_threshold=de_escalation_threshold,
                de_escalation_enabled=de_escalation_enabled,
            )
            self._metrics.record_success(keyword)

            # Track time-of-day success for adaptive backoff (US-114-004)
            self.record_time_of_day_result(success=True)

            # Track player_client success rate for dynamic selection (US-123-003)
            if player_client:
                self.record_extractor_args_result(player_client, success=True)

            # Track tier success for graceful degradation (US-123-009)
            if self._tier_graceful_degradation:
                self._tier_failure_tracker.record_attempt(state.current_tier, success=True)

            if de_escalated:
                logger.info(
                    f"De-escalation: keyword={keyword} tier {old_tier.name}->{state.current_tier.name} "
                    f"after {de_escalation_threshold} consecutive successes"
                )
                # Log de-escalation tier transition
                from_tier_num = old_tier.value
                to_tier_num = state.current_tier.value
                log_rate_limit(
                    logger, "tier_deescalation", "escalation_manager", "de_escalation",
                    keyword=keyword, from_tier=from_tier_num, to_tier=to_tier_num,
                    consecutive_successes=de_escalation_threshold
                )

    def record_slow_speed(self, keyword: str, speed_mbps: float = 0.0) -> None:
        """Record a slow download speed signal for preemptive escalation.

        When called 3+ times for the same keyword, preemptively escalates
        one tier without waiting for a 403 error. Speed-triggered escalations
        do NOT count toward budget rotations (they are preventive, not reactive).

        Args:
            keyword: The download keyword or video ID.
            speed_mbps: The detected download speed in MB/s (for logging).
        """
        lock = self._get_lock(keyword)
        with lock:
            # Increment slow speed count for this keyword
            count = self._slow_speed_counts.get(keyword, 0) + 1
            self._slow_speed_counts[keyword] = count

            state = self._get_state(keyword)
            # Delegate decision to strategy
            decision = self._strategy.should_escalate_on_slow_speed(state, count)

            if decision.should_escalate:
                old_tier = state.current_tier
                state.escalate()
                # Delegate metrics tracking to EscalationMetrics
                self._metrics.record_escalation(
                    keyword, old_tier, state.current_tier,
                    trigger_category='slow_speed', speed_triggered=True,
                )

                logger.info(
                    f"Preemptive escalation for {keyword}: sustained low speed "
                    f"({speed_mbps:.3f} MB/s) - "
                    f"{old_tier.name} -> {state.current_tier.name} "
                    f"(after {count} slow speed signals)"
                )

                # Log preemptive escalation
                from_tier_num = old_tier.value
                to_tier_num = state.current_tier.value
                log_rate_limit(
                    logger, "tier_escalation", "escalation_manager", "preemptive_escalation",
                    keyword=keyword, from_tier=from_tier_num, to_tier=to_tier_num,
                    slow_speed_count=count, speed_mbps=speed_mbps
                )

                # Reset slow speed count after escalation
                self._slow_speed_counts[keyword] = 0
                # NOTE: No budget.record_rotation() here - speed signals
                # are preventive, not reactive, so they don't consume budget
            elif self._strategy.is_at_max_tier(state.current_tier):
                logger.debug(
                    f"Slow speed for {keyword} ({speed_mbps:.3f} MB/s) "
                    f"but already at max tier"
                )
                # Reset counter since we can't escalate further
                self._slow_speed_counts[keyword] = 0
            elif "cooldown" in decision.reason.lower():
                logger.debug(
                    f"Speed escalation suppressed for {keyword}: cooldown active"
                )
                # Reset counter so signals accumulate again after cooldown
                self._slow_speed_counts[keyword] = 0

    def _should_escalate(self, state: EscalationState) -> bool:
        """Check if escalation should proceed, considering cooldown.

        Returns False if the keyword was escalated within the cooldown
        period, even if the 403 threshold has been reached again.

        Note: This method delegates to the strategy but is kept for
        backward compatibility with any external code that may call it.

        Args:
            state: The per-keyword escalation state.

        Returns:
            True if escalation should proceed, False if in cooldown.
        """
        decision = self._strategy.should_escalate_on_failure(state, budget_exhausted=False)
        return decision.should_escalate

    def _is_past_cooldown(self, state: EscalationState) -> bool:
        """Check if enough time has passed since the last escalation.

        Used by record_slow_speed() to prevent rapid speed-triggered
        escalations within the cooldown window.

        Note: This method delegates to the strategy but is kept for
        backward compatibility.

        Args:
            state: The per-keyword escalation state.

        Returns:
            True if past cooldown (or never escalated), False if in cooldown.
        """
        return self._strategy.is_past_cooldown(state)

    def get_cooldown_remaining(self, keyword: str) -> float:
        """Get remaining cooldown seconds for a keyword.

        Args:
            keyword: The download keyword or video ID.

        Returns:
            Seconds remaining in cooldown, or 0.0 if not in cooldown.
        """
        lock = self._get_lock(keyword)
        with lock:
            state = self._get_state(keyword)
            return self._strategy.get_cooldown_remaining(state)

    def reset_keyword(self, keyword: str) -> None:
        """Clear all escalation state for a keyword.

        Useful for healer or manual recovery scenarios.

        Args:
            keyword: The download keyword or video ID to reset.
        """
        lock = self._get_lock(keyword)
        with lock:
            if keyword in self._keyword_states:
                del self._keyword_states[keyword]
                logger.debug(f"Escalation state reset for keyword={keyword}")

    def reset_all(self) -> None:
        """Clear all keyword escalation states.

        Useful for session restart or full recovery.
        """
        with self._global_lock:
            self._keyword_states.clear()
            self._keyword_locks.clear()
            self._slow_speed_counts.clear()
            self._metrics.reset()
            self._player_client_tracker.reset()
            logger.debug("All escalation states reset")

    def reset_player_client_tracker(self) -> None:
        """Reset the player client success rate tracking.

        Called on config reload to avoid using stale data from previous sessions.
        This ensures dynamic player_client selection starts fresh after config changes.
        """
        self._player_client_tracker.reset()
        logger.debug("Player client tracker reset for config reload")

    def get_active_keyword_count(self) -> int:
        """Get the number of keywords with tracked escalation state.

        Used by circuit breaker to determine what percentage of keywords
        are at Tier 3, which informs whether pause duration should be extended.

        Returns:
            Number of keywords currently tracked.
        """
        with self._global_lock:
            return len(self._keyword_states)

    def get_keywords_at_tier(self, tier: EscalationTier) -> List[str]:
        """Get list of keywords currently at a specific escalation tier.

        Used by circuit breaker to check how many keywords are at Tier 3
        and decide whether to extend pause duration.

        Args:
            tier: The escalation tier to query.

        Returns:
            List of keyword strings at the given tier.
        """
        with self._global_lock:
            return [
                kw for kw, state in self._keyword_states.items()
                if state.current_tier == tier
            ]

    def to_dict(self) -> Dict:
        """Serialize all keyword escalation states for checkpoint persistence.

        Returns:
            Dict with keyword states, metrics data, and a timestamp.
            Format: {
                'keyword_states': {keyword: {tier, consecutive_403s,
                    extractor_args_index, last_escalation_time}},
                'total_403s': int,
                'total_successes': int,
                'total_escalations': int,
                'escalations_per_tier': {tier_name: count},
                'speed_escalations': int,
                'timeline': {keyword: [{timestamp, from_tier, to_tier, trigger_category}]},
                'tier_outcomes': {trigger_category: {tier_value: {successes, attempts}}},
                'saved_at': float (epoch timestamp)
            }
        """
        with self._global_lock:
            keyword_states = {}
            for keyword, state in self._keyword_states.items():
                keyword_states[keyword] = {
                    'tier': state.current_tier.value,
                    'consecutive_403s': state.consecutive_403s,
                    'extractor_args_index': state.extractor_args_index,
                    'last_escalation_time': state.last_escalation_time,
                }
            # Get metrics data from delegated EscalationMetrics
            metrics_data = self._metrics.to_dict()
            # Get adaptive backoff time-of-day data (US-114-004)
            adaptive_backoff_data = self._adaptive_backoff_time_of_day.to_dict()
            # Get player_client tracker data (US-123-003)
            player_client_tracker_data = self._player_client_tracker.to_dict()
            # Get tier failure tracker data (US-123-009)
            tier_failure_tracker_data = self._tier_failure_tracker.to_dict()
            # Get rate limit predictor data (US-136-004)
            predictor_data = None
            if self._rate_limit_predictor is not None:
                predictor_data = self._rate_limit_predictor.to_dict()
            return {
                'keyword_states': keyword_states,
                'total_403s': metrics_data['total_403s'],
                'total_successes': metrics_data['total_successes'],
                'total_escalations': metrics_data['total_escalations'],
                'escalations_per_tier': metrics_data['escalations_per_tier'],
                'speed_escalations': metrics_data['speed_escalations'],
                'timeline': metrics_data['timeline'],
                'tier_outcomes': metrics_data['tier_outcomes'],
                'adaptive_backoff_time_of_day': adaptive_backoff_data,
                'player_client_tracker': player_client_tracker_data,
                'tier_failure_tracker': tier_failure_tracker_data,
                'rate_limit_predictor': predictor_data,
                'predictor_enabled': self._predictor_enabled,
                'saved_at': time.time(),
            }

    @classmethod
    def from_dict(
        cls,
        data: Dict,
        impersonation_manager: "ImpersonationManager",
        extractor_args_config: Optional["ExtractorArgsConfig"] = None,
        budget: Optional["RateLimitBudget"] = None,
        stale_threshold: float = 3600.0,
        strategy: Optional["EscalationStrategy"] = None,
        rate_limit_predictor: Optional[RateLimitPredictor] = None,
    ) -> "EscalationManager":
        """Restore an EscalationManager from checkpoint data.

        Handles stale state: if the checkpoint data is older than
        ``stale_threshold`` seconds, all keywords are de-escalated by one tier
        (YouTube may have relaxed blocking since the last session).

        Args:
            data: Dict previously returned by ``to_dict()``.
            impersonation_manager: Provides Tier 1 --impersonate args.
            extractor_args_config: Configuration for Tier 2 player_client rotation.
            budget: Optional RateLimitBudget for budget-aware escalation.
            stale_threshold: Seconds after which saved data is considered stale
                and keywords are de-escalated by one tier. Default: 3600 (1 hour).
            strategy: Optional EscalationStrategy for decision logic.
                If not provided, a default strategy is created from config.
            rate_limit_predictor: Optional RateLimitPredictor to use. If not provided,
                will be restored from checkpoint data if available.

        Returns:
            A new EscalationManager with restored keyword states.
        """
        # Restore metrics from checkpoint data
        metrics = EscalationMetrics.from_dict(data) if data else EscalationMetrics()

        # Restore or create rate limit predictor (US-136-004)
        predictor_enabled = data.get('predictor_enabled', True) if data else True
        predictor = rate_limit_predictor
        if predictor is None:
            predictor_data = data.get('rate_limit_predictor') if data else None
            if predictor_data:
                predictor = RateLimitPredictor.from_dict(predictor_data)

        manager = cls(
            impersonation_manager=impersonation_manager,
            extractor_args_config=extractor_args_config,
            budget=budget,
            strategy=strategy,
            metrics=metrics,
            rate_limit_predictor=predictor,
            predictor_enabled=predictor_enabled,
        )

        if not data or not isinstance(data, dict):
            logger.warning("Empty or invalid escalation checkpoint data, starting fresh")
            return manager

        # Check staleness
        saved_at = data.get('saved_at', 0.0)
        age = time.time() - saved_at
        is_stale = age > stale_threshold

        if is_stale:
            logger.info(
                f"Escalation checkpoint is stale ({age:.0f}s > {stale_threshold:.0f}s threshold), "
                f"de-escalating all keywords by one tier"
            )

        # Restore keyword states
        keyword_states = data.get('keyword_states', {})
        for keyword, state_data in keyword_states.items():
            tier_value = state_data.get('tier', EscalationTier.IMPERSONATE_ONLY.value)
            # Clamp to valid tier range
            tier_value = max(
                EscalationTier.IMPERSONATE_ONLY.value,
                min(tier_value, EscalationTier.FULL_BYPASS.value),
            )

            if is_stale and tier_value > EscalationTier.IMPERSONATE_ONLY.value:
                tier_value -= 1

            state = EscalationState(
                current_tier=EscalationTier(tier_value),
                consecutive_403s=state_data.get('consecutive_403s', 0),
                last_escalation_time=state_data.get('last_escalation_time'),
                extractor_args_index=state_data.get('extractor_args_index', 0),
            )
            manager._keyword_states[keyword] = state

        # Restore adaptive backoff time-of-day data (US-114-004)
        adaptive_backoff_data = data.get('adaptive_backoff_time_of_day')
        if adaptive_backoff_data:
            manager._adaptive_backoff_time_of_day = AdaptiveBackoffTimeOfDay.from_dict(adaptive_backoff_data)

        # Restore player_client tracker data (US-123-003)
        player_client_tracker_data = data.get('player_client_tracker')
        if player_client_tracker_data:
            manager._player_client_tracker = PlayerClientTracker.from_dict(player_client_tracker_data)

        # Restore tier failure tracker data (US-123-009)
        tier_failure_tracker_data = data.get('tier_failure_tracker')
        if tier_failure_tracker_data:
            manager._tier_failure_tracker = TierFailureTracker.from_dict(tier_failure_tracker_data)

        restored_count = len(keyword_states)
        logger.info(
            f"Restored escalation state for {restored_count} keywords"
            f"{' (de-escalated due to stale data)' if is_stale else ''}"
        )

        return manager

    @classmethod
    def create_with_mullvad(
        cls,
        impersonation_manager: "ImpersonationManager",
        extractor_args_config: Optional["ExtractorArgsConfig"] = None,
        mullvad_config: Optional["MullvadConfig"] = None,
        budget: Optional["RateLimitBudget"] = None,
        strategy: Optional["EscalationStrategy"] = None,
    ) -> "EscalationManager":
        """Create an EscalationManager with MullvadVPN pre-wired if enabled.

        Factory method that instantiates both EscalationManager and MullvadVPN
        (if mullvad_config.enabled is True), wiring them together automatically.
        This simplifies setup compared to manually creating both and calling
        set_mullvad_vpn().

        Args:
            impersonation_manager: Provides Tier 1 --impersonate args.
            extractor_args_config: Configuration for Tier 2 player_client rotation.
            mullvad_config: MullvadConfig for Tier 4 VPN rotation. If None or
                mullvad_config.enabled is False, MullvadVPN is not created.
            budget: Optional RateLimitBudget for budget-aware escalation.
            strategy: Optional EscalationStrategy for decision logic.

        Returns:
            A new EscalationManager with MullvadVPN wired if enabled.
        """
        manager = cls(
            impersonation_manager=impersonation_manager,
            extractor_args_config=extractor_args_config,
            budget=budget,
            strategy=strategy,
        )

        # Wire MullvadVPN if config is enabled
        if mullvad_config and getattr(mullvad_config, 'enabled', False):
            # Import here to avoid circular import at module level
            from .mullvad_vpn import MullvadVPN
            mullvad_vpn = MullvadVPN(mullvad_config)
            manager.set_mullvad_vpn(mullvad_vpn)
            logger.debug(
                "EscalationManager created with MullvadVPN for Tier 4 bypass"
            )
        else:
            logger.debug(
                "EscalationManager created without MullvadVPN (disabled or no config)"
            )

        return manager

    def get_metrics(self) -> Dict:
        """Get escalation metrics summary.

        Returns:
            Dict with:
                total_escalations: Total number of tier escalations
                escalations_per_tier: Dict mapping tier name to escalation count
                keywords_at_each_tier: Dict mapping tier name to list of keywords
                total_403s: Total 403/bot-detection errors recorded
                total_successes: Total successful downloads recorded
                average_tier: Weighted average tier across all tracked keywords (1.0-3.0)
                speed_escalations: Total escalations triggered by slow speed signals
        """
        with self._global_lock:
            keywords_at_each_tier: Dict[str, List[str]] = {}
            tier_sum = 0.0
            keyword_count = 0

            for keyword, state in self._keyword_states.items():
                tier_name = state.current_tier.name
                if tier_name not in keywords_at_each_tier:
                    keywords_at_each_tier[tier_name] = []
                keywords_at_each_tier[tier_name].append(keyword)
                tier_sum += float(state.current_tier.value)
                keyword_count += 1

            average_tier = round(tier_sum / keyword_count, 2) if keyword_count > 0 else 1.0

            # Merge keyword-state info with metrics summary
            summary = self._metrics.get_summary()
            summary['keywords_at_each_tier'] = keywords_at_each_tier
            summary['average_tier'] = average_tier
            return summary

    def get_keyword_escalation_timeline(self) -> Dict[str, List[Dict]]:
        """Get per-keyword escalation event timeline.

        Delegates to EscalationMetrics.get_timeline().

        Returns:
            Dict mapping keyword to list of event dicts.
        """
        return self._metrics.get_timeline()

    def get_hot_keywords(self, window_seconds: float = 1800.0) -> List[Dict]:
        """Get keywords with frequent escalations in a recent time window.

        Delegates to EscalationMetrics.get_hot_keywords().

        Args:
            window_seconds: Time window in seconds (default: 1800 = 30 min).

        Returns:
            List of dicts with 'keyword' and 'escalation_count', sorted by
            count descending. Only includes keywords with >3 escalations.
        """
        return self._metrics.get_hot_keywords(window_seconds)

    def record_outcome(
        self, trigger_category: str, tier: EscalationTier, success: bool
    ) -> None:
        """Record a download outcome for tier effectiveness tracking.

        Delegates to EscalationMetrics.record_outcome().

        Args:
            trigger_category: Category from classify_trigger() (e.g., '403', '429').
            tier: The escalation tier used for this attempt.
            success: True if the download succeeded, False if it failed.
        """
        self._metrics.record_outcome(trigger_category, tier, success)

    def get_tier_effectiveness(self) -> Dict[str, Dict[str, float]]:
        """Get success rate per trigger category per escalation tier.

        Delegates to EscalationMetrics.get_tier_effectiveness().

        Returns:
            Dict mapping trigger_category to {tier_name: success_rate}.
            Empty dict if no outcomes have been recorded.
        """
        return self._metrics.get_tier_effectiveness()

    def get_tier_recommendations(
        self,
        remaining_budget: Optional[int] = None,
    ) -> List[str]:
        """Generate recommendations based on tier effectiveness data.

        Delegates to EscalationMetrics.get_tier_recommendations().

        Args:
            remaining_budget: Optional remaining budget for downloads. When provided,
                recommendations will suggest skipping low-effectiveness tiers when
                budget is low (< 20).

        Returns:
            List of recommendation strings. Empty list if no data or
            no recommendations apply.
        """
        return self._metrics.get_tier_recommendations(remaining_budget)

    def get_expected_success_rate(self, trigger_category: str) -> Optional[float]:
        """Get expected success rate for a trigger category.

        Delegates to EscalationMetrics.get_expected_success_rate().

        Args:
            trigger_category: Category from classify_trigger() (e.g., '403', '429').

        Returns:
            Weighted average success rate (0.0-1.0), or None if no data available.
        """
        return self._metrics.get_expected_success_rate(trigger_category)

    def get_best_tier_for_category(
        self,
        trigger_category: str,
        min_success_rate: float = 0.60,
    ) -> Optional[EscalationTier]:
        """Get the best tier for a trigger category based on historical success rates.

        Delegates to EscalationMetrics.get_best_tier_for_category().

        Args:
            trigger_category: Category from classify_trigger() (e.g., '403', '429').
            min_success_rate: Minimum success rate threshold (default 0.60 = 60%).

        Returns:
            The best EscalationTier that meets the threshold, or None if
            no tier meets the minimum.
        """
        return self._metrics.get_best_tier_for_category(trigger_category, min_success_rate)

    def select_weighted_tier(
        self,
        trigger_category: str,
        default_tier: EscalationTier = EscalationTier.IMPERSONATE_ONLY,
    ) -> EscalationTier:
        """Select tier using weighted selection that prefers >60% historical success rate.

        This implements ML-style pattern recognition for tier selection:
        1. Check if we have historical data for this trigger category
        2. If a tier has >60% historical success rate, prefer it
        3. Otherwise fall back to the default tier

        Args:
            trigger_category: Category from classify_trigger() (e.g., '403', '429').
            default_tier: Default tier to use if no historical data or no tier
                meets the success rate threshold.

        Returns:
            The selected EscalationTier, preferring tiers with >60% historical
            success rate when data is available.
        """
        # Try to find a tier with >60% success rate
        best_tier = self.get_best_tier_for_category(
            trigger_category, min_success_rate=0.60
        )

        if best_tier is not None:
            return best_tier

        # Fall back to default tier
        return default_tier
