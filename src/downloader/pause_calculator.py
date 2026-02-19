"""Composable pause duration calculator for circuit breakers.

Extracted from CircuitBreaker's 5-step pause pipeline to make pause logic
independently testable and reusable. Each step is a standalone method that
takes a PauseContext and returns an updated pause value.

Pipeline order: base_pause → escalation_adjusted → budget_adjusted → region_adjusted → time_adjusted → apply_jitter → cap_duration

Extracted as part of US-82-008.

US-109-002 Improvements:
- Added jitter strategies: 'random', 'adaptive', 'deterministic'
- Added time-of-day based adaptive jitter (higher during peak hours)
- Added jitter correlation check to prevent similar values across clients
- Hard cap at 50% jitter regardless of strategy

US-109-011 Improvements:
- Added region detection via VPN country or IP geolocation
- Added region-specific backoff multipliers: US=1.0, EU=1.2, ASIA=1.5, OTHER=2.0
- Added regional rate limit tracking with separate counters per region

US-144-010 Improvements:
- Added time-of-day aware backoff multipliers
- Time windows: overnight (10pm-6am), morning (6am-12pm), afternoon (12pm-6pm), evening (6pm-10pm)
- Higher multipliers for evening (1.5x) and afternoon (1.2x)
- Lower multipliers for overnight (0.8x)
- Weekend vs weekday differentiation (1.3x weekend multiplier)
"""

from __future__ import annotations

import hashlib
import logging
import random
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from .escalation_manager import EscalationManager
    from .rate_limit_budget import RateLimitBudget

# Use the circuit_breaker logger to preserve backward-compatible log filtering
logger = logging.getLogger('src.downloader.circuit_breaker')


class JitterStrategy(str, Enum):
    """Jitter calculation strategy for backoff delays.

    - RANDOM: Classic uniform random jitter
    - ADAPTIVE: Time-of-day based jitter (higher during peak hours)
    - DETERMINISTIC: Seeded random based on client_id for reproducibility
    """
    RANDOM = "random"
    ADAPTIVE = "adaptive"
    DETERMINISTIC = "deterministic"


@dataclass
class PauseContext:
    """Context for pause calculation pipeline steps.

    Carries the current pause value plus all inputs needed by
    individual pipeline steps. Each step reads from this context
    and returns the updated pause value.
    """
    current_pause: float = 0.0
    base_pause_seconds: float = 60.0
    max_pause_seconds: float = 300.0
    jitter_factor: float = 0.2
    # US-109-002: New jitter configuration
    jitter_strategy: str = "random"  # 'random', 'adaptive', 'deterministic'
    jitter_max_factor: float = 0.5  # Hard cap at 50%
    jitter_correlation_check: bool = False  # Check for similar values
    client_id: str = ""  # For deterministic jitter
    escalation_manager: Optional['EscalationManager'] = None
    budget: Optional['RateLimitBudget'] = None
    # US-109-011: Region-specific backoff configuration
    region: str = ""  # Current region: 'us', 'eu', 'asia', 'other'
    country_code: str = ""  # ISO country code (e.g., 'us', 'de')
    region_multipliers: dict = field(default_factory=lambda: {
        'us': 1.0, 'eu': 1.2, 'asia': 1.5, 'other': 2.0
    })  # Region-specific multiplier mapping
    region_enabled: bool = False  # Whether region-specific backoff is enabled

    # US-113-008: Dynamic region-based backoff
    # Success rates per region (region -> success_rate 0-1)
    region_success_rates: dict = field(default_factory=dict)
    # Minimum success rate threshold - regions below this get increased backoff
    region_success_rate_threshold: float = 0.7
    # Whether dynamic region adjustment is enabled
    dynamic_region_adjustment: bool = False

    # US-144-010: Time-of-day aware backoff
    time_of_day_enabled: bool = False  # Whether time-of-day backoff is enabled
    time_of_day_hour: Optional[int] = None  # Hour of day (0-23) for backoff calculation
    time_of_day_is_weekend: bool = False  # Whether it's a weekend
    time_of_day_timestamp: Optional[float] = None  # Unix timestamp (overrides hour/weekend)
    # Time window multipliers
    time_window_multipliers: dict = field(default_factory=lambda: {
        'overnight': 0.8, 'morning': 1.0, 'afternoon': 1.2, 'evening': 1.5
    })
    # Weekend multiplier (applied on top of time window multiplier)
    weekend_multiplier: float = 1.3


# Class-level store for recent jitter values (for correlation check)
class JitterCorrelationStore:
    """Store recent jitter values to detect correlation patterns."""

    _recent_values: list = []
    _max_store_size: int = 100

    @classmethod
    def add(cls, value: float) -> None:
        """Add a jitter value to the store."""
        cls._recent_values.append(value)
        if len(cls._recent_values) > cls._max_store_size:
            cls._recent_values.pop(0)

    @classmethod
    def get_recent(cls, count: int = 10) -> list:
        """Get the most recent N jitter values."""
        return cls._recent_values[-count:]

    @classmethod
    def is_correlated(cls, value: float, threshold: float = 0.1) -> bool:
        """Check if value is too similar to recent values."""
        recent = cls.get_recent(10)
        if not recent:
            return False
        # Check if within threshold of average of recent values
        avg_recent = sum(recent) / len(recent)
        return abs(value - avg_recent) < threshold

    @classmethod
    def clear(cls) -> None:
        """Clear the store (useful for testing)."""
        cls._recent_values = []


class RegionalRateLimitTracker:
    """Track rate limits separately per region (US-109-011).

    When region-specific tracking is enabled, each region maintains
    independent rate limit counters. This allows:
    - US region to recover while ASIA is still rate-limited
    - Different backoff strategies per region
    - Better utilization of multi-region VPN setups

    Region mapping:
    - 'us': United States, Canada
    - 'eu': European countries
    - 'asia': Asian countries
    - 'other': All other countries
    """

    _region_counts: dict = {}
    _enabled: bool = False

    @classmethod
    def enable(cls) -> None:
        """Enable regional rate limit tracking."""
        cls._enabled = True
        cls._region_counts = {'us': 0, 'eu': 0, 'asia': 0, 'other': 0}

    @classmethod
    def disable(cls) -> None:
        """Disable regional rate limit tracking."""
        cls._enabled = False

    @classmethod
    def is_enabled(cls) -> bool:
        """Check if regional tracking is enabled."""
        return cls._enabled

    @classmethod
    def _country_to_region(cls, country_code: str) -> str:
        """Map ISO country code to region name.

        Args:
            country_code: ISO 3166-1 alpha-2 country code or region name

        Returns:
            Region name: 'us', 'eu', 'asia', or 'other'
        """
        if not country_code:
            return 'other'

        country = country_code.lower()

        # First check if it's already a valid region name
        if country in ('us', 'eu', 'asia', 'other'):
            return country

        # US/Canada
        if country in ('us', 'ca'):
            return 'us'

        # European countries
        eu_countries = {
            'gb', 'de', 'nl', 'se', 'ch', 'fr', 'it', 'es', 'pl', 'be',
            'at', 'ie', 'no', 'fi', 'dk', 'pt', 'el', 'cz', 'hu', 'ro',
            'bg', 'sk', 'si', 'hr', 'lt', 'lv', 'ee', 'cy', 'lu', 'mt'
        }
        if country in eu_countries:
            return 'eu'

        # Asian countries
        asia_countries = {
            'jp', 'sg', 'kr', 'in', 'id', 'my', 'th', 'vn', 'ph', 'tw',
            'hk', 'cn', 'pk', 'bd', 'lk', 'np', 'mm', 'kh', 'la'
        }
        if country in asia_countries:
            return 'asia'

        return 'other'

    @classmethod
    def increment(cls, region: str) -> None:
        """Increment the rate limit counter for a region.

        Args:
            region: ISO country code (e.g., 'us', 'de') or region name (e.g., 'eu')
        """
        if not cls._enabled:
            return
        # Map country code to region
        normalized = cls._country_to_region(region) if region else 'other'
        cls._region_counts[normalized] = cls._region_counts.get(normalized, 0) + 1

    @classmethod
    def get_count(cls, region: str) -> int:
        """Get the rate limit count for a region."""
        if not cls._enabled:
            return 0
        normalized = region.lower() if region else 'other'
        return cls._region_counts.get(normalized, 0)

    @classmethod
    def reset(cls, region: Optional[str] = None) -> None:
        """Reset rate limit counter(s).

        Args:
            region: If provided, reset only that region. Otherwise reset all.
        """
        if not cls._enabled:
            return
        if region:
            normalized = region.lower() if region else 'other'
            cls._region_counts[normalized] = 0
        else:
            cls._region_counts = {'us': 0, 'eu': 0, 'asia': 0, 'other': 0}

    @classmethod
    def get_all_counts(cls) -> dict:
        """Get all region rate limit counts."""
        return cls._region_counts.copy() if cls._enabled else {}

    @classmethod
    def clear(cls) -> None:
        """Clear all tracking (useful for testing)."""
        cls._region_counts = {}
        cls._enabled = False


class RegionSuccessTracker:
    """Tracks success/failure rates per region for dynamic backoff adjustment (US-113-008).

    Aggregates success/failure data from individual servers (countries) into
    region-level statistics. Used to dynamically adjust backoff multipliers
    based on regional success rates.

    Usage:
        # Record successes and failures by country code
        RegionSuccessTracker.record_success('us')  # Server in US succeeded
        RegionSuccessTracker.record_failure('de')  # Server in Germany failed

        # Get success rate for a region (aggregates all countries in that region)
        rate = RegionSuccessTracker.get_success_rate('eu')  # Returns 0.0-1.0

        # Get all region success rates
        rates = RegionSuccessTracker.get_all_rates()
    """

    # Class-level storage for region success tracking
    _region_attempts: dict = {'us': 0, 'eu': 0, 'asia': 0, 'other': 0}
    _region_successes: dict = {'us': 0, 'eu': 0, 'asia': 0, 'other': 0}
    _enabled: bool = False
    _min_sample_size: int = 3  # Minimum attempts before trusting the rate

    # Country to region mapping (same as RegionalRateLimitTracker)
    _COUNTRY_MAP: dict = {
        # US/Canada
        'us': 'us', 'ca': 'us',
        # EU countries
        'gb': 'eu', 'de': 'eu', 'fr': 'eu', 'nl': 'eu', 'se': 'eu', 'ch': 'eu',
        'it': 'eu', 'es': 'eu', 'pl': 'eu', 'be': 'eu', 'at': 'eu', 'ie': 'eu',
        'no': 'eu', 'fi': 'eu', 'dk': 'eu',
        # Asian countries
        'jp': 'asia', 'sg': 'asia', 'kr': 'asia', 'in': 'asia', 'id': 'asia',
        'my': 'asia', 'th': 'asia', 'vn': 'asia', 'ph': 'asia', 'tw': 'asia',
    }

    @classmethod
    def _country_to_region(cls, country: str) -> str:
        """Map country code to region name."""
        if not country:
            return 'other'
        normalized = country.lower()
        return cls._COUNTRY_MAP.get(normalized, 'other')

    @classmethod
    def enable(cls) -> None:
        """Enable region success tracking."""
        cls._enabled = True
        logger.debug("RegionSuccessTracker enabled")

    @classmethod
    def disable(cls) -> None:
        """Disable region success tracking."""
        cls._enabled = False
        logger.debug("RegionSuccessTracker disabled")

    @classmethod
    def is_enabled(cls) -> bool:
        """Check if tracking is enabled."""
        return cls._enabled

    @classmethod
    def record_success(cls, country: str) -> None:
        """Record a successful download attempt for a country.

        Args:
            country: ISO country code (e.g., 'us', 'de', 'jp')
        """
        if not cls._enabled:
            return
        region = cls._country_to_region(country)
        cls._region_attempts[region] = cls._region_attempts.get(region, 0) + 1
        cls._region_successes[region] = cls._region_successes.get(region, 0) + 1
        logger.debug(
            f"RegionSuccessTracker: recorded success for {country} -> {region} "
            f"(rate={cls.get_success_rate(region):.1%})"
        )

    @classmethod
    def record_failure(cls, country: str) -> None:
        """Record a failed download attempt for a country.

        Args:
            country: ISO country code (e.g., 'us', 'de', 'jp')
        """
        if not cls._enabled:
            return
        region = cls._country_to_region(country)
        cls._region_attempts[region] = cls._region_attempts.get(region, 0) + 1
        # successes not incremented for failures
        logger.debug(
            f"RegionSuccessTracker: recorded failure for {country} -> {region} "
            f"(rate={cls.get_success_rate(region):.1%})"
        )

    @classmethod
    def get_success_rate(cls, region: str) -> float:
        """Get success rate for a region.

        Args:
            region: Region name ('us', 'eu', 'asia', 'other')

        Returns:
            Success rate as float between 0 and 1.
            Returns 0.5 for untested regions (neutral rate).
            Returns 0.5 for regions with fewer than min_sample_size attempts.
        """
        if not cls._enabled:
            return 0.5
        normalized = region.lower() if region else 'other'
        attempts = cls._region_attempts.get(normalized, 0)
        successes = cls._region_successes.get(normalized, 0)

        # Return neutral rate for untested or low-sample regions
        if attempts < cls._min_sample_size:
            return 0.5

        return successes / attempts

    @classmethod
    def get_all_rates(cls) -> dict:
        """Get success rates for all regions.

        Returns:
            Dict mapping region name to success rate (0-1).
        """
        if not cls._enabled:
            return {}
        return {
            region: cls.get_success_rate(region)
            for region in ['us', 'eu', 'asia', 'other']
        }

    @classmethod
    def get_attempts(cls, region: str) -> int:
        """Get total attempts for a region."""
        if not cls._enabled:
            return 0
        normalized = region.lower() if region else 'other'
        return cls._region_attempts.get(normalized, 0)

    @classmethod
    def get_all_attempts(cls) -> dict:
        """Get attempts for all regions."""
        if not cls._enabled:
            return {}
        return cls._region_attempts.copy()

    @classmethod
    def is_below_threshold(cls, region: str, threshold: float) -> bool:
        """Check if region's success rate is below threshold.

        Args:
            region: Region name ('us', 'eu', 'asia', 'other')
            threshold: Success rate threshold (0-1)

        Returns:
            True if success rate is below threshold.
            Returns False for untested regions (allow them to be tried).
        """
        if not cls._enabled:
            return False
        rate = cls.get_success_rate(region)
        # Don't avoid untested regions - let them be tried
        if cls.get_attempts(region) < cls._min_sample_size:
            return False
        return rate < threshold

    @classmethod
    def reset(cls, region: Optional[str] = None) -> None:
        """Reset success tracking.

        Args:
            region: If provided, reset only that region. Otherwise reset all.
        """
        if region:
            normalized = region.lower() if region else 'other'
            cls._region_attempts[normalized] = 0
            cls._region_successes[normalized] = 0
        else:
            cls._region_attempts = {'us': 0, 'eu': 0, 'asia': 0, 'other': 0}
            cls._region_successes = {'us': 0, 'eu': 0, 'asia': 0, 'other': 0}

    @classmethod
    def get_best_region(cls, exclude_regions: Optional[List[str]] = None) -> Optional[str]:
        """Get the best performing region based on success rate.

        US-136-007: Added method to get best performing region for next connection.

        Args:
            exclude_regions: Optional list of regions to exclude from selection

        Returns:
            Region name with highest success rate, or None if no data.
            Only considers regions with at least min_sample_size attempts.
            If all regions have insufficient data, returns None.
        """
        if not cls._enabled:
            return None

        exclude_set = set(exclude_regions) if exclude_regions else set()
        best_region = None
        best_rate = -1.0

        for region in ['us', 'eu', 'asia', 'other']:
            if region in exclude_set:
                continue
            attempts = cls.get_attempts(region)
            # Only consider regions with sufficient sample size
            if attempts < cls._min_sample_size:
                continue
            rate = cls.get_success_rate(region)
            if rate > best_rate:
                best_rate = rate
                best_region = region

        return best_region

    @classmethod
    def get_regions_below_threshold(cls, threshold: float) -> List[str]:
        """Get list of regions with success rate below threshold.

        US-136-007: Added method for identifying underperforming regions.

        Args:
            threshold: Success rate threshold (0-1)

        Returns:
            List of region names with success rate below threshold.
        """
        if not cls._enabled:
            return []

        below_threshold = []
        for region in ['us', 'eu', 'asia', 'other']:
            if cls.is_below_threshold(region, threshold):
                below_threshold.append(region)
        return below_threshold

    @classmethod
    def to_checkpoint_state(cls) -> dict:
        """Serialize region success tracking state for checkpoint persistence.

        US-136-007: Add persistence of regional success data to checkpoint.

        Returns:
            Dict with region attempts and successes
        """
        return {
            'region_attempts': cls._region_attempts.copy(),
            'region_successes': cls._region_successes.copy(),
            'enabled': cls._enabled,
            'min_sample_size': cls._min_sample_size,
        }

    @classmethod
    def restore_from_checkpoint(cls, state: dict) -> None:
        """Restore region success tracking state from checkpoint.

        US-136-007: Add persistence of regional success data to checkpoint.

        Args:
            state: Dict with region attempts and successes from checkpoint
        """
        if not state:
            return

        cls._region_attempts = state.get('region_attempts', {
            'us': 0, 'eu': 0, 'asia': 0, 'other': 0
        })
        cls._region_successes = state.get('region_successes', {
            'us': 0, 'eu': 0, 'asia': 0, 'other': 0
        })
        cls._enabled = state.get('enabled', False)
        cls._min_sample_size = state.get('min_sample_size', 3)

    @classmethod
    def clear(cls) -> None:
        """Clear all tracking (useful for testing)."""
        cls._region_attempts = {'us': 0, 'eu': 0, 'asia': 0, 'other': 0}
        cls._region_successes = {'us': 0, 'eu': 0, 'asia': 0, 'other': 0}
        cls._enabled = False


class TimeOfDayBackoffManager:
    """Time-of-day aware backoff multiplier manager (US-144-010).

    Applies different backoff multipliers based on time of day and day of week:
    - Time windows:
      - OVERNIGHT (10pm-6am): 0.8x (YouTube less busy)
      - MORNING (6am-12pm): 1.0x (baseline)
      - AFTERNOON (12pm-6pm): 1.2x (higher usage)
      - EVENING (6pm-10pm): 1.5x (peak usage)
    - Weekend multiplier: 1.3x additional boost for weekends

    Usage:
        manager = TimeOfDayBackoffManager()
        multiplier = manager.get_multiplier(hour=20, is_weekend=False)  # Returns 1.5
        multiplier = manager.get_multiplier(hour=3, is_weekend=False)   # Returns 0.8
        multiplier = manager.get_multiplier(hour=14, is_weekend=True)  # Returns 1.56 (1.2 * 1.3)
    """

    # Time window definitions (hour ranges)
    WINDOW_RANGES = {
        'overnight': (22, 6),   # 10pm - 6am
        'morning': (6, 12),      # 6am - 12pm
        'afternoon': (12, 18),   # 12pm - 6pm
        'evening': (18, 22),     # 6pm - 10pm
    }

    # Default time window multipliers
    DEFAULT_WINDOW_MULTIPLIERS = {
        'overnight': 0.8,
        'morning': 1.0,
        'afternoon': 1.2,
        'evening': 1.5,
    }

    # Weekend multiplier (applied on top of time window multiplier)
    DEFAULT_WEEKEND_MULTIPLIER = 1.3

    def __init__(self, window_multipliers: Optional[dict] = None,
                 weekend_multiplier: float = DEFAULT_WEEKEND_MULTIPLIER,
                 enabled: bool = True):
        """Initialize the time-of-day backoff manager.

        Args:
            window_multipliers: Custom time window multipliers (optional)
            weekend_multiplier: Weekend multiplier (default 1.3)
            enabled: Whether time-of-day backoff is enabled (default True)
        """
        self._window_multipliers = window_multipliers or self.DEFAULT_WINDOW_MULTIPLIERS.copy()
        self._weekend_multiplier = weekend_multiplier
        self._enabled = enabled

    @property
    def enabled(self) -> bool:
        """Check if time-of-day backoff is enabled."""
        return self._enabled

    def enable(self) -> None:
        """Enable time-of-day backoff."""
        self._enabled = True

    def disable(self) -> None:
        """Disable time-of-day backoff."""
        self._enabled = False

    def _get_time_window(self, hour: int) -> str:
        """Determine time window from hour of day.

        Args:
            hour: Hour of day (0-23)

        Returns:
            Time window name: 'overnight', 'morning', 'afternoon', 'evening'
        """
        # Handle overnight specially (22-23 and 0-5)
        if hour >= 22 or hour < 6:
            return 'overnight'
        elif 6 <= hour < 12:
            return 'morning'
        elif 12 <= hour < 18:
            return 'afternoon'
        else:  # 18 <= hour < 22
            return 'evening'

    def get_multiplier(self, hour: Optional[int] = None,
                       is_weekend: bool = False,
                       timestamp: Optional[float] = None) -> float:
        """Get time-of-day aware backoff multiplier.

        Args:
            hour: Hour of day (0-23). If not provided, uses current hour.
            is_weekend: Whether it's a weekend. If not provided, derived from timestamp.
            timestamp: Unix timestamp. If provided, extracts hour and is_weekend from it.

        Returns:
            Time-of-day multiplier (0.0-2.0 typically)
        """
        if not self._enabled:
            return 1.0

        # Determine hour and weekend from timestamp if provided
        if timestamp is not None:
            dt = datetime.fromtimestamp(timestamp)
            hour = dt.hour
            is_weekend = dt.weekday() >= 5  # Saturday=5, Sunday=6

        # Default to current time if not provided
        if hour is None:
            hour = datetime.now().hour

        # Get time window
        window = self._get_time_window(hour)

        # Get base multiplier for time window
        multiplier = self._window_multipliers.get(window, 1.0)

        # Apply weekend multiplier if applicable
        if is_weekend:
            multiplier = multiplier * self._weekend_multiplier

        return multiplier

    def get_window_info(self, hour: Optional[int] = None,
                        timestamp: Optional[float] = None) -> dict:
        """Get detailed time window information.

        Args:
            hour: Hour of day (0-23). If not provided, uses current hour.
            timestamp: Unix timestamp. If provided, extracts hour from it.

        Returns:
            Dict with window name, multiplier breakdown, and is_weekend
        """
        if timestamp is not None:
            dt = datetime.fromtimestamp(timestamp)
            hour = dt.hour
            is_weekend = dt.weekday() >= 5
        elif hour is None:
            hour = datetime.now().hour
            is_weekend = False
        else:
            is_weekend = False

        window = self._get_time_window(hour)
        base_multiplier = self._window_multipliers.get(window, 1.0)
        weekend_mult = self._weekend_multiplier if is_weekend else 1.0
        total_multiplier = base_multiplier * weekend_mult

        return {
            'hour': hour,
            'time_window': window,
            'is_weekend': is_weekend,
            'base_multiplier': base_multiplier,
            'weekend_multiplier': weekend_mult,
            'total_multiplier': total_multiplier,
        }


# Global instance for convenience
_time_of_day_manager: Optional[TimeOfDayBackoffManager] = None


def get_time_of_day_manager() -> TimeOfDayBackoffManager:
    """Get the global TimeOfDayBackoffManager instance."""
    global _time_of_day_manager
    if _time_of_day_manager is None:
        _time_of_day_manager = TimeOfDayBackoffManager()
    return _time_of_day_manager


def set_time_of_day_manager(manager: TimeOfDayBackoffManager) -> None:
    """Set the global TimeOfDayBackoffManager instance."""
    global _time_of_day_manager
    _time_of_day_manager = manager


class PauseCalculator:
    """Composable pause duration calculator.

    Composes 5 steps into a pipeline:
    1. base_pause() - Get base duration from config
    2. escalation_adjusted() - Double pause when >50% keywords at Tier 3
    3. budget_adjusted() - Scale pause based on rate limit budget state
    4. apply_jitter() - Randomize ±jitter_factor to prevent thundering herd
    5. cap_duration() - Enforce max_pause_seconds ceiling

    Each step can be called independently for unit testing, or composed
    via calculate() for the full pipeline.
    """

    def __init__(self) -> None:
        self._last_jitter_applied: float = 0.0

    @property
    def last_jitter_applied(self) -> float:
        """Last jitter multiplier offset applied (for debugging/metrics)."""
        return self._last_jitter_applied

    def calculate(self, ctx: PauseContext) -> float:
        """Run the full 7-step pause calculation pipeline.

        Args:
            ctx: PauseContext with all inputs for the pipeline.

        Returns:
            Final pause duration in seconds.
        """
        pause = self.base_pause(ctx)
        pause = self.escalation_adjusted(pause, ctx)
        pause = self.budget_adjusted(pause, ctx)
        pause = self.region_adjusted(pause, ctx)  # US-109-011
        pause = self.time_adjusted(pause, ctx)  # US-144-010
        pause = self.apply_jitter(pause, ctx)
        pause = self.cap_duration(pause, ctx)
        return pause

    def base_pause(self, ctx: PauseContext) -> float:
        """Step 1: Get the base pause duration from context."""
        return ctx.base_pause_seconds

    def escalation_adjusted(self, pause: float, ctx: PauseContext) -> float:
        """Step 2: Apply escalation-based extension to a pause duration.

        Doubles the pause when >50% of active keywords are at Tier 3.
        """
        if ctx.escalation_manager is None:
            return pause

        try:
            from .types import EscalationTier

            total_keywords = ctx.escalation_manager.get_active_keyword_count()
            if total_keywords > 0:
                tier3_keywords = ctx.escalation_manager.get_keywords_at_tier(
                    EscalationTier.FULL_BYPASS
                )
                tier3_pct = len(tier3_keywords) / total_keywords

                if tier3_pct > 0.5:
                    adjusted = pause * 2.0
                    logger.info(
                        f"Circuit breaker extended: {tier3_pct:.0%} keywords at Tier 3 "
                        f"(pause {ctx.base_pause_seconds:.0f}s -> {adjusted:.0f}s)"
                    )
                    return adjusted
        except ImportError:
            pass

        return pause

    def budget_adjusted(self, pause: float, ctx: PauseContext) -> float:
        """Step 3: Apply budget-aware extension to a pause duration."""
        if ctx.budget is None:
            return pause

        original_pause = pause
        if ctx.budget.is_exhausted():
            pause = pause * 2.5
            budget_status = "exhausted"
        elif ctx.budget.is_nearly_exhausted():
            pause = pause * 1.5
            budget_status = "nearly exhausted"
        else:
            budget_status = None

        if budget_status is not None:
            logger.info(
                f"Circuit breaker pause extended {original_pause:.0f}s -> "
                f"{pause:.0f}s (budget {budget_status})"
            )

        return pause

    def region_adjusted(self, pause: float, ctx: PauseContext) -> float:
        """Step 4 (US-109-011, US-113-008): Apply region-specific multiplier to pause duration.

        Different geographic regions have different rate limit tolerance from YouTube.
        This applies region-specific multipliers:
        - US: 1.0 (baseline)
        - EU: 1.2 (20% longer backoff)
        - ASIA: 1.5 (50% longer backoff)
        - OTHER: 2.0 (100% longer backoff)

        US-113-008: When dynamic_region_adjustment is enabled, also applies additional
        multiplier based on regional success rate:
        - Regions with success rate below threshold get increased backoff
        - Success rate is calculated from tracked successes/failures per region
        - Regions below threshold get: base_multiplier * (1 + (threshold - success_rate))

        Region can be set via:
        - VPN rotation country code (preferred)
        - IP geolocation lookup
        - Manual configuration
        """
        # Skip if region-specific backoff is disabled
        if not ctx.region_enabled:
            return pause

        # Use region from context, default to 'other' if not set
        region = ctx.region or 'other'

        # Get multiplier from context, default to 1.0 if region not found
        multiplier = ctx.region_multipliers.get(region, 1.0)

        # US-113-008: Apply dynamic adjustment based on success rate
        if ctx.dynamic_region_adjustment and ctx.region_success_rates:
            success_rate = ctx.region_success_rates.get(region, 0.5)
            threshold = ctx.region_success_rate_threshold

            # Apply additional multiplier if below threshold
            if success_rate < threshold and success_rate >= 0:
                # Calculate penalty: more severe for lower success rates
                # e.g., threshold=0.7, rate=0.3 -> penalty = 1 + (0.7-0.3) = 1.4x
                penalty = 1.0 + (threshold - success_rate)
                original_multiplier = multiplier
                multiplier = multiplier * penalty
                logger.info(
                    f"Circuit breaker dynamic region adjustment: {region} success rate "
                    f"{success_rate:.1%} below threshold {threshold:.0%} - applying "
                    f"penalty ({original_multiplier}x -> {multiplier:.2f}x)"
                )
            elif success_rate >= threshold:
                # Good success rate - apply small boost to help recovery
                # Only if we're not at the minimum multiplier
                if multiplier > 0.8:
                    recovery_boost = 0.9  # Slightly reduce backoff for good regions
                    multiplier = multiplier * recovery_boost
                    logger.debug(
                        f"Circuit breaker region recovery: {region} success rate "
                        f"{success_rate:.1%} above threshold {threshold:.0%} - "
                        f"reducing multiplier ({multiplier/recovery_boost:.2f}x -> {multiplier:.2f}x)"
                    )

        if multiplier != 1.0:
            original_pause = pause
            pause = pause * multiplier
            logger.info(
                f"Circuit breaker pause region-adjusted: {original_pause:.0f}s -> "
                f"{pause:.0f}s (region={region}, multiplier={multiplier:.2f})"
            )

        return pause

    def time_adjusted(self, pause: float, ctx: PauseContext) -> float:
        """Step 5 (US-144-010): Apply time-of-day aware multiplier to pause duration.

        Applies different backoff multipliers based on time of day and day of week:
        - Time windows:
          - OVERNIGHT (10pm-6am): 0.8x (YouTube less busy)
          - MORNING (6am-12pm): 1.0x (baseline)
          - AFTERNOON (12pm-6pm): 1.2x (higher usage)
          - EVENING (6pm-10pm): 1.5x (peak usage)
        - Weekend multiplier: 1.3x additional boost for weekends

        Can use either:
        - hour + is_weekend from context
        - timestamp from context (hour and weekend derived from timestamp)

        Args:
            pause: Current pause duration in seconds
            ctx: PauseContext with time-of-day configuration

        Returns:
            Adjusted pause duration with time-of-day multiplier applied
        """
        # Skip if time-of-day backoff is disabled
        if not ctx.time_of_day_enabled:
            return pause

        # Determine hour and weekend
        hour = ctx.time_of_day_hour
        is_weekend = ctx.time_of_day_is_weekend

        # If timestamp is provided, use it to determine hour and weekend
        if ctx.time_of_day_timestamp is not None:
            dt = datetime.fromtimestamp(ctx.time_of_day_timestamp)
            hour = dt.hour
            is_weekend = dt.weekday() >= 5  # Saturday=5, Sunday=6
        elif hour is None:
            # Default to current time if not specified
            hour = datetime.now().hour

        # Determine time window from hour
        window = self._get_time_window(hour)

        # Get multiplier for time window
        multipliers = ctx.time_window_multipliers
        window_multiplier = multipliers.get(window, 1.0)

        # Apply weekend multiplier if applicable
        if is_weekend:
            window_multiplier = window_multiplier * ctx.weekend_multiplier

        # Apply multiplier
        if window_multiplier != 1.0:
            original_pause = pause
            pause = pause * window_multiplier
            logger.info(
                f"Circuit breaker pause time-adjusted: {original_pause:.0f}s -> "
                f"{pause:.0f}s (window={window}, is_weekend={is_weekend}, multiplier={window_multiplier:.2f})"
            )

        return pause

    def _get_time_window(self, hour: int) -> str:
        """Determine time window from hour of day.

        Args:
            hour: Hour of day (0-23)

        Returns:
            Time window name: 'overnight', 'morning', 'afternoon', 'evening'
        """
        # Handle overnight specially (22-23 and 0-5)
        if hour >= 22 or hour < 6:
            return 'overnight'
        elif 6 <= hour < 12:
            return 'morning'
        elif 12 <= hour < 18:
            return 'afternoon'
        else:  # 18 <= hour < 22
            return 'evening'

    def _get_adaptive_jitter_factor(self, base_factor: float) -> float:
        """Calculate adaptive jitter factor based on time-of-day patterns.

        During peak hours (9am-9pm), uses higher jitter to prevent thundering herd.
        During off-peak, uses lower jitter for faster recovery.

        Returns:
            Adjusted jitter factor (capped at jitter_max_factor).
        """
        current_hour = time.localtime().tm_hour
        # Peak hours: 9am-9pm (9-21)
        is_peak_hour = 9 <= current_hour < 21

        if is_peak_hour:
            # During peak: increase jitter by up to 50%
            adaptive_factor = base_factor * 1.5
        else:
            # Off-peak: reduce jitter by 25%
            adaptive_factor = base_factor * 0.75

        # Hard cap at max factor (50%)
        return min(adaptive_factor, 0.5)

    def _get_deterministic_jitter(self, base_factor: float, client_id: str, delay: float) -> float:
        """Calculate deterministic jitter based on client_id for reproducibility.

        Uses a hash of client_id + timestamp to create consistent but
        distributed jitter values for the same client.

        Returns:
            Jitter multiplier offset.
        """
        # Create seed from client_id and delay (for variety across delays)
        seed_input = f"{client_id}:{delay}"
        # Use first 8 chars of hash as integer seed
        seed = int(hashlib.md5(seed_input.encode()).hexdigest()[:8], 16)
        # Create a random.Random instance with this seed
        rng = random.Random(seed)
        # Generate jitter in range [-factor, +factor]
        return rng.uniform(-base_factor, base_factor)

    def apply_jitter(self, delay: float, ctx: PauseContext) -> float:
        """Step 6: Apply random jitter to a delay value.

        Jitter helps prevent thundering herd when multiple downloads
        resume simultaneously after circuit breaker recovery.

        Supports three strategies (US-109-002):
        - 'random': Classic uniform jitter (default)
        - 'adaptive': Time-of-day based (higher during peak hours)
        - 'deterministic': Seeded based on client_id for reproducibility

        The jitter formula is: delay * (1 + jitter_offset)
        Jitter is always capped at jitter_max_factor (default 50%).
        """
        jitter_factor = ctx.jitter_factor
        jitter_strategy = getattr(ctx, 'jitter_strategy', 'random')
        jitter_max = getattr(ctx, 'jitter_max_factor', 0.5)
        client_id = getattr(ctx, 'client_id', '')

        # Clamp jitter factor to valid range first
        if jitter_factor < 0.0:
            jitter_factor = 0.0
        elif jitter_factor > 1.0:
            jitter_factor = 1.0

        # Apply hard cap (AC: jitter never exceeds 50% of base delay)
        jitter_factor = min(jitter_factor, jitter_max)

        if jitter_factor <= 0.0:
            self._last_jitter_applied = 0.0
            return delay

        # Calculate jitter offset based on strategy
        if jitter_strategy == 'adaptive':
            jitter_offset = random.uniform(-self._get_adaptive_jitter_factor(jitter_factor), self._get_adaptive_jitter_factor(jitter_factor))
        elif jitter_strategy == 'deterministic' and client_id:
            jitter_offset = self._get_deterministic_jitter(jitter_factor, client_id, delay)
        else:
            # Default: random strategy
            jitter_offset = random.uniform(-jitter_factor, jitter_factor)

        # Ensure jitter doesn't exceed max (hard cap)
        jitter_offset = max(-jitter_max, min(jitter_max, jitter_offset))

        # Apply correlation check if enabled
        if getattr(ctx, 'jitter_correlation_check', False):
            # Check if this value is too similar to recent ones
            if JitterCorrelationStore.is_correlated(jitter_offset, threshold=0.05):
                # Add small adjustment to avoid correlation
                adjustment = random.uniform(-0.05, 0.05)
                jitter_offset = max(-jitter_max, min(jitter_max, jitter_offset + adjustment))
                logger.debug(f"Jitter correlation adjusted: {jitter_offset:.3f}")

            # Store for future correlation checks
            JitterCorrelationStore.add(jitter_offset)

        # Apply jitter
        jitter_multiplier = 1 + jitter_offset
        jittered_delay = delay * jitter_multiplier
        self._last_jitter_applied = jitter_offset

        return jittered_delay

    def cap_duration(self, pause: float, ctx: PauseContext) -> float:
        """Step 7: Cap a pause duration at max_pause_seconds."""
        max_pause = ctx.max_pause_seconds
        if pause > max_pause:
            logger.debug(
                f"Circuit breaker pause capped: {pause:.1f}s -> {max_pause:.0f}s "
                f"(max_pause_seconds={max_pause:.0f})"
            )
            pause = max_pause
        return pause
