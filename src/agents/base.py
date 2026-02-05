"""
Base Healer class for self-healing pipeline agents.

Healers detect specific error categories and attempt automatic remediation.
Each healer implements can_handle() to claim errors and fix() to remediate.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Type, runtime_checkable, Protocol

if TYPE_CHECKING:
    from ..config import Config
    from ..state import PipelineState

logger = logging.getLogger(__name__)


def get_config_value(config_obj: Any, field_name: str, default: Any = None) -> Any:
    """Safely get a config value from either a dict or dataclass/object.

    Handles the common Rule 6 pattern where config sections may be
    either dicts (from JSON loads) or dataclass objects.

    Args:
        config_obj: Config object (dict, dataclass, or None)
        field_name: Field name to retrieve
        default: Default value if field not found or config is None

    Returns:
        The field value, or default if not found
    """
    if config_obj is None:
        return default
    if isinstance(config_obj, dict):
        return config_obj.get(field_name, default)
    return getattr(config_obj, field_name, default)


def set_config_value(config_obj: Any, field_name: str, value: Any) -> bool:
    """Safely set a config value on either a dict or dataclass/object.

    Args:
        config_obj: Config object (dict, dataclass, or None)
        field_name: Field name to set
        value: Value to set

    Returns:
        True if the value was set, False if config_obj is None
    """
    if config_obj is None:
        return False
    if isinstance(config_obj, dict):
        config_obj[field_name] = value
        return True
    setattr(config_obj, field_name, value)
    return True


class HealerEvent(Enum):
    """Events for cross-healer coordination."""
    CONFIG_CHANGED = "config_changed"       # A healer modified pipeline config
    CACHE_CLEARED = "cache_cleared"         # Cache was cleaned/invalidated
    RATE_LIMITED = "rate_limited"            # Rate limiting detected
    PROVIDER_SWITCHED = "provider_switched"  # LLM/API provider was switched


@dataclass
class HealerEventData:
    """Data payload for a healer event."""
    event: HealerEvent
    source_healer: str
    details: Dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class SupportsBackoff(Protocol):
    """Protocol for healers that support backoff/retry state.

    Healers implementing this protocol have exponential backoff with
    reset capability. Used by orchestrator for event subscriptions
    and session persistence.

    Implementors must also have `backoff_time: float` and `retry_count: int`
    instance attributes (set in __init__).
    """

    def reset_backoff(self) -> None: ...


@runtime_checkable
class SupportsPreflight(Protocol):
    """Protocol for healers that provide preflight checks.

    Healers implementing this return a list of warning messages
    during orchestrator preflight.
    """

    def preflight_check(self, state: Any) -> List[str]: ...



class HealerAction(Enum):
    """Actions a healer can take."""
    RETRY = "retry"           # Retry the same stage
    SKIP = "skip"             # Skip to next stage
    MODIFY_CONFIG = "modify"  # Changed config, retry
    RESTORE = "restore"       # Restored from backup, retry
    ABORT = "abort"           # Cannot fix, stop pipeline


@dataclass
class HealerResult:
    """Result of a healing attempt."""
    success: bool
    action: HealerAction
    message: str
    modified_config: bool = False
    details: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def fixed(cls, message: str, action: HealerAction = HealerAction.RETRY, **details) -> 'HealerResult':
        """Create a successful healing result."""
        return cls(success=True, action=action, message=message, details=details)

    @classmethod
    def failed(cls, message: str, **details) -> 'HealerResult':
        """Create a failed healing result."""
        return cls(success=False, action=HealerAction.ABORT, message=message, details=details)

    @classmethod
    def config_changed(cls, message: str, **details) -> 'HealerResult':
        """Create result indicating config was modified."""
        return cls(success=True, action=HealerAction.MODIFY_CONFIG, message=message, modified_config=True, details=details)


class Healer(ABC):
    """
    Base class for pipeline healers.

    Healers are responsible for:
    1. Detecting if they can handle a specific error
    2. Attempting to fix the error
    3. Reporting what was done
    """

    name: str = "base"
    description: str = "Base healer"

    # Error patterns this healer can handle (regex or exact match)
    error_patterns: List[str] = []

    # Exception types this healer can handle
    exception_types: List[Type[Exception]] = []

    def __init__(self, config: 'Config', project_dir: Any):
        """
        Initialize healer.

        Args:
            config: Pipeline configuration
            project_dir: Project directory path
        """
        self.config = config
        self.project_dir = project_dir

    def can_handle(self, error: Exception, stage_name: str) -> bool:
        """
        Check if this healer can handle the given error.

        Args:
            error: The exception that occurred
            stage_name: Name of the stage that failed

        Returns:
            True if this healer can attempt to fix the error
        """
        # Check exception type
        for exc_type in self.exception_types:
            if isinstance(error, exc_type):
                return True

        # Check error message patterns
        error_str = str(error).lower()
        for pattern in self.error_patterns:
            if pattern.lower() in error_str:
                return True

        return False

    @abstractmethod
    def fix(
        self,
        error: Exception,
        state: 'PipelineState',
        stage_name: str
    ) -> HealerResult:
        """
        Attempt to fix the error.

        Args:
            error: The exception that occurred
            state: Current pipeline state
            stage_name: Name of the stage that failed

        Returns:
            HealerResult indicating success/failure and action to take
        """
        pass

    def handle_event(self, event_data: HealerEventData) -> None:
        """Handle a cross-healer coordination event.

        Override in subclasses to react to events from other healers.
        Default implementation is a no-op.

        Args:
            event_data: The event data including type, source, and details
        """
        pass

    def log_attempt(self, message: str):
        """Log a healing attempt."""
        logger.info(f"[{self.name}] {message}")

    def log_success(self, message: str):
        """Log successful healing."""
        logger.info(f"[{self.name}] ✓ {message}")

    def log_failure(self, message: str):
        """Log failed healing."""
        logger.warning(f"[{self.name}] ✗ {message}")
