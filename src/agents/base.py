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
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Type

if TYPE_CHECKING:
    from ..config import Config
    from ..state import PipelineState

logger = logging.getLogger(__name__)


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

    def log_attempt(self, message: str):
        """Log a healing attempt."""
        logger.info(f"[{self.name}] {message}")

    def log_success(self, message: str):
        """Log successful healing."""
        logger.info(f"[{self.name}] ✓ {message}")

    def log_failure(self, message: str):
        """Log failed healing."""
        logger.warning(f"[{self.name}] ✗ {message}")
