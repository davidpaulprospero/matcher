"""
Result Types for Pipeline Stage Operations

Provides a structured way to return success/failure from operations
instead of using None or empty lists which can't distinguish
"no results" from "error occurred".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional, TypeVar, Generic

T = TypeVar('T')


@dataclass
class StageResult(Generic[T]):
    """
    Result of a pipeline stage execution.

    Usage:
        # Success case
        return StageResult.ok(downloaded_videos)

        # Failure case
        return StageResult.fail("Network timeout downloading videos")

        # Checking result
        if result.success:
            videos = result.data
        else:
            logger.error(result.error)
    """
    success: bool
    data: Optional[T] = None
    error: Optional[str] = None
    warnings: List[str] = field(default_factory=list)

    @classmethod
    def ok(cls, data: T, warnings: List[str] = None) -> 'StageResult[T]':
        """Create a successful result with data"""
        return cls(success=True, data=data, warnings=warnings or [])

    @classmethod
    def fail(cls, error: str, warnings: List[str] = None) -> 'StageResult[T]':
        """Create a failed result with error message"""
        return cls(success=False, error=error, warnings=warnings or [])

    @classmethod
    def partial(cls, data: T, error: str, warnings: List[str] = None) -> 'StageResult[T]':
        """Create a partial success (some data, but also errors)"""
        return cls(success=True, data=data, error=error, warnings=warnings or [])

    def __bool__(self) -> bool:
        """Allow using result in boolean context"""
        return self.success


@dataclass
class OperationResult:
    """
    Result of a single operation within a stage.

    Lighter weight than StageResult for individual operations
    like downloading a single video or transcribing a file.
    """
    success: bool
    message: str = ""
    data: Any = None

    @classmethod
    def ok(cls, message: str = "", data: Any = None) -> 'OperationResult':
        return cls(success=True, message=message, data=data)

    @classmethod
    def fail(cls, message: str, data: Any = None) -> 'OperationResult':
        return cls(success=False, message=message, data=data)


@dataclass
class ValidationResult:
    """
    Result of validating data (checkpoint, config, etc.)

    Collects multiple issues rather than failing on first error.
    """
    valid: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def add_error(self, message: str):
        """Add an error and mark as invalid"""
        self.errors.append(message)
        self.valid = False

    def add_warning(self, message: str):
        """Add a warning (doesn't affect validity)"""
        self.warnings.append(message)

    def merge(self, other: 'ValidationResult'):
        """Merge another validation result into this one"""
        if not other.valid:
            self.valid = False
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)

    def __bool__(self) -> bool:
        return self.valid
