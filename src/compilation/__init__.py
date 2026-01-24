"""
Compilation Video Pipeline

Generates compilation videos from topic/keywords without voiceover.
Creates N alternative tracks with unique clips filling the target duration.
"""

from .state import (
    CompilationCandidate,
    CompilationState,
    DownloadedClip,
    GapReport,
)
from .orchestrator import (
    CompilationOrchestrator,
    load_compilation_config,
    get_default_config,
)

__all__ = [
    "CompilationCandidate",
    "CompilationState",
    "DownloadedClip",
    "GapReport",
    "CompilationOrchestrator",
    "load_compilation_config",
    "get_default_config",
]
