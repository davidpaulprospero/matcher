"""Health check module for pipeline stage validation.

US-88-005: Pre-stage health checks that validate external dependencies
before running each pipeline stage.
"""

import os
import re
import socket
import subprocess
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, TYPE_CHECKING
from enum import Enum

if TYPE_CHECKING:
    from config.base import Config

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health check status."""
    OK = "ok"
    WARNING = "warning"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class HealthCheckResult:
    """Result of a single health check."""
    name: str
    status: HealthStatus
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    duration_ms: float = 0.0

    def is_ok(self) -> bool:
        return self.status == HealthStatus.OK

    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'status': self.status.value,
            'message': self.message,
            'details': self.details,
            'duration_ms': self.duration_ms,
        }


@dataclass
class HealthCheckConfig:
    """Configuration for health checks."""
    # Network check settings
    network_timeout_seconds: float = 5.0
    network_check_hosts: List[str] = field(default_factory=lambda: [
        '8.8.8.8',  # Google DNS
        '1.1.1.1',  # Cloudflare DNS
    ])

    # Disk space check settings
    disk_space_warning_gb: float = 10.0  # Warn below this
    disk_space_critical_gb: float = 2.0   # Fail below this
    disk_space_check_paths: List[str] = field(default_factory=list)  # Empty = use project dir

    # Embedding provider settings
    embedding_timeout_seconds: float = 10.0
    embedding_check_enabled: bool = True

    # FFmpeg check settings
    ffmpeg_required: bool = True  # Fail if FFmpeg not found
    ffmpeg_min_version: str = "4.0"  # Minimum required version

    # Required commands check
    required_commands: List[str] = field(default_factory=lambda: ['ffmpeg'])

    # Memory check settings
    memory_warning_percent: float = 85.0  # Warn above this
    memory_critical_percent: float = 95.0   # Fail above this

    # Health check timeouts per stage (stage_name -> timeout seconds)
    stage_timeouts: Dict[str, float] = field(default_factory=dict)

    # yt-dlp check settings
    ytdlp_required: bool = False  # Fail if yt-dlp not found (default: warning only)
    ytdlp_min_version: str = "2024"  # Minimum required version year

    # LLM provider check settings
    llm_provider_check_enabled: bool = True
    llm_provider_timeout_seconds: float = 10.0

    # Enable/disable specific health checks
    enabled_checks: Dict[str, bool] = field(default_factory=lambda: {
        'network': True,
        'disk_space': True,
        'memory': True,
        'embedding': True,
        'ffmpeg': True,
        'ytdlp': True,
        'llm_provider': True,
    })

    def get_stage_timeout(self, stage_name: str, default: float = 30.0) -> float:
        """Get timeout for a specific stage, or default if not configured."""
        return self.stage_timeouts.get(stage_name, default)


class HealthChecker:
    """Performs health checks for pipeline stages."""

    def __init__(self, config: 'Config'):
        self.config = config
        self.health_config = self._load_health_config(config)

    def _load_health_config(self, config: 'Config') -> HealthCheckConfig:
        """Load health check config from Config object."""
        # Check if config has health_check section
        hc_dict = {}
        if hasattr(config, 'health_check'):
            hc = config.health_check
            if isinstance(hc, dict):
                hc_dict = hc
            elif hasattr(hc, '__dict__'):
                hc_dict = hc.__dict__

        # Support both config.yaml naming (min_disk_space_gb, warn_disk_space_gb, etc.)
        # and the internal naming (disk_space_critical_gb, disk_space_warning_gb, etc.)
        disk_space_warning = (
            hc_dict.get('warn_disk_space_gb') or
            hc_dict.get('disk_space_warning_gb') or
            10.0
        )
        disk_space_critical = (
            hc_dict.get('min_disk_space_gb') or
            hc_dict.get('disk_space_critical_gb') or
            2.0
        )
        memory_warning = (
            hc_dict.get('min_memory_percent') or
            hc_dict.get('memory_warning_percent') or
            85.0
        )
        memory_critical = (
            hc_dict.get('max_memory_percent') or
            hc_dict.get('memory_critical_percent') or
            95.0
        )

        return HealthCheckConfig(
            network_timeout_seconds=hc_dict.get('network_timeout_seconds', 5.0),
            network_check_hosts=hc_dict.get('network_check_hosts', ['8.8.8.8', '1.1.1.1']),
            disk_space_warning_gb=disk_space_warning,
            disk_space_critical_gb=disk_space_critical,
            disk_space_check_paths=hc_dict.get('disk_space_check_paths', []),
            embedding_timeout_seconds=hc_dict.get('embedding_timeout_seconds', 10.0),
            embedding_check_enabled=hc_dict.get('embedding_check_enabled', True),
            ffmpeg_required=hc_dict.get('ffmpeg_required', True),
            ffmpeg_min_version=hc_dict.get('ffmpeg_min_version', '4.0'),
            required_commands=hc_dict.get('required_commands', ['ffmpeg']),
            memory_warning_percent=memory_warning,
            memory_critical_percent=memory_critical,
            stage_timeouts=hc_dict.get('stage_timeouts', {}),
            ytdlp_required=hc_dict.get('ytdlp_required', False),
            ytdlp_min_version=hc_dict.get('ytdlp_min_version', '2024'),
            llm_provider_check_enabled=hc_dict.get('llm_provider_check_enabled', True),
            llm_provider_timeout_seconds=hc_dict.get('llm_provider_timeout_seconds', 10.0),
            enabled_checks=hc_dict.get('enabled_checks', {
                'network': True,
                'disk_space': True,
                'memory': True,
                'embedding': True,
                'ffmpeg': True,
                'ytdlp': True,
                'llm_provider': True,
            }),
        )

    def check_network(self, timeout: Optional[float] = None) -> HealthCheckResult:
        """Check network connectivity.

        Tests connectivity by attempting to connect to specified hosts.
        """
        import time
        start = time.perf_counter()

        if not self.health_config.enabled_checks.get('network', True):
            return HealthCheckResult(
                name="network",
                status=HealthStatus.SKIPPED,
                message="Network check disabled",
                duration_ms=0.0,
            )

        timeout_val = timeout or self.health_config.network_timeout_seconds
        hosts = self.health_config.network_check_hosts

        # Try to connect to each host
        port = 53  # DNS port
        for host in hosts:
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(timeout_val)
                sock.connect((host, port))
                sock.close()
                duration_ms = (time.perf_counter() - start) * 1000
                return HealthCheckResult(
                    name="network",
                    status=HealthStatus.OK,
                    message=f"Network connectivity OK (connected to {host})",
                    details={'host': host, 'port': port},
                    duration_ms=duration_ms,
                )
            except (socket.timeout, socket.error) as e:
                continue

        duration_ms = (time.perf_counter() - start) * 1000
        return HealthCheckResult(
            name="network",
            status=HealthStatus.FAILED,
            message=f"Network connectivity failed (tried: {hosts})",
            details={'hosts': hosts, 'error': 'All hosts unreachable'},
            duration_ms=duration_ms,
        )

    def check_disk_space(self, path: Optional[str] = None) -> HealthCheckResult:
        """Check available disk space.

        Returns warning if below warning threshold, failed if below critical.
        """
        import time
        start = time.perf_counter()

        if not self.health_config.enabled_checks.get('disk_space', True):
            return HealthCheckResult(
                name="disk_space",
                status=HealthStatus.SKIPPED,
                message="Disk space check disabled",
                duration_ms=0.0,
            )

        # Use configured paths or provided path or default to project dir
        check_paths = [path] if path else self.health_config.disk_space_check_paths
        if not check_paths:
            # Use project directory if available
            if hasattr(self.config, 'project') and self.config.project:
                check_paths = [self.config.project]
            else:
                check_paths = [os.getcwd()]

        for check_path in check_paths:
            try:
                if not os.path.exists(check_path):
                    continue
                stat = os.statvfs(check_path) if hasattr(os, 'statvfs') else None
                if stat:
                    free_bytes = stat.f_bavail * stat.f_frsize
                    free_gb = free_bytes / (1024 ** 3)

                    duration_ms = (time.perf_counter() - start) * 1000

                    if free_gb < self.health_config.disk_space_critical_gb:
                        return HealthCheckResult(
                            name="disk_space",
                            status=HealthStatus.FAILED,
                            message=f"Critical: Only {free_gb:.1f}GB free at {check_path}",
                            details={'path': check_path, 'free_gb': free_gb},
                            duration_ms=duration_ms,
                        )
                    elif free_gb < self.health_config.disk_space_warning_gb:
                        return HealthCheckResult(
                            name="disk_space",
                            status=HealthStatus.WARNING,
                            message=f"Low disk space: {free_gb:.1f}GB free at {check_path}",
                            details={'path': check_path, 'free_gb': free_gb},
                            duration_ms=duration_ms,
                        )
                    else:
                        return HealthCheckResult(
                            name="disk_space",
                            status=HealthStatus.OK,
                            message=f"Disk space OK: {free_gb:.1f}GB free at {check_path}",
                            details={'path': check_path, 'free_gb': free_gb},
                            duration_ms=duration_ms,
                        )
            except OSError as e:
                continue

        duration_ms = (time.perf_counter() - start) * 1000
        return HealthCheckResult(
            name="disk_space",
            status=HealthStatus.WARNING,
            message="Could not check disk space (paths not accessible)",
            details={'check_paths': check_paths},
            duration_ms=duration_ms,
        )

    def check_memory(self) -> HealthCheckResult:
        """Check available memory/RAM usage.

        Returns warning if above warning threshold, failed if above critical.
        This is particularly important for stages that load large embeddings.
        """
        import time
        start = time.perf_counter()

        if not self.health_config.enabled_checks.get('memory', True):
            return HealthCheckResult(
                name="memory",
                status=HealthStatus.SKIPPED,
                message="Memory check disabled",
                duration_ms=0.0,
            )

        try:
            # Try to get memory info - cross-platform
            if hasattr(os, 'sysconf'):
                # Unix-like systems
                total_mem = os.sysconf('SC_PHYS_PAGES') * os.sysconf('SC_PAGESIZE')
                # Use psutil if available for more accurate reading
                try:
                    import psutil
                    mem = psutil.virtual_memory()
                    used_percent = mem.percent
                    available_gb = mem.available / (1024 ** 3)
                    total_gb = mem.total / (1024 ** 3)
                except ImportError:
                    # Fallback: just check total memory
                    used_percent = 50.0  # Assume 50% if we can't measure
                    total_gb = total_mem / (1024 ** 3)
                    available_gb = total_gb * 0.5
            else:
                # Windows - try psutil
                try:
                    import psutil
                    mem = psutil.virtual_memory()
                    used_percent = mem.percent
                    available_gb = mem.available / (1024 ** 3)
                    total_gb = mem.total / (1024 ** 3)
                except ImportError:
                    duration_ms = (time.perf_counter() - start) * 1000
                    return HealthCheckResult(
                        name="memory",
                        status=HealthStatus.SKIPPED,
                        message="Memory check not available on this platform (psutil recommended)",
                        details={},
                        duration_ms=duration_ms,
                    )

            duration_ms = (time.perf_counter() - start) * 1000

            if used_percent >= self.health_config.memory_critical_percent:
                return HealthCheckResult(
                    name="memory",
                    status=HealthStatus.FAILED,
                    message=f"Critical: Memory usage at {used_percent:.1f}%",
                    details={'used_percent': used_percent, 'available_gb': available_gb, 'total_gb': total_gb},
                    duration_ms=duration_ms,
                )
            elif used_percent >= self.health_config.memory_warning_percent:
                return HealthCheckResult(
                    name="memory",
                    status=HealthStatus.WARNING,
                    message=f"High memory usage: {used_percent:.1f}%",
                    details={'used_percent': used_percent, 'available_gb': available_gb, 'total_gb': total_gb},
                    duration_ms=duration_ms,
                )
            else:
                return HealthCheckResult(
                    name="memory",
                    status=HealthStatus.OK,
                    message=f"Memory OK: {used_percent:.1f}% used",
                    details={'used_percent': used_percent, 'available_gb': available_gb, 'total_gb': total_gb},
                    duration_ms=duration_ms,
                )
        except Exception as e:
            duration_ms = (time.perf_counter() - start) * 1000
            return HealthCheckResult(
                name="memory",
                status=HealthStatus.WARNING,
                message=f"Could not check memory: {str(e)}",
                details={'error': str(e)},
                duration_ms=duration_ms,
            )

    def check_embedding_provider(self) -> HealthCheckResult:
        """Check embedding provider availability.

        Tests if the configured embedding provider is accessible.
        """
        import time
        start = time.perf_counter()

        if not self.health_config.enabled_checks.get('embedding', True):
            return HealthCheckResult(
                name="embedding_provider",
                status=HealthStatus.SKIPPED,
                message="Embedding check disabled",
                duration_ms=0.0,
            )

        if not self.health_config.embedding_check_enabled:
            return HealthCheckResult(
                name="embedding_provider",
                status=HealthStatus.SKIPPED,
                message="Embedding check disabled in config",
                duration_ms=0.0,
            )

        # Check embedding provider configuration
        embedding_config = getattr(self.config, 'embedding', None)
        if not embedding_config:
            duration_ms = (time.perf_counter() - start) * 1000
            return HealthCheckResult(
                name="embedding_provider",
                status=HealthStatus.FAILED,
                message="No embedding configuration found",
                details={},
                duration_ms=duration_ms,
            )

        provider = None
        if isinstance(embedding_config, dict):
            provider = embedding_config.get('provider', 'gemini')
        elif hasattr(embedding_config, 'provider'):
            provider = embedding_config.provider

        # Check based on provider type
        if provider == 'local':
            # Check if local embedding service is available
            return self._check_local_embedding()
        elif provider in ('gemini', 'anthropic'):
            # For cloud providers, check network and API key
            return self._check_cloud_embedding(provider)
        else:
            duration_ms = (time.perf_counter() - start) * 1000
            return HealthCheckResult(
                name="embedding_provider",
                status=HealthStatus.WARNING,
                message=f"Unknown embedding provider: {provider}",
                details={'provider': provider},
                duration_ms=duration_ms,
            )

    def _check_local_embedding(self) -> HealthCheckResult:
        """Check local embedding provider (Ollama)."""
        import time
        start = time.perf_counter()

        # Check if Ollama is running
        try:
            result = subprocess.run(
                ['curl', '-s', 'http://localhost:11434/api/tags'],
                capture_output=True,
                timeout=self.health_config.embedding_timeout_seconds,
                encoding='utf-8',
                errors='replace',
            )
            duration_ms = (time.perf_counter() - start) * 1000

            if result.returncode == 0:
                return HealthCheckResult(
                    name="embedding_provider",
                    status=HealthStatus.OK,
                    message="Local embedding provider (Ollama) is running",
                    details={'provider': 'local', 'url': 'http://localhost:11434'},
                    duration_ms=duration_ms,
                )
            else:
                return HealthCheckResult(
                    name="embedding_provider",
                    status=HealthStatus.FAILED,
                    message="Local embedding provider (Ollama) not responding",
                    details={'provider': 'local', 'returncode': result.returncode},
                    duration_ms=duration_ms,
                )
        except FileNotFoundError:
            duration_ms = (time.perf_counter() - start) * 1000
            return HealthCheckResult(
                name="embedding_provider",
                status=HealthStatus.FAILED,
                message="curl not found - cannot check local embedding",
                details={'provider': 'local', 'error': 'curl not available'},
                duration_ms=duration_ms,
            )
        except subprocess.TimeoutExpired:
            duration_ms = (time.perf_counter() - start) * 1000
            return HealthCheckResult(
                name="embedding_provider",
                status=HealthStatus.FAILED,
                message="Local embedding provider (Ollama) timeout",
                details={'provider': 'local', 'timeout': self.health_config.embedding_timeout_seconds},
                duration_ms=duration_ms,
            )

    def _check_cloud_embedding(self, provider: str) -> HealthCheckResult:
        """Check cloud embedding provider (Gemini/Anthropic)."""
        import time
        start = time.perf_counter()

        # Check if API key is configured
        api_key = None
        if provider == 'gemini':
            api_key = os.environ.get('GEMINI_API_KEY')
        elif provider == 'anthropic':
            api_key = os.environ.get('ANTHROPIC_API_KEY')

        if not api_key:
            duration_ms = (time.perf_counter() - start) * 1000
            return HealthCheckResult(
                name="embedding_provider",
                status=HealthStatus.FAILED,
                message=f"No API key configured for {provider}",
                details={'provider': provider, 'error': 'API key not set'},
                duration_ms=duration_ms,
            )

        # For cloud providers, just verify network connectivity
        network_result = self.check_disk_space()  # Reuse network check
        if network_result.status == HealthStatus.OK:
            duration_ms = (time.perf_counter() - start) * 1000
            return HealthCheckResult(
                name="embedding_provider",
                status=HealthStatus.OK,
                message=f"Cloud embedding provider ({provider}) configured and network OK",
                details={'provider': provider, 'network': 'ok'},
                duration_ms=duration_ms,
            )
        else:
            duration_ms = (time.perf_counter() - start) * 1000
            return HealthCheckResult(
                name="embedding_provider",
                status=HealthStatus.FAILED,
                message=f"Cloud embedding provider ({provider}) - network unavailable",
                details={'provider': provider, 'network': 'failed'},
                duration_ms=duration_ms,
            )

    def check_ffmpeg(self) -> HealthCheckResult:
        """Check FFmpeg availability.

        Tests if FFmpeg is installed and meets minimum version requirements.
        """
        import time
        start = time.perf_counter()

        # Check if FFmpeg check is enabled
        if not self.health_config.ffmpeg_required:
            return HealthCheckResult(
                name="ffmpeg",
                status=HealthStatus.SKIPPED,
                message="FFmpeg check disabled in config",
                duration_ms=0.0,
            )

        # Try to get FFmpeg version
        try:
            result = subprocess.run(
                ['ffmpeg', '-version'],
                capture_output=True,
                timeout=10,
                encoding='utf-8',
                errors='replace',
            )
            duration_ms = (time.perf_counter() - start) * 1000

            if result.returncode == 0:
                # Parse version from output
                version_output = result.stdout.split('\n')[0] if result.stdout else ''
                # Extract version number (e.g., "ffmpeg version 4.4")
                version_match = re.search(r'ffmpeg\s+version\s+(\d+\.\d+)', version_output, re.IGNORECASE)
                if version_match:
                    version = version_match.group(1)
                    # Check minimum version
                    min_version = self.health_config.ffmpeg_min_version
                    return HealthCheckResult(
                        name="ffmpeg",
                        status=HealthStatus.OK,
                        message=f"FFmpeg {version} available (minimum: {min_version})",
                        details={'version': version, 'min_version': min_version},
                        duration_ms=duration_ms,
                    )
                else:
                    return HealthCheckResult(
                        name="ffmpeg",
                        status=HealthStatus.OK,
                        message="FFmpeg available",
                        details={'output': version_output[:100]},
                        duration_ms=duration_ms,
                    )
            else:
                duration_ms = (time.perf_counter() - start) * 1000
                return HealthCheckResult(
                    name="ffmpeg",
                    status=HealthStatus.FAILED,
                    message="FFmpeg not available or returned error",
                    details={'returncode': result.returncode, 'stderr': result.stderr[:200]},
                    duration_ms=duration_ms,
                )
        except FileNotFoundError:
            duration_ms = (time.perf_counter() - start) * 1000
            return HealthCheckResult(
                name="ffmpeg",
                status=HealthStatus.FAILED,
                message="FFmpeg not found in PATH",
                details={'error': 'ffmpeg command not found'},
                duration_ms=duration_ms,
            )
        except subprocess.TimeoutExpired:
            duration_ms = (time.perf_counter() - start) * 1000
            return HealthCheckResult(
                name="ffmpeg",
                status=HealthStatus.FAILED,
                message="FFmpeg check timed out",
                details={'timeout': 10},
                duration_ms=duration_ms,
            )

    def check_ytdlp(self) -> HealthCheckResult:
        """Check yt-dlp availability.

        Tests if yt-dlp is installed and meets minimum version requirements.
        """
        import time
        start = time.perf_counter()

        if not self.health_config.enabled_checks.get('ytdlp', True):
            return HealthCheckResult(
                name="ytdlp",
                status=HealthStatus.SKIPPED,
                message="yt-dlp check disabled",
                duration_ms=0.0,
            )

        # Try to get yt-dlp version
        try:
            result = subprocess.run(
                ['yt-dlp', '--version'],
                capture_output=True,
                timeout=10,
                encoding='utf-8',
                errors='replace',
            )
            duration_ms = (time.perf_counter() - start) * 1000

            if result.returncode == 0:
                # Parse version from output
                version_output = result.stdout.strip() if result.stdout else ''
                # Extract version number (e.g., "2024.12.23")
                version_match = re.search(r'(\d{4})\.(\d+)\.(\d+)', version_output)
                if version_match:
                    version_year = version_match.group(1)
                    # Check minimum version year
                    min_version_year = self.health_config.ytdlp_min_version
                    if version_year >= min_version_year:
                        return HealthCheckResult(
                            name="ytdlp",
                            status=HealthStatus.OK,
                            message=f"yt-dlp {version_output} available (minimum year: {min_version_year})",
                            details={'version': version_output, 'min_version': min_version_year},
                            duration_ms=duration_ms,
                        )
                    else:
                        status = HealthStatus.FAILED if self.health_config.ytdlp_required else HealthStatus.WARNING
                        return HealthCheckResult(
                            name="ytdlp",
                            status=status,
                            message=f"yt-dlp {version_output} too old (minimum year: {min_version_year})",
                            details={'version': version_output, 'min_version': min_version_year},
                            duration_ms=duration_ms,
                        )
                else:
                    return HealthCheckResult(
                        name="ytdlp",
                        status=HealthStatus.OK,
                        message=f"yt-dlp available: {version_output}",
                        details={'version': version_output},
                        duration_ms=duration_ms,
                    )
            else:
                duration_ms = (time.perf_counter() - start) * 1000
                return HealthCheckResult(
                    name="ytdlp",
                    status=HealthStatus.FAILED,
                    message="yt-dlp not available or returned error",
                    details={'returncode': result.returncode, 'stderr': result.stderr[:200]},
                    duration_ms=duration_ms,
                )
        except FileNotFoundError:
            duration_ms = (time.perf_counter() - start) * 1000
            status = HealthStatus.FAILED if self.health_config.ytdlp_required else HealthStatus.WARNING
            return HealthCheckResult(
                name="ytdlp",
                status=status,
                message="yt-dlp not found in PATH",
                details={'error': 'yt-dlp command not found'},
                duration_ms=duration_ms,
            )
        except subprocess.TimeoutExpired:
            duration_ms = (time.perf_counter() - start) * 1000
            return HealthCheckResult(
                name="ytdlp",
                status=HealthStatus.FAILED,
                message="yt-dlp check timed out",
                details={'timeout': 10},
                duration_ms=duration_ms,
            )

    def check_llm_provider(self) -> HealthCheckResult:
        """Check LLM provider connectivity.

        Tests if the configured LLM provider is accessible (Gemini, Anthropic, or Ollama).
        """
        import time
        import json
        start = time.perf_counter()

        if not self.health_config.enabled_checks.get('llm_provider', True):
            return HealthCheckResult(
                name="llm_provider",
                status=HealthStatus.SKIPPED,
                message="LLM provider check disabled",
                duration_ms=0.0,
            )

        if not self.health_config.llm_provider_check_enabled:
            return HealthCheckResult(
                name="llm_provider",
                status=HealthStatus.SKIPPED,
                message="LLM provider check disabled in config",
                duration_ms=0.0,
            )

        # Get LLM config to determine provider
        llm_config = getattr(self.config, 'llm', None)
        if not llm_config:
            duration_ms = (time.perf_counter() - start) * 1000
            return HealthCheckResult(
                name="llm_provider",
                status=HealthStatus.FAILED,
                message="No LLM configuration found",
                details={},
                duration_ms=duration_ms,
            )

        # Determine provider
        provider = None
        if isinstance(llm_config, dict):
            provider = llm_config.get('provider', 'gemini')
        elif hasattr(llm_config, 'provider'):
            provider = llm_config.provider

        if not provider:
            provider = 'gemini'  # Default

        # Check based on provider type
        if provider == 'ollama':
            return self._check_ollama_llm(start)
        elif provider in ('gemini', 'anthropic'):
            return self._check_cloud_llm(provider, start)
        else:
            duration_ms = (time.perf_counter() - start) * 1000
            return HealthCheckResult(
                name="llm_provider",
                status=HealthStatus.WARNING,
                message=f"Unknown LLM provider: {provider}",
                details={'provider': provider},
                duration_ms=duration_ms,
            )

    def _check_ollama_llm(self, start: float) -> HealthCheckResult:
        """Check Ollama LLM provider."""
        import time
        import json
        try:
            result = subprocess.run(
                ['curl', '-s', 'http://localhost:11434/api/tags'],
                capture_output=True,
                timeout=self.health_config.llm_provider_timeout_seconds,
                encoding='utf-8',
                errors='replace',
            )
            duration_ms = (time.perf_counter() - start) * 1000

            if result.returncode == 0:
                # Try to parse response to see if it's valid JSON
                try:
                    models = json.loads(result.stdout)
                    model_count = len(models.get('models', [])) if isinstance(models, dict) else 0
                    return HealthCheckResult(
                        name="llm_provider",
                        status=HealthStatus.OK,
                        message=f"Ollama LLM provider is running ({model_count} models available)",
                        details={'provider': 'ollama', 'url': 'http://localhost:11434', 'models': model_count},
                        duration_ms=duration_ms,
                    )
                except json.JSONDecodeError:
                    return HealthCheckResult(
                        name="llm_provider",
                        status=HealthStatus.OK,
                        message="Ollama LLM provider is running",
                        details={'provider': 'ollama', 'url': 'http://localhost:11434'},
                        duration_ms=duration_ms,
                    )
            else:
                duration_ms = (time.perf_counter() - start) * 1000
                return HealthCheckResult(
                    name="llm_provider",
                    status=HealthStatus.FAILED,
                    message="Ollama LLM provider not responding",
                    details={'provider': 'ollama', 'returncode': result.returncode},
                    duration_ms=duration_ms,
                )
        except FileNotFoundError:
            duration_ms = (time.perf_counter() - start) * 1000
            return HealthCheckResult(
                name="llm_provider",
                status=HealthStatus.FAILED,
                message="curl not found - cannot check Ollama LLM",
                details={'provider': 'ollama', 'error': 'curl not available'},
                duration_ms=duration_ms,
            )
        except subprocess.TimeoutExpired:
            duration_ms = (time.perf_counter() - start) * 1000
            return HealthCheckResult(
                name="llm_provider",
                status=HealthStatus.FAILED,
                message="Ollama LLM provider timeout",
                details={'provider': 'ollama', 'timeout': self.health_config.llm_provider_timeout_seconds},
                duration_ms=duration_ms,
            )

    def _check_cloud_llm(self, provider: str, start: float) -> HealthCheckResult:
        """Check cloud LLM provider (Gemini/Anthropic)."""
        import time
        # Check if API key is configured
        api_key = None
        if provider == 'gemini':
            api_key = os.environ.get('GEMINI_API_KEY') or os.environ.get('GOOGLE_API_KEY')
        elif provider == 'anthropic':
            api_key = os.environ.get('ANTHROPIC_API_KEY')

        if not api_key:
            duration_ms = (time.perf_counter() - start) * 1000
            return HealthCheckResult(
                name="llm_provider",
                status=HealthStatus.FAILED,
                message=f"No API key configured for {provider}",
                details={'provider': provider, 'error': 'API key not set'},
                duration_ms=duration_ms,
            )

        # For cloud providers, verify network connectivity
        # Check if we can reach the provider's API endpoint
        try:
            timeout = self.health_config.llm_provider_timeout_seconds

            if provider == 'gemini':
                # Check Gemini API endpoint
                result = subprocess.run(
                    ['curl', '-s', '-o', '/dev/null', '-w', '%{http_code}',
                     f'https://generativelanguage.googleapis.com/v1/models?key={api_key[:20]}...'],
                    capture_output=True,
                    timeout=timeout,
                    encoding='utf-8',
                    errors='replace',
                )
            elif provider == 'anthropic':
                # Check Anthropic API endpoint (just connectivity, not auth)
                result = subprocess.run(
                    ['curl', '-s', '-o', '/dev/null', '-w', '%{http_code}',
                     'https://api.anthropic.com'],
                    capture_output=True,
                    timeout=timeout,
                    encoding='utf-8',
                    errors='replace',
                )

            duration_ms = (time.perf_counter() - start) * 1000

            if result.returncode == 0:
                http_code = result.stdout.strip()
                # 200 = OK, 401 = auth issue (but connectivity OK), 403 = forbidden
                if http_code in ('200', '401', '403'):
                    return HealthCheckResult(
                        name="llm_provider",
                        status=HealthStatus.OK,
                        message=f"Cloud LLM provider ({provider}) is reachable (HTTP {http_code})",
                        details={'provider': provider, 'http_code': http_code},
                        duration_ms=duration_ms,
                    )
                else:
                    return HealthCheckResult(
                        name="llm_provider",
                        status=HealthStatus.FAILED,
                        message=f"Cloud LLM provider ({provider}) returned HTTP {http_code}",
                        details={'provider': provider, 'http_code': http_code},
                        duration_ms=duration_ms,
                    )
            else:
                return HealthCheckResult(
                    name="llm_provider",
                    status=HealthStatus.FAILED,
                    message=f"Cannot reach {provider} API endpoint",
                    details={'provider': provider, 'returncode': result.returncode},
                    duration_ms=duration_ms,
                )
        except FileNotFoundError:
            duration_ms = (time.perf_counter() - start) * 1000
            return HealthCheckResult(
                name="llm_provider",
                status=HealthStatus.WARNING,
                message="curl not found - cannot verify cloud LLM connectivity",
                details={'provider': provider, 'error': 'curl not available'},
                duration_ms=duration_ms,
            )
        except subprocess.TimeoutExpired:
            duration_ms = (time.perf_counter() - start) * 1000
            return HealthCheckResult(
                name="llm_provider",
                status=HealthStatus.FAILED,
                message=f"{provider} LLM provider timeout",
                details={'provider': provider, 'timeout': timeout},
                duration_ms=duration_ms,
            )

    def check_stage(self, stage_name: str, project_path: Optional[str] = None) -> List[HealthCheckResult]:
        """Run appropriate health checks for a specific stage.

        Args:
            stage_name: Name of the stage to check
            project_path: Optional project path for disk space check

        Returns:
            List of health check results
        """
        results = []

        # Network checks - for any stage that uses network
        if stage_name in ('VIDEO_SEARCH', 'CAPTION', 'DOWNLOAD_SEGMENTS', 'MATCH', 'ITERATIVE_MATCH'):
            results.append(self.check_network())

        # Disk space checks - for download and output stages
        if stage_name in ('DOWNLOAD_SEGMENTS', 'OUTPUT'):
            results.append(self.check_disk_space(project_path))

        # Memory checks - for stages that load large embeddings
        if stage_name in ('MATCH', 'ITERATIVE_MATCH'):
            results.append(self.check_memory())

        # Embedding checks - for matching stages
        if stage_name in ('MATCH', 'ITERATIVE_MATCH'):
            results.append(self.check_embedding_provider())

        # FFmpeg checks - for download and output stages
        if stage_name in ('DOWNLOAD_SEGMENTS', 'OUTPUT'):
            results.append(self.check_ffmpeg())

        # yt-dlp checks - for download stages
        if stage_name in ('DOWNLOAD_SEGMENTS', 'VIDEO_SEARCH', 'CAPTION'):
            results.append(self.check_ytdlp())

        # LLM provider checks - for stages that use LLM
        if stage_name in ('MATCH', 'ITERATIVE_MATCH', 'VIDEO_SEARCH'):
            results.append(self.check_llm_provider())

        return results

    def check_before_stage(self, stage_name: str, project_path: Optional[str] = None) -> List[HealthCheckResult]:
        """Run pre-stage health checks before executing a stage.

        This is an alias for check_stage() that logs warnings but continues execution.
        Health checks are advisory - they warn but don't block stage execution.

        Args:
            stage_name: Name of the stage to check before execution
            project_path: Optional project path for disk space check

        Returns:
            List of health check results
        """
        results = self.check_stage(stage_name, project_path)

        # Log warnings but continue execution (health checks are advisory)
        for result in results:
            if result.status == HealthStatus.WARNING:
                logger.warning(f"Health check warning for {stage_name}: {result.message}")
            elif result.status == HealthStatus.FAILED:
                logger.warning(f"Health check failed for {stage_name}: {result.message} (continuing anyway)")

        return results

    def get_stage_timeout(self, stage_name: str) -> float:
        """Get the configured timeout for a specific stage."""
        return self.health_config.get_stage_timeout(stage_name, 30.0)

    def check_circuit_breakers(self) -> HealthCheckResult:
        """Check circuit breaker states (US-120-006).

        Reports on the health of search and caption circuit breakers,
        showing current state and any open circuits.
        """
        import time
        start = time.perf_counter()

        if not self.health_config.enabled_checks.get('circuit_breaker', True):
            return HealthCheckResult(
                name="circuit_breaker",
                status=HealthStatus.SKIPPED,
                message="Circuit breaker check disabled",
                duration_ms=0.0,
            )

        try:
            # Import circuit breakers
            from src.downloader.circuit_breaker import CircuitBreaker, CircuitBreakerConfig
            from src.caption.circuit_breaker import CaptionCircuitBreaker

            # Create search circuit breaker
            search_config = CircuitBreakerConfig()
            search_cb = CircuitBreaker(config=search_config)

            # Create caption circuit breaker
            caption_cb = CaptionCircuitBreaker()

            # Get states
            search_state = search_cb.get_state()
            caption_state = caption_cb.get_state()

            # Build status message
            issues = []

            if search_state == 'open':
                remaining = search_cb.get_remaining_pause_time()
                issues.append(f"search circuit OPEN (paused {remaining:.0f}s)")
            elif search_state == 'half_open':
                issues.append(f"search circuit HALF-OPEN ({search_cb.state.consecutive_failures} failures)")

            if caption_state == 'open':
                remaining = caption_cb.get_remaining_pause_time()
                issues.append(f"caption circuit OPEN (paused {remaining:.0f}s)")
            elif caption_state == 'half_open':
                issues.append(f"caption circuit HALF-OPEN ({caption_cb.state.consecutive_failures} failures)")

            duration_ms = (time.perf_counter() - start) * 1000

            if issues:
                # Circuits are open or half-open - report warning status
                message = "; ".join(issues)
                return HealthCheckResult(
                    name="circuit_breaker",
                    status=HealthStatus.WARNING,
                    message=f"Circuit breaker state: {message}",
                    details={
                        'search': {
                            'state': search_state,
                            'consecutive_failures': search_cb.state.consecutive_failures,
                            'total_trips': search_cb.state.total_trips,
                        },
                        'caption': {
                            'state': caption_state,
                            'consecutive_failures': caption_cb.state.consecutive_failures,
                            'total_trips': caption_cb.state.total_trips,
                        },
                    },
                    duration_ms=duration_ms,
                )
            else:
                # All circuits closed - healthy
                return HealthCheckResult(
                    name="circuit_breaker",
                    status=HealthStatus.OK,
                    message="Circuit breakers: search CLOSED, caption CLOSED",
                    details={
                        'search': {'state': 'closed', 'consecutive_failures': 0, 'total_trips': search_cb.state.total_trips},
                        'caption': {'state': 'closed', 'consecutive_failures': 0, 'total_trips': caption_cb.state.total_trips},
                    },
                    duration_ms=duration_ms,
                )

        except Exception as e:
            duration_ms = (time.perf_counter() - start) * 1000
            return HealthCheckResult(
                name="circuit_breaker",
                status=HealthStatus.WARNING,
                message=f"Circuit breaker check error: {str(e)}",
                details={'error': str(e)},
                duration_ms=duration_ms,
            )

    def run_all_checks(self, project_path: Optional[str] = None) -> List[HealthCheckResult]:
        """Run all health checks regardless of stage.

        This is used by --health-check CLI flag to run comprehensive diagnostics.

        Args:
            project_path: Optional project path for disk space check

        Returns:
            List of all health check results
        """
        results = []

        # Run all available checks
        results.append(self.check_network())
        results.append(self.check_disk_space(project_path))
        results.append(self.check_memory())
        results.append(self.check_embedding_provider())
        results.append(self.check_ffmpeg())
        results.append(self.check_ytdlp())
        results.append(self.check_llm_provider())
        results.append(self.check_circuit_breakers())

        return results


def run_health_checks(
    stage_name: str,
    config: 'Config',
    project_path: Optional[str] = None,
) -> List[HealthCheckResult]:
    """Convenience function to run health checks for a stage.

    Args:
        stage_name: Name of the stage
        config: Pipeline configuration
        project_path: Optional project path

    Returns:
        List of health check results
    """
    checker = HealthChecker(config)
    return checker.check_stage(stage_name, project_path)


def run_all_health_checks(
    config: 'Config',
    project_path: Optional[str] = None,
) -> List[HealthCheckResult]:
    """Run all health checks (used by --health-check CLI flag).

    Args:
        config: Pipeline configuration
        project_path: Optional project path

    Returns:
        List of all health check results
    """
    checker = HealthChecker(config)
    return checker.run_all_checks(project_path)


# Registry for stage-specific health checks
STAGE_HEALTH_CHECKS: Dict[str, List[str]] = {
    'VIDEO_SEARCH': ['network', 'ytdlp', 'llm_provider'],
    'CAPTION': ['network', 'ytdlp'],
    'MATCH': ['network', 'memory', 'embedding', 'llm_provider'],
    'ITERATIVE_MATCH': ['network', 'memory', 'embedding', 'llm_provider'],
    'DOWNLOAD_SEGMENTS': ['network', 'disk_space', 'ffmpeg', 'ytdlp'],
    'OUTPUT': ['disk_space', 'ffmpeg'],
}


def get_circuit_breaker_health(
    circuit_breaker: Optional[Any] = None,
    per_keyword_breaker: Optional[Any] = None,
) -> Optional[Dict[str, Any]]:
    """Get circuit breaker health metrics.

    Args:
        circuit_breaker: Optional CircuitBreaker instance to get metrics from
        per_keyword_breaker: Optional PerKeywordCircuitBreaker instance

    Returns:
        Dict with health metrics or None if no circuit breaker provided
    """
    result: Dict[str, Any] = {}

    # Get main circuit breaker metrics
    if circuit_breaker is not None:
        try:
            # Try to get health metrics from the circuit breaker
            if hasattr(circuit_breaker, 'get_health_metrics'):
                cb_metrics = circuit_breaker.get_health_metrics()
                result['main_breaker'] = cb_metrics
            # Fallback to get_stats if get_health_metrics not available
            elif hasattr(circuit_breaker, 'get_stats'):
                stats = circuit_breaker.get_stats()
                result['main_breaker'] = {
                    'trip_count': stats.get('total_trips', 0),
                    'current_state': 'open' if stats.get('is_open', False) else 'closed',
                    'average_pause_duration': stats.get('total_paused_seconds', 0.0),
                    'consecutive_failures': stats.get('consecutive_failures', 0),
                    'is_tripped': stats.get('is_open', False),
                    'enabled': stats.get('enabled', True),
                    'success_count': stats.get('success_count', 0),
                    'failure_count': stats.get('failure_count', 0),
                }
                # Calculate failure rate
                total = result['main_breaker']['success_count'] + result['main_breaker']['failure_count']
                if total > 0:
                    result['main_breaker']['failure_rate'] = round(
                        result['main_breaker']['failure_count'] / total, 4
                    )
        except Exception as e:
            result['main_breaker'] = {'error': str(e)}
    else:
        result['main_breaker'] = None

    # Get per-keyword circuit breaker metrics
    if per_keyword_breaker is not None:
        try:
            keyword_states = {}
            active_keywords = 0
            rate_limited_keywords = 0
            total_trips = 0

            # Access internal keyword states (assumes _keyword_states attribute)
            if hasattr(per_keyword_breaker, '_keyword_states'):
                for keyword, state in per_keyword_breaker._keyword_states.items():
                    active_keywords += 1
                    if state.is_open:
                        rate_limited_keywords += 1
                    total_trips += getattr(state, 'total_trips', 0)

                    keyword_states[keyword] = {
                        'is_open': state.is_open,
                        'consecutive_failures': getattr(state, 'consecutive_failures', 0),
                        'total_trips': getattr(state, 'total_trips', 0),
                        'consecutive_successes': getattr(state, 'consecutive_successes', 0),
                        'pause_history_count': len(getattr(state, 'pause_history', [])),
                    }

            # Global fallback state
            global_tripped = False
            if hasattr(per_keyword_breaker, 'is_global_tripped'):
                global_tripped = per_keyword_breaker.is_global_tripped()

            result['per_keyword_breaker'] = {
                'enabled': getattr(per_keyword_breaker, '_config', None) is not None,
                'active_keywords': active_keywords,
                'rate_limited_keywords': rate_limited_keywords,
                'rate_limit_percentage': round(rate_limited_keywords / active_keywords, 4) if active_keywords > 0 else 0,
                'total_trips': total_trips,
                'global_fallback_tripped': global_tripped,
                'keyword_states': keyword_states,
            }
        except Exception as e:
            result['per_keyword_breaker'] = {'error': str(e)}
    else:
        result['per_keyword_breaker'] = None

    # Add recommendations
    recommendations = []
    if result.get('per_keyword_breaker'):
        pkb = result['per_keyword_breaker']
        if pkb.get('global_fallback_tripped'):
            recommendations.append({
                'severity': 'critical',
                'message': 'Global circuit breaker tripped - most keywords are rate-limited',
                'action': 'Wait for global recovery or increase pause durations',
            })
        if pkb.get('rate_limit_percentage', 0) > 0.5:
            recommendations.append({
                'severity': 'warning',
                'message': f"{pkb['rate_limit_percentage']*100:.1f}% of keywords are rate-limited",
                'action': 'Consider using different keywords or waiting longer between requests',
            })

    if result.get('main_breaker'):
        mb = result['main_breaker']
        if mb.get('is_tripped'):
            recommendations.append({
                'severity': 'warning',
                'message': 'Main circuit breaker is open',
                'action': 'Wait for recovery or check for persistent failures',
            })

    result['recommendations'] = recommendations

    return result if result else None


def get_tier_effectiveness_health(
    escalation_manager: Optional[Any] = None,
) -> Optional[Dict[str, Any]]:
    """Get tier effectiveness health metrics from escalation manager.

    Provides ML-style pattern recognition for tier effectiveness, including
    success rates per trigger category and recommendations.

    Args:
        escalation_manager: Optional EscalationManager instance to get metrics from

    Returns:
        Dict with tier effectiveness metrics or None if no escalation manager provided
    """
    if escalation_manager is None:
        return None

    try:
        # Get tier effectiveness data
        tier_effectiveness = escalation_manager.get_tier_effectiveness()
        tier_recommendations = escalation_manager.get_tier_recommendations()

        # Get tier counts if available (success/failure counts)
        tier_counts: Dict[str, Dict[str, int]] = {}
        if hasattr(escalation_manager, 'get_tier_counts'):
            tier_counts = escalation_manager.get_tier_counts()

        # Build visualization data - tier comparisons
        tier_visualization: Dict[str, Any] = {}
        tier_names = ['tier1', 'tier2', 'tier3', 'tier4', 'tier5']

        for category, rates in tier_effectiveness.items():
            vis_data = {
                'tiers': [],
                'best_tier': None,
                'best_rate': 0.0,
            }

            for tier_name in tier_names:
                rate = rates.get(tier_name, 0.0)
                count_data = tier_counts.get(category, {}).get(tier_name, {})
                success_count = count_data.get('successes', 0)
                failure_count = count_data.get('failures', 0)
                total = success_count + failure_count

                # Create visual bar (0-50 chars based on rate)
                bar_length = int(rate * 50) if rate > 0 else 0
                bar = '█' * bar_length + '░' * (50 - bar_length)

                tier_info = {
                    'name': tier_name,
                    'success_rate': round(rate, 4),
                    'success_count': success_count,
                    'failure_count': failure_count,
                    'total_attempts': total,
                    'visual_bar': bar,
                    'status': 'excellent' if rate >= 0.8 else 'good' if rate >= 0.6 else 'fair' if rate >= 0.4 else 'poor' if rate > 0 else 'no_data',
                }
                vis_data['tiers'].append(tier_info)

                if rate > vis_data['best_rate']:
                    vis_data['best_rate'] = rate
                    vis_data['best_tier'] = tier_name

            tier_visualization[category] = vis_data

        # Build metrics dict
        metrics: Dict[str, Any] = {
            'tier_effectiveness': tier_effectiveness,
            'tier_recommendations': tier_recommendations,
            'tracked_categories': list(tier_effectiveness.keys()),
            'visualization': tier_visualization,
            'tier_counts': tier_counts,
        }

        # Add per-category expected success rates
        expected_rates: Dict[str, float] = {}
        for category in tier_effectiveness.keys():
            rate = escalation_manager.get_expected_success_rate(category)
            if rate is not None:
                expected_rates[category] = round(rate, 4)
        metrics['expected_success_rates'] = expected_rates

        # Add best tier suggestions (60% threshold)
        best_tiers: Dict[str, str] = {}
        for category in tier_effectiveness.keys():
            best_tier = escalation_manager.get_best_tier_for_category(
                category, min_success_rate=0.60
            )
            if best_tier is not None:
                best_tiers[category] = best_tier.name
        metrics['recommended_tiers_60pct'] = best_tiers

        # Count overall effectiveness
        total_categories = len(tier_effectiveness)
        high_effectiveness = sum(
            1 for cat, rates in tier_effectiveness.items()
            if any(r >= 0.60 for r in rates.values())
        )

        # Add actionable recommendations
        action_recommendations: List[Dict[str, Any]] = []

        # Analyze tier patterns
        for category, vis in tier_visualization.items():
            if not vis['tiers']:
                continue

            # Check if any tier is working well
            working_tiers = [t for t in vis['tiers'] if t['status'] in ('excellent', 'good')]
            if not working_tiers and vis['best_rate'] == 0:
                action_recommendations.append({
                    'severity': 'critical',
                    'category': category,
                    'message': f'No tier has succeeded for category "{category}"',
                    'action': 'Check network connectivity and authentication',
                })
            elif working_tiers:
                best = working_tiers[0]
                action_recommendations.append({
                    'severity': 'info',
                    'category': category,
                    'message': f'Tier "{best["name"]}" works best for "{category}" ({best["success_rate"]*100:.0f}% success)',
                    'action': f'Consider prioritizing tier {best["name"]} for this category',
                })

        # Add tier recommendations to metrics
        metrics['actionable_recommendations'] = action_recommendations

        metrics['summary'] = {
            'total_tracked_categories': total_categories,
            'high_effectiveness_categories': high_effectiveness,
            'has_recommendations': len(tier_recommendations) > 0,
            'has_actionable_recommendations': len(action_recommendations) > 0,
        }

        return metrics
    except Exception as e:
        return {'error': str(e)}
