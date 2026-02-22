"""Health check module for pipeline stage validation.

US-88-005: Pre-stage health checks that validate external dependencies
before running each pipeline stage.
"""

import os
import re
import socket
import subprocess
import logging
import time
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
        'youtube_api': True,
    })

    # YouTube API check settings
    youtube_api_check_enabled: bool = True
    youtube_api_timeout_seconds: float = 10.0

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
            youtube_api_check_enabled=hc_dict.get('youtube_api_check_enabled', True),
            youtube_api_timeout_seconds=hc_dict.get('youtube_api_timeout_seconds', 10.0),
            enabled_checks=hc_dict.get('enabled_checks', {
                'network': True,
                'disk_space': True,
                'memory': True,
                'embedding': True,
                'ffmpeg': True,
                'ytdlp': True,
                'llm_provider': True,
                'youtube_api': True,
            }),
        )

    def check_network(self, timeout: Optional[float] = None) -> HealthCheckResult:
        """Check network connectivity.

        Tests connectivity by attempting to connect to specified hosts.
        """
        import time
        start = time.perf_counter()
        logger.debug(f"[HEALTH_CHECK] Network check starting (timeout: {timeout or self.health_config.network_timeout_seconds}s)")

        if not self.health_config.enabled_checks.get('network', True):
            logger.debug("[HEALTH_CHECK] Network check disabled")
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

                logger.info(f"[HEALTH_CHECK] Network: OK (connected to {host}:{port}, duration: {duration_ms:.1f}ms)")
                logger.debug(f"[HEALTH_CHECK] Network details: host={host}, port={port}")

                return HealthCheckResult(
                    name="network",
                    status=HealthStatus.OK,
                    message=f"Network connectivity OK (connected to {host})",
                    details={'host': host, 'port': port},
                    duration_ms=duration_ms,
                )
            except (socket.timeout, socket.error) as e:
                logger.debug(f"[HEALTH_CHECK] Network check failed for {host}: {e}")
                continue

        duration_ms = (time.perf_counter() - start) * 1000
        logger.error(f"[HEALTH_CHECK] Network: FAILED - All hosts unreachable ({hosts}, duration: {duration_ms:.1f}ms). Check internet connectivity.")

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
        logger.debug(f"[HEALTH_CHECK] Disk space check starting (warning threshold: {self.health_config.disk_space_warning_gb}GB, critical: {self.health_config.disk_space_critical_gb}GB)")

        if not self.health_config.enabled_checks.get('disk_space', True):
            logger.debug("[HEALTH_CHECK] Disk space check disabled")
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
                    logger.debug(f"[HEALTH_CHECK] Disk space check path does not exist: {check_path}")
                    continue
                stat = os.statvfs(check_path) if hasattr(os, 'statvfs') else None
                if stat:
                    free_bytes = stat.f_bavail * stat.f_frsize
                    free_gb = free_bytes / (1024 ** 3)

                    duration_ms = (time.perf_counter() - start) * 1000

                    if free_gb < self.health_config.disk_space_critical_gb:
                        logger.error(f"[HEALTH_CHECK] Disk space: CRITICAL - Only {free_gb:.1f}GB free at {check_path} (threshold: {self.health_config.disk_space_critical_gb}GB, duration: {duration_ms:.1f}ms). Free up disk space to continue.")

                        return HealthCheckResult(
                            name="disk_space",
                            status=HealthStatus.FAILED,
                            message=f"Critical: Only {free_gb:.1f}GB free at {check_path}",
                            details={'path': check_path, 'free_gb': free_gb},
                            duration_ms=duration_ms,
                        )
                    elif free_gb < self.health_config.disk_space_warning_gb:
                        logger.warning(f"[HEALTH_CHECK] Disk space: LOW - {free_gb:.1f}GB free at {check_path} (warning threshold: {self.health_config.disk_space_warning_gb}GB, duration: {duration_ms:.1f}ms). Consider freeing up disk space.")

                        return HealthCheckResult(
                            name="disk_space",
                            status=HealthStatus.WARNING,
                            message=f"Low disk space: {free_gb:.1f}GB free at {check_path}",
                            details={'path': check_path, 'free_gb': free_gb},
                            duration_ms=duration_ms,
                        )
                    else:
                        logger.info(f"[HEALTH_CHECK] Disk space: OK - {free_gb:.1f}GB free at {check_path} (duration: {duration_ms:.1f}ms)")
                        logger.debug(f"[HEALTH_CHECK] Disk space details: path={check_path}, free_gb={free_gb:.2f}, warning_threshold={self.health_config.disk_space_warning_gb}, critical_threshold={self.health_config.disk_space_critical_gb}")

                        return HealthCheckResult(
                            name="disk_space",
                            status=HealthStatus.OK,
                            message=f"Disk space OK: {free_gb:.1f}GB free at {check_path}",
                            details={'path': check_path, 'free_gb': free_gb},
                            duration_ms=duration_ms,
                        )
            except OSError as e:
                logger.debug(f"[HEALTH_CHECK] Disk space check error for {check_path}: {e}")
                continue

        duration_ms = (time.perf_counter() - start) * 1000
        logger.warning(f"[HEALTH_CHECK] Disk space: WARNING - Could not check disk space (paths not accessible: {check_paths}, duration: {duration_ms:.1f}ms)")

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
        logger.debug(f"[HEALTH_CHECK] Memory check starting (warning threshold: {self.health_config.memory_warning_percent}%, critical: {self.health_config.memory_critical_percent}%)")

        if not self.health_config.enabled_checks.get('memory', True):
            logger.debug("[HEALTH_CHECK] Memory check disabled")
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
                    logger.debug("[HEALTH_CHECK] Memory: SKIPPED - psutil not available")
                    return HealthCheckResult(
                        name="memory",
                        status=HealthStatus.SKIPPED,
                        message="Memory check not available on this platform (psutil recommended)",
                        details={},
                        duration_ms=duration_ms,
                    )

            duration_ms = (time.perf_counter() - start) * 1000

            if used_percent >= self.health_config.memory_critical_percent:
                logger.error(f"[HEALTH_CHECK] Memory: CRITICAL - {used_percent:.1f}% used (threshold: {self.health_config.memory_critical_percent}%, duration: {duration_ms:.1f}ms). Close other applications or the pipeline may fail.")

                return HealthCheckResult(
                    name="memory",
                    status=HealthStatus.FAILED,
                    message=f"Critical: Memory usage at {used_percent:.1f}%",
                    details={'used_percent': used_percent, 'available_gb': available_gb, 'total_gb': total_gb},
                    duration_ms=duration_ms,
                )
            elif used_percent >= self.health_config.memory_warning_percent:
                logger.warning(f"[HEALTH_CHECK] Memory: HIGH - {used_percent:.1f}% used (warning threshold: {self.health_config.memory_warning_percent}%, duration: {duration_ms:.1f}ms). Available: {available_gb:.1f}GB of {total_gb:.1f}GB.")

                return HealthCheckResult(
                    name="memory",
                    status=HealthStatus.WARNING,
                    message=f"High memory usage: {used_percent:.1f}%",
                    details={'used_percent': used_percent, 'available_gb': available_gb, 'total_gb': total_gb},
                    duration_ms=duration_ms,
                )
            else:
                logger.info(f"[HEALTH_CHECK] Memory: OK - {used_percent:.1f}% used (duration: {duration_ms:.1f}ms)")
                logger.debug(f"[HEALTH_CHECK] Memory details: used_percent={used_percent:.1f}, available_gb={available_gb:.2f}, total_gb={total_gb:.2f}, warning_threshold={self.health_config.memory_warning_percent}, critical_threshold={self.health_config.memory_critical_percent}")

                return HealthCheckResult(
                    name="memory",
                    status=HealthStatus.OK,
                    message=f"Memory OK: {used_percent:.1f}% used",
                    details={'used_percent': used_percent, 'available_gb': available_gb, 'total_gb': total_gb},
                    duration_ms=duration_ms,
                )
        except Exception as e:
            duration_ms = (time.perf_counter() - start) * 1000
            logger.warning(f"[HEALTH_CHECK] Memory: WARNING - Could not check memory: {str(e)}")

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
        logger.debug("[HEALTH_CHECK] Embedding provider check starting")

        if not self.health_config.enabled_checks.get('embedding', True):
            logger.debug("[HEALTH_CHECK] Embedding provider check disabled")
            return HealthCheckResult(
                name="embedding_provider",
                status=HealthStatus.SKIPPED,
                message="Embedding check disabled",
                duration_ms=0.0,
            )

        if not self.health_config.embedding_check_enabled:
            logger.debug("[HEALTH_CHECK] Embedding provider check disabled in config")
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
            logger.error(f"[HEALTH_CHECK] Embedding provider: FAILED - No embedding configuration found (duration: {duration_ms:.1f}ms)")

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

        logger.debug(f"[HEALTH_CHECK] Embedding provider: {provider}")

        # Check based on provider type
        if provider == 'local':
            # Check if local embedding service is available
            return self._check_local_embedding()
        elif provider in ('gemini', 'anthropic'):
            # For cloud providers, check network and API key
            return self._check_cloud_embedding(provider)
        else:
            duration_ms = (time.perf_counter() - start) * 1000
            logger.warning(f"[HEALTH_CHECK] Embedding provider: WARNING - Unknown provider '{provider}' (duration: {duration_ms:.1f}ms)")

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
        logger.debug("[HEALTH_CHECK] Embedding provider (local/Ollama) check starting")

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
                logger.info(f"[HEALTH_CHECK] Embedding provider: OK - Local (Ollama) is running (duration: {duration_ms:.1f}ms)")
                logger.debug(f"[HEALTH_CHECK] Embedding provider details: provider=local, url=http://localhost:11434")

                return HealthCheckResult(
                    name="embedding_provider",
                    status=HealthStatus.OK,
                    message="Local embedding provider (Ollama) is running",
                    details={'provider': 'local', 'url': 'http://localhost:11434'},
                    duration_ms=duration_ms,
                )
            else:
                logger.error(f"[HEALTH_CHECK] Embedding provider: FAILED - Local (Ollama) not responding (returncode: {result.returncode}, duration: {duration_ms:.1f}ms). Start Ollama to enable local embeddings.")

                return HealthCheckResult(
                    name="embedding_provider",
                    status=HealthStatus.FAILED,
                    message="Local embedding provider (Ollama) not responding",
                    details={'provider': 'local', 'returncode': result.returncode},
                    duration_ms=duration_ms,
                )
        except FileNotFoundError:
            duration_ms = (time.perf_counter() - start) * 1000
            logger.error(f"[HEALTH_CHECK] Embedding provider: FAILED - curl not found (duration: {duration_ms:.1f}ms). Install curl to check local embedding provider.")

            return HealthCheckResult(
                name="embedding_provider",
                status=HealthStatus.FAILED,
                message="curl not found - cannot check local embedding",
                details={'provider': 'local', 'error': 'curl not available'},
                duration_ms=duration_ms,
            )
        except subprocess.TimeoutExpired:
            duration_ms = (time.perf_counter() - start) * 1000
            logger.error(f"[HEALTH_CHECK] Embedding provider: FAILED - Local (Ollama) timeout (duration: {duration_ms:.1f}ms). Ollama may be overloaded or not responding.")

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
        logger.debug(f"[HEALTH_CHECK] Embedding provider ({provider}/cloud) check starting")

        # Check if API key is configured
        api_key = None
        if provider == 'gemini':
            api_key = os.environ.get('GEMINI_API_KEY')
        elif provider == 'anthropic':
            api_key = os.environ.get('ANTHROPIC_API_KEY')

        if not api_key:
            duration_ms = (time.perf_counter() - start) * 1000
            logger.error(f"[HEALTH_CHECK] Embedding provider: FAILED - No API key configured for {provider} (duration: {duration_ms:.1f}ms). Set {provider.upper()}_API_KEY environment variable.")

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
            logger.info(f"[HEALTH_CHECK] Embedding provider: OK - {provider} cloud configured with network OK (duration: {duration_ms:.1f}ms)")
            logger.debug(f"[HEALTH_CHECK] Embedding provider details: provider={provider}, network=ok")

            return HealthCheckResult(
                name="embedding_provider",
                status=HealthStatus.OK,
                message=f"Cloud embedding provider ({provider}) configured and network OK",
                details={'provider': provider, 'network': 'ok'},
                duration_ms=duration_ms,
            )
        else:
            duration_ms = (time.perf_counter() - start) * 1000
            logger.error(f"[HEALTH_CHECK] Embedding provider: FAILED - {provider} cloud network unavailable (duration: {duration_ms:.1f}ms). Check internet connectivity.")

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
        logger.debug(f"[HEALTH_CHECK] FFmpeg check starting (min version: {self.health_config.ffmpeg_min_version})")

        # Check if FFmpeg check is enabled
        if not self.health_config.ffmpeg_required:
            logger.debug("[HEALTH_CHECK] FFmpeg check disabled")
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

                    logger.info(f"[HEALTH_CHECK] FFmpeg: OK - version {version} (minimum: {min_version}, duration: {duration_ms:.1f}ms)")
                    logger.debug(f"[HEALTH_CHECK] FFmpeg details: version={version}, min_version={min_version}")

                    return HealthCheckResult(
                        name="ffmpeg",
                        status=HealthStatus.OK,
                        message=f"FFmpeg {version} available (minimum: {min_version})",
                        details={'version': version, 'min_version': min_version},
                        duration_ms=duration_ms,
                    )
                else:
                    logger.info(f"[HEALTH_CHECK] FFmpeg: OK - available (duration: {duration_ms:.1f}ms)")
                    logger.debug(f"[HEALTH_CHECK] FFmpeg details: output={version_output[:100]}")

                    return HealthCheckResult(
                        name="ffmpeg",
                        status=HealthStatus.OK,
                        message="FFmpeg available",
                        details={'output': version_output[:100]},
                        duration_ms=duration_ms,
                    )
            else:
                duration_ms = (time.perf_counter() - start) * 1000
                logger.error(f"[HEALTH_CHECK] FFmpeg: FAILED - returned error (returncode: {result.returncode}, duration: {duration_ms:.1f}ms). Install FFmpeg to enable video processing.")

                return HealthCheckResult(
                    name="ffmpeg",
                    status=HealthStatus.FAILED,
                    message="FFmpeg not available or returned error",
                    details={'returncode': result.returncode, 'stderr': result.stderr[:200]},
                    duration_ms=duration_ms,
                )
        except FileNotFoundError:
            duration_ms = (time.perf_counter() - start) * 1000
            logger.error(f"[HEALTH_CHECK] FFmpeg: FAILED - not found in PATH (duration: {duration_ms:.1f}ms). Install FFmpeg to enable video processing.")

            return HealthCheckResult(
                name="ffmpeg",
                status=HealthStatus.FAILED,
                message="FFmpeg not found in PATH",
                details={'error': 'ffmpeg command not found'},
                duration_ms=duration_ms,
            )
        except subprocess.TimeoutExpired:
            duration_ms = (time.perf_counter() - start) * 1000
            logger.error(f"[HEALTH_CHECK] FFmpeg: FAILED - check timed out (duration: {duration_ms:.1f}ms)")

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
        logger.debug(f"[HEALTH_CHECK] yt-dlp check starting (min version year: {self.health_config.ytdlp_min_version})")

        if not self.health_config.enabled_checks.get('ytdlp', True):
            logger.debug("[HEALTH_CHECK] yt-dlp check disabled")
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
                        logger.info(f"[HEALTH_CHECK] yt-dlp: OK - version {version_output} (min year: {min_version_year}, duration: {duration_ms:.1f}ms)")
                        logger.debug(f"[HEALTH_CHECK] yt-dlp details: version={version_output}, min_version={min_version_year}")

                        return HealthCheckResult(
                            name="ytdlp",
                            status=HealthStatus.OK,
                            message=f"yt-dlp {version_output} available (minimum year: {min_version_year})",
                            details={'version': version_output, 'min_version': min_version_year},
                            duration_ms=duration_ms,
                        )
                    else:
                        status = HealthStatus.FAILED if self.health_config.ytdlp_required else HealthStatus.WARNING
                        msg = f"[HEALTH_CHECK] yt-dlp: {'FAILED' if status == HealthStatus.FAILED else 'WARNING'} - version {version_output} too old (min year: {min_version_year}, duration: {duration_ms:.1f}ms)"
                        if status == HealthStatus.FAILED:
                            logger.error(msg)
                        else:
                            logger.warning(msg)

                        return HealthCheckResult(
                            name="ytdlp",
                            status=status,
                            message=f"yt-dlp {version_output} too old (minimum year: {min_version_year})",
                            details={'version': version_output, 'min_version': min_version_year},
                            duration_ms=duration_ms,
                        )
                else:
                    logger.info(f"[HEALTH_CHECK] yt-dlp: OK - {version_output} (duration: {duration_ms:.1f}ms)")
                    logger.debug(f"[HEALTH_CHECK] yt-dlp details: version={version_output}")

                    return HealthCheckResult(
                        name="ytdlp",
                        status=HealthStatus.OK,
                        message=f"yt-dlp available: {version_output}",
                        details={'version': version_output},
                        duration_ms=duration_ms,
                    )
            else:
                duration_ms = (time.perf_counter() - start) * 1000
                logger.error(f"[HEALTH_CHECK] yt-dlp: FAILED - returned error (returncode: {result.returncode}, duration: {duration_ms:.1f}ms)")

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
            msg = f"[HEALTH_CHECK] yt-dlp: {'FAILED' if status == HealthStatus.FAILED else 'WARNING'} - not found in PATH (duration: {duration_ms:.1f}ms)"
            if status == HealthStatus.FAILED:
                logger.error(msg)
            else:
                logger.warning(msg)

            return HealthCheckResult(
                name="ytdlp",
                status=status,
                message="yt-dlp not found in PATH",
                details={'error': 'yt-dlp command not found'},
                duration_ms=duration_ms,
            )
        except subprocess.TimeoutExpired:
            duration_ms = (time.perf_counter() - start) * 1000
            logger.error(f"[HEALTH_CHECK] yt-dlp: FAILED - check timed out (duration: {duration_ms:.1f}ms)")

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
        logger.debug("[HEALTH_CHECK] LLM provider check starting")

        if not self.health_config.enabled_checks.get('llm_provider', True):
            logger.debug("[HEALTH_CHECK] LLM provider check disabled")
            return HealthCheckResult(
                name="llm_provider",
                status=HealthStatus.SKIPPED,
                message="LLM provider check disabled",
                duration_ms=0.0,
            )

        if not self.health_config.llm_provider_check_enabled:
            logger.debug("[HEALTH_CHECK] LLM provider check disabled in config")
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
            logger.error(f"[HEALTH_CHECK] LLM provider: FAILED - No LLM configuration found (duration: {duration_ms:.1f}ms)")

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

        logger.debug(f"[HEALTH_CHECK] LLM provider: {provider}")

        # Check based on provider type
        if provider == 'ollama':
            return self._check_ollama_llm(start)
        elif provider in ('gemini', 'anthropic'):
            return self._check_cloud_llm(provider, start)
        else:
            duration_ms = (time.perf_counter() - start) * 1000
            logger.warning(f"[HEALTH_CHECK] LLM provider: WARNING - Unknown provider '{provider}' (duration: {duration_ms:.1f}ms)")

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
        logger.debug("[HEALTH_CHECK] LLM provider (Ollama) check starting")

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

                    logger.info(f"[HEALTH_CHECK] LLM provider: OK - Ollama running with {model_count} models (duration: {duration_ms:.1f}ms)")
                    logger.debug(f"[HEALTH_CHECK] LLM provider details: provider=ollama, url=http://localhost:11434, models={model_count}")

                    return HealthCheckResult(
                        name="llm_provider",
                        status=HealthStatus.OK,
                        message=f"Ollama LLM provider is running ({model_count} models available)",
                        details={'provider': 'ollama', 'url': 'http://localhost:11434', 'models': model_count},
                        duration_ms=duration_ms,
                    )
                except json.JSONDecodeError:
                    logger.info(f"[HEALTH_CHECK] LLM provider: OK - Ollama running (duration: {duration_ms:.1f}ms)")
                    logger.debug(f"[HEALTH_CHECK] LLM provider details: provider=ollama, url=http://localhost:11434")

                    return HealthCheckResult(
                        name="llm_provider",
                        status=HealthStatus.OK,
                        message="Ollama LLM provider is running",
                        details={'provider': 'ollama', 'url': 'http://localhost:11434'},
                        duration_ms=duration_ms,
                    )
            else:
                duration_ms = (time.perf_counter() - start) * 1000
                logger.error(f"[HEALTH_CHECK] LLM provider: FAILED - Ollama not responding (returncode: {result.returncode}, duration: {duration_ms:.1f}ms). Start Ollama to enable local LLM.")

                return HealthCheckResult(
                    name="llm_provider",
                    status=HealthStatus.FAILED,
                    message="Ollama LLM provider not responding",
                    details={'provider': 'ollama', 'returncode': result.returncode},
                    duration_ms=duration_ms,
                )
        except FileNotFoundError:
            duration_ms = (time.perf_counter() - start) * 1000
            logger.error(f"[HEALTH_CHECK] LLM provider: FAILED - curl not found (duration: {duration_ms:.1f}ms). Install curl to check Ollama LLM.")

            return HealthCheckResult(
                name="llm_provider",
                status=HealthStatus.FAILED,
                message="curl not found - cannot check Ollama LLM",
                details={'provider': 'ollama', 'error': 'curl not available'},
                duration_ms=duration_ms,
            )
        except subprocess.TimeoutExpired:
            duration_ms = (time.perf_counter() - start) * 1000
            logger.error(f"[HEALTH_CHECK] LLM provider: FAILED - Ollama timeout (duration: {duration_ms:.1f}ms). Ollama may be overloaded or not responding.")

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
        logger.debug(f"[HEALTH_CHECK] LLM provider ({provider}/cloud) check starting")

        # Check if API key is configured
        api_key = None
        if provider == 'gemini':
            api_key = os.environ.get('GEMINI_API_KEY') or os.environ.get('GOOGLE_API_KEY')
        elif provider == 'anthropic':
            api_key = os.environ.get('ANTHROPIC_API_KEY')

        if not api_key:
            duration_ms = (time.perf_counter() - start) * 1000
            logger.error(f"[HEALTH_CHECK] LLM provider: FAILED - No API key configured for {provider} (duration: {duration_ms:.1f}ms). Set {provider.upper()}_API_KEY environment variable.")

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
                    logger.info(f"[HEALTH_CHECK] LLM provider: OK - {provider} cloud reachable (HTTP {http_code}, duration: {duration_ms:.1f}ms)")
                    logger.debug(f"[HEALTH_CHECK] LLM provider details: provider={provider}, http_code={http_code}")

                    return HealthCheckResult(
                        name="llm_provider",
                        status=HealthStatus.OK,
                        message=f"Cloud LLM provider ({provider}) is reachable (HTTP {http_code})",
                        details={'provider': provider, 'http_code': http_code},
                        duration_ms=duration_ms,
                    )
                else:
                    logger.error(f"[HEALTH_CHECK] LLM provider: FAILED - {provider} returned HTTP {http_code} (duration: {duration_ms:.1f}ms). Check API key validity.")

                    return HealthCheckResult(
                        name="llm_provider",
                        status=HealthStatus.FAILED,
                        message=f"Cloud LLM provider ({provider}) returned HTTP {http_code}",
                        details={'provider': provider, 'http_code': http_code},
                        duration_ms=duration_ms,
                    )
            else:
                logger.error(f"[HEALTH_CHECK] LLM provider: FAILED - Cannot reach {provider} API endpoint (returncode: {result.returncode}, duration: {duration_ms:.1f}ms). Check internet connectivity.")

                return HealthCheckResult(
                    name="llm_provider",
                    status=HealthStatus.FAILED,
                    message=f"Cannot reach {provider} API endpoint",
                    details={'provider': provider, 'returncode': result.returncode},
                    duration_ms=duration_ms,
                )
        except FileNotFoundError:
            duration_ms = (time.perf_counter() - start) * 1000
            logger.warning(f"[HEALTH_CHECK] LLM provider: WARNING - curl not found (duration: {duration_ms:.1f}ms). Cannot verify {provider} cloud connectivity.")

            return HealthCheckResult(
                name="llm_provider",
                status=HealthStatus.WARNING,
                message="curl not found - cannot verify cloud LLM connectivity",
                details={'provider': provider, 'error': 'curl not available'},
                duration_ms=duration_ms,
            )
        except subprocess.TimeoutExpired:
            duration_ms = (time.perf_counter() - start) * 1000
            logger.error(f"[HEALTH_CHECK] LLM provider: FAILED - {provider} timeout (duration: {duration_ms:.1f}ms). Check internet connectivity.")

            return HealthCheckResult(
                name="llm_provider",
                status=HealthStatus.FAILED,
                message=f"{provider} LLM provider timeout",
                details={'provider': provider, 'timeout': timeout},
                duration_ms=duration_ms,
            )

    def check_youtube_api(self) -> HealthCheckResult:
        """Check YouTube API availability and key validity (US-149-002).

        Validates that the YouTube API key is valid by performing a lightweight
        health check call. This runs without consuming significant quota.
        """
        import time
        start = time.perf_counter()
        logger.debug("[HEALTH_CHECK] YouTube API check starting")

        if not self.health_config.enabled_checks.get('youtube_api', True):
            logger.debug("[HEALTH_CHECK] YouTube API check disabled")
            return HealthCheckResult(
                name="youtube_api",
                status=HealthStatus.SKIPPED,
                message="YouTube API check disabled",
                duration_ms=0.0,
            )

        if not self.health_config.youtube_api_check_enabled:
            logger.debug("[HEALTH_CHECK] YouTube API check disabled in config")
            return HealthCheckResult(
                name="youtube_api",
                status=HealthStatus.SKIPPED,
                message="YouTube API check disabled in config",
                duration_ms=0.0,
            )

        # Get YouTube API key from config
        api_key = None

        # Check download section for API key
        download_config = getattr(self.config, 'download', None)
        if download_config:
            if isinstance(download_config, dict):
                api_key = download_config.get('youtube_api_key')
                if not api_key:
                    api_keys_list = download_config.get('youtube_api_keys', [])
                    if api_keys_list:
                        api_key = api_keys_list[0] if api_keys_list else None
            elif hasattr(download_config, 'youtube_api_key'):
                api_key = download_config.youtube_api_key
                if not api_key:
                    api_keys = getattr(download_config, 'youtube_api_keys', [])
                    if api_keys:
                        api_key = api_keys[0] if api_keys else None

        # Also check environment variable as fallback
        if not api_key:
            api_key = os.environ.get('YOUTUBE_API_KEY')

        if not api_key:
            duration_ms = (time.perf_counter() - start) * 1000
            logger.error(f"[HEALTH_CHECK] YouTube API: FAILED - No API key configured (duration: {duration_ms:.1f}ms). Set YOUTUBE_API_KEY environment variable or configure in config.")

            return HealthCheckResult(
                name="youtube_api",
                status=HealthStatus.FAILED,
                message="No YouTube API key configured",
                details={'error': 'API key not found in config or environment'},
                duration_ms=duration_ms,
            )

        # Perform health check using YouTubeAPIClient
        try:
            from src.downloader.youtube_api_client import YouTubeAPIClient

            client = YouTubeAPIClient(
                api_key=api_key,
                quota_limit=10000,
                timeout=int(self.health_config.youtube_api_timeout_seconds),
                auto_scale_quota=False,  # Don't auto-scale for health check
            )

            is_valid, error_message, quota_info = client.health_check()

            # US-155-012: Get retry budget stats
            retry_budget_info = {}
            try:
                if hasattr(client, 'get_retry_budget_stats'):
                    retry_budget_info = client.get_retry_budget_stats()
                elif hasattr(client, '_retry_budget'):
                    retry_budget_info = client._retry_budget.get_budget_status()
            except Exception:
                pass

            client.close()

            duration_ms = (time.perf_counter() - start) * 1000

            # Build details including retry budget info and rotation strategy
            details = {
                'quota_used': quota_info.get('quota_used'),
                'quota_limit': quota_info.get('quota_limit'),
                'percent_used': quota_info.get('percent_used'),
                'keys_available': quota_info.get('keys_available'),
                'keys_total': quota_info.get('keys_total'),
                'keys_exhausted_count': quota_info.get('keys_exhausted_count'),
                'rotation_strategy': quota_info.get('rotation_strategy'),
            }

            # US-155-012: Add retry budget to details if available
            if retry_budget_info:
                details['retry_budget'] = {
                    'attempts_used': retry_budget_info.get('attempts_used'),
                    'attempts_remaining': retry_budget_info.get('attempts_remaining'),
                    'attempts_max': retry_budget_info.get('attempts_max'),
                    'attempts_utilization_percent': retry_budget_info.get('attempts_utilization_percent'),
                    'budget_exhausted': retry_budget_info.get('budget_exhausted'),
                    'backoff_time_spent': retry_budget_info.get('backoff_time_spent'),
                    'backoff_time_remaining': retry_budget_info.get('backoff_time_remaining'),
                }

            if is_valid:
                logger.info(f"[HEALTH_CHECK] YouTube API: OK - API key valid (quota: {quota_info.get('percent_used', 0):.1f}% used, {quota_info.get('keys_available', 0)}/{quota_info.get('keys_total', 0)} keys, duration: {duration_ms:.1f}ms)")
                logger.debug(f"[HEALTH_CHECK] YouTube API details: quota_used={quota_info.get('quota_used')}, quota_limit={quota_info.get('quota_limit')}, keys_available={quota_info.get('keys_available')}")

                return HealthCheckResult(
                    name="youtube_api",
                    status=HealthStatus.OK,
                    message="YouTube API key is valid",
                    details=details,
                    duration_ms=duration_ms,
                )
            else:
                # US-155-011: Check for all keys exhausted - this is a failure
                keys_available = quota_info.get('keys_available', 0)
                keys_total = quota_info.get('keys_total', 1)
                rotation_strategy = quota_info.get('rotation_strategy', 'unknown')

                # Determine if it's a warning or failure
                status = HealthStatus.WARNING
                if keys_available == 0:
                    # All keys exhausted - this is a failure
                    status = HealthStatus.FAILED
                    error_message = (
                        f"All {keys_total} API keys exhausted. "
                        f"Rotation strategy: {rotation_strategy}. "
                        f"Wait for quota reset or add new API keys."
                    )
                elif 'quota' in error_message.lower() or 'exceeded' in error_message.lower():
                    status = HealthStatus.WARNING  # Quota issues are warnings
                elif 'invalid' in error_message.lower() or 'authentication' in error_message.lower():
                    status = HealthStatus.FAILED  # Invalid key is a failure

                if status == HealthStatus.FAILED:
                    logger.error(f"[HEALTH_CHECK] YouTube API: FAILED - {error_message} (duration: {duration_ms:.1f}ms)")
                else:
                    logger.warning(f"[HEALTH_CHECK] YouTube API: WARNING - {error_message} (duration: {duration_ms:.1f}ms)")

                return HealthCheckResult(
                    name="youtube_api",
                    status=status,
                    message=f"YouTube API issue: {error_message}",
                    details=details,
                    duration_ms=duration_ms,
                )

        except ImportError:
            duration_ms = (time.perf_counter() - start) * 1000
            logger.warning(f"[HEALTH_CHECK] YouTube API: SKIPPED - Module not available (duration: {duration_ms:.1f}ms)")

            return HealthCheckResult(
                name="youtube_api",
                status=HealthStatus.SKIPPED,
                message="YouTube API client not available",
                details={'error': 'Module import failed'},
                duration_ms=duration_ms,
            )
        except Exception as e:
            duration_ms = (time.perf_counter() - start) * 1000
            logger.warning(f"[HEALTH_CHECK] YouTube API: WARNING - Check failed: {str(e)} (duration: {duration_ms:.1f}ms)")

            return HealthCheckResult(
                name="youtube_api",
                status=HealthStatus.WARNING,
                message=f"YouTube API check failed: {str(e)}",
                details={'error': str(e)},
                duration_ms=duration_ms,
            )

    def check_youtube_api_cache(self) -> HealthCheckResult:
        """Check YouTube API cache database integrity (US-155-011).

        Validates that the SQLite cache database is accessible and not corrupted.
        """
        import time
        start = time.perf_counter()
        logger.debug("[HEALTH_CHECK] YouTube API cache check starting")

        if not self.health_config.enabled_checks.get('youtube_api_cache', True):
            logger.debug("[HEALTH_CHECK] YouTube API cache check disabled")
            return HealthCheckResult(
                name="youtube_api_cache",
                status=HealthStatus.SKIPPED,
                message="YouTube API cache check disabled",
                duration_ms=0.0,
            )

        try:
            from src.downloader.youtube_api_cache import YouTubeAPISQLCache
            import sqlite3
            from pathlib import Path

            # Initialize the cache to get the db path
            cache = YouTubeAPISQLCache()
            db_path = cache._db_path

            if not db_path.exists():
                duration_ms = (time.perf_counter() - start) * 1000
                logger.warning(f"[HEALTH_CHECK] YouTube API cache: WARNING - Database does not exist yet (duration: {duration_ms:.1f}ms)")

                return HealthCheckResult(
                    name="youtube_api_cache",
                    status=HealthStatus.WARNING,
                    message="Cache database does not exist yet",
                    details={'db_path': str(db_path)},
                    duration_ms=duration_ms,
                )

            # Perform integrity check using sqlite3
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()

            # Run PRAGMA integrity_check
            cursor.execute("PRAGMA integrity_check")
            integrity_result = cursor.fetchone()

            # Get table info
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = cursor.fetchall()

            # Get cache stats
            cache_stats = {}
            try:
                cursor.execute("SELECT COUNT(*) FROM api_responses")
                cache_stats['total_entries'] = cursor.fetchone()[0]
            except sqlite3.OperationalError:
                cache_stats['total_entries'] = 0

            # Get database size
            db_size_bytes = db_path.stat().st_size
            cache_stats['db_size_mb'] = round(db_size_bytes / (1024 * 1024), 2)

            conn.close()

            duration_ms = (time.perf_counter() - start) * 1000

            if integrity_result and integrity_result[0] == 'ok':
                logger.info(f"[HEALTH_CHECK] YouTube API cache: OK - {len(tables)} tables, {cache_stats.get('total_entries', 0)} entries, {cache_stats.get('db_size_mb', 0)}MB (duration: {duration_ms:.1f}ms)")
                logger.debug(f"[HEALTH_CHECK] YouTube API cache details: db_path={db_path}, tables_count={len(tables)}, total_entries={cache_stats.get('total_entries', 0)}, db_size_mb={cache_stats.get('db_size_mb', 0)}")

                return HealthCheckResult(
                    name="youtube_api_cache",
                    status=HealthStatus.OK,
                    message=f"Cache database integrity OK ({len(tables)} tables, {cache_stats.get('total_entries', 0)} entries)",
                    details={
                        'db_path': str(db_path),
                        'tables_count': len(tables),
                        'total_entries': cache_stats.get('total_entries', 0),
                        'db_size_mb': cache_stats.get('db_size_mb', 0),
                    },
                    duration_ms=duration_ms,
                )
            else:
                logger.error(f"[HEALTH_CHECK] YouTube API cache: FAILED - Integrity check failed: {integrity_result} (duration: {duration_ms:.1f}ms). Database may be corrupted.")

                return HealthCheckResult(
                    name="youtube_api_cache",
                    status=HealthStatus.FAILED,
                    message=f"Cache database integrity check failed: {integrity_result}",
                    details={
                        'db_path': str(db_path),
                        'integrity_result': integrity_result,
                        'tables_count': len(tables),
                    },
                    duration_ms=duration_ms,
                )

        except ImportError:
            duration_ms = (time.perf_counter() - start) * 1000
            logger.warning(f"[HEALTH_CHECK] YouTube API cache: SKIPPED - Module not available (duration: {duration_ms:.1f}ms)")

            return HealthCheckResult(
                name="youtube_api_cache",
                status=HealthStatus.SKIPPED,
                message="YouTube API cache module not available",
                duration_ms=duration_ms,
            )
        except Exception as e:
            duration_ms = (time.perf_counter() - start) * 1000
            logger.warning(f"[HEALTH_CHECK] YouTube API cache: WARNING - Check failed: {str(e)} (duration: {duration_ms:.1f}ms)")

            return HealthCheckResult(
                name="youtube_api_cache",
                status=HealthStatus.WARNING,
                message=f"Cache database check failed: {str(e)}",
                details={'error': str(e)},
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

        # Determine which checks will run for logging
        planned_checks = []
        if stage_name in ('VIDEO_SEARCH', 'CAPTION', 'DOWNLOAD_SEGMENTS', 'MATCH', 'ITERATIVE_MATCH'):
            planned_checks.append('network')
        if stage_name in ('DOWNLOAD_SEGMENTS', 'OUTPUT'):
            planned_checks.append('disk_space')
        if stage_name in ('MATCH', 'ITERATIVE_MATCH'):
            planned_checks.append('memory')
        if stage_name in ('MATCH', 'ITERATIVE_MATCH'):
            planned_checks.append('embedding')
        if stage_name in ('DOWNLOAD_SEGMENTS', 'OUTPUT'):
            planned_checks.append('ffmpeg')
        if stage_name in ('DOWNLOAD_SEGMENTS', 'VIDEO_SEARCH', 'CAPTION'):
            planned_checks.append('ytdlp')
        if stage_name in ('VIDEO_SEARCH', 'CAPTION'):
            planned_checks.append('youtube_api')
            planned_checks.append('youtube_api_cache')
        if stage_name in ('MATCH', 'ITERATIVE_MATCH', 'VIDEO_SEARCH'):
            planned_checks.append('llm_provider')

        # Log health check start with check names
        logger.info(f"[HEALTH_CHECK] Starting checks for stage '{stage_name}': {', '.join(planned_checks)}")

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

        # YouTube API checks - for stages that use YouTube API
        if stage_name in ('VIDEO_SEARCH', 'CAPTION'):
            results.append(self.check_youtube_api())
            # US-155-011: Also check cache database integrity
            results.append(self.check_youtube_api_cache())

        # LLM provider checks - for stages that use LLM
        if stage_name in ('MATCH', 'ITERATIVE_MATCH', 'VIDEO_SEARCH'):
            results.append(self.check_llm_provider())

        # Log summary of checks performed
        ok_count = sum(1 for r in results if r.status == HealthStatus.OK)
        warning_count = sum(1 for r in results if r.status == HealthStatus.WARNING)
        failed_count = sum(1 for r in results if r.status == HealthStatus.FAILED)
        skipped_count = sum(1 for r in results if r.status == HealthStatus.SKIPPED)
        total_duration_ms = sum(r.duration_ms for r in results)

        logger.info(f"[HEALTH_CHECK] Stage '{stage_name}' checks complete: {ok_count} OK, {warning_count} warnings, {failed_count} failed, {skipped_count} skipped (total duration: {total_duration_ms:.1f}ms)")

        # Log failed checks with actionable context
        for result in results:
            if result.status == HealthStatus.FAILED:
                logger.error(f"[HEALTH_CHECK] {stage_name}: {result.name} check FAILED - {result.message}")
            elif result.status == HealthStatus.WARNING:
                logger.warning(f"[HEALTH_CHECK] {stage_name}: {result.name} check WARNING - {result.message}")

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

        # Log summary of pre-stage health check status
        warning_count = sum(1 for r in results if r.status == HealthStatus.WARNING)
        failed_count = sum(1 for r in results if r.status == HealthStatus.FAILED)

        if failed_count > 0:
            logger.warning(f"[PRE-STAGE] {stage_name}: {failed_count} health check(s) failed, {warning_count} warning(s) - continuing with stage execution")
        elif warning_count > 0:
            logger.info(f"[PRE-STAGE] {stage_name}: {warning_count} health check warning(s) - continuing with stage execution")
        else:
            logger.info(f"[PRE-STAGE] {stage_name}: All health checks passed")

        # Log warnings but continue execution (health checks are advisory)
        for result in results:
            if result.status == HealthStatus.WARNING:
                logger.warning(f"[PRE-STAGE] {stage_name}: {result.name} - {result.message}")
            elif result.status == HealthStatus.FAILED:
                logger.warning(f"[PRE-STAGE] {stage_name}: {result.name} - {result.message} (continuing anyway)")

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
        logger.debug("[HEALTH_CHECK] Circuit breaker check starting")

        if not self.health_config.enabled_checks.get('circuit_breaker', True):
            logger.debug("[HEALTH_CHECK] Circuit breaker check disabled")
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
                logger.warning(f"[HEALTH_CHECK] Circuit breaker: WARNING - {message} (duration: {duration_ms:.1f}ms)")
                logger.debug(f"[HEALTH_CHECK] Circuit breaker details: search_state={search_state}, caption_state={caption_state}, search_failures={search_cb.state.consecutive_failures}, caption_failures={caption_cb.state.consecutive_failures}")

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
                logger.info(f"[HEALTH_CHECK] Circuit breaker: OK - All circuits closed (duration: {duration_ms:.1f}ms)")
                logger.debug(f"[HEALTH_CHECK] Circuit breaker details: search_state={search_state}, caption_state={caption_state}, search_total_trips={search_cb.state.total_trips}, caption_total_trips={caption_cb.state.total_trips}")

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
            logger.warning(f"[HEALTH_CHECK] Circuit breaker: WARNING - Check error: {str(e)} (duration: {duration_ms:.1f}ms)")
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
        import time
        start = time.perf_counter()

        logger.info("[HEALTH_CHECK] Running all health checks: network, disk_space, memory, embedding, ffmpeg, ytdlp, youtube_api, youtube_api_cache, llm_provider, circuit_breaker")

        results = []

        # Run all available checks
        results.append(self.check_network())
        results.append(self.check_disk_space(project_path))
        results.append(self.check_memory())
        results.append(self.check_embedding_provider())
        results.append(self.check_ffmpeg())
        results.append(self.check_ytdlp())
        results.append(self.check_youtube_api())
        # US-155-011: Add cache database integrity check
        results.append(self.check_youtube_api_cache())
        results.append(self.check_llm_provider())
        results.append(self.check_circuit_breakers())

        # Log summary
        total_duration_ms = (time.perf_counter() - start) * 1000
        ok_count = sum(1 for r in results if r.status == HealthStatus.OK)
        warning_count = sum(1 for r in results if r.status == HealthStatus.WARNING)
        failed_count = sum(1 for r in results if r.status == HealthStatus.FAILED)
        skipped_count = sum(1 for r in results if r.status == HealthStatus.SKIPPED)

        logger.info(f"[HEALTH_CHECK] All checks complete: {ok_count} OK, {warning_count} warnings, {failed_count} failed, {skipped_count} skipped (total duration: {total_duration_ms:.1f}ms)")

        # Log failed checks with actionable context
        for result in results:
            if result.status == HealthStatus.FAILED:
                logger.error(f"[HEALTH_CHECK] {result.name}: FAILED - {result.message}")
            elif result.status == HealthStatus.WARNING:
                logger.warning(f"[HEALTH_CHECK] {result.name}: WARNING - {result.message}")

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
    import time
    start_time = time.perf_counter()

    checker = HealthChecker(config)
    results = checker.check_stage(stage_name, project_path)

    # Log health check summary at INFO level
    total_duration_ms = (time.perf_counter() - start_time) * 1000

    # Count results by status
    ok_count = sum(1 for r in results if r.status == HealthStatus.OK)
    warning_count = sum(1 for r in results if r.status == HealthStatus.WARNING)
    failed_count = sum(1 for r in results if r.status == HealthStatus.FAILED)
    skipped_count = sum(1 for r in results if r.status == HealthStatus.SKIPPED)

    # Collect resource metrics for logging
    resource_metrics = {}
    for result in results:
        if result.name in ('memory', 'disk_space') and result.details:
            if result.name == 'memory' and 'used_percent' in result.details:
                resource_metrics['memory_percent'] = result.details['used_percent']
            if result.name == 'disk_space' and 'free_gb' in result.details:
                resource_metrics['disk_free_gb'] = result.details['free_gb']

    # Log summary
    logger.info(
        f"[HEALTH_CHECK] Stage '{stage_name}': "
        f"OK={ok_count}, WARNING={warning_count}, FAILED={failed_count}, SKIPPED={skipped_count} "
        f"(duration: {total_duration_ms:.1f}ms){' | ' + ', '.join(f'{k}={v}' for k, v in resource_metrics.items()) if resource_metrics else ''}"
    )

    # Log individual check results at DEBUG level
    for result in results:
        logger.debug(
            f"[HEALTH_CHECK] {result.name}: {result.status.value.upper()} - {result.message} "
            f"(duration: {result.duration_ms:.1f}ms)"
        )

    return results


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
    import time
    start_time = time.perf_counter()

    checker = HealthChecker(config)
    results = checker.run_all_checks(project_path)

    # Log health check summary at INFO level
    total_duration_ms = (time.perf_counter() - start_time) * 1000

    # Count results by status
    ok_count = sum(1 for r in results if r.status == HealthStatus.OK)
    warning_count = sum(1 for r in results if r.status == HealthStatus.WARNING)
    failed_count = sum(1 for r in results if r.status == HealthStatus.FAILED)
    skipped_count = sum(1 for r in results if r.status == HealthStatus.SKIPPED)

    # Log summary
    logger.info(
        f"[HEALTH_CHECK] All checks complete: "
        f"OK={ok_count}, WARNING={warning_count}, FAILED={failed_count}, SKIPPED={skipped_count} "
        f"(duration: {total_duration_ms:.1f}ms)"
    )

    # Log individual check results at DEBUG level
    for result in results:
        logger.debug(
            f"[HEALTH_CHECK] {result.name}: {result.status.value.upper()} - {result.message} "
            f"(duration: {result.duration_ms:.1f}ms)"
        )

    return results


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
