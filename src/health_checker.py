"""Health check module for pipeline stage validation.

US-88-005: Pre-stage health checks that validate external dependencies
before running each pipeline stage.
"""

import os
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

    # Health check timeouts per stage (stage_name -> timeout seconds)
    stage_timeouts: Dict[str, float] = field(default_factory=dict)

    # Enable/disable specific health checks
    enabled_checks: Dict[str, bool] = field(default_factory=lambda: {
        'network': True,
        'disk_space': True,
        'embedding': True,
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

        return HealthCheckConfig(
            network_timeout_seconds=hc_dict.get('network_timeout_seconds', 5.0),
            network_check_hosts=hc_dict.get('network_check_hosts', ['8.8.8.8', '1.1.1.1']),
            disk_space_warning_gb=hc_dict.get('disk_space_warning_gb', 10.0),
            disk_space_critical_gb=hc_dict.get('disk_space_critical_gb', 2.0),
            disk_space_check_paths=hc_dict.get('disk_space_check_paths', []),
            embedding_timeout_seconds=hc_dict.get('embedding_timeout_seconds', 10.0),
            embedding_check_enabled=hc_dict.get('embedding_check_enabled', True),
            stage_timeouts=hc_dict.get('stage_timeouts', {}),
            enabled_checks=hc_dict.get('enabled_checks', {
                'network': True,
                'disk_space': True,
                'embedding': True,
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

        # Disk space checks - for download stages
        if stage_name in ('DOWNLOAD_SEGMENTS',):
            results.append(self.check_disk_space(project_path))

        # Embedding checks - for matching stages
        if stage_name in ('MATCH', 'ITERATIVE_MATCH'):
            results.append(self.check_embedding_provider())

        return results

    def get_stage_timeout(self, stage_name: str) -> float:
        """Get the configured timeout for a specific stage."""
        return self.health_config.get_stage_timeout(stage_name, 30.0)


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


# Registry for stage-specific health checks
STAGE_HEALTH_CHECKS: Dict[str, List[str]] = {
    'VIDEO_SEARCH': ['network'],
    'CAPTION': ['network'],
    'MATCH': ['network', 'embedding'],
    'ITERATIVE_MATCH': ['network', 'embedding'],
    'DOWNLOAD_SEGMENTS': ['network', 'disk_space'],
    'OUTPUT': ['disk_space'],
}


def get_circuit_breaker_health(
    circuit_breaker: Optional[Any] = None,
) -> Optional[Dict[str, Any]]:
    """Get circuit breaker health metrics.

    Args:
        circuit_breaker: Optional CircuitBreaker instance to get metrics from

    Returns:
        Dict with health metrics or None if no circuit breaker provided
    """
    if circuit_breaker is None:
        return None

    try:
        # Try to get health metrics from the circuit breaker
        if hasattr(circuit_breaker, 'get_health_metrics'):
            return circuit_breaker.get_health_metrics()
        # Fallback to get_stats if get_health_metrics not available
        elif hasattr(circuit_breaker, 'get_stats'):
            stats = circuit_breaker.get_stats()
            return {
                'trip_count': stats.get('total_trips', 0),
                'current_state': 'open' if stats.get('is_open', False) else 'closed',
                'average_pause_duration': stats.get('total_paused_seconds', 0.0),
                'consecutive_failures': stats.get('consecutive_failures', 0),
                'is_tripped': stats.get('is_open', False),
                'enabled': stats.get('enabled', True),
            }
    except Exception as e:
        return {'error': str(e)}

    return None
