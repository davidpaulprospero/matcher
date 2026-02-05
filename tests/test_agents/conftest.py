"""
Pytest fixtures for agent tests.
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, MagicMock
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any


@pytest.fixture
def project_dir(tmp_path) -> Path:
    """Create a temporary project directory."""
    project = tmp_path / "test_project"
    project.mkdir()
    (project / ".cache").mkdir()
    (project / "videos").mkdir()
    (project / "output").mkdir()
    return project


@pytest.fixture
def mock_config():
    """Create a mock config object."""
    config = Mock()

    # Output config
    config.output = Mock()
    config.output.gap_mode = "scale"
    config.output.frame_rate = 30.0
    config.output.include_alternatives = True
    config.output.include_strategy_tracks = True
    config.output.include_entity_images = True
    config.output.include_entity_videos = True
    config.output.export_edl = True
    config.output.export_xml = True
    config.output.num_alternatives = 2
    config.output.align_to_voiceover = True

    # Download config
    config.download = Mock()
    config.download.format = "bestvideo+bestaudio/best"
    config.download.socket_timeout = 30
    config.download.continue_dl = False
    config.download.root_dir = None

    # Cookie rotation config - disabled by default to prevent CookieRotator initialization
    config.download.cookie_rotation = Mock()
    config.download.cookie_rotation.enabled = False
    config.download.cookie_rotation.cookie_files = []
    config.download.cookie_rotation.rotate_on_errors = ["429", "rate limit"]
    config.download.cookie_rotation.rotation_strategy = "on_error"
    config.download.cookie_rotation.cooldown_seconds = 300
    config.download.cookie_rotation.max_rotations_per_session = 0

    # Caption-first config
    config.download.caption_first = Mock()
    config.download.caption_first.negative_cache_ttl_seconds = 3600
    config.download.caption_first.negative_cache_ttl_hours = 1.0

    # VPN config - disabled by default
    config.download.vpn = Mock()
    config.download.vpn.enabled = False

    # LLM config
    config.llm = Mock()
    config.llm.provider = "gemini"
    config.llm.default_provider = "gemini"
    config.llm.timeout = 30

    # Matching config
    config.matching = Mock()

    return config


@dataclass
class MockSegment:
    """Mock voiceover segment."""
    start: float = 0.0
    end: float = 5.0
    text: str = "Test segment"
    duration: float = 5.0
    metadata: Dict = field(default_factory=dict)


@dataclass
class MockMatch:
    """Mock match result."""
    segment: MockSegment = field(default_factory=MockSegment)
    video_path: str = ""
    video_id: str = "test_video_123"
    source_id: str = "test_video_123"
    start_time: float = 0.0
    end_time: float = 5.0
    confidence: float = 0.8
    time_scalar: float = 1.0
    metadata: Dict = field(default_factory=dict)


@pytest.fixture
def mock_segment():
    """Create a mock segment."""
    return MockSegment()


@pytest.fixture
def mock_match(tmp_path):
    """Create a mock match with existing video file."""
    video_file = tmp_path / "videos" / "test_video_123.mp4"
    video_file.parent.mkdir(parents=True, exist_ok=True)
    video_file.write_bytes(b"fake video content")

    return MockMatch(
        video_path=str(video_file),
        segment=MockSegment(start=0.0, end=5.0)
    )


@pytest.fixture
def mock_matches(tmp_path):
    """Create multiple mock matches."""
    videos_dir = tmp_path / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)

    matches = []
    for i in range(5):
        video_file = videos_dir / f"test_video_{i}.mp4"
        video_file.write_bytes(b"fake video content")

        match = MockMatch(
            video_path=str(video_file),
            video_id=f"test_video_{i}",
            source_id=f"test_video_{i}",
            segment=MockSegment(
                start=i * 10.0,
                end=(i + 1) * 10.0 - 1.0,
                duration=9.0
            ),
            start_time=0.0,
            end_time=9.0
        )
        matches.append(match)

    return matches


@pytest.fixture
def mock_state(mock_matches):
    """Create a mock pipeline state."""
    state = Mock()
    state.matches = mock_matches
    state.voiceover_path = "/path/to/voiceover.srt"
    state.entity_images = {}
    state.entity_videos = {}
    state.downloads = []
    state.stage_timings = {}
    return state


@pytest.fixture
def mock_checkpoint():
    """Create a mock checkpoint manager."""
    checkpoint = Mock()
    checkpoint.exists.return_value = False
    checkpoint.load.return_value = None
    checkpoint.save.return_value = True
    return checkpoint


@pytest.fixture
def mock_stage():
    """Create a mock pipeline stage."""
    from src.stages import StageResult

    stage = Mock()
    stage.name = "TEST_STAGE"
    stage.can_skip.return_value = False
    stage.validate_inputs.return_value = None
    stage.run.return_value = StageResult.ok({})
    stage.restore.return_value = True
    return stage


# =============================================================================
# Healer-specific fixtures (AC1, AC2, AC3)
# =============================================================================

@dataclass
class MockHealingContext:
    """
    Isolated context for healer tests.

    Provides all dependencies needed to test healer behavior without
    instantiating real orchestrators or runners.
    """
    config: Any = None
    project_dir: Path = None
    state: Any = None
    stage_name: str = "TEST_STAGE"
    error: Exception = None
    metrics: Any = None

    def with_error(self, error: Exception) -> 'MockHealingContext':
        """Create a copy with different error."""
        return MockHealingContext(
            config=self.config,
            project_dir=self.project_dir,
            state=self.state,
            stage_name=self.stage_name,
            error=error,
            metrics=self.metrics
        )

    def with_stage(self, stage_name: str) -> 'MockHealingContext':
        """Create a copy with different stage name."""
        return MockHealingContext(
            config=self.config,
            project_dir=self.project_dir,
            state=self.state,
            stage_name=stage_name,
            error=self.error,
            metrics=self.metrics
        )


@pytest.fixture
def mock_healing_context(mock_config, project_dir, mock_state):
    """
    Create an isolated healing context for healer tests.

    This fixture provides everything needed to test healer behavior
    without requiring a full orchestrator or runner setup.

    Usage:
        def test_healer_handles_error(mock_healing_context):
            ctx = mock_healing_context.with_error(ValueError("test"))
            healer = MyHealer(ctx.config, ctx.project_dir)
            result = healer.fix(ctx.error, ctx.state, ctx.stage_name)
    """
    from src.agents.strategy import HealingMetrics

    return MockHealingContext(
        config=mock_config,
        project_dir=project_dir,
        state=mock_state,
        stage_name="TEST_STAGE",
        error=Exception("Test error"),
        metrics=HealingMetrics()
    )


@pytest.fixture
def mock_orchestrator(mock_config, project_dir):
    """
    Create a mock HealingOrchestrator for runner integration tests.

    This fixture creates a real HealingOrchestrator but with controlled
    healer behavior, useful for testing runner-orchestrator interaction.

    Usage:
        def test_runner_with_orchestrator(mock_orchestrator, mock_config, project_dir):
            runner = ResilientRunner(mock_config, project_dir, orchestrator=mock_orchestrator)
            assert runner.orchestrator == mock_orchestrator
    """
    from src.agents.orchestrator import HealingOrchestrator
    from src.agents.strategy import HealingStrategy

    orchestrator = HealingOrchestrator(
        mock_config,
        project_dir,
        strategy=HealingStrategy.conservative()
    )
    return orchestrator


@pytest.fixture
def mock_orchestrator_factory(mock_config, project_dir):
    """
    Factory fixture for creating mock orchestrators with custom strategies.

    Usage:
        def test_aggressive_healing(mock_orchestrator_factory):
            orchestrator = mock_orchestrator_factory(strategy="aggressive")
            assert orchestrator.strategy.mode == HealingMode.AGGRESSIVE
    """
    from src.agents.orchestrator import HealingOrchestrator
    from src.agents.strategy import HealingStrategy

    def _create_orchestrator(strategy: str = "conservative", **kwargs):
        if strategy == "aggressive":
            strat = HealingStrategy.aggressive()
        elif strategy == "minimal":
            strat = HealingStrategy.minimal()
        elif strategy == "interactive":
            strat = HealingStrategy.interactive()
        else:
            strat = HealingStrategy.conservative()

        return HealingOrchestrator(mock_config, project_dir, strategy=strat, **kwargs)

    return _create_orchestrator


# =============================================================================
# Agent factory fixture (AC4)
# =============================================================================

@pytest.fixture
def agent_factory(mock_config, project_dir):
    """
    Factory fixture for creating agent components with sensible defaults.

    Follows the factory pattern from tests/conftest.py (US-006).
    Allows creating healers, orchestrators, and runners with minimal boilerplate.

    Usage:
        def test_healer(agent_factory):
            healer = agent_factory.create_healer("checkpoint")
            assert healer.name == "checkpoint-healer"

        def test_runner(agent_factory):
            runner = agent_factory.create_runner(with_orchestrator=True)
            assert runner.orchestrator is not None
    """
    from src.agents.base import Healer
    from src.agents.healers import HEALER_REGISTRY
    from src.agents.orchestrator import HealingOrchestrator
    from src.agents.runner import ResilientRunner
    from src.agents.strategy import HealingStrategy

    class AgentFactory:
        """Factory for creating agent test instances."""

        def __init__(self, config, proj_dir):
            self.config = config
            self.project_dir = proj_dir
            self._healer_map = {cls.name.replace("-healer", ""): cls for cls in HEALER_REGISTRY}

        def create_healer(self, healer_type: str) -> Healer:
            """
            Create a healer instance by type.

            Args:
                healer_type: Short name like "checkpoint", "disk", "api", etc.

            Returns:
                Instantiated healer
            """
            # Normalize the name
            normalized = healer_type.lower().replace("-healer", "").replace("_healer", "")

            if normalized in self._healer_map:
                return self._healer_map[normalized](self.config, self.project_dir)

            # Try to find by prefix match
            for name, cls in self._healer_map.items():
                if name.startswith(normalized):
                    return cls(self.config, self.project_dir)

            raise ValueError(f"Unknown healer type: {healer_type}. Available: {list(self._healer_map.keys())}")

        def create_orchestrator(
            self,
            strategy: str = "conservative",
            healers: List = None
        ) -> HealingOrchestrator:
            """
            Create an orchestrator with specified strategy.

            Args:
                strategy: "conservative", "aggressive", "minimal", or "interactive"
                healers: Optional list of healer classes

            Returns:
                HealingOrchestrator instance
            """
            strat_map = {
                "conservative": HealingStrategy.conservative,
                "aggressive": HealingStrategy.aggressive,
                "minimal": HealingStrategy.minimal,
                "interactive": HealingStrategy.interactive,
            }
            strat = strat_map.get(strategy, HealingStrategy.conservative)()
            return HealingOrchestrator(self.config, self.project_dir, strategy=strat, healers=healers)

        def create_runner(
            self,
            with_orchestrator: bool = False,
            strategy: str = "conservative"
        ) -> ResilientRunner:
            """
            Create a runner optionally with orchestrator.

            Args:
                with_orchestrator: Whether to attach an orchestrator
                strategy: Strategy for orchestrator if created

            Returns:
                ResilientRunner instance
            """
            orchestrator = self.create_orchestrator(strategy) if with_orchestrator else None
            return ResilientRunner(self.config, self.project_dir, orchestrator=orchestrator)

        def available_healers(self) -> List[str]:
            """List available healer types."""
            return list(self._healer_map.keys())

    return AgentFactory(mock_config, project_dir)
