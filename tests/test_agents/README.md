# test_agents - Self-Healing Pipeline Agent Tests

This directory contains tests for the self-healing pipeline agent system, which automatically recovers from common pipeline errors.

## Directory Overview

| File | Purpose |
|------|---------|
| `conftest.py` | Pytest fixtures for healer/agent tests |
| `test_base.py` | Base healer class and abstract interface tests |
| `test_checkpoint_healer.py` | Checkpoint corruption recovery tests |
| `test_disk_healer.py` | Disk space management and cleanup tests |
| `test_download_healer_retry.py` | Download retry logic and backoff tests |
| `test_fallback.py` | Fallback provider selection tests |
| `test_healers.py` | Individual healer class tests (API, Checkpoint, Download, Disk, Path) |
| `test_healing_logger.py` | Healing event logging and audit trail tests |
| `test_llm_healer.py` | LLM provider failover and recovery tests |
| `test_orchestrator.py` | HealingOrchestrator coordination tests |
| `test_otio.py` | OTIO timeline healer tests |
| `test_path_healer.py` | Path length and unicode sanitization tests |
| `test_runner.py` | ResilientRunner pipeline execution tests |
| `test_strategy.py` | HealingStrategy configuration tests |
| `test_watcher.py` | Pipeline watcher and monitoring tests |

## Key Fixtures

### MockHealingContext

The `MockHealingContext` fixture provides isolated context for testing healers without full orchestrator setup.

```python
@dataclass
class MockHealingContext:
    config: Any = None
    project_dir: Path = None
    state: Any = None
    stage_name: str = "TEST_STAGE"
    error: Exception = None
    metrics: Any = None

    def with_error(self, error: Exception) -> 'MockHealingContext':
        """Create a copy with different error."""
        ...

    def with_stage(self, stage_name: str) -> 'MockHealingContext':
        """Create a copy with different stage name."""
        ...
```

**Usage Example:**

```python
def test_healer_handles_rate_limit(mock_healing_context):
    # Create context with specific error
    ctx = mock_healing_context.with_error(Exception("Rate limit exceeded"))

    # Create healer with context's dependencies
    healer = APIHealer(ctx.config, ctx.project_dir)

    # Test fix behavior
    result = healer.fix(ctx.error, ctx.state, ctx.stage_name)
    assert result.success
    assert result.action == HealerAction.RETRY
```

### agent_factory

The `agent_factory` fixture provides a factory pattern for creating test instances of healers, orchestrators, and runners.

```python
class AgentFactory:
    def create_healer(self, healer_type: str) -> Healer:
        """Create healer by type: 'checkpoint', 'disk', 'api', etc."""
        ...

    def create_orchestrator(
        self,
        strategy: str = "conservative",
        healers: List = None
    ) -> HealingOrchestrator:
        """Create orchestrator with specified strategy."""
        ...

    def create_runner(
        self,
        with_orchestrator: bool = False,
        strategy: str = "conservative"
    ) -> ResilientRunner:
        """Create runner optionally with orchestrator."""
        ...

    def available_healers(self) -> List[str]:
        """List available healer types."""
        ...
```

**Usage Examples:**

```python
def test_healer_via_factory(agent_factory):
    # Create specific healer type
    healer = agent_factory.create_healer("checkpoint")
    assert healer.name == "checkpoint-healer"

def test_orchestrator_strategy(agent_factory):
    # Create orchestrator with specific strategy
    orchestrator = agent_factory.create_orchestrator(strategy="aggressive")
    assert orchestrator.strategy.mode == HealingMode.AGGRESSIVE

def test_runner_integration(agent_factory):
    # Create runner with attached orchestrator
    runner = agent_factory.create_runner(with_orchestrator=True)
    assert runner.orchestrator is not None
```

### mock_orchestrator_factory

Factory fixture for creating mock orchestrators with custom strategies:

```python
def test_aggressive_healing(mock_orchestrator_factory):
    orchestrator = mock_orchestrator_factory(strategy="aggressive")
    assert orchestrator.strategy.mode == HealingMode.AGGRESSIVE
```

### Other Fixtures

| Fixture | Purpose |
|---------|---------|
| `project_dir` | Temporary project directory with `.cache`, `videos`, `output` subdirs |
| `mock_config` | Mock config object with output, download, LLM settings |
| `mock_segment` | Mock voiceover segment dataclass |
| `mock_match` | Mock match result with video file |
| `mock_matches` | Multiple mock matches for batch testing |
| `mock_state` | Mock pipeline state with matches and paths |
| `mock_checkpoint` | Mock checkpoint manager |
| `mock_stage` | Mock pipeline stage returning StageResult.ok |

## Testing Healing Strategies

The system supports four healing strategies, each with different behaviors:

### Strategy Comparison

| Strategy | Attempts/Stage | Total Heals | Behavior |
|----------|----------------|-------------|----------|
| `aggressive` | 5 | 50 | Try everything, auto-fix, minimal interaction |
| `conservative` | 3 | 20 | Safe fixes only, preserve user config (default) |
| `interactive` | 3 | 20 | Ask user before major changes |
| `minimal` | 1 | 5 | Fail fast, critical fixes only |

### Testing Strategy Selection

```python
def test_aggressive_strategy(agent_factory):
    orchestrator = agent_factory.create_orchestrator(strategy="aggressive")

    # Aggressive has more attempts
    assert orchestrator.strategy.max_attempts_per_stage == 5
    assert orchestrator.strategy.max_total_heals == 50
    assert orchestrator.strategy.heal_delay == 1.0

def test_minimal_strategy(agent_factory):
    orchestrator = agent_factory.create_orchestrator(strategy="minimal")

    # Minimal has strict limits
    assert orchestrator.strategy.max_attempts_per_stage == 1
    assert orchestrator.strategy.max_total_heals == 5
    assert orchestrator.strategy.enable_rollback is False

def test_conservative_is_default():
    strategy = HealingStrategy()
    assert strategy.mode == HealingMode.CONSERVATIVE
```

### Testing Strategy-Specific Behavior

```python
def test_interactive_asks_user(agent_factory):
    orchestrator = agent_factory.create_orchestrator(strategy="interactive")

    # Interactive strategy doesn't auto-fix preflight
    assert orchestrator.strategy.auto_fix_preflight is False
```

## Mocking Pipeline Stages

When testing healers in isolation, mock the pipeline stages to control inputs:

### Basic Stage Mock

```python
from src.stages import StageResult

@pytest.fixture
def mock_stage():
    stage = Mock()
    stage.name = "TEST_STAGE"
    stage.can_skip.return_value = False
    stage.validate_inputs.return_value = None
    stage.run.return_value = StageResult.ok({})
    stage.restore.return_value = True
    return stage
```

### Stage That Fails Then Succeeds

```python
def test_healer_retries_stage(mock_healing_context, mock_stage):
    # Stage fails first, then succeeds
    mock_stage.run.side_effect = [
        ValueError("Transient error"),
        StageResult.ok({"recovered": True})
    ]

    healer = APIHealer(mock_healing_context.config, mock_healing_context.project_dir)
    result = healer.fix(ValueError("Transient error"), mock_healing_context.state, "MATCH")

    assert result.action == HealerAction.RETRY
```

### Mocking Specific Stage Errors

```python
def test_download_stage_403(mock_healing_context):
    # Create context for DOWNLOAD stage with 403 error
    ctx = mock_healing_context.with_stage("DOWNLOAD").with_error(
        Exception("HTTP Error 403: Forbidden")
    )

    healer = DownloadHealer(ctx.config, ctx.project_dir)
    assert healer.can_handle(ctx.error, ctx.stage_name)

    result = healer.fix(ctx.error, ctx.state, ctx.stage_name)
    assert result.action in [HealerAction.RETRY, HealerAction.MODIFY_CONFIG]
```

### Testing Stage Restoration

```python
def test_orchestrator_restores_stage(agent_factory, mock_stage):
    orchestrator = agent_factory.create_orchestrator()

    # Simulate checkpoint with stage data
    checkpoint_data = {"stages": {"ANALYZE": {"keywords": ["test"]}}}

    # Test restoration
    mock_stage.restore.return_value = True
    success = mock_stage.restore(checkpoint_data["stages"]["ANALYZE"])
    assert success
```

## Common Testing Patterns

### Testing Error Pattern Matching

```python
def test_healer_error_patterns(agent_factory):
    healer = agent_factory.create_healer("download")

    # Test various error patterns
    assert healer.can_handle(Exception("yt-dlp error"), "DOWNLOAD")
    assert healer.can_handle(Exception("429 Too Many Requests"), "DOWNLOAD")
    assert not healer.can_handle(Exception("OTIO serialization failed"), "OUTPUT")
```

### Testing Backoff Behavior

```python
@patch('time.sleep')
def test_rate_limit_backoff(mock_sleep, agent_factory):
    healer = agent_factory.create_healer("api")
    state = Mock()

    initial_backoff = healer.backoff_time
    healer._handle_rate_limit(Exception("rate limit"), state)

    assert healer.backoff_time > initial_backoff
    mock_sleep.assert_called()
```

### Testing Config Modification

```python
def test_healer_modifies_config(mock_healing_context):
    ctx = mock_healing_context
    ctx.config.llm.timeout = 30

    healer = APIHealer(ctx.config, ctx.project_dir)
    healer._handle_timeout(Exception("timeout"), ctx.state)

    # Config should be modified, not replaced
    assert ctx.config.llm.timeout > 30
```

### Testing Healer Result Actions

```python
def test_healer_result_actions(agent_factory):
    healer = agent_factory.create_healer("checkpoint")
    state = Mock()

    # Test different result actions
    result = healer._start_fresh(Exception("corrupted"), state)

    assert result.success
    assert result.action in [
        HealerAction.RETRY,      # Re-run stage
        HealerAction.RESTORE,    # Restore from backup
        HealerAction.SKIP,       # Skip this item
        HealerAction.ABORT,      # Stop pipeline
        HealerAction.MODIFY_CONFIG  # Change settings
    ]
```

## Running Tests

```bash
# Run all agent tests
pytest tests/test_agents/ -v

# Run specific healer tests
pytest tests/test_agents/test_healers.py -v

# Run orchestrator tests
pytest tests/test_agents/test_orchestrator.py -v

# Run with markers
pytest tests/test_agents/ -m "not slow" -v

# Run single test
pytest tests/test_agents/test_healers.py::TestAPIHealer::test_can_handle_rate_limit -v
```

## Test Markers

| Marker | Description |
|--------|-------------|
| `@pytest.mark.slow` | Tests that take longer to run |
| `@pytest.mark.integration` | Tests requiring real file system |
| `@pytest.mark.fast` | Quick unit tests |

## Related Documentation

- `src/agents/README.md` - Agent implementation details
- `CLAUDE.md` - Self-healing configuration section
- `docs/healing-system.md` - Healing system architecture
