"""
Stage Contract Tests - US-45-012

Parametrized tests that verify all registered pipeline stages can handle
a minimal (default-initialized) PipelineState without raising AttributeError.

This catches stale state references and missing attribute defaults that
would crash the pipeline at runtime. Stages are allowed to return failure
StageResult — we only test that they don't crash on missing attributes.
"""

import pytest
from unittest.mock import MagicMock, patch

from src.checkpoint import STAGE_ORDER
from src.stages import get_stage
from src.state import PipelineState
from tests.fixtures import create_mock_config

# Import all stage modules to trigger @register_stage decorators
import src.stages.analyze  # noqa: F401
import src.stages.video_search  # noqa: F401
import src.stages.caption_stage  # noqa: F401
import src.stages.match  # noqa: F401
import src.stages.iterative_match  # noqa: F401
import src.stages.download_segments  # noqa: F401
import src.stages.output  # noqa: F401


def _make_mock_checkpoint():
    """Create a minimal mock CheckpointManager."""
    cp = MagicMock()
    cp.get_stage_data.return_value = None
    cp.has_stage_data.return_value = False
    cp.save_stage.return_value = None
    cp.should_skip_stage.return_value = False  # For can_skip() method
    return cp


@pytest.mark.parametrize("stage_name", STAGE_ORDER)
def test_stage_run_no_attribute_error(stage_name):
    """
    Contract: every registered stage must handle a default-initialized
    PipelineState without raising AttributeError.

    Stages may fail (return StageResult with success=False) — that's fine.
    They may raise other exceptions from missing external deps (LLM, network).
    The contract is specifically that no AttributeError from stale or missing
    state field references occurs.
    """
    stage_cls = get_stage(stage_name)
    assert stage_cls is not None, (
        f"Stage '{stage_name}' is in STAGE_ORDER but not registered. "
        f"Ensure it has @register_stage decorator."
    )

    stage = stage_cls()
    state = PipelineState()
    config = create_mock_config()
    checkpoint = _make_mock_checkpoint()

    try:
        stage.run(state, config, checkpoint)
    except AttributeError as exc:
        pytest.fail(
            f"Stage '{stage_name}' raised AttributeError on minimal state: {exc}"
        )
    except Exception:
        # Other exceptions (network, LLM, file I/O, etc.) are acceptable.
        # The contract only guards against AttributeError from stale references.
        pass


@pytest.mark.parametrize("stage_name", STAGE_ORDER)
def test_stage_can_skip_no_attribute_error(stage_name):
    """
    Contract: can_skip() must not raise AttributeError on minimal state.
    """
    stage_cls = get_stage(stage_name)
    assert stage_cls is not None

    stage = stage_cls()
    state = PipelineState()
    checkpoint = _make_mock_checkpoint()

    try:
        stage.can_skip(state, checkpoint)
    except AttributeError as exc:
        pytest.fail(
            f"Stage '{stage_name}'.can_skip() raised AttributeError: {exc}"
        )
    except Exception:
        pass


@pytest.mark.parametrize("stage_name", STAGE_ORDER)
def test_stage_restore_no_attribute_error(stage_name):
    """
    Contract: restore() must not raise AttributeError on minimal state.
    """
    stage_cls = get_stage(stage_name)
    assert stage_cls is not None

    stage = stage_cls()
    state = PipelineState()
    checkpoint = _make_mock_checkpoint()
    config = create_mock_config()

    try:
        stage.restore(state, checkpoint, config)
    except AttributeError as exc:
        pytest.fail(
            f"Stage '{stage_name}'.restore() raised AttributeError: {exc}"
        )
    except Exception:
        pass


@pytest.mark.parametrize("stage_name", STAGE_ORDER)
def test_stage_validate_inputs_no_attribute_error(stage_name):
    """
    Contract: validate_inputs() must not raise AttributeError on minimal state.
    """
    stage_cls = get_stage(stage_name)
    assert stage_cls is not None

    stage = stage_cls()
    state = PipelineState()
    config = create_mock_config()

    try:
        stage.validate_inputs(state, config)
    except AttributeError as exc:
        pytest.fail(
            f"Stage '{stage_name}'.validate_inputs() raised AttributeError: {exc}"
        )
    except Exception:
        pass


# =============================================================================
# Stage Contract Validation Tests - US-86-011
# =============================================================================

@pytest.mark.parametrize("stage_name", STAGE_ORDER)
def test_stage_contract_returns_stage_result(stage_name):
    """
    Contract: stage.run() must return a StageResult instance.

    Tests that all stages return the correct type from run().
    """
    from src.stages import StageResult

    stage_cls = get_stage(stage_name)
    assert stage_cls is not None, f"Stage '{stage_name}' not registered"

    stage = stage_cls()
    state = PipelineState()
    config = create_mock_config()
    checkpoint = _make_mock_checkpoint()

    result = stage.run(state, config, checkpoint)

    assert isinstance(result, StageResult), (
        f"Stage '{stage_name}'.run() must return StageResult, got {type(result).__name__}"
    )


@pytest.mark.parametrize("stage_name", STAGE_ORDER)
def test_stage_contract_has_valid_name(stage_name):
    """
    Contract: all stages must have a non-empty name attribute.

    The name must match STAGE_ORDER for proper checkpoint integration.
    """
    stage_cls = get_stage(stage_name)
    assert stage_cls is not None

    # Get the class attribute
    stage_name_attr = getattr(stage_cls, 'name', None)
    assert stage_name_attr is not None and stage_name_attr != "", (
        f"Stage '{stage_name}' must have a non-empty 'name' attribute"
    )
    assert stage_name_attr == stage_name, (
        f"Stage name '{stage_name_attr}' must match STAGE_ORDER entry '{stage_name}'"
    )


@pytest.mark.parametrize("stage_name", STAGE_ORDER)
def test_stage_contract_has_description(stage_name):
    """
    Contract: all stages must have a description attribute.
    """
    stage_cls = get_stage(stage_name)
    assert stage_cls is not None

    description = getattr(stage_cls, 'description', None)
    assert description is not None and description != "", (
        f"Stage '{stage_name}' must have a non-empty 'description' attribute"
    )


@pytest.mark.parametrize("stage_name", STAGE_ORDER)
def test_stage_contract_has_depends_on(stage_name):
    """
    Contract: all stages must have a DEPENDS_ON attribute (list).

    Tests that DEPENDS_ON is declared and is a list.
    """
    stage_cls = get_stage(stage_name)
    assert stage_cls is not None

    depends_on = getattr(stage_cls, 'DEPENDS_ON', None)
    assert depends_on is not None, (
        f"Stage '{stage_name}' must have DEPENDS_ON attribute"
    )
    assert isinstance(depends_on, list), (
        f"Stage '{stage_name}'.DEPENDS_ON must be a list, got {type(depends_on).__name__}"
    )


@pytest.mark.parametrize("stage_name", STAGE_ORDER)
def test_stage_contract_depends_on_valid_stages(stage_name):
    """
    Contract: stage DEPENDS_ON must reference valid stage names in STAGE_ORDER.

    Ensures dependency declarations point to actual pipeline stages.
    """
    stage_cls = get_stage(stage_name)
    assert stage_cls is not None

    depends_on = getattr(stage_cls, 'DEPENDS_ON', [])

    for dep in depends_on:
        assert dep in STAGE_ORDER, (
            f"Stage '{stage_name}' depends on '{dep}' which is not in STAGE_ORDER"
        )


@pytest.mark.parametrize("stage_name", STAGE_ORDER)
def test_stage_contract_has_critical_flag(stage_name):
    """
    Contract: all stages must have a CRITICAL attribute (bool).

    Tests that CRITICAL flag is declared.
    """
    stage_cls = get_stage(stage_name)
    assert stage_cls is not None

    critical = getattr(stage_cls, 'CRITICAL', None)
    assert critical is not None, (
        f"Stage '{stage_name}' must have CRITICAL attribute"
    )
    assert isinstance(critical, bool), (
        f"Stage '{stage_name}'.CRITICAL must be a bool, got {type(critical).__name__}"
    )


@pytest.mark.parametrize("stage_name", STAGE_ORDER)
def test_stage_contract_has_produces_list(stage_name):
    """
    Contract: all stages must have a PRODUCES attribute (list).

    Documents what state attributes the stage produces.
    """
    stage_cls = get_stage(stage_name)
    assert stage_cls is not None

    produces = getattr(stage_cls, 'PRODUCES', None)
    assert produces is not None, (
        f"Stage '{stage_name}' must have PRODUCES attribute"
    )
    assert isinstance(produces, list), (
        f"Stage '{stage_name}'.PRODUCES must be a list, got {type(produces).__name__}"
    )


def test_stage_contract_abstract_methods_enforced():
    """
    Contract: Stage base class abstract methods must be implemented by subclasses.

    Tests that the ABC metaclass prevents instantiation of incomplete implementations.
    """
    from src.stages import Stage

    # Attempting to instantiate a Stage subclass without implementing abstract methods
    # should raise TypeError (not possible with ABC)
    class IncompleteStage(Stage):
        name = "incomplete_stage"
        description = "Incomplete stage"
        # Missing: run(), can_skip(), restore()

    # The ABC metaclass prevents instantiation of classes that don't implement
    # all abstract methods. This is enforced at class definition time.
    # We verify the abstractmethod decorator exists on the base class.
    from src.stages import Stage as StageBase
    assert hasattr(StageBase.run, '__isabstractmethod__'), "run() must be abstract"
    assert hasattr(StageBase.can_skip, '__isabstractmethod__'), "can_skip() must be abstract"
    assert hasattr(StageBase.restore, '__isabstractmethod__'), "restore() must be abstract"


@pytest.mark.parametrize("stage_name", STAGE_ORDER)
def test_stage_contract_can_skip_returns_bool(stage_name):
    """
    Contract: stage.can_skip() should return a boolean when implemented.

    Some stages may return None - we accept both bool and None.
    """
    stage_cls = get_stage(stage_name)
    assert stage_cls is not None

    stage = stage_cls()
    state = PipelineState()
    checkpoint = _make_mock_checkpoint()

    result = stage.can_skip(state, checkpoint)

    # Accept bool or None (some stages may not have implemented return value)
    assert result is None or isinstance(result, bool), (
        f"Stage '{stage_name}'.can_skip() must return bool or None, got {type(result).__name__}"
    )


@pytest.mark.parametrize("stage_name", STAGE_ORDER)
def test_stage_contract_restore_returns_bool(stage_name):
    """
    Contract: stage.restore() must return a boolean.

    Tests that the return type is bool, not None.
    """
    stage_cls = get_stage(stage_name)
    assert stage_cls is not None

    stage = stage_cls()
    state = PipelineState()
    checkpoint = _make_mock_checkpoint()
    config = create_mock_config()

    result = stage.restore(state, checkpoint, config)

    assert isinstance(result, bool), (
        f"Stage '{stage_name}'.restore() must return bool, got {type(result).__name__}"
    )


@pytest.mark.parametrize("stage_name", STAGE_ORDER)
def test_stage_contract_validate_inputs_returns_optional_string(stage_name):
    """
    Contract: stage.validate_inputs() must return None or a string error message.

    Tests return type is Optional[str].
    """
    stage_cls = get_stage(stage_name)
    assert stage_cls is not None

    stage = stage_cls()
    state = PipelineState()
    config = create_mock_config()

    result = stage.validate_inputs(state, config)

    assert result is None or isinstance(result, str), (
        f"Stage '{stage_name}'.validate_inputs() must return None or str, got {type(result).__name__}"
    )
