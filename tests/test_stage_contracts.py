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
