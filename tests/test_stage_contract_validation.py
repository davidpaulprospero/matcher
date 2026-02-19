"""
Tests for stage input/output contract validation (US-108-011).

Tests:
- StageInputSchema and StageOutputSchema definitions
- register_stage_schemas function
- get_stage_schemas function
- PipelineValidator.validate_stage_io method
- ContractViolation dataclass
- ErrorCategory.CONTRACT integration with error_aggregator
"""

import pytest
from unittest.mock import MagicMock, patch

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestStageSchemas:
    """Test StageInputSchema and StageOutputSchema classes."""

    def test_stage_input_schema_creation(self):
        """Test creating a StageInputSchema."""
        from src.stages import StageInputSchema, FieldSchema

        schema = StageInputSchema(
            stage_name="TEST_STAGE",
            fields=[
                FieldSchema(
                    field_name="test_field",
                    expected_type=dict,
                    required=True,
                    description="A test field",
                ),
            ],
        )

        assert schema.stage_name == "TEST_STAGE"
        assert len(schema.fields) == 1
        assert schema.fields[0].field_name == "test_field"

    def test_stage_output_schema_creation(self):
        """Test creating a StageOutputSchema."""
        from src.stages import StageOutputSchema, FieldSchema

        schema = StageOutputSchema(
            stage_name="TEST_STAGE",
            fields=[
                FieldSchema(
                    field_name="output_field",
                    expected_type=list,
                    required=True,
                    description="An output field",
                ),
            ],
        )

        assert schema.stage_name == "TEST_STAGE"
        assert len(schema.fields) == 1

    def test_get_required_fields(self):
        """Test getting required fields from schema."""
        from src.stages import StageInputSchema, FieldSchema

        schema = StageInputSchema(
            stage_name="TEST_STAGE",
            fields=[
                FieldSchema(field_name="required_field", expected_type=dict, required=True),
                FieldSchema(field_name="optional_field", expected_type=list, required=False),
            ],
        )

        required = schema.get_required_fields()
        assert "required_field" in required
        assert "optional_field" not in required

    def test_get_all_fields(self):
        """Test getting all fields from schema."""
        from src.stages import StageInputSchema, FieldSchema

        schema = StageInputSchema(
            stage_name="TEST_STAGE",
            fields=[
                FieldSchema(field_name="required_field", expected_type=dict, required=True),
                FieldSchema(field_name="optional_field", expected_type=list, required=False),
            ],
        )

        all_fields = schema.get_all_fields()
        assert "required_field" in all_fields
        assert "optional_field" in all_fields


class TestSchemaRegistration:
    """Test stage schema registration functions."""

    def test_register_stage_schemas(self):
        """Test registering stage schemas."""
        from src.stages import (
            register_stage_schemas,
            get_stage_schemas,
            StageInputSchema,
            StageOutputSchema,
            FieldSchema,
        )

        # Clear any existing schemas for this test
        input_schema = StageInputSchema(
            stage_name="REG_TEST",
            fields=[
                FieldSchema(field_name="input1", expected_type=dict, required=True),
            ],
        )
        output_schema = StageOutputSchema(
            stage_name="REG_TEST",
            fields=[
                FieldSchema(field_name="output1", expected_type=list, required=True),
            ],
        )

        register_stage_schemas(input_schema=input_schema, output_schema=output_schema)

        # Get the schemas back
        schemas = get_stage_schemas("REG_TEST")
        assert "input" in schemas
        assert "output" in schemas
        assert schemas["input"].stage_name == "REG_TEST"
        assert schemas["output"].stage_name == "REG_TEST"

    def test_get_stage_schemas_not_found(self):
        """Test getting schemas for unregistered stage."""
        from src.stages import get_stage_schemas

        schemas = get_stage_schemas("NONEXISTENT_STAGE")
        assert schemas == {}

    def test_register_input_only(self):
        """Test registering only input schema."""
        from src.stages import (
            register_stage_schemas,
            get_stage_schemas,
            StageInputSchema,
            FieldSchema,
        )

        input_schema = StageInputSchema(
            stage_name="INPUT_ONLY",
            fields=[
                FieldSchema(field_name="input1", expected_type=dict, required=True),
            ],
        )

        register_stage_schemas(input_schema=input_schema)

        schemas = get_stage_schemas("INPUT_ONLY")
        assert "input" in schemas
        assert "output" not in schemas


class TestContractViolation:
    """Test ContractViolation dataclass."""

    def test_contract_violation_creation(self):
        """Test creating a ContractViolation."""
        from src.pipeline_validator import ContractViolation

        violation = ContractViolation(
            stage_name="TEST_STAGE",
            violation_type="input",
            field_name="test_field",
            expected="required field of type dict",
            actual="field not present in checkpoint",
            severity="error",
        )

        assert violation.stage_name == "TEST_STAGE"
        assert violation.violation_type == "input"
        assert violation.field_name == "test_field"
        assert violation.severity == "error"


class TestPipelineValidatorIO:
    """Test PipelineValidator.validate_stage_io method."""

    def test_validate_missing_input(self):
        """Test validation detects missing required input."""
        from src.pipeline_validator import PipelineValidator, ContractViolation
        from src.stages import (
            register_stage_schemas,
            StageInputSchema,
            FieldSchema,
        )

        # Register schema for test stage
        input_schema = StageInputSchema(
            stage_name="VALIDATE_TEST",
            fields=[
                FieldSchema(field_name="required_input", expected_type=dict, required=True),
            ],
        )
        register_stage_schemas(input_schema=input_schema)

        # Create validator with mock stages
        mock_config = MagicMock()
        mock_stage = MagicMock()
        mock_stage.name = "VALIDATE_TEST"

        validator = PipelineValidator(mock_config, [mock_stage])

        # Validate against empty checkpoint data
        checkpoint_data = {}
        violations = validator.validate_stage_io("VALIDATE_TEST", checkpoint_data, "input")

        assert len(violations) == 1
        assert violations[0].field_name == "required_input"
        assert violations[0].violation_type == "input"

    def test_validate_present_input(self):
        """Test validation passes when required input is present."""
        from src.pipeline_validator import PipelineValidator
        from src.stages import (
            register_stage_schemas,
            StageInputSchema,
            FieldSchema,
        )

        # Register schema for test stage
        input_schema = StageInputSchema(
            stage_name="VALIDATE_TEST_OK",
            fields=[
                FieldSchema(field_name="required_input", expected_type=dict, required=True),
            ],
        )
        register_stage_schemas(input_schema=input_schema)

        mock_config = MagicMock()
        mock_stage = MagicMock()
        mock_stage.name = "VALIDATE_TEST_OK"

        validator = PipelineValidator(mock_config, [mock_stage])

        # Validate against checkpoint with required field present
        checkpoint_data = {"required_input": {"key": "value"}}
        violations = validator.validate_stage_io("VALIDATE_TEST_OK", checkpoint_data, "input")

        assert len(violations) == 0

    def test_validate_wrong_type(self):
        """Test validation detects wrong type."""
        from src.pipeline_validator import PipelineValidator
        from src.stages import (
            register_stage_schemas,
            StageInputSchema,
            FieldSchema,
        )

        # Register schema expecting dict
        input_schema = StageInputSchema(
            stage_name="TYPE_TEST",
            fields=[
                FieldSchema(field_name="typed_field", expected_type="dict", required=True),
            ],
        )
        register_stage_schemas(input_schema=input_schema)

        mock_config = MagicMock()
        mock_stage = MagicMock()
        mock_stage.name = "TYPE_TEST"

        validator = PipelineValidator(mock_config, [mock_stage])

        # Validate with wrong type (list instead of dict)
        checkpoint_data = {"typed_field": ["item1", "item2"]}
        violations = validator.validate_stage_io("TYPE_TEST", checkpoint_data, "input")

        assert len(violations) == 1
        assert "type" in violations[0].expected.lower()

    def test_validate_optional_field_missing(self):
        """Test validation passes for missing optional field."""
        from src.pipeline_validator import PipelineValidator
        from src.stages import (
            register_stage_schemas,
            StageInputSchema,
            FieldSchema,
        )

        # Register schema with optional field
        input_schema = StageInputSchema(
            stage_name="OPTIONAL_TEST",
            fields=[
                FieldSchema(field_name="optional_field", expected_type=list, required=False),
            ],
        )
        register_stage_schemas(input_schema=input_schema)

        mock_config = MagicMock()
        mock_stage = MagicMock()
        mock_stage.name = "OPTIONAL_TEST"

        validator = PipelineValidator(mock_config, [mock_stage])

        # Validate with missing optional field
        checkpoint_data = {}
        violations = validator.validate_stage_io("OPTIONAL_TEST", checkpoint_data, "input")

        assert len(violations) == 0

    def test_validate_output(self):
        """Test output validation."""
        from src.pipeline_validator import PipelineValidator
        from src.stages import (
            register_stage_schemas,
            StageOutputSchema,
            FieldSchema,
        )

        # Register output schema
        output_schema = StageOutputSchema(
            stage_name="OUTPUT_TEST",
            fields=[
                FieldSchema(field_name="output_field", expected_type=list, required=True),
            ],
        )
        register_stage_schemas(output_schema=output_schema)

        mock_config = MagicMock()
        mock_stage = MagicMock()
        mock_stage.name = "OUTPUT_TEST"

        validator = PipelineValidator(mock_config, [mock_stage])

        # Validate with missing output
        checkpoint_data = {}
        violations = validator.validate_stage_io("OUTPUT_TEST", checkpoint_data, "output")

        assert len(violations) == 1
        assert violations[0].violation_type == "output"

    def test_validate_both_input_and_output(self):
        """Test validating both input and output."""
        from src.pipeline_validator import PipelineValidator
        from src.stages import (
            register_stage_schemas,
            StageInputSchema,
            StageOutputSchema,
            FieldSchema,
        )

        # Register both schemas
        input_schema = StageInputSchema(
            stage_name="BOTH_TEST",
            fields=[
                FieldSchema(field_name="input_field", expected_type=dict, required=True),
            ],
        )
        output_schema = StageOutputSchema(
            stage_name="BOTH_TEST",
            fields=[
                FieldSchema(field_name="output_field", expected_type=list, required=True),
            ],
        )
        register_stage_schemas(input_schema=input_schema, output_schema=output_schema)

        mock_config = MagicMock()
        mock_stage = MagicMock()
        mock_stage.name = "BOTH_TEST"

        validator = PipelineValidator(mock_config, [mock_stage])

        # Validate with only input present
        checkpoint_data = {"input_field": {"key": "value"}}
        violations = validator.validate_stage_io("BOTH_TEST", checkpoint_data, "both")

        assert len(violations) == 1
        assert violations[0].violation_type == "output"


class TestErrorAggregatorContract:
    """Test error_aggregator integration with contract violations."""

    def test_error_category_contract(self):
        """Test ErrorCategory.CONTRACT exists."""
        from src.stages.error_aggregator import ErrorCategory

        assert hasattr(ErrorCategory, 'CONTRACT')
        assert ErrorCategory.CONTRACT.value == 'contract'

    def test_record_contract_violation(self):
        """Test recording contract violations."""
        from src.stages.error_aggregator import ErrorAggregator, ErrorCategory

        agg = ErrorAggregator()
        agg.record("Test contract violation", ErrorCategory.CONTRACT)

        assert agg.total_errors == 1
        assert ErrorCategory.CONTRACT in agg.categories

    def test_contract_suggestions(self):
        """Test contract violation suggestions exist."""
        from src.stages.error_aggregator import ERROR_SUGGESTIONS

        assert 'contract' in ERROR_SUGGESTIONS
        assert 'contract.input' in ERROR_SUGGESTIONS
        assert 'contract.output' in ERROR_SUGGESTIONS


class TestCLIValidateOnly:
    """Test --validate-only CLI flag."""

    def test_validate_only_flag_exists(self):
        """Test that --validate-only flag is defined in args."""
        from src.cli.args import parse_arguments
        import sys

        # Save original argv
        original_argv = sys.argv

        try:
            sys.argv = ['main.py', '--validate-only']
            args = parse_arguments()
            assert hasattr(args, 'validate_only')
            assert args.validate_only is True
        finally:
            sys.argv = original_argv


class TestSchemaValidationAtRegistration:
    """Test schema validation at registration time (US-125-004)."""

    def test_validate_duplicate_input_fields(self):
        """Test validation detects duplicate field names in input schema."""
        from src.stages import validate_stage_schema, StageInputSchema, FieldSchema

        # Create schema with duplicate field names
        input_schema = StageInputSchema(
            stage_name="DUPE_TEST",
            fields=[
                FieldSchema(field_name="same_field", expected_type=dict, required=True),
                FieldSchema(field_name="same_field", expected_type=list, required=False),
            ],
        )

        errors = validate_stage_schema(input_schema=input_schema)

        assert len(errors) == 1
        assert "duplicate" in errors[0].lower()
        assert "same_field" in errors[0]

    def test_validate_duplicate_output_fields(self):
        """Test validation detects duplicate field names in output schema."""
        from src.stages import validate_stage_schema, StageOutputSchema, FieldSchema

        output_schema = StageOutputSchema(
            stage_name="DUPE_OUTPUT_TEST",
            fields=[
                FieldSchema(field_name="matches", expected_type=list, required=True),
                FieldSchema(field_name="matches", expected_type=dict, required=False),
            ],
        )

        errors = validate_stage_schema(output_schema=output_schema)

        assert len(errors) == 1
        assert "duplicate" in errors[0].lower()

    def test_validate_invalid_output_field(self):
        """Test validation detects output fields not in PipelineState."""
        from src.stages import validate_stage_schema, StageOutputSchema, FieldSchema

        output_schema = StageOutputSchema(
            stage_name="INVALID_OUTPUT",
            fields=[
                FieldSchema(field_name="invalid_field_name", expected_type=list, required=True),
            ],
        )

        errors = validate_stage_schema(output_schema=output_schema)

        assert len(errors) == 1
        assert "not a valid PipelineState attribute" in errors[0]
        assert "invalid_field_name" in errors[0]

    def test_validate_valid_output_fields(self):
        """Test validation passes for valid PipelineState output fields."""
        from src.stages import validate_stage_schema, StageOutputSchema, FieldSchema

        # Valid PipelineState attributes
        output_schema = StageOutputSchema(
            stage_name="VALID_OUTPUT",
            fields=[
                FieldSchema(field_name="matches", expected_type=list, required=True),
                FieldSchema(field_name="video_ids", expected_type=list, required=True),
            ],
        )

        errors = validate_stage_schema(output_schema=output_schema)

        assert len(errors) == 0

    def test_validate_invalid_identifier(self):
        """Test validation detects invalid identifier field names."""
        from src.stages import validate_stage_schema, StageInputSchema, FieldSchema

        input_schema = StageInputSchema(
            stage_name="INVALID_ID",
            fields=[
                FieldSchema(field_name="not-valid", expected_type=dict, required=True),
            ],
        )

        errors = validate_stage_schema(input_schema=input_schema)

        assert len(errors) == 1
        assert "not a valid identifier" in errors[0]

    def test_validate_both_input_and_output_errors(self):
        """Test validation detects errors in both input and output."""
        from src.stages import validate_stage_schema, StageInputSchema, StageOutputSchema, FieldSchema

        input_schema = StageInputSchema(
            stage_name="BOTH_ERRORS",
            fields=[
                FieldSchema(field_name="dup_field", expected_type=dict, required=True),
                FieldSchema(field_name="dup_field", expected_type=list, required=False),
            ],
        )

        output_schema = StageOutputSchema(
            stage_name="BOTH_ERRORS",
            fields=[
                FieldSchema(field_name="not_valid_attr", expected_type=list, required=True),
            ],
        )

        errors = validate_stage_schema(input_schema=input_schema, output_schema=output_schema)

        assert len(errors) == 2

    def test_check_orphan_stages(self):
        """Test orphan stage detection."""
        from src.stages import check_orphan_stages

        # This should return warnings for stages without schemas
        warnings = check_orphan_stages()

        # Should have warnings since no schemas are registered yet
        assert isinstance(warnings, list)

    def test_valid_schema_registration_no_errors(self):
        """Test that valid schema registration produces no validation errors."""
        from src.stages import (
            validate_stage_schema,
            StageInputSchema,
            StageOutputSchema,
            FieldSchema,
            register_stage_schemas,
            _STAGE_SCHEMAS,
        )

        # Clear schemas for this test
        _STAGE_SCHEMAS.clear()

        input_schema = StageInputSchema(
            stage_name="VALID_SCHEMA_TEST",
            fields=[
                FieldSchema(field_name="voiceover_segments", expected_type=list, required=True),
            ],
        )

        output_schema = StageOutputSchema(
            stage_name="VALID_SCHEMA_TEST",
            fields=[
                FieldSchema(field_name="matches", expected_type=list, required=True),
            ],
        )

        # Validate first
        errors = validate_stage_schema(input_schema=input_schema, output_schema=output_schema)

        # Should have no errors
        assert len(errors) == 0

        # Register should also succeed (validation runs inside)
        register_stage_schemas(input_schema=input_schema, output_schema=output_schema)

        # Verify schemas were registered
        assert "VALID_SCHEMA_TEST" in _STAGE_SCHEMAS


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
