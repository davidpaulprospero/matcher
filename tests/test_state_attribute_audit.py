"""
Audit Tests: PipelineState Attribute Access in Stage Files (US-47-003)

Uses AST parsing to scan all files in src/stages/ and verify every
attribute accessed on `state` (the PipelineState parameter) actually
exists on the PipelineState dataclass. Catches:
  - Direct access: state.field_name
  - Defensive access: getattr(state, 'field_name', default)
  - Existence check: hasattr(state, 'field_name')

Prevents the class of bug where a field is removed from PipelineState
but stage code still references it (stale state references).
"""

import ast
from dataclasses import fields as dataclass_fields
from pathlib import Path

import pytest

from src.state import PipelineState


# ============================================================================
# Helpers
# ============================================================================

STAGES_DIR = Path(__file__).parent.parent / "src" / "stages"

# Known legacy backward-compat references that are guarded by hasattr() checks.
# These access fields that don't exist on PipelineState but are safely handled.
# When legacy code is cleaned up, remove entries here — the test will then
# enforce that no new references to these fields are added.
# Format: {(filename, field_name): "reason for exception"}
KNOWN_LEGACY_EXCEPTIONS = {
    ("caption_stage.py", "downloaded_videos"): "hasattr-guarded legacy backward compat for pre-caption-first pipeline",
    ("caption_stage.py", "downloaded_audio"): "hasattr-guarded legacy backward compat for pre-caption-first pipeline",
    ("caption_stage.py", "pending_streams"): "dynamic attribute set during streaming caption fetch",
    ("iterative_match.py", "downloaded_videos"): "hasattr-guarded legacy backward compat for video ID extraction",
    ("caption_stage.py", "project_dir"): "getattr-guarded optional project_dir lookup with None fallback",
}


def _is_known_exception(filepath: Path, field_name: str) -> bool:
    """Check if a stale reference is a known legacy exception."""
    return (filepath.name, field_name) in KNOWN_LEGACY_EXCEPTIONS


def _get_pipeline_state_attrs() -> set:
    """Get all valid attribute names on PipelineState (fields + methods + properties)."""
    # Dataclass fields
    field_names = {f.name for f in dataclass_fields(PipelineState)}

    # Methods and properties (non-dunder)
    method_names = set()
    for name in dir(PipelineState):
        if not name.startswith("_"):
            method_names.add(name)

    return field_names | method_names


def _get_stage_files() -> list:
    """Get all .py files in src/stages/."""
    return sorted(STAGES_DIR.glob("*.py"))


class _StateAccessCollector(ast.NodeVisitor):
    """AST visitor that collects all attribute accesses on a variable named 'state'."""

    def __init__(self):
        self.direct_accesses = []   # (line, col, field_name)
        self.getattr_accesses = []  # (line, col, field_name)
        self.hasattr_accesses = []  # (line, col, field_name)

    def visit_Attribute(self, node: ast.Attribute):
        """Catch state.field_name patterns."""
        if isinstance(node.value, ast.Name) and node.value.id == "state":
            self.direct_accesses.append((node.lineno, node.col_offset, node.attr))
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        """Catch getattr(state, 'field', ...) and hasattr(state, 'field') patterns."""
        if isinstance(node.func, ast.Name) and node.func.id in ("getattr", "hasattr"):
            if (
                len(node.args) >= 2
                and isinstance(node.args[0], ast.Name)
                and node.args[0].id == "state"
                and isinstance(node.args[1], ast.Constant)
                and isinstance(node.args[1].value, str)
            ):
                field_name = node.args[1].value
                if node.func.id == "getattr":
                    self.getattr_accesses.append(
                        (node.lineno, node.col_offset, field_name)
                    )
                else:
                    self.hasattr_accesses.append(
                        (node.lineno, node.col_offset, field_name)
                    )
        self.generic_visit(node)


def _collect_state_accesses(filepath: Path) -> _StateAccessCollector:
    """Parse a Python file and collect all state attribute accesses."""
    source = filepath.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(filepath))
    collector = _StateAccessCollector()
    collector.visit(tree)
    return collector


def _format_violations(violations: list, filepath: Path) -> str:
    """Format violation list into readable error message."""
    rel = filepath.relative_to(STAGES_DIR.parent.parent)
    lines = [f"\nStale PipelineState references found in {rel}:"]
    for lineno, col, field_name in violations:
        lines.append(f"  line {lineno}, col {col}: state.{field_name}")
    return "\n".join(lines)


# ============================================================================
# Tests
# ============================================================================

VALID_ATTRS = _get_pipeline_state_attrs()


class TestDirectStateAccess:
    """AC1+AC2: Validate state.FIELD_NAME accesses in all stage files."""

    @pytest.fixture(scope="class")
    def all_violations(self):
        """Scan all stage files and collect direct access violations."""
        all_violations = {}
        for filepath in _get_stage_files():
            collector = _collect_state_accesses(filepath)
            violations = [
                (lineno, col, field_name)
                for lineno, col, field_name in collector.direct_accesses
                if field_name not in VALID_ATTRS
                and not _is_known_exception(filepath, field_name)
            ]
            if violations:
                all_violations[filepath] = violations
        return all_violations

    def test_no_stale_direct_state_access(self, all_violations):
        """Every state.FIELD_NAME in src/stages/*.py must exist on PipelineState."""
        if all_violations:
            messages = []
            for filepath, violations in all_violations.items():
                messages.append(_format_violations(violations, filepath))
            pytest.fail(
                f"Found stale PipelineState attribute accesses "
                f"({sum(len(v) for v in all_violations.values())} total):"
                + "\n".join(messages)
                + f"\n\nValid PipelineState fields: {sorted(VALID_ATTRS)}"
            )


class TestGetAttrStateAccess:
    """AC3: Validate getattr(state, 'field', default) patterns."""

    @pytest.fixture(scope="class")
    def all_violations(self):
        """Scan all stage files and collect getattr violations."""
        all_violations = {}
        for filepath in _get_stage_files():
            collector = _collect_state_accesses(filepath)
            violations = [
                (lineno, col, field_name)
                for lineno, col, field_name in collector.getattr_accesses
                if field_name not in VALID_ATTRS
                and not _is_known_exception(filepath, field_name)
            ]
            if violations:
                all_violations[filepath] = violations
        return all_violations

    def test_no_stale_getattr_state_access(self, all_violations):
        """Every getattr(state, 'field', ...) in src/stages/*.py must reference a valid field."""
        if all_violations:
            messages = []
            for filepath, violations in all_violations.items():
                messages.append(_format_violations(violations, filepath))
            pytest.fail(
                f"Found stale getattr(state, ...) references "
                f"({sum(len(v) for v in all_violations.values())} total):"
                + "\n".join(messages)
                + f"\n\nValid PipelineState fields: {sorted(VALID_ATTRS)}"
            )


class TestHasAttrStateAccess:
    """AC4: Validate hasattr(state, 'field') patterns."""

    @pytest.fixture(scope="class")
    def all_violations(self):
        """Scan all stage files and collect hasattr violations."""
        all_violations = {}
        for filepath in _get_stage_files():
            collector = _collect_state_accesses(filepath)
            violations = [
                (lineno, col, field_name)
                for lineno, col, field_name in collector.hasattr_accesses
                if field_name not in VALID_ATTRS
                and not _is_known_exception(filepath, field_name)
            ]
            if violations:
                all_violations[filepath] = violations
        return all_violations

    def test_no_stale_hasattr_state_access(self, all_violations):
        """Every hasattr(state, 'field') in src/stages/*.py must reference a valid field."""
        if all_violations:
            messages = []
            for filepath, violations in all_violations.items():
                messages.append(_format_violations(violations, filepath))
            pytest.fail(
                f"Found stale hasattr(state, ...) references "
                f"({sum(len(v) for v in all_violations.values())} total):"
                + "\n".join(messages)
                + f"\n\nValid PipelineState fields: {sorted(VALID_ATTRS)}"
            )


class TestCombinedAudit:
    """AC5: Combined test with clear file/line/field reporting."""

    def test_full_audit_all_patterns(self):
        """
        Comprehensive audit: scan all stage files for ALL state access patterns
        (direct, getattr, hasattr) and report every stale reference with
        file path, line number, and invalid field name.
        """
        all_violations = []

        for filepath in _get_stage_files():
            collector = _collect_state_accesses(filepath)
            rel_path = filepath.relative_to(STAGES_DIR.parent.parent)

            for lineno, col, field_name in collector.direct_accesses:
                if field_name not in VALID_ATTRS and not _is_known_exception(filepath, field_name):
                    all_violations.append(
                        f"  {rel_path}:{lineno} - state.{field_name} "
                        f"(direct access)"
                    )

            for lineno, col, field_name in collector.getattr_accesses:
                if field_name not in VALID_ATTRS and not _is_known_exception(filepath, field_name):
                    all_violations.append(
                        f"  {rel_path}:{lineno} - getattr(state, '{field_name}', ...) "
                        f"(defensive access)"
                    )

            for lineno, col, field_name in collector.hasattr_accesses:
                if field_name not in VALID_ATTRS and not _is_known_exception(filepath, field_name):
                    all_violations.append(
                        f"  {rel_path}:{lineno} - hasattr(state, '{field_name}') "
                        f"(existence check)"
                    )

        if all_violations:
            pytest.fail(
                f"Found {len(all_violations)} stale PipelineState reference(s) "
                f"in src/stages/:\n"
                + "\n".join(all_violations)
                + f"\n\nValid PipelineState attributes:\n  "
                + "\n  ".join(sorted(VALID_ATTRS))
            )


class TestAuditInfrastructure:
    """Verify the audit infrastructure itself works correctly."""

    def test_stages_directory_exists(self):
        """src/stages/ directory must exist for audit to be meaningful."""
        assert STAGES_DIR.is_dir(), f"Stages directory not found: {STAGES_DIR}"

    def test_stage_files_found(self):
        """At least 5 stage files should exist (7 stages + __init__.py)."""
        files = _get_stage_files()
        assert len(files) >= 5, (
            f"Expected at least 5 stage files, found {len(files)}: "
            f"{[f.name for f in files]}"
        )

    def test_pipeline_state_has_fields(self):
        """PipelineState should have a reasonable number of fields."""
        assert len(VALID_ATTRS) >= 15, (
            f"PipelineState has only {len(VALID_ATTRS)} attributes, "
            f"expected at least 15"
        )

    def test_ast_collector_finds_direct_access(self):
        """Verify AST collector detects state.field patterns."""
        source = "state.matches\nstate.fake_field\n"
        tree = ast.parse(source)
        collector = _StateAccessCollector()
        collector.visit(tree)
        fields_found = {f for _, _, f in collector.direct_accesses}
        assert "matches" in fields_found
        assert "fake_field" in fields_found

    def test_ast_collector_finds_getattr(self):
        """Verify AST collector detects getattr(state, 'field', ...) patterns."""
        source = "getattr(state, 'caption_results', {})\ngetattr(state, 'bogus', None)\n"
        tree = ast.parse(source)
        collector = _StateAccessCollector()
        collector.visit(tree)
        fields_found = {f for _, _, f in collector.getattr_accesses}
        assert "caption_results" in fields_found
        assert "bogus" in fields_found

    def test_ast_collector_finds_hasattr(self):
        """Verify AST collector detects hasattr(state, 'field') patterns."""
        source = "hasattr(state, 'text_metadata')\nhasattr(state, 'nonexistent')\n"
        tree = ast.parse(source)
        collector = _StateAccessCollector()
        collector.visit(tree)
        fields_found = {f for _, _, f in collector.hasattr_accesses}
        assert "text_metadata" in fields_found
        assert "nonexistent" in fields_found

    def test_known_exceptions_still_exist_in_code(self):
        """Known legacy exceptions must still be present — remove entries when code is cleaned up."""
        found_exceptions = set()
        for filepath in _get_stage_files():
            collector = _collect_state_accesses(filepath)
            all_fields = (
                [(f, "direct") for _, _, f in collector.direct_accesses]
                + [(f, "getattr") for _, _, f in collector.getattr_accesses]
                + [(f, "hasattr") for _, _, f in collector.hasattr_accesses]
            )
            for field_name, _ in all_fields:
                key = (filepath.name, field_name)
                if key in KNOWN_LEGACY_EXCEPTIONS:
                    found_exceptions.add(key)

        stale_exceptions = set(KNOWN_LEGACY_EXCEPTIONS.keys()) - found_exceptions
        if stale_exceptions:
            details = [
                f"  ({fn}, '{fld}'): {KNOWN_LEGACY_EXCEPTIONS[(fn, fld)]}"
                for fn, fld in sorted(stale_exceptions)
            ]
            pytest.fail(
                f"KNOWN_LEGACY_EXCEPTIONS contains {len(stale_exceptions)} entries "
                f"that are no longer in the code — remove them:\n"
                + "\n".join(details)
            )

    def test_ast_collector_ignores_non_state_variables(self):
        """AST collector should NOT flag other_obj.field or getattr(other, ...)."""
        source = (
            "other.matches\n"
            "getattr(config, 'missing', None)\n"
            "hasattr(self, 'thing')\n"
        )
        tree = ast.parse(source)
        collector = _StateAccessCollector()
        collector.visit(tree)
        assert len(collector.direct_accesses) == 0
        assert len(collector.getattr_accesses) == 0
        assert len(collector.hasattr_accesses) == 0
