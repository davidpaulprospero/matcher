"""AST-based audit: every subprocess.run/Popen call with text=True must have encoding='utf-8'.

Rule 27 compliance: ALL subprocess.Popen/run with text=True MUST add encoding='utf-8', errors='replace'.
This test parses all Python files in src/ using the ast module and asserts compliance.
"""

import ast
import pytest
from pathlib import Path

SRC_DIR = Path(__file__).parent.parent / "src"


def _find_subprocess_calls_missing_encoding():
    """Walk all .py files in src/ and find subprocess calls with text=True but no encoding kwarg."""
    violations = []

    for py_file in sorted(SRC_DIR.rglob("*.py")):
        try:
            source = py_file.read_text(encoding="utf-8", errors="replace")
            tree = ast.parse(source, filename=str(py_file))
        except SyntaxError:
            continue

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue

            # Check if this is subprocess.run(...) or subprocess.Popen(...)
            func = node.func
            is_subprocess_call = False

            if isinstance(func, ast.Attribute):
                if func.attr in ("run", "Popen"):
                    # Check if it's subprocess.run or subprocess.Popen
                    if isinstance(func.value, ast.Name) and func.value.id == "subprocess":
                        is_subprocess_call = True
                    elif isinstance(func.value, ast.Attribute) and func.value.attr == "subprocess":
                        is_subprocess_call = True

            if not is_subprocess_call:
                continue

            # Extract keyword arguments
            kwargs = {kw.arg: kw.value for kw in node.keywords if kw.arg is not None}

            # Check if text=True is present
            has_text_true = False
            if "text" in kwargs:
                val = kwargs["text"]
                if isinstance(val, ast.Constant) and val.value is True:
                    has_text_true = True
                elif isinstance(val, ast.NameConstant) and getattr(val, "value", None) is True:
                    has_text_true = True

            if not has_text_true:
                continue

            # text=True is present - check for encoding kwarg
            has_encoding = "encoding" in kwargs
            has_errors = "errors" in kwargs

            rel_path = py_file.relative_to(SRC_DIR.parent)

            if not has_encoding:
                violations.append(
                    f"{rel_path}:{node.lineno} - subprocess call has text=True but missing encoding='utf-8'"
                )
            if not has_errors:
                violations.append(
                    f"{rel_path}:{node.lineno} - subprocess call has text=True but missing errors='replace'"
                )

    return violations


class TestSubprocessEncodingCompliance:
    """Ensure all subprocess calls with text=True have encoding parameters (Rule 27)."""

    def test_all_subprocess_calls_have_encoding(self):
        """Every subprocess.run/Popen with text=True must also have encoding='utf-8' and errors kwarg."""
        violations = _find_subprocess_calls_missing_encoding()
        assert violations == [], (
            f"Found {len(violations)} subprocess call(s) missing encoding parameters:\n"
            + "\n".join(f"  - {v}" for v in violations)
        )

    def test_audit_finds_files(self):
        """Sanity check: the audit actually finds subprocess calls to verify."""
        # Count total subprocess calls with text=True
        count = 0
        for py_file in SRC_DIR.rglob("*.py"):
            try:
                source = py_file.read_text(encoding="utf-8", errors="replace")
                tree = ast.parse(source, filename=str(py_file))
            except SyntaxError:
                continue

            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                func = node.func
                if isinstance(func, ast.Attribute) and func.attr in ("run", "Popen"):
                    if isinstance(func.value, ast.Name) and func.value.id == "subprocess":
                        kwargs = {kw.arg: kw.value for kw in node.keywords if kw.arg is not None}
                        if "text" in kwargs:
                            val = kwargs["text"]
                            if isinstance(val, ast.Constant) and val.value is True:
                                count += 1

        # We know there are many subprocess calls in src/
        assert count >= 10, f"Expected at least 10 subprocess calls with text=True, found {count}"
