"""Recursive documentation and typing contracts include all implementation helpers."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from runpy import run_path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CHECK = ROOT / "scripts/check_source_structure.py"
source_contract_issues = run_path(str(CHECK))["source_contract_issues"]


def test_nested_private_and_async_definitions_have_the_same_contract(tmp_path):
    source = tmp_path / "contracts.py"
    source.write_text(
        '''"""A module with deliberately incomplete implementation contracts."""
class _Record:
    def _method(self, value, /, *, count, **kwargs):
        async def inner(*values):
            return values
        return value
'''
    )
    issues, count = source_contract_issues(source)
    assert count == 3
    assert len(issues) == 9
    assert any("_Record has no docstring" in issue for issue in issues)
    assert any("_Record._method argument 'kwargs'" in issue for issue in issues)
    assert any("_Record._method.inner argument 'values'" in issue for issue in issues)
    assert not any("argument 'self'" in issue for issue in issues)


def test_receiver_and_annotated_variadic_arguments_are_accepted(tmp_path):
    source = tmp_path / "contracts.py"
    source.write_text(
        '''"""Documented module."""
class Record:
    """Documented class."""
    @classmethod
    def make(cls, value: float, /, *values: float, count: int = 1, **kw: float) -> float:
        """Documented class constructor."""
        def _combine(delta: float) -> float:
            """Documented private nested operation."""
            return value + delta
        return _combine(sum(values))
async def _task(value: int) -> int:
    """Documented asynchronous private operation."""
    return value
'''
    )
    assert source_contract_issues(source) == ([], 4)


@pytest.mark.parametrize("documented", [True, False])
def test_overload_signatures_share_their_implementations_documentation(tmp_path, documented):
    source = tmp_path / "contracts.py"
    source.write_text(
        '''"""Documented overload module."""
from typing import overload
@overload
def operation(value: int) -> int: ...
def operation(value: int) -> int:
'''
        + ('    """Documented implementation."""\n' if documented else "")
        + "    return value\n"
    )
    issues, count = source_contract_issues(source)
    assert count == 2
    assert len(issues) == (0 if documented else 2)
    assert all("operation has no docstring" in issue for issue in issues)


@pytest.mark.parametrize("problem", ["empty", "missing_module", "missing_return", "valid"])
def test_command_recurses_without_importing_sources(tmp_path, problem):
    source = tmp_path / "nested"
    source.mkdir()
    if problem != "empty":
        prefix = "" if problem == "missing_module" else '"""A runtime module."""\n'
        result = "" if problem == "missing_return" else " -> None"
        (source / "contracts.py").write_text(
            prefix
            + f'def operation(){result}:\n    """A documented operation."""\n    pass\n'
            + 'raise RuntimeError("The checker must not import the source")\n'
        )
    run = subprocess.run(
        [sys.executable, str(CHECK), str(tmp_path)], capture_output=True, text=True, check=False
    )
    if problem == "valid":
        assert run.returncode == 0
        assert "1 modules and 1 definitions" in run.stdout
    else:
        assert run.returncode != 0
        assert {
            "empty": "No Python modules",
            "missing_module": "module has no docstring",
            "missing_return": "no return annotation",
        }[problem] in run.stderr
    assert "RuntimeError" not in run.stderr
