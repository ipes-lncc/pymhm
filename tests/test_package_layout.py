"""Canonical public imports reject removed paths and leave native adapters unopened."""

from __future__ import annotations

import importlib
import inspect
import pickle
import subprocess
import sys
from dataclasses import is_dataclass
from pathlib import Path

import pytest

import pymhm
from pymhm._registry import PUBLIC_EXPORTS


def test_package_namespace_contains_only_canonical_responsibilities() -> None:
    """The installed top-level namespace contains logical owners and no aliases."""
    import pkgutil

    assert {module.name for module in pkgutil.iter_modules(pymhm.__path__)} == {
        "_legacy",
        "_registry",
        "adaptivity",
        "backends",
        "core",
        "estimators",
        "execution",
        "fem",
        "io",
        "linalg",
        "materials",
        "meshes",
        "methods",
        "postprocessing",
        "recovery",
    }


@pytest.mark.parametrize("name,owner", sorted(PUBLIC_EXPORTS.items()))
def test_root_exports_are_canonical_objects(name: str, owner: tuple[str, str]) -> None:
    """The root API returns the implementation held by its declared owner."""
    module, symbol = owner
    assert getattr(pymhm, name) is getattr(importlib.import_module(module), symbol)


def test_generic_api_exposes_free_operations_and_equation_records() -> None:
    """User forms compose through functions around explicit problem/solution records."""
    from pymhm import Equation, LocalEquations, MultiscaleProblem

    assert all(is_dataclass(record) for record in (Equation, LocalEquations, MultiscaleProblem))
    for name in ("assemble", "solve", "compile_local_equations", "leaf_moment", "newmark_step"):
        assert inspect.isfunction(getattr(pymhm, name))
    assert set(pymhm.__all__) == set(PUBLIC_EXPORTS)
    assert set(pymhm.__all__) <= set(dir(pymhm))
    with pytest.raises(AttributeError, match="no attribute 'unknown_export'"):
        _ = pymhm.unknown_export


def test_current_pickle_globals_roundtrip_their_canonical_owners() -> None:
    """New records serialize only the modules where their implementations live."""
    from pymhm.core.contracts import LocalProblem
    from pymhm.meshes.triangle import TriangleMesh

    for record in (LocalProblem, TriangleMesh):
        serialized = pickle.dumps(record)
        assert record.__module__.encode() in serialized
        assert pickle.loads(serialized) is record
    with pytest.raises(ModuleNotFoundError, match="pymhm.hybrid"):
        pickle.loads(b"cpymhm.hybrid\nLocalProblem\n.")
    with pytest.raises(ModuleNotFoundError, match="pymhm.mesh"):
        pickle.loads(b"cpymhm.mesh\nTriangleMesh\n.")


def test_cold_namespace_has_no_alias_finder_or_physical_dispatch() -> None:
    """A fresh interpreter has no import hook or hidden root fallback for old APIs."""
    script = """
import sys
import pymhm
assert 'LocalEquations' in dir(pymhm)
for name in ('solve_darcy', 'solve_mh2m', 'DarcySolution', 'MaxwellStepper', 'mesh', 'models'):
    assert name not in dir(pymhm), name
    assert not hasattr(pymhm, name), name
assert 'pymhm._compat' not in sys.modules
assert not any(type(finder).__module__ == 'pymhm._compat' for finder in sys.meta_path)
try:
    from pymhm import solve_darcy
except ImportError:
    pass
else:
    raise AssertionError('removed root solver unexpectedly imported')
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", script],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_algebraic_import_does_not_load_native_backends() -> None:
    """Root and local algebra imports leave optional/native FEM resources unopened."""
    script = """
import sys
import pymhm
assert 'pymhm.core.contracts' not in sys.modules
from pymhm.core.contracts import LocalProblem
from pymhm.core.equations import Equation, LocalEquations
from pymhm.core.multiscale import assemble
for name in ('basix', 'dolfinx', 'ufl', 'mpi4py', 'cupy', 'gmsh', 'netgen'):
    assert name not in sys.modules, name
assert pymhm.LocalProblem is LocalProblem
assert pymhm.Equation is Equation and pymhm.LocalEquations is LocalEquations
assert pymhm.assemble is assemble
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", script],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
