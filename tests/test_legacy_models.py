"""Current physical records serialize and spawn through their explicit owners."""

from __future__ import annotations

import importlib
import pickle
import subprocess
import sys

import pytest


@pytest.mark.parametrize(
    "current,symbol",
    [
        ("darcy.primal", "DarcySolution"),
        ("darcy.primal", "_DarcyLocalFactory"),
        ("darcy.mixed_bdm", "BDMDarcySolution"),
        ("elasticity.primal", "PrimalElasticitySolution"),
        ("flow.solver", "solve_flow"),
        ("transport.solver", "ScalarSolution"),
        ("waves.helmholtz", "HelmholtzSolution"),
        ("waves.maxwell", "MaxwellSolution"),
    ],
)
def test_current_physical_globals_roundtrip_the_declared_owner(current: str, symbol: str) -> None:
    """New GLOBAL records reference the actual module and require no import fallback."""
    owner = f"pymhm._legacy.models.{current}"
    expected = getattr(importlib.import_module(owner), symbol)
    record = pickle.dumps(expected)
    assert owner.encode("ascii") in record
    assert pickle.loads(record) is expected
    assert expected.__module__ == owner


def test_current_package_first_import_and_spawn_keep_canonical_owner() -> None:
    """Fresh workers resolve currently serialized globals directly to their owners."""
    script = """
import importlib
import multiprocessing
import pickle
from concurrent.futures import ProcessPoolExecutor

import pymhm._legacy.models
import pymhm._legacy.models.darcy.primal
from pymhm._legacy.models.darcy.primal import DarcySolution

canonical_package = importlib.import_module('pymhm._legacy.models')
canonical_module = importlib.import_module('pymhm._legacy.models.darcy.primal')
assert pymhm._legacy.models is canonical_package
assert pymhm._legacy.models.darcy.primal is canonical_module
assert canonical_module.__spec__.name == canonical_module.__name__
assert canonical_package.__path__
current = pickle.loads(pickle.dumps(DarcySolution))
assert current is DarcySolution
with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context('spawn')) as pool:
    owner = pool.submit(getattr, current, '__module__').result(timeout=30)
assert owner == 'pymhm._legacy.models.darcy.primal'
assert not multiprocessing.active_children()
assert 'pymhm.models' not in __import__('sys').modules
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", script],
        capture_output=True,
        text=True,
        timeout=45,
        check=False,
    )
    assert result.returncode == 0, result.stderr
