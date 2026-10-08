"""Current physical records serialize and spawn through their explicit owners."""

from __future__ import annotations

import importlib
import pickle
import subprocess
import sys

import pytest


@pytest.mark.parametrize(
    "current,symbol,canonical",
    [
        ("darcy.primal", "DarcySolution", "pymhm.postprocessing.solutions"),
        ("darcy.primal", "_DarcyLocalFactory", "pymhm._legacy.models.darcy.primal"),
        ("darcy.mixed_bdm", "BDMDarcySolution", "pymhm.postprocessing.solutions"),
        ("elasticity.primal", "PrimalElasticitySolution", "pymhm.postprocessing.primal_elasticity"),
        ("flow.solver", "solve_flow", "pymhm._legacy.models.flow.solver"),
        ("transport.solver", "ScalarSolution", "pymhm.postprocessing.solutions"),
        ("waves.helmholtz", "HelmholtzSolution", "pymhm.postprocessing.acoustics"),
        ("waves.maxwell", "MaxwellSolution", "pymhm.postprocessing.electromagnetic"),
    ],
)
def test_current_physical_globals_roundtrip_the_declared_owner(
    current: str, symbol: str, canonical: str
) -> None:
    """New records use public owners; persisted legacy globals resolve identically."""
    owner = f"pymhm._legacy.models.{current}"
    expected = getattr(importlib.import_module(owner), symbol)
    record = pickle.dumps(expected)
    assert canonical.encode("ascii") in record
    assert pickle.loads(record) is expected
    assert expected.__module__ == canonical
    assert pickle.loads(f"c{owner}\n{symbol}\n.".encode("ascii")) is expected


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
assert owner == 'pymhm.postprocessing.solutions'
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
