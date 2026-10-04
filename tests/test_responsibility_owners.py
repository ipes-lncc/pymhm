"""Numerical operations have explicit owners without lower-layer PDE dispatch."""

from __future__ import annotations

import importlib
import pickle
import subprocess
import sys

import numpy as np
import pytest


@pytest.mark.parametrize(
    "owner,symbol",
    [
        ("pymhm.materials.evaluation", "scalar_values"),
        ("pymhm.materials.evaluation", "vector_values"),
        ("pymhm.materials.evaluation", "tensor_values"),
        ("pymhm.materials.evaluation", "scalar_values_3d"),
        ("pymhm.materials.evaluation", "tensor_values_3d"),
        ("pymhm.materials.evaluation", "vector_values_3d"),
        ("pymhm.meshes.cartesian", "CartesianMacroMesh"),
        ("pymhm.fem.scalar.quadrilateral", "qk_basis"),
        ("pymhm.fem.scalar.quadrilateral", "quadrilateral_operators"),
        ("pymhm.meshes.hexahedron", "HexMesh"),
        ("pymhm.fem.hdiv.mapped", "mapped_rt_basis"),
        ("pymhm.meshes.tetrahedron", "TetraMesh"),
        ("pymhm.meshes.tetrahedron", "tetra_barycentric_gradients"),
        ("pymhm.fem.traces.triangle_3d", "TriangularSkeleton"),
        ("pymhm.fem.traces.triangle_3d", "tetra_trace_coupling"),
        ("pymhm._legacy.models.geometry", "solve_darcy_polygons"),
        ("pymhm._legacy.models.geometry", "solve_transport_polygons"),
        ("pymhm._legacy.models.geometry", "solve_brinkman_polygons"),
        ("pymhm._legacy.models.geometry", "solve_mshho_polygons"),
        ("pymhm._legacy.models.geometry", "solve_elasticity_mixed_polygons"),
        ("pymhm.materials.cartesian", "CartesianCellField"),
        ("pymhm.io.datasets.spe10", "load_spe10_model2"),
        ("pymhm.io.datasets.spe10", "download_spe10_model2"),
    ],
)
def test_current_globals_roundtrip_their_numerical_owner(owner: str, symbol: str) -> None:
    """Serialized free functions and records refer directly to their implementation."""
    expected = getattr(importlib.import_module(owner), symbol)
    assert pickle.loads(pickle.dumps(expected)) is expected
    assert expected.__module__ == owner


def test_mesh_and_fem_imports_do_not_load_problem_models() -> None:
    """Geometry, fields, bases and traces can be consumed without PDE dispatch."""
    script = """
import sys
from pymhm.meshes.cartesian import CartesianMacroMesh
from pymhm.meshes.hexahedron import HexMesh
from pymhm.meshes.tetrahedron import TetraMesh
from pymhm.meshes.polygonal import PolygonMesh
from pymhm.materials.cartesian import CartesianCellField
from pymhm.fem.scalar.quadrilateral import qk_basis
from pymhm.fem.hdiv.mapped import mapped_rt_basis
from pymhm.fem.traces.triangle_3d import TriangularSkeleton
assert not any(name.startswith('pymhm._legacy.models.') for name in sys.modules)
assert 'pymhm.io.reservoir' not in sys.modules
assert 'pymhm.io.datasets.spe10' not in sys.modules
for name in ('basix', 'dolfinx', 'mpi4py', 'petsc4py', 'ufl'):
    assert name not in sys.modules, name
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", script],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_polygon_mesh_does_not_dispatch_physical_solvers() -> None:
    """A geometry owner exposes no lazy physical driver or solver fallback."""
    module = importlib.import_module("pymhm.meshes.polygonal")
    assert "__getattr__" not in vars(module)
    for name in ("solve_darcy_polygons", "solve_brinkman_polygons", "unknown_solver"):
        with pytest.raises(AttributeError, match=f"no attribute '{name}'"):
            getattr(module, name)


def test_polygon_star_import_exposes_geometry_without_physical_dispatch() -> None:
    """Importing the mesh namespace cannot acquire a problem solver implicitly."""
    namespace = {}
    exec("from pymhm.meshes.polygonal import *", namespace)
    from pymhm.meshes.polygonal import PolygonMesh

    assert namespace["PolygonMesh"] is PolygonMesh
    assert not any(name.startswith("solve_") for name in namespace)


def test_rectangular_scatter_sums_shared_dofs_in_the_declared_spaces() -> None:
    """Two cell contributions share DOFs while row and column spaces differ."""
    from pymhm.fem.assembly import assemble_element_blocks

    blocks = np.arange(1, 13, dtype=float).reshape(2, 2, 3)
    rows = np.array([[0, 1], [1, 2]])
    columns = np.array([[0, 1, 2], [1, 2, 3]])
    matrix = assemble_element_blocks(blocks, rows, columns, (3, 4))
    np.testing.assert_array_equal(matrix.toarray(), [[1, 2, 3, 0], [4, 12, 14, 9], [0, 10, 11, 12]])
    assert matrix.format == "csc"
    assert assemble_element_blocks.__module__ == "pymhm.fem.assembly"
    assert pickle.loads(pickle.dumps(assemble_element_blocks)) is assemble_element_blocks
