"""Visible introduction controls reach meshes, operators and profile acquisition."""

import ast
import builtins
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def formulation(
    name: str, controls: dict[str, Any], *, definitions: tuple[str, ...] = ()
) -> dict[str, Any]:
    """Load numerical definitions and their imports without campaigns or display cells.

    Explicitly selected portable definitions use the supplied namespace. Native
    formulations also import names referenced by their function and class bodies.
    """
    notebook = json.loads((ROOT / f"notebooks/introduction/{name}.ipynb").read_text())
    nodes = [
        node
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
        for node in ast.parse("".join(cell["source"])).body
    ]
    namespace = {
        "__name__": "introduction_control_test",
        "ROOT": ROOT,
        "np": np,
        "dataclass": dataclass,
        **controls,
    }
    selected = [
        node
        for node in nodes
        if isinstance(node, ast.FunctionDef | ast.ClassDef)
        and (not definitions or node.name in definitions)
    ]
    referenced = {
        child.id
        for node in selected
        for child in ast.walk(node)
        if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load)
    }
    for node in nodes:
        if (
            not definitions
            and isinstance(node, ast.Import | ast.ImportFrom)
            and any(
                (alias.asname or alias.name.split(".")[0]) in referenced for alias in node.names
            )
        ):
            exec(compile(ast.Module([node], type_ignores=[]), name, "exec"), namespace)
    for node in selected:
        exec(compile(ast.Module([node], type_ignores=[]), name, "exec"), namespace)
    return namespace


@pytest.fixture
def numerical_imports_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reject display dependencies even when the complete test profile provides them."""
    original_import = builtins.__import__

    def import_numerical(name: str, *args: Any, **kwargs: Any) -> Any:
        """Keep native assembly independent of notebook rendering dependencies."""
        if name.split(".")[0] in {"matplotlib", "IPython"}:
            raise AssertionError(f"Numerical formulation imports a display dependency: {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_numerical)


@pytest.mark.fem
@pytest.mark.usefixtures("numerical_imports_only")
def test_rad_nondefault_diffusion_refinement_and_assembly_quadrature(tmp_path: Path) -> None:
    """The visible controls change actual P1 operators and the independent local mesh."""
    pytest.importorskip("dolfinx")
    from pymhm.execution.cpu import ExecutionConfig
    from pymhm.fem.scalar.triangle import scalar_operators

    controls = dict(
        EPSILON=0.002,
        LOCAL_SUBDIVISIONS=4,
        LOCAL_QUADRATURE_DEGREE=10,
        OPERATOR_CHECK_ORDER=12,
        ERROR_ORDER=16,
        RAD_JIT_OPTIONS={"cache_dir": tmp_path / "rad-jit", "timeout": 7200},
    )
    namespace = formulation("mhm_usfem_rad", controls)
    macro, skeleton, system, solution = namespace["solve_rad_case"](
        2, controls["EPSILON"], False, execution=ExecutionConfig(backend="serial")
    )
    assert len(macro.cells) == 8 and all(face.degrees == (0,) for face in skeleton.faces)
    assert all(
        len(data["mesh"].cells) == 16 and data["local_subdivisions"] == 4
        for data in system.local_metadata
    )
    data = system.local_metadata[0]
    operator, _, load = scalar_operators(
        data["mesh"], 1, diffusion=controls["EPSILON"], reaction=1, source=1, order=12
    )
    free = data["free"]
    np.testing.assert_allclose(
        system.responses[0].problem.matrix.toarray(),
        operator[free][:, free].toarray(),
        atol=1e-12,
        rtol=1e-10,
    )
    np.testing.assert_allclose(system.responses[0].problem.load, load[free], atol=1e-12, rtol=1e-10)
    assert solution.raw_residual < 1e-12
    from pymhm.backends.spaces import create_native_mesh

    native = create_native_mesh(data["mesh"])
    _, a, _ = namespace["rad_forms"](native, epsilon=controls["EPSILON"], stabilized=False)
    assert a.integrals()[0].metadata()["quadrature_degree"] == 10


@pytest.mark.fem
@pytest.mark.usefixtures("numerical_imports_only")
def test_brinkman_nondefault_subdivision_viscosity_drag_and_quadrature(tmp_path: Path) -> None:
    """The main Taylor-Hood mesh and physical gauge follow the displayed controls."""
    pytest.importorskip("dolfinx")
    controls = dict(
        NU=0.02,
        GAMMA=1.5,
        LOCAL_SUBDIVISIONS=8,
        LOCAL_QUADRATURE_DEGREE=32,
        ERROR_ORDER=16,
        INVERSE_M=0.01,
        BRINKMAN_JIT_OPTIONS={"cache_dir": tmp_path / "brinkman-jit", "timeout": 7200},
    )
    namespace = formulation("stokes_brinkman_boundary_layer", controls)
    namespace["truth"] = namespace["BrinkmanLayer"](controls["NU"], controls["GAMMA"])
    macro, _, system, solution, gauge = namespace["solve_brinkman_case"](1, False)
    assert len(macro.cells) == 2
    assert all(len(data["mesh"].cells) == 64 for data in system.local_metadata)
    assert solution.raw_residual < 1e-12
    assert abs(gauge[0] @ solution.trace - gauge[1]) < 1e-12
    from pymhm.backends.spaces import create_native_mesh

    native = create_native_mesh(system.local_metadata[0]["mesh"])
    _, a, _, _ = namespace["brinkman_forms"](native, stabilized=False)
    assert a.integrals()[0].metadata()["quadrature_degree"] == 32


@pytest.mark.visualization
def test_rad_profiles_use_the_configured_primary_diffusion(tmp_path: Path) -> None:
    """A changed primary epsilon selects its actual references, fields and exact profile."""
    plt = pytest.importorskip("matplotlib.pyplot")

    from examples.introduction.transport import plot_rad_profiles
    from pymhm.fem.scalar.triangle import nodal_space
    from pymhm.meshes.triangle import TriangleMesh

    namespace = formulation(
        "mhm_usfem_rad",
        dict(
            EPSILON=0.002,
            LOCAL_SUBDIVISIONS=4,
            LOCAL_QUADRATURE_DEGREE=10,
            OPERATOR_CHECK_ORDER=12,
            ERROR_ORDER=16,
        ),
        definitions=("ReactionLayer",),
    )
    layer = namespace["ReactionLayer"]
    macro = TriangleMesh.unit_square()
    meshes = tuple(macro.submesh(cell, 4) for cell in range(len(macro.cells)))
    classical = TriangleMesh.unit_square(2)
    references, challenging, resolved = {}, {}, {}
    for epsilon in (0.002, 1e-5):
        exact = layer(epsilon)
        fields = tuple(exact.value(nodal_space(mesh, 1)[1]) for mesh in meshes)
        references[epsilon, 128 if epsilon == 0.002 else 1024] = (
            classical,
            exact.value(nodal_space(classical, 2)[1]),
        )
        for method in ("MHM-Galerkin", "MHM-USFEM"):
            challenging[epsilon, method] = meshes, fields
            resolved[epsilon, 16, method] = macro, None, meshes, fields
    plt.close("all")
    plot_rad_profiles(
        macro,
        challenging,
        resolved,
        references,
        layer,
        epsilon_primary=0.002,
        underresolved_subdivisions=4,
        reports=tmp_path,
    )
    figure = plt.gcf()
    assert "epsilon=0.002" in figure.axes[0].get_title()
    assert "local subdivisions=4" in figure.axes[0].get_title()
    assert "epsilon=1e-05" in figure.axes[1].get_title()
    x = figure.axes[0].lines[0].get_xdata()
    np.testing.assert_allclose(
        figure.axes[0].lines[0].get_ydata(),
        layer(0.002).value(np.column_stack((x, x * 0 + 0.37))),
        atol=1e-12,
        rtol=1e-10,
    )
    assert (tmp_path / "rad-layer-profiles.png").is_file()
    plt.close("all")
