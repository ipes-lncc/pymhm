"""Independent conforming UFL reference for the smooth 2D dynamic tutorial.

The exact displacement is t²/2 times two sine products. Its force is derived
symbolically in UFL, independently of the MHM acquisition's analytic Hessians.
No MHM trace, local response or local coefficient operator enters this solve.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from threadpoolctl import threadpool_limits

from pymhm import TriangleMesh, compile_form
from pymhm.backends.spaces import bind_space, coefficient_map
from pymhm.fem.scalar.triangle import nodal_space
from pymhm.io.provenance import current_source_manifest, file_digest
from pymhm.io.workspace import case_workspace, source_file, source_identity
from pymhm.linalg.dynamics import newmark_step
from pymhm.linalg.linear import factorize

ROOT = case_workspace()


def reference_row(n: int, *, dt: float = 0.005, final: float = 0.1) -> dict[str, Any]:
    """Assemble global P3 displacement, exact force and strong homogeneous data."""
    import basix
    import basix.ufl
    import ufl
    from dolfinx import fem

    started = perf_counter()
    mesh = TriangleMesh.unit_square(n)
    native = bind_space(
        mesh,
        basix.ufl.element(
            "Lagrange", "triangle", 3, shape=(2,), lagrange_variant=basix.LagrangeVariant.equispaced
        ),
    )
    try:
        u, v = ufl.TrialFunction(native.space), ufl.TestFunction(native.space)
        domain = native.space.mesh
        x = ufl.SpatialCoordinate(domain)
        w = ufl.as_vector(
            (
                ufl.sin(np.pi * x[0]) * ufl.sin(np.pi * x[1]),
                ufl.sin(2 * np.pi * x[0]) * ufl.sin(np.pi * x[1]),
            )
        )
        stress_shape = 2 * ufl.sym(ufl.grad(w)) + ufl.div(w) * ufl.Identity(2)
        dx = ufl.Measure("dx", domain=domain, metadata={"quadrature_degree": 14})
        M = compile_form(ufl.inner(u, v) * dx)
        K = compile_form(
            (2 * ufl.inner(ufl.sym(ufl.grad(u)), ufl.sym(ufl.grad(v))) + ufl.div(u) * ufl.div(v))
            * dx
        )
        F0 = compile_form(ufl.inner(w, v) * dx)
        F1 = compile_form(-ufl.inner(ufl.div(stress_shape), v) * dx)
        _, points = nodal_space(mesh, 3)
        nodes = np.flatnonzero(np.any(np.isclose(points, 0) | np.isclose(points, 1), axis=1))
        portable = (2 * nodes[:, None] + np.arange(2)).ravel()
        boundary = coefficient_map(native)[portable]
        free = np.setdiff1d(np.arange(native.size), boundary)
        mass, stiffness = M[free][:, free], K[free][:, free]
        force0, force1 = F0[free], F1[free]
        displacement, velocity = np.zeros(len(free)), np.zeros(len(free))
        original = 0.0
        count = round(final / dt)
        if abs(count * dt - final) > 1e-13:
            raise ValueError("The final time must be a whole number of steps")
        with factorize(mass) as mf, factorize(mass + dt * dt / 4 * stiffness) as sf:
            for step in range(count):
                old_u, old_v = displacement, velocity
                f0 = force0 + 0.5 * (step * dt) ** 2 * force1
                f1 = force0 + 0.5 * ((step + 1) * dt) ** 2 * force1
                displacement, velocity = newmark_step(
                    mass, stiffness, sf, mf, dt, old_u, old_v, f0, f1
                )
                inertia = mass @ (velocity - old_v)
                force = dt / 2 * (f0 + f1 - stiffness @ (old_u + displacement))
                original = max(
                    original,
                    float(
                        np.linalg.norm(inertia - force)
                        / max(np.linalg.norm(inertia) + np.linalg.norm(force), np.finfo(float).tiny)
                    ),
                )
        uh, vh = fem.Function(native.space), fem.Function(native.space)
        uh.x.array[free], vh.x.array[free] = displacement, velocity
        sigma_h = 2 * ufl.sym(ufl.grad(uh)) + ufl.div(uh) * ufl.Identity(2)
        norms = {}
        for order in (14, 18):
            measure = ufl.Measure("dx", domain=domain, metadata={"quadrature_degree": order})
            differences = {
                "displacement_l2": uh - 0.5 * final * final * w,
                "velocity_l2": vh - final * w,
                "stress_l2": sigma_h - 0.5 * final * final * stress_shape,
            }
            norms[str(order)] = {
                name: float(
                    np.sqrt(fem.assemble_scalar(fem.form(ufl.inner(delta, delta) * measure)))
                )
                for name, delta in differences.items()
            }
        low, high = norms.values()
        changes = {
            name: abs(low[name] - high[name]) / max(high[name], np.finfo(float).tiny)
            for name in high
        }
        return {
            "resolution": n,
            "degree": 3,
            "dimension": 2,
            "dt": dt,
            "final_time": final,
            "dofs": native.size,
            **high,
            "original_momentum_relative_max": original,
            "quadrature": {
                "assembly_degree": 14,
                "error_degrees": [14, 18],
                "errors": norms,
                "relative_changes": changes,
            },
            "wall_seconds": perf_counter() - started,
            "accepted": bool(original <= 1e-10 and max(changes.values()) <= 1e-6),
        }
    finally:
        native.close()


def main() -> None:
    """Acquire several sufficiently refined classical meshes in one fresh record."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--levels", type=int, nargs="+", default=[16, 32, 64])
    parser.add_argument("--dt", type=float, default=0.005)
    parser.add_argument("--final", type=float, default=0.1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("An existing numerical record cannot be overwritten")
    paths = [
        Path(__file__),
        *sorted((ROOT / "src/pymhm").rglob("*.py")),
        ROOT / "pixi.lock",
        ROOT / "pixi.toml",
        ROOT / "pyproject.toml",
    ]
    sources = current_source_manifest(source_identity(ROOT, paths), packages=("pymhm",))
    record = {
        "schema": "pymhm-independent-2d-elastodynamic-reference-v1",
        "source_sha256": sources,
        "case": "u=t²/2*(sin(pi*x)sin(pi*y),sin(2pi*x)sin(pi*y))",
        "operator": "rho=lambda=mu=1",
        "boundary": "homogeneous full Dirichlet",
        "time_integrator": "Newmark beta=1/4,gamma=1/2",
        "force_derivation": "independent symbolic UFL divergence of physical stress",
        "coefficient_vectors_persisted": False,
        "field_replay_claimed": False,
        "rows": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(1):
        for n in args.levels:
            row = reference_row(n, dt=args.dt, final=args.final)
            record["rows"].append(row)
            args.output.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
            print(json.dumps(row), flush=True)
            if not row["accepted"]:
                raise RuntimeError(f"Classical physical equations or quadrature rejected n={n}")
    if any(file_digest(source_file(name, root=ROOT)) != digest for name, digest in sources.items()):
        raise RuntimeError("An executed classical source changed during acquisition")


if __name__ == "__main__":
    main()
