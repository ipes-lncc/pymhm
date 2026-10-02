"""Acquire the oscillatory-modulus geometry and spaces of L18 Table 3.

The printed rotation-error column contains entries incompatible with its stated
rates; this driver records independently integrated norms without changing those
published values or asserting a complete historical reproduction.
"""

import argparse
import hashlib
import json
from pathlib import Path

from plot_mixed_elasticity import metrics, oscillatory_fields, oscillatory_modulus
from threadpoolctl import threadpool_limits

from pymhm import FaceSpace, SkeletonSpace, TriangleMesh
from pymhm.elasticity_mixed import solve_elasticity_mixed

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "examples/results/elasticity-families"


def main() -> None:
    """Use H=1/4, h_in=h_sk/2 and explicitly distinguish macro and local degrees."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-degree", type=int, choices=[1, 2], default=1)
    parser.add_argument("--enrichment", type=int, choices=[0, 1], default=0)
    parser.add_argument("--segments", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--order", type=int, default=12)
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    mesh = TriangleMesh.unit_square(4)
    boundary = set(mesh.boundary_faces)
    degree, poisson = args.trace_degree + 1, 0.3
    rows = []
    with threadpool_limits(1):
        for segments in args.segments:
            refinement = 2 * segments
            skeleton = SkeletonSpace(
                mesh,
                tuple(
                    FaceSpace.uniform(degree, refinement)
                    if face in boundary
                    else FaceSpace.uniform(args.trace_degree, segments)
                    for face in range(len(mesh.faces))
                ),
                2,
            )
            solution = solve_elasticity_mixed(
                mesh,
                skeleton=skeleton,
                stress_degree=degree,
                enrichment=args.enrichment,
                local_refinement=refinement,
                quadrature_order=args.order,
                lame_mu=lambda x: oscillatory_modulus(x) / (2 * (1 + poisson)),
                lame_lambda=lambda x: (
                    oscillatory_modulus(x) * poisson / ((1 + poisson) * (1 - 2 * poisson))
                ),
                source=lambda x: oscillatory_fields(x)[2],
                dirichlet=lambda x: oscillatory_fields(x)[0],
            )
            row = dict(
                segments=segments,
                local_refinement=refinement,
                h_sk=1 / (4 * segments),
                h_in=1 / (8 * segments),
                trace_degree=args.trace_degree,
                local_normal_degree=degree,
                interior_degree=degree + args.enrichment,
                displacement_degree=solution.displacement_degree,
                trace_dofs=skeleton.size,
                algebraic_residual=solution.hybrid.residual,
                **metrics(solution, oscillatory_fields),
            )
            print(json.dumps(row), flush=True)
            rows.append(row)
    sources = [
        Path(__file__),
        ROOT / "examples/plot_mixed_elasticity.py",
        ROOT / "src/pymhm/elasticity_mixed.py",
        ROOT / "src/pymhm/bdm_family.py",
    ]
    report = dict(
        case="L18 section 6.1.2/Table 3 oscillatory Young modulus",
        macro_geometry="unit_square(4), 32 diagonal triangles",
        poisson_ratio=0.3,
        exterior_trace="degree k_in on every fine exterior face",
        interior_trace="degree k_sk on segments h_sk; h_in=h_sk/2",
        boundary="analytical displacement trace, including nonzero values",
        assembly_quadrature=args.order,
        error_quadrature=10,
        rows=rows,
        source_hashes={
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources
        },
    )
    (OUTPUT / f"l18-k{args.trace_degree}-enrichment{args.enrichment}.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
