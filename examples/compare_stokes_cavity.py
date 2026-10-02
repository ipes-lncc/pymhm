"""Integrate broken MHM cavity fields against a refined conforming Taylor--Hood reference."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from solve_spe10_taylor_hood import TaylorHoodField
from threadpoolctl import threadpool_limits

from pymhm.cut_cells import _clip_polygon
from pymhm.elements import triangle_quadrature
from pymhm.lagrange import element_tabulate, nodal_space, reference_basis
from pymhm.mesh import TriangleMesh

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "examples/results/stokes-adaptive"


def interior_quadrature(mesh: TriangleMesh, order: int, cutout: float) -> tuple:
    """Clip triangles to two disjoint rectangles excluding the upper corner boxes."""
    if not 0 < cutout < 0.5:
        raise ValueError("corner cutout must lie in (0,0.5)")
    base, weight = triangle_quadrature(order)
    rules = []
    for vertices, area in zip(mesh.points[mesh.cells], mesh.areas, strict=True):
        parts, weights = [], []
        transform = (vertices[1:] - vertices[0]).T
        for bounds in ((0, 1, 0, 1 - cutout), (cutout, 1 - cutout, 1 - cutout, 1)):
            polygon = vertices.copy()
            for axis, level, lower in (
                (0, bounds[0], True),
                (0, bounds[1], False),
                (1, bounds[2], True),
                (1, bounds[3], False),
            ):
                polygon = _clip_polygon(polygon, axis, level, lower)
                if len(polygon) < 3:
                    break
            for index in range(1, len(polygon) - 1):
                triangle = polygon[[0, index, index + 1]]
                size = np.linalg.det((triangle[1:] - triangle[0]).T) / 2
                if size <= 0:
                    continue
                local = np.linalg.solve(transform, (base @ triangle - vertices[0]).T).T
                parts.append(np.column_stack((1 - local.sum(axis=1), local)))
                weights.append(weight * size / area)
        rules.append(
            (np.concatenate(parts), np.concatenate(weights))
            if parts
            else (np.full((1, 3), 1 / 3), np.zeros(1))
        )
    count = max(len(weights) for _, weights in rules)
    bary = np.full((len(rules), count, 3), 1 / 3)
    weights = np.zeros((len(rules), count))
    for cell, (coordinates, values) in enumerate(rules):
        bary[cell, : len(values)], weights[cell, : len(values)] = coordinates, values
    return bary, weights


def norms(
    arrays: dict,
    reference: TaylorHoodField,
    order: int,
    *,
    drag: float = 0.0,
    corner_cutout: float = 0.0,
) -> dict:
    """Integrate on actual MHM fine triangles; repeat orders to resolve reference intersections."""
    bary, weights = triangle_quadrature(order)
    totals = np.zeros(6)
    for macro in range(len(arrays["macro_cells"])):
        mesh = TriangleMesh(arrays[f"points_{macro}"], arrays[f"cells_{macro}"])
        local_bary, local_weights = (
            interior_quadrature(mesh, order, corner_cutout)
            if corner_cutout
            else (
                np.broadcast_to(bary, (len(mesh.cells), *bary.shape)),
                np.broadcast_to(weights, (len(mesh.cells), len(weights))),
            )
        )
        udofs, _, ubasis, gradients, _ = element_tabulate(
            mesh, int(arrays["velocity_degree"]), local_bary
        )
        pdofs, _, pbasis, _, _ = element_tabulate(mesh, int(arrays["pressure_degree"]), local_bary)
        for start in range(0, len(mesh.cells), 128):
            part = slice(start, start + 128)
            points = np.einsum("tqi,tij->tqj", local_bary[part], mesh.points[mesh.cells[part]])
            u = np.einsum("tqi,tia->tqa", ubasis[part], arrays[f"velocity_{macro}"][udofs[part]])
            p = np.einsum("tqi,ti->tq", pbasis[part], arrays[f"pressure_{macro}"][pdofs[part]])
            grad = np.einsum(
                "tqia,tib->tqba", gradients[part], arrays[f"velocity_{macro}"][udofs[part]]
            )
            ru, rp = reference.evaluate(points.reshape(-1, 2))
            rg, _ = reference.gradient(points.reshape(-1, 2))
            ru, rp, rg = ru.reshape(u.shape), rp.reshape(p.shape), rg.reshape(grad.shape)
            weight = mesh.areas[part, None] * local_weights[part]
            totals += [
                np.sum(weight * np.sum((u - ru) ** 2, axis=2)),
                np.sum(weight * (p - rp) ** 2),
                np.sum(weight * np.sum(ru**2, axis=2)),
                np.sum(weight * rp**2),
                np.sum(weight * np.sum((grad - rg) ** 2, axis=(2, 3))),
                np.sum(weight * np.sum(rg**2, axis=(2, 3))),
            ]
    absolute = np.sqrt(totals[[0, 1, 4]])
    denominator = np.sqrt(totals[[2, 3, 5]])
    energy = np.sqrt(totals[4] + drag * totals[0])
    reference_energy = np.sqrt(totals[5] + drag * totals[2])
    return dict(
        quadrature_order=order,
        corner_cutout=corner_cutout,
        velocity_l2=float(absolute[0]),
        pressure_l2=float(absolute[1]),
        velocity_h1_seminorm=float(absolute[2]),
        velocity_reference_l2=float(denominator[0]),
        pressure_reference_l2=float(denominator[1]),
        velocity_reference_h1_seminorm=float(denominator[2]),
        velocity_relative=float(absolute[0] / denominator[0]),
        pressure_relative=float(absolute[1] / denominator[1]),
        velocity_h1_seminorm_relative=float(absolute[2] / denominator[2]),
        velocity_energy=float(energy),
        velocity_reference_energy=float(reference_energy),
        velocity_energy_relative=float(energy / reference_energy),
    )


def profiles(arrays: dict, reference: TaylorHoodField) -> dict:
    """Sample individual fine-cell line segments without averaging coincident macro traces."""
    output = {}
    for label, axis, level in (("upper", 1, 0.99), ("vertical", 0, 0.5)):
        points, velocity, pressure, intersections = [], [], [], []
        for macro in range(len(arrays["macro_cells"])):
            mesh = TriangleMesh(arrays[f"points_{macro}"], arrays[f"cells_{macro}"])
            udofs = nodal_space(mesh, int(arrays["velocity_degree"]))[0]
            pdofs = nodal_space(mesh, int(arrays["pressure_degree"]))[0]
            vertices = mesh.points[mesh.cells]
            selected = np.flatnonzero(
                (vertices[:, :, axis].min(axis=1) <= level)
                & (vertices[:, :, axis].max(axis=1) >= level)
            )
            for cell in selected:
                triangle = vertices[cell]
                crossing = []
                for first, second in ((0, 1), (1, 2), (2, 0)):
                    a, b = triangle[[first, second]]
                    if a[axis] == b[axis]:
                        if a[axis] == level:
                            crossing.extend((a, b))
                    else:
                        parameter = (level - a[axis]) / (b[axis] - a[axis])
                        if 0 <= parameter <= 1:
                            crossing.append(a + parameter * (b - a))
                if len(crossing) < 2:
                    continue
                endpoints = np.asarray(crossing)
                a = endpoints[np.argmin(endpoints[:, 1 - axis])]
                b = endpoints[np.argmax(endpoints[:, 1 - axis])]
                if np.linalg.norm(b - a) < 1e-14:
                    continue
                sample = a + np.linspace(0, 1, 9)[:, None] * (b - a)
                local = np.linalg.solve((triangle[1:] - triangle[0]).T, (sample - triangle[0]).T).T
                bary = np.column_stack((1 - local.sum(axis=1), local))
                ubasis = reference_basis(int(arrays["velocity_degree"]), bary)[0]
                pbasis = reference_basis(int(arrays["pressure_degree"]), bary)[0]
                points.append(sample)
                velocity.append(ubasis @ arrays[f"velocity_{macro}"][udofs[cell]])
                pressure.append(pbasis @ arrays[f"pressure_{macro}"][pdofs[cell]])
        macro_mesh = TriangleMesh(arrays["macro_points"], arrays["macro_cells"])
        for a, b in macro_mesh.points[macro_mesh.faces]:
            if a[axis] != b[axis]:
                parameter = (level - a[axis]) / (b[axis] - a[axis])
                if 0 <= parameter <= 1:
                    intersections.append(float((a + parameter * (b - a))[1 - axis]))
        coordinates = np.asarray(points)
        refu, refp = reference.evaluate(coordinates.reshape(-1, 2))
        output.update(
            {
                f"profile_{label}_points": coordinates,
                f"profile_{label}_velocity": np.asarray(velocity),
                f"profile_{label}_pressure": np.asarray(pressure),
                f"profile_{label}_reference_velocity": refu.reshape(*coordinates.shape[:-1], 2),
                f"profile_{label}_reference_pressure": refp.reshape(coordinates.shape[:-1]),
                f"profile_{label}_macro_intersections": np.unique(intersections),
            }
        )
    return output


def main() -> None:
    """Compare all four final adaptive cavity states using recorded coefficient checksums."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lid", choices=["regularized", "constant"], default="regularized")
    parser.add_argument("--macro-configuration", default="")
    parser.add_argument("--face-configuration", default="")
    parser.add_argument("--drag", type=int, nargs="+", default=[0, 10000])
    parser.add_argument(
        "--strategy", nargs="+", choices=["macro", "face"], default=["macro", "face"]
    )
    parser.add_argument("--history-stride", type=int, default=2)
    args = parser.parse_args()
    prefix = "cavity-constant" if args.lid == "constant" else "cavity"
    cutout = 1 / 32 if args.lid == "constant" else 0.0
    if args.history_stride < 1:
        parser.error("history stride must be positive")
    for drag in args.drag:
        candidates = []
        for path in DATA.glob(f"{prefix}-classical-gamma{drag:g}-n*.json"):
            record = json.loads(path.read_text())
            candidates.append((record["n"], path, record))
        if len(candidates) < 3:
            raise ValueError("at least three conforming reference resolutions are required")
        _, reference_path, record = max(candidates, key=lambda value: value[0])
        path = ROOT / record["archive"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError("classical coefficient digest mismatch")
        with np.load(path) as saved:
            reference = TaylorHoodField(
                saved["velocity"], saved["pressure"], tuple(saved["bounds"])
            )
        for strategy in args.strategy:
            name = f"{prefix}-{strategy}-nu1-g{drag:g}-l0"
            configuration = (
                args.macro_configuration if strategy == "macro" else args.face_configuration
            )
            if configuration:
                name += "-" + configuration
            row = json.loads((DATA / (name + ".json")).read_text())
            path = ROOT / row["coefficients"]
            if hashlib.sha256(path.read_bytes()).hexdigest() != row["coefficients_sha256"]:
                raise ValueError("MHM coefficient digest mismatch")
            with np.load(path) as saved:
                arrays = dict(saved)
            with threadpool_limits(1):
                measured = {
                    str(order): norms(arrays, reference, order, drag=drag, corner_cutout=cutout)
                    for order in (12, 20, 28)
                }
                global_velocity = (
                    {
                        str(order): {
                            key: value
                            for key, value in norms(arrays, reference, order, drag=drag).items()
                            if key in ("velocity_l2", "velocity_reference_l2", "velocity_relative")
                        }
                        for order in (12, 20, 28)
                    }
                    if cutout
                    else None
                )
                history = []
                saved_history = row.get("coefficient_history", [])
                indices = sorted(
                    set(range(0, len(saved_history), args.history_stride))
                    | ({len(saved_history) - 1} if saved_history else set())
                )
                for state in indices:
                    saved_state = saved_history[state]
                    state_path = ROOT / saved_state["path"]
                    if hashlib.sha256(state_path.read_bytes()).hexdigest() != saved_state["sha256"]:
                        raise ValueError("adaptive history coefficient digest mismatch")
                    with np.load(state_path) as saved:
                        state_arrays = dict(saved)
                    history.append(
                        dict(
                            state=state,
                            trace_and_coarse_dofs=row["rows"][state]["trace_and_coarse_dofs"],
                            fine_cells=row["rows"][state]["fine_cells"],
                            coefficients_sha256=saved_state["sha256"],
                            norms={
                                str(order): norms(
                                    state_arrays,
                                    reference,
                                    order,
                                    drag=drag,
                                    corner_cutout=cutout,
                                )
                                for order in (12, 20)
                            },
                        )
                    )
            with np.load(DATA / row["archive"]) as saved:
                display = dict(saved)
            if hashlib.sha256((DATA / row["archive"]).read_bytes()).hexdigest() != row["sha256"]:
                raise ValueError("MHM display archive digest mismatch")
            refu, refp = reference.evaluate(display["points"])
            display.update(reference_velocity=refu, reference_pressure=refp)
            display.update(profiles(arrays, reference))
            if cutout:
                points, cells, offset = [], [], 0
                for macro in range(len(arrays["macro_cells"])):
                    local_points = arrays[f"points_{macro}"]
                    points.append(local_points)
                    cells.append(arrays[f"cells_{macro}"] + offset)
                    offset += len(local_points)
                display.update(fine_points=np.vstack(points), fine_cells=np.vstack(cells))
            output = DATA / (name + "-comparison.npz")
            np.savez_compressed(output, **display)
            report = dict(
                case=f"{args.lid} cavity: MHM versus conforming Taylor-Hood",
                lid=args.lid,
                corner_cutout=cutout,
                strategy=strategy,
                drag=drag,
                reference=reference_path.name,
                reference_sha256=hashlib.sha256(reference_path.read_bytes()).hexdigest(),
                reference_coefficients_sha256=record["sha256"],
                mhm_coefficients_sha256=row["coefficients_sha256"],
                norms=measured,
                global_velocity_norms=global_velocity,
                history=history,
                integration=(
                    "MHM fine triangles; Gaussian order study resolves intersections "
                    "with the continuous reference"
                ),
                archive=output.name,
                sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            )
            (DATA / (name + "-comparison.json")).write_text(json.dumps(report, indent=2) + "\n")
            print(name, json.dumps(measured), flush=True)


if __name__ == "__main__":
    main()
