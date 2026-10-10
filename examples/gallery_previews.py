"""Render single-field Gallery previews from verified executed application archives.

These functions perform no assembly or solve. Broken samples keep their own
connectivity, and every preview overlays the actual acquired macro partition.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from threadpoolctl import threadpool_limits

from examples.nested_field_archive import array_digest
from examples.plot_mesh import draw_macro_mesh
from pymhm import TriangleMesh

if TYPE_CHECKING:
    from matplotlib.axes import Axes

ROOT = Path(__file__).resolve().parents[1]


def _digest(path: Path) -> str:
    """Return the SHA-256 digest of literal input or output bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_record(
    output: Path, archive: Path, record: dict[str, Any], sources: Sequence[Path]
) -> dict[str, Any]:
    """Bind the preview to its executed archive and current evaluation owners."""
    record.update(
        schema="pymhm-gallery-preview-v1",
        input_archive=archive.relative_to(ROOT).as_posix(),
        input_archive_sha256=_digest(archive),
        figure_sha256=_digest(output),
        source_sha256={
            path.relative_to(ROOT).as_posix(): _digest(path)
            for path in (Path(__file__).resolve(), *sources)
        },
        attribution="IPES Research Group",
        license="CC-BY-4.0",
        acquisition=record.get(
            "acquisition", "Replay of existing executed fields; no numerical solve"
        ),
    )
    output.with_suffix(".json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record


def plot_preview(
    output: Path,
    draw: Callable[[Axes], None],
    *,
    macro: TriangleMesh | None = None,
    edge_layers: Sequence[tuple[np.ndarray, str, float]] = (),
    bounds: tuple[float, float, float, float] = (0, 1, 0, 1),
) -> None:
    """Render one field at equal physical aspect with mesh edges and no card text.

    ``draw`` supplies the existing physical samples without welding their
    connectivity. ``edge_layers`` can overlay several acquired hierarchy levels.
    The figure contains the entire domain; it is not cropped from a larger plot.
    """
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    horizontal = abs(bounds[1] - bounds[0])
    vertical = abs(bounds[3] - bounds[2])
    if not np.isfinite(bounds).all() or min(horizontal, vertical) <= 0:
        raise ValueError("finite nondegenerate physical bounds are required")
    figure, axis = plt.subplots(figsize=(5, 5 * vertical / horizontal))
    figure.subplots_adjust(left=0, right=1, bottom=0, top=1)
    draw(axis)
    if macro is not None:
        edges = draw_macro_mesh(axis, macro)
        edges.set_path_effects([])
        edges.set_linewidth(max(0.08, min(0.6, 3 / np.sqrt(len(macro.faces)))))
        edges.set_alpha(0.45)
    for edges, color, width in edge_layers:
        axis.add_collection(LineCollection(edges, colors=color, linewidths=width, zorder=3))
    axis.set(xlim=bounds[:2], ylim=bounds[2:], aspect="equal")
    axis.axis("off")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, facecolor="white")
    plt.close(figure)


def recursive_pressure(output: Path) -> dict[str, Any]:
    """Replay recursive Q2 pressure using its saved one-sided sampling tables.

    The displayed n=4 case has 16 outer quadrilaterals, 64 inner leaves and
    four Q2 fine cells per leaf. Both hierarchy levels use their archived faces.
    """
    from examples.nested_field_archive import display_fields, read_archive
    from examples.plot_nested import sample_edges

    record_path = ROOT / "examples/results/nested.json"
    published = json.loads(record_path.read_text())
    selected = published["display_field"]
    archive = record_path.parent / selected["archive"]
    if _digest(archive) != selected["archive_sha256"]:
        raise ValueError("recursive field archive does not match the published record")
    acquisition, arrays = read_archive(archive)
    points, pressure, _ = display_fields(arrays, kind="recursive")
    if acquisition["n"] != 4 or acquisition["boundary_case"] != "sine_zero_dirichlet":
        raise ValueError("the verified n=4 homogeneous recursive field is required")
    limits = float(pressure.min()), float(pressure.max())

    def draw(axis: Axes) -> None:
        """Color bounded pixels from each independent saved 17-by-17 leaf."""
        for coordinates, samples in zip(points, pressure, strict=True):
            physical = coordinates.reshape(17, 17, 2)
            axis.pcolormesh(
                sample_edges(physical[0, :, 0]),
                sample_edges(physical[:, 0, 1]),
                samples.reshape(17, 17),
                shading="flat",
                cmap="viridis",
                vmin=limits[0],
                vmax=limits[1],
                rasterized=True,
            )

    plot_preview(
        output,
        draw,
        edge_layers=(
            (arrays["flat_face_endpoints"], ".50", 0.5),
            (arrays["outer_face_endpoints"], ".12", 1.0),
        ),
    )
    return _write_record(
        output,
        archive,
        {
            "field": "Recursive MHM pressure",
            "local_space": published["local_space"],
            "trace_space": published["trace"],
            "hierarchy": "16 outer macroquadrilaterals; 64 inner leaves; 256 fine quadrilaterals",
            "basis": "Saved executed Q2 nodal injection and physical value tables",
            "one_sided_values": "Independent bounded leaf pixels; no macroface averaging",
            "sample_points_sha256": array_digest(points),
            "sample_values_sha256": array_digest(pressure),
            "color_range": limits,
        },
        (record_path, ROOT / "examples/nested_field_archive.py", ROOT / "examples/plot_nested.py"),
    )


def obstacle_flux(output: Path) -> dict[str, Any]:
    """Render archived cellwise Darcy flux magnitude around the square obstacle.

    The mixed RT0/P0 MHM case uses 200 macrotriangles, 51,200 fine triangles
    and two constant physical-flux trace segments per macroface. Values remain
    physical cell samples; power normalization resolves the low-flux square
    without clipping the complete acquired magnitude range.
    """
    from matplotlib.colors import PowerNorm
    from matplotlib.patches import Rectangle
    from matplotlib.tri import Triangulation

    from examples.plot_quarter_reference import load_record
    from examples.quarter_spot_problem import OBSTACLE_LOWER, OBSTACLE_SIDE

    record_path = ROOT / "examples/results/quarter-five-spot/reference/classical-convergence.json"
    published = json.loads(record_path.read_text())
    row = next(row for row in published["mhm_trace_enrichment"] if row["segments"] == 2)
    arrays = load_record(row)
    archive = record_path.parent / row["fields"]
    points, cells = arrays["points"], arrays["cells"]
    magnitude = np.linalg.norm(arrays["pymhm_flux"], axis=1)
    limits = float(magnitude.min()), float(magnitude.max())
    macro = TriangleMesh(arrays["macro_points"], arrays["macro_cells"])

    def draw(axis: Axes) -> None:
        """Color exact archived fine-cell values and mark the material boundary."""
        axis.tripcolor(
            Triangulation(points[:, 0], points[:, 1], cells),
            facecolors=magnitude,
            shading="flat",
            norm=PowerNorm(0.35, vmin=limits[0], vmax=limits[1]),
            cmap="viridis",
            rasterized=True,
        )
        axis.add_patch(
            Rectangle(
                (OBSTACLE_LOWER, OBSTACLE_LOWER),
                OBSTACLE_SIDE,
                OBSTACLE_SIDE,
                fill=False,
                edgecolor="white",
                linewidth=1.4,
                zorder=4,
            )
        )

    plot_preview(output, draw, macro=macro)
    return _write_record(
        output,
        archive,
        {
            "field": "MHM physical Darcy flux magnitude",
            "local_space": "RT0 flux / P0 pressure on 51200 fitted fine triangles",
            "trace_space": "Two P0 physical normal-flux segments per macroface",
            "macro_space": "200 macrotriangles on the actual archived partition",
            "basis": "Already executed physical RT0 cell samples; no coefficient reinterpretation",
            "sample_values_sha256": array_digest(magnitude),
            "one_sided_values": "Cellwise constant display of the stored fine-cell samples",
            "color_scale": "Power normalization with exponent 0.35; full magnitude range",
            "color_range": limits,
        },
        (
            record_path,
            ROOT / "examples/plot_quarter_reference.py",
            ROOT / "examples/quarter_spot_problem.py",
            ROOT / "examples/plot_mesh.py",
        ),
    )


def brinkman_velocity(output: Path, exact: Any) -> dict[str, Any]:
    """Replay and validate the n=64 single-element MHM-USFEM velocity preview.

    Supply the analytical ``BrinkmanLayer`` object declared in the introductory
    notebook. The saved source/lifts/retained matrices are replayed literally.
    Recomputed physical field norms must match the independent published values
    before the existing Basix evaluator is allowed to produce display samples.
    """
    from matplotlib.tri import Triangulation

    from examples.introduction.vector import display_samples, flow_error_norms

    record_path = ROOT / "examples/results/introduction-layers/stokes-brinkman-boundary-layer.json"
    published = json.loads(record_path.read_text())
    identity = next(
        row
        for row in published["state_archives"]
        if row["path"] == "build/introduction/brinkman-single-ell2-n64-state.npz"
    )
    archive = ROOT / identity["path"]
    if _digest(archive) != identity["sha256"]:
        raise ValueError("Brinkman state does not match the published executed archive")
    expected = next(
        row for row in published["single_element_family"] if row["n"] == 64 and row["ell"] == 2
    )
    meshes, velocities, pressures = [], [], []
    basis_digest = hashlib.sha256()
    with np.load(archive) as saved, threadpool_limits(1):
        macro = TriangleMesh(saved["macro_points"], saved["macro_cells"])
        trace = saved["trace"]
        for cell in range(len(macro.cells)):
            basis = saved[f"basis_{cell}"]
            basis_digest.update(str(basis.shape).encode())
            basis_digest.update(basis.tobytes())
            coefficients = (
                saved[f"source_{cell}"]
                - saved[f"lifts_{cell}"] @ trace[saved[f"trace_dofs_{cell}"]]
                + basis @ saved[f"coarse_{cell}"]
            )
            np.testing.assert_allclose(coefficients, saved[f"field_{cell}"], rtol=1e-11, atol=1e-12)
            if (
                int(saved[f"velocity_degree_{cell}"]) != 4
                or int(saved[f"pressure_degree_{cell}"]) != 4
            ):
                raise ValueError("the verified P4/P4 archive is required")
            size = 2 * int(saved[f"velocity_nodes_{cell}"])
            meshes.append(TriangleMesh(saved[f"local_points_{cell}"], saved[f"local_cells_{cell}"]))
            velocities.append(coefficients[:size].reshape(-1, 2))
            pressures.append(coefficients[size:])
        if basis_digest.hexdigest() != identity["executed_basis_sha256"]:
            raise ValueError("executed retained-basis digest differs")
        metrics = flow_error_norms(
            meshes, velocities, pressures, 4, 4, exact, order=published["error_duffy_order"]
        )
        for name in (
            "velocity_l2",
            "pressure_l2",
            "velocity_h1_seminorm",
            "divergence_l2",
            "pseudostress_l2",
        ):
            np.testing.assert_allclose(metrics[name], expected[name], rtol=1e-8, atol=1e-12)
        samples = display_samples(meshes, velocities, 4, subdivisions=4)
    magnitude = np.linalg.norm(samples["values"], axis=1)
    limits = float(magnitude.min()), float(magnitude.max())

    def draw(axis: Axes) -> None:
        """Evaluate each macrotriangle independently without welding its traces."""
        points = samples["points"]
        axis.tripcolor(
            Triangulation(points[:, 0], points[:, 1], samples["cells"]),
            magnitude,
            shading="gouraud",
            cmap="viridis",
            vmin=limits[0],
            vmax=limits[1],
            rasterized=True,
        )

    plot_preview(output, draw, macro=macro)
    return _write_record(
        output,
        archive,
        {
            "field": "Single-element MHM-USFEM velocity magnitude",
            "local_space": "P4 velocity / P4 pressure; one fine triangle per macrotriangle",
            "trace_space": "Vector P2 multiplier on each macroface",
            "macro_space": "8192 macrotriangles on the actual n=64 partition",
            "basis": (
                "Literal saved source/lifts/retained-basis replay; "
                "interleaved velocity then pressure"
            ),
            "executed_basis_sha256": basis_digest.hexdigest(),
            "evaluator_validation": (
                "Physical norms reproduced against independent published archive measurements"
            ),
            "physical_norms": metrics,
            "sample_points_sha256": array_digest(samples["points"]),
            "sample_values_sha256": array_digest(magnitude),
            "one_sided_values": "Independent P4 polynomial display on every macrotriangle",
            "color_range": limits,
        },
        (
            record_path,
            ROOT / "examples/introduction/vector.py",
            ROOT / "examples/plot_mesh.py",
            ROOT / "notebooks/introduction/stokes_brinkman_boundary_layer.ipynb",
        ),
    )


def _rad_samples(exact: Any) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    """Replay the executed finest epsilon=1e-5 P1/P0 USFEM scalar field.

    ``exact`` is the notebook's independent ReactionLayer(1e-5) object.
    The current archive has its literal digest recorded separately from the
    older article receipt; physical norms and the retained basis are checked.
    """
    from examples.introduction.transport import scalar_error_norms
    from examples.introduction.vector import display_samples
    from pymhm import TriangleMesh

    archive = ROOT / "build/introduction/rad-resolved-eps1e-05-n16-mhm-usfem-state.npz"
    if _digest(archive) != "81ac5919e0ab61f924738d1d1f73e29a4a40c3ad9243d9a56fdc46567f818ff1":
        raise ValueError("The independently checked current RAD archive is required")
    published = json.loads(
        (ROOT / "examples/results/introduction-layers/mhm-usfem-rad.json").read_text()
    )
    expected = next(
        row
        for row in published["refined_convergence"]
        if row["n"] == 16 and row["method"] == "MHM-USFEM" and row["epsilon"] == 1e-5
    )
    identity = next(
        row
        for row in published["state_archives"]
        if row["path"] == archive.relative_to(ROOT).as_posix()
    )
    meshes, fields = [], []
    basis_digest = hashlib.sha256()
    with np.load(archive) as saved:
        macro = TriangleMesh(saved["macro_points"], saved["macro_cells"])
        if len(macro.cells) != 512 or len(saved["trace"]) != 3200:
            raise ValueError("The finest declared RAD partition is required")
        for cell in range(len(macro.cells)):
            mesh = TriangleMesh(saved[f"local_points_{cell}"], saved[f"local_cells_{cell}"])
            if int(saved[f"scalar_degree_{cell}"]) != 1 or len(mesh.cells) != 64:
                raise ValueError("The declared P1/eight-subdivision local mesh is required")
            basis = saved[f"basis_{cell}"]
            basis_digest.update(str(basis.shape).encode())
            basis_digest.update(basis.tobytes())
            for count in (1, 2):
                with threadpool_limits(count):
                    coefficients = (
                        saved[f"source_{cell}"]
                        - saved[f"lifts_{cell}"] @ saved["trace"][saved[f"trace_dofs_{cell}"]]
                        + basis @ saved[f"coarse_{cell}"]
                    )
                np.testing.assert_allclose(
                    coefficients, saved[f"field_{cell}"], rtol=1e-11, atol=1e-12
                )
            field = np.zeros(len(mesh.points))
            field[saved[f"free_nodes_{cell}"]] = coefficients
            meshes.append(mesh)
            fields.append(field)
    if basis_digest.hexdigest() != identity["executed_basis_sha256"]:
        raise ValueError("The actual retained basis differs from the original executed basis")
    measurements = {}
    with threadpool_limits(1):
        for order in (24, 32):
            values = scalar_error_norms(
                meshes, fields, 1, exact.value, exact.gradient, 1e-5, order=order
            )
            for name in ("scalar_l2", "flux_l2", "gradient_l2", "nodal_maximum", "nodal_minimum"):
                np.testing.assert_allclose(values[name], expected[name], rtol=1e-8, atol=1e-12)
            measurements[str(order)] = values
    samples = display_samples(meshes, fields, 1, subdivisions=2)
    return (
        macro,
        samples,
        {
            "field": "MHM-USFEM scalar u",
            "epsilon": 1e-5,
            "macro_triangles": 512,
            "fine_triangles": 32768,
            "local_space": "P1 on eight-subdivision red-refined local triangles",
            "trace_space": "Four P0 segments per macroface",
            "basis": (
                "Literal executed source, lifts and retained basis; portable free-node injection"
            ),
            "replay_blas_threads": [1, 2],
            "retained_basis_sha256": basis_digest.hexdigest(),
            "current_input_archive": archive.relative_to(ROOT).as_posix(),
            "current_input_archive_sha256": _digest(archive),
            "published_input_archive_sha256": identity["sha256"],
            "evaluator_validation": (
                "Physical fields reproduce the published norms at two quadrature orders"
            ),
            "physical_norms_by_duffy_order": measurements,
            "scalar_extrema": [float(samples["values"].min()), float(samples["values"].max())],
            "one_sided_values": "Independent original fine-cell triangles; no macroface averaging",
        },
    )


def rad_scalar(output: Path, exact: Any) -> dict[str, Any]:
    """Replay the finest severe-layer USFEM scalar as one independent field.

    Supply the introductory notebook's ``ReactionLayer(1e-5)`` object. The
    preview retains the actual 512 macrotriangles and checks physical scalar
    and raw-flux errors using two adequate Duffy quadratures before plotting.
    """
    from matplotlib.tri import Triangulation

    macro, samples, record = _rad_samples(exact)
    values = samples["values"]
    limits = float(values.min()), float(values.max())

    def draw(axis: Axes) -> None:
        """Show actual independent P1 triangles and their physical scalar values."""
        points = samples["points"]
        axis.tripcolor(
            Triangulation(points[:, 0], points[:, 1], samples["cells"]),
            values,
            shading="gouraud",
            cmap="viridis",
            vmin=limits[0],
            vmax=limits[1],
            rasterized=True,
        )

    plot_preview(output, draw, macro=macro)
    record.update(
        color_range=limits,
        sample_points_sha256=array_digest(samples["points"]),
        sample_values_sha256=array_digest(values),
    )
    return _write_record(
        output,
        ROOT / record["current_input_archive"],
        record,
        (
            ROOT / "examples/results/introduction-layers/mhm-usfem-rad.json",
            ROOT / "examples/introduction/vector.py",
            ROOT / "examples/introduction/transport.py",
            ROOT / "examples/plot_mesh.py",
            ROOT / "notebooks/introduction/mhm_usfem_rad.ipynb",
        ),
    )


def marmousi_velocity(output: Path) -> dict[str, Any]:
    """Display the pinned primary P-wave material on the H=20 m comparison grid.

    Primary SEG-Y files are read with the shared crop owner and verified
    mandatory digests. The entire 10,240 by 2,560 m crop retains its physical
    aspect and 5 m sample cells; the preview represents material, not a solved
    acoustic pressure field. Coordinates increase downwards in depth.
    """
    from examples.marmousi_data import load_marmousi_crop
    from pymhm.meshes.cartesian import CartesianMacroMesh

    material = load_marmousi_crop(ROOT / "build/datasets/marmousi")
    velocity = material.velocity
    if velocity.values.shape != (2048, 512) or velocity.spacing != (5.0, 5.0):
        raise ValueError("the complete declared primary material crop is required")
    macro = CartesianMacroMesh(512, 128, (0, 10240, 0, 2560))
    limits = float(velocity.values.min()), float(velocity.values.max())

    def draw(axis: Axes) -> None:
        """Display real primary velocity samples with faint real macro boundaries."""
        axis.imshow(
            velocity.values.T,
            origin="upper",
            extent=(0, 10240, 2560, 0),
            interpolation="nearest",
            cmap="viridis",
            vmin=limits[0],
            vmax=limits[1],
            rasterized=True,
        )
        lines = draw_macro_mesh(axis, macro)
        lines.set_path_effects([])
        lines.set_linewidth(0.07)
        lines.set_alpha(0.25)

    plot_preview(output, draw, bounds=(0, 10240, 2560, 0))
    return _write_record(
        output,
        ROOT / "build/datasets/marmousi/vp_marmousi-ii.segy",
        {
            "field": "Marmousi II P-wave velocity",
            "scope": "Primary material coefficient; no acoustic solution is represented",
            "material": material.provenance,
            "coordinate_bounds_m": [0, 10240, 0, 2560],
            "velocity_unit": "m/s",
            "color_range": limits,
            "macro_overlay": "H=20 m comparison partition; 512 by 128 actual macrorectangles",
            "array_shape": list(velocity.values.shape),
            "sample_values_sha256": array_digest(velocity.values),
            "display": "Nearest primary material samples; full physical aspect; depth downwards",
            "input_kind": "Pinned primary SEG-Y material file",
            "acquisition": "Checksum-verified primary material sampling; no numerical solve",
        },
        (
            ROOT / "examples/marmousi_data.py",
            ROOT / "src/pymhm/materials/cartesian.py",
            ROOT / "examples/plot_mesh.py",
        ),
    )


def heterogeneous_lame(output: Path, modulus: Callable[[np.ndarray], np.ndarray]) -> dict[str, Any]:
    """Plot the application notebook's exact Lamé material on its actual macro mesh.

    Supply ``micro_modulus`` declared in ``multiscale_elasticity.ipynb``.
    Here lambda=mu and the 1/8 wavelength is represented by independent
    macrotriangle samples. This thumbnail represents coefficients; the
    application's nine-panel result figure supplies computed physical stresses.
    """
    from matplotlib.tri import Triangulation

    from examples.introduction.vector import elasticity_panel

    macro = TriangleMesh.unit_square(4)
    points, cells, values = elasticity_panel(macro, None, "modulus", refinement=32, modulus=modulus)
    limits = float(values.min()), float(values.max())

    def draw(axis: Axes) -> None:
        """Draw the supplied exact coefficient with the application's real partition."""
        axis.tripcolor(
            Triangulation(points[:, 0], points[:, 1], cells),
            values,
            shading="gouraud",
            cmap="viridis",
            vmin=limits[0],
            vmax=limits[1],
            rasterized=True,
        )

    plot_preview(output, draw, macro=macro)
    notebook = ROOT / "notebooks/introduction/multiscale_elasticity.ipynb"
    return _write_record(
        output,
        notebook,
        {
            "field": "Exact Lamé material lambda=mu",
            "scope": (
                "Declared heterogeneous material coefficient; no computed stress is represented"
            ),
            "macro_triangles": 32,
            "wavelength": 0.125,
            "coefficient_contrast": float(np.exp(3)),
            "color_range": limits,
            "sample_points_sha256": array_digest(points),
            "sample_values_sha256": array_digest(values),
            "one_sided_values": "Independent samples on each actual macrotriangle",
            "input_kind": "Application notebook declaring the exact coefficient",
            "acquisition": (
                "Evaluation of the notebook-supplied exact Lamé material; no numerical solve"
            ),
        },
        (ROOT / "examples/introduction/vector.py", ROOT / "examples/plot_mesh.py"),
    )
