"""Archive resource and physical-field identity contracts for scientific replay."""

from collections import Counter

import numpy as np
import pytest

from examples import compare_pgmhm_inclusions as inclusion
from examples import solve_pgmhm_spe10 as pgmhm
from examples import solve_unusual_spe10 as unusual
from examples import spe10_adaptive_norms as adaptive
from examples.archive_precision import precision_fields
from pymhm.fem.hdiv.rt import rt_interpolate
from pymhm.fem.scalar.triangle import nodal_space
from pymhm.meshes.refinement import refine_triangles
from pymhm.meshes.triangle import TriangleMesh


def field_archive(tmp_path, kind, *, ragged=True):
    """Store independent affine fields on unequal local partitions and portable components."""
    macro = TriangleMesh.unit_square()
    meshes = tuple(TriangleMesh(macro.points[cell], np.array([[0, 1, 2]])) for cell in macro.cells)
    if ragged:
        meshes = (meshes[0], refine_triangles(meshes[1], np.array([True])).mesh)
    degree = 1 if kind == "unusual" else 2
    coordinates = tuple(mesh.points if degree == 1 else nodal_space(mesh, 2)[1] for mesh in meshes)
    coefficients = tuple(p[:, 0] + 2 * p[:, 1] + i for i, p in enumerate(coordinates))
    arrays = dict(macro_points=macro.points, macro_cells=macro.cells)
    if ragged:
        arrays.update(
            local_points=np.concatenate([mesh.points for mesh in meshes]),
            local_cells=np.concatenate([mesh.cells for mesh in meshes]),
            point_offsets=np.r_[0, np.cumsum([len(mesh.points) for mesh in meshes])],
            cell_offsets=np.r_[0, np.cumsum([len(mesh.cells) for mesh in meshes])],
            coefficient_offsets=np.r_[0, np.cumsum([len(c) for c in coefficients])],
        )
        values = np.concatenate(coefficients)
    else:
        arrays.update(
            local_points=np.stack([mesh.points for mesh in meshes]),
            local_cells=np.stack([mesh.cells for mesh in meshes]),
        )
        values = np.stack(coefficients)
    if kind == "unusual":
        arrays["pressure"] = values
    elif kind == "pgmhm":
        arrays["component"] = np.array("kx")
        for multiplier, name in enumerate(("pressure", "enriched_pressure"), 1):
            arrays.update(precision_fields(name, multiplier * values.astype(np.longdouble)))
    elif kind == "inclusion":
        for multiplier, name in enumerate(("mhm", "pgmhm", "enriched"), 1):
            arrays.update(precision_fields(name, multiplier * values.astype(np.longdouble)))
    else:
        fluxes = tuple(rt_interpolate(mesh, (0.0, 0.0), 2) for mesh in meshes)
        if ragged:
            arrays.update(
                pressure_offsets=arrays["coefficient_offsets"],
                flux_offsets=np.r_[0, np.cumsum([len(q) for q in fluxes])],
            )
            arrays.update(precision_fields("pressure", values.astype(np.longdouble)))
            arrays.update(precision_fields("reconstructed_flux", np.concatenate(fluxes)))
        else:
            arrays.update(pressure=values, reconstructed_flux=np.stack(fluxes))
    path = tmp_path / (kind + ".npz")
    np.savez_compressed(path, **arrays)
    return path, arrays


class CountedArchive:
    """Observe actual decompressions and closure without changing NumPy's array values."""

    def __init__(self, archive):
        self.archive = archive
        self.reads = Counter()
        self.closed = False

    def __contains__(self, key):
        return key in self.archive

    def __getitem__(self, key):
        self.reads[key] += 1
        return self.archive[key]

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.archive.close()
        self.closed = True


@pytest.mark.parametrize("kind", ["unusual", "pgmhm", "inclusion", "adaptive"])
def test_ragged_arrays_are_read_once_and_preserve_broken_fields(tmp_path, monkeypatch, kind):
    """Local partition count cannot multiply retained full coefficient vectors."""
    path, arrays = field_archive(tmp_path, kind)
    original_load = np.load
    opened = CountedArchive(original_load(path))
    monkeypatch.setattr(np, "load", lambda *args, **kwargs: opened)

    def tensor(points):
        """Use constant diffusion to expose physical field values independently of material data."""
        return np.broadcast_to(2 * np.eye(2), (len(points), 2, 2))

    monkeypatch.setattr(unusual, "load_layer", lambda: tensor)
    monkeypatch.setattr(adaptive, "load_layer", lambda: tensor)
    monkeypatch.setattr(pgmhm, "load_material", lambda component: lambda x: np.full(len(x), 2.0))
    cls = {
        "unusual": unusual.UnusualSPE10Field,
        "pgmhm": pgmhm.PGMHMField,
        "inclusion": inclusion.InclusionMHMField,
        "adaptive": adaptive.BrokenP2,
    }[kind]
    field = cls(path)
    assert opened.closed
    assert opened.reads["local_points"] == opened.reads["local_cells"] == 1
    names = ("mhm", "pgmhm", "enriched") if kind == "inclusion" else ("pressure",)
    assert all(opened.reads[name] == 1 for name in names)
    for owner, mesh in enumerate(field.meshes):
        points = mesh.points[mesh.cells].mean(axis=1)
        exact = points[:, 0] + 2 * points[:, 1] + owner
        if kind == "unusual":
            actual, gradient, flux = field.evaluate_local(owner, points, np.arange(len(points)))
            np.testing.assert_allclose(actual, exact, rtol=0, atol=2e-15)
            np.testing.assert_allclose(gradient, np.broadcast_to([1, 2], gradient.shape))
            np.testing.assert_allclose(flux, np.broadcast_to([-2, -4], flux.shape))
        elif kind == "adaptive":
            actual, flux, reconstructed = field.evaluate_local(owner, points)
            np.testing.assert_allclose(actual, exact, rtol=0, atol=2e-15)
            np.testing.assert_allclose(flux, np.broadcast_to([-2, -4], flux.shape))
            np.testing.assert_array_equal(reconstructed, np.zeros_like(reconstructed))
        else:
            for cell, point in enumerate(points):
                actual, _ = field.evaluate(owner, cell, point[None])
                np.testing.assert_allclose(
                    actual[:, 0], np.arange(1, len(actual) + 1) * exact[cell], rtol=0, atol=4e-15
                )
    if kind == "unusual":
        assert field.pressure[0].base is field.pressure[1].base
        assert field.pressure[0].base.nbytes == arrays["pressure"].nbytes
        np.testing.assert_array_equal(np.concatenate(field.pressure), arrays["pressure"])


@pytest.mark.parametrize("kind", ["pgmhm", "adaptive"])
def test_rectangular_archive_branch_closes_and_preserves_precision(tmp_path, monkeypatch, kind):
    """Existing fixed-size archives retain their array values and native dtypes."""
    path, arrays = field_archive(tmp_path, kind, ragged=False)
    opened = CountedArchive(np.load(path))
    monkeypatch.setattr(np, "load", lambda *args, **kwargs: opened)
    monkeypatch.setattr(adaptive, "load_layer", lambda: None)
    monkeypatch.setattr(pgmhm, "load_material", lambda component: None)
    field = (pgmhm.PGMHMField if kind == "pgmhm" else adaptive.BrokenP2)(path)
    pressure = field.fields["pressure"] if kind == "pgmhm" else field.pressure
    np.testing.assert_array_equal(np.stack(pressure), arrays["pressure"])
    assert np.stack(pressure).dtype == (
        np.dtype(np.longdouble) if kind == "pgmhm" else arrays["pressure"].dtype
    )
    assert opened.closed and opened.reads["pressure"] == 1


def test_broken_archive_is_closed_when_geometry_is_rejected(tmp_path, monkeypatch):
    """An invalid archive must not leak an open ZIP descriptor through its failure path."""
    path = tmp_path / "invalid.npz"
    np.savez(path, macro_points=np.zeros((3, 2)), macro_cells=np.array([[0, 1, 2]]))
    opened = CountedArchive(np.load(path))
    monkeypatch.setattr(np, "load", lambda *args, **kwargs: opened)
    with pytest.raises(ValueError, match="degenerate"):
        adaptive.BrokenP2(path)
    assert opened.closed
