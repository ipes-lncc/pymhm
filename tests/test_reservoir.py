"""Reservoir ordering, discontinuous evaluation, units and acquisition integrity."""

import hashlib
from io import BytesIO

import numpy as np
import pytest

import pymhm.io.datasets.spe10 as reservoir
import pymhm.io.reservoir as property_io
from pymhm.io.datasets.spe10 import download_spe10_model2, load_spe10_model2
from pymhm.io.reservoir import ReservoirData, read_eclipse_properties
from pymhm.materials.cartesian import CartesianCellField


def test_cartesian_index_order_interfaces_endpoints_and_copy():
    values = np.arange(24).reshape(2, 3, 4)
    field = CartesianCellField(values, (2, 3, 4), (1, -2, 3))
    points = [[1, -2, 3], [5, 7, 19], [3, 1, 7], [3 - 1e-10, 1 - 1e-10, 7 - 1e-10]]
    np.testing.assert_array_equal(field(points), [values[0, 0, 0], 23, values[1, 1, 1], 0])
    values[:] = -10
    assert field.values[1, 2, 3] == 23
    assert not field.values.flags.writeable
    assert field(np.empty((0, 3))).shape == (0,)


def test_cartesian_tensor_values_and_boundary_roundoff():
    tensors = np.arange(24).reshape(2, 3, 2, 2)
    field = CartesianCellField(tensors, (2, 3))
    np.testing.assert_array_equal(field([[3, 8]])[0], tensors[1, 2])
    np.testing.assert_array_equal(field([[-1e-15, 9 + 1e-15]])[0], tensors[0, 2])


@pytest.mark.parametrize("spacing", [(1,), (1, 2, 3, 4), (0, 1), (np.inf, 2), (-1, 2)])
def test_cartesian_rejects_spacing(spacing):
    with pytest.raises(ValueError, match="spacing"):
        CartesianCellField(np.ones((2, 2)), spacing)


@pytest.mark.parametrize("origin", [(1,), (1, np.nan)])
def test_cartesian_rejects_origin(origin):
    with pytest.raises(ValueError, match="origin"):
        CartesianCellField(np.ones((2, 2)), (1, 1), origin)


@pytest.mark.parametrize(
    "values", [np.ones(2), np.ones((0, 2)), [["x"]], [[1j]], [[np.nan]], [[np.inf]]]
)
def test_cartesian_rejects_values(values):
    with pytest.raises(ValueError, match="values"):
        CartesianCellField(values, (1, 1))


@pytest.mark.parametrize("points", [[1, 2], [[1, 2, 3]], [[np.nan, 0]], [[np.inf, 0]]])
def test_cartesian_rejects_points(points):
    with pytest.raises(ValueError, match="points"):
        CartesianCellField(np.ones((2, 2)), (1, 1))(points)


@pytest.mark.parametrize("points", [[[-1e-5, 0]], [[2.0001, 0]], [[0, 2.001]]])
def test_cartesian_rejects_extrapolation(points):
    with pytest.raises(ValueError, match="outside"):
        CartesianCellField(np.ones((2, 2)), (1, 1))(points)


def test_eclipse_comments_repetitions_d_exponents_and_order(tmp_path):
    path = tmp_path / "properties.inc"
    path.write_text("-- comment\nPERMX\n 1 2*3.5 4D-2 -- ignore\n /\nPORO .2 2*.4 5d-1 /")
    props = read_eclipse_properties(path)
    np.testing.assert_array_equal(props["PERMX"], [1, 3.5, 3.5, 0.04])
    np.testing.assert_array_equal(props["PORO"], [0.2, 0.4, 0.4, 0.5])


@pytest.mark.parametrize(
    "text,message",
    [
        ("PERMX 1 2", "slash"),
        ("/", "keyword"),
        ("perm 1 /", "keyword"),
        ("PERM /", "nonempty"),
        ("PERM 1 / PERM 2 /", "unique"),
        ("PERM 2bad /", "token"),
        ("PERM 2* /", "token"),
        ("PERM 0*3 /", "positive"),
        ("PERM 1e400 /", "finite"),
        ("-- no properties\n", "no numeric"),
    ],
)
def test_eclipse_rejects_invalid_data(tmp_path, text, message):
    path = tmp_path / "bad.inc"
    path.write_text(text)
    with pytest.raises(ValueError, match=message):
        read_eclipse_properties(path)


def test_reservoir_layer_number_tensor_porosity_and_units():
    permeability = 1 + np.arange(2 * 3 * 4 * 3).reshape(2, 3, 4, 3)
    porosity = np.linspace(0, 1, 24).reshape(2, 3, 4)
    data = ReservoirData(permeability, porosity)
    k, phi = data.layer(2)
    np.testing.assert_array_equal(k.values[..., 0, 0], permeability[:, :, 1, 0])
    np.testing.assert_array_equal(k.values[..., 1, 1], permeability[:, :, 1, 1])
    np.testing.assert_array_equal(k.values[..., 0, 1], 0)
    np.testing.assert_array_equal(phi.values, porosity[:, :, 1])
    assert k.spacing == (20, 10)
    si = data.to_si()
    np.testing.assert_allclose(si.permeability, permeability * 9.869233e-16, rtol=1e-15, atol=0)
    np.testing.assert_allclose(si.spacing, np.array([20, 10, 2]) * 0.3048)
    np.testing.assert_array_equal(si.porosity, porosity)
    assert (si.permeability_unit, si.length_unit) == ("m2", "m")
    assert si.to_si() is si
    permeability[:] = 0
    porosity[:] = 0
    assert data.permeability.min() == 1
    assert data.porosity.max() == 1
    assert not data.permeability.flags.writeable
    assert not data.porosity.flags.writeable


@pytest.mark.parametrize("layer", [0, 1.5, True, 5])
def test_reservoir_rejects_layer(layer):
    data = ReservoirData(np.ones((2, 3, 4, 3)), np.ones((2, 3, 4)))
    with pytest.raises(ValueError, match="layer"):
        data.layer(layer)


@pytest.mark.parametrize(
    "k,phi,kwargs,message",
    [
        (np.ones((2, 2, 2, 2)), np.ones((2, 2, 2)), {}, "shapes"),
        (np.ones((2, 2, 2, 2, 3)), np.ones((2, 2, 2, 2)), {}, "shapes"),
        (np.ones((2, 2, 2, 3)), np.ones((2, 2, 2)), {"spacing": (1, 1)}, "shapes"),
        (np.zeros((2, 2, 2, 3)), np.ones((2, 2, 2)), {}, "positive"),
        (np.ones((2, 2, 2, 3)), -np.ones((2, 2, 2)), {}, "porosity"),
        (np.ones((2, 2, 2, 3)), 2 * np.ones((2, 2, 2)), {}, "porosity"),
        (np.ones((2, 2, 2, 3)), np.ones((2, 2, 2)), {"length_unit": ""}, "units"),
        (np.ones((2, 2, 2, 3)), np.ones((2, 2, 2)), {"permeability_unit": ""}, "units"),
    ],
)
def test_reservoir_rejects_invalid_data(k, phi, kwargs, message):
    with pytest.raises(ValueError, match=message):
        ReservoirData(k, phi, **kwargs)


def test_reservoir_rejects_unrecognized_conversion():
    data = ReservoirData(np.ones((1, 1, 1, 3)), np.ones((1, 1, 1)), length_unit="cm")
    with pytest.raises(ValueError, match="conversion"):
        data.to_si()


def test_spe10_loader_shape_and_i_fastest_order(monkeypatch, tmp_path):
    shape = (60, 220, 85)
    index = np.arange(np.prod(shape), dtype=float)
    mapping = {"PERMX": index + 1, "PERMY": index + 2, "PERMZ": index + 3}
    monkeypatch.setattr(
        property_io,
        "read_eclipse_properties",
        lambda path: {"PORO": index / index.max()} if "PHI" in path.name else mapping,
    )
    data = load_spe10_model2(tmp_path)
    assert data.permeability.shape == shape + (3,)
    assert data.permeability[0, 0, 0, 0] == 1
    assert data.permeability[1, 0, 0, 0] == 2
    assert data.permeability[0, 1, 0, 0] == 61
    assert data.permeability[0, 0, 1, 0] == 13201
    assert data.porosity[-1, -1, -1] == 1


@pytest.mark.parametrize("keys", [("PERMX",), ("PERMX", "PERMY", "PERMZ", "PORO")])
def test_spe10_loader_rejects_missing_or_short_properties(monkeypatch, tmp_path, keys):
    monkeypatch.setattr(
        property_io, "read_eclipse_properties", lambda path: {key: np.ones(2) for key in keys}
    )
    with pytest.raises(ValueError, match="60"):
        load_spe10_model2(tmp_path)


def test_download_is_pinned_validated_and_reuses_cache(monkeypatch, tmp_path):
    content = b"PERMX 1 /"
    checksum = hashlib.sha256(content).hexdigest()
    monkeypatch.setattr(reservoir, "SPE10_FILES", {"a.inc": checksum, "b.inc": checksum})
    urls = []

    def request(url, timeout):
        urls.append(url)
        assert timeout > 0
        return BytesIO(content)

    monkeypatch.setattr(reservoir, "urlopen", request)
    assert download_spe10_model2(tmp_path) == tmp_path
    assert len(urls) == 2
    assert all(reservoir.SPE10_REVISION in url for url in urls)
    download_spe10_model2(tmp_path)
    assert len(urls) == 2
    assert sorted(p.name for p in tmp_path.iterdir()) == ["a.inc", "b.inc"]
    (tmp_path / "a.inc").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="cached"):
        download_spe10_model2(tmp_path)


def test_download_corruption_cleans_temporary_file(monkeypatch, tmp_path):
    monkeypatch.setattr(reservoir, "SPE10_FILES", {"a.inc": "wrong"})
    monkeypatch.setattr(reservoir, "urlopen", lambda *a, **kw: BytesIO(b"corrupt"))
    with pytest.raises(ValueError, match="downloaded"):
        download_spe10_model2(tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_download_failure_before_temporary_creation(monkeypatch, tmp_path):
    def request(*args, **kwargs):
        raise OSError("offline")

    monkeypatch.setattr(reservoir, "urlopen", request)
    with pytest.raises(OSError, match="offline"):
        download_spe10_model2(tmp_path)
    assert list(tmp_path.iterdir()) == []
