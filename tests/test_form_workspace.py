"""Portable native API contracts for reusable arbitrary UFL assembly workspaces."""

from __future__ import annotations

import pickle
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from numpy.testing import assert_array_equal
from scipy import sparse

from pymhm.backends import forms, workspace


class Space:
    """Unhashable native-like space, matching DOLFINx's function-space behavior."""

    __hash__ = None  # type: ignore[assignment]

    def __init__(self, mesh: Any, size: int = 2) -> None:
        """Retain a native mesh and owned scalar coefficient dimensions."""
        self._cpp_object = object()
        self.mesh = mesh
        self.dofmap = SimpleNamespace(
            index_map=SimpleNamespace(size_local=size),
            index_map_bs=1,
            list=np.arange(size).reshape(1, size),
        )

    def ufl_element(self) -> str:
        """Expose finite-element identity independently of its bound native mesh."""
        return "P1"

    def ufl_domain(self) -> Any:
        """Expose the native mesh cargo required by pre-JIT domain validation."""
        return SimpleNamespace(ufl_cargo=lambda: self.mesh, ufl_coordinate_element=lambda: "P1")


class Form:
    """Named arbitrary vector/matrix payload with changing coefficient dependence."""

    def __init__(
        self,
        space: Any,
        payload: Any,
        *,
        zero: bool = False,
        coefficients: Any = (),
        constants: Any = (),
        integrals: Any = None,
    ) -> None:
        """Declare ranks, source symbols and optional tagged integration entities."""
        self.space, self.payload, self.zero = space, np.asarray(payload), zero
        self.coefficients, self.constants = coefficients, constants
        self._integrals = integrals

    def arguments(self) -> tuple[Any, ...]:
        """Return ordered test/trial spaces, or no arguments for an erased zero."""
        return tuple(
            SimpleNamespace(ufl_function_space=lambda: self.space, number=lambda i=i: i)
            for i in range(self.payload.ndim)
        )

    def integrals(self) -> tuple[Any, ...]:
        """Expose zero and tagged integration protocols without a FEM implementation."""
        if self._integrals is not None:
            return self._integrals
        return (
            SimpleNamespace(
                integrand=lambda: 0 if self.zero else 1,
                subdomain_data=lambda: None,
                subdomain_id=lambda: "everywhere",
            ),
        )

    def ufl_domains(self) -> tuple[Any, ...]:
        """Expose a native single-rank form domain before compilation."""
        return (self.space.ufl_domain(),)

    def signature(self) -> str:
        """Represent structural rank and coefficient-layout compatibility."""
        return str((self.payload.shape, self.zero))


@pytest.fixture
def native_api(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Model data-independent compile/bind and accumulating native assembly APIs."""
    observed = SimpleNamespace(compilations=0, bindings=[], allocations=0, domains=[])
    geometry = SimpleNamespace(
        x=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        dim=2,
        cmap=SimpleNamespace(degree=1, variant=0),
        dofmap=np.array([[0, 1]]),
    )
    topology = SimpleNamespace(
        dim=2,
        cell_type="triangle",
        create_connectivity=lambda *x: None,
        connectivity=lambda *x: SimpleNamespace(array=np.array([0, 1]), offsets=np.array([0, 2])),
    )
    observed.mesh = SimpleNamespace(
        comm=SimpleNamespace(size=1), geometry=geometry, topology=topology
    )
    observed.space = Space(observed.mesh)
    observed.mesh.ufl_domain = lambda: SimpleNamespace(ufl_coordinate_element=lambda: "P1")
    observed.coefficient = SimpleNamespace()

    # UFL symbols must be hashable while their native data handles are arbitrary.
    class Symbol:
        """Hashable UFL-like symbol with a native-independent approximation space."""

        ufl_shape = ()

        def ufl_function_space(self) -> Any:
            """Return the coefficient's declared approximation space."""
            return observed.space

    observed.symbol, observed.constant = Symbol(), Symbol()
    observed.function = SimpleNamespace(
        _cpp_object=object(),
        function_space=observed.space,
        x=SimpleNamespace(array=np.array([1.0, 2.0]), scatter_forward=lambda: None),
    )
    observed.function.interpolate = lambda callback: observed.function.x.array.__setitem__(
        slice(None), callback(observed.mesh.geometry.x.T)
    )
    observed.native_constant = SimpleNamespace(
        _cpp_object=object(),
        value=np.array(1.0),
        ufl_shape=(),
    )

    def compile_native(comm: Any, form: Any, **options: Any) -> Any:
        """Count kernel compilation and copy options as a mutating native API would."""
        observed.compilations += 1
        options["form_compiler_options"]["mutated"] = True
        return SimpleNamespace(source=form)

    def bind(
        compiled: Any, spaces: Any, mesh: Any, domains: Any, coeffs: Any, consts: Any, **kw: Any
    ) -> Any:
        """Record only the symbols actually present in this form."""
        observed.bindings.append((spaces, domains, coeffs, consts, kw))
        return SimpleNamespace(source=compiled.source, mesh=mesh, coeffs=coeffs, consts=consts)

    def values(compiled: Any) -> Any:
        """Recompute payloads from current independent native data each time."""
        scale = sum(float(x.value) for x in compiled.consts.values()) or 1.0
        shift = sum(float(x.x.array.sum()) for x in compiled.coeffs.values())
        return compiled.source.payload * scale + shift

    class Buffer:
        """Native assembly owner supporting vector and CSR storage reuse."""

        def __init__(self, value: Any) -> None:
            """Allocate a native-like vector or matrix buffer."""
            observed.allocations += 1
            self.array = np.array(value, copy=True)
            self.matrix = sparse.csr_matrix(value) if self.array.ndim == 2 else None

        @property
        def data(self) -> Any:
            """Expose mutable native CSR data for explicit zero-before-assembly."""
            return self.matrix.data

        def scatter_reverse(self, *args: Any) -> None:
            """Model completion of owned single-rank assembly."""

        def to_scipy(self) -> Any:
            """Expose the matrix view whose data must be copied before returning."""
            return self.matrix

    def vector(first: Any, compiled: Any = None) -> Any:
        """Implement allocate or accumulate overloads of native vector assembly."""
        if compiled is None:
            return Buffer(values(first))
        first[:] += values(compiled)

    def matrix(first: Any, compiled: Any = None) -> Any:
        """Implement allocate or accumulate overloads of native sparse assembly."""
        if compiled is None:
            return Buffer(values(first))
        first.data[:] += sparse.csr_matrix(values(compiled)).data

    def domains(kind: Any, topology: Any, entities: Any, dim: int) -> Any:
        """Record explicit tagged integration entities independently from geometry."""
        observed.domains.append((kind, entities.copy(), dim))
        return entities

    fem = SimpleNamespace(
        compile_form=compile_native,
        create_form=bind,
        assemble_matrix=matrix,
        assemble_vector=vector,
        IntegralType=SimpleNamespace(exterior_facet="facet"),
        compute_integration_domains=domains,
    )
    ufl = SimpleNamespace(
        algorithms=SimpleNamespace(
            extract_coefficients=lambda form: form.coefficients,
            analysis=SimpleNamespace(extract_constants=lambda form: form.constants),
        )
    )
    actual = forms.import_module
    monkeypatch.setattr(
        forms,
        "import_module",
        lambda name: (
            {
                "dolfinx.fem": fem,
                "dolfinx.la": SimpleNamespace(InsertMode=SimpleNamespace(add="add")),
                "ufl": ufl,
            }.get(name)
            or actual(name)
        ),
    )
    return observed


def bound_workspace(native_api: Any) -> Any:
    """Bind a rectangular bilinear block, two loads and explicit literal zeros."""
    forms_by_name = {
        "a": Form(native_api.space, [[1.0, 2.0], [3.0, 4.0]], constants=(native_api.constant,)),
        "f": Form(native_api.space, [1.0, 2.0], coefficients=(native_api.symbol,)),
        "z": Form(native_api.space, [0.0, 0.0], zero=True),
        "Z": Form(native_api.space, [[0.0, 0.0], [0.0, 0.0]], zero=True),
    }
    bundle = workspace.compile_form_bundle(forms_by_name, native_api.mesh.comm)
    return workspace.create_workspace(
        bundle,
        native_api.mesh,
        coefficient_map={native_api.symbol: native_api.function},
        constant_map={native_api.constant: native_api.native_constant},
    )


def test_current_data_buffers_owned_copies_and_no_recompilation(native_api: Any) -> None:
    """A→B→A data changes preserve exact loads and reuse only two native buffers."""
    with bound_workspace(native_api) as local:
        a, f = local.assemble("a"), local.assemble("f")
        workspace.update_workspace(
            local,
            constants={native_api.constant: np.array(2.0)},
            coefficients={native_api.symbol: np.array([4.0, 5.0])},
        )
        assert_array_equal(local.assemble("a").toarray(), 2 * a.toarray())
        assert_array_equal(local.assemble("f"), [10.0, 11.0])
        workspace.update_workspace(
            local,
            constants={native_api.constant: np.array(1.0)},
            coefficients={native_api.symbol: lambda x: np.array([1.0, 2.0])},
        )
        assert_array_equal(local.assemble("a").toarray(), a.toarray())
        assert_array_equal(local.assemble("f"), f)
        assert_array_equal(local.assemble("z"), [0.0, 0.0])
        assert_array_equal(local.assemble("Z").toarray(), np.zeros((2, 2)))
        assert native_api.compilations == 2 and native_api.allocations == 2
        assert len(native_api.bindings[0][2]) == 0
        assert len(native_api.bindings[1][3]) == 0
        with pytest.raises(TypeError, match="cannot be pickled"):
            pickle.dumps(local)
        with ThreadPoolExecutor(1) as pool, pytest.raises(RuntimeError, match="owning thread"):
            pool.submit(local.assemble, "a").result()
    assert not local.buffers and local.mesh is None and local.bundle is None
    local.close()
    with pytest.raises(RuntimeError, match="closed"):
        local.assemble("a")
    with pytest.raises(RuntimeError, match="closed"):
        local.__enter__()


def test_geometry_updates_and_structural_keys_exclude_values(native_api: Any) -> None:
    """Coordinates/material values change without altering structural compatibility."""
    form = Form(native_api.space, [1.0, 2.0])
    key = workspace.workspace_key(native_api.mesh, {"f": form})
    with bound_workspace(native_api) as local:
        workspace.update_workspace(local, geometry=native_api.mesh.geometry.x + 3.0)
        assert workspace.workspace_key(native_api.mesh, {"f": form}) == key
        native_api.mesh.geometry.dofmap[0, 0] = 1
        assert workspace.workspace_key(native_api.mesh, {"f": form}) != key
        before = native_api.mesh.geometry.x.copy()
        with pytest.raises(ValueError, match="original shape"):
            workspace.update_workspace(local, geometry=np.zeros((1, 3)))
        assert_array_equal(native_api.mesh.geometry.x, before)
        with pytest.raises(KeyError):
            workspace.update_workspace(local, coefficients={object(): [1.0, 2.0]})


def test_structural_key_records_tags_maps_options_and_real_connectivity(native_api: Any) -> None:
    """Equal sizes never merge different entity, topology, FE-layout or compiler contracts."""
    form = Form(native_api.space, [1.0, 2.0])
    baseline = workspace.workspace_key(native_api.mesh, {"f": form})
    assert (
        workspace.workspace_key(
            native_api.mesh,
            {"f": form},
            space_map={0: native_api.space},
        )
        == baseline
    )
    assert (
        workspace.workspace_key(
            native_api.mesh,
            {"f": form},
            form_compiler_options={"quadrature_degree": 7},
        )
        != baseline
    )
    assert (
        workspace.workspace_key(
            native_api.mesh,
            {"f": form},
            subdomains={"facet": [(1, np.array([2]))]},
            entity_maps={object(): np.array([0, 1])},
        )
        != baseline
    )
    tags = SimpleNamespace(indices=np.array([0, 1]), values=np.array([1, 2]))
    form._integrals = (SimpleNamespace(integrand=lambda: 1, subdomain_data=lambda: tags),)
    tagged = workspace.workspace_key(native_api.mesh, {"f": form})
    assert tagged != baseline
    tags.values[:] = [2, 1]
    assert workspace.workspace_key(native_api.mesh, {"f": form}) != tagged
    form._integrals = (
        SimpleNamespace(integrand=lambda: 1, subdomain_data=lambda: [(1, np.array([2]))]),
    )
    assert workspace.workspace_key(native_api.mesh, {"f": form}) != baseline


def test_structural_key_records_native_entity_map_correspondence(native_api: Any) -> None:
    """Native map keys distinguish declared indices, dimensions and topology identities."""
    form = Form(native_api.space, [1.0, 2.0])
    parent = SimpleNamespace(_cpp_object=object())
    submesh = SimpleNamespace(
        _cpp_object=object(),
        index_map=lambda dim: SimpleNamespace(size_local=1, num_ghosts=1),
    )
    correspondence = np.array([1, 3], dtype=np.int32)
    observed_entities: list[np.ndarray[Any, Any]] = []

    def map_entities(entities: Any, inverse: bool) -> Any:
        """Return only the supplied forward correspondence, including ghost entries."""
        assert inverse is False
        observed_entities.append(entities.copy())
        return correspondence[entities]

    entity_map = SimpleNamespace(
        topology=parent,
        sub_topology=submesh,
        dim=1,
        sub_topology_to_topology=map_entities,
    )
    maps = [entity_map]
    key = workspace.workspace_key(native_api.mesh, {"f": form}, entity_maps=maps)
    assert_array_equal(observed_entities[0], [0, 1])
    assert workspace.workspace_key(native_api.mesh, {"f": form}, entity_maps=maps) == key
    assert workspace.workspace_key(native_api.mesh, {"f": form}, entity_maps=[]) != key
    correspondence[:] = [3, 1]
    assert workspace.workspace_key(native_api.mesh, {"f": form}, entity_maps=maps) != key
    correspondence[:] = [1, 3]
    entity_map.dim = 2
    assert workspace.workspace_key(native_api.mesh, {"f": form}, entity_maps=maps) != key
    entity_map.dim = 1
    original_parent = parent._cpp_object
    parent._cpp_object = object()
    assert workspace.workspace_key(native_api.mesh, {"f": form}, entity_maps=maps) != key
    parent._cpp_object = original_parent
    submesh._cpp_object = object()
    assert workspace.workspace_key(native_api.mesh, {"f": form}, entity_maps=maps) != key


def test_native_binding_mesh_element_and_constant_shape_validation(native_api: Any) -> None:
    """Reject a stale mesh or incompatible element/tensor before native form creation."""
    form = Form(native_api.space, [1.0, 2.0])
    bundle = workspace.compile_form_bundle(
        {"f": form}, native_api.mesh.comm, form_compiler_options={"scalar_type": np.float64}
    )
    other_mesh = SimpleNamespace(_cpp_object=object())
    native_api.mesh._cpp_object = object()
    other_space = Space(other_mesh)
    with pytest.raises(ValueError, match="belong to the bound mesh"):
        workspace.create_workspace(bundle, native_api.mesh, space_map={0: other_space})
    other_mesh._cpp_object = native_api.mesh._cpp_object
    other_space.ufl_element = lambda: "P2"
    with pytest.raises(ValueError, match="compiled finite element"):
        workspace.create_workspace(bundle, native_api.mesh, space_map={0: other_space})
    other_space.ufl_element = lambda: "P1"
    with workspace.create_workspace(bundle, native_api.mesh, space_map={0: other_space}):
        pass
    form.constants = (native_api.constant,)
    bundle = workspace.compile_form_bundle({"f": form}, native_api.mesh.comm)
    native_api.native_constant.ufl_shape = (2,)
    with pytest.raises(ValueError, match="tensor shape"):
        workspace.create_workspace(
            bundle, native_api.mesh, constant_map={native_api.constant: native_api.native_constant}
        )


def test_geometry_element_integration_domain_and_precision_are_structural(native_api: Any) -> None:
    """Reject incompatible coordinate kernels and native precision before pointer binding."""
    form = Form(native_api.space, [1.0, 2.0])
    bundle = workspace.compile_form_bundle({"f": form}, native_api.mesh.comm)
    native_api.mesh.ufl_domain = lambda: SimpleNamespace(ufl_coordinate_element=lambda: "P2")
    with pytest.raises(ValueError, match="compiled coordinate element"):
        workspace.create_workspace(bundle, native_api.mesh)
    native_api.mesh.ufl_domain = lambda: SimpleNamespace(ufl_coordinate_element=lambda: "P1")
    native_api.mesh.geometry.x = native_api.mesh.geometry.x.astype(np.float32)
    with pytest.raises(ValueError, match="binary64"):
        workspace.compile_form_bundle({"f": form}, native_api.mesh.comm)
    with pytest.raises(ValueError, match="binary64"):
        workspace.create_workspace(bundle, native_api.mesh)
    native_api.mesh.geometry.x = native_api.mesh.geometry.x.astype(np.float64)
    form.ufl_domains = lambda: (native_api.space.ufl_domain(), native_api.space.ufl_domain())
    with pytest.raises(ValueError, match="exactly one integration domain"):
        workspace.compile_form_bundle({"f": form}, native_api.mesh.comm)
    form.ufl_domains = lambda: (
        SimpleNamespace(ufl_cargo=lambda: None, ufl_coordinate_element=lambda: "P1"),
    )
    with workspace.create_workspace(
        workspace.compile_form_bundle({"f": form}, native_api.mesh.comm),
        native_api.mesh,
    ):
        pass


@pytest.mark.parametrize("kind", ["coefficient", "constant"])
def test_native_data_precision_cannot_mismatch_kernel(kind: str, native_api: Any) -> None:
    """Mixed precision native handles cannot enter binary64 compiled forms implicitly."""
    form = Form(
        native_api.space,
        [1.0, 2.0],
        coefficients=(native_api.symbol,),
        constants=(native_api.constant,),
    )
    bundle = workspace.compile_form_bundle({"f": form}, native_api.mesh.comm)
    if kind == "coefficient":
        native_api.function.x.array = native_api.function.x.array.astype(np.float32)
    else:
        native_api.native_constant.value = native_api.native_constant.value.astype(np.float32)
    with pytest.raises(ValueError, match="binary64"):
        workspace.create_workspace(
            bundle,
            native_api.mesh,
            coefficient_map={native_api.symbol: native_api.function},
            constant_map={native_api.constant: native_api.native_constant},
        )


@pytest.mark.parametrize(
    "forms_by_name, shapes",
    [
        ({}, None),
        ({"": object()}, None),
        ({1: object()}, None),
        ({"a": object()}, {"wrong": (2,)}),
    ],
)
def test_bundle_names_are_unambiguous(native_api: Any, forms_by_name: Any, shapes: Any) -> None:
    """Invalid identifiers fail before native compiler invocations."""
    with pytest.raises(ValueError, match="names|shape names"):
        workspace.compile_form_bundle(forms_by_name, native_api.mesh.comm, shapes=shapes)


def test_bundle_shape_rank_dtype_domain_and_plain_binding_contracts(native_api: Any) -> None:
    """Reject incompatible spaces/ranks/scalars before binding native data."""
    form = Form(native_api.space, [1.0, 2.0])
    with pytest.raises(ValueError, match="scalar_type"):
        workspace.compile_form_bundle(
            {"f": form}, native_api.mesh.comm, form_compiler_options={"scalar_type": "complex128"}
        )
    with pytest.raises(ValueError, match="shape rank"):
        workspace.compile_form_bundle({"f": form}, native_api.mesh.comm, shapes={"f": (2, 2)})
    with pytest.raises(ValueError, match="linear/bilinear"):
        workspace.compile_form_bundle({"f": Form(native_api.space, 1.0)}, native_api.mesh.comm)
    native_api.mesh.comm.size = 2
    with pytest.raises(ValueError, match="single-rank"):
        workspace.compile_form_bundle({"f": form}, native_api.mesh.comm)
    bundle = workspace.CompiledFormBundle({"f": form}, {"f": object()}, {"f": None})
    with pytest.raises(ValueError, match="single-rank"):
        workspace.create_workspace(bundle, native_api.mesh)
    native_api.mesh.comm.size = 1
    with pytest.raises(ValueError, match="explicit native bindings"):
        workspace.create_workspace(bundle, native_api.mesh, space_map={0: object()})
    scalar_zero = Form(native_api.space, 0.0, zero=True)
    bundle = workspace.compile_form_bundle(
        {"z": scalar_zero}, native_api.mesh.comm, shapes={"z": (3,)}
    )
    with workspace.create_workspace(bundle, native_api.mesh) as local:
        assert_array_equal(local.assemble("z"), np.zeros(3))


@pytest.mark.parametrize("domain_arity", [3, 4])
@pytest.mark.parametrize("map_kind", ["mapping", "sequence"])
def test_tagged_domains_explicit_domains_and_space_map(
    domain_arity: int,
    map_kind: str,
    native_api: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Integrals pack unique marker ids; caller domains/entity maps are copied."""
    tags = SimpleNamespace(find=lambda identifier: np.array([identifier], dtype=np.int32), dim=1)
    if domain_arity == 3:

        def domains(kind: Any, topology: Any, entities: Any) -> Any:
            """Model the public DOLFINx 0.10 integration-domain signature."""
            native_api.domains.append((kind, entities.copy(), None))
            return entities

        monkeypatch.setattr(forms._require("dolfinx.fem"), "compute_integration_domains", domains)

    def integral(data: Any, marker: Any) -> Any:
        """Declare one exterior-facet integration domain."""
        return SimpleNamespace(
            integrand=lambda: 1,
            integral_type=lambda: "exterior_facet",
            subdomain_data=lambda: data,
            subdomain_id=lambda: marker,
        )

    form = Form(
        native_api.space,
        [1.0, 2.0],
        integrals=(
            integral(tags, (1, 2)),
            integral(tags, 1),
            integral(None, "everywhere"),
        ),
    )
    options, jit = {"quadrature_degree": 4}, {"timeout": 10}
    bundle = workspace.compile_form_bundle(
        {"f": form}, native_api.mesh.comm, form_compiler_options=options, jit_options=jit
    )
    assert options == {"quadrature_degree": 4} and jit == {"timeout": 10}
    with workspace.create_workspace(bundle, native_api.mesh, space_map={0: native_api.space}):
        assert len(native_api.domains) == 2
        assert all(item[2] == (1 if domain_arity == 4 else None) for item in native_api.domains)
    manual = [(1, np.array([4]))]
    form._integrals = (integral(manual, 1),)
    with workspace.create_workspace(bundle, native_api.mesh):
        assert_array_equal(native_api.bindings[-1][1]["facet"][0][1], [4])
    supplied = {"facet": manual}
    native_map = np.array([0])
    entity_maps: Any = {"mesh": native_map} if map_kind == "mapping" else (native_map,)
    with workspace.create_workspace(
        bundle, native_api.mesh, subdomains=supplied, entity_maps=entity_maps
    ):
        assert native_api.bindings[-1][4]["entity_maps"] is not entity_maps
        supplied_map = native_api.bindings[-1][4]["entity_maps"]
        assert (supplied_map["mesh"] if map_kind == "mapping" else supplied_map[0]) is native_map
        assert native_api.bindings[-1][1]["facet"][0][1] is not manual[0][1]


def test_generic_record_lru_eviction_cleanup_and_thread_boundaries() -> None:
    """Application records close with native workspaces at eviction and scope exit."""
    closed: list[str] = []

    def factory(name: str) -> Any:
        """Create a record with an observable native-resource lifetime."""
        return SimpleNamespace(close=lambda: closed.append(name))

    with workspace.WorkspaceCache[Any](2) as cache:
        a = cache.get("a", lambda: factory("a"))
        cache.get("b", lambda: factory("b"))
        assert cache.get("a", lambda: pytest.fail("cache hit reconstructed")) is a
        cache.get("c", lambda: factory("c"))
        assert closed == ["b"]
        with pytest.raises(TypeError, match="cannot be pickled"):
            pickle.dumps(cache)
        with ThreadPoolExecutor(1) as pool, pytest.raises(RuntimeError, match="another thread"):
            pool.submit(cache.get, "a", lambda: a).result()
    assert closed == ["b", "a", "c"]
    cache.close()
    with pytest.raises(RuntimeError, match="closed"):
        cache.get("a", lambda: a)
    with pytest.raises(RuntimeError, match="closed"):
        cache.__enter__()
    for invalid in (0, -1, True, 1.5):
        with pytest.raises(ValueError, match="positive integer"):
            workspace.WorkspaceCache(invalid)


def test_cache_cleanup_attempts_every_record_even_when_a_close_fails() -> None:
    """A failing native resource cannot prevent independent records from closing."""
    closed = []

    def failed_close() -> None:
        """Represent a backend resource that reports an error while closing."""
        closed.append("failed")
        raise ValueError("native shutdown failed")

    cache = workspace.WorkspaceCache[Any](3)
    cache.get("a", lambda: SimpleNamespace(close=failed_close))
    cache.get("b", lambda: SimpleNamespace(close=failed_close))
    cache.get("c", lambda: SimpleNamespace(close=lambda: closed.append("last")))
    with pytest.raises(ValueError, match="native shutdown failed"):
        cache.close()
    assert closed == ["failed", "failed", "last"]
    cache.close()
