"""Native 3D generator lifecycle/contracts and real Gmsh/Netgen integration."""

from types import SimpleNamespace

import numpy as np
import pytest
from numpy.testing import assert_allclose
from threadpoolctl import threadpool_limits

from pymhm import meshing_native3d as native
from pymhm.darcy3d import solve_darcy_3d
from pymhm.tetrahedral import TetraMesh


class GmshModel:
    """Small explicit Gmsh API contract with sparse node tags and physical entities."""

    def __init__(self):
        """Build only topology and API state; this is not a substitute for native meshing."""
        self.domain = TetraMesh.unit_cube()
        self.dimension = 3
        self.kind = 4
        self.invalid_node = False
        self.conflict = False
        self.current = "caller"
        self.nodes = 11 + 3 * np.arange(len(self.domain.points))
        self.mesh = SimpleNamespace(
            getNodes=lambda: (self.nodes, self.domain.points.ravel(), []),
            getElements=self.elements,
            setSize=lambda *a: None,
            generate=lambda *a: None,
        )
        self.occ = SimpleNamespace(
            addBox=lambda *a: 1, synchronize=lambda: None, getCenterOfMass=self.center
        )

    def center(self, dim, tag):
        """Return the cube surface barycenter for its geometric entity."""
        result = np.full(3, 0.5)
        result[(tag - 1) // 2] = (tag - 1) % 2
        return result

    def elements(self, dim, entity=-1):
        """Resolve volume or boundary triangles, with noncontiguous global IDs."""
        if dim == 3:
            nodes = self.nodes[self.domain.cells].ravel().copy()
            if self.invalid_node:
                nodes[0] = 999
            return [self.kind], [np.arange(len(self.domain.cells)) + 100], [nodes]
        faces = self.domain.boundary_faces
        if entity != -1:
            centers = self.domain.points[self.domain.faces[faces]].mean(axis=1)
            axis, side = (entity - 1) // 2, (entity - 1) % 2
            faces = faces[np.isclose(centers[:, axis], side)]
        return [2], [500 + faces], [self.nodes[self.domain.faces[faces]].ravel()]

    def getDimension(self):
        """Return the declared geometric dimension."""
        return self.dimension

    def getPhysicalGroups(self):
        """Expose one volume and six boundary groups, with an optional invalid overlap."""
        return (
            [(3, 1)]
            + [(2, i) for i in range(1, 7)]
            + ([(3, 2)] if self.conflict else [])
            + [(1, 99)]
        )

    def getEntitiesForPhysicalGroup(self, dim, tag):
        """Map physical IDs to their actual dimensional geometric entity."""
        return [1 if dim == 3 else tag]

    def getEntities(self, dim):
        """List the cube's geometric vertices or six faces."""
        return [(dim, i) for i in range(1, 9 if dim == 0 else 7)]

    def getCurrent(self):
        """Return the caller's active model name."""
        return self.current

    def setCurrent(self, name):
        """Record an explicit active-model switch."""
        self.current = name

    def add(self, name):
        """Select a temporary owned model."""
        self.current = name

    def remove(self):
        """Remove only the active temporary model."""
        self.current = ""

    def addPhysicalGroup(self, *args):
        """Accept the deterministic unit-cube physical group API call."""


@pytest.fixture
def gmsh_contract(monkeypatch):
    """Supply session state whose ownership can be asserted without optional imports."""
    model = GmshModel()
    state = {"initialized": False}
    module = SimpleNamespace(
        model=model,
        isInitialized=lambda: state["initialized"],
        initialize=lambda: state.update(initialized=True),
        finalize=lambda: state.update(initialized=False),
    )
    monkeypatch.setattr(native, "_optional", lambda name: module)
    return model, state


def test_gmsh_ownership_and_physical_ids(gmsh_contract):
    """Import sparse tags and preserve caller model/session when generating another model."""
    model, state = gmsh_contract
    with pytest.raises(RuntimeError, match="initialize"):
        native.from_gmsh_3d()
    result = native.unit_cube_gmsh()
    assert not state["initialized"] and model.current == "caller"
    assert_allclose(result.mesh.volumes.sum(), 1.0)
    assert set(result.face_tags) == set(range(7))
    state["initialized"] = True
    native.from_gmsh_3d()
    native.unit_cube_gmsh()
    assert state["initialized"] and model.current == "caller"
    model.current = ""
    native.unit_cube_gmsh()
    assert model.current == ""


@pytest.mark.parametrize(
    "bad,match",
    [
        ("dimension", "dimension"),
        ("kind", "first-order"),
        ("invalid_node", "node tags"),
        ("conflict", "conflicting"),
    ],
)
def test_gmsh_failures_and_cleanup(gmsh_contract, bad, match):
    """Invalid contracts never remove the caller's pre-existing model."""
    model, state = gmsh_contract
    setattr(model, bad, 2 if bad == "dimension" else 11 if bad == "kind" else True)
    with pytest.raises(ValueError, match=match):
        native.unit_cube_gmsh()
    assert not state["initialized"] and model.current == "caller"


def netgen_source():
    """Build the Netgen API contract on an actual tetrahedral topology."""
    mesh = TetraMesh.unit_cube()

    def element(vertices, index):
        """Create primary point IDs and an explicit descriptor/material index."""
        return SimpleNamespace(
            points=vertices,
            vertices=[SimpleNamespace(nr=int(i) + 1) for i in vertices],
            index=index,
        )

    volumes = [element(cell, 3) for cell in mesh.cells]
    surfaces = [element(mesh.faces[face], 1) for face in mesh.boundary_faces]
    return (
        SimpleNamespace(
            dim=3,
            Points=lambda: [SimpleNamespace(p=p) for p in mesh.points],
            Elements3D=lambda: volumes,
            Elements2D=lambda: surfaces,
            FaceDescriptor=lambda index: SimpleNamespace(bc=8),
        ),
        volumes,
        surfaces,
    )


def test_netgen_descriptor_contract_and_factory(monkeypatch):
    """Boundary IDs come from descriptor.bc, not its storage index."""
    source, volumes, surfaces = netgen_source()
    data = native.from_netgen_3d(source)
    assert set(data.cell_tags) == {3}
    assert set(data.face_tags) == {0, 8}
    monkeypatch.setattr(
        native,
        "_optional",
        lambda name: SimpleNamespace(
            unit_cube=SimpleNamespace(GenerateMesh=lambda **kwargs: source)
        ),
    )
    assert_allclose(native.unit_cube_netgen().mesh.volumes.sum(), 1.0)
    source.dim = 2
    with pytest.raises(ValueError, match="dimension"):
        native.from_netgen_3d(source)
    source.dim = 3
    volumes[0].points = list(range(10))
    with pytest.raises(ValueError, match="tetrahedra"):
        native.from_netgen_3d(source)
    volumes[0].points = list(range(4))
    surfaces[0].points = list(range(6))
    with pytest.raises(ValueError, match="triangles"):
        native.from_netgen_3d(source)


@pytest.mark.meshing
@pytest.mark.parametrize(
    "generator,dependency",
    [(native.unit_cube_gmsh, "gmsh"), (native.unit_cube_netgen, "netgen.csg")],
)
def test_actual_native_volume_generator_and_darcy_patch(generator, dependency):
    """Real generators preserve volume, IDs and a physical affine Darcy pressure patch."""
    pytest.importorskip(dependency)
    with threadpool_limits(1):
        data = generator(0.9)
        assert_allclose(data.mesh.volumes.sum(), 1.0, atol=2e-14)
        assert set(data.face_tags[data.mesh.boundary_faces]) == set(range(1, 7))

        def exact(x):
            """Nonhomogeneous affine pressure on the generated unstructured mesh."""
            return 1 + x[:, 0] + 2 * x[:, 1] + 3 * x[:, 2]

        solution = solve_darcy_3d(data.mesh, degree=1, local_refinement=1, dirichlet=exact)
        assert solution.l2_error(exact) < 2e-10
        assert solution.flux_l2_error([-1, -2, -3]) < 2e-10
