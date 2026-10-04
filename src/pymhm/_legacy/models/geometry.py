"""Dispatch existing PDE formulations over polygonal macroelement geometry."""

from typing import Any, cast

from pymhm.meshes.polygonal import PolygonMesh
from pymhm.meshes.triangle import TriangleMesh


def solve_darcy_polygons(mesh: PolygonMesh, **options: Any) -> Any:
    """Use the shared primal/RT0 Darcy assembly on a polygonal macro partition.

    All local cells remain triangles. Discrete point wells are currently defined
    on triangular macro partitions; use distributed volume sources on polygons.
    """
    from pymhm._legacy.models.darcy.primal import solve_darcy

    if options.get("point_sources") is not None:
        raise ValueError("point wells require a triangular macro mesh")
    return solve_darcy(cast(TriangleMesh, mesh), **options)


def solve_transport_polygons(mesh: PolygonMesh, **options: Any) -> Any:
    """Use the conservative RAD variational forms on polygonal macrocells."""
    from pymhm._legacy.models.transport.solver import solve_transport

    return solve_transport(cast(TriangleMesh, mesh), **options)


def solve_brinkman_polygons(mesh: PolygonMesh, **options: Any) -> Any:
    """Use the shared incompressible velocity/pressure operators on polygonal cells."""
    from pymhm._legacy.models.vector import solve_brinkman

    return solve_brinkman(cast(TriangleMesh, mesh), **options)


def solve_mshho_polygons(mesh: PolygonMesh, **options: Any) -> Any:
    """Use total-degree cell moments and energy minimization on polygonal macrocells."""
    from pymhm.methods.hho import solve_mshho

    return solve_mshho(cast(TriangleMesh, mesh), **options)


def solve_elasticity_mixed_polygons(mesh: PolygonMesh, **options: Any) -> Any:
    """Solve weak-symmetry BDM elasticity on straight-sided polygonal macrocells.

    The local stress and displacement spaces use conforming triangulations of
    each polygon; skeletal tractions remain on its original boundary edges.
    Nonconvex simple cells are supported. Lamé and full anisotropic compliance,
    mixed displacement/traction boundaries and rigid gauges use the shared
    mixed-elasticity implementation. Trace partitions must align with the local
    boundary mesh, and the chosen local spaces must resolve every rigid mode.
    """
    from pymhm._legacy.models.elasticity.stress import solve_elasticity_mixed

    return solve_elasticity_mixed(cast(TriangleMesh, mesh), **options)
