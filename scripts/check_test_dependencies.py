"""Require the native dependencies and resources used by the complete test suite."""

from __future__ import annotations

import importlib
import json
import shutil

REQUIRED_MODULES = (
    "pytest",
    "pytest_cov",
    "xdist",
    "hypothesis",
    "numpy",
    "scipy",
    "threadpoolctl",
    "basix",
    "ufl",
    "dolfinx.fem",
    "mpi4py.MPI",
    "petsc4py.PETSc",
    "pypardiso",
    "pyamg",
    "cupy",
    "nvmath.sparse.advanced",
    "pyamgx",
    "gmsh",
    "netgen.csg",
    "meshio",
    "pyvista",
    "vtk",
    "nbformat",
    "nbclient",
    "matplotlib",
)


def check_dependencies() -> dict[str, object]:
    """Import every native integration and require MUMPS, MPI, FreeFEM and two GPUs.

    This preflight belongs to the full Linux CUDA test profile. Portable test
    profiles run without it. Missing modules and unsupported native resources
    fail before pytest, instead of leaving dependency-driven skips in a passing
    complete-suite report. The isolated-wheel assertion runs separately after
    building and installing the distribution outside the source checkout.
    """
    modules = {name: importlib.import_module(name) for name in REQUIRED_MODULES}
    devices = modules["cupy"].cuda.runtime.getDeviceCount()
    if devices < 2:
        raise RuntimeError("The complete test profile requires two visible CUDA devices")
    if not modules["petsc4py.PETSc"].Sys.hasExternalPackage("mumps"):
        raise RuntimeError("The complete test profile requires PETSc with MUMPS")
    executables = {name: shutil.which(name) for name in ("mpiexec", "FreeFem++-nw")}
    if not all(executables.values()):
        raise RuntimeError(f"Required native executables are unavailable: {executables}")
    return {
        "modules": list(modules),
        "cuda_devices": devices,
        "mpi_library": modules["mpi4py.MPI"].Get_library_version(),
        "petsc_version": modules["petsc4py.PETSc"].Sys.getVersion(),
        "executables": executables,
    }


def main() -> None:
    """Print the verified full-profile capabilities and propagate dependency errors."""
    print(json.dumps(check_dependencies(), indent=2))


if __name__ == "__main__":
    main()
