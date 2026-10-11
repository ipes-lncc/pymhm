"""Load suite contracts and prepare isolated, checksum-verified notebook companions."""

from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import Any, NoReturn

import pytest

from scripts.build_notebook_companions import (
    build_notebook_companions,
    validate_notebook_companion_pins,
)

pytest_plugins = ["scripts.pytest_suite"]


@pytest.fixture
def oscillatory_q2_reference() -> dict[str, Any]:
    """Independently integrate a Q2 cell spanning one nonpolynomial material period.

    The physical cell is [0,1/8]^2, with x-fastest equispaced nodal ordering.
    Explicit cardinal polynomials and adaptive one-dimensional integrals avoid
    PyMHM/Basix tables and their tensor Gauss rule. The manufactured pressure
    is sin(pi*x)*sin(pi*y); the source is -div(K*grad(pressure)).
    """
    import numpy as np
    from numpy.polynomial import Polynomial
    from scipy.integrate import quad

    from pymhm.meshes.cartesian import CartesianMacroMesh

    width, amplitude, omega = 1 / 8, 1.5, 16 * np.pi
    cardinal = (Polynomial([1, -3, 2]), Polynomial([0, 4, -4]), Polynomial([0, -1, 2]))
    derivative = tuple(polynomial.deriv() for polynomial in cardinal)

    def kxx(x: Any) -> Any:
        """Evaluate the x-dependent diagonal permeability."""
        return np.exp(amplitude * np.sin(omega * x))

    def kyy(y: Any) -> Any:
        """Evaluate the y-dependent diagonal permeability."""
        return np.exp(amplitude * np.cos(omega * y))

    def force_x(x: Any) -> Any:
        """Differentiate -d/dx(Kxx*d/dx(sin(pi*x))) explicitly."""
        return kxx(x) * (
            np.pi**2 * np.sin(np.pi * x)
            - amplitude * omega * np.pi * np.cos(omega * x) * np.cos(np.pi * x)
        )

    def force_y(y: Any) -> Any:
        """Differentiate -d/dy(Kyy*d/dy(sin(pi*y))) explicitly."""
        return kyy(y) * (
            np.pi**2 * np.sin(np.pi * y)
            + amplitude * omega * np.pi * np.sin(omega * y) * np.cos(np.pi * y)
        )

    def integrate(function: Callable[[float], Any]) -> float:
        """Integrate one physical coordinate with independent adaptive subdivision."""
        return float(quad(function, 0, width, epsabs=2e-12, epsrel=2e-12)[0])

    mass_1d = np.array(
        [
            [
                integrate(lambda x, i=i, j=j: cardinal[i](x / width) * cardinal[j](x / width))
                for j in range(3)
            ]
            for i in range(3)
        ]
    )
    stiffness = []
    for coefficient in (kxx, kyy):
        stiffness.append(
            np.array(
                [
                    [
                        integrate(
                            lambda x, i=i, j=j, coefficient=coefficient: (
                                coefficient(x)
                                * derivative[i](x / width)
                                * derivative[j](x / width)
                                / width**2
                            )
                        )
                        for j in range(3)
                    ]
                    for i in range(3)
                ]
            )
        )
    sine = np.array(
        [integrate(lambda x, i=i: np.sin(np.pi * x) * cardinal[i](x / width)) for i in range(3)]
    )
    forcing = tuple(
        np.array(
            [
                integrate(lambda x, i=i, function=function: function(x) * cardinal[i](x / width))
                for i in range(3)
            ]
        )
        for function in (force_x, force_y)
    )

    def tensor(points: Any) -> Any:
        """Return the positive diagonal material at physical (x,y) points."""
        values = np.zeros((len(points), 2, 2))
        values[:, 0, 0], values[:, 1, 1] = kxx(points[:, 0]), kyy(points[:, 1])
        return values

    def source(points: Any) -> Any:
        """Return the independently differentiated separable physical forcing."""
        x, y = points.T
        return force_x(x) * np.sin(np.pi * y) + np.sin(np.pi * x) * force_y(y)

    return {
        "mesh": CartesianMacroMesh(1, bounds=(0, width, 0, width)),
        "permeability": tensor,
        "source": source,
        "stiffness": np.kron(mass_1d, stiffness[0]) + np.kron(stiffness[1], mass_1d),
        "mass": np.kron(mass_1d, mass_1d),
        "load": np.kron(sine, forcing[0]) + np.kron(forcing[1], sine),
    }


@pytest.fixture(scope="session")
def notebook_companion_archives(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Path, dict[str, dict[str, Any]]]:
    """Build deterministic support ZIPs from tracked sources, without numerical inputs."""
    root = Path(__file__).resolve().parents[1]
    output = tmp_path_factory.mktemp("notebook-companions")
    return output, build_notebook_companions(root, output)


@pytest.fixture
def prepare_notebook_companion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    notebook_companion_archives: tuple[Path, dict[str, dict[str, Any]]],
) -> Callable[[str], Path]:
    """Run unchanged notebook acquisition with exact local ZIPs and no external downloads."""
    output, descriptors = notebook_companion_archives
    root = Path(__file__).resolve().parents[1]
    cache, workspace = tmp_path / "cache", tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setenv("PYMHM_CACHE_DIR", str(cache))
    monkeypatch.setenv("PYMHM_WORKSPACE", str(workspace))
    for name in ("PYMHM_NOTEBOOK_PREPARED", "PYMHM_NOTEBOOK_HISTORICAL", "PYMHM_NOTEBOOK_STUDY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.syspath_prepend(str(workspace))

    def forbid_download(*args: Any, **kwargs: Any) -> NoReturn:
        """Reject any undeclared dependency on the public website or a developer cache."""
        raise AssertionError(f"Notebook tests must use locally built resources: {args}")

    monkeypatch.setattr("pymhm.io.resources.urlopen", forbid_download)

    def prepare(selector: str) -> Path:
        """Seed the selected ZIP and its locally available, verified numerical inputs."""
        descriptor = descriptors[selector]
        validate_notebook_companion_pins(root, {selector: descriptor}, require_pins=True)
        digest = descriptor["sha256"]
        archive = output / "downloads" / digest / (Path(selector).stem + "-companion.zip")
        destination = cache / "archives" / digest / "resources.zip"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(archive, destination)
        with zipfile.ZipFile(archive) as zipped:
            resources = json.loads(zipped.read(".pymhm-resources.json"))["resources"]
        for name, identity in resources.items():
            source = root / name
            if "url" not in identity or not source.is_file():
                continue
            assert hashlib.sha256(source.read_bytes()).hexdigest() == identity["sha256"], name
            target = cache / "resources" / identity["sha256"] / source.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        return workspace

    return prepare
