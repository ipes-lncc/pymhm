"""Selected physical inputs for the three-layer elastodynamic experiment of 2017.

The 341-cell connectivity and digitized horizons are declared selected geometry.
They do not identify the paper's historical mesh. Physical plane-strain Lamé
coefficients preserve the tabulated P/S wave speeds and differ from printed
Equation (56). Spatial assembly, signed trace moments and radial-force integration
belong to the shared PyMHM owners; this module supplies only physical data.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from examples.campaign_provenance import verify_archive
from pymhm import PolylineLayerField, RadialDiskLoad, TriangleMesh
from pymhm.core.validation import positive_int
from pymhm.fem.traces.interval import FaceSpace, SkeletonSpace
from pymhm.io.provenance import file_digest
from pymhm.materials.elasticity import constitutive_values

DATA = Path(__file__).resolve().parent / "data/three-layer-2017"
Array = NDArray[np.float64]


def _positive(value: Any, name: str) -> float:
    """Reject boolean, complex, nonfinite and nonpositive physical parameters."""
    if isinstance(value, bool) or np.iscomplexobj(value) or not np.isscalar(value):
        raise ValueError(f"{name} must be a finite positive real number")
    result = float(cast(Any, value))
    if not np.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be a finite positive real number")
    return result


@dataclass(frozen=True)
class ThreeLayerCase:
    """Frozen selected geometry with SI material/source parameters and declared scales.

    ``mesh`` and horizon coordinates use x*=x/L0, with depth increasing downward.
    Density and Lamé arrays retain SI units. Scales are (L0 m, rho0 kg/m³, u0 m,
    t0 s). No artificial rigid-motion gauge is imposed: positive mass and the
    prescribed zero initial state determine transient rigid motion.
    """

    directory: Path
    mesh: TriangleMesh
    abscissae: Array
    heights: Array
    density: Array
    lame_lambda: Array
    lame_mu: Array
    scales: tuple[float, float, float, float]
    source_center_m: tuple[float, float]
    source_radius_m: float
    source_amplitude_N_m3: float
    source_frequencies_Hz: tuple[float, float, float, float]
    source_center_time_s: float
    source_duration_s: float
    input_sha256: dict[str, str]

    def materials(self) -> tuple[PolylineLayerField, PolylineLayerField]:
        """Scale physical density and use the shared Kelvin plane-strain tensor owner."""
        length, density, _, time = self.scales
        stiffness = constitutive_values(
            None,
            np.zeros((len(self.density), 2)),
            lame_lambda=self.lame_lambda,
            lame_mu=self.lame_mu,
        )
        return (
            PolylineLayerField(self.abscissae, self.heights, self.density / density),
            PolylineLayerField(
                self.abscissae, self.heights, stiffness * time**2 / (density * length**2)
            ),
        )

    def wavelet(self, physical_time_s: Any) -> Array:
        """Evaluate the unit-peak inverse transform of the selected trapezoidal spectrum.

        The four corner frequencies are physical Hz; np.sinc(z)=sin(pi*z)/(pi*z).
        The denominator is the spectrum's integral, not a fitted field amplitude.
        The selected truncation has zero value/derivative at its endpoints, since
        every corner times the delay is an integer. Finite support is not strict
        spectral band limitation. Source support extends beyond T=.3 s.
        """
        if np.iscomplexobj(physical_time_s):
            raise ValueError("physical time must be finite and real")
        time = np.asarray(physical_time_s, dtype=float)
        if not np.isfinite(time).all():
            raise ValueError("physical time must be finite and real")
        tau = time - self.source_center_time_s
        first, second, third, fourth = self.source_frequencies_Hz
        upper = (fourth**2 * np.sinc(fourth * tau) ** 2 - third**2 * np.sinc(third * tau) ** 2) / (
            fourth - third
        )
        lower = (second**2 * np.sinc(second * tau) ** 2 - first**2 * np.sinc(first * tau) ** 2) / (
            second - first
        )
        return np.where(
            (time > 0) & (time < self.source_duration_s),
            (upper - lower) / (fourth + third - second - first),
            0.0,
        )

    def scaled_wavelet(self, time: float) -> float:
        """Convert nondimensional integration time to physical seconds."""
        return float(self.wavelet(time * self.scales[3]))

    def source(self) -> RadialDiskLoad:
        """Supply force per volume through the shared unsmoothed cut-disk provider.

        From rho*u_tt-div(sigma)=f, f*=f*t0²/(rho0*u0). No additional material
        density multiplies the spatial force. The center and open disk are in
        normalized coordinates; value at the disk center is zero.
        """
        length, density, displacement, time = self.scales
        return RadialDiskLoad(
            np.asarray(self.source_center_m) / length,
            self.source_radius_m / length,
            self.source_amplitude_N_m3 * time**2 / (density * displacement),
            time_function=self.scaled_wavelet,
        )

    def skeleton(self) -> SkeletonSpace:
        """Use exactly vector discontinuous P2 on eight oriented segments per macroface."""
        return SkeletonSpace(
            self.mesh, tuple(FaceSpace.uniform(2, 8) for _ in self.mesh.faces), components=2
        )

    def metadata(self) -> dict[str, Any]:
        """Return verified physical attribution separately from any numerical acceptance."""
        self.verify_inputs()
        contract = json.loads((self.directory / "case.json").read_text())
        return {
            "case": "Selected three-layer completion of Gomes et al. (2017), Section 5.2",
            "historical_literal_reproduction": False,
            "selected_geometry": contract["geometry"],
            "input_interpretation": contract["input_interpretation"],
            "paper": contract["paper"],
            "material_reference": contract["material_reference"],
            "constitutive_convention": contract["constitutive_convention"],
            "source": contract["source"],
            "nondimensionalization": contract["nondimensionalization"],
            "input_sha256": dict(self.input_sha256),
            "units": {"displacement": "m", "velocity": "m/s", "Cauchy_stress": "Pa"},
            "trace_convention": (
                "Negative mean physical traction on the preceding time slab in first-owner "
                "Cartesian P2/s8 coordinates; physical traction is minus the multiplier"
            ),
            "gauge": "No added rigid-motion gauge; positive inertia and zero initial state",
            "scientific_acceptance": False,
        }

    def verify_inputs(self) -> None:
        """Reject altered input bytes before resuming or exposing acquired results."""
        for name, digest in self.input_sha256.items():
            verify_archive(self.directory / name, digest)


def load_case(directory: Path = DATA) -> ThreeLayerCase:
    """Validate the pinned lightweight inputs and preserve their exact selected connectivity.

    A different declared input directory receives its own immutable acquisition
    identity; geometry/horizon bytes must match that directory's manifest. The
    selected published spaces/time grid remain P3/r8, P2/s8, dt=.001/T=.3.
    """
    directory = Path(directory)
    contract = json.loads((directory / "case.json").read_text())
    hashes = {"case.json": file_digest(directory / "case.json")}
    for name in ("macro-mesh.json", "horizons.csv"):
        hashes[name] = verify_archive(directory / name, contract["inputs"][name]["sha256"])
    expected = {
        "macro_triangles": 341,
        "local_displacement_degree": 3,
        "local_uniform_refinement": 8,
        "local_triangles_per_macro": 64,
        "trace_degree": 2,
        "trace_segments_per_face": 8,
        "trace_components": 2,
        "time_step_s": 0.001,
        "final_time_s": 0.3,
        "newmark_beta": 0.25,
        "newmark_gamma": 0.5,
        "initial_displacement": 0,
        "initial_velocity": 0,
        "boundary_condition": "Homogeneous physical traction on all exterior faces",
    }
    if json.dumps(contract["discretization"], sort_keys=True) != json.dumps(
        expected, sort_keys=True
    ):
        raise ValueError("preserve the declared P3/r8, P2/s8, zero initial/traction and time grid")
    scales_data = contract["nondimensionalization"]
    scale_length, scale_density, scale_displacement, scale_time = (
        _positive(scales_data[key], key)
        for key in (
            "reference_length_m",
            "reference_density_kg_m3",
            "reference_displacement_m",
            "reference_time_s",
        )
    )
    scales = (scale_length, scale_density, scale_displacement, scale_time)
    geometry = contract["geometry"]
    if geometry["box_m"] != [0, 1000, 0, 450]:
        raise ValueError("the selected physical box is [0,1000] by [0,450] m")
    raw_mesh = json.loads((directory / "macro-mesh.json").read_text())
    mesh = TriangleMesh(np.asarray(raw_mesh["points"]) * (1000 / scales[0]), raw_mesh["cells"])
    if (
        len(mesh.cells) != 341
        or not np.array_equal(
            np.stack([mesh.points.min(axis=0), mesh.points.max(axis=0)]) * scales[0],
            [[0, 0], [1000, 450]],
        )
        or not np.isclose(mesh.areas.sum() * scales[0] ** 2, 450000, rtol=0, atol=2e-9)
    ):
        raise ValueError("the selected macro mesh must cover the complete physical box")
    horizons = np.loadtxt(directory / "horizons.csv", delimiter=",", skiprows=1)
    abscissae, heights = horizons[:, 0] * 1000 / scales[0], horizons[:, 1:].T * 450 / scales[0]
    rows = contract["materials"]
    if [row["layer"] for row in rows] != ["1", "7", "15"]:
        raise ValueError("retain the declared selected material mapping 1,7,15")
    density = np.array([_positive(row["density_kg_m3"], "density") for row in rows])
    vp = np.array([_positive(row["vp_m_s"], "P-wave velocity") for row in rows])
    vs = np.array([_positive(row["vs_m_s"], "S-wave velocity") for row in rows])
    mu, lam = density * vs**2, density * (vp**2 - 2 * vs**2)
    constitutive_values(None, np.zeros((3, 2)), lame_lambda=lam, lame_mu=mu)
    if not np.allclose(
        [lam, mu],
        [[row["lambda_Pa"] for row in rows], [row["mu_Pa"] for row in rows]],
        rtol=4e-15,
        atol=0,
    ):
        raise ValueError("recorded Lamé coefficients do not preserve the physical wave speeds")
    PolylineLayerField(abscissae, heights, density)
    if heights.shape[0] != 2 or heights.min() <= 0 or heights.max() >= 450 / scales[0]:
        raise ValueError("both selected horizons must lie strictly inside the physical box")
    source = contract["source"]
    frequencies = tuple(
        _positive(value, "Ormsby corner frequency")
        for value in source["ormsby_corner_frequencies_Hz"]
    )
    delay = _positive(source["center_time_s"], "source delay")
    duration = _positive(source["duration_s"], "source duration")
    if (
        len(frequencies) != 4
        or not np.all(np.diff(frequencies) > 0)
        or duration != 2 * delay
        or any(frequency * delay != round(frequency * delay) for frequency in frequencies)
    ):
        raise ValueError("the selected Ormsby pulse requires ordered corners and zero C1 endpoints")
    center = tuple(float(value) for value in geometry["source_center_m"])
    radius = _positive(source["radius_m"], "source radius")
    amplitude = _positive(source["peak_amplitude_N_m3"], "physical force amplitude")
    if (
        len(center) != 2
        or not np.isfinite(center).all()
        or any(
            coordinate - radius <= 0 or coordinate + radius >= bound
            for coordinate, bound in zip(center, (1000, 450), strict=True)
        )
    ):
        raise ValueError("the selected source disk must be wholly inside the physical box")
    for array in (abscissae, heights, density, lam, mu):
        array.setflags(write=False)
    case = ThreeLayerCase(
        directory,
        mesh,
        abscissae,
        heights,
        density,
        lam,
        mu,
        scales,
        center,
        radius,
        amplitude,
        frequencies,
        delay,
        duration,
        hashes,
    )
    case.verify_inputs()
    return case


def assembly_order(order: int) -> int:
    """Require the shared P3 assembly minimum q5; q7 supplies its integration control."""
    return positive_int(order, "quadrature order", 5)
