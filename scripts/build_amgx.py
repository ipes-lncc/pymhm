"""Build and install optional native AmgX/PyAMGX into the Linux GPU environment.

Run with the pinned ``tools/amgx/pixi.toml`` toolchain. Native libraries are
installed into the target Python prefix; the wheel is local to that prefix and
must not be published as a portable binary distribution.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

AMGX_REVISION = "91a8413ef267b1c32aff4014c02820e1c5897ac2"
PYAMGX_REVISION = "6229ff008ee5a264cfc1799eeb2f83d96da0aadc"


def run(command: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    """Execute a build step with argument boundaries preserved and failure propagated."""
    subprocess.run(command, cwd=cwd, env=env, check=True)


def checkout(path: Path, url: str, revision: str) -> Path:
    """Fetch a pinned upstream checkout, refusing to change an existing revision."""
    if not path.exists():
        run(["git", "clone", "--filter=blob:none", "--no-checkout", url, str(path)])
        run(["git", "checkout", "--detach", revision], cwd=path)
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=path, text=True).strip()
    if actual != revision:
        raise SystemExit(f"Expected upstream revision {revision}; {path} contains {actual}")
    return path


def main() -> None:
    """Build pinned sources, install relative runtime paths, and smoke-test AmgX."""
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-python", type=Path, default=root / ".pixi/envs/gpu/bin/python")
    parser.add_argument("--prefix", type=Path, default=root / ".native/amgx")
    parser.add_argument("--cuda-arch", default="native", help="CMake CUDA architecture, e.g. 86")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--reuse-library", type=Path, help="Reuse a precompiled libamgxsh.so")
    parser.add_argument(
        "--binding-source", type=Path, help="Reuse a checkout at the pinned binding revision"
    )
    args = parser.parse_args()
    if sys.platform != "linux":
        raise SystemExit("This optional native build recipe currently supports Linux only")
    if args.jobs < 1:
        parser.error("jobs must be positive")
    target = args.target_python.resolve()
    prefix = args.prefix.resolve()
    prefix.mkdir(parents=True, exist_ok=True)
    configuration = json.loads(
        subprocess.check_output(
            [
                str(target),
                "-c",
                "import json,sys; print(json.dumps([sys.prefix, list(sys.version_info[:2])]))",
            ],
            text=True,
        )
    )
    target_prefix, target_version = Path(configuration[0]), tuple(configuration[1])
    if target_version != sys.version_info[:2]:
        raise SystemExit("Build and target Python versions must have matching major/minor versions")
    amgx = checkout(prefix / "AmgX", "https://github.com/NVIDIA/AMGX.git", AMGX_REVISION)
    binding = checkout(
        args.binding_source.resolve() if args.binding_source else prefix / "PyAMGX",
        "https://github.com/shwina/pyamgx.git",
        PYAMGX_REVISION,
    )
    build = prefix / "build"
    if args.reuse_library is None:
        cmake = amgx / "CMakeLists.txt"
        text = cmake.read_text(encoding="utf-8")
        # Disable optional profiling ranges for the pinned CUDA 12 NVTX interface.
        text = text.replace("set(NVTXRANGE_FLAG -DNVTX_RANGES)", 'set(NVTXRANGE_FLAG "")')
        cmake.write_text(text, encoding="utf-8")
        run(
            [
                "cmake",
                "-S",
                str(amgx),
                "-B",
                str(build),
                "-G",
                "Ninja",
                "-DCMAKE_BUILD_TYPE=Release",
                f"-DCMAKE_CUDA_ARCHITECTURES={args.cuda_arch}",
                "-DCMAKE_NO_MPI=ON",
            ]
        )
        run(["cmake", "--build", str(build), "--target", "amgxsh", "--parallel", str(args.jobs)])
        library = build / "libamgxsh.so"
    else:
        library = args.reuse_library.resolve()
    if not library.is_file() or library.name != "libamgxsh.so":
        raise SystemExit("A compiled libamgxsh.so is required")
    installed = target_prefix / "lib/libamgxsh.so"
    shutil.copy2(library, installed)
    run(["patchelf", "--set-rpath", "$ORIGIN", str(installed)])
    environment = dict(os.environ, AMGX_DIR=str(amgx), AMGX_BUILD_DIR=str(installed.parent))
    raw_wheels = prefix / "raw-wheels"
    final_wheels = prefix / "wheels"
    unpacked = prefix / "wheel-unpacked"
    for directory in (raw_wheels, final_wheels, unpacked):
        directory.mkdir(exist_ok=True)
    run([sys.executable, "setup.py", "build_ext", "--force"], cwd=binding, env=environment)
    run(
        [sys.executable, "setup.py", "bdist_wheel", "--dist-dir", str(raw_wheels)],
        cwd=binding,
        env=environment,
    )
    wheels = sorted(raw_wheels.glob("pyamgx-*.whl"))
    if len(wheels) != 1:
        raise SystemExit("Expected one freshly built PyAMGX wheel")
    run([sys.executable, "-m", "wheel", "unpack", "--dest", str(unpacked), str(wheels[0])])
    unpacked_wheel = unpacked / "pyamgx-0.1"
    extensions = list(unpacked_wheel.glob("pyamgx*.so"))
    if len(extensions) != 1:
        raise SystemExit("Expected one native PyAMGX extension")
    run(["patchelf", "--set-rpath", "$ORIGIN/../..", str(extensions[0])])
    run(
        [
            sys.executable,
            "-m",
            "wheel",
            "pack",
            "--dest-dir",
            str(final_wheels),
            str(unpacked_wheel),
        ]
    )
    wheel = final_wheels / wheels[0].name
    run(
        [
            sys.executable,
            "-m",
            "pip",
            "--python",
            str(target),
            "install",
            "--no-deps",
            "--force-reinstall",
            str(wheel),
        ]
    )
    smoke = (
        "import numpy as np; from pymhm.linalg.linear import solve_linear; "
        "from scipy.sparse import diags; "
        "a=diags([-np.ones(99),2*np.ones(100),-np.ones(99)],[-1,0,1]); "
        "x=solve_linear(a,np.ones(100),solver='amgx'); "
        "assert np.linalg.norm(a@x-np.ones(100))<1e-8; print('Native AmgX installation verified')"
    )
    clean_environment = {
        key: value
        for key, value in os.environ.items()
        if key not in {"PYTHONPATH", "LD_LIBRARY_PATH"}
    }
    run([str(target), "-c", smoke], cwd=root, env=clean_environment)
    print(f"Installed AmgX and PyAMGX into {target_prefix}; no PYTHONPATH override is required")


if __name__ == "__main__":
    main()
