"""Install and exercise a wheel using only dependencies from the active locked environment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

_INSTALLED_IDENTITY_PROGRAM = """
import json
import os
import platform
from pathlib import Path
import pymhm

package = Path(pymhm.__file__).resolve()
installation = Path(os.environ["PYMHM_EXPECT_INSTALL_ROOT"]).resolve()
assert package.is_relative_to(installation), (
    f"Imported package {package} is outside wheel installation {installation}"
)
print(json.dumps({"platform": platform.platform(), "package": str(package)}))
"""


def check_wheel(directory: Path, report: Path, *, require_pardiso: bool = False) -> int:
    """Check an installed wheel outside the checkout with native dependencies inherited unchanged.

    The temporary virtual environment uses the active interpreter's site packages
    for NumPy, SciPy, Basix, pytest and optional PARDISO. Installing the package
    uses ``--no-deps --no-index``; no dependency or runtime is downloaded. The
    copied tests run outside the repository and verify the imported package is
    owned by that wheel installation. No scientific performance claim is made.
    """
    wheels = sorted(directory.resolve().glob("pymhm-*.whl"))
    if len(wheels) != 1:
        raise ValueError("the wheel directory must contain exactly one pymhm wheel")
    wheel = wheels[0]
    with zipfile.ZipFile(wheel) as archive:
        for name in archive.namelist():
            path = Path(name)
            library_file = path.parts[0] == "pymhm" and (
                path.suffix in {".py", ".pyi"} or path.name == "py.typed"
            )
            prefix = wheel.name.removesuffix("-py3-none-any.whl") + ".dist-info/"
            metadata = name.startswith(prefix) and name.removeprefix(prefix) in {
                "METADATA",
                "WHEEL",
                "RECORD",
                "licenses/LICENSE",
            }
            if not library_file and not metadata:
                raise ValueError(f"Wheel contains a non-library payload: {name}")
    repository = Path(__file__).resolve().parents[1]
    native_library = None
    if require_pardiso:
        from pypardiso import ps

        # A venv inherits Python dependencies, but PyPardiso searches sys.prefix
        # for MKL. Bind the library already loaded by the active locked profile.
        library = Path(ps.libmkl._name)
        if not library.is_absolute():
            matches = sorted(Path(sys.prefix).rglob(library.name))
            if len(matches) != 1:
                raise ValueError("the active environment must identify one MKL runtime")
            library = matches[0]
        if not library.is_file():
            raise ValueError("the active environment's loaded MKL runtime must exist")
        with library.open("rb") as stream:
            native_library = {
                "path": str(library),
                "sha256": hashlib.file_digest(stream, "sha256").hexdigest(),
            }
    with tempfile.TemporaryDirectory(prefix="pymhm-wheel-") as temporary:
        root = Path(temporary)
        environment = root / "environment"
        subprocess.run(
            [sys.executable, "-m", "venv", "--system-site-packages", str(environment)],
            check=True,
        )
        python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        subprocess.run(
            [
                str(python),
                "-m",
                "pip",
                "install",
                "--ignore-installed",
                "--no-deps",
                "--no-index",
                str(wheel),
            ],
            check=True,
            cwd=root,
        )
        variables = os.environ.copy()
        variables.pop("PYTHONPATH", None)
        variables["PYMHM_EXPECT_INSTALL_ROOT"] = str(environment)
        guard = root / "import-guard"
        guard.mkdir()
        (guard / "sitecustomize.py").write_text(
            "import os, sys\n"
            "from pathlib import Path\n"
            f"FORBIDDEN = Path({str(repository)!r}).resolve()\n"
            f"DEPENDENCIES = Path({sys.prefix!r}).resolve()\n"
            "sys.path[:] = [value for value in sys.path\n"
            "    if not Path(value).resolve().is_relative_to(FORBIDDEN)\n"
            "    or Path(value).resolve().is_relative_to(DEPENDENCIES)]\n"
            "def audit(event, arguments):\n"
            "    if event in {'open', 'os.listdir', 'os.scandir'} and arguments:\n"
            "        value = arguments[0]\n"
            "        if isinstance(value, (str, bytes, os.PathLike)):\n"
            "            path = Path(os.fsdecode(value)).resolve()\n"
            "            denied = path.is_relative_to(FORBIDDEN)\n"
            "            if denied and not path.is_relative_to(DEPENDENCIES):\n"
            "                raise RuntimeError(f'Wheel check accessed checkout: {path}')\n"
            "sys.addaudithook(audit)\n",
            encoding="utf-8",
        )
        variables["PYTHONPATH"] = str(guard)
        if native_library is not None:
            variables["PYPARDISO_MKL_RT"] = native_library["path"]
        identity = subprocess.run(
            [str(python), "-c", _INSTALLED_IDENTITY_PROGRAM],
            cwd=root,
            env=variables,
            check=False,
            capture_output=True,
            text=True,
        )
        print(identity.stderr, end="", file=sys.stderr)
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(
            json.dumps(
                {
                    "wheel": wheel.name,
                    "identity_exit_code": identity.returncode,
                    "stdout": identity.stdout,
                    "stderr": identity.stderr,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        identity.check_returncode()
        if require_pardiso:
            subprocess.run(
                [str(python), "-c", "import pypardiso"],
                cwd=root,
                env=variables,
                check=True,
            )
        tests = root / "test_windows_portability.py"
        shutil.copyfile(repository / "tests/test_windows_portability.py", tests)
        result = subprocess.run(
            [
                str(python),
                "-m",
                "pytest",
                "-q",
                str(tests),
                "--junitxml=portable-wheel.xml",
            ],
            cwd=root,
            env=variables,
            check=False,
            capture_output=True,
            text=True,
        )
        print(result.stdout, end="")
        print(result.stderr, end="", file=sys.stderr)
        report.write_text(
            json.dumps(
                {
                    "wheel": wheel.name,
                    "wheel_sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
                    "python": sys.version,
                    "host_platform": platform.platform(),
                    "installed_identity": json.loads(identity.stdout),
                    "checkout_io_guard": str(repository),
                    "locked_dependency_root": sys.prefix,
                    "library_only_payload": True,
                    "required_pardiso": require_pardiso,
                    "native_library": native_library,
                    "test_exit_code": result.returncode,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        shutil.copyfile(root / "portable-wheel.xml", report.with_suffix(".xml"))
        return result.returncode


def main() -> int:
    """Parse the wheel directory and retain an explicit installation/test receipt."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--report", type=Path, default=Path("build/reports/portable-wheel.json"))
    parser.add_argument("--require-pardiso", action="store_true")
    arguments = parser.parse_args()
    return check_wheel(
        arguments.directory, arguments.report, require_pardiso=arguments.require_pardiso
    )


if __name__ == "__main__":
    raise SystemExit(main())
