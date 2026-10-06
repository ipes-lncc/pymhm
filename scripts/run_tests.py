"""Run independent tests with xdist, then isolated serial tests, with fresh coverage."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NATIVE_THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def available_cpus() -> int:
    """Return CPUs available to this process, including Linux affinity restrictions."""
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        return max(1, os.cpu_count() or 1)


def _pytest_arguments(arguments: list[str]) -> list[str]:
    """Reject flags that would override phase isolation or coverage ownership."""
    if arguments[:1] == ["--"]:
        arguments = arguments[1:]
    for argument in arguments:
        if (
            argument.startswith("-n")
            or argument.startswith(
                ("--numprocesses", "--dist", "--max-worker-restart", "--cov", "--no-cov")
            )
            or argument.startswith("--pymhm-test-phase")
        ):
            raise ValueError(
                "Use the runner's --workers and --coverage options; "
                "pytest scheduling, coverage and phase options are reserved."
            )
    return arguments


def _phase_arguments(arguments: list[str], phase: str) -> list[str]:
    """Give each phase its own JUnit XML instead of overwriting the first report."""
    result = []
    junit_path = False
    for argument in arguments:
        if junit_path:
            path = Path(argument)
            result.append(str(path.with_name(f"{path.stem}-{phase}{path.suffix}")))
            junit_path = False
        elif argument in {"--junitxml", "--junit-xml"}:
            result.append(argument)
            junit_path = True
        elif argument.startswith(("--junitxml=", "--junit-xml=")):
            option, value = argument.split("=", 1)
            path = Path(value)
            result.append(f"{option}={path.with_name(f'{path.stem}-{phase}{path.suffix}')}")
        else:
            result.append(argument)
    return result


def _command(command: list[str], environment: dict[str, str]) -> int:
    """Run a child to completion so its workers have exited before the next phase."""
    print("Running:", " ".join(command), flush=True)
    return subprocess.run(command, env=environment, check=False).returncode


def _coverage_reports(data_paths: list[Path], report: Path, environment: dict[str, str]) -> int:
    """Combine only this completed run and publish the conventional coverage outputs."""
    prefix = [sys.executable, "-m", "coverage"]
    data = f"--data-file={report / '.coverage'}"
    commands = [
        [*prefix, "combine", "--keep", data, *(str(path) for path in data_paths)],
        [*prefix, "report", data, "--fail-under=0"],
        [*prefix, "xml", data, "--fail-under=0", "-o", str(report / "coverage.xml")],
        [*prefix, "json", data, "--fail-under=0", "-o", str(report / "coverage.json")],
    ]
    for command in commands:
        status = _command(command, environment)
        if status:
            return status
    return 0


def run_suite(
    arguments: list[str],
    *,
    workers: int,
    coverage: bool = False,
    distribution: str = "loadscope",
    report: Path = ROOT / "build/reports/coverage",
) -> int:
    """Execute both phases without overlap; incomplete runs never publish coverage.

    ``workers`` is capped to the process's available CPUs. Each child gets one
    native numerical thread. User pytest selectors apply to both phases. An
    empty phase is accepted, but an entirely empty selection returns pytest's
    exit code 5. A failed parallel phase stops before serial execution. Coverage
    uses separate temporary measurements and combines them only after both
    phases succeed; the existing check_coverage.py enforces the independent
    line and branch gates on the resulting JSON. A native worker crash stops
    the run without restarting workers or repeating the crashing test.
    """
    if workers < 1:
        raise ValueError("workers must be a positive integer")
    if distribution not in {"loadscope", "worksteal"}:
        raise ValueError("distribution must be loadscope or worksteal")
    environment = os.environ.copy()
    arguments = [
        *_pytest_arguments(shlex.split(environment.pop("PYTEST_ADDOPTS", ""))),
        *_pytest_arguments(arguments),
    ]
    workers = min(workers, available_cpus())
    environment.update(dict.fromkeys(NATIVE_THREAD_VARIABLES, "1"))
    environment.pop("PYTEST_XDIST_WORKER", None)
    environment.pop("PYTEST_XDIST_WORKER_COUNT", None)
    if coverage:
        report = report.resolve()
        report.mkdir(parents=True, exist_ok=True)
        for name in (".coverage", "coverage.xml", "coverage.json"):
            (report / name).unlink(missing_ok=True)
    with tempfile.TemporaryDirectory(prefix="pymhm-tests-") as temporary:
        data_paths = []
        statuses = []
        for phase, scheduling in (
            (
                "parallel",
                ["-n", str(workers), f"--dist={distribution}", "--max-worker-restart=0"],
            ),
            ("serial", ["-n", "0"]),
        ):
            command = [
                sys.executable,
                "-m",
                "pytest",
                "-p",
                "scripts.pytest_suite",
                "-q",
                f"--pymhm-test-phase={phase}",
                *scheduling,
                *_phase_arguments(arguments, phase),
            ]
            phase_environment = environment.copy()
            data = Path(temporary) / phase / ".coverage"
            if coverage:
                data.parent.mkdir()
                phase_environment["COVERAGE_FILE"] = str(data)
                command.extend(
                    [
                        "--cov=pymhm",
                        "--cov-branch",
                        f"--cov-config={ROOT / 'pyproject.toml'}",
                        "--cov-report=",
                        "--cov-fail-under=0",
                    ]
                )
            status = _command(command, phase_environment)
            statuses.append(status)
            if status not in (0, 5):
                print(f"The {phase} phase failed; no qualified coverage is published.", flush=True)
                return status
            if coverage and status == 0:
                if not data.is_file():
                    print(f"Missing {phase} coverage measurement: {data}", file=sys.stderr)
                    return 1
                data_paths.append(data)
        if all(status == 5 for status in statuses):
            return 5
        if coverage:
            return _coverage_reports(data_paths, report, environment)
    return 0


def main() -> int:
    """Parse bounded worker settings and pass pytest selectors to both phases."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workers",
        type=int,
        default=os.environ.get("PYMHM_TEST_WORKERS", available_cpus()),
        help=(
            "xdist workers, capped to available CPUs "
            "(default: PYMHM_TEST_WORKERS or all available CPUs)"
        ),
    )
    parser.add_argument(
        "--distribution",
        choices=("loadscope", "worksteal"),
        default="loadscope",
        help="xdist scheduler (default: loadscope preserves shared module fixtures)",
    )
    parser.add_argument("--coverage", action="store_true", help="Combine fresh branch coverage")
    parser.add_argument("--report-dir", type=Path, default=ROOT / "build/reports/coverage")
    parser.add_argument("pytest_args", nargs=argparse.REMAINDER, help="pytest arguments after --")
    options = parser.parse_args()
    try:
        return run_suite(
            options.pytest_args,
            workers=options.workers,
            coverage=options.coverage,
            distribution=options.distribution,
            report=options.report_dir,
        )
    except ValueError as error:
        parser.error(str(error))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
