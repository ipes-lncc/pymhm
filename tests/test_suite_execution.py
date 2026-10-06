"""Exercise process isolation, serial selection and coverage union with real pytest workers."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/run_tests.py"


def _run(
    command: list[str], directory: Path, *, extra_environment: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    """Invoke a private test subprocess with explicit import and environment ownership."""
    environment = os.environ.copy()
    environment.pop("PYTEST_ADDOPTS", None)
    environment["PYTHONPATH"] = os.pathsep.join((str(directory), str(ROOT)))
    if extra_environment:
        environment.update(extra_environment)
    return subprocess.run(
        [sys.executable, *command],
        cwd=directory,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )


def _cases(directory: Path, source: str) -> Path:
    """Write a private pytest case with a declared serial marker."""
    directory.mkdir()
    (directory / "pytest.ini").write_text(
        "[pytest]\nmarkers = serial: exclusive test\n", encoding="utf-8"
    )
    (directory / "test_cases.py").write_text(textwrap.dedent(source), encoding="utf-8")
    return directory / "test_cases.py"


def test_parallel_workers_exit_before_serial_and_coverage_is_the_union(tmp_path: Path) -> None:
    """Verify genuine worker overlap, complete shutdown and fresh union coverage."""
    directory = tmp_path / "suite"
    _cases(
        directory,
        """
        import os
        from pathlib import Path
        import pytest
        from pymhm import choose

        def test_parallel_positive():
            assert os.environ["PYTEST_XDIST_WORKER"].startswith("gw")
            assert os.environ["OPENBLAS_NUM_THREADS"] == "1"
            assert os.environ["MKL_NUM_THREADS"] == "1"
            assert choose(True) == 1

        @pytest.mark.serial
        def test_serial_negative():
            assert "PYTEST_XDIST_WORKER" not in os.environ
            assert choose(False) == -1
            started = list(Path(".").glob("gw*.started"))
            assert len(started) == int(Path("worker-count").read_text())
            assert all(p.with_suffix(".closed").is_file() for p in started)
            Path("serial.completed").write_text(str(os.getpid()))
        """,
    )
    (directory / "conftest.py").write_text(
        textwrap.dedent(
            """
            import os
            from pathlib import Path
            import time
            import pytest

            @pytest.fixture(scope="session", autouse=True)
            def record_worker_exit():
                worker = os.environ.get("PYTEST_XDIST_WORKER")
                if worker:
                    count = int(os.environ["PYTEST_XDIST_WORKER_COUNT"])
                    Path("worker-count").write_text(str(count))
                    Path(f"{worker}.started").write_text(str(os.getpid()))
                    deadline = time.monotonic() + 15
                    while len(list(Path(".").glob("gw*.started"))) < count:
                        assert time.monotonic() < deadline
                        time.sleep(0.01)
                yield
                if worker:
                    Path(f"{worker}.closed").write_text(str(os.getpid()))
            """
        ),
        encoding="utf-8",
    )
    (directory / "test_other.py").write_text(
        "from pymhm import choose\ndef test_parallel_other():\n    assert choose(True) == 1\n",
        encoding="utf-8",
    )
    package = directory / "pymhm"
    package.mkdir()
    (package / "__init__.py").write_text(
        "def choose(positive):\n    if positive:\n        return 1\n    return -1\n",
        encoding="utf-8",
    )
    report = directory / "reports"
    report.mkdir()
    for name in (".coverage", "coverage.json", "coverage.xml"):
        (report / name).write_text("stale", encoding="utf-8")
    result = _run(
        [
            str(RUNNER),
            "--workers",
            "2",
            "--coverage",
            "--report-dir",
            str(report),
            "--",
            "-q",
            "--junitxml=tests.xml",
            str(directory),
        ],
        directory,
        extra_environment={"OPENBLAS_NUM_THREADS": "5", "MKL_NUM_THREADS": "5"},
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert (directory / "serial.completed").is_file()
    totals = json.loads((report / "coverage.json").read_text())["totals"]
    assert totals["num_branches"] == totals["covered_branches"] == 2
    assert totals["num_statements"] == totals["covered_lines"] == 4
    assert (report / ".coverage").is_file()
    assert (report / "coverage.xml").is_file()
    assert (directory / "tests-parallel.xml").is_file()
    assert (directory / "tests-serial.xml").is_file()
    assert not (directory / "tests.xml").exists()


@pytest.mark.parametrize("selector", [[], ["-m", "not serial"], ["-k", "parallel"]])
def test_direct_xdist_rejects_serial_but_honors_user_selection(
    tmp_path: Path, selector: list[str]
) -> None:
    """Protect marked tests while allowing both explicit marker and keyword selectors."""
    case = _cases(
        tmp_path / "suite",
        """
        from pathlib import Path
        import pytest
        def test_parallel():
            Path("parallel.executed").write_text("executed")
        @pytest.mark.serial
        def test_exclusive():
            Path("serial.executed").write_text("executed")
        """,
    )
    result = _run(
        ["-m", "pytest", "-p", "scripts.pytest_suite", "-q", "-n", "2", str(case), *selector],
        case.parent,
    )
    if selector:
        assert result.returncode == 0, result.stdout + result.stderr
        assert "1 passed" in result.stdout
        assert (case.parent / "parallel.executed").is_file()
    else:
        assert result.returncode != 0
        assert "Selected @pytest.mark.serial tests cannot run with xdist" in result.stdout
        assert not (case.parent / "parallel.executed").exists()
    assert not (case.parent / "serial.executed").exists()


@pytest.mark.parametrize("failing_phase", ["parallel", "serial", "native-crash"])
def test_failed_phase_removes_stale_reports_and_cannot_publish_coverage(
    tmp_path: Path, failing_phase: str
) -> None:
    """A failed phase stops the run and cannot retain qualified reports from an older run."""
    case = _cases(
        tmp_path / "suite",
        f"""
        import os
        from pathlib import Path
        import pytest
        def test_parallel():
            with Path("parallel.started").open("a") as stream:
                stream.write(os.environ["PYTEST_XDIST_WORKER"] + "\\n")
            if {failing_phase == "native-crash"}:
                os._exit(1)
            assert {failing_phase != "parallel"}
        @pytest.mark.serial
        def test_serial():
            Path("serial.started").write_text("started")
            assert {failing_phase != "serial"}
        """,
    )
    report = case.parent / "reports"
    report.mkdir()
    for name in (".coverage", "coverage.json", "coverage.xml"):
        (report / name).write_text("stale", encoding="utf-8")
    result = _run(
        [
            str(RUNNER),
            "--workers",
            "1",
            "--coverage",
            "--report-dir",
            str(report),
            "--",
            "-q",
            str(case),
        ],
        case.parent,
    )
    assert result.returncode == 1, result.stdout + result.stderr
    assert (case.parent / "serial.started").exists() == (failing_phase == "serial")
    assert not list(report.iterdir())
    if failing_phase == "native-crash":
        assert (case.parent / "parallel.started").read_text().splitlines() == ["gw0"]
        assert "crashed while running" in result.stdout
        assert "test_parallel" in result.stdout
        assert "INTERNALERROR" not in result.stdout + result.stderr


@pytest.mark.parametrize("selector,expected", [(["-m", "serial"], 0), (["-k", "absent"], 5)])
def test_empty_phase_is_allowed_but_empty_entire_selection_is_not(
    tmp_path: Path, selector: list[str], expected: int
) -> None:
    """Keep focused serial selections valid without treating an empty suite as a pass."""
    case = _cases(
        tmp_path / "suite",
        """
        import pytest
        @pytest.mark.serial
        def test_serial():
            assert True
        """,
    )
    result = _run(
        [str(RUNNER), "--workers", "1", "--", "-q", str(case), *selector],
        case.parent,
    )
    assert result.returncode == expected, result.stdout + result.stderr


@pytest.mark.parametrize("flag", ["-nauto", "--max-worker-restart=1"])
def test_runner_rejects_environment_scheduling_flags_before_launch(
    tmp_path: Path, flag: str
) -> None:
    """Environment flags cannot bypass the runner's bounded worker ownership."""
    result = _run(
        [str(RUNNER), "--workers", "1"],
        tmp_path,
        extra_environment={"PYTEST_ADDOPTS": flag},
    )
    assert result.returncode == 2
    assert "pytest scheduling, coverage and phase options are reserved" in result.stderr
    assert "Running:" not in result.stdout
