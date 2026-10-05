"""Protect release authorization, optional GPU scheduling and native CI acceptance gates."""

from __future__ import annotations

import shlex
import tomllib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def workflow() -> dict[str, Any]:
    """Read the four workflows without YAML 1.1 converting trigger keys to booleans."""
    return {
        path.stem: yaml.load(path.read_text(), Loader=yaml.BaseLoader)
        for path in (ROOT / ".github/workflows").glob("*.yml")
    }


def _condition(
    expression: str,
    *,
    event: str,
    ref: str,
    repository: str,
    full_native: bool,
    publish: bool = False,
) -> bool:
    """Evaluate the workflow's simple scheduling expression against independent event records."""
    if expression.startswith("${{"):
        expression = expression[3:-2].strip()
    expression = expression.replace("&&", " and ").replace("||", " or ")
    return bool(
        eval(
            expression,
            {"__builtins__": {}},
            {
                "github": SimpleNamespace(event_name=event, ref=ref, repository=repository),
                "inputs": SimpleNamespace(full_native=full_native, publish=publish),
                "startsWith": str.startswith,
            },
        )
    )


@pytest.mark.parametrize(
    "event,ref,repository,full_native,publish,gpu",
    [
        ("pull_request", "refs/pull/12/merge", "volpatto/pymhm", True, False, False),
        ("push", "refs/heads/main", "volpatto/pymhm", False, False, False),
        ("push", "refs/tags/v0.1.0", "volpatto/pymhm", False, True, False),
        ("push", "refs/tags/v0.1.0", "another/pymhm", False, False, False),
        ("workflow_dispatch", "refs/tags/v0.1.0", "volpatto/pymhm", False, False, False),
        ("workflow_dispatch", "refs/heads/main", "volpatto/pymhm", True, False, True),
    ],
)
def test_actual_conditions_require_a_release_push_or_explicit_native_dispatch(
    workflow: dict[str, Any],
    event: str,
    ref: str,
    repository: str,
    full_native: bool,
    publish: bool,
    gpu: bool,
) -> None:
    """PRs, forks and manual tag runs cannot inherit release authority or automatic GPU access."""
    context = {"event": event, "ref": ref, "repository": repository, "full_native": full_native}
    release = workflow["publish-pypi"]["jobs"]
    assert _condition(release["publish"]["if"], **context) is publish
    assert _condition(release["docs"]["if"], **context) is publish
    assert _condition(workflow["docs"]["jobs"]["deploy"]["if"], **context, publish=True) is publish
    assert not _condition(workflow["docs"]["jobs"]["deploy"]["if"], **context)
    assert _condition(workflow["tests"]["jobs"]["full-native"]["if"], **context) is gpu


def test_pixi_version_is_shared_by_both_workspaces_and_every_ci_install(
    workflow: dict[str, Any],
) -> None:
    """Local and CI workspace validation use the same explicitly qualified Pixi version."""
    required = tomllib.loads((ROOT / "pixi.toml").read_text())["workspace"]["requires-pixi"]
    native = tomllib.loads((ROOT / "tools/amgx/pixi.toml").read_text())
    assert required.startswith("==") and native["workspace"]["requires-pixi"] == required
    for definition in workflow.values():
        for job in definition["jobs"].values():
            for step in job.get("steps", []):
                if step.get("uses", "").startswith("prefix-dev/setup-pixi@"):
                    assert step["with"]["pixi-version"] == "v" + required.removeprefix("==")


def test_lock_acceptance_checks_the_actual_complete_workspaces_before_job_fanout(
    workflow: dict[str, Any],
) -> None:
    """An unsatisfied optional platform fails validation before any environment is installed."""
    jobs = workflow["tests"]["jobs"]
    workspace = jobs["workspace"]
    setup = next(step for step in workspace["steps"] if "setup-pixi@" in step.get("uses", ""))
    assert setup["with"]["run-install"] == "false"
    commands = [shlex.split(step["run"]) for step in workspace["steps"] if "run" in step]
    assert len(commands) == 2
    for command in commands:
        assert command[:2] == ["pixi", "list"]
        assert "--locked" in command and "--no-install" in command
    assert "--manifest-path" not in commands[0] and "test-core" in commands[0]
    assert commands[1][commands[1].index("--manifest-path") + 1] == "tools/amgx/pixi.toml"
    for name in ("core", "integration", "full-native"):
        assert jobs[name]["needs"] == "workspace"
    assert all(
        "--frozen" not in step.get("run", "")
        for definition in workflow.values()
        for job in definition["jobs"].values()
        for step in job.get("steps", [])
    )


def test_parallel_matrices_preserve_windows_and_native_backend_acceptance(
    workflow: dict[str, Any],
) -> None:
    """Dedicated workflows retain portable coverage and each native integration platform."""
    assert {key: definition["name"] for key, definition in workflow.items()} == {
        "tests": "Tests",
        "lint-and-quality": "Lint and Quality",
        "docs": "Docs",
        "publish-pypi": "Publish to PyPI",
    }
    for name in ("tests", "lint-and-quality", "docs"):
        trigger = workflow[name]["on"]
        assert {"pull_request", "push", "workflow_dispatch", "workflow_call"} == set(trigger)
        assert trigger["push"] == {"branches": ["main"]}
    jobs = workflow["tests"]["jobs"]
    assert set(jobs) == {"workspace", "core", "integration", "full-native"}
    core = jobs["core"]
    assert {(row["os"], row["environment"]) for row in core["strategy"]["matrix"]["include"]} == {
        ("ubuntu-latest", "test-core"),
        ("windows-latest", "test-core"),
        ("macos-15", "test-core"),
        ("macos-15-intel", "test-core"),
        ("ubuntu-latest", "test-py311"),
        ("ubuntu-latest", "test-py312"),
    }
    integration = jobs["integration"]
    assert {(row["kind"], row["os"]) for row in integration["strategy"]["matrix"]["include"]} == {
        ("fem", "ubuntu-latest"),
        ("mpi", "ubuntu-latest"),
        ("intel", "ubuntu-latest"),
        ("intel", "windows-latest"),
        ("meshing", "ubuntu-latest"),
        ("remeshing", "ubuntu-latest"),
        ("visualization", "ubuntu-latest"),
        ("visualization", "windows-latest"),
        ("visualization", "macos-15"),
        ("visualization", "macos-15-intel"),
    }
    assert core["strategy"]["fail-fast"] == integration["strategy"]["fail-fast"] == "false"
    assert any(" test-cov" in step.get("run", "") for step in core["steps"])
    commands = "\n".join(step.get("run", "") for step in integration["steps"])
    for acceptance in (
        "test-fem",
        "native_optional_cpu_integration and petsc",
        'PETSc.Sys.hasExternalPackage("mumps")',
        "test-mpi",
        "import pypardiso",
        "tests/test_windows_portability.py",
        "--require-pardiso",
        "test-meshing",
        "tests/test_metric_freefem.py",
        "test-visualization",
    ):
        assert acceptance in commands


def test_release_waits_for_all_required_checks_and_reuses_the_validated_artifacts(
    workflow: dict[str, Any],
) -> None:
    """A skipped GPU job cannot block releases, while failed required checks cannot be bypassed."""
    definition = workflow["publish-pypi"]
    jobs = definition["jobs"]
    publisher = jobs["publish"]
    assert set(publisher["needs"]) == {"tests", "quality", "docs"}
    assert definition["on"] == {"push": {"tags": ["v*"]}}
    assert "workflow_call" not in definition["on"]
    assert jobs["tests"]["uses"] == "./.github/workflows/tests.yml"
    assert jobs["tests"]["with"] == {"full_native": "false"}
    assert jobs["tests"]["secrets"] == "inherit"
    assert jobs["quality"]["uses"] == "./.github/workflows/lint-and-quality.yml"
    for name in ("tests", "quality"):
        assert "needs" not in jobs[name]
        assert jobs[name]["permissions"] == {"contents": "read"}
    assert jobs["docs"]["needs"] == ["tests", "quality"]
    assert jobs["docs"]["uses"] == "./.github/workflows/docs.yml"
    assert jobs["docs"]["with"] == {"publish": "true"}
    assert jobs["docs"]["if"] == publisher["if"]
    assert jobs["docs"]["permissions"] == {
        "contents": "read",
        "pages": "write",
        "id-token": "write",
    }
    assert "always()" not in publisher["if"]
    assert publisher["environment"]["name"] == "pypi"
    assert publisher["permissions"] == {"contents": "read", "id-token": "write"}
    assert all(item["permissions"] == {"contents": "read"} for item in workflow.values())
    assert all(
        "id-token" not in job.get("permissions", {})
        for name in ("tests", "lint-and-quality")
        for job in workflow[name]["jobs"].values()
    )
    quality = workflow["lint-and-quality"]["jobs"]["quality"]["steps"]
    release_validation = next(
        step for step in quality if "metadata-check --tag" in step.get("run", "")
    )
    assert (
        release_validation["if"]
        == "github.event_name == 'push' && startsWith(github.ref, 'refs/tags/v')"
    )
    assert quality.index(release_validation) < next(
        index for index, step in enumerate(quality) if step.get("run", "").endswith(" build")
    )
    artifact = next(
        step for step in quality if step.get("with", {}).get("name") == "python-distributions"
    )
    assert artifact["with"]["path"] == "dist/" and artifact["with"]["if-no-files-found"] == "error"
    assert publisher["steps"][0]["with"] == {"name": "python-distributions", "path": "dist/"}
    assert publisher["steps"][-1]["uses"] == "pypa/gh-action-pypi-publish@release/v1"
    assert "uses" not in publisher and "runs-on" in publisher
    assert definition["concurrency"]["cancel-in-progress"] == "false"
    groups = {item["concurrency"]["group"] for item in workflow.values()}
    assert len(groups) == len(workflow)


def test_documentation_publishes_the_checked_site_under_the_same_release_gates(
    workflow: dict[str, Any],
) -> None:
    """Pages deploys the checked release site with isolated permissions and serialized updates."""
    definition = workflow["docs"]
    jobs = definition["jobs"]
    option = definition["on"]["workflow_call"]["inputs"]["publish"]
    assert option["type"] == "boolean" and option["default"] == "false"
    assert jobs["build"]["permissions"] == {"contents": "read"}
    docs = jobs["build"]["steps"]
    upload = next(step for step in docs if "upload-pages-artifact@" in step.get("uses", ""))
    validation = next(step for step in docs if step.get("run", "").endswith(" docs-check"))
    publisher = jobs["deploy"]
    assert docs.index(validation) < docs.index(upload)
    assert upload["with"]["path"] == "site/"
    assert upload["if"] == publisher["if"]
    assert publisher["needs"] == "build"
    assert "always()" not in publisher["if"]
    assert publisher["permissions"] == {"contents": "read", "pages": "write", "id-token": "write"}
    assert publisher["environment"] == {
        "name": "github-pages",
        "url": "${{ steps.deployment.outputs.page_url }}",
    }
    assert publisher["concurrency"] == {"group": "github-pages", "cancel-in-progress": "false"}
    assert publisher["steps"] == [
        {"uses": "actions/configure-pages@v6"},
        {
            "name": "Deploy the validated release site",
            "id": "deployment",
            "uses": "actions/deploy-pages@v5",
        },
    ]


def test_complete_native_suite_is_explicit_exclusive_and_uses_all_available_workers(
    workflow: dict[str, Any],
) -> None:
    """GPU execution keeps native setup and acceptance controls without overlapping GPU jobs."""
    definition = workflow["tests"]
    for trigger in ("workflow_dispatch", "workflow_call"):
        option = definition["on"][trigger]["inputs"]["full_native"]
        assert option["type"] == "boolean" and option["default"] == "false"
    native = definition["jobs"]["full-native"]
    assert set(native["runs-on"]) == {"self-hosted", "linux", "x64", "gpu"}
    assert native["concurrency"]["cancel-in-progress"] == "false"
    commands = [step["run"] for step in native["steps"] if "run" in step]
    assert commands == [
        "pixi run --locked -e test test-setup-amgx",
        "pixi run --locked -e test test-dependencies",
        "pixi run --locked -e test test-cov",
    ]
    for name in ("tests", "lint-and-quality", "docs"):
        environment = workflow[name]["env"]
        assert "PYMHM_TEST_WORKERS" not in environment
        for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            assert environment[variable] == "1"
    assert not any("--workers" in command for command in commands)
