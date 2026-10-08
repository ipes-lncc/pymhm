"""Execute scientific notebooks and retain outputs without modifying sources."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

import nbformat
from jupyter_client import KernelManager
from nbclient import NotebookClient

from pymhm.io.workspace import case_workspace
from scripts.notebook_data import (
    DEFAULT_MAX_BYTES,
    dependency_plan,
    notebook_selector,
    select_notebooks,
    validate_archives,
)
from scripts.notebook_reproduction import (
    execute_in_environment,
    execution_inputs,
    native_requirements,
    notebook_contract,
    notebook_workspace,
    preparation_plan,
    prepare_notebook_inputs,
    required_environment,
    stage_notebook_resources,
    validate_native_requirements,
    validate_reproduction_manifest,
)


@contextmanager
def _temporary_environment(values: Mapping[str, str]) -> Iterator[None]:
    """Scope preparation flags without modifying the caller's Python environment."""
    previous = {name: os.environ.get(name) for name in values}
    try:
        os.environ.update(values)
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def output_notebook_path(root: Path, source: Path) -> Path:
    """Keep problem folders in outputs and separate external sources by path digest.

    External notebook paths are encoded by the SHA-256 of the absolute
    UTF-8 path, preventing equal basenames from overwriting each other's outputs.
    Source notebooks are never overwritten.
    """
    source = source.resolve()
    base = (root / "notebooks").resolve()
    output = root / "build" / "notebooks"
    if source.is_relative_to(base):
        return output / source.relative_to(base)
    digest = hashlib.sha256(str(source).encode("utf-8")).hexdigest()
    return output / "external" / digest / source.name


def main() -> None:
    """Run selected notebooks with a bounded timeout and reproducible paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths", nargs="*", help="Notebook IDs, paths or problem folders (default: all)"
    )
    parser.add_argument("--timeout", type=int, default=600, help="Cell timeout in seconds")
    parser.add_argument(
        "--plan", action="store_true", help="List selected inputs and public preparation commands"
    )
    parser.add_argument(
        "--no-prepare",
        action="store_true",
        help="Require existing inputs without running producers",
    )
    parser.add_argument(
        "--historical",
        action="store_true",
        help="Require and replay original historical field archives",
    )
    parser.add_argument(
        "--check", action="store_true", help="Also validate every declared source and recipe"
    )
    parser.add_argument("--no-dispatch", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--dispatch",
        action="store_true",
        help="Explicit development-only dispatch into a locked Pixi environment",
    )
    parser.add_argument(
        "--study",
        action="store_true",
        help="Acquire complete declared current studies before execution",
    )
    parser.add_argument(
        "--max-data-bytes",
        type=int,
        default=DEFAULT_MAX_BYTES,
        help="Explicit size budget for selected computed field archives",
    )
    args = parser.parse_args()
    if args.dispatch and args.no_dispatch:
        parser.error("--dispatch and --no-dispatch cannot be combined")
    root = case_workspace()
    try:
        notebooks = select_notebooks(root, args.paths, allow_external=True)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    if not notebooks:
        raise SystemExit("No notebooks found")
    if args.timeout < 1:
        raise SystemExit("Cell timeout must be positive")
    if args.max_data_bytes < 1:
        raise SystemExit("Archive size budget must be positive")
    if args.check:
        try:
            validate_reproduction_manifest(root, notebooks)
        except ValueError as error:
            raise SystemExit(str(error)) from error
    preparation_environment = {
        "PYMHM_WORKSPACE": str(root),
        "PYMHM_NOTEBOOK_HISTORICAL": "1" if args.historical else "0",
        "PYMHM_NOTEBOOK_STUDY": "1" if args.study else "0",
        "PYMHM_NOTEBOOK_MAX_DATA_BYTES": str(args.max_data_bytes),
    }
    if not args.plan:
        try:
            if not args.dispatch:
                validate_native_requirements(root, notebooks)
            with _temporary_environment(preparation_environment):
                for source in notebooks:
                    selector = notebook_selector(root, source)
                    if selector is not None:
                        stage_notebook_resources(selector)
        except ValueError as error:
            raise SystemExit(str(error)) from error
    data = dependency_plan(
        root, *execution_inputs(root, notebooks, historical=args.historical, study=args.study)
    )
    plan = preparation_plan(root, notebooks, data, study=args.study, historical=args.historical)
    if args.plan:
        print(json.dumps({**data, **plan}, indent=2), flush=True)
        return
    try:
        if not args.no_prepare:
            prepare_notebook_inputs(root, plan)
        validate_archives(
            root,
            dependency_plan(
                root,
                *execution_inputs(root, notebooks, historical=args.historical, study=args.study),
            ),
            args.max_data_bytes,
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error
    output = root / "build" / "notebooks"
    output.mkdir(parents=True, exist_ok=True)
    kernel_environment = {
        **os.environ,
        "MPLBACKEND": "module://matplotlib_inline.backend_inline",
        **preparation_environment,
    }
    receipt = {
        "schema": "pymhm-notebook-execution-v1",
        "started_utc": datetime.now(UTC).isoformat(),
        "python_executable": sys.executable,
        "pixi_environment": os.environ.get("PIXI_ENVIRONMENT_NAME"),
        "lockfile_sha256": hashlib.sha256((root / "pixi.lock").read_bytes()).hexdigest()
        if (root / "pixi.lock").is_file()
        else None,
        "preparation": []
        if args.no_prepare
        else plan.get("executed_preparation", plan["preparation"]),
        "acquisition_id": plan.get("acquisition_id"),
        "notebooks": [],
    }
    for path in notebooks:
        destination = output_notebook_path(root, path)
        source_identity = hashlib.sha256(path.read_bytes()).hexdigest()
        environment = required_environment(root, path) if args.dispatch else None
        if environment is not None:
            flags = ["--timeout", str(args.timeout), "--max-data-bytes", str(args.max_data_bytes)]
            if args.no_prepare:
                flags.append("--no-prepare")
            if args.historical:
                flags.append("--historical")
            if args.study:
                flags.extend(("--study", "--no-prepare"))
            execute_in_environment(root, path, environment, flags)
            if hashlib.sha256(path.read_bytes()).hexdigest() != source_identity:
                raise RuntimeError(f"Notebook source changed during execution: {path}")
            receipt["notebooks"].append(
                {
                    "source": str(path),
                    "source_sha256": source_identity,
                    "output": str(destination),
                    "output_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
                    "execution_environment": environment,
                }
            )
            (output / "execution.json").write_text(
                json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
            )
            continue
        print(f"Executing {path}", flush=True)
        selector = notebook_selector(root, path)
        per_notebook_environment = {
            **kernel_environment,
            "PYMHM_NOTEBOOK_SOURCE": str(path.resolve()),
            "PYMHM_NOTEBOOK_PREPARED": selector or "",
        }
        if selector is not None:
            # The producers above run once. Workspace staging acquires this
            # selector's data and checks it; the kernel repeats validation only.
            with _temporary_environment(
                {**preparation_environment, "PYMHM_NOTEBOOK_PREPARED": selector}
            ):
                notebook_workspace(selector)
        notebook = nbformat.read(path, as_version=4)
        manager = KernelManager(kernel_name="python3")
        if manager.kernel_spec is None:
            raise RuntimeError("The notebook extra requires an installed Python ipykernel")
        manager.kernel_spec.argv = [
            sys.executable,
            "-m",
            "ipykernel_launcher",
            "-f",
            "{connection_file}",
        ]
        client = NotebookClient(
            notebook,
            km=manager,
            timeout=args.timeout,
            kernel_name="python3",
            resources={"metadata": {"path": str(root)}},
        )
        client.execute(env=per_notebook_environment, cleanup_kc=True)
        if hashlib.sha256(path.read_bytes()).hexdigest() != source_identity:
            raise RuntimeError(f"Notebook source changed during execution: {path}")
        notebook.metadata["pymhm_execution"] = {
            "source_sha256": source_identity,
            "python_executable": sys.executable,
            "execution_contract": notebook_contract(root, path),
            "native_requirements_checked": sorted(
                native_requirements(path) | set(notebook_contract(root, path)["requires"])
            ),
        }
        destination.parent.mkdir(parents=True, exist_ok=True)
        nbformat.write(notebook, destination)
        receipt["notebooks"].append(
            {
                "source": str(path),
                "source_sha256": source_identity,
                "output": str(destination),
                "output_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
            }
        )
        (output / "execution.json").write_text(
            json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
        )
    print(f"Executed {len(notebooks)} notebooks; outputs: {output}")


if __name__ == "__main__":
    main()
