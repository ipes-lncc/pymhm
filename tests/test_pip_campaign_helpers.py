"""Downloaded case companions preserve inputs, archive identity and spawn outside a clone."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from scripts.build_notebook_companions import build_companion, module_sources


def _environment(workspace: Path) -> dict[str, str]:
    """Select one writable case directory and one native thread for child interpreters."""
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment["PYMHM_WORKSPACE"] = str(workspace)
    for name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
        environment[name] = "1"
    return environment


class _QuietHandler(SimpleHTTPRequestHandler):
    """Serve actual companion/resource bytes without request messages in test output."""

    def log_message(self, format: str, *args: object) -> None:
        """Suppress transport diagnostics without changing HTTP responses."""


@contextmanager
def _downloaded_companions(
    tmp_path: Path,
    modules: Sequence[str],
    *,
    configurations: Sequence[str] = (),
    resources: Sequence[str] = (),
) -> Iterator[Path]:
    """Use the website's closure builder and the generic downloader outside the checkout."""
    from pymhm.io.workspace import materialize_resources, workspace_from_archive

    repository = Path(__file__).resolve().parents[1]
    site = tmp_path / "site"
    site.mkdir()
    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(_QuietHandler, directory=str(site)))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_port}/"
    try:
        inventory = json.loads((repository / "examples/resource_manifest.json").read_text())
        selected = {}
        for name in resources:
            record = dict(inventory["remote"][name])
            relative = f"downloads/{record['sha256']}/{Path(name).name}"
            destination = site / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(repository / name, destination)
            record["url"] = base_url + relative
            selected[name] = record
        descriptor = build_companion(
            repository,
            [*module_sources(repository, modules), *configurations],
            site,
            name="physical-case-companion.zip",
            resources=selected,
        )
        workspace = tmp_path / "workspace"
        workspace_from_archive(
            base_url + f"downloads/{descriptor['sha256']}/physical-case-companion.zip",
            sha256=descriptor["sha256"],
            directory=workspace,
        )
        assert not (workspace / "src").exists()
        assert not (workspace / "examples/resource_manifest.json").exists()
        assert not (workspace / "pixi.lock").exists()
        assert (workspace / ".pymhm-resources.json").is_file()
        assert all(not (workspace / name).exists() for name in resources)
        materialize_resources(resources, workspace)
        yield workspace
    finally:
        server.shutdown()
        thread.join(timeout=10)
        server.server_close()


def test_case_inputs_and_source_manifests_without_checkout(tmp_path: Path) -> None:
    """A physical acquisition keeps literal source/input bytes with no invented Git/lock."""
    script = r"""
import json
from pathlib import Path
import numpy as np
from pymhm.io.provenance import file_digest, workspace_revision
from pymhm.io.workspace import case_workspace, source_file
from examples.helmholtz_campaign import source_hashes
from examples.three_layer_2017 import load_case
from examples.three_layer_2017_acquire import prepare
from examples.transport_random_problem import INPUT

workspace = case_workspace()
assert not (workspace / 'src').exists()
assert not (workspace / 'pixi.lock').exists()
assert workspace_revision(workspace) is None
sources = source_hashes()
assert 'pixi.lock' not in sources
assert sources['examples/helmholtz_campaign.py'] == file_digest(
    source_file('examples/helmholtz_campaign.py', root=workspace)
)
assert all(file_digest(source_file(name, root=workspace)) == digest
           for name, digest in sources.items())
case = load_case()
output = workspace / 'acquired'
record = prepare(case, output)
assert prepare(case, output) == record
assert 'pixi.lock' not in record['source_sha256']
with np.load(output / record['archive'], allow_pickle=False) as arrays:
    np.testing.assert_array_equal(arrays['macro_points'], case.mesh.points)
    np.testing.assert_array_equal(arrays['macro_cells'], case.mesh.cells)
    np.testing.assert_array_equal(arrays['density_kg_m3'], case.density)
    np.testing.assert_array_equal(arrays['wavelet'], case.wavelet(arrays['physical_time_s']))
assert INPUT.is_file()
owner = source_file('examples/three_layer_2017_acquire.py', root=workspace)
assert owner.is_relative_to(workspace)
previous_source = owner.read_bytes()
previous_record = (output / 'inputs.json').read_bytes()
owner.write_bytes(previous_source + b'\n# changed source identity\n')
try:
    prepare(case, output)
except ValueError:
    pass
else:
    raise AssertionError('changed companion sources must reject an acquired identity')
assert (output / 'inputs.json').read_bytes() == previous_record
owner.write_bytes(previous_source)
print(json.dumps({'macro_cells': len(case.mesh.cells), 'sources': len(sources)}))
"""
    with _downloaded_companions(
        tmp_path,
        [
            "examples.helmholtz_campaign",
            "examples.three_layer_2017_acquire",
            "examples.transport_random_problem",
        ],
        configurations=[
            "examples/data/three-layer-2017/case.json",
            "examples/data/three-layer-2017/macro-mesh.json",
        ],
        resources=[
            "examples/data/transport-random-2015/permeability.json",
            "examples/data/three-layer-2017/horizons.csv",
        ],
    ) as workspace:
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=workspace,
            env=_environment(workspace),
            check=False,
            capture_output=True,
            text=True,
            timeout=120,
        )
    assert result.returncode == 0, result.stderr
    record = json.loads(result.stdout)
    assert record["macro_cells"] == 341
    assert record["sources"] > 250


@pytest.mark.parametrize("formulation", ["primal", "mixed"])
@pytest.mark.parametrize("entrypoint", ["downloaded", "module"])
def test_downloaded_entrypoint_spawns_canonical_providers(
    tmp_path: Path, formulation: str, entrypoint: str
) -> None:
    """A downloaded script and its minimal companions preserve Neumann equations under spawn."""
    repository = Path(__file__).resolve().parents[1]
    companions = tmp_path / "examples"
    companions.mkdir()
    for name in ("__init__.py", "tutorial_local_provider.py"):
        (companions / name).write_bytes((repository / "examples" / name).read_bytes())
    source = repository / "examples/tutorial_local_provider.py"
    downloaded = tmp_path / "darcy_example.py"
    downloaded.write_bytes(source.read_bytes())
    command = (
        [sys.executable, str(downloaded)]
        if entrypoint == "downloaded"
        else [sys.executable, "-m", "examples.tutorial_local_provider"]
    )
    result = subprocess.run(
        [
            *command,
            "--formulation",
            formulation,
            "--boundary",
            "neumann",
            "--backend",
            "process",
            "--workers",
            "2",
            "--batch-size",
            "1",
            "--element-backend",
            "portable",
        ],
        cwd=tmp_path,
        env=_environment(tmp_path),
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    diagnostics = json.loads(result.stdout)
    assert diagnostics["formulation"] == formulation
    assert diagnostics["boundary"] == "neumann"
    assert diagnostics["field_L2_errors"]["Darcy_flux"] < 1e-12
    assert diagnostics["field_L2_errors"]["divergence"] < 1e-12
    assert diagnostics["global_original_relative_residual"] < 1e-12
    assert max(diagnostics["maximum_local_original_row_l2_by_block"].values()) < 1e-12


def test_transport_companion_acquires_real_field_and_rejects_changed_source(tmp_path: Path) -> None:
    """A tiny real P1/P0 acquisition verifies source identity before final acceptance."""
    script = r"""
import json
from pathlib import Path
import numpy as np
from pymhm.io.provenance import file_digest
from pymhm.io.workspace import case_workspace, source_file
from examples import transport_coefficient_controls as owner

workspace = case_workspace()
output = workspace / 'acquired'
row = owner.acquire((1, 1.0, 2), output_directory=output)
assert row['source_changed_during_run'] is False
assert row['macro_triangles'] == 4
assert row['local_degree'] == 1 and row['trace_degree'] == 0
assert row['original_hybrid_residual'] < 1e-12
assert set(row['quadrature']) == {'8', '12'}
assert all(file_digest(source_file(label, root=workspace)) == digest
           for label, digest in row['source_hashes'].items())
archive = output / row['archive']
assert file_digest(archive) == row['archive_sha256']
with np.load(archive, allow_pickle=False) as field:
    assert field['macro_cells'].shape == (4, 3)
    assert field['coefficients'].shape[0] == 4
    assert np.isfinite(field['coefficients']).all()
source = source_file('examples/transport_coefficient_controls.py', root=workspace)
assert source.is_relative_to(workspace)
previous = source.read_bytes()
original_norm = owner.norm_contribution
changed = False

def modify_after_physical_field(task):
    global changed
    result = original_norm(task)
    if not changed:
        source.write_bytes(previous + b'\n# changed during acquisition\n')
        changed = True
    return result

owner.norm_contribution = modify_after_physical_field
rejected = workspace / 'rejected'
try:
    owner.acquire((1, 1.0, 2), output_directory=rejected)
except AssertionError:
    pass
else:
    raise AssertionError('changed sources must not produce a final accepted case record')
finally:
    source.write_bytes(previous)
    owner.norm_contribution = original_norm
assert changed
assert not list(rejected.glob('*.json')) or all(
    path.name.endswith('.progress.json') for path in rejected.glob('*.json')
)
assert list(rejected.glob('*.progress.json'))
assert json.loads(archive.with_suffix('.json').read_text()) == row
print(json.dumps({'verified_macro_cells': row['macro_triangles'], 'changed_source_rejected': True}))
"""
    with _downloaded_companions(tmp_path, ["examples.transport_coefficient_controls"]) as workspace:
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=workspace,
            env=_environment(workspace),
            check=False,
            capture_output=True,
            text=True,
            timeout=120,
        )
    assert result.returncode == 0, result.stderr
    record = json.loads(result.stdout.splitlines()[-1])
    assert record == {"verified_macro_cells": 4, "changed_source_rejected": True}
