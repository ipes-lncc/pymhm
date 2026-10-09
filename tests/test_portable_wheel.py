"""Wheel ownership checks accept equivalent paths and reject external imports."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.check_portable_wheel import _INSTALLED_IDENTITY_PROGRAM


@pytest.mark.parametrize("expected", ["installation", "parent_alias", "external"])
def test_wheel_identity_resolves_both_paths_and_rejects_external_imports(
    tmp_path: Path, expected: str
) -> None:
    """Exercise the actual child program with a package owned by a real directory."""
    installation = tmp_path / "installation"
    package = installation / "pymhm"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text('"""Temporary package identity."""\n')
    external = tmp_path / "external"
    external.mkdir()
    expected_root = {
        "installation": installation,
        "parent_alias": installation / ".." / installation.name,
        "external": external,
    }[expected]
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment["PYMHM_EXPECT_INSTALL_ROOT"] = str(expected_root)
    result = subprocess.run(
        [sys.executable, "-c", _INSTALLED_IDENTITY_PROGRAM],
        cwd=installation,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    if expected == "external":
        assert result.returncode != 0
        assert "outside wheel installation" in result.stderr
    else:
        assert result.returncode == 0, result.stderr
        identity = json.loads(result.stdout)
        assert Path(identity["package"]) == (package / "__init__.py").resolve()
