"""Importable scientific examples maintained in the PyMHM repository.

Case definitions, acquisition helpers and spawn-worker callables use the public
PyMHM numerical API. Writable outputs belong to the selected working directory.
"""

import json
from pathlib import Path

from pymhm.io.workspace import case_workspace, register_resources

_inventory_path = Path(__file__).with_name("resource_manifest.json")
if _inventory_path.is_file():
    _inventory = json.loads(_inventory_path.read_text())
    register_resources(
        case_workspace(),
        {
            "resources": _inventory["remote"],
            "unavailable": _inventory["unavailable"],
        },
    )
