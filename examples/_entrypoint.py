"""Resolve repository example imports when a driver is executed by file path."""

import sys
from pathlib import Path


def prepare_example_imports(script: str, package: str | None) -> None:
    """Expose the script's repository root for direct execution without changing cwd.

    Package imports already resolve their dependencies. A file entrypoint under
    ``examples/`` needs its parent directory on the module search path so that
    canonical example imports and their spawn-worker callables remain importable.
    """
    if package:
        return
    root = str(Path(script).resolve().parent.parent)
    if root not in sys.path:
        sys.path.insert(0, root)
