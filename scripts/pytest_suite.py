"""Select explicitly marked test phases and reject unsafe direct xdist runs."""

from __future__ import annotations

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register the runner's explicit phase; ordinary pytest has no phase filter."""
    parser.addoption(
        "--pymhm-test-phase",
        choices=("parallel", "serial"),
        default=None,
        help="Internal selection used by scripts/run_tests.py.",
    )


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Filter after user selectors, then protect serial tests from worker execution."""
    phase = config.getoption("pymhm_test_phase")
    if phase is not None:
        selected = []
        deselected = []
        for item in items:
            serial = item.get_closest_marker("serial") is not None
            (selected if serial == (phase == "serial") else deselected).append(item)
        items[:] = selected
        config.hook.pytest_deselected(items=deselected)
    distributed = hasattr(config, "workerinput") or bool(config.getoption("numprocesses", 0))
    if distributed and any(item.get_closest_marker("serial") is not None for item in items):
        # Collection reports are transported to the xdist controller; a worker
        # UsageError can otherwise lose its explanation during worker shutdown.
        items.clear()
        config.hook.pytest_collectreport(
            report=pytest.CollectReport(
                nodeid="serial test selection",
                outcome="failed",
                longrepr=(
                    "Selected @pytest.mark.serial tests cannot run with xdist. "
                    "Use pixi run -e test test, or exclude them with -m 'not serial'."
                ),
                result=[],
            )
        )
