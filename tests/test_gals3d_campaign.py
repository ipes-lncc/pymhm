"""Light provenance tests for safely resuming accepted elasticity campaign fields."""

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
for name in ("gals3d_data", "solve_gals3d"):
    specification = importlib.util.spec_from_file_location(name, EXAMPLES / f"{name}.py")
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
resume_report = sys.modules["solve_gals3d"].resume_report


def test_validated_resume_preserves_row_source_provenance(tmp_path):
    """Unchanged sources and bytes preserve row signatures; changed drivers are explicit."""
    archive = tmp_path / "accepted.npz"
    archive.write_bytes(b"immutable numerical archive")
    old = {"examples/solve_gals3d.py": "old", "operator.py": "operator"}
    row = dict(
        case="gals-p1",
        formulation="gals",
        degree=1,
        local_refinement=4,
        face_degree=1,
        face_subdivisions=1,
        assembly_quadrature_order=7,
        error_quadrature_orders=[8, 9],
        shear="1+x/4+z/8",
        fields=archive.name,
        fields_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
        displacement_l2=0.1,
        pressure_l2=0.1,
        gradient_l2=0.1,
        stress_l2=0.1,
        compressibility_l2=0.1,
        backward_residual=1e-15,
    )
    report = dict(source_sha256=old, rows=[row], primal_control=[])
    path = tmp_path / "campaign.json"
    path.write_text(json.dumps(report))
    result = resume_report(path, old)
    assert result["acquisition_sources"] == [old]
    assert result["rows"][0]["acquisition_source_index"] == 0
    path.write_text(json.dumps(result))
    assert resume_report(path, old) == result
    for changed in ({**old, "examples/solve_gals3d.py": "new"}, {**old, "operator.py": "changed"}):
        with pytest.raises(ValueError, match="sources differ"):
            resume_report(path, changed)
    archive.write_bytes(b"different numerical archive")
    with pytest.raises(ValueError, match="digest"):
        resume_report(path, old)
