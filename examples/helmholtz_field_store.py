"""Stream uniform Helmholtz nodal fields without retaining the complete coefficient array."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

from examples.helmholtz_compact_family import CompactFamily, CompactTrace
from pymhm.core.validation import positive_int
from pymhm.io.provenance import file_digest


def write_coefficients(
    family: CompactFamily, trace: CompactTrace, path: Path, *, cell_count: int
) -> dict[str, Any]:
    """Write exact cell-ordered complex coefficients to one atomic NPY file.

    All cells must have the same local nodal count and executed dtype. Fields
    come from the shared reconstruction and retain independent macro limits.
    The caller archives geometry, local degree/refinement and source/basis
    provenance alongside this file. Loading with ``np.load(..., mmap_mode='r')``
    permits sampling without first copying every local coefficient into RAM.
    """
    cell_count = positive_int(cell_count, "cell_count")
    temporary = path.with_suffix(path.suffix + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = family.reconstruct(trace.prepared_coordinates)
    balance_max, local_residual_max, count = 0.0, 0.0, 0
    try:
        with temporary.open("wb") as stream:
            for coefficients, balance, residual in fields:
                if count == 0:
                    dtype, width = coefficients.dtype, len(coefficients)
                    np.lib.format.write_array_header_2_0(
                        stream,
                        {
                            "descr": np.lib.format.dtype_to_descr(dtype),
                            "fortran_order": False,
                            "shape": (cell_count, width),
                        },
                    )
                if (
                    count >= cell_count
                    or coefficients.shape != (width,)
                    or coefficients.dtype != dtype
                ):
                    raise ValueError(
                        "streamed fields differ from the declared uniform coefficient contract"
                    )
                stream.write(coefficients.tobytes(order="C"))
                balance_max = max(balance_max, abs(balance))
                local_residual_max = max(local_residual_max, residual)
                count += 1
            if count != cell_count:
                raise ValueError("streamed fields differ from the declared cell count")
            stream.flush()
            os.fsync(stream.fileno())
        mapped = np.load(temporary, mmap_mode="r", allow_pickle=False)
        try:
            original_trace_residual = family.verify_fields(trace, mapped)
        finally:
            mapped._mmap.close()
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "coefficient_file": path.name,
        "coefficient_file_sha256": file_digest(path),
        "coefficient_shape": [cell_count, width],
        "coefficient_dtype": dtype.str,
        "coefficient_real_mantissa_bits": np.finfo(dtype).nmant,
        "algebraic_residual": trace.residual,
        "original_field_trace_residual": original_trace_residual,
        "macro_balance_max": float(balance_max),
        "local_equation_residual_max": local_residual_max,
    }
