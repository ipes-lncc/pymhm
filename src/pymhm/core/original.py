"""Monolithic assembly of explicitly declared leaf A/B/C/D coefficient equations."""

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import sparse

from pymhm.core.multiscale import MultiscaleSystem


@dataclass(frozen=True)
class OriginalBlocks:
    """Original local fields followed by global trace coordinates, without elimination.

    ``local_offsets`` delimits each original local vector. ``trace_offset``
    starts the global trace vector. Prescribed trace indices must therefore
    be shifted by this offset before a caller applies essential data. No gauge,
    symmetry, invertibility or physical interpretation is inferred.
    """

    matrix: Any
    load: Any
    local_offsets: Any
    trace_offset: int


def assemble_original_blocks(system: MultiscaleSystem) -> OriginalBlocks:
    """Assemble the declared uncondensed equations without invoking a physical solver.

    The local rows are ``A u+B lambda=L``. Global rows sum
    ``C u+D lambda=g`` and the user's additional global Equation. The signed
    C is taken literally from the compiled test coupling, independently of B.
    Matrix contributions are scattered in original local/global coordinate
    order. The source-dependent rows come from the supplied system, including
    updates made through OfflineMultiscaleSystem.

    This operation currently accepts leaf problems with no retained modes.
    Recursive hierarchies and retained-mode gauges require a separately
    declared original layout and are rejected instead of being flattened.
    It builds a larger sparse operator for direct comparisons; it does not
    establish stability or constitute an independent physical discretization.
    """
    if not isinstance(system, MultiscaleSystem):
        raise TypeError("original blocks require a MultiscaleSystem")
    if any(record.child is not None for record in system.cells) or any(system.layout.coarse_sizes):
        raise ValueError("original blocks currently require leaf equations without retained modes")
    offsets = np.r_[0, np.cumsum([len(response.problem.load) for response in system.responses])]
    volume_size = int(offsets[-1])
    size = volume_size + system.trace_size
    matrix = sparse.lil_matrix((size, size), dtype=system.matrix.dtype)
    load = np.zeros(size, dtype=system.rhs.dtype)
    # The executed additional form is retained literally. No harmonic lift,
    # Schur complement or numerical subtraction enters this original assembly.
    direct = system.global_matrix.copy().tolil()
    for response, record in zip(system.responses, system.cells, strict=True):
        ids = response.problem.trace_dofs
        direct[np.ix_(ids, ids)] += record.equations.matrix
    matrix[volume_size:, volume_size:] = direct
    load[volume_size:] = system.global_load
    for cell, (response, record) in enumerate(zip(system.responses, system.cells, strict=True)):
        problem = response.problem
        fields = np.arange(offsets[cell], offsets[cell + 1])
        traces = volume_size + problem.trace_dofs
        matrix[np.ix_(fields, fields)] = problem.matrix
        matrix[np.ix_(fields, traces)] = problem.coupling
        matrix[np.ix_(traces, fields)] = -problem.test_coupling.T
        load[fields] = problem.load
        np.add.at(load, traces, record.equations.load)
    offsets.setflags(write=False)
    return OriginalBlocks(matrix.tocsc(), load, offsets, volume_size)
