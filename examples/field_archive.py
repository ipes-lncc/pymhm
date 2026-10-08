"""Scientific archive helpers for complete executed public field definitions.

These helpers store acquisition data, not numerical algorithms. The caller
verifies the archive's recorded SHA256 before explicitly loading its trusted
pickled definition. Geometry, coordinate maps, reconstruction and every executed
basis factor travel together; a historical coefficient-only archive returns no
definition. A separate matrix is included only for a declared single-matrix basis.
"""

from __future__ import annotations

import pickle
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from pymhm.postprocessing.fields import DiscreteField, FieldDefinition


def field_archive_arrays(
    fields: Sequence[DiscreteField], *, prefix: str = "scalar"
) -> dict[str, Any]:
    """Return NPZ arrays containing full executed definitions and their coefficient vectors.

    Keys use ``prefix_index`` followed by definition, basis_digest and
    coefficients. ``basis_matrix`` is an additional key when the definition
    declares a single coordinate matrix. Composite Piola bases retain their
    polynomial, moment, orientation and mapping factors in the full definition.
    Definitions are explicit pickle bytes in uint8 arrays, so opening the NPZ
    never implicitly deserializes a Python object.
    """
    arrays: dict[str, Any] = {}
    for index, field in enumerate(fields):
        key = f"{prefix}_{index}"
        definition = field.definition
        arrays[f"{key}_definition"] = np.frombuffer(pickle.dumps(definition), dtype=np.uint8)
        try:
            matrix = definition.basis_matrix
        except TypeError:
            pass  # A composed basis declares its factors in the full definition.
        else:
            arrays[f"{key}_basis_matrix"] = matrix
        arrays[f"{key}_basis_digest"] = np.asarray(definition.basis_digest)
        arrays[f"{key}_coefficients"] = field.coefficients
    return arrays


def load_trusted_field(
    archive: Mapping[str, Any], index: int, *, prefix: str = "scalar"
) -> DiscreteField | None:
    """Replay a trusted definition, checking its digest and any separate basis matrix.

    The archive file's own source/field SHA256 is checked by its scientific
    reader before this explicit deserialization. Missing definitions identify
    historical coefficient-only data; their legacy replay cannot assert a
    persisted executed-basis identity. A separate single matrix is optional;
    composite scalar/vector bases replay all factors from their full definition.
    No polynomial basis is regenerated here.
    """
    key = f"{prefix}_{index}"
    if f"{key}_definition" not in archive:
        return None
    definition = pickle.loads(np.asarray(archive[f"{key}_definition"], dtype=np.uint8).tobytes())
    if not isinstance(definition, FieldDefinition):
        raise TypeError("field archive must contain a public FieldDefinition")
    if str(archive[f"{key}_basis_digest"]) != definition.basis_digest:
        raise ValueError("field archive does not match its executed basis definition")
    if f"{key}_basis_matrix" in archive:
        try:
            matrix = definition.basis_matrix
        except TypeError as error:
            raise ValueError("field archive matrix has no executed basis definition") from error
        if not np.array_equal(archive[f"{key}_basis_matrix"], matrix):
            raise ValueError("field archive does not match its executed basis definition")
    return DiscreteField(definition, archive[f"{key}_coefficients"])
