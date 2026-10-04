"""Checkpoint exact Helmholtz responses in bounded, independently verified batches.

The store retains the executed matrices, Schur blocks and response coefficients.
It delegates all numerical assembly, restriction and reconstruction to the same
owners used by the in-memory campaign. Reading never enables NumPy object pickles.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Iterator, Mapping
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Literal

import numpy as np
from scipy import sparse

from examples.campaign_checkpoint import require_sources
from examples.campaign_provenance import require_equal
from examples.helmholtz_compact_family import CompactLocal, acquire_local_responses
from examples.local_response_cache import ExactResponseCache
from pymhm.core.validation import positive_int
from pymhm.io.provenance import file_digest

_ARRAYS = ("load", "dofs", "source", "lifts", "schur", "rhs", "boundary")


def _sync_directory(directory: Path) -> None:
    """Flush renamed directory entries where directory descriptors are supported."""
    if hasattr(os, "O_DIRECTORY"):
        descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def _write_record(path: Path, record: dict[str, Any]) -> None:
    """Commit a complete finite manifest after flushing its temporary file."""
    temporary = path.with_suffix(".json.tmp")
    with temporary.open("w") as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    _sync_directory(path.parent)


def _write_batch(path: Path, local: list[CompactLocal]) -> None:
    """Persist every numeric buffer without compression or precision conversion."""
    arrays = {}
    for index, item in enumerate(local):
        prefix = f"{index}_"
        arrays.update({prefix + name: getattr(item, name) for name in _ARRAYS})
        arrays[prefix + "fixed_dofs"] = np.asarray(tuple(item.fixed), dtype=np.int64)
        arrays[prefix + "fixed_values"] = np.asarray(tuple(item.fixed.values()), dtype=float)
        for name in ("matrix", "coupling"):
            matrix = getattr(item, name)
            arrays[prefix + name + "_shape"] = np.asarray(matrix.shape, dtype=np.int64)
            arrays[prefix + name + "_format"] = np.asarray(matrix.format)
            for buffer in ("data", "indices", "indptr"):
                arrays[prefix + name + "_" + buffer] = getattr(matrix, buffer)
    temporary = path.with_suffix(".npz.tmp")
    with temporary.open("wb") as stream:
        np.savez(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    _sync_directory(path.parent)


def _read_batch(directory: Path, row: Mapping[str, Any]) -> Iterator[CompactLocal]:
    """Verify the opened archive before decoding its original response buffers."""
    path = directory / row["archive"]
    if file_digest(path) != row["sha256"]:
        raise ValueError("Helmholtz response batch digest differs")
    with np.load(path, allow_pickle=False) as saved:
        for index in range(row["stop"] - row["start"]):
            prefix = f"{index}_"
            matrices = {}
            for name in ("matrix", "coupling"):
                key = prefix + name
                buffers = tuple(saved[key + "_" + part] for part in ("data", "indices", "indptr"))
                constructors = {"csc": sparse.csc_matrix, "csr": sparse.csr_matrix}
                matrix = constructors[str(saved[key + "_format"])](
                    buffers, shape=tuple(saved[key + "_shape"])
                )
                # SciPy may downcast sparse indices; the archive preserves the executed dtype.
                matrix.indices, matrix.indptr = buffers[1:]
                matrices[name] = matrix
            fixed = dict(
                zip(
                    saved[prefix + "fixed_dofs"].tolist(),
                    saved[prefix + "fixed_values"].tolist(),
                    strict=True,
                )
            )
            yield CompactLocal(
                **matrices,
                **{name: saved[prefix + name] for name in _ARRAYS},
                fixed=fixed,
            )


class ResponseStore:
    """Repeatable cell-ordered responses with no retained batch during a global solve.

    A committed manifest identifies exact source and physical-input contracts.
    An incomplete final batch is reacquired; any corrupted committed batch fails
    closed. Construct stores through :meth:`prepare`, including for resumption.
    """

    def __init__(self, directory: Path, record: dict[str, Any]) -> None:
        """Retain only the verified manifest, never the local coefficient arrays."""
        self.directory = directory
        self.record = record

    def __iter__(self) -> Iterator[CompactLocal]:
        """Load one response at a time in unchanged macrocell order."""
        for row in self.record["batches"]:
            yield from _read_batch(self.directory, row)

    @property
    def storage_bytes(self) -> int:
        """Return the acquired numeric-buffer count without loading coefficients."""
        return sum(row["storage_bytes"] for row in self.record["batches"])

    @property
    def acquisition_seconds(self) -> float | None:
        """Sum measured batch assembly/export times, including prior resumed batches.

        Historical batches without acquisition timing return ``None`` rather
        than inventing a zero setup cost. Global assembly, archive validation,
        factorization and field evaluation have separate caller measurements.
        """
        values = [row.get("acquisition_seconds") for row in self.record["batches"]]
        return None if any(value is None for value in values) else float(sum(values))

    @classmethod
    def prepare(
        cls,
        factory: Any,
        directory: Path,
        *,
        sources: Mapping[str, str],
        configuration: Mapping[str, Any],
        batch_size: int = 128,
        backend: str = "serial",
        workers: int = 1,
        local_solver: str = "scipy",
        local_refinement_precision: Literal["double", "extended"] = "double",
        exact_response_cache: bool = False,
    ) -> ResponseStore:
        """Acquire missing batches with the original assembler and exact condensation.

        The caller's configuration must identify materials, domain, source,
        boundaries, spaces and quadrature. Source hashes identify the executed
        numerical owners. Both contracts are required to resume acquisition.
        Execution worker count may change without altering those contracts.
        Nondefault local solver/precision choices are bound to the identity.
        Local source/trace response arrays retain their executed dtype; their
        subsequent global trace factorization has its own solver contract.
        """
        batch_size = positive_int(batch_size, "batch_size")
        if not isinstance(exact_response_cache, bool):
            raise ValueError("exact_response_cache must be a boolean")
        if exact_response_cache and (backend != "serial" or workers != 1):
            raise ValueError("exact response caching requires one serial native factor owner")
        count = len(factory.mesh.cells)
        if not configuration:
            raise ValueError("response storage requires a physical-input configuration")
        require_sources(sources, sources)
        identity = json.loads(
            json.dumps(
                dict(
                    schema=1, cells=count, batch_size=batch_size, configuration=dict(configuration)
                ),
                allow_nan=False,
            )
        )
        # Retain the default schema/identity so existing double archives remain
        # readable and resumable under their unchanged numerical source contract.
        if local_solver != "scipy" or local_refinement_precision != "double":
            identity["local_solver"] = local_solver
            identity["local_refinement_precision"] = local_refinement_precision
        if exact_response_cache:
            identity["exact_response_cache"] = True
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "responses.json"
        if path.exists():
            record = json.loads(path.read_text())
            require_sources(record["sources"], sources)
            require_equal(record["identity"], identity, label="response-store identity")
        else:
            record = dict(identity=identity, sources=dict(sources), batches=[], complete=False)
            _write_record(path, record)
        start = 0
        for row in record["batches"]:
            if (
                start >= count
                or row["start"] != start
                or row["stop"] != min(start + batch_size, count)
                or row["archive"] != f"responses-{start:07d}.npz"
            ):
                raise ValueError("response-store batch ordering differs")
            if file_digest(directory / row["archive"]) != row["sha256"]:
                raise ValueError("Helmholtz response batch digest differs")
            start = row["stop"]
        if record["complete"] and start != count:
            raise ValueError("complete response-store manifest has missing cells")
        if start == count and not record["complete"]:
            record["complete"] = True
            _write_record(path, record)
        owner = (
            ExactResponseCache(solver=local_solver, refinement_precision=local_refinement_precision)
            if exact_response_cache
            else nullcontext(None)
        )
        with owner as cache:
            cls._acquire(
                factory,
                directory,
                record,
                path,
                start,
                count,
                batch_size,
                backend,
                workers,
                local_solver,
                local_refinement_precision,
                cache,
            )
        return cls(directory, record)

    @staticmethod
    def _acquire(
        factory: Any,
        directory: Path,
        record: dict[str, Any],
        path: Path,
        start: int,
        count: int,
        batch_size: int,
        backend: str,
        workers: int,
        local_solver: str,
        local_refinement_precision: Literal["double", "extended"],
        cache: ExactResponseCache | None,
    ) -> None:
        """Commit bounded batches while one optional exact factor owner stays open."""
        for first in range(start, count, batch_size):
            started = time.perf_counter()
            stop = min(first + batch_size, count)
            before = (
                (cache.factorizations, cache.source_solves, cache.response_hits) if cache else None
            )
            local = acquire_local_responses(
                factory,
                range(first, stop),
                backend=backend,
                workers=workers,
                local_solver=local_solver,
                local_refinement_precision=local_refinement_precision,
                cache=cache,
            )
            archive = directory / f"responses-{first:07d}.npz"
            _write_batch(archive, local)
            record["batches"].append(
                dict(
                    start=first,
                    stop=stop,
                    archive=archive.name,
                    sha256=file_digest(archive),
                    storage_bytes=sum(item.storage_bytes for item in local),
                    acquisition_seconds=time.perf_counter() - started,
                )
            )
            if cache is not None:
                assert before is not None
                after = (cache.factorizations, cache.source_solves, cache.response_hits)
                record["batches"][-1]["exact_response_cache"] = dict(
                    zip(
                        ("factorizations", "source_solves", "exact_source_hits"),
                        (value - prior for value, prior in zip(after, before, strict=True)),
                        strict=True,
                    )
                )
            record["complete"] = stop == count
            _write_record(path, record)
            del local
