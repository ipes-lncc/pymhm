"""Bounded acquisition of immutable scientific resources with literal byte checks."""

from __future__ import annotations

import hashlib
import math
import os
import tempfile
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit
from urllib.request import urlopen

from pymhm.core.validation import positive_int
from pymhm.io.provenance import file_digest


def download_resource(
    url: str,
    destination: str | Path,
    *,
    sha256: str,
    maximum_bytes: int = 512 * 1024**2,
    timeout: float = 60.0,
) -> Path:
    """Acquire immutable bytes with a size limit and an expected SHA256 digest.

    ``maximum_bytes`` limits the complete payload in bytes, independently of
    HTTP headers. ``timeout`` is the positive finite network timeout in seconds.
    An existing file is reused only if its checksum matches; an unexpected
    existing file is preserved and raises ``ValueError``. Failed, oversized or
    corrupt downloads cannot replace the destination. This function performs
    a network request only when explicitly called for a missing destination.
    PYMHM_RESOURCE_ORIGIN can select an HTTP mirror while retaining the source
    path and query; checksum and size verification remain unchanged.
    """
    limit = positive_int(maximum_bytes, "maximum_bytes")
    if (
        not isinstance(sha256, str)
        or len(sha256) != 64
        or any(character not in "0123456789abcdef" for character in sha256)
    ):
        raise ValueError("sha256 must be a lowercase 64-character hexadecimal digest")
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be positive and finite")
    target = Path(destination).expanduser()
    if target.exists():
        if file_digest(target) != sha256:
            raise ValueError(f"SHA256 mismatch for existing resource: {target}")
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    origin = os.environ.get("PYMHM_RESOURCE_ORIGIN")
    if origin:
        mirror, original = urlsplit(origin), urlsplit(url)
        url = urlunsplit((mirror.scheme, mirror.netloc, original.path, original.query, ""))
    with tempfile.TemporaryDirectory(prefix=".pymhm-resource-", dir=target.parent) as temporary:
        pending = Path(temporary) / target.name
        digest = hashlib.sha256()
        total = 0
        with urlopen(url, timeout=timeout) as response, pending.open("wb") as stream:
            while chunk := response.read(min(1024**2, limit + 1)):
                total += len(chunk)
                if total > limit:
                    raise ValueError(f"resource exceeds the {limit}-byte acquisition limit")
                digest.update(chunk)
                stream.write(chunk)
        if digest.hexdigest() != sha256:
            raise ValueError(f"SHA256 mismatch for downloaded resource: {url}")
        if target.exists():
            if file_digest(target) != sha256:
                raise ValueError(f"SHA256 mismatch for concurrently created resource: {target}")
        else:
            pending.replace(target)
    return target
