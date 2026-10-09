# Resources and writable workspaces

The installed library contains its Python modules and typing information.
Application sources, notebooks and scientific inputs are separate downloads.
These utilities accept explicit directories, URLs and catalogues; they contain
no built-in case inventory.

`download_resource` acquires a file with a declared SHA256 and byte limit. It
reuses a matching local copy and preserves that copy when acquisition fails.
`workspace_from_archive` applies the same identity check to a ZIP, validates
its paths and expanded size, then extracts regular files into a writable
directory. Extraction preserves existing local files and does not execute code.
Importing companion sources is an explicit step in the consuming notebook.

A workspace may register an external `.pymhm-resources.json`, or callers can
pass a mapping to `register_resources`. For example:

```python
from pymhm.io.workspace import register_resources, resource_file

register_resources(
    "run",
    {
        "resources": {
            "data/coefficients.csv": {
                "url": coefficient_url,
                "sha256": coefficient_sha256,
                "size_bytes": coefficient_size_bytes,
            },
        },
    },
)
coefficients = resource_file("data/coefficients.csv", directory="run")
```

Registration describes inputs without downloading them. `resource_file` checks
immutable inputs against the catalogue and acquires only the requested file.
`materialize_resources` copies selected inputs and declared attribution files
into the workspace, preserving user edits. `local_resource` supports existing
mutable local files. Unavailable original inputs retain explicit reasons and
are not replaced with different data.

`source_identity` hashes the actual source files used by an application. A
logical `src/pymhm/...` path identifies the installed library's corresponding
module; application sources resolve in the declared workspace. Missing Git or
lockfile metadata remain absent from provenance records.

For the notebook workflow and download catalogue, see the
[notebook guide](../tutorials/notebooks.md#execute-downloaded-notebooks) and
[data downloads](../data.md).

::: pymhm.io.resources

::: pymhm.io.workspace

::: pymhm.io.provenance

## Source modules

::: pymhm.io.reservoir
    options:
      show_source: false
