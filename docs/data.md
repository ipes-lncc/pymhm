# Data and notebook downloads

The Python distribution contains only the `pymhm` library and its typing and
release metadata. Notebook sources, example helpers, case configurations,
datasets, recorded fields and figures are separate repository and website files.
No case collection or resource registry is installed with the library.

The [download catalogue](https://ipes-lncc.github.io/pymhm/downloads/catalogue.json)
lists individual notebooks, their companion ZIPs, available records and figures
with byte sizes, SHA-256 checksums, source attribution and scope. A notebook's
first cell identifies its companion URL and checksum. The companion contains
its importable Python helpers, acquisition recipes, small configurations and
selected input catalogue. It contains no datasets, computed fields or images.
Acquiring and extracting it verifies its bytes without executing its code;
the subsequent notebook cells import the declared helpers.

```bash
python -m pip install "pymhm[notebooks]"
jupyter lab primal_galerkin.ipynb
```

The notebook uses a writable workspace for acquired inputs and generated outputs.
Selected data downloads are verified against their SHA-256 identities; valid
cached files can be reused. Install the native backend identified by the
notebook when required; see [installation](installation.md#native-ufl-assembly).
Opening a notebook or displaying a recorded figure does not recompute its fields.

## Primary datasets

| Dataset | Public source and identity | Acquisition scope |
| --- | --- | --- |
| SPE10 Model 2 | [OPM/opm-data, pinned revision](https://github.com/OPM/opm-data/tree/eaa2261683a97027e057c2bc49612ad1c86390b3/spe10model2) | Full 60×220×85 permeability/porosity, or the separately downloaded layers 1, 36, 85 |
| Marmousi II | [Madagascar primary SEG-Y files](https://ahay.org/data/marm2/) | Two verified files, about 310 MB total; the declared 2048×512 material crop retains its coordinates and units |
| HPC4e | [labmec/MHM, pinned Data_13_Set](https://github.com/labmec/MHM/tree/f978f29d657d28fe58bcea20fabee68953093482/Data_13_Set) | Three verified material arrays with the article's physical bounds and axis conventions |

The separately downloadable dataset identity records contain the source revisions, checksums,
units and array conventions. The corresponding public acquisition helpers reuse
valid caches and reject mismatched downloads. They keep primary dataset files
outside the Python installation.

The compact SPE10 downloads are
[layer 1](https://ipes-lncc.github.io/pymhm/downloads/685c6b7e51bfa56a4b2c6e3a802ea57b7f8a9590f2f99fd403312284f5546e23/layer-1.npz),
[layer 36](https://ipes-lncc.github.io/pymhm/downloads/ffe319ce6c126d338a8ad95e497a0121c87e3d7d3eadafb4021ebbe7d3998512/layer-36.npz)
and [layer 85](https://ipes-lncc.github.io/pymhm/downloads/eeccea0043307ed8d016f811062d05b8f1315d8c6d8ea48f08a2162cc700d533/layer-85.npz).
Their catalogue entries link the accompanying metadata and dataset notices.

SPE10 is attributed to Mike Christie and Martin Blunt. The two pinned property
include-file headers state that their data are public domain. OPM's repository
also supplies ODbL 1.0 database/deck and DbCL 1.0 content notices unless otherwise
specified. Those notices accompany the downloadable slices and are separate from
PyMHM's LGPL license. The porosity convention, including OPM's recorded replacement
of most zeros by 1e-7, is preserved. Downloading or redistributing other datasets
requires observing the terms at their stated primary source; the package license
does not grant additional third-party rights.

## Current studies and historical comparisons

Current computed fields use the notebook's public acquisition recipe. After
acquiring its companion, use its runner from the companion workspace. Inspect
its discretization, numerical scope and resource requirements before a full study:

```bash
cd /path/to/extracted-companion  # the notebook's ROOT directory
python -m scripts.run_notebooks /path/to/72_marmousi.ipynb --study --plan
```

The command performs no acquisition in planning mode. Complete studies retain
their stated computational cost; a small current control is labeled separately
from a matched article reproduction.

Original external field archives and article image crops are optional historical
inputs. The catalogue records unavailable payloads and links their primary
sources. Article crops without verified redistribution permission are not exported
as downloads. Supply the exact attributed payload when requesting historical
replay; a fresh PyMHM field cannot replace an original external comparison.
