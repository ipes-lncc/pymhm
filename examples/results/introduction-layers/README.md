# Analytical reaction–diffusion and Brinkman layers

The numerical records accompany the introductory notebooks:

- [Scalar reaction–diffusion](../../../notebooks/introduction/mhm_usfem_rad.ipynb)
- [Incompressible Brinkman flow](../../../notebooks/introduction/stokes_brinkman_boundary_layer.ipynb)

Run the notebooks with the checked-in lockfile and a 3600-second cell timeout:

```bash
pixi run --locked -e introduction notebooks-run introduction/mhm_usfem_rad.ipynb --timeout 3600
pixi run --locked -e introduction notebooks-run introduction/stokes_brinkman_boundary_layer.ipynb --timeout 3600
```

The notebooks define the physical data, UFL forms,
local and global equations, conforming references, field norms and plots.

The scalar record separates the coarse resolution control from the family with
four face segments and eight local subdivisions per macro edge. USFEM reduces
nodal oscillations on the severe layer while retaining a larger scalar L2 error
than Galerkin. The severe layer's physical flux remains underresolved;
its relative error is approximately
58% on the finest displayed MHM mesh. The scalar and flux comparisons therefore
have separate scientific interpretations.

The Brinkman record distinguishes the equal-order P2/P2–P4/P4 family with one
local triangle, the Taylor–Hood comparison with admissible local interior
vertices, and the further refined local/skeletal control. Exact velocity and pressure are
available. Independently assembled conforming Taylor–Hood fields supply a
classical comparison whose own refinement error is recorded.

These applications use published analytical data and declared approximation
families. Historical macro connectivity, matching local triangulations and
numerical inverse constants are not completely specified in the papers;
individual historical figure values are not claimed as reproduced.

`provenance.json` binds the current records and figures to the executed notebook
sources, runtime source digests, native dependency versions and Pixi lockfile.
The numerical records declare generated state archives and executed basis
digests. Re-execution saves those matrices and checks field replay with one
and two BLAS threads. Generated state archives remain outside the release
payload. `SHA256SUMS` verifies this publication's literal files; a second
manifest accompanies the figures in `docs/figures/introduction-layers`.
Source distributions contain the notebooks and numerical records; execute the
notebooks to regenerate their rendered figures.

See the [case documentation](../../../docs/cases/introduction-layers.md) for
method conditions, measured rates, field plots and literature attribution.
