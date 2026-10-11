# Methods

PyMHM provides local/global constructions for MHM and related multiscale
formulations. The tables identify the available formulations and their primary
references. Consult the cited publications for mathematical definitions,
approximation-space requirements and assumptions.

Use the [Tutorials](tutorials/index.md) for the implementation workflow and
the [API](api.md) for the supported operators, spaces and input contracts.
The [Gallery](gallery/index.md) presents applications with their chosen data
and discretizations.

## MHM formulations {#original-mhm}

| Formulation | Local problem or model | Primary reference |
| --- | --- | --- |
| Primal MHM | Scalar diffusion; primal local problems | [Harder, Paredes and Valentin (2013)](https://doi.org/10.1016/j.jcp.2013.03.019) |
| Mixed H(div) MHM | Local flux–pressure mixed problems | [Durán et al. (2019)](https://doi.org/10.1016/j.cma.2019.05.013) |
| Petrov–Galerkin MHM | Residual-enriched scalar formulation | [Fernando et al. (2023)](https://doi.org/10.1007/s40314-023-02304-y) |
| Scalar RAD MHM | Reaction–advection–diffusion | [Harder et al. (2015)](https://doi.org/10.1137/130938499); [Araya et al. (2024)](https://doi.org/10.1016/j.cma.2024.117089) |
| Stokes–Brinkman MHM | Velocity–pressure local problems | [Araya et al. (2017)](https://doi.org/10.1016/j.cma.2017.05.027) |
| Oseen MHM | Flow with prescribed convection | [Araya et al. (2021)](https://doi.org/10.1007/s10444-020-09833-8) |
| Primal elasticity MHM | Displacement local problems | [Harder, Madureira and Valentin (2016)](https://doi.org/10.1051/m2an/2015046) |
| GaLS elasticity MHM | Stabilized displacement–pressure local problems | [Gomes, Pereira and Valentin (2024, preprint v1)](https://arxiv.org/abs/2403.16890v1) |
| Mixed elasticity MHM | Stress, displacement and rotation local problems | [Devloo et al. (2021)](https://doi.org/10.1051/m2an/2021013) |
| Elastodynamic MHM | Time-dependent elasticity | [Gomes et al. (2017)](https://doi.org/10.20906/CPS/CILAMCE2017-0399) |
| Helmholtz MHM | Frequency-domain wave problems | [Chaumont-Frelet and Valentin (2020)](https://doi.org/10.1137/19M1255616) |
| Maxwell MHM | Time-domain electromagnetic problems | [Lanteri et al. (2018)](https://doi.org/10.1137/16M110037X) |

## Related formulations {#related-formulations}

These formulations have their own local and global equations.

| Formulation | Primary reference |
| --- | --- |
| Robin Multiscale Hybrid method (MH) | [Barrenechea, Gomes and Paredes (2024)](https://doi.org/10.1137/22M1542556) |
| Three-field Multiscale Method (MH²M) | [de Barros, Madureira and Valentin (2026, preprint v3)](https://arxiv.org/abs/2404.16978v3) |
| Multiscale Hybrid High-Order method (MsHHO) | [Cicuttin, Ern and Lemaire (2019)](https://doi.org/10.1515/cmam-2018-0013); [Chaumont-Frelet et al. (2022)](https://doi.org/10.1051/m2an/2021082) |

## Complementary strategies {#complementary-strategies}

| Strategy | Primary reference |
| --- | --- |
| MHM-USFEM: stabilized reaction–diffusion local problems | [Martins, Pacazuca and Valentin (CILAMCE 2025)](https://doi.org/10.55592/cilamce2025.v5i.14270) |
| Independent skeletal refinement | [Paredes, Valentin and Versieux (2024)](https://doi.org/10.1016/j.cam.2023.115415) |
| Unfitted macro meshes with material-fitted skeletal partitions | [Chaumont-Frelet, Paredes and Valentin (2026)](https://doi.org/10.1016/j.camwa.2026.01.016) |
| Flux reconstruction and associated estimators | [Barrenechea et al. (2026)](https://doi.org/10.1137/24M1673073) |
| Residual estimators and adaptive refinement | [Araya et al. (2013)](https://doi.org/10.1137/120888223); [Araya, Rebolledo and Valentin (2021)](https://doi.org/10.1093/imanum/drz053) |

The package also provides [transient scalar integration](api/transport.md)
and [nested local/global providers](guides/providers.md).
Their supported software interfaces are documented separately from the
reference formulations above.

## Citing

If you are using PyMHM or any of the supported methods, please **cite the publications for the methods and strategies used, together with the PyMHM version.** The [Bibliography](literature.md) provides the complete
bibliographic records, including versioned preprints.
