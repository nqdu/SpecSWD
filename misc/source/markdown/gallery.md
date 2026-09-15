# Gallery

## Two-layer attenuating Love-wave model

| Layer | Thickness (km) | $\rho$ (g/cm³) | $V_{SV}$ (km/s) | $V_{SH}$ (km/s) | $Q_L$ | $Q_N$ |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 35 | 2.8 | 3.0 | 3.3 | 200 | 220 |
| Half-space | $\infty$ | 3.2 | 5.0 | 5.5 | 300 | 330 |

![Love-wave dispersion benchmark](../../../example/love/eigenvalues_att.jpg)

Phase velocity, group velocity, and propagation-Q benchmark against the
analytical solution.

![Love-wave phase kernels](../../../example/love/phase_deriv_veloc_att.jpg)

Phase-velocity sensitivity kernels compared with the analytical solution.

![Love-wave group kernels](../../../example/love/group_deriv_veloc_att.jpg)

Group-velocity sensitivity kernels compared with the analytical solution.

## Coupled fluid/solid Scholte-wave model

| Layer | Thickness (km) | $\rho$ (g/cm³) | $V_P$ (km/s) | $V_S$ (km/s) |
|---:|---:|---:|---:|---:|
| Fluid | 5 | 1.00 | 1.50 | 0.00 |
| Solid 1 | 45 | 2.57 | 5.22 | 3.10 |
| Solid 2 | 50 | 2.95 | 6.94 | 4.00 |
| Half-space | $\infty$ | 3.57 | 8.75 | 5.00 |

![Scholte-wave dispersion](../../../example/scholte/eigenvalues.jpg)

Phase and group velocity benchmark against CPS330.

![Scholte-wave eigenfunctions](../../../example/scholte/eigenvecs.jpg)

Normalized coupled solid displacement and acoustic-potential eigenfunctions.

![Scholte-wave phase kernels](../../../example/scholte/phase_deriv.jpg)

Phase kernels compared with centred finite differences.

![Scholte-wave group kernels](../../../example/scholte/group_deriv.jpg)

Group kernels compared with centred finite differences.

The test suite also covers attenuating fully anisotropic group velocity,
radial/x/y group-kernel consistency, and coupled acoustic bulk-modulus group
kernels. These checks use the same analytic frequency and model derivatives as
the production solver.
