# Tutorial

## Layered model convention

SpecSWD represents piecewise-linear one-dimensional models. Depth increases
downward. Repeat a depth to mark a discontinuity: the first occurrence is the
bottom value of the region above and the second is the top value of the region
below. The final finite-depth point must be repeated before the half-space
sample.

For example, the rows at 35 km below describe a discontinuity:

```text
0.0   ... upper material ...
35.0  ... upper material ...
35.0  ... lower material ...
```

Python inputs normally use kilometres, kilometres per second, and grams per
cubic centimetre. SpecSWD chooses nondimensional scales automatically unless
`scale_z`, `scale_v`, and `scale_rho` are supplied to
`SpecWorkSpace.initialize`.

## Love-wave VTI model

A Love-wave model uses density, $V_{SH}$, and $V_{SV}$. Attenuating models also
provide $Q_N$ and $Q_L$:

```python
ws.initialize(
    "love", z, rho,
    vsh=vsh,
    vsv=vsv,
    Qn=qn,
    Ql=ql,
    reference_frequency=1.0,
)
```

Love waves cannot propagate in a fluid layer.

## Rayleigh and Scholte VTI model

A VTI Rayleigh model uses $V_{PH}$, $V_{PV}$, $V_{SV}$, density, and $\eta$.
Attenuation uses $Q_A$, $Q_C$, and $Q_L$:

```python
ws.initialize(
    "rayl", z, rho,
    vph=vph,
    vpv=vpv,
    vsv=vsv,
    eta=eta,
    Qa=qa,
    Qc=qc,
    Ql=ql,
    reference_frequency=1.0,
)
```

Set `vsv == 0` for acoustic points. At those points `vph` and `vpv` must be
equal. In an attenuating acoustic region, `Qa` and `Qc` must also be equal. A
solid/fluid boundary must coincide with a repeated-depth discontinuity.

## Fully anisotropic model

The Python compatibility API accepts stiffness in shape `(21, nz)`. Components
are the upper triangle of the symmetric $6\times6$ Voigt matrix in this order:

```text
c11 c12 c13 c14 c15 c16
    c22 c23 c24 c25 c26
        c33 c34 c35 c36
            c44 c45 c46
                c55 c56
                    c66
```

The following function creates the required array from matrices stored by
depth:

```python
def pack_c21(c66):
    # c66.shape == (nz, 6, 6)
    upper = np.triu_indices(6)
    return np.ascontiguousarray(c66[:, upper[0], upper[1]].T)
```

The supported anisotropic attenuation mapping uses two arrays, $Q_\kappa$ and
$Q_\mu$, with `Qani.shape == (2, nz)` and `qfunc_id=1`:

```python
ws.initialize(
    "aniso", z, rho,
    c21=c21,
    Qani=np.vstack([q_kappa, q_mu]),
    qfunc_id=1,
    reference_frequency=1.0,
)
```

Internally, the interpolated SEM arrays are point-major:
`xC21[point*21 + component]` and
`xQani[point*nQani + q_component]`.

For an isotropic acoustic point, only
`c11`, `c12`, `c13`, `c22`, `c23`, and `c33` are nonzero and all equal the
bulk modulus. All Q values at an acoustic point must be equal.

## Constant-Q attenuation

SpecSWD uses a direct causal constant-Q modulus for positive frequency:

$$
M(\omega)=M_0\left(1+\frac{i}{Q}\right)
\left(\frac{\omega}{\omega_0}\right)^\alpha,
\qquad
\alpha=\frac{2}{\pi}\tan^{-1}\left(\frac{1}{Q}\right).
$$

`M0` is the real storage modulus supplied by the user at the reference
frequency: $M_0=\operatorname{Re}M(\omega_0)$. Consequently,
$M(\omega_0)=M_0(1+i/Q)$; `M0` is not the full complex modulus at the reference
frequency. The default reference frequency is 1 Hz and can be changed through
`reference_frequency`.

Velocity inputs are converted to their corresponding reference storage
moduli. Acoustic input uses the bulk modulus even though inverse bulk modulus
appears on the left side of the acoustic equation. Frequency derivatives and
mixed frequency/model derivatives are evaluated analytically for group
velocity and group kernels. SLS coefficient fitting is not part of the runtime
attenuation model.

You can inspect the same constitutive response without running the solver:

```python
from specd import constant_q_response

factor = constant_q_response(
    freq=np.array([0.1, 1.0, 10.0]),
    quality_factor=200.0,
    reference_frequency=1.0,
)
```

## Phase and group velocity

`compute_egn` returns all selected modes at one frequency. Set
`only_phase=False` to retain the generalized Schur data required by group
velocity, eigenfunctions, and kernels:

```python
phase = ws.compute_egn(freq=0.1, ang_in_deg=30.0, only_phase=False)
group = ws.group_velocity()
```

For Love and Rayleigh waves, `group.shape == (nmodes,)`. For fully anisotropic
waves, `group.shape == (2, nmodes)`, with x and y components. Radial and
transverse values are

```python
phi = np.deg2rad(30.0)
radial = np.cos(phi) * group[0] + np.sin(phi) * group[1]
transverse = -np.sin(phi) * group[0] + np.cos(phi) * group[1]
```

## Sensitivity kernels

After a QZ solve, request phase or group kernels for a zero-based mode:

```python
phase_velocity_kernel, phase_qinv_kernel = ws.get_phase_kl(0)
group_velocity_kernel, group_qinv_kernel = ws.get_group_kl(0)
names = ws.get_kernel_names()
```

Attenuating calls return two arrays. The first differentiates the propagation
velocity; the second differentiates the propagation inverse quality factor.
Elastic calls return only the velocity array.

For fully anisotropic media, the current scalar compatibility entry point
`get_group_kl` returns the radial group-velocity kernel. The numerical core also
supports x- and y-component group kernels. The new result API will expose
those components explicitly.

## Variable-mode frequency grids

The number of modes may change with frequency. `compute_dispersion` therefore
returns CSR containers instead of a padded rectangular array:

```python
table = ws.compute_dispersion(
    frequencies_hz=np.array([0.05, 0.1, 0.2]),
    azimuths_deg=np.array([0.0, 30.0]),
    include_group=True,
)

modes, phase = table.phase_row(frequency_index=1, azimuth_index=0)
group_modes, group = table.group_row(1, 0)
table.to_hdf5("dispersion.h5")
```

See [CSR and HDF5 output](io.md) for the array layout and cross-language file
schema.
