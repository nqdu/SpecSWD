# Differentiable SpecSWD

`specd.torch` implements the spectral surface-wave solver directly in PyTorch.
Install the optional dependency with `pip install ".[torch]"` (or
`pip install "specd[torch]"` for a published distribution). The implementation
lives in `specd/torch`.
It does not import `specd.lib`, call C++/Fortran, or use the native adjoint
kernels during differentiation. The forward path includes model interpolation,
GLL finite elements, a GRL infinite half-space element, matrix assembly,
eigensolving, mode filtering, and eigenfunctions. It supports:

- Love waves with VTI `vsh` and `vsv`;
- Rayleigh/Scholte waves with VTI solids and acoustic layers;
- general anisotropic `c21` tensors and propagation azimuths;
- the native constant-Q attenuation law and reference-frequency convention;
- phase velocity, complex group velocity, and their model gradients;
- frequency/azimuth sweeps with variable mode counts and interoperable HDF5.

## Environment

From the repository root:

```bash
conda env create -f specd/torch/environment.yml
conda activate specswd-torch
python -m specd.torch.example
OMP_NUM_THREADS=1 python -m pytest specd/torch/tests -q
```

If the environment already exists, activate it directly. To update it:

```bash
conda env update -n specswd-torch -f specd/torch/environment.yml
```

The package imports from the repository root. For installation elsewhere,
`python -m pip install -e '.[torch]'` installs both Python packages; the torch
solver itself needs no compiled extensions. The tests comparing against the
native solver skip when its extensions are unavailable.

If an installed extension predates the current Python API, those comparisons
also skip. To test against a separate current build, set
`SPECSWD_REFERENCE_EXTENSION` to the built `libswd` extension file when running
pytest; this avoids replacing an installed extension.

On macOS the environment selects the pthreads OpenBLAS build so NumPy and
PyTorch can coexist without loading two OpenMP runtimes.

## Forward solve and gradients

Use km, km/s, g/cm³, and Hz, as in the existing Python interface. Repeated
depths define discontinuities; the last row is the constant half-space and
must repeat the preceding depth. Depths are fixed geometry.

```python
import torch
from specd.torch import SpecWorkSpace

vsh = torch.tensor([3.3, 3.3, 5.5], dtype=torch.float64, requires_grad=True)
vsv = torch.tensor([3.0, 3.0, 5.0], dtype=torch.float64, requires_grad=True)
rho = torch.tensor([2.8, 2.8, 3.2], dtype=torch.float64, requires_grad=True)

ws = SpecWorkSpace().initialize(
    "love", [0., 35., 35.], rho, vsh=vsh, vsv=vsv,
)
c = ws.compute_egn(0.1)  # complex128, sorted by increasing real phase speed
u = ws.group_velocity()
loss = (c[0].real - 3.5).square() + (u[0].real - 3.3).square()
loss.backward()
print(vsh.grad, vsv.grad, rho.grad)
```

Tensor inputs retain their autograd graphs, including non-leaf tensors derived
from trainable parameters. Lists/NumPy arrays become internal trainable tensors
available through `ws.parameters`. Inputs are converted to float64 on the input
device (or `SpecWorkSpace(device="cpu")` / `device="cuda"`). Complex operators
use complex128. CPU is the default; float64/complex eigensolves exclude MPS.
Call `compute_egn` again after changing parameters or performing backward before
starting a new optimization step. As with ordinary PyTorch computations, reuse
of an existing graph after backward requires `retain_graph=True`.

Group velocity uses `Uradial = 2*pi / (dk/df)`. For the discrete wave operator
`P(k,f)`, eigenvalue perturbation gives
`dk/df = -(leftᴴ @ P_f @ right) / (leftᴴ @ P_k @ right)`.
Forward-mode autodiff (`torch.func.jvp`) computes operator partials with geometry
held fixed, including frequency-dependent attenuation and fluid coupling.
Tensor contractions evaluate every retained mode together. An azimuth JVP
supplies the transverse component for anisotropy; `group_velocity()` returns
Cartesian `x,y` rows with shape `(2,nmode)`. The eigenvectors and operator
partials retain their graphs, so reverse-mode autodiff computes group kernels
without a separate eigensolver backward for each mode. Gradients remain
enabled when computing group velocity or kernels.

## Tensor implementation

The public workspace API remains the same. Internally, each mesh stores
contiguous node, interpolation, connectivity and scatter tensors, with offsets
for variable element lengths. There are no per-element objects or growing
lists in mesh construction, assembly, group evaluation or displacement
reconstruction. GLL/GRL quadrature and constitutive index maps are cached by
order/device; adaptive geometry is rebuilt for every solve.

Assembly processes at most four batches: finite/half-space elements in
solid/fluid material. `assembly._solid_operators` contains the local tensor
math, separate from discrete mesh construction. Constitutive laws use broadcast
masks, stiffnesses use batched matrix products, and indexed additions assemble
the global operators. Interpolated parameters and scatter plans are reused
within a solve and its operator JVPs. Tensor batches use the input device;
adaptive element counts and mode selection still require discrete decisions.

The horizontal operator is diagonal for Love, triangular for Rayleigh, and
node-local 3×3 blocks plus acoustic diagonal entries for anisotropy. These
structures replace general dense solves. Dual eigenvectors are obtained only
for selected modes by solving against selected unit columns; no full
eigenvector-basis inverse is constructed. The dense eigendecomposition still
computes the full spectrum. Ragged frequency/azimuth outputs retain their
existing list interface, with row slots allocated before the sweep.

## Models

`initialize` takes `wavetype`, `z`, `rho` and these keyword arguments:

| Wave | Elastic parameters | Optional quality factors |
| --- | --- | --- |
| `love` | `vsh`, `vsv` | `Qn`, `Ql` |
| `rayl` | `vph`, `vpv`, `vsv`, `eta` | `Qa`, `Qc`, `Ql` |
| `aniso` | `c21`, shape `(21,nz)` | `Qani`, shape `(2,nz)`, Qκ then Qμ |

Other arrays have shape `(nz,)`. The tensor ordering is the upper triangle of
the 6×6 Voigt stiffness matrix: `c11,c12,...,c16,c22,...,c66`, with Voigt order
`xx,yy,zz,yz,xz,xy`. `isotropic_c21(rho,vp,vs)` builds an isotropic tensor while
retaining gradients. Anisotropic attenuation implements native Q model 1.
Supply every quality-factor array listed for the chosen wave type, or none.
Model quality factors must be positive and finite; omit them for elastic models.
`reference_frequency=1.0` sets the storage-modulus reference in Hz.

Fluid Rayleigh rows have `vsv=0`, `vph=vpv`, and `Qa=Qc` when attenuating.
Fluid anisotropic rows contain an isotropic bulk block with zero shear and
equal Qκ/Qμ. Material transitions require a repeated depth.

## Kernels, eigenfunctions and IO

```python
ws.compute_egn(0.1, ang_in_deg=30.)
named = ws.kernels(0, observable="phase", component="real")
group_named = ws.kernels(0, observable="group", component="real")
qinv_named = ws.kernels(0, observable="phase", component="qinv")
displacement = ws.get_egnfunc(0)  # (ncomponent, nSEMpoints)
depth = ws.mesh.depth

table = ws.compute_dispersion([0.05, 0.1, 0.2], include_group=True)
mode_indices, phase_row = table.phase_row(0)
table.to_hdf5("torch-dispersion.h5")
```

`kernels()` returns a dictionary with gradients in original input units;
Q entries are derivatives with respect to **Q itself**. `component="qinv"`
differentiates the wave observable `2*Im(value)/Re(value)`. Set
`create_graph=True` if further differentiation of the kernels is needed.
Anisotropic group kernels refer to radial group velocity.

`get_phase_kl(mode)` and `get_group_kl(mode)` stack kernels in
`get_kernel_names()` order and convert Q parameter derivatives to derivatives
with respect to inverse material Q, matching the native convention. For
attenuating models they return `(velocity_kernel, wave_qinv_kernel)`.
Fluid and solid contributions are combined at the original input rows;
there are no separate `vp_ac`/`rho_ac` rows in this API.

Eigenfunctions are right displacement amplitudes; anisotropic components are
radial/transverse/vertical and Rayleigh uses the native polarization convention.
Raw right/left generalized eigenvectors are available with
`get_egnfunc(mode, return_displ=False, return_left=False/True)`.
Only phase computation can be requested with `compute_egn(..., only_phase=True)`.

Sweeps keep tensors and their graphs in ragged rows. HDF5 export explicitly
detaches tensors and writes the existing `specswd-dispersion` v1 CSR schema,
readable by `specd.DispersionTable.from_hdf5`. The torch table also has
`DispersionTable.from_hdf5`, which loads tensors without a gradient history.
This module exports dispersion tables; it does not reproduce the command-line
programs, the separate CPS330 reference solver, or full `SolverOutput` files.

## Efficiency comparison with C++

Use **forward + phase kernels** (`--workloads phase_kernel`) as the primary
benchmark. Each measurement includes a fresh forward solve followed by all
fundamental-mode phase-kernel parameter rows. For attenuating models it also
includes wave inverse-Q kernels. Do not add the separate `forward` timing.

The reproducible CPU benchmark separately compares phase-only solves,
forward solves with eigenvectors, phase kernels, group velocities, and group
kernels. It verifies numerical
agreement before timing, sets only `OMP_NUM_THREADS=1` by default, warms up both
backends, alternates their measurement order, and reports median times.
Every timed solve includes mesh creation and assembly; model initialization
and IO are excluded. Attenuating kernel workloads include both real velocity
and wave inverse-Q derivatives. Forward inference disables the torch graph;
gradient workloads construct a fresh graph on every solve.
Other thread pools use their defaults; the benchmark does not call PyTorch
thread-setting functions. The JSON records the thread environment and actual
PyTorch thread counts.

```bash
conda activate specswd-torch
for wave in love rayl scholte aniso; do
    OMP_NUM_THREADS=1 python -m specd.torch.benchmark \
        --waves "$wave" --layers 1 4 --frequencies .05 .2 \
        --workloads forward phase_kernel group group_kernel \
        --repeats 5 --warmup 1 --output "specd/torch/benchmarks/$wave"
done
OMP_NUM_THREADS=1 python misc/plot_benchmarks.py
```

Use `--native-extension /path/to/libswd.so` to select a separate current
native build; omit it when the installed `specd.lib.libswd` matches the
source. `phase_only` sets `only_phase=True`; `forward` includes eigenvectors.
Use `--help` to change cases, sizes, frequencies, threads and output.
Run comparisons in a fresh Python process. The requested extension is loaded
before importing `specd`, so an installed build cannot affect backend
selection through shared C++ template symbols.

Only four saved comparisons are retained:
[Love](benchmarks/love/efficiency.md),
[Rayleigh](benchmarks/rayl/efficiency.md),
[Scholte](benchmarks/scholte/efficiency.md), and
[anisotropic](benchmarks/aniso/efficiency.md).
Each contains a report, raw JSON samples, and a runtime-ratio figure for one
and four solid layers, elastic and attenuating models, and four workloads.
The [manual](../../misc/source/markdown/benchmarks.md) displays the primary
forward-plus-phase-kernel timing figure. GPU performance and peak memory are
outside this benchmark.

## Numerical scope

The defaults match the native 7-node GLL and 30-node GRL discretization and
50-wavelength half-space scale. These can be configured in the constructor.
The dense torch eigensolvers favor modest model sizes and experimentation
with inverse problems; the native implementation remains useful for large
sweeps. Mode selection and sorting are discrete. Gradients follow a selected
mode locally and do not differentiate a change in mode count or ordering.

Adaptive mesh geometry, interpolation positions and element counts are held
fixed during backward, as in the native model kernels. Depth gradients are
rejected explicitly. Finite differences should hold this geometry fixed too.
Frequency and azimuth derivatives describe the physical operator on this
fixed mesh rather than the numerical mesh adaptation.

Eigenvector-based derivatives require separated eigenvalues, and losses on
complex eigenvectors must be independent of their arbitrary phase; see the
[PyTorch eigensolver documentation](https://docs.pytorch.org/docs/stable/generated/torch.linalg.eig.html).
The general problems use `torch.linalg.solve` followed by `torch.linalg.eig`
instead of native generalized Schur/QZ. Very ill-conditioned or singular
stiffness matrices therefore need additional care.
