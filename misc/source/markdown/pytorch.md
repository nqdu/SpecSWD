# PyTorch backend

`specd.torch` computes dispersion, eigenfunctions, and model gradients using
PyTorch tensors and automatic differentiation. It supports Love, VTI
Rayleigh/Scholte, and fully anisotropic waves, with optional constant-Q
attenuation. It runs independently of the C++/Fortran extensions.

## Installation

From the repository root, use Python 3.10 or newer and install the optional
dependency:

```bash
python -m pip install -e '.[torch]'
```

`python -m pip install .` installs the base package without requiring PyTorch.
For a published distribution, use `python -m pip install 'specd[torch]'`.
The provided Conda environment is another option:

```bash
conda env create -f specd/torch/environment.yml
conda activate specswd-torch
python -m pip install --no-deps -e .
OMP_NUM_THREADS=1 python -m specd.torch.example
```

To use both backends, also build and install the native extensions as described
in the [build guide](readme.md). Native imports remain `from specd import
SpecWorkSpace`; tensor imports use `from specd.torch import SpecWorkSpace`.

## Forward solve and automatic differentiation

Inputs use km, km/s, g/cm³, and Hz. Depths start at zero; repeated depths mark
discontinuities. The last row describes the constant half-space and repeats
the preceding depth.

```python
import torch
from specd.torch import SpecWorkSpace

vsh = torch.tensor([3.3, 3.3, 5.5], dtype=torch.float64, requires_grad=True)
ws = SpecWorkSpace(device="cpu").initialize(
    "love", [0., 35., 35.], [2.8, 2.8, 3.2],
    vsh=vsh, vsv=[3., 3., 5.],
)
phase = ws.compute_egn(0.1)
group = ws.group_velocity()
loss = (phase[0].real - 3.5).square() + (group[0].real - 3.3).square()
loss.backward()
print(vsh.grad)
```

Tensor inputs preserve their autograd graphs, including tensors derived from
trainable parameters. Lists and NumPy arrays become internal trainable tensors
available through `ws.parameters`. Real inputs use `float64`, complex results
use `complex128`. CPU is the default; CUDA can be selected through `device`.
MPS does not support the required double-precision eigensolves.

After updating parameters or finishing a backward pass, call `compute_egn`
again to construct the graph for the next optimization step. Keep gradients
enabled for group velocity and kernel evaluation. For phase-only inference:

```python
with torch.no_grad():
    phase_only = ws.compute_egn(0.1, only_phase=True)
```

`only_phase=True` omits eigenvectors; recompute with `only_phase=False` before
requesting group velocity or eigenfunctions.

## Model parameters

`initialize(wavetype, z, rho, **parameters)` accepts:

| `wavetype` | Required elastic parameters | Optional attenuation parameters |
| --- | --- | --- |
| `"love"` | `vsh`, `vsv` | `Qn`, `Ql` |
| `"rayl"` | `vph`, `vpv`, `vsv`, `eta` | `Qa`, `Qc`, `Ql` |
| `"aniso"` | `c21` | `Qani` |

Arrays have shape `(nz,)`, except `c21` with shape `(21,nz)` and `Qani` with
shape `(2,nz)`. Stiffness uses the upper triangle of the symmetric Voigt matrix
in order `c11,c12,...,c16,c22,...,c66`, with Voigt axes `xx,yy,zz,yz,xz,xy`.
`Qani` contains Qκ then Qμ. Supply all quality-factor arrays for the wave type,
or omit all for an elastic model. Q values must be positive and finite.
`reference_frequency=1.0` specifies the storage-modulus reference in Hz.

For acoustic Rayleigh rows, set `vsv=0`, `vph=vpv`, and `Qa=Qc` when
attenuating. Anisotropic acoustic rows require an isotropic bulk block with
zero shear and equal Qκ/Qμ. Solid/fluid transitions require repeated depths.

`specd.torch.isotropic_c21(rho, vp, vs)` constructs the stiffness tensor while
preserving gradients. The tensor backend implements anisotropic attenuation
model 1; it does not accept the native `qfunc_id` argument.

## Outputs and kernels

`compute_egn(freq, ang_in_deg=0., only_phase=False)` returns complex phase
velocities of shape `(nmode,)`, sorted by increasing real phase speed.
`group_velocity()` returns `(nmode,)` for Love/Rayleigh and `(2,nmode)` for
anisotropy, whose rows are Cartesian x and y components. The azimuth is in
degrees.

```python
ws.compute_egn(0.1)
phase_grad = ws.kernels(0, observable="phase", component="real")
group_grad = ws.kernels(0, observable="group", component="real")
print(phase_grad["vsh"])  # original input rows, derivative with respect to vsh
displacement = ws.get_egnfunc(0)  # (ncomponent, nSEMpoints)
depth = ws.mesh.depth
```

`kernels` returns a dictionary matching the parameter names and shapes.
`component` selects `"real"`, `"imag"`, or `"qinv"`; the last differentiates
the wave observable `2*Im(value)/Re(value)`. Q parameter entries differentiate
with respect to **Q itself**. Set `create_graph=True` to differentiate kernels
further. Anisotropic group kernels differentiate radial group velocity.

The compatibility methods `get_phase_kl(mode)` and `get_group_kl(mode)` stack
rows in `get_kernel_names()` order and convert material-Q derivatives to
derivatives with respect to **inverse Q**. Elastic models return a tensor;
attenuating models return `(velocity_kernel, wave_qinv_kernel)`. Tensor kernels
combine acoustic and solid contributions on the original model rows; there
are no separate `vp_ac` or `rho_ac` rows.

Displacement components are radial/vertical for Rayleigh and
radial/transverse/vertical for anisotropy. Use
`get_egnfunc(mode, return_displ=False, return_left=True)` for a raw left
eigenvector, or omit `return_left` for the raw right eigenvector.

## Frequency and azimuth sweeps

Tensor tables retain autograd graphs in variable-length rows:

```python
from specd.torch import DispersionTable

table = ws.compute_dispersion(
    [0.05, 0.1, 0.2], azimuths_deg=[0., 30.], include_group=True,
)
row = 1 * table.azimuth_deg.numel() + 0  # frequency index 1, azimuth index 0
mode_indices, phase_row = table.phase_row(row)
group_modes, group_row = table.group_row(row)
table.to_hdf5("torch-dispersion.h5")
restored = DispersionTable.from_hdf5("torch-dispersion.h5")
```

Unlike the native table, `phase_row` and `group_row` take a single flattened
row index, in frequency-major, azimuth-minor order. A group row has shape
`(nmode,ncomponent)`. Export explicitly detaches tensors and writes the shared
[`specswd-dispersion` v1 HDF5 schema](io.md), readable by the native
`specd.DispersionTable`. Loaded tensors have no gradient history. The tensor
backend exports dispersion tables rather than complete native `SolverOutput`
files.

## Numerical scope

Defaults are `ngll=7`, `ngrl=30`, and `halfspace_wavelengths=50`, matching the
native discretization. They are constructor arguments in the tensor backend.
Increasing the half-space scale requires checking GRL order: a larger scale
improves slowly decaying tails but spreads nodes farther apart. Phase and,
especially, group kernels near cutoff require a convergence check in both
scale and order.

The general quadratic problem is linearized and reduced with structured
linear solves before `torch.linalg.eig`. The tensor backend does not perform
generalized Schur/QZ decomposition. Group velocity uses operator frequency
and azimuth derivatives from `torch.func.jvp`; model kernels use reverse-mode
autodiff. Eigenvector derivatives require separated eigenvalues.

Mesh geometry, interpolation positions, and element counts are fixed during
differentiation; depth gradients are unsupported. Mode sorting and selection
are discrete. A gradient follows a selected mode locally, rather than
differentiating changes in mode count or ordering.

## Examples and validation

From the repository root:

```bash
OMP_NUM_THREADS=1 python -m pytest specd/torch/tests -q
OMP_NUM_THREADS=1 python example/paper/reproduce_paper_pytorch.py
OMP_NUM_THREADS=1 python -m specd.torch.benchmark \
    --layers 1 4 --frequencies .05 .2 --repeats 5 --warmup 1 \
    --workloads phase_kernel --output /tmp/specswd-phase-kernel
```

Native comparisons and benchmarks require a current compiled extension;
the remaining tensor tests run without it. The primary benchmark workload
`phase_kernel` includes a fresh forward solve and fundamental-mode phase
kernels. Paper figures are written to `example/paper/pytorch`.

The [PyTorch API reference](../python/torch.rst) lists public signatures.
See [C++ and PyTorch efficiency](benchmarks.md) for the four comparisons and
their timing figure.
