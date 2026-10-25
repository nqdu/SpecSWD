# SpecSWD

SpecSWD computes surface-wave dispersion, eigenfunctions, and sensitivity
kernels for one-dimensional layered media. Its spectral-element formulation
supports discontinuities, anisotropy, attenuation, and coupled solid-fluid
models within the same numerical framework.

The code provides C++ executables and a NumPy-based Python interface. Results
whose number of modes changes with frequency are stored as compressed sparse
row (CSR) arrays and can be exchanged between C++ and Python through a shared,
versioned HDF5 format.

## Capabilities

- Love waves in vertically transversely isotropic (VTI) media.
- Rayleigh and Scholte waves in solid and coupled solid-fluid VTI media.
- Fully anisotropic surface waves at arbitrary propagation azimuths.
- Causal constant-Q attenuation referenced to a user-selected frequency.
- Complex phase velocity, group velocity, and wave quality factor.
- Left and right eigenvectors and physical displacement eigenfunctions.
- Adjoint Fréchet kernels for phase and group observables.
- Gauss-Legendre-Lobatto elements in finite layers and a
  Gauss-Radau-Legendre element for the half-space.
- HDF5 output implemented with the HDF5 C library and exposed as contiguous
  NumPy arrays in Python.

## Requirements

The numerical core requires:

- CMake 3.25 or newer;
- a C++17 compiler and a Fortran compiler;
- [Eigen 5](https://eigen.tuxfamily.org/);
- Apple Accelerate on macOS (selected automatically), or LAPACK and LAPACKE
  on other platforms; and
- the HDF5 C library and development headers.

Building the Python extension also requires Python, NumPy, and pybind11. The
Python package uses SciPy, h5py, and Matplotlib. On Ubuntu, install the system
dependencies with:

```bash
sudo apt-get update
sudo apt-get install -y \
    build-essential cmake gfortran \
    liblapack-dev liblapacke-dev libhdf5-dev
```

Install the Python dependencies with:

```bash
python -m pip install numpy scipy h5py matplotlib pybind11-global
```

## Build

From the repository root, configure and build SpecSWD with:

```bash
cmake -S . -B build \
    -DCMAKE_C_COMPILER=gcc \
    -DCMAKE_CXX_COMPILER=g++ \
    -DCMAKE_Fortran_COMPILER=gfortran \
    -DEIGEN_INC=/path/to/eigen-5.0.0 \
    -DBUILD_LIBS=ON \
    -DBUILD_TESTS=ON
cmake --build build --parallel 4
OMP_NUM_THREADS=1 ctest --test-dir build --output-on-failure
cmake --install build --component python
```

The executables are written to `bin/`. The Python extension is installed into
`specd/lib/` and can be imported when the repository root is on
`PYTHONPATH`.

On macOS, CMake defaults to `SPECSWD_USE_ACCELERATE=ON`. The existing solver
and Eigen LAPACKE calls use a local adapter linked to the system Accelerate
framework; OpenBLAS/MKL and a separate LAPACKE installation are not needed.
Compiler and Eigen/HDF5 requirements remain the same.

To select an external backend instead, configure a separate build with:

```bash
cmake -S . -B build-openblas \
    -DSPECSWD_USE_ACCELERATE=OFF \
    -DBLA_VENDOR=OpenBLAS \
    -DLAPACKE_DIR=/path/to/openblas-prefix \
    -DEIGEN_INC=/path/to/eigen-5.0.0
```

With Accelerate disabled, the existing LAPACK/LAPACKE discovery also supports
other vendors such as MKL through `BLA_VENDOR`. Non-Apple platforms retain
that discovery by default.

The Accelerate adapter always uses the modern LAPACK interface with 32-bit
integers. CMake supplies Apple's required `ACCELERATE_NEW_LAPACK` definition;
there is no legacy interface option. Deployment targets below macOS 13.3 use
external LAPACK/LAPACKE instead.
The adapter implements the routines used by this solver: column-major
eigensolvers, unsorted QZ, selected eigenvectors, and row-major eigenvalues
without vectors for the Christoffel calculation, plus Eigen's pivoted LU.
New LAPACKE routines require
extending the adapter. Eigenproblem formulations and kernel algorithms are
unchanged by backend selection.

Archived comparisons explicitly labeled "legacy Accelerate" were measured
before this change and retain their original backend metadata.

Both backends use a default half-space scale of 50 nominal wavelengths.
The default GRL order is 30 to retain resolution of confined modes at this scale.
Override it with `SPECSWD_MAX_LAMBDA_HALF_SPACE` for the native solver or
`SpecWorkSpace(halfspace_wavelengths=...)` for `specd.torch`.

To enable ordinary Schur instead of QZ for nonsingular pencils, configure with
`-DSPECSWD_REGULAR_SCHUR=ON` (the default). This defines
`SPECSWD_REGULAR_SCHUR` in C++. The solver computes `C` from `B*C=A` using
diagonal division, sparse triangular substitution for VTI, or node-local
3×3 solves for anisotropy. Acoustic blocks use scalar divisions independently
of the solid components. Component ordering stays unchanged. After ordinary
Schur, QR recovery preserves the existing unitary `Q/Z/S/Sp` factor format,
selected eigenvectors and adjoint routines. Real 2×2 blocks are canonicalized
with small QZ solves. Failed reductions/decompositions or excessive original
pencil residuals fall back to full QZ. The eigenvalues-only path uses the same
reduction and GEEV with GGEV fallback; elastic Love retains its symmetric
definite eigensolver.

See the [four C++/PyTorch comparisons](specd/torch/benchmarks/README.md)
for current timings with only `OMP_NUM_THREADS=1`.

## Python quick start

Repeated depths describe a discontinuity. In this two-layer Love-wave model,
the final row is the half-space value:

```python
import numpy as np

from specd import DispersionTable, SpecWorkSpace

z = np.array([0.0, 35.0, 35.0])
rho = np.array([2.8, 2.8, 3.2])
vsh = np.array([3.3, 3.3, 5.5])
vsv = np.array([3.0, 3.0, 5.0])

workspace = SpecWorkSpace()
workspace.initialize("love", z, rho, vsh=vsh, vsv=vsv)

# Keep the Schur data needed by group velocity, eigenfunctions, and kernels.
phase_velocity = workspace.compute_egn(0.1, only_phase=False)
group_velocity = workspace.group_velocity()
phase_kernel = workspace.get_phase_kl(0)

# A frequency sweep may contain a different number of modes in every row.
table = workspace.compute_dispersion(
    frequencies_hz=np.geomspace(0.02, 1.0, 50),
    include_group=True,
)
table.to_hdf5("dispersion.h5")

restored = DispersionTable.from_hdf5("dispersion.h5")
mode_index, phase_row = restored.phase_row(0)
group_mode_index, group_row = restored.group_row(0)
```

For attenuation, supply the quality-factor arrays and the frequency at which
the input real velocities or storage moduli are defined:

```python
workspace.initialize(
    "love", z, rho,
    vsh=vsh,
    vsv=vsv,
    Qn=np.full(z.size, 220.0),
    Ql=np.full(z.size, 200.0),
    reference_frequency=1.0,
)
```

The runtime constitutive model is the direct constant-Q power law. Frequency
and mixed frequency/model derivatives used by group velocities and group
kernels are evaluated analytically.

## PyTorch automatic differentiation

The optional [`specd.torch`](specd/torch/README.md) implementation includes
automatic differentiation of phase and group velocities. Choose dependencies
when installing:

```bash
pip install .             # Native Python API, using the CMake modules built above
pip install ".[torch]"    # Also install the optional PyTorch dependency
```

The two APIs are imported independently:

```python
from specd import SpecWorkSpace        # Native backend
from specd.torch import SpecWorkSpace  # PyTorch backend
```

The PyTorch implementation is in `specd/torch` and works without compiled
extensions.
It supports Love, Rayleigh/Scholte, fully anisotropic media and constant-Q
attenuation without loading the compiled extensions. Set up its environment
and run the gradient example with:

```bash
conda env create -f specd/torch/environment.yml
conda activate specswd-torch
python -m specd.torch.example
```

Generate the paper examples with `OMP_NUM_THREADS=1 python
example/paper/reproduce_paper_pytorch.py`; figures and numerical data are written
to `example/paper/pytorch/`.

## Command-line programs

The three solvers accept logarithmic frequency bounds `f1`, `f2`, and a sample
count `nt`:

```bash
bin/surflove model.txt 0.02 1.0 50
bin/surfrayl model.txt 0.02 1.0 50
bin/surfani model.txt 0.02 1.0 50 30.0
```

`surfani` takes the propagation azimuth in degrees. Each program writes a
self-describing `out/specswd.h5` file containing the dispersion data and any
requested eigenfunctions or kernels. The HDF5 schema uses independent CSR
tables for phase and group results, allowing their valid modes to differ at
each frequency and azimuth.

## Examples and paper figures

The `example/love`, `example/scholte`, and `example/aniso-bench` directories
contain model construction and benchmark programs. The figures used by the
paper, together with their reproduction script, are in
[`example/paper`](example/paper).

Regenerate the numerical figures from the current solver with:

```bash
OMP_NUM_THREADS=1 python example/paper/reproduce_figures.py
```

The script writes both vector PDF and 300-dpi JPEG files into
`example/paper/`. Figure 1 is the original static model diagram and is retained
when the numerical figures are regenerated. Compile the manuscript with:

```bash
cd theory/spec
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

## Documentation

The full manual is available at <https://nqdu.github.io/SpecSWD/>. Source
documentation is organized as follows:

- [`misc/source/markdown/Tutorial.md`](misc/source/markdown/Tutorial.md): model
  conventions, solver calls, attenuation, eigenfunctions, and kernels.
- [`misc/source/markdown/io.md`](misc/source/markdown/io.md): NumPy CSR
  containers and the cross-language HDF5 schema.
- [`misc/source/markdown/pytorch.md`](misc/source/markdown/pytorch.md): optional
  PyTorch installation, tensor API, and automatic differentiation.
- [`misc/source/markdown/benchmarks.md`](misc/source/markdown/benchmarks.md):
  four C++/PyTorch comparisons with the forward-plus-phase-kernel figure.
- [`src/shared/io.hpp`](src/shared/io.hpp): public C++ IO declarations.

Build the local HTML documentation with:

```bash
python -m pip install -r misc/requirements-docs.txt
python -m pip install -e '.[torch]'
doxygen misc/doxygen/doxygen.cfg
sphinx-build -b html misc/source docs
```

The theoretical formulation and numerical examples are described in the
[SpecSWD preprint](https://doi.org/10.31223/X57J37).
