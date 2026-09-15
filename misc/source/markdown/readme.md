# SpecSWD

SpecSWD computes surface-wave dispersion and sensitivity kernels for
one-dimensional layered media. It supports:

- Love waves in VTI media;
- Rayleigh and Scholte waves in coupled solid/fluid VTI media;
- fully anisotropic media with arbitrary propagation azimuth;
- causal constant-Q attenuation referenced to a user-selected frequency;
- phase velocity, group velocity, eigenfunctions, and Fréchet kernels; and
- NumPy/CSR storage of variable-mode dispersion results in HDF5.

The numerical core uses the spectral-element method in depth. Fully
anisotropic dispersion is solved as a quadratic eigenvalue problem. Group
kernels reuse the generalized Schur factors, so each constrained adjoint solve
after the decomposition costs $O(n^2)$.

## Requirements

- CMake 3.25 or newer;
- a C++17 compiler and a Fortran compiler;
- Eigen 5;
- LAPACK and LAPACKE through MKL, OpenBLAS, FlexiBLAS, or equivalent;
- Python, NumPy, and pybind11 when `BUILD_LIBS=ON`;
- the HDF5 C library;
- h5py for Python HDF5 input and output; and
- Doxygen and Sphinx only when building documentation.


## Configure and build

From the repository root:

```bash
cmake -S . -B build \
    -DEIGEN_INC=/path/to/eigen \
    -DBUILD_LIBS=ON \
    -DBUILD_TESTS=ON
cmake --build build -j 8
ctest --test-dir build --output-on-failure
```

Set `CC`, `CXX`, and `FC` in the environment when non-default compilers are
needed:

```bash
CC=gcc CXX=g++ FC=gfortran cmake -S . -B build [options]
```

Install the Python extension into `specd/lib`:

```bash
cmake --install build --component python
```

Run Python from the repository root, or add the repository to `PYTHONPATH`:

```bash
export PYTHONPATH=/path/to/SpecSWD:${PYTHONPATH}
python -c "import specd; print(specd.__file__)"
```

HDF5 IO is part of the `shared` C++ target and uses the HDF5 C API. The command
line solvers write `out/specswd.h5`; the old `swd.txt` and `database.bin`
outputs have been removed.

## Numerical types

Public C++ code uses `specswd::Real` and `specswd::Complex`, declared in
`include/numerical.hpp`. `Real` defaults to `double` and can be selected with
`SPECSWD_REAL_TYPE`; `Complex` is `std::complex<Real>`.

Generalized Schur solves use double precision by default. Configure with
`-DSPECSWD_EGN_DOUBLE=OFF` only when reduced eigensolver precision is an
acceptable performance tradeoff.

## Documentation

Generate the C++ reference from the repository root:

```bash
doxygen misc/doxygen/doxygen.cfg
```

Generate the Sphinx manual when its documentation dependencies are installed:

```bash
sphinx-build -b html misc/source misc/_build/html
```

Continue with the [tutorial](Tutorial.md) for model construction and solver
usage, and see [CSR and HDF5 output](io.md) for variable mode counts.
