# NumPy and HDF5 output

## Ragged dispersion rows

Each frequency and azimuth can produce a different number of valid modes.
SpecSWD represents phase and group results with independent CSR containers:

- `indptr`: `uint64`, shape `(nsolve + 1,)`;
- `mode_index`: `int32`, shape `(nentry,)`;
- `values`: `complex128`, shape `(nentry, ncomponent)`; and
- `component_names`: labels for the final dimension.

Solve rows are frequency-major and azimuth-minor:

```python
row = frequency_index * table.azimuth_deg.size + azimuth_index
start, stop = csr.indptr[row:row + 2]
modes = csr.mode_index[start:stop]
values = csr.values[start:stop]
```

An empty row has equal adjacent offsets. Independent phase and group CSR
tables allow their mode counts and mode selections to differ at the same solve
point.

## Python API

`SpecWorkSpace.compute_dispersion` returns NumPy-backed data directly:

```python
import numpy as np

table = ws.compute_dispersion(
    frequencies_hz=np.geomspace(0.02, 1.0, 50),
    azimuths_deg=np.array([0.0, 30.0, 60.0]),
    include_group=True,
)
table.to_hdf5("dispersion.h5")

restored = DispersionTable.from_hdf5("dispersion.h5")
modes, phase = restored.phase_row(4, 1)
group_modes, group_values = restored.group_row(4, 1)
```

Use `DispersionTable.from_rows` to pack existing variable-length NumPy arrays:

```python
from specd import DispersionTable

table = DispersionTable.from_rows(
    frequency_hz=np.array([0.05, 0.1, 0.2]),
    phase_rows=[
        np.array([3.0 + 0.01j, 3.4 + 0.02j]),
        np.array([3.1 + 0.01j]),
        np.empty(0, dtype=np.complex128),
    ],
    phase_mode_rows=[np.array([0, 2]), np.array([0]), np.empty(0, dtype=np.int32)],
)
```

The complete command-line result is available through `SolverOutput`:

```python
from specd import SolverOutput

result = SolverOutput.from_hdf5("out/specswd.h5")
solve_row = result.dispersion.solve_row(0, 0)
mode_start = int(result.dispersion.phase_velocity.indptr[solve_row])

depth = result.sem_depth_row(solve_row)
eigenfunction = result.eigenfunctions.row(mode_start)
elastic_kernel = result.elastic_kernels.velocity[mode_start]
```

`eigenfunction` has shape `(n_sem_point, ncomponent)`. A kernel table has shape
`(nphase_entry, nparameter, n_model_depth)`. Attenuating runs also provide
`propagation_q_inverse` with the same shape. All public numeric containers are
contiguous NumPy arrays; xarray and NumPy object arrays are not used.
For VTI Rayleigh and Scholte waves, horizontal and vertical displacement
components use real polarization amplitudes. The physical quadrature phase of
the vertical component is implicit rather than stored as a factor of `i`.

## Command-line output

The solvers write one self-describing file and replace it on each run:

```bash
surflove model.txt 0.02 1.0 50 0
surfrayl model.txt 0.02 1.0 50 1
surfani model.txt 0.02 1.0 50 30.0 1
```

The last value selects phase kernels (`0`) or group kernels (`1`). Anisotropic
group velocity is saved with `x` and `y` components. Its group kernel is the
radial derivative. Each command writes `out/specswd.h5`.

## File schema

The implemented schema is `specswd-dispersion`, version 1:

```text
/
  attrs: schema, schema_version, layout, complex_encoding
  attrs for complete output: content, wave_type, attenuation, kernel_type
  /solutions
    frequency_hz                 float64 (nfreq), units="Hz"
    azimuth_deg                  float64 (nazimuth), units="degree"
    /phase
      indptr                     uint64 (nsolve+1)
      mode_index                 int32 (nphase)
      component_names            ["phase"]
      values                     complex (nphase,1), units="km/s"
    /group                       optional, independent CSR
      indptr                     uint64 (nsolve+1)
      mode_index                 int32 (ngroup)
      component_names            ["radial"] or ["x","y"]
      values                     complex (ngroup,ncomponent), units="km/s"
  /model                         complete output only
    depth                        float64 (n_model_depth), units="km"
  /mesh
    depth_indptr                 uint64 (nsolve+1)
    depth_values                 float64 (n_sem_point_total), units="km"
  /eigenfunctions
    indptr                       uint64 (nphase+1)
    component_names              wave-dependent component labels
    values                       complex (n_eigen_point_total,ncomponent)
  /kernels
    attrs: observable, entry_alignment="solutions/phase"
    /elastic                     optional
      parameter_names
      velocity                   float64 (nphase,nparameter,n_model_depth)
      propagation_q_inverse      optional, same shape
    /acoustic                    optional, same datasets
```

`depth_indptr` handles a different SEM mesh at every solve point.
`eigenfunctions/indptr` handles the corresponding different eigenfunction
length for every phase entry. Projected kernels use the fixed input model depth
grid and therefore remain dense after the ragged phase-entry dimension has
been packed.

HDF5 has no portable native complex type. The file stores complex values as a
compound with float64 members `r` and `i`. Python exposes this representation
as `numpy.complex128`; C++ converts it to and from `specswd::Complex`.

## C++ API

IO is declared in `shared/io.hpp` and is part of the `shared` target:

```cpp
#include "shared/io.hpp"

using specswd::Complex;
using specswd::io::ComplexCSR;
using specswd::io::DispersionTable;

DispersionTable table;
table.frequency_hz = {0.05, 0.1, 0.2};
table.azimuth_deg = {0.0};
table.phase_velocity = ComplexCSR::from_rows(
    {{{3.0, 0.01}, {3.4, 0.02}}, {{3.1, 0.01}}, {}},
    {"phase"},
    {{0, 2}, {0}, {}}
);

specswd::io::write_hdf5("dispersion.h5", table);
auto restored = specswd::io::read_hdf5("dispersion.h5");
auto [begin, end] = restored.phase_velocity.row_range(1);
```

`SolverOutput`, `RaggedComplexField`, and `KernelTable` provide the complete
output form used by the executables. The implementation is in
`src/shared/io.cpp` and calls the HDF5 C API directly. Linking `shared`
propagates the required HDF5 include and library settings.

Both implementations validate offsets, shapes, mode indices, coordinate
counts, units, schema metadata, eigenfunction-to-depth alignment, and kernel
dimensions before accepting or writing data.
