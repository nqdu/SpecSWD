"""NumPy containers and HDF5 IO for ragged dispersion results."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np


@dataclass
class ComplexCSR:
    """Complex CSR rows indexed by solve point and local mode number."""

    indptr: np.ndarray
    mode_index: np.ndarray
    values: np.ndarray
    component_names: Tuple[str, ...]

    def __post_init__(self):
        self.indptr = np.ascontiguousarray(self.indptr, dtype=np.uint64)
        self.mode_index = np.ascontiguousarray(self.mode_index, dtype=np.int32)
        values = np.asarray(self.values, dtype=np.complex128)
        if values.ndim == 1:
            values = values[:, np.newaxis]
        self.values = np.ascontiguousarray(values)
        self.component_names = tuple(str(name) for name in self.component_names)

    @classmethod
    def from_rows(
        cls,
        rows: Sequence[np.ndarray],
        component_names: Sequence[str],
        mode_rows: Optional[Sequence[np.ndarray]] = None,
    ) -> "ComplexCSR":
        """Pack variable-length NumPy rows into a CSR container."""
        names = tuple(str(name) for name in component_names)
        if not names:
            raise ValueError("component_names must not be empty")
        if mode_rows is not None and len(mode_rows) != len(rows):
            raise ValueError("mode_rows and rows must have the same length")

        indptr = np.empty(len(rows) + 1, dtype=np.uint64)
        indptr[0] = 0
        packed_values = []
        packed_modes = []
        for row_index, row in enumerate(rows):
            values = np.asarray(row, dtype=np.complex128)
            if values.ndim == 1:
                if len(names) != 1:
                    raise ValueError(
                        f"row {row_index} requires shape (n,{len(names)})"
                    )
                values = values[:, np.newaxis]
            if values.ndim != 2 or values.shape[1] != len(names):
                raise ValueError(
                    f"row {row_index} requires shape (n,{len(names)}), "
                    f"received {values.shape}"
                )
            if mode_rows is None:
                modes = np.arange(values.shape[0], dtype=np.int32)
            else:
                modes = np.asarray(mode_rows[row_index], dtype=np.int32)
                if modes.ndim != 1 or modes.size != values.shape[0]:
                    raise ValueError(
                        f"mode row {row_index} must have {values.shape[0]} entries"
                    )
            packed_values.append(values)
            packed_modes.append(modes)
            indptr[row_index + 1] = indptr[row_index] + values.shape[0]

        if packed_values:
            values = np.ascontiguousarray(np.concatenate(packed_values, axis=0))
            modes = np.ascontiguousarray(np.concatenate(packed_modes))
        else:
            values = np.empty((0, len(names)), dtype=np.complex128)
            modes = np.empty(0, dtype=np.int32)
        result = cls(indptr, modes, values, names)
        result.validate(len(rows), "csr")
        return result

    @property
    def nrow(self) -> int:
        return max(0, self.indptr.size - 1)

    @property
    def nentry(self) -> int:
        return self.mode_index.size

    def validate(self, nrow: int, name: str = "csr") -> None:
        if not self.component_names or any(not item for item in self.component_names):
            raise ValueError(f"{name} component names must be nonempty")
        if len(set(self.component_names)) != len(self.component_names):
            raise ValueError(f"{name} component names must be unique")
        if self.indptr.ndim != 1 or self.indptr.size != nrow + 1:
            raise ValueError(f"{name} indptr must have nsolve+1 entries")
        if self.indptr[0] != 0 or np.any(self.indptr[1:] < self.indptr[:-1]):
            raise ValueError(f"{name} indptr must start at zero and be nondecreasing")
        if self.indptr[-1] != self.nentry:
            raise ValueError(f"{name} indptr[-1] must equal the entry count")
        required_shape = (self.nentry, len(self.component_names))
        if self.values.shape != required_shape:
            raise ValueError(
                f"{name} values require shape {required_shape}, "
                f"received {self.values.shape}"
            )
        for row in range(nrow):
            start, end = self.indptr[row : row + 2]
            modes = self.mode_index[int(start) : int(end)]
            if np.any(modes < 0) or np.unique(modes).size != modes.size:
                raise ValueError(
                    f"{name} mode indices must be unique and nonnegative per row"
                )

    def row(self, row: int) -> Tuple[np.ndarray, np.ndarray]:
        """Return ``(mode_index, values)`` as zero-copy NumPy views."""
        if row < 0:
            row += self.nrow
        if row < 0 or row >= self.nrow:
            raise IndexError("CSR row out of range")
        start, end = (int(value) for value in self.indptr[row : row + 2])
        return self.mode_index[start:end], self.values[start:end]


@dataclass
class DispersionTable:
    """CSR dispersion data on a frequency/azimuth grid."""

    frequency_hz: np.ndarray
    azimuth_deg: np.ndarray
    phase_velocity: ComplexCSR
    group_velocity: Optional[ComplexCSR] = None
    velocity_units: str = "km/s"

    def __post_init__(self):
        self.frequency_hz = np.ascontiguousarray(
            self.frequency_hz, dtype=np.float64
        )
        self.azimuth_deg = np.ascontiguousarray(self.azimuth_deg, dtype=np.float64)
        self.velocity_units = str(self.velocity_units)
        self.validate()

    @classmethod
    def from_rows(
        cls,
        frequency_hz,
        phase_rows: Sequence[np.ndarray],
        *,
        azimuth_deg=(0.0,),
        phase_mode_rows: Optional[Sequence[np.ndarray]] = None,
        group_rows: Optional[Sequence[np.ndarray]] = None,
        group_components: Sequence[str] = ("radial",),
        group_mode_rows: Optional[Sequence[np.ndarray]] = None,
        velocity_units: str = "km/s",
    ) -> "DispersionTable":
        """Create a table directly from variable-length solver outputs."""
        frequencies = np.asarray(frequency_hz, dtype=np.float64)
        azimuths = np.atleast_1d(np.asarray(azimuth_deg, dtype=np.float64))
        nsolve = frequencies.size * azimuths.size
        if len(phase_rows) != nsolve:
            raise ValueError(f"phase_rows must contain {nsolve} solve-point rows")
        phase = ComplexCSR.from_rows(
            phase_rows, ("phase",), phase_mode_rows
        )
        group = None
        if group_rows is not None:
            if len(group_rows) != nsolve:
                raise ValueError(f"group_rows must contain {nsolve} solve-point rows")
            group = ComplexCSR.from_rows(
                group_rows, group_components, group_mode_rows
            )
        return cls(frequencies, azimuths, phase, group, velocity_units)

    @property
    def nsolve(self) -> int:
        return self.frequency_hz.size * self.azimuth_deg.size

    def solve_row(self, frequency_index: int, azimuth_index: int = 0) -> int:
        """Map grid indices to the frequency-major CSR row."""
        if frequency_index < 0:
            frequency_index += self.frequency_hz.size
        if azimuth_index < 0:
            azimuth_index += self.azimuth_deg.size
        if not 0 <= frequency_index < self.frequency_hz.size:
            raise IndexError("frequency index out of range")
        if not 0 <= azimuth_index < self.azimuth_deg.size:
            raise IndexError("azimuth index out of range")
        return frequency_index * self.azimuth_deg.size + azimuth_index

    def validate(self) -> None:
        if self.frequency_hz.ndim != 1 or self.frequency_hz.size == 0:
            raise ValueError("frequency_hz must be a nonempty one-dimensional array")
        if np.any(~np.isfinite(self.frequency_hz)) or np.any(self.frequency_hz <= 0):
            raise ValueError("frequencies must be finite and positive")
        if self.azimuth_deg.ndim != 1 or self.azimuth_deg.size == 0:
            raise ValueError("azimuth_deg must be a nonempty one-dimensional array")
        if np.any(~np.isfinite(self.azimuth_deg)):
            raise ValueError("azimuths must be finite")
        if not self.velocity_units:
            raise ValueError("velocity_units must not be empty")
        self.phase_velocity.validate(self.nsolve, "phase_velocity")
        if self.phase_velocity.component_names != ("phase",):
            raise ValueError("phase_velocity must have the single component 'phase'")
        if self.group_velocity is not None:
            self.group_velocity.validate(self.nsolve, "group_velocity")

    def phase_row(
        self, frequency_index: int, azimuth_index: int = 0
    ) -> Tuple[np.ndarray, np.ndarray]:
        modes, values = self.phase_velocity.row(
            self.solve_row(frequency_index, azimuth_index)
        )
        return modes, values[:, 0]

    def group_row(
        self, frequency_index: int, azimuth_index: int = 0
    ) -> Tuple[np.ndarray, np.ndarray]:
        if self.group_velocity is None:
            raise ValueError("group velocity was not saved")
        return self.group_velocity.row(
            self.solve_row(frequency_index, azimuth_index)
        )

    def to_hdf5(self, path) -> None:
        """Write schema-versioned CSR data using HDF5 compound complex values."""
        import h5py

        self.validate()
        with h5py.File(Path(path), "w") as handle:
            handle.attrs["schema"] = "specswd-dispersion"
            handle.attrs["schema_version"] = np.int32(1)
            handle.attrs["layout"] = "csr"
            handle.attrs["complex_encoding"] = "compound-r-i-f64"
            solutions = handle.create_group("solutions")
            frequency = solutions.create_dataset("frequency_hz", data=self.frequency_hz)
            frequency.attrs["units"] = "Hz"
            azimuth = solutions.create_dataset("azimuth_deg", data=self.azimuth_deg)
            azimuth.attrs["units"] = "degree"
            solutions.attrs["row_order"] = "frequency-major,azimuth-minor"
            _write_csr(
                solutions.create_group("phase"),
                self.phase_velocity,
                self.velocity_units,
            )
            if self.group_velocity is not None:
                _write_csr(
                    solutions.create_group("group"),
                    self.group_velocity,
                    self.velocity_units,
                )

    @classmethod
    def from_hdf5(cls, path) -> "DispersionTable":
        """Read and validate a version-1 SpecSWD CSR HDF5 file."""
        import h5py

        with h5py.File(Path(path), "r") as handle:
            expected = {
                "schema": "specswd-dispersion",
                "schema_version": 1,
                "layout": "csr",
                "complex_encoding": "compound-r-i-f64",
            }
            for key, value in expected.items():
                if handle.attrs.get(key) != value:
                    raise ValueError(f"unsupported HDF5 {key}")
            solutions = handle["solutions"]
            if solutions.attrs.get("row_order") != "frequency-major,azimuth-minor":
                raise ValueError("unsupported solution row order")
            if solutions["frequency_hz"].attrs.get("units") != "Hz":
                raise ValueError("frequency_hz has unsupported units")
            if solutions["azimuth_deg"].attrs.get("units") != "degree":
                raise ValueError("azimuth_deg has unsupported units")
            phase = _read_csr(solutions["phase"])
            group = _read_csr(solutions["group"]) if "group" in solutions else None
            velocity_units = str(solutions["phase/values"].attrs.get("units", ""))
            if group is not None:
                group_units = str(solutions["group/values"].attrs.get("units", ""))
                if group_units != velocity_units:
                    raise ValueError("phase and group velocity units must match")
            return cls(
                solutions["frequency_hz"][...],
                solutions["azimuth_deg"][...],
                phase,
                group,
                velocity_units,
            )


@dataclass
class RaggedComplexField:
    """Variable-length complex samples with a fixed component dimension."""

    indptr: np.ndarray
    values: np.ndarray
    component_names: Tuple[str, ...]
    units: str = "normalized"

    def __post_init__(self):
        self.indptr = np.ascontiguousarray(self.indptr, dtype=np.uint64)
        values = np.asarray(self.values, dtype=np.complex128)
        if values.ndim == 1:
            values = values[:, np.newaxis]
        self.values = np.ascontiguousarray(values)
        self.component_names = tuple(str(item) for item in self.component_names)
        self.units = str(self.units)
        self.validate()

    @property
    def nrow(self) -> int:
        return max(0, self.indptr.size - 1)

    def validate(self, nrow: Optional[int] = None) -> None:
        expected_rows = self.nrow if nrow is None else int(nrow)
        if not self.component_names or len(set(self.component_names)) != len(
            self.component_names
        ) or any(not item for item in self.component_names):
            raise ValueError("field component names must be unique and nonempty")
        if not self.units:
            raise ValueError("field units must not be empty")
        if self.indptr.ndim != 1 or self.indptr.size != expected_rows + 1:
            raise ValueError("field indptr must have nrow+1 entries")
        if self.indptr[0] != 0 or np.any(self.indptr[1:] < self.indptr[:-1]):
            raise ValueError("field indptr must start at zero and be nondecreasing")
        if self.values.ndim != 2 or self.values.shape[1] != len(
            self.component_names
        ) or self.indptr[-1] != self.values.shape[0]:
            raise ValueError("field values have an inconsistent shape")

    def row(self, row: int) -> np.ndarray:
        """Return one field as a zero-copy ``(npoint,ncomponent)`` view."""
        if row < 0:
            row += self.nrow
        if not 0 <= row < self.nrow:
            raise IndexError("field row out of range")
        start, stop = (int(value) for value in self.indptr[row : row + 2])
        return self.values[start:stop]


@dataclass
class KernelTable:
    """Projected kernels with shape ``(nmode,nparameter,ndepth)``."""

    parameter_names: Tuple[str, ...]
    velocity: np.ndarray
    propagation_q_inverse: Optional[np.ndarray] = None

    def __post_init__(self):
        self.parameter_names = tuple(str(item) for item in self.parameter_names)
        self.velocity = np.ascontiguousarray(self.velocity, dtype=np.float64)
        if self.propagation_q_inverse is not None:
            self.propagation_q_inverse = np.ascontiguousarray(
                self.propagation_q_inverse, dtype=np.float64
            )
        if not self.parameter_names or any(not item for item in self.parameter_names):
            raise ValueError("kernel parameter names must be nonempty")
        if len(set(self.parameter_names)) != len(self.parameter_names):
            raise ValueError("kernel parameter names must be unique")
        if self.velocity.ndim != 3 or self.velocity.shape[1] != len(
            self.parameter_names
        ):
            raise ValueError("kernel velocity values have an inconsistent shape")
        if self.propagation_q_inverse is not None and (
            self.propagation_q_inverse.shape != self.velocity.shape
        ):
            raise ValueError("propagation-Q-inverse kernel shape differs")


@dataclass
class SolverOutput:
    """Complete command-line result backed only by NumPy containers."""

    dispersion: DispersionTable
    wave_type: int
    attenuation: bool
    kernel_type: int
    kernel_observable: str
    model_depth: np.ndarray
    sem_depth_indptr: np.ndarray
    sem_depth_values: np.ndarray
    eigenfunctions: RaggedComplexField
    elastic_kernels: Optional[KernelTable] = None
    acoustic_kernels: Optional[KernelTable] = None

    def __post_init__(self):
        self.wave_type = int(self.wave_type)
        self.attenuation = bool(self.attenuation)
        self.kernel_type = int(self.kernel_type)
        self.kernel_observable = str(self.kernel_observable)
        self.model_depth = np.ascontiguousarray(self.model_depth, dtype=np.float64)
        self.sem_depth_indptr = np.ascontiguousarray(
            self.sem_depth_indptr, dtype=np.uint64
        )
        self.sem_depth_values = np.ascontiguousarray(
            self.sem_depth_values, dtype=np.float64
        )
        self.validate()

    def validate(self) -> None:
        self.dispersion.validate()
        if self.wave_type not in (0, 1, 2):
            raise ValueError("wave_type must be 0, 1, or 2")
        if self.kernel_type not in (0, 1):
            raise ValueError("kernel_type must be 0 or 1")
        if not self.kernel_observable or self.model_depth.ndim != 1 or not len(
            self.model_depth
        ):
            raise ValueError("kernel observable and model depth are required")
        if (
            self.sem_depth_indptr.ndim != 1
            or self.sem_depth_indptr.size != self.dispersion.nsolve + 1
            or self.sem_depth_indptr[0] != 0
            or np.any(self.sem_depth_indptr[1:] < self.sem_depth_indptr[:-1])
            or self.sem_depth_indptr[-1] != self.sem_depth_values.size
        ):
            raise ValueError("SEM depth CSR has an inconsistent shape")
        self.eigenfunctions.validate(self.dispersion.phase_velocity.nentry)
        for solve_row in range(self.dispersion.nsolve):
            depth_count = int(
                self.sem_depth_indptr[solve_row + 1]
                - self.sem_depth_indptr[solve_row]
            )
            start, stop = self.dispersion.phase_velocity.indptr[
                solve_row : solve_row + 2
            ]
            for entry in range(int(start), int(stop)):
                if (
                    self.eigenfunctions.indptr[entry + 1]
                    - self.eigenfunctions.indptr[entry]
                    != depth_count
                ):
                    raise ValueError("eigenfunction and SEM depth lengths differ")
        expected = (
            self.dispersion.phase_velocity.nentry,
            len(self.model_depth),
        )
        for kernels in (self.elastic_kernels, self.acoustic_kernels):
            if kernels is None:
                continue
            if (kernels.velocity.shape[0], kernels.velocity.shape[2]) != expected:
                raise ValueError("kernel mode/depth dimensions are inconsistent")
            if self.attenuation != (kernels.propagation_q_inverse is not None):
                raise ValueError("kernel attenuation fields are inconsistent")

    def sem_depth_row(self, solve_row: int) -> np.ndarray:
        """Return the physical SEM depth coordinates for one solve row."""
        if solve_row < 0:
            solve_row += self.dispersion.nsolve
        if not 0 <= solve_row < self.dispersion.nsolve:
            raise IndexError("solve row out of range")
        start, stop = (
            int(value) for value in self.sem_depth_indptr[solve_row : solve_row + 2]
        )
        return self.sem_depth_values[start:stop]

    def to_hdf5(self, path) -> None:
        """Write the complete version-1 solver output."""
        import h5py

        self.validate()
        self.dispersion.to_hdf5(path)
        with h5py.File(Path(path), "r+") as handle:
            handle.attrs["content"] = "solver-output"
            handle.attrs["wave_type"] = np.int32(self.wave_type)
            handle.attrs["attenuation"] = np.int32(self.attenuation)
            handle.attrs["kernel_type"] = np.int32(self.kernel_type)
            model = handle.create_group("model")
            depth = model.create_dataset("depth", data=self.model_depth)
            depth.attrs["units"] = "km"
            mesh = handle.create_group("mesh")
            mesh.create_dataset("depth_indptr", data=self.sem_depth_indptr, dtype="<u8")
            values = mesh.create_dataset("depth_values", data=self.sem_depth_values)
            values.attrs["units"] = "km"
            mesh.attrs["row_alignment"] = "solutions"
            eigenfunctions = handle.create_group("eigenfunctions")
            eigenfunctions.create_dataset(
                "indptr", data=self.eigenfunctions.indptr, dtype="<u8"
            )
            eigenfunctions.create_dataset(
                "component_names",
                data=np.asarray(
                    self.eigenfunctions.component_names,
                    dtype=h5py.string_dtype("utf-8"),
                ),
            )
            eigenvalues = eigenfunctions.create_dataset(
                "values", data=self.eigenfunctions.values, dtype=np.complex128
            )
            eigenvalues.attrs["units"] = self.eigenfunctions.units
            kernel_group = handle.create_group("kernels")
            kernel_group.attrs["observable"] = self.kernel_observable
            kernel_group.attrs["entry_alignment"] = "solutions/phase"
            _write_kernel(kernel_group, "elastic", self.elastic_kernels)
            _write_kernel(kernel_group, "acoustic", self.acoustic_kernels)

    @classmethod
    def from_hdf5(cls, path) -> "SolverOutput":
        """Read complete output written by a C++ or Python solver."""
        import h5py

        dispersion = DispersionTable.from_hdf5(path)
        with h5py.File(Path(path), "r") as handle:
            if handle.attrs.get("content") != "solver-output":
                raise ValueError("HDF5 file does not contain complete solver output")
            eigenfunctions = handle["eigenfunctions"]
            field = RaggedComplexField(
                eigenfunctions["indptr"][...],
                eigenfunctions["values"][...],
                _read_strings(eigenfunctions["component_names"]),
                str(eigenfunctions["values"].attrs.get("units", "")),
            )
            kernels = handle["kernels"]
            return cls(
                dispersion=dispersion,
                wave_type=int(handle.attrs["wave_type"]),
                attenuation=bool(handle.attrs["attenuation"]),
                kernel_type=int(handle.attrs["kernel_type"]),
                kernel_observable=str(kernels.attrs["observable"]),
                model_depth=handle["model/depth"][...],
                sem_depth_indptr=handle["mesh/depth_indptr"][...],
                sem_depth_values=handle["mesh/depth_values"][...],
                eigenfunctions=field,
                elastic_kernels=_read_kernel(kernels, "elastic"),
                acoustic_kernels=_read_kernel(kernels, "acoustic"),
            )


def _write_csr(group, csr: ComplexCSR, units: str) -> None:
    group.create_dataset("indptr", data=csr.indptr, dtype="<u8")
    group.create_dataset("mode_index", data=csr.mode_index, dtype="<i4")
    import h5py

    group.create_dataset(
        "component_names",
        data=np.asarray(csr.component_names, dtype=h5py.string_dtype("utf-8")),
    )
    values = group.create_dataset("values", data=csr.values, dtype=np.complex128)
    values.attrs["units"] = units


def _read_csr(group) -> ComplexCSR:
    names = _read_strings(group["component_names"])
    return ComplexCSR(
        group["indptr"][...],
        group["mode_index"][...],
        group["values"][...],
        names,
    )


def _read_strings(dataset) -> Tuple[str, ...]:
    return tuple(
        value.decode("utf-8") if isinstance(value, bytes) else str(value)
        for value in dataset[...]
    )


def _write_kernel(parent, name: str, kernels: Optional[KernelTable]) -> None:
    if kernels is None:
        return
    import h5py

    group = parent.create_group(name)
    group.create_dataset(
        "parameter_names",
        data=np.asarray(kernels.parameter_names, dtype=h5py.string_dtype("utf-8")),
    )
    velocity = group.create_dataset("velocity", data=kernels.velocity)
    velocity.attrs["units"] = "kernel"
    if kernels.propagation_q_inverse is not None:
        quality = group.create_dataset(
            "propagation_q_inverse", data=kernels.propagation_q_inverse
        )
        quality.attrs["units"] = "kernel"


def _read_kernel(parent, name: str) -> Optional[KernelTable]:
    if name not in parent:
        return None
    group = parent[name]
    return KernelTable(
        _read_strings(group["parameter_names"]),
        group["velocity"][...],
        group["propagation_q_inverse"][...]
        if "propagation_q_inverse" in group
        else None,
    )
