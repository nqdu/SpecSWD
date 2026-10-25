"""Ragged tensor dispersion rows with explicit, detached HDF5 export."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class DispersionTable:
    frequency_hz: torch.Tensor
    azimuth_deg: torch.Tensor
    phase_rows: list
    group_rows: list | None = None
    group_components: tuple = ("radial",)
    phase_mode_rows: list | None = None
    group_mode_rows: list | None = None

    def __post_init__(self):
        count = self.frequency_hz.numel() * self.azimuth_deg.numel()
        if len(self.phase_rows) != count or (self.group_rows is not None and len(self.group_rows) != count):
            raise ValueError("dispersion rows must match the frequency/azimuth grid")
        for name, rows in (("phase", self.phase_rows), ("group", self.group_rows)):
            modes = getattr(self, name + "_mode_rows")
            if rows is None:
                continue
            if modes is None:
                modes = [torch.arange(len(row), device=row.device) for row in rows]
                setattr(self, name + "_mode_rows", modes)
            if len(modes) != len(rows) or any(len(m) != len(r) for m, r in zip(modes, rows)):
                raise ValueError("mode indices must match dispersion row lengths")

    def phase_row(self, row):
        value = self.phase_rows[row]
        return self.phase_mode_rows[row], value

    def group_row(self, row):
        if self.group_rows is None:
            raise ValueError("group velocities were not computed")
        value = self.group_rows[row]
        return self.group_mode_rows[row], value

    def to_hdf5(self, path):
        """Write the existing specswd-dispersion v1 schema; detach at IO only."""
        import h5py
        import numpy as np

        def array(value):
            return value.detach().cpu().numpy()

        with h5py.File(path, "w") as handle:
            handle.attrs.update(schema="specswd-dispersion", schema_version=np.int32(1),
                                layout="csr", complex_encoding="compound-r-i-f64")
            solutions = handle.create_group("solutions")
            solutions.attrs["row_order"] = "frequency-major,azimuth-minor"
            solutions.create_dataset("frequency_hz", data=array(self.frequency_hz)).attrs["units"] = "Hz"
            solutions.create_dataset("azimuth_deg", data=array(self.azimuth_deg)).attrs["units"] = "degree"
            for name, rows, components in (("phase", self.phase_rows, ("phase",)),
                                           ("group", self.group_rows, self.group_components)):
                if rows is None:
                    continue
                group = solutions.create_group(name)
                lengths = [len(row) for row in rows]
                group.create_dataset("indptr", data=np.cumsum([0] + lengths), dtype="<u8")
                modes = getattr(self, name + "_mode_rows")
                group.create_dataset("mode_index", data=array(torch.cat(modes)), dtype="<i4")
                group.create_dataset("component_names", data=components, dtype=h5py.string_dtype("utf-8"))
                values = torch.cat([row.reshape(len(row), len(components)) for row in rows])
                group.create_dataset("values", data=array(values).astype(np.complex128)).attrs["units"] = "km/s"

    @classmethod
    def from_hdf5(cls, path, *, device=None):
        """Read detached float64/complex128 tensors from a v1 dispersion file."""
        import h5py
        import numpy as np

        with h5py.File(path, "r") as handle:
            if handle.attrs.get("schema") != "specswd-dispersion" or handle.attrs.get("schema_version") != 1:
                raise ValueError("unsupported SpecSWD dispersion schema")
            solutions = handle["solutions"]
            frequencies = torch.as_tensor(solutions["frequency_hz"][...], device=device)
            angles = torch.as_tensor(solutions["azimuth_deg"][...], device=device)

            def rows(name):
                group = solutions[name]
                offsets = group["indptr"][...]
                data = group["values"][...]
                if data.dtype.fields:
                    data = data["r"] + 1j * data["i"]
                data = torch.as_tensor(np.asarray(data, dtype=np.complex128), device=device)
                values = [data[int(a):int(b)] for a, b in zip(offsets[:-1], offsets[1:])]
                modes = torch.as_tensor(group["mode_index"][...].astype(np.int64), device=device)
                return values, [modes[int(a):int(b)] for a, b in zip(offsets[:-1], offsets[1:])]

            phase, phase_modes = rows("phase")
            phase = [row.reshape(-1) for row in phase]
            group, group_modes = rows("group") if "group" in solutions else (None, None)
            components = tuple(solutions["group/component_names"].asstr()[...]) if group is not None else ("radial",)
            return cls(frequencies, angles, phase, group, components, phase_modes, group_modes)
