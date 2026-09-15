import importlib.util
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


def load_io_module(path):
    spec = importlib.util.spec_from_file_location("specswd_numpy_io", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def example(io):
    return io.DispersionTable.from_rows(
        np.array([0.1, 0.2]),
        [
            np.array([3.0 + 0.1j, 3.2 + 0.2j]),
            np.empty(0, dtype=np.complex128),
            np.array([3.5 + 0.3j]),
            np.array([4.0 + 0.4j, 4.1 + 0.5j, 4.4 + 0.6j]),
        ],
        azimuth_deg=np.array([10.0, 20.0]),
        phase_mode_rows=[
            np.array([0, 2]), np.array([], dtype=int),
            np.array([1]), np.array([0, 1, 4]),
        ],
        group_rows=[
            np.array([[2.0 + 0.1j, 0.2 + 0.01j]]),
            np.array([[2.1 + 0.2j, 0.3 + 0.02j], [2.3 + 0.3j, 0.4 + 0.03j]]),
            np.empty((0, 2), dtype=np.complex128),
            np.array([[2.5 + 0.4j, 0.5 + 0.04j]]),
        ],
        group_components=("x", "y"),
        group_mode_rows=[
            np.array([0]), np.array([0, 3]),
            np.array([], dtype=int), np.array([1]),
        ],
    )


def check(actual, expected):
    np.testing.assert_array_equal(actual.frequency_hz, expected.frequency_hz)
    np.testing.assert_array_equal(actual.azimuth_deg, expected.azimuth_deg)
    for left, right in (
        (actual.phase_velocity, expected.phase_velocity),
        (actual.group_velocity, expected.group_velocity),
    ):
        np.testing.assert_array_equal(left.indptr, right.indptr)
        np.testing.assert_array_equal(left.mode_index, right.mode_index)
        np.testing.assert_array_equal(left.values, right.values)
        assert left.component_names == right.component_names


def main():
    bridge = Path(sys.argv[1])
    io = load_io_module(Path(sys.argv[2]))
    expected = example(io)
    with tempfile.TemporaryDirectory() as directory:
        directory = Path(directory)
        python_file = directory / "python.h5"
        cpp_file = directory / "cpp.h5"
        full_file = directory / "cpp-full.h5"
        full_python_file = directory / "python-full.h5"

        expected.to_hdf5(python_file)
        subprocess.run([bridge, "check", python_file], check=True)

        subprocess.run([bridge, "write", cpp_file], check=True)
        check(io.DispersionTable.from_hdf5(cpp_file), expected)

        subprocess.run([bridge, "write-full", full_file], check=True)
        full = io.SolverOutput.from_hdf5(full_file)
        assert full.wave_type == 0
        assert full.kernel_observable == "phase_velocity"
        assert full.eigenfunctions.component_names == ("transverse",)
        np.testing.assert_array_equal(full.sem_depth_row(1), [0.0, 10.0, 30.0])
        np.testing.assert_array_equal(
            full.elastic_kernels.velocity,
            np.arange(1.0, 7.0).reshape(3, 1, 2),
        )
        full.to_hdf5(full_python_file)
        restored = io.SolverOutput.from_hdf5(full_python_file)
        np.testing.assert_array_equal(
            restored.eigenfunctions.values, full.eigenfunctions.values
        )


if __name__ == "__main__":
    main()
