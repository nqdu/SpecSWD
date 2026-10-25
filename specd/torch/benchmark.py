"""Compare the public PyTorch and C++ APIs on identical CPU workloads.

Run from the repository root with ``python -m specd.torch.benchmark --help``.
Model setup and correctness checks are excluded from timing; mesh creation,
assembly, eigensolving and requested derivatives are included on every solve.
"""

import argparse
import gc
import importlib.util
import json
import os
from pathlib import Path
import platform
import statistics
import sys
import time


WORKLOADS = {
    "phase_only": "Phase velocities only (no eigenvectors)",
    "forward": "Phase and eigenvectors (no autograd graph)",
    "phase_kernel": "Forward + fundamental-mode phase kernels",
    "group": "Phase and all group velocities",
    "group_kernel": "Phase, all group velocities and fundamental-mode group kernels",
}


def make_model(wave, layers, attenuation):
    """Layered crust with a faster half-space; Scholte adds a 2 km ocean."""
    import numpy as np

    z, rho, vp, vs = [], [], [], []
    if wave == "scholte":
        z.extend([0., 2.])
        rho.extend([1., 1.])
        vp.extend([1.5, 1.5])
        vs.extend([0., 0.])
    top = 2. if wave == "scholte" else 0.
    for layer in range(layers):
        fraction = layer / max(1, layers - 1)
        z.extend([top + 35 * layer / layers, top + 35 * (layer + 1) / layers])
        rho.extend([2.65 + .2 * fraction] * 2)
        vp.extend([5.6 + .8 * fraction] * 2)
        vs.extend([3.1 + .5 * fraction] * 2)
    z.append(top + 35.)
    rho.append(3.2)
    vp.append(8.)
    vs.append(5.5)
    z, rho, vp, vs = [np.asarray(x, dtype=np.float64) for x in (z, rho, vp, vs)]
    if wave == "love":
        params = dict(vsh=vs, vsv=.95 * vs)
        if attenuation:
            params.update(Qn=np.full_like(vs, 220), Ql=np.full_like(vs, 180))
    elif wave in ("rayl", "scholte"):
        params = dict(vph=vp, vpv=vp, vsv=vs, eta=np.ones_like(vs))
        if attenuation:
            params.update(Qa=np.full_like(vs, 300), Qc=np.full_like(vs, 300), Ql=np.full_like(vs, 180))
    else:
        mu = rho * vs**2
        lam = rho * vp**2 - 2 * mu
        c21 = np.stack([
            lam + 2 * mu if i == j and i < 3 else
            lam if i < j < 3 else mu if i == j else np.zeros_like(mu)
            for i in range(6) for j in range(i, 6)
        ])
        c21[0] *= 1.02
        c21[5] += .1
        params = dict(c21=c21)
        if attenuation:
            params["Qani"] = np.stack((np.full_like(vs, 300), np.full_like(vs, 180)))
    return "rayl" if wave == "scholte" else wave, z, rho, params


def load_native(extension):
    if extension is not None:
        path = Path(extension).expanduser().resolve()
        # Load the requested binary before importing specd. Importing the
        # installed extension first can coalesce weak C++/Eigen symbols across
        # the two binaries on macOS and contaminate backend measurements.
        name = "specd.lib.libswd"
        module = sys.modules.get(name)
        if module is not None and Path(module.__file__).resolve() != path:
            raise RuntimeError("a different native extension is already loaded; run this benchmark in a fresh process")
        if module is None:
            spec = importlib.util.spec_from_file_location(name, path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            sys.modules[name] = module
    import specd.specd as native
    if "reference_frequency" not in native.libswd.init_mesh.__doc__:
        raise RuntimeError("the native extension is stale; build libswd and pass --native-extension /path/to/libswd.so")
    return native.SpecWorkSpace, Path(native.libswd.__file__).resolve()


def native_build_info(extension_path):
    """Record the local CMake configuration when a build tree is available."""
    keys = ("CMAKE_CXX_COMPILER:", "CMAKE_CXX_FLAGS:", "CMAKE_CXX_FLAGS_RELEASE:",
            "SPECSWD_EGN_DOUBLE:", "CMAKE_BUILD_TYPE:", "SPECSWD_USE_ACCELERATE:",
            "SPECSWD_ACCELERATE_NEW_LAPACK:", "SPECSWD_LAPACK_BACKEND:",
            "SPECSWD_REGULAR_SCHUR:", "BLA_VENDOR:")
    for parent in extension_path.parents:
        cache = parent / "CMakeCache.txt"
        if cache.is_file():
            settings = {line.split(":", 1)[0]: line.split("=", 1)[1]
                        for line in cache.read_text().splitlines() if line.startswith(keys) and "=" in line}
            flags = parent / "src/libswd/CMakeFiles/libswd.dir/flags.make"
            if flags.is_file():
                settings.update({"effective_" + line.split(" = ", 1)[0].lower(): line.split(" = ", 1)[1]
                                 for line in flags.read_text().splitlines() if line.startswith("CXX_FLAGS = ")})
            return settings
    return {}


def align_native_kernel(array, wave, nrows, attenuation):
    """Combine native fluid rows with their corresponding torch input rows."""
    result = array[:nrows].copy()
    if wave in ("rayl", "scholte"):
        result[0] += array[nrows]  # vp_ac -> vph
        result[3] += array[nrows + 1]  # rho_ac -> rho
        if attenuation:
            result[6] += array[nrows + 2]  # Qk_ac -> Qc (inverse-Q convention)
    elif wave == "aniso":
        result[0] += array[nrows]  # kappa_ac -> c11
        result[21] += array[nrows + 1]
        if attenuation:
            result[22] += array[nrows + 2]
    return result


def validate(native, tensor, wave, attenuation, frequencies, azimuth):
    import numpy as np
    import torch

    errors = dict(phase=0., group=0., phase_kernel=0., group_kernel=0.)
    meshes, modes = [], []
    for freq in frequencies:
        expected = native.compute_egn(freq, azimuth, only_phase=False)
        expected_phase = expected.copy()
        actual = tensor.compute_egn(freq, azimuth, only_phase=False)
        np.testing.assert_allclose(actual.detach().numpy(), expected, atol=2e-6, rtol=2e-6)
        if not len(actual):
            raise RuntimeError(f"no modes at {freq} Hz; cannot benchmark fundamental-mode kernels")
        errors["phase"] = max(errors["phase"], float(np.max(np.abs(actual.detach().numpy() - expected))))
        expected_group = native.group_velocity()
        actual_group = tensor.group_velocity().detach().numpy()
        np.testing.assert_allclose(actual_group, expected_group, atol=2e-5, rtol=2e-5)
        errors["group"] = max(errors["group"], float(np.max(np.abs(actual_group - expected_group))))
        native_vector = native.get_egnfunc(0, return_displ=False)
        native_size = (native_vector[0] if attenuation else native_vector).size
        torch_size = tensor.get_egnfunc(0, return_displ=False).numel()
        if native_size != torch_size:
            raise AssertionError(f"different physical matrix sizes: C++ {native_size}, torch {torch_size}")
        meshes.append(torch_size)
        modes.append(len(actual))
        for observable in ("phase", "group"):
            expected = getattr(native, "get_" + observable + "_kl")(0)
            actual = getattr(tensor, "get_" + observable + "_kl")(0)
            expected = expected if attenuation else (expected,)
            actual = actual if attenuation else (actual,)
            for ref, value in zip(expected, actual):
                ref = align_native_kernel(ref, wave, len(value), attenuation)
                value = value.detach().numpy()
                np.testing.assert_allclose(value, ref, atol=2e-5, rtol=2e-4)
                key = observable + "_kernel"
                errors[key] = max(errors[key], float(np.max(np.abs(value - ref))))
        # Validate the separate eigenvalues-only driver as well as the vector path.
        phase_only_native = native.compute_egn(freq, azimuth, only_phase=True)
        with torch.no_grad():
            phase_only_torch = tensor.compute_egn(freq, azimuth, only_phase=True).numpy()
        for phase_only in (phase_only_native, phase_only_torch):
            np.testing.assert_allclose(phase_only, expected_phase, atol=2e-6, rtol=2e-6)
            errors["phase"] = max(errors["phase"], float(np.max(np.abs(phase_only - expected_phase))))
    return errors, meshes, modes


def action(workspace, backend, workload, frequencies, azimuth):
    import torch

    def run():
        # no_grad applies only to inference. Group velocity itself uses autograd.
        with torch.no_grad() if backend == "torch" and workload in ("phase_only", "forward") else torch.enable_grad():
            for frequency in frequencies:
                workspace.compute_egn(frequency, azimuth, only_phase=workload == "phase_only")
                if workload in ("group", "group_kernel"):
                    workspace.group_velocity()
                if workload == "phase_kernel":
                    workspace.get_phase_kl(0)
                elif workload == "group_kernel":
                    workspace.get_group_kl(0)
    return run


def elapsed(run, iterations):
    enabled = gc.isenabled()
    gc.disable()
    try:
        start = time.perf_counter_ns()
        for _ in range(iterations):
            run()
        return (time.perf_counter_ns() - start) / 1e9
    finally:
        if enabled:
            gc.enable()


def measure_pair(runs, repeats, warmup, target_seconds, nfrequency):
    """Calibrate batches; alternate backend order and retain raw samples."""
    iterations = {}
    for backend, run in runs.items():
        for _ in range(warmup):
            run()
        count = 1
        while elapsed(run, count) < target_seconds and count < 64:
            count *= 2
        iterations[backend] = count
    samples = {backend: [] for backend in runs}
    for repeat in range(repeats):
        order = list(runs) if repeat % 2 == 0 else list(reversed(runs))
        for backend in order:
            gc.collect()
            seconds = elapsed(runs[backend], iterations[backend])
            samples[backend].append(seconds * 1000 / iterations[backend] / nfrequency)
    return {backend: dict(median_ms=statistics.median(values), min_ms=min(values),
                           max_ms=max(values), samples_ms=values,
                           iterations_per_sample=iterations[backend])
            for backend, values in samples.items()}


def write_report(result, directory):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "efficiency.json").write_text(json.dumps(result, indent=2) + "\n")
    metadata = result["metadata"]
    native_build = metadata['native_build']
    backend = native_build.get('SPECSWD_LAPACK_BACKEND',
                              f"Accelerate={native_build.get('SPECSWD_USE_ACCELERATE', 'not recorded')}; "
                              f"modern={native_build.get('SPECSWD_ACCELERATE_NEW_LAPACK', 'not recorded')}")
    workloads = {name: title for name, title in WORKLOADS.items()
                 if any(row["workload"] == name for row in result["results"])}
    lines = ["# PyTorch versus C++ efficiency", "",
             f"Measured {metadata['measured_at_utc']} on {metadata['platform']} ({metadata['machine']}).",
             f"Python {metadata['python']}, PyTorch {metadata['torch']}, NumPy {metadata['numpy']}; "
             f"OMP_NUM_THREADS={metadata['threads']}; other thread pools use their defaults.", "",
             f"Native compiler: `{metadata['native_build'].get('CMAKE_CXX_COMPILER', 'unavailable')}`. "
             f"Effective C++ flags: `{metadata['native_build'].get('effective_cxx_flags', 'unavailable')}`.", "",
             f"C++ LAPACK backend: `{backend}`.", "",
             f"C++ ordinary Schur option: `{metadata['native_build'].get('SPECSWD_REGULAR_SCHUR', 'not recorded')}`.", "",
             "The primary comparison workload is `phase_kernel`: a fresh forward solve "
             "followed by fundamental-mode phase kernels. Its measured time includes "
             "both operations; separate forward timings must not be added to it.", "",
             "The C++ backend is called through its existing pybind11/Python interface. "
             "Each timing includes mesh creation, interpolation, assembly, eigensolving, "
             "mode filtering and the requested output/derivatives. Model initialization, "
             "imports, correctness checks, IO and plotting are excluded. Both backends use "
             f"float64/complex128, {metadata.get('ngll', 7)}-node GLL, "
             f"{metadata.get('ngrl', 20)}-node GRL and a "
             f"{metadata.get('halfspace_wavelengths', 20):g}-wavelength half-space.", "",
             "Phase-only timings request only eigenvalues with `only_phase=True`. "
             "Forward timings include eigenvectors in both implementations. Both disable "
             "autograd graph construction in PyTorch. Other tasks build a fresh graph/solve "
             "at every frequency. Kernel tasks return every model-parameter derivative for "
             "mode 0; attenuating cases include both velocity and wave inverse-Q kernels. "
             "Group tasks evaluate every retained mode; group-kernel tasks additionally "
             "compute the radial group kernels of mode 0.", "",
             f"Frequencies: {metadata['frequencies_hz']} Hz; azimuth {metadata['azimuth_deg']} degrees. "
             f"Each entry is the median of {metadata['repeats']} repeated samples, divided "
             "by the number of frequencies and calibrated batch iterations. Backend order "
             "alternates between repeats; garbage collection runs outside timed regions. "
             "The ratio is PyTorch time / C++ time, so values above 1 favor C++.", "",
             "Physical DOF and retained mode counts are recorded per frequency in the JSON. "
             "Both anisotropic eigensystems have twice the physical DOF. "
             "The C++ implementation uses QZ/Schur; PyTorch uses dense eigensolvers and "
             "higher-order autograd. These are measurements of the current implementations, "
             "rather than a general limit on either language or GPU performance.", "",
             "All cases first passed phase, group and both kernel comparisons. Native "
             "fluid kernel rows are combined into their corresponding original torch "
             "parameter rows before validation. CPU timing only; no GPU or peak-memory "
             "comparison was performed.", "", "## Runtime ratio ranges", "",
             "Ranges span the measured layer counts and attenuation settings. "
             "Every ratio is PyTorch / C++ time; below 1 favors PyTorch.", "",
             "| Wave | " + " | ".join(workloads.values()) + " |",
             "| --- | " + " | ".join("---:" for _ in workloads) + " |"]
    for wave in dict.fromkeys(row["wave"] for row in result["results"]):
        ranges = []
        for workload in workloads:
            ratios = [row["ratio"] for row in result["results"]
                      if row["wave"] == wave and row["workload"] == workload]
            ranges.append(f"{min(ratios):.2f}–{max(ratios):.2f}×" if ratios else "—")
        lines.append("| " + " | ".join([wave, *ranges]) + " |")
    lines += ["", "![Runtime ratios](efficiency.svg)", ""]
    for workload, title in workloads.items():
        lines += ["## " + title, "",
                  "| Wave | Attenuation | Solid layers | Physical DOF range | Modes | C++ ms/frequency | PyTorch ms/frequency | Ratio |",
                  "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for row in result["results"]:
            if row["workload"] != workload:
                continue
            bounds = lambda values: f"{min(values)}–{max(values)}"
            lines.append(f"| {row['wave']} | {'yes' if row['attenuation'] else 'no'} | "
                         f"{row['solid_layers']} | {bounds(row['physical_dof'])} | {bounds(row['modes'])} | "
                         f"{row['cpp']['median_ms']:.3f} | {row['torch']['median_ms']:.3f} | {row['ratio']:.1f}× |")
        lines.append("")
    lines += ["## Reproduce", "", "Run from the repository root with the `specswd-torch` environment:", "",
              "```bash", metadata["command"], "```", "",
              "A current native extension is required; `--native-extension` selects a separate "
              "build without replacing the installed one. The JSON includes extension identity, "
              "model arrays, raw timing samples, batch counts and correctness errors. "
              "Use `--help` to change frequencies, models, layer counts, repeats or threads.", ""]
    (directory / "efficiency.md").write_text("\n".join(lines))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nrows = (len(workloads) + 1) // 2
    fig, axes = plt.subplots(nrows, 2, figsize=(15, 5 * nrows), constrained_layout=True)
    for axis, (workload, title) in zip(axes.flat, workloads.items()):
        rows = [r for r in result["results"] if r["workload"] == workload]
        labels = [f"{r['wave']} / {r['solid_layers']}L" + (" / Q" if r["attenuation"] else "") for r in rows]
        colors = ["#d97706" if r["attenuation"] else "#2563eb" for r in rows]
        axis.bar(range(len(rows)), [r["ratio"] for r in rows], color=colors)
        axis.set_xticks(range(len(rows)), labels, rotation=65, ha="right", fontsize=7)
        axis.set_yscale("log")
        axis.axhline(1, color="black", linewidth=.8)
        axis.set_title(title, fontsize=10)
        axis.set_ylabel("PyTorch time / C++ time")
        axis.grid(axis="y", alpha=.2)
    for axis in axes.flat[len(workloads):]:
        axis.set_visible(False)
    fig.suptitle(f"SpecSWD CPU runtime ratios (OMP_NUM_THREADS={metadata['threads']}; higher favors C++)")
    fig.savefig(directory / "efficiency.svg")
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-extension", default=os.environ.get("SPECSWD_REFERENCE_EXTENSION"))
    parser.add_argument("--waves", nargs="+", choices=["love", "rayl", "scholte", "aniso"],
                        default=["love", "rayl", "scholte", "aniso"])
    parser.add_argument("--layers", nargs="+", type=int, default=[1, 4])
    parser.add_argument("--frequencies", nargs="+", type=float, default=[.05, .1, .2])
    parser.add_argument("--azimuth", type=float, default=25.)
    parser.add_argument("--attenuation", choices=["both", "elastic", "attenuating"], default="both")
    parser.add_argument("--workloads", nargs="+", choices=list(WORKLOADS), default=list(WORKLOADS))
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--sample-ms", type=float, default=10.)
    parser.add_argument("--threads", type=int, default=1,
                        help="OMP_NUM_THREADS only; other thread pools are not configured")
    parser.add_argument("--output", type=Path, default=Path("specd/torch/benchmarks"))
    arguments = sys.argv[1:] if argv is None else argv
    args = parser.parse_args(arguments)
    if (min(args.layers) < 1 or min(args.frequencies) <= 0 or args.repeats < 1
            or args.warmup < 0 or args.sample_ms <= 0 or args.threads < 1):
        parser.error("layers, frequencies, repeats, sample-ms and threads must be positive; warmup must be nonnegative")

    # -m imports the package before this module. Restart once if necessary so
    # the requested OpenMP setting takes effect before libraries are loaded.
    limits = {"OMP_NUM_THREADS": str(args.threads)}
    if any(os.environ.get(key) != value for key, value in limits.items()):
        os.execve(sys.executable, [sys.executable, "-m", "specd.torch.benchmark", *arguments],
                  dict(os.environ, **limits))
    import datetime
    import hashlib
    import shlex
    import numpy as np
    import torch
    from .solver import SpecWorkSpace

    Native, extension_path = load_native(args.native_extension)
    result = {"metadata": dict(
        measured_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        platform=platform.platform(), machine=platform.machine(), python=platform.python_version(),
        torch=torch.__version__, numpy=np.__version__, threads=args.threads,
        halfspace_wavelengths=SpecWorkSpace().halfspace_wavelengths,
        ngll=SpecWorkSpace().ngll, ngrl=SpecWorkSpace().ngrl,
        frequencies_hz=args.frequencies, azimuth_deg=args.azimuth, repeats=args.repeats,
        warmup=args.warmup, target_sample_ms=args.sample_ms, native_extension=str(extension_path),
        native_extension_sha256=hashlib.sha256(extension_path.read_bytes()).hexdigest(),
        native_build=native_build_info(extension_path),
        native_loading="requested extension preloaded before specd import" if args.native_extension else "installed extension",
        thread_environment={key: os.environ.get(key) for key in
                            ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                             "VECLIB_MAXIMUM_THREADS")},
        torch_num_threads=torch.get_num_threads(),
        torch_num_interop_threads=torch.get_num_interop_threads(),
        torch_parallel_info=torch.__config__.parallel_info(),
        torch_build=torch.__config__.show(),
        command=shlex.join(["python", "-m", "specd.torch.benchmark", *arguments])),
        "models": [], "results": []}
    attenuation_cases = [False, True] if args.attenuation == "both" else [args.attenuation == "attenuating"]
    for wave in args.waves:
        for layers in args.layers:
            for attenuation in attenuation_cases:
                wave_type, z, rho, parameters = make_model(wave, layers, attenuation)
                native = Native()
                native.initialize(wave_type, z, rho, **parameters)
                tensor = SpecWorkSpace(device="cpu").initialize(wave_type, z, rho, **parameters)
                errors, dof, modes = validate(native, tensor, wave, attenuation, args.frequencies, args.azimuth)
                case = dict(wave=wave, attenuation=attenuation, solid_layers=layers)
                result["models"].append(dict(**case, depth=z.tolist(), rho=rho.tolist(),
                                             parameters={k: v.tolist() for k, v in parameters.items()}))
                for workload in args.workloads:
                    runs = {name: action(ws, name, workload, args.frequencies, args.azimuth)
                            for name, ws in (("cpp", native), ("torch", tensor))}
                    timing = measure_pair(runs, args.repeats, args.warmup, args.sample_ms / 1000, len(args.frequencies))
                    ratio = timing["torch"]["median_ms"] / timing["cpp"]["median_ms"]
                    result["results"].append(dict(**case, workload=workload, physical_dof=dof,
                                                  modes=modes, max_absolute_errors=errors, ratio=ratio, **timing))
                    print(f"{wave:7} layers={layers} Q={attenuation!s:5} {workload:12} "
                          f"C++={timing['cpp']['median_ms']:.3f} ms "
                          f"torch={timing['torch']['median_ms']:.3f} ms ratio={ratio:.1f}x", flush=True)
    write_report(result, args.output)
    print(f"Saved {args.output / 'efficiency.md'}", flush=True)


if __name__ == "__main__":
    main()
