"""Render the four saved C++/PyTorch comparisons for the Sphinx manual."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
WAVES = {"love": "Love", "rayl": "Rayleigh", "scholte": "Scholte", "aniso": "Anisotropic"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "specd/torch/benchmarks")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "misc/source/static/pytorch-benchmarks.svg")
    args = parser.parse_args()
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    metadata = None
    for axis, (wave, title) in zip(axes.flat, WAVES.items()):
        result = json.loads((args.input / wave / "efficiency.json").read_text())
        current = result["metadata"]
        if metadata is not None:
            for key in ("ngll", "ngrl", "halfspace_wavelengths", "threads", "frequencies_hz"):
                if current[key] != metadata[key]:
                    raise ValueError(f"inconsistent {key} across benchmark reports")
        metadata = current
        rows = [row for row in result["results"] if row["workload"] == "phase_kernel"]
        if not rows or any(row["wave"] != wave for row in result["results"]):
            raise ValueError(f"expected the {wave} comparison with phase kernels")
        positions = np.arange(len(rows))
        cpp = np.array([row["cpp"]["median_ms"] for row in rows])
        tensor = np.array([row["torch"]["median_ms"] for row in rows])
        axis.bar(positions - .2, cpp, .4, label="C++", color="#2563eb")
        axis.bar(positions + .2, tensor, .4, label="PyTorch", color="#d97706")
        axis.set_xticks(positions, [f"{row['solid_layers']} layer(s)\n" +
                                   ("attenuating" if row["attenuation"] else "elastic")
                                   for row in rows])
        axis.set_title(title)
        axis.set_ylabel("Milliseconds per frequency (lower is faster)")
        axis.set_ylim(0, max(cpp.max(), tensor.max()) * 1.25)
        axis.grid(axis="y", alpha=.2)
        axis.set_axisbelow(True)
        for position, ratio in zip(positions, tensor / cpp):
            axis.annotate(f"{ratio:.2f}×", (position, max(cpp[position], tensor[position])),
                          xytext=(0, 5), textcoords="offset points", ha="center", fontsize=9)
    axes.flat[0].legend()
    fig.suptitle(
        "Forward + fundamental-mode phase kernels\n"
        f"{metadata['ngll']} GLL / {metadata['ngrl']} GRL; half-space factor "
        f"{metadata['halfspace_wavelengths']:g}; OMP_NUM_THREADS={metadata['threads']}\n"
        "Annotations: PyTorch time / C++ time", fontsize=12)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output)
    plt.close(fig)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
