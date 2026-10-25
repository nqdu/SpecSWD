# C++ and PyTorch comparisons

Only four current wave comparisons are retained. Numerical regression tests
remain in `../tests`.

| Wave | Report | Raw measurements | Runtime ratios |
| --- | --- | --- | --- |
| Love | [Report](love/efficiency.md) | [JSON](love/efficiency.json) | [Figure](love/efficiency.svg) |
| Rayleigh | [Report](rayl/efficiency.md) | [JSON](rayl/efficiency.json) | [Figure](rayl/efficiency.svg) |
| Scholte | [Report](scholte/efficiency.md) | [JSON](scholte/efficiency.json) | [Figure](scholte/efficiency.svg) |
| Anisotropic | [Report](aniso/efficiency.md) | [JSON](aniso/efficiency.json) | [Figure](aniso/efficiency.svg) |

Each comparison covers forward, forward plus phase kernels, group velocities,
and group kernels, with one/four solid layers and elastic/attenuating models.
The primary workload is **forward plus phase kernels**. Measurements use
7 GLL / 30 GRL nodes, half-space factor 50, and only `OMP_NUM_THREADS=1`.

See the [manual](../../../misc/source/markdown/benchmarks.md) for the combined
figure, method, and reproduction commands. Raw JSON records the actual
measurement command and shared-run provenance; each report is filtered from
the same validated run.
