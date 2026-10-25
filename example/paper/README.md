# Paper figures

Run the PyTorch reproduction from the repository root:

```bash
pip install ".[torch]"
OMP_NUM_THREADS=1 python example/paper/reproduce_paper_pytorch.py
```

This generates 14 numerical figures as PDF/JPEG pairs in `example/paper/pytorch/`,
plus Love/Scholte numerical arrays and a `manifest.json` with validation errors,
runtime, solver counts and thread settings. The default frequency and azimuth
grids match `reproduce_figures.py`. The static model diagram is not generated.
The current 50-wavelength half-space scale is paired with 30 GRL nodes;
20 nodes at that scale can lose accuracy for confined modes. Love branch
indices are recovered from the dispersion relation so filtering one root
does not relabel the remaining branches.

All SEM solves and kernels use `specd.torch`; plots label derivatives as
autodiff. Love curves are checked against the analytical dispersion equation.
Scholte curves and finite differences use the independent CPS330 reference
when the optional prebuilt `specd.lib.cps330` module is available. The script
automatically omits those reference points when unavailable; `--reference cps`
requires CPS330 and `--reference none` omits it explicitly. The C++ SEM module
is never loaded.

The manifest reports reference discrepancies rather than declaring every
point exact: higher Love modes near cutoff can have large group-kernel
discretization errors on the original mesh. CPS zero sentinels for unavailable
modes are excluded from phase/group error metrics.

Use `--quick` for a preview, `--sections love scholte aniso` to select cases,
and `--output /path/to/figures` to change the output directory.

The original native reproduction remains available:

```bash
python example/paper/reproduce_figures.py
```
