#!/usr/bin/env python3
"""Reproduce the paper's 14 numerical figures using specd.torch autodiff.

Run: OMP_NUM_THREADS=1 python example/paper/reproduce_paper_pytorch.py
Outputs are written to example/paper/pytorch. CPS330 is used only as an
optional independent Scholte reference; every SEM solve/kernel uses PyTorch.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

os.environ['OMP_NUM_THREADS'] = '1'
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

import numpy as np
import torch
from specd.torch import SpecWorkSpace
import reproduce_figures as paper


STATISTICS = dict(solves=0, phase_kernel_calls=0, group_kernel_calls=0)


def numpy(value: torch.Tensor) -> np.ndarray:
    return value.detach().cpu().numpy().copy()


class PlotWorkspace:
    """NumPy plotting boundary; the underlying graph stays in specd.torch."""

    def __init__(self):
        self.workspace = SpecWorkSpace(device='cpu')

    def initialize(self, *args, **kwargs):
        self.workspace.initialize(*args, **kwargs)
        return self

    def compute_egn(self, *args, **kwargs):
        result = self.workspace.compute_egn(*args, **kwargs)
        if not torch.isfinite(result).all():
            raise FloatingPointError('nonfinite phase velocity')
        STATISTICS['solves'] += 1
        if STATISTICS['solves'] % 25 == 0:
            print(f"Completed {STATISTICS['solves']} PyTorch solves", flush=True)
        return numpy(result if self.workspace._has_att else result.real)

    def group_velocity(self):
        result = self.workspace.group_velocity()
        if not torch.isfinite(result).all():
            raise FloatingPointError('nonfinite group velocity')
        return numpy(result if self.workspace._has_att else result.real)

    def _kernels(self, mode, observable):
        # The paper plots real-velocity derivatives, including material 1/Q
        # rows. Avoid computing unused wave inverse-Q output derivatives.
        result = self.workspace._kernel_array(mode, observable, 'real')
        if not torch.isfinite(result).all():
            raise FloatingPointError(f'nonfinite {observable} kernel')
        STATISTICS[observable + '_kernel_calls'] += 1
        values = numpy(result)
        return (values, None) if self.workspace._has_att else values

    def get_phase_kl(self, mode):
        return self._kernels(mode, 'phase')

    def get_group_kl(self, mode):
        return self._kernels(mode, 'group')

    def get_znodes(self):
        return numpy(self.workspace.mesh.depth)

    def get_egnfunc(self, mode, return_displ):
        values = self.workspace.get_egnfunc(mode, return_displ=return_displ)
        if not self.workspace._has_att:
            pivot = values.flatten()[values.abs().flatten().argmax()]
            values = values * (pivot.conj()/pivot.abs())
            values = values.real
        return numpy(values)


class NoReference:
    """Leave reference points absent when the optional CPS module is absent."""

    def __init__(self, *args, **kwargs):
        pass

    def compute_swd(self, wave, mode, periods):
        return np.full_like(periods, np.nan, dtype=float)


def difference(actual, expected):
    valid = np.isfinite(actual) & np.isfinite(expected)
    if not valid.any():
        return dict(count=0)
    error = np.abs(actual[valid]-expected[valid])
    scale = max(float(np.linalg.norm(expected[valid])), 1.e-30)
    return dict(count=int(valid.sum()), max_absolute=float(error.max()),
                relative_l2=float(np.linalg.norm(error)/scale))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent/'pytorch')
    parser.add_argument('--sections', nargs='+', choices=('love', 'scholte', 'aniso'),
                        default=['love', 'scholte', 'aniso'])
    parser.add_argument('--reference', choices=('auto', 'cps', 'none'), default='auto')
    parser.add_argument('--quick', action='store_true', help='Use smaller grids for a preview')
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    reference = False
    if args.reference != 'none' and 'scholte' in args.sections:
        try:
            from specd import THSolver
            paper.THSolver = THSolver
            reference = True
        except ImportError:
            if args.reference == 'cps':
                raise
    if not reference:
        paper.THSolver = NoReference
        paper.REFERENCE_LABEL = ''
    paper.SpecWorkSpace = PlotWorkspace
    paper.OUT = output
    paper.KERNEL_LABEL = 'PyTorch autodiff'
    paper.SOLVER_LABEL = 'PyTorch SEM'
    # Native rows include vp_ac separately. specd.torch already accumulates
    # the fluid derivative into vph; adding an acoustic row would count twice.
    paper.SCHOLTE_P_ROWS = (0, 1)
    if args.quick:
        paper.LOVE_FREQUENCY_COUNT = 12
        paper.SCHOLTE_FREQUENCY_COUNT = 10
        paper.ANISO_FREQUENCY_COUNT = 8
        paper.ANISO_AZIMUTH_COUNT = 13
    started = time.perf_counter()
    manifest = dict(backend='specd.torch', torch_version=torch.__version__,
                    python=sys.executable, omp_num_threads=os.environ['OMP_NUM_THREADS'],
                    torch_num_threads=torch.get_num_threads(),
                    ngll=SpecWorkSpace().ngll, ngrl=SpecWorkSpace().ngrl,
                    halfspace_wavelengths=SpecWorkSpace().halfspace_wavelengths,
                    cps_reference=reference, quick=args.quick, sections=args.sections,
                    statistics=STATISTICS, validation={}, figures=[],
                    validation_notes=[
                        'Love analytic errors include cutoff-sensitive higher modes; these quantify SEM discretization error, not backend agreement.',
                        'CPS zero sentinels for unavailable phase/group modes are excluded from reference errors.'])

    def write_manifest():
        manifest['elapsed_seconds'] = time.perf_counter()-started
        (output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')

    original_save = paper.save_figure

    def save_figure(figure, stem):
        original_save(figure, stem)
        manifest['figures'].append(dict(pdf=stem+'.pdf', jpeg=stem+'.jpg'))
        write_manifest()
        print('Saved', output/(stem+'.pdf'), flush=True)

    def record_results(name, **arrays):
        np.savez_compressed(output/(name+'_data.npz'), **arrays)
        if name == 'love':
            manifest['validation']['love'] = {
                'group_vs_analytical': difference(arrays['group'], arrays['analytical_group']),
                **{key+'_vs_analytical': difference(arrays[key], arrays['analytical_'+key])
                   for key in ('phase_kernels', 'group_kernels')}}
        else:
            manifest['validation']['scholte'] = {
                key+'_vs_cps': difference(arrays[key],
                    np.where(arrays['reference_'+key] > 0, arrays['reference_'+key], np.nan)
                    if key in ('phase', 'group') else arrays['reference_'+key])
                for key in ('phase', 'group', 'kernels')}
        write_manifest()

    paper.save_figure = save_figure
    paper.record_results = record_results
    for section in args.sections:
        print('Generating', section, 'figures with specd.torch', flush=True)
        {'love': paper.love_figures, 'scholte': paper.scholte_figures,
         'aniso': paper.orthorhombic_figures}[section]()
    if 'specd.lib.libswd' in sys.modules:
        raise RuntimeError('the PyTorch example unexpectedly loaded the C++ SEM extension')
    write_manifest()
    print(f"Generated {len(manifest['figures'])} PDF/JPEG figure pairs in {output}", flush=True)


if __name__ == '__main__':
    main()
