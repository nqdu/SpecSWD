"""Physical-root checks for the paper examples, beyond backend agreement."""

import importlib.util
from pathlib import Path

import numpy as np

from specd.torch import SpecWorkSpace


path = Path(__file__).resolve().parents[3] / 'example/paper/reproduce_figures.py'
spec = importlib.util.spec_from_file_location('specswd_paper_reference', path)
paper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(paper)


def test_love_fundamental_is_resolved_across_the_full_paper_grid():
    model = paper.love_model()
    workspace = SpecWorkSpace().initialize(
        'love', model['z'], model['rho'], vsh=model['vsh'], vsv=model['vsv'],
        Qn=model['Qn'], Ql=model['Ql'])
    for frequency in paper.logarithmic_frequencies(.01, 1., 500):
        phase = workspace.compute_egn(frequency, only_phase=True).detach().numpy()
        assert len(phase)
        assert paper.love_residual_error(phase[0], frequency, model) < 1.e-5, frequency
        assert paper.love_mode_numbers(model, frequency, phase[:1])[0] == 0


def test_rejecting_a_lower_love_mode_does_not_renumber_the_next_branch():
    # Two-layer analytical roots at 0.08 Hz: fundamental and first overtone.
    phase = np.array([3.39521570+.00820904j, 4.66089251+.01730908j])
    model = paper.love_model()
    np.testing.assert_array_equal(paper.love_mode_numbers(model, .08, phase), [0, 1])
    np.testing.assert_array_equal(paper.love_mode_numbers(model, .08, phase[1:]), [1])
