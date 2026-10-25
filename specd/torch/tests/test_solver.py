"""Numerical forward and derivative checks, with optional native comparisons."""

import importlib.util
import os
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

from specd.torch import DispersionTable, SpecWorkSpace, constant_q_response, isotropic_c21
from specd.torch.assembly import assemble, solve_horizontal
from specd.torch.quadrature import gll, grl


def model(wave="love", attenuation=False, fluid=False):
    if fluid:
        z = [0., 2., 2., 35., 35.]
        rho, vp, vs = [1., 1., 2.8, 2.8, 3.2], [1.5, 1.5, 6., 6., 8.], [0., 0., 3.3, 3.3, 5.5]
    else:
        z = [0., 35., 35.]
        rho, vp, vs = [2.8, 2.8, 3.2], [6., 6., 8.], [3.3, 3.3, 5.5]
    rho, vp, vs = [torch.tensor(x, dtype=torch.float64) for x in (rho, vp, vs)]
    if wave == "love":
        kwargs = dict(vsh=vs, vsv=vs * .95)
        if attenuation:
            kwargs.update(Qn=torch.full_like(vs, 220), Ql=torch.full_like(vs, 180))
    elif wave == "rayl":
        kwargs = dict(vph=vp, vpv=vp, vsv=vs, eta=torch.ones_like(vs))
        if attenuation:
            kwargs.update(Qa=torch.full_like(vs, 300), Qc=torch.full_like(vs, 300), Ql=torch.full_like(vs, 180))
    else:
        c = isotropic_c21(rho, vp, vs)
        # Break the horizontal symmetry and couple transverse polarization.
        c[0] = c[0] * 1.02
        c[5] = c[5] + torch.where(vs > 0, .1, 0)
        if fluid:
            c[0, :2] /= 1.02
        kwargs = dict(c21=c)
        if attenuation:
            q = torch.stack((torch.full_like(vs, 300), torch.full_like(vs, 180)))
            if fluid:
                q[:, :2] = 300
            kwargs["Qani"] = q
    return z, rho, kwargs


def workspace(wave="love", attenuation=False, fluid=False):
    z, rho, kwargs = model(wave, attenuation, fluid)
    return SpecWorkSpace().initialize(wave, z, rho, **kwargs)


def test_quadrature():
    x, w, d = gll()
    torch.testing.assert_close(w.sum(), torch.tensor(2., dtype=x.dtype))
    torch.testing.assert_close(d @ x.pow(5), 5 * x.pow(4), atol=1e-12, rtol=1e-12)
    x, w, d = grl()
    f = torch.exp(-x / 2)
    torch.testing.assert_close((w * f.square()).sum(), torch.tensor(1., dtype=x.dtype), atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(d @ f, -.5 * f, atol=1e-12, rtol=1e-10)


def test_constant_q_gradcheck():
    f = torch.tensor(.1, dtype=torch.float64, requires_grad=True)
    q = torch.tensor(180., dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(constant_q_response, (f, q))
    value = constant_q_response(1., q)
    torch.testing.assert_close(value.real, torch.ones_like(q))
    torch.testing.assert_close(value.imag, 1 / q)


@pytest.mark.parametrize("wave,fluid", [("love", False), ("rayl", False), ("rayl", True), ("aniso", False), ("aniso", True)])
@pytest.mark.parametrize("attenuation", [False, True])
def test_phase_and_group_gradients(wave, fluid, attenuation):
    ws = workspace(wave, attenuation, fluid)
    phase = ws.compute_egn(.1, 25.)
    assert len(phase) > 0
    assert torch.isfinite(phase).all()
    group = ws.group_velocity()
    assert torch.isfinite(group).all()
    for observable in ("phase", "group"):
        gradients = ws.kernels(0, observable=observable)
        assert all(torch.isfinite(g).all() for g in gradients.values())
        key = "c21" if wave == "aniso" else "vsh" if wave == "love" else "vsv"
        parameter = ws.parameters[key]
        index = (18, 2 if fluid else 0) if wave == "aniso" else (2 if fluid else 0,)
        expected = gradients[key][index].detach()
        eps = 1e-4
        original = parameter.detach().clone()
        results = []
        old_mesh = ws.mesh
        for sign in (-1, 1):
            perturbed = original.clone()
            perturbed[index] += sign * eps
            ws.parameters[key] = perturbed.requires_grad_()
            # Freeze geometry to compare the derivative of the same discrete operator.
            f = torch.tensor(.1, dtype=torch.float64, requires_grad=True)
            angle = torch.tensor(25 * torch.pi / 180, dtype=torch.float64, requires_grad=True)
            c, k, _, _ = ws._eigensystem(f, angle, old_mesh, True)
            if observable == "phase":
                value = c[0].real
            else:
                from specd.torch.solver import _derivative
                value = (2 * torch.pi / _derivative(k[0], f)).real
            results.append(value.detach())
        ws.parameters[key] = parameter
        fd = (results[1] - results[0]) / (2 * eps)
        torch.testing.assert_close(expected, fd, atol=3e-6, rtol=4e-3)
    displ = ws.get_egnfunc(0)
    assert displ.shape[1] == len(ws.mesh.depth)
    assert torch.isfinite(displ).all()


@pytest.mark.parametrize("wave,fluid", [("love", False), ("rayl", False), ("rayl", True), ("aniso", False), ("aniso", True)])
@pytest.mark.parametrize("attenuation", [False, True])
def test_matches_native_solver(wave, fluid, attenuation):
    extension_path = os.environ.get("SPECSWD_REFERENCE_EXTENSION")
    if extension_path:
        from specd.torch.benchmark import load_native
        load_native(extension_path)
    extension = pytest.importorskip("specd.lib.libswd", reason="optional compiled reference solver")
    import specd as native
    if "reference_frequency" not in extension.init_mesh.__doc__:
        pytest.skip("compiled reference predates the current API; rebuild libswd")
    z, rho, kwargs = model(wave, attenuation, fluid)
    ws = SpecWorkSpace().initialize(wave, z, rho, **kwargs)
    reference = native.SpecWorkSpace()
    reference.initialize(wave, np.array(z), rho.numpy(), **{k: v.numpy() for k, v in kwargs.items()})
    expected = reference.compute_egn(.1, 25., only_phase=False)
    actual = ws.compute_egn(.1, 25.)
    np.testing.assert_allclose(actual.detach().numpy(), expected, atol=2e-6, rtol=2e-6)
    np.testing.assert_allclose(ws.group_velocity().detach().numpy(), reference.group_velocity(), atol=2e-5, rtol=2e-5)
    # The phase-only driver must return the same spectrum as the QZ path.
    phase_only = reference.compute_egn(.1, 25., only_phase=True)
    np.testing.assert_allclose(phase_only, expected, atol=2e-6, rtol=2e-6)


def test_external_tensors_receive_gradients():
    z, rho, kwargs = model()
    vsh = kwargs["vsh"].requires_grad_()
    ws = SpecWorkSpace().initialize("love", z, rho, **kwargs)
    ws.compute_egn(.1)[0].real.backward()
    assert vsh.grad is not None and vsh.grad.abs().sum() > 0


def test_phase_autograd_gradcheck():
    ws = workspace(attenuation=True)
    ws.compute_egn(.1)
    mesh = ws.mesh
    vsh, qn = ws.parameters["vsh"], ws.parameters["Qn"]
    f, angle = ws._freq, ws._angle

    def solve(v, q):
        ws.parameters["vsh"], ws.parameters["Qn"] = v, q
        return ws._eigensystem(f, angle, mesh, False)[0][0]

    assert torch.autograd.gradcheck(solve, (vsh, qn), eps=1e-5, atol=1e-6, rtol=1e-3)
    ws.parameters["vsh"], ws.parameters["Qn"] = vsh, qn


def test_quality_and_wave_qinv_kernels():
    ws = workspace(attenuation=True)
    ws.compute_egn(.1)
    original = ws.parameters["Qn"]
    for observable in ("phase", "group"):
        derivative = ws.kernels(0, observable=observable, component="qinv")["Qn"][0]
        results = []
        for sign in (-1, 1):
            perturbed = original.detach().clone()
            perturbed[0] += sign * .01
            ws.parameters["Qn"] = perturbed.requires_grad_()
            f = ws._freq.detach().clone().requires_grad_()
            c, k, _, _ = ws._eigensystem(f, ws._angle, ws.mesh, True)
            if observable == "phase":
                value = c[0]
            else:
                from specd.torch.solver import _derivative
                value = 2 * torch.pi / _derivative(k[0], f)
            results.append((2 * value.imag / value.real).detach())
        ws.parameters["Qn"] = original
        torch.testing.assert_close(derivative, (results[1] - results[0]) / .02, atol=1e-8, rtol=1e-3)


def test_hdf5_roundtrip_and_numpy_compatibility(tmp_path):
    ws = workspace()
    table = ws.compute_dispersion([.08, .1], include_group=True)
    assert table.phase_rows[0].grad_fn is not None
    path = tmp_path / "dispersion.h5"
    table.to_hdf5(path)
    restored = DispersionTable.from_hdf5(path)
    for actual, expected in zip(restored.phase_rows, table.phase_rows):
        torch.testing.assert_close(actual, expected)
    location = Path(__file__).resolve().parents[2] / "io.py"
    spec = importlib.util.spec_from_file_location("specswd_numpy_io", location)
    numpy_io = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = numpy_io
    spec.loader.exec_module(numpy_io)
    numpy_table = numpy_io.DispersionTable.from_hdf5(path)
    np.testing.assert_allclose(numpy_table.phase_row(0)[1], table.phase_rows[0].detach().numpy())


def test_left_and_right_eigenvector_residuals():
    for wave in ("love", "rayl", "aniso"):
        ws = workspace(wave, attenuation=True)
        ws.compute_egn(.1, 25.)
        m, k, h, e = assemble(ws.model, ws.mesh, ws._freq, ws._angle)
        value = ws._k[0]
        polynomial = value.square() * k + e - (2 * torch.pi * ws._freq).square() * m
        if wave == "aniso":
            polynomial = polynomial + 1j * value * h
        right = ws.get_egnfunc(0, return_displ=False)
        left = ws.get_egnfunc(0, return_displ=False, return_left=True)
        torch.testing.assert_close(polynomial @ right, torch.zeros_like(right), atol=1e-7, rtol=0)
        torch.testing.assert_close(left.conj() @ polynomial, torch.zeros_like(left), atol=1e-7, rtol=0)


@pytest.mark.parametrize("wave,fluid", [("love", False), ("rayl", True), ("aniso", True)])
@pytest.mark.parametrize("attenuation", [False, True])
def test_structured_horizontal_solve(wave, fluid, attenuation):
    ws = workspace(wave, attenuation, fluid)
    ws.compute_egn(.1, 25.)
    _, k, _, _ = assemble(ws.model, ws.mesh, ws._freq, ws._angle)
    generator = torch.Generator().manual_seed(42)
    rhs = torch.randn((len(k), 5), dtype=k.dtype, generator=generator)
    for adjoint in (False, True):
        actual = solve_horizontal(wave, ws.mesh, k, rhs, adjoint=adjoint)
        expected = torch.linalg.solve(k.mH if adjoint else k, rhs)
        torch.testing.assert_close(actual, expected, rtol=1e-11, atol=1e-11)


@pytest.mark.parametrize("attenuation", [False, True])
def test_operator_jvp_groups_match_eigensolver_derivatives(attenuation):
    from specd.torch.solver import _derivative

    ws = workspace("aniso", attenuation, fluid=True)
    ws.compute_egn(.1, 25.)
    actual = ws.group_velocity()
    # Include the last retained mode, not just the fundamental mode.
    for mode in (0, len(ws._k) - 1):
        k = ws._k[mode]
        radial = 2 * torch.pi / _derivative(k, ws._freq)
        transverse = -radial * _derivative(k, ws._angle) / k
        co, si = ws._angle.cos(), ws._angle.sin()
        expected = torch.stack((radial * co - transverse * si, radial * si + transverse * co))
        if not attenuation:
            expected = expected.real.to(actual.dtype)
        torch.testing.assert_close(actual[:, mode], expected, rtol=1e-8, atol=1e-8)


@pytest.mark.parametrize("ngll,ngrl", [(4, 8), (7, 20)])
def test_tensor_mesh_discontinuous_regions_and_multiple_elements(ngll, ngrl):
    from specd.torch.mesh import Mesh

    z = torch.tensor([0., 1., 2., 2., 10., 35., 35., 40., 40.], dtype=torch.float64)
    fluid = torch.tensor([True, True, True, False, False, False, True, True, False])
    rho = torch.where(fluid, 1., 2.8).to(z.dtype)
    vs = torch.where(fluid, 0., 3.3).to(z.dtype)
    vp = torch.where(fluid, 1.5, 6.).to(z.dtype)
    ws = SpecWorkSpace(ngll=ngll, ngrl=ngrl).initialize(
        "rayl", z, rho, vph=vp, vpv=vp, vsv=vs, eta=torch.ones_like(z))
    mesh = Mesh(ws.model, z.new_tensor(.4), z.new_zeros(()), ngll, ngrl)
    assert len(mesh.counts) > 4
    torch.testing.assert_close(mesh.interpolation.sum(dim=1), torch.ones_like(mesh.depth))
    finite = ~mesh.halfspace[mesh.element]
    torch.testing.assert_close((mesh.interpolation @ z)[finite], mesh.depth[finite])
    assert mesh.indices[0] == -1
    assert mesh.interface_normal.tolist() == [-1., 1., -1.]
    phase = ws.compute_egn(.4)
    assert phase.numel() and torch.isfinite(ws.group_velocity()).all()
    assert torch.isfinite(ws.get_egnfunc(0)).all()
    assert all(torch.isfinite(value).all() for value in ws.kernels(0, observable="group").values())


@pytest.mark.parametrize("z", [[0, 35, 40], [0, 35, 34], [0, 0, 35, 35]])
def test_invalid_depths(z):
    with pytest.raises(ValueError):
        SpecWorkSpace().initialize("love", z, [2.8] * len(z), vsh=[3.3] * len(z), vsv=[3.] * len(z))
