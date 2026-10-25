"""Batched tensor element operators and differentiable indexed assembly."""

import torch

from .materials import attenuate_c21, c21_to_tensor, _constant_q_response


def _blocks(value):
    """(element, component, component, node, node) -> element matrices."""
    b, c, _, n, _ = value.shape
    return value.permute(0, 1, 3, 2, 4).reshape(b, c * n, c * n)


def _solid_operators(wave, p, w, d, j, freq, angle, reference):
    """Pure tensor kernel; all elements of one quadrature order at once."""
    r = p["rho"]
    wj = w * j[:, None]

    def stiff(value):
        return d.T @ ((w * value / j[:, None])[..., None] * d)

    def modulus(name, qname):
        value = r * p[name].square()
        return value * _constant_q_response(freq, p[qname], reference) if qname in p else value

    mass = torch.diag_embed(wj * r)
    if wave == "love":
        k = torch.diag_embed(wj * modulus("vsh", "Qn"))
        e = stiff(modulus("vsv", "Ql"))
        return torch.stack((mass, k, torch.zeros_like(k), e))
    nc = 2 if wave == "rayl" else 3
    eye = torch.eye(nc, dtype=mass.dtype, device=mass.device)
    m = _blocks(mass[:, None, None] * eye[None, :, :, None, None])
    if wave == "rayl":
        a, c, l = modulus("vph", "Qa"), modulus("vpv", "Qc"), modulus("vsv", "Ql")
        f = p["eta"] * (a - 2 * l)
        k21 = d.T * (w * f)[:, None, :] - (w * l)[:, :, None] * d
        e12 = (w * f)[:, :, None] * d - d.T * (w * l)[:, None, :]
        zero = torch.zeros_like(k21)
        b, n = r.shape
        k = _blocks(torch.stack((torch.diag_embed(wj * a), zero, k21,
                                torch.diag_embed(wj * l)), dim=1).reshape(b, 2, 2, n, n))
        e = _blocks(torch.stack((stiff(l), e12, zero, stiff(c)), dim=1).reshape(b, 2, 2, n, n))
        return torch.stack((m, k, torch.zeros_like(k), e))

    c = p["c21"]
    if "Qani" in p:
        c = attenuate_c21(c, p["Qani"], freq, reference)
    c = c21_to_tensor(c)
    direction = torch.stack((angle.cos(), angle.sin())).to(c.dtype)
    horizontal = torch.einsum("baijpq,j,q->bipa", c[:, :, :, :2, :, :2], direction, direction)
    mixed = torch.einsum("baijp,j->bipa", c[:, :, :, :2, :, 2], direction)
    mixed_t = torch.einsum("baipq,q->bipa", c[:, :, :, 2, :, :2], direction)
    vertical = c[:, :, :, 2, :, 2].permute(0, 2, 3, 1)
    k = _blocks(torch.diag_embed(horizontal * wj[:, None, None, :]))
    h = _blocks((mixed * w[:, None, None, :])[..., None] * d
                - d.T * (mixed_t * w[:, None, None, :])[..., None, :])
    e = _blocks(d.T @ ((vertical * (w / j[:, None])[:, None, None, :])[..., None] * d))
    return torch.stack((m, k, h, e))


def interpolate(model, mesh):
    """Interpolate model fields once; these tensors retain parameter gradients."""
    return {key: mesh.interpolation @ value if value.ndim == 1 else
            mesh.interpolation @ value.T for key, value in model.parameters.items()}


def assemble(model, mesh, freq, angle, nodal=None):
    nc = {"love": 1, "rayl": 2, "aniso": 3}[model.wave_type]
    ne, nf = mesh.nsolid, mesh.nfluid
    ndof = nc * ne + nf
    p = interpolate(model, mesh) if nodal is None else nodal
    operators = torch.zeros((4, ndof * ndof), dtype=torch.complex128, device=freq.device)
    for solid, nodes, jac, derivative, indices, valid in mesh.groups:
        if not nodes.numel():
            continue
        local = {key: value[nodes] for key, value in p.items()}
        w, d = mesh.weights[nodes], derivative.to(torch.complex128)
        if solid:
            matrices = _solid_operators(model.wave_type, local, w, d, jac, freq, angle,
                                        model.reference_frequency)
        else:
            r = local["rho"]
            if model.wave_type == "aniso":
                bulk = local["c21"][..., 0]
                if "Qani" in local:
                    bulk = bulk * _constant_q_response(freq, local["Qani"][..., 0], model.reference_frequency)
            else:
                bulk = r * local["vph"].square()
                if "Qc" in local:
                    bulk = bulk * _constant_q_response(freq, local["Qc"], model.reference_frequency)
            mass = torch.diag_embed(w * jac[:, None] / bulk)
            k = torch.diag_embed(w * jac[:, None] / r)
            e = d.T @ ((w / r / jac[:, None])[..., None] * d)
            matrices = torch.stack((mass, k, torch.zeros_like(k), e))
        values = matrices.flatten(1).to(torch.complex128) * valid[None, :]
        operators = operators.index_add(1, indices, values)
    si = mesh.interface_solid + (nc - 1) * ne
    fi = mesh.interface_fluid + nc * ne
    coupling_indices = torch.cat((si * ndof + fi, fi * ndof + si))
    coupling = torch.cat(((2 * torch.pi * freq).square() * mesh.interface_normal,
                          mesh.interface_normal)).to(operators.dtype)
    e = operators[3].index_add(0, coupling_indices, coupling)
    return (operators[0].reshape(ndof, ndof), operators[1].reshape(ndof, ndof),
            operators[2].reshape(ndof, ndof), e.reshape(ndof, ndof))


def solve_horizontal(wave, mesh, k, rhs, *, adjoint=False):
    """Apply K^-1 using its diagonal/triangular/node-local block structure."""
    if adjoint:
        k = k.mH
    if wave == "love":
        return rhs / k.diag()[:, None]
    if wave == "rayl":
        return torch.linalg.solve_triangular(k, rhs, upper=adjoint)
    ne = mesh.nsolid
    ids = torch.arange(ne, device=k.device)[:, None] + ne * torch.arange(3, device=k.device)
    blocks = k[ids[:, :, None], ids[:, None, :]]
    solid = torch.linalg.solve(blocks, rhs[ids]).transpose(0, 1).flatten(0, 1)
    fluid = rhs[3 * ne:] / k.diag()[3 * ne:, None]
    return torch.cat((solid, fluid), dim=0)
