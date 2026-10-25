"""Differentiable constitutive laws in SpecSWD's Voigt convention."""

from functools import lru_cache

import torch


PAIRS = tuple((i, j) for i in range(6) for j in range(i, 6))
VOIGT = ((0, 5, 4), (5, 1, 3), (4, 3, 2))


@lru_cache(maxsize=16)
def _indices(device):
    """Immutable device-local constitutive maps, shared by all solves."""
    pairs = torch.tensor(PAIRS, device=device)
    voigt = torch.tensor(VOIGT, device=device)
    lookup = torch.empty((6, 6), dtype=torch.long, device=device)
    lookup[pairs[:, 0], pairs[:, 1]] = torch.arange(21, device=device)
    lookup[pairs[:, 1], pairs[:, 0]] = torch.arange(21, device=device)
    tensor = lookup[voigt[:, :, None, None], voigt[None, None, :, :]]
    normal = (pairs[:, 0] == pairs[:, 1]) & (pairs[:, 0] < 3)
    cross = (pairs[:, 0] < pairs[:, 1]) & (pairs[:, 1] < 3)
    shear = (pairs[:, 0] == pairs[:, 1]) & (pairs[:, 0] >= 3)
    return tensor.flatten(), normal, cross, shear


def constant_q_response(freq, quality_factor, reference_frequency=1.0):
    q = torch.as_tensor(quality_factor)
    if not q.is_floating_point():
        q = q.to(torch.float64)
    f = torch.as_tensor(freq, dtype=q.dtype, device=q.device)
    ref = torch.as_tensor(reference_frequency, dtype=q.dtype, device=q.device)
    if (not torch.isfinite(f).all() or (f <= 0).any()
            or not torch.isfinite(ref).all() or (ref <= 0).any()):
        raise ValueError("frequencies must be positive and finite")
    if torch.isnan(q).any() or (q <= 0).any():
        raise ValueError("quality factors must be positive (infinity is elastic)")
    return _constant_q_response(f, q, ref)


def _constant_q_response(f, q, ref):
    """Tensor kernel for already validated solver inputs."""
    qi = q.reciprocal()
    return (1 + 1j * qi) * torch.exp(2 / torch.pi * torch.atan(qi) * torch.log(f / ref))


def c21_to_tensor(c21):
    """Convert (...,21) upper-triangular Voigt values to (...,3,3,3,3)."""
    index = _indices(c21.device)[0]
    return c21[..., index].reshape(*c21.shape[:-1], 3, 3, 3, 3)


def attenuate_c21(c, q, freq, reference_frequency):
    """Carcione Qkappa/Qmu map used by the native anisotropic solver."""
    sk = _constant_q_response(freq, q[..., 0], reference_frequency)
    sm = _constant_q_response(freq, q[..., 1], reference_frequency)
    eps = (c[..., 0] + c[..., 6] + c[..., 11]) / 3
    mu = (c[..., 15] + c[..., 18] + c[..., 20]) / 3
    bulk = eps - 4 / 3 * mu
    _, normal, cross, shear = _indices(c.device)
    normal_shift = -eps + bulk * sk + 4 / 3 * mu * sm
    cross_shift = -eps + bulk * sk + 2 * mu * (1 - sm / 3)
    return (c * (1 + (sm[..., None] - 1) * shear)
            + normal_shift[..., None] * normal + cross_shift[..., None] * cross)


def isotropic_c21(rho, vp, vs):
    """Build (21,nz) stiffnesses without breaking the input autograd graph."""
    mu = rho * vs.square()
    lam = rho * vp.square() - 2 * mu
    _, normal, cross, shear = _indices(rho.device)
    return ((lam + 2 * mu)[None, :] * normal[:, None]
            + lam[None, :] * cross[:, None] + mu[None, :] * shear[:, None])
