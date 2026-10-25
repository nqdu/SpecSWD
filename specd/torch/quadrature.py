"""GLL polynomials and exponentially weighted GRL functions (all torch)."""

from functools import lru_cache

import torch


def _legendre(n, x):
    p, q = torch.ones_like(x), x
    if n == 0:
        return p
    for k in range(1, n):
        p, q = q, ((2 * k + 1) * x * q - k * p) / (k + 1)
    return q


def _laguerre(n, x):
    p, q = torch.ones_like(x), 1 - x
    if n == 0:
        return p
    for k in range(1, n):
        p, q = q, ((2 * k + 1 - x) * q - k * p) / (k + 1)
    return q


@lru_cache(maxsize=32)
def gll(n=7, *, device=None):
    if n < 3:
        raise ValueError("GLL requires at least three nodes")
    x = -torch.cos(torch.linspace(0, torch.pi, n, dtype=torch.float64, device=device))
    for _ in range(40):
        pn, pm = _legendre(n - 1, x), _legendre(n - 2, x)
        step = (x * pn - pm) / (n * pn)
        x = x - step
        if step.abs().max() < 1e-15:
            break
    w = 2 / (n * (n - 1) * _legendre(n - 1, x).square())
    delta = x[:, None] - x[None, :]
    eye = torch.eye(n, dtype=torch.bool, device=device)
    bary = torch.where(eye, 1.0, delta).prod(dim=1).reciprocal()
    d = (bary[None, :] / bary[:, None]) / torch.where(eye, 1.0, delta)
    d = d.masked_fill(eye, 0)
    d = d - torch.diag(d.sum(dim=1))
    return x, w, d


@lru_cache(maxsize=32)
def grl(n=30, *, device=None):
    if n < 2:
        raise ValueError("GRL requires at least two nodes")
    k = torch.arange(n - 1, dtype=torch.float64, device=device)
    off = torch.sqrt((k[:-1] + 1) * (k[:-1] + 2))
    jacobi = torch.diag(2 * k + 2) + torch.diag(off, 1) + torch.diag(off, -1)
    x = torch.cat((k.new_zeros(1), torch.linalg.eigvalsh(jacobi)))
    w = 1 / (n * (_laguerre(n - 1, x) * torch.exp(-x / 2)).square())
    f = _laguerre(n, x) * torch.exp(-x / 2)
    delta = x[:, None] - x[None, :]
    eye = torch.eye(n, dtype=torch.bool, device=device)
    # d[a,b] = derivative of basis b evaluated at node a.
    d = (f[:, None] / f[None, :]) / torch.where(eye, 1.0, delta)
    d = d.masked_fill(eye, 0)
    d[0, 0] = -n / 2
    return x, w, d
