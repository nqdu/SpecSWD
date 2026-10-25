"""Frequency-adaptive mesh stored as arrays, fixed during differentiation."""

import torch

from .materials import c21_to_tensor
from .quadrature import gll, grl


class Mesh:
    """One tensor container per solve; no per-element Python objects.

    Nodes and interpolation rows are contiguous in element order. Offsets and
    counts describe their ragged layout. Assembly batches equal-order elements.
    """

    @torch.no_grad()
    def __init__(self, model, freq, angle, ngll=7, ngrl=30, halfspace_wavelengths=50):
        z, device = model.z, model.z.device
        cuts = torch.where(z[1:] == z[:-1])[0] + 1
        boundaries = torch.cat((cuts.new_zeros(1), cuts, cuts.new_full((1,), len(z))))
        starts, ends = boundaries[:-1], boundaries[1:]
        lengths = ends - starts
        regions = torch.repeat_interleave(torch.arange(len(starts), device=device), lengths)
        fluid = model.fluid[starts]
        if model.wave_type == "love":
            velocities = model.parameters["vsh"].detach()[:, None]
        elif model.wave_type == "rayl":
            velocities = torch.where(model.fluid, model.parameters["vpv"],
                                     model.parameters["vsv"]).detach()[:, None]
        else:
            c = c21_to_tensor(model.parameters["c21"].detach().T)
            direction = torch.stack((angle.cos(), angle.sin(), angle.new_zeros(())))
            gamma = torch.einsum("nijpq,j,q->nip", c, direction, direction)
            velocities = torch.linalg.eigvalsh(gamma / model.parameters["rho"][:, None, None])
            velocities = velocities.clamp_min(0).sqrt()
        node_min = torch.where(model.fluid, velocities[:, -1], velocities.amin(dim=1))
        vmin = z.new_full(starts.shape, torch.inf).scatter_reduce(0, regions, node_min, reduce="amin")
        vmax = z.new_zeros(starts.shape).scatter_reduce(0, regions, velocities.amax(dim=1), reduce="amax")
        self.velocity_min, self.velocity_max = 0.85 * vmin.amin(), vmax.amax()
        thickness = z[ends - 1] - z[starts]
        nfinite = (1.5 * thickness * freq / vmin).to(torch.long) + 1
        counts = torch.where(lengths == 1, 1, nfinite.clamp_min(1))
        region = torch.repeat_interleave(torch.arange(len(starts), device=device), counts)
        number = torch.arange(len(region), device=device)
        local_element = number - torch.repeat_interleave(counts.cumsum(0) - counts, counts)
        self.solid = ~fluid[region]
        self.halfspace = lengths[region] == 1
        self.counts = torch.where(self.halfspace, ngrl, ngll)
        self.offsets = self.counts.cumsum(0) - self.counts
        self.element = torch.repeat_interleave(number, self.counts)
        self.local_node = torch.arange(int(self.counts.sum()), device=device) - self.offsets[self.element]
        xg, wg, self.dg = gll(ngll, device=device)
        xr, wr, self.dr = grl(ngrl, device=device)
        self.ngll, self.ngrl = ngll, ngrl
        h = thickness[region] / counts[region]
        self.jacobian = torch.where(self.halfspace,
            vmax[region] / freq / xr[-1] * halfspace_wavelengths, h / 2)
        qindex = self.local_node + self.halfspace[self.element] * ngll
        x = torch.cat((xg, xr))[qindex]
        self.weights = torch.cat((wg, wr))[qindex]
        finite_offset = h * local_element
        self.depth = (z[starts[region]][self.element]
                      + torch.where(self.halfspace[self.element], 0, finite_offset[self.element])
                      + (x + (~self.halfspace[self.element])) * self.jacobian[self.element])

        # Region bounds keep interpolation on the correct side of discontinuities.
        a, b = starts[region][self.element], (ends[region] - 1)[self.element]
        lower = torch.searchsorted(z, self.depth, right=True) - 1
        lower = torch.maximum(a, torch.minimum(lower, b - 1))
        upper = torch.minimum(lower + 1, b)
        fraction = ((self.depth - z[lower]) / torch.where(upper == lower, 1, z[upper] - z[lower])).clamp(0, 1)
        self.interpolation = z.new_zeros((len(self.depth), len(z))).scatter_add(
            1, torch.stack((lower, upper), dim=1), torch.stack((1 - fraction, fraction), dim=1))

        previous_same = torch.cat((self.solid.new_zeros(1), self.solid[1:] == self.solid[:-1]))
        increments = self.counts - previous_same.to(self.counts.dtype)
        increments[0] -= (~self.solid[0]).to(increments.dtype)
        solid_end = (increments * self.solid).cumsum(0)
        fluid_end = (increments * ~self.solid).cumsum(0)
        self.nsolid, self.nfluid = int(solid_end[-1]), int(fluid_end[-1])
        self.ncomponent = {"love": 1, "rayl": 2, "aniso": 3}[model.wave_type]
        start = torch.where(self.solid, solid_end, fluid_end) - self.counts
        self.indices = start[self.element] + self.local_node

        interface = torch.where(self.solid[1:] != self.solid[:-1])[0]
        top_solid = self.solid[interface]
        top_node = self.offsets[interface] + self.counts[interface] - 1
        bottom_node = self.offsets[interface + 1]
        self.interface_solid = self.indices[torch.where(top_solid, top_node, bottom_node)]
        self.interface_fluid = self.indices[torch.where(top_solid, bottom_node, top_node)]
        self.interface_normal = 2 * top_solid.to(z.dtype) - 1

        # Four fixed groups, independent of the number of elements.
        self.groups = tuple(self._group(half, solid) for half, solid in
                            ((False, True), (True, True), (False, False), (True, False)))

    def _group(self, half, solid):
        elements = torch.where((self.halfspace == half) & (self.solid == solid))[0]
        n = self.ngrl if half else self.ngll
        nodes = self.offsets[elements, None] + torch.arange(n, device=self.depth.device)
        nc, ne = self.ncomponent, self.nsolid
        ndof = nc * ne + self.nfluid
        ids = self.indices[nodes]
        if solid:
            ids = (ids[:, None, :] + torch.arange(nc, device=ids.device)[None, :, None] * ne).flatten(1)
        else:
            ids = torch.where(ids >= 0, ids + nc * ne, -1)
        valid = ((ids[:, :, None] >= 0) & (ids[:, None, :] >= 0)).flatten()
        safe = ids.clamp_min(0)
        scatter = (safe[:, :, None] * ndof + safe[:, None, :]).flatten()
        return solid, nodes, self.jacobian[elements], self.dr if half else self.dg, scatter, valid
