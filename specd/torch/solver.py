"""Torch-native surface-wave dispersion, eigenfunctions and autograd kernels."""

from dataclasses import dataclass

import torch

from .assembly import assemble, interpolate, solve_horizontal
from .materials import PAIRS
from .mesh import Mesh


def _derivative(value, variable, create_graph=True):
    """Complex derivative with respect to a real scalar (not a Wirtinger VJP)."""
    def grad(part):
        if not part.requires_grad:
            return variable * 0
        result, = torch.autograd.grad(part, variable, retain_graph=True,
                                      create_graph=create_graph, allow_unused=True)
        return variable * 0 if result is None else result
    return grad(value.real) + 1j * grad(value.imag)


@dataclass
class WaveModel:
    wave_type: str
    z: torch.Tensor
    parameters: dict
    fluid: torch.Tensor
    reference_frequency: torch.Tensor


class SpecWorkSpace:
    """Independent PyTorch implementation of the SpecSWD spectral solver.

    Model tensors retain their graphs. Non-tensor inputs become float64 leaf
    tensors requiring gradients, accessible through ``parameters``. Geometry,
    adaptive element counts, interpolation locations and mode selection are
    discrete, fixed decisions during differentiation.
    """

    def __init__(self, wavetype=None, *, device=None, ngll=7, ngrl=30,
                 halfspace_wavelengths=50):
        if halfspace_wavelengths <= 0 or not torch.isfinite(torch.tensor(halfspace_wavelengths)):
            raise ValueError("halfspace_wavelengths must be positive and finite")
        self.device = torch.device(device) if device is not None else None
        self.ngll, self.ngrl = ngll, ngrl
        self.halfspace_wavelengths = halfspace_wavelengths
        self.model = None
        self._phase = None
        self._right = self._left = self._group = None

    @property
    def parameters(self):
        if self.model is None:
            raise RuntimeError("initialize a model first")
        return self.model.parameters

    def initialize(self, wavetype, z, rho, *, vph=None, vpv=None, vsh=None,
                   vsv=None, eta=None, Qa=None, Qc=None, Qn=None, Ql=None,
                   c21=None, Qani=None, reference_frequency=1.0):
        wave = wavetype.lower()
        if wave not in ("love", "rayl", "aniso"):
            raise ValueError("wavetype must be love, rayl or aniso")
        device = self.device
        if device is None:
            tensors = [value for value in (rho, vsv, c21, z) if isinstance(value, torch.Tensor)]
            device = tensors[0].device if tensors else torch.device("cpu")
        if isinstance(z, torch.Tensor) and z.requires_grad:
            raise ValueError("depth gradients are unsupported; supply fixed depths")
        depth = torch.as_tensor(z, dtype=torch.float64, device=device)
        if (depth.ndim != 1 or len(depth) < 3 or not torch.isfinite(depth).all()
                or depth[0] != 0 or (depth[1:] < depth[:-1]).any()
                or depth[-1] != depth[-2] or depth[-1] <= 0):
            raise ValueError("depths must start at zero, be nondecreasing and end with a repeated half-space depth")
        cuts = torch.where(depth[1:] == depth[:-1])[0] + 1
        if any(int(b - a) < 2 for a, b in zip([0] + cuts.tolist(), cuts.tolist())):
            raise ValueError("finite regions require at least two distinct depths")
        ref = torch.as_tensor(reference_frequency, dtype=torch.float64, device=device)
        if ref.ndim != 0 or not torch.isfinite(ref) or ref <= 0:
            raise ValueError("reference_frequency must be a positive finite scalar")
        inputs = dict(rho=rho)
        if wave == "love":
            inputs.update(vsh=vsh, vsv=vsv)
            qualities = dict(Qn=Qn, Ql=Ql)
        elif wave == "rayl":
            inputs.update(vph=vph, vpv=vpv, vsv=vsv, eta=eta)
            qualities = dict(Qa=Qa, Qc=Qc, Ql=Ql)
        else:
            inputs.update(c21=c21)
            qualities = dict(Qani=Qani)
        if any(value is not None for value in qualities.values()):
            if any(value is None for value in qualities.values()):
                raise ValueError("supply all quality-factor arrays for the selected wave type")
            inputs.update(qualities)
        parameters = {}
        for key, value in inputs.items():
            if value is None:
                raise ValueError(f"{key} is required for {wave}")
            tensor = torch.as_tensor(value, dtype=torch.float64, device=device)
            shape = (21, len(depth)) if key == "c21" else (2, len(depth)) if key == "Qani" else depth.shape
            if tensor.shape != shape:
                raise ValueError(f"{key} must have shape {tuple(shape)}")
            quality = key.startswith("Q")
            if torch.isnan(tensor).any() or (not quality and not torch.isfinite(tensor).all()):
                raise ValueError(f"{key} contains invalid values")
            if quality and (not torch.isfinite(tensor).all() or (tensor <= 0).any()):
                raise ValueError(f"{key} must be positive and finite; omit quality arrays for elastic models")
            if key == "rho" and (tensor <= 0).any():
                raise ValueError("rho must be positive")
            if key.startswith("v") and ((tensor < 0).any() or (wave == "love" and (tensor == 0).any())):
                raise ValueError("velocities must be nonnegative, and positive for Love waves")
            if key in ("vph", "vpv") and (tensor <= 0).any():
                raise ValueError("P velocities must be positive")
            if not tensor.requires_grad:
                tensor = tensor.clone().requires_grad_()
            parameters[key] = tensor
        if wave == "love":
            fluid = torch.zeros_like(depth, dtype=torch.bool)
        elif wave == "rayl":
            fluid = parameters["vsv"].detach() < 1e-6
            if not torch.equal(parameters["vph"][fluid], parameters["vpv"][fluid]):
                raise ValueError("fluid vph and vpv must agree")
            if "Qa" in parameters and not torch.equal(parameters["Qa"][fluid], parameters["Qc"][fluid]):
                raise ValueError("fluid Qa and Qc must agree")
        else:
            # A fluid has only the isotropic bulk block in its Voigt matrix.
            c = parameters["c21"].detach()
            fluid = (c[[15, 18, 20]].abs() < 1e-6).all(dim=0)
            for index, (i, j) in enumerate(PAIRS):
                expected = c[0] if j < 3 else torch.zeros_like(c[0])
                if not torch.allclose(c[index, fluid], expected[fluid], atol=1e-6, rtol=0):
                    raise ValueError("fluid c21 must contain an isotropic bulk block and zero shear")
            if "Qani" in parameters and not torch.equal(parameters["Qani"][0, fluid], parameters["Qani"][1, fluid]):
                raise ValueError("fluid Qkappa and Qmu must agree")
        bounds = [0] + cuts.tolist() + [len(depth)]
        for a, b in zip(bounds[:-1], bounds[1:]):
            if (fluid[a:b] != fluid[a]).any():
                raise ValueError("solid/fluid transitions require repeated depths")
        self.model = WaveModel(wave, depth, parameters, fluid, ref)
        self._phase = None
        self._right = self._left = self._group = None
        return self

    def _eigensystem(self, freq, angle, mesh, vectors, operators=None):
        m, k, h, e = assemble(self.model, mesh, freq, angle) if operators is None else operators
        a = (2 * torch.pi * freq).square() * m - e
        if self.model.wave_type == "love" and not self._has_att:
            root = k.diag().real.sqrt()
            symmetric = a.real / (root[:, None] * root[None, :])
            if vectors:
                values, right = torch.linalg.eigh(symmetric)
                right = (right / root[:, None]).to(torch.complex128)
                left = right
            else:
                values, right = torch.linalg.eigvalsh(symmetric), None
            wavenumber = values.to(torch.complex128).sqrt()
        elif self.model.wave_type != "aniso":
            operator = solve_horizontal(self.model.wave_type, mesh, k, a)
            if vectors:
                values, right = torch.linalg.eig(operator)
            else:
                values, right = torch.linalg.eigvals(operator), None
            wavenumber = values.sqrt()
        else:
            size = len(a)
            operator = a.new_zeros((2 * size, 2 * size))
            operator[:size, size:] = torch.eye(size, dtype=a.dtype, device=a.device)
            operator[size:] = solve_horizontal("aniso", mesh, k, torch.cat((a, -1j * h), dim=1))
            if vectors:
                wavenumber, right = torch.linalg.eig(operator)
            else:
                wavenumber, right = torch.linalg.eigvals(operator), None
        phase = 2 * torch.pi * freq / wavenumber
        with torch.no_grad():
            factor = 1 if self._has_att else 10
            valid = (torch.isfinite(phase) & (phase.real >= mesh.velocity_min)
                     & (phase.real <= mesh.velocity_max)
                     & (wavenumber.real.abs() >= factor * wavenumber.imag.abs()))
            ids = torch.where(valid)[0]
            ids = ids[torch.argsort(phase[ids].real)]
        phase, wavenumber = phase[ids], wavenumber[ids]
        if not self._has_att:
            phase = phase.real.to(torch.complex128)
        if right is None:
            return phase, wavenumber, None, None
        if self.model.wave_type == "love" and not self._has_att:
            return phase, wavenumber, right[:, ids], left[:, ids]
        # Solve for only the selected dual vectors, instead of inverting the full basis.
        unit = right.new_zeros((len(right), len(ids)))
        unit[ids, torch.arange(len(ids), device=ids.device)] = 1
        dual = torch.linalg.solve(right.mH, unit)
        if self.model.wave_type == "aniso":
            dual, right = dual[len(a):], right[:len(a)]
        left = solve_horizontal(self.model.wave_type, mesh, k, dual, adjoint=True)
        return phase, wavenumber, right[:, ids], left

    @property
    def _has_att(self):
        return any(key.startswith("Q") for key in self.parameters)

    def compute_egn(self, freq, ang_in_deg=0.0, only_phase=False):
        if self.model is None:
            raise RuntimeError("initialize a model first")
        f = torch.as_tensor(freq, dtype=torch.float64, device=self.model.z.device)
        angle = torch.as_tensor(ang_in_deg, dtype=f.dtype, device=f.device) * torch.pi / 180
        if f.ndim != 0 or not torch.isfinite(f) or f <= 0:
            raise ValueError("frequency must be a positive finite scalar")
        if angle.ndim != 0 or not torch.isfinite(angle):
            raise ValueError("angle must be a finite scalar")
        if not only_phase:
            if not f.requires_grad:
                f = f.clone().requires_grad_()
            if not angle.requires_grad:
                angle = angle.clone().requires_grad_()
        mesh = Mesh(self.model, f.detach(), angle.detach(), self.ngll, self.ngrl,
                    self.halfspace_wavelengths)
        self._nodal = interpolate(self.model, mesh)
        self._operators = assemble(self.model, mesh, f, angle, self._nodal)
        self._phase, self._k, self._right, self._left = self._eigensystem(
            f, angle, mesh, not only_phase, self._operators)
        self._freq, self._angle, self.mesh = f, angle, mesh
        self._group = None
        self._only_phase = only_phase
        return self._phase

    def group_velocity(self):
        """All modes at once using operator JVPs and eigenvalue perturbation.

        For P(k,f)r=0, dk/df = -(l^H P_f r)/(l^H P_k r). Forward-mode
        autodiff computes the operator partials without one eigensolver backward
        per mode. Keeping the eigenvectors in the graph supports group kernels.
        """
        if self._phase is None or self._only_phase:
            raise RuntimeError("call compute_egn(..., only_phase=False) first")
        if self._group is not None:
            return self._group
        if not self._k.numel():
            shape = (2, 0) if self.model.wave_type == "aniso" else (0,)
            self._group = self._phase.new_empty(shape)
            return self._group
        m, kmat, h, _ = self._operators
        _, df = torch.func.jvp(lambda f: assemble(self.model, self.mesh, f, self._angle, self._nodal),
                              (self._freq,), (torch.ones_like(self._freq),))
        left, right, k = self._left.conj(), self._right, self._k

        def project(matrix):
            return (left * (matrix @ right)).sum(dim=0)

        anisotropic = self.model.wave_type == "aniso"
        omega2 = (2 * torch.pi * self._freq).square()
        pf = (project(df[3] - omega2 * df[0] - 8 * torch.pi**2 * self._freq * m)
              + k.square() * project(df[1]))
        pk = 2 * k * project(kmat)
        if anisotropic:
            pf = pf + 1j * k * project(df[2])
            pk = pk + 1j * project(h)
        ur = -2 * torch.pi * pk / pf
        if anisotropic:
            _, da = torch.func.jvp(lambda a: assemble(self.model, self.mesh, self._freq, a, self._nodal),
                                  (self._angle,), (torch.ones_like(self._angle),))
            pa = project(da[3] - omega2 * da[0]) + k.square() * project(da[1]) + 1j * k * project(da[2])
            ut = ur * pa / (pk * k)
            co, si = self._angle.cos(), self._angle.sin()
            self._group = torch.stack((ur * co - ut * si, ur * si + ut * co))
        else:
            self._group = ur
        if not self._has_att:
            self._group = self._group.real.to(torch.complex128)
        return self._group

    def kernels(self, imode=0, *, observable="phase", component="real", create_graph=False):
        """Return named autograd derivatives on the original model grid.

        Q derivatives here are d/dQ. ``get_phase_kl`` and ``get_group_kl``
        convert these to d/d(Q^-1), matching the native kernel convention.
        Anisotropic group kernels use the radial group component.
        """
        if self._phase is None:
            raise RuntimeError("compute eigenvalues first")
        if observable == "phase":
            value = self._phase[imode]
        elif observable == "group":
            group = self.group_velocity()
            value = (group[0, imode] * self._angle.cos() + group[1, imode] * self._angle.sin()
                     if self.model.wave_type == "aniso" else group[imode])
        else:
            raise ValueError("observable must be phase or group")
        if component == "real":
            scalar = value.real
        elif component == "imag":
            scalar = value.imag
        elif component == "qinv":
            scalar = 2 * value.imag / value.real
        else:
            raise ValueError("component must be real, imag or qinv")
        values = torch.autograd.grad(scalar, tuple(self.parameters.values()),
                                     retain_graph=True, create_graph=create_graph, allow_unused=True)
        return {key: torch.zeros_like(parameter) if derivative is None else derivative
                for (key, parameter), derivative in zip(self.parameters.items(), values)}

    def get_kernel_names(self):
        if self.model.wave_type == "love":
            names = ["vsh", "vsv", "rho"]
        elif self.model.wave_type == "rayl":
            names = ["vph", "vpv", "vsv", "rho", "eta"]
        else:
            names = [f"c{i + 1}{j + 1}" for i, j in PAIRS] + ["rho"]
        if self._has_att:
            names += {"love": ["Qn", "Ql"], "rayl": ["Qa", "Qc", "Ql"],
                      "aniso": ["Qk", "Qm"]}[self.model.wave_type]
        return names

    def _kernel_array(self, imode, observable, component):
        derivatives = self.kernels(imode, observable=observable, component=component)
        if self.model.wave_type == "aniso":
            real = torch.cat((derivatives["c21"], derivatives["rho"][None, :]))
            return (torch.cat((real, -self.parameters["Qani"].square() * derivatives["Qani"]))
                    if self._has_att else real)
        return torch.stack(tuple(-self.parameters[name].square() * derivatives[name]
                                 if name.startswith("Q") else derivatives[name]
                                 for name in self.get_kernel_names()))

    def get_phase_kl(self, imode):
        real = self._kernel_array(imode, "phase", "real")
        return (real, self._kernel_array(imode, "phase", "qinv")) if self._has_att else real

    def get_group_kl(self, imode):
        real = self._kernel_array(imode, "group", "real")
        return (real, self._kernel_array(imode, "group", "qinv")) if self._has_att else real

    def get_egnfunc(self, imode, return_displ=True, return_left=False):
        """Return displacement at SEM nodes or the raw right eigenvector.

        Anisotropic displacements use radial/transverse/vertical components;
        Rayleigh uses the native real polarization amplitude convention.
        """
        if self._right is None:
            raise RuntimeError("call compute_egn(..., only_phase=False) first")
        if return_left and return_displ:
            raise ValueError("displacement is defined for right eigenvectors; use return_displ=False for left vectors")
        u = (self._left if return_left else self._right)[:, imode]
        u = u / torch.linalg.vector_norm(u)
        if not return_displ:
            return u
        wave, ne, k = self.model.wave_type, self.mesh.nsolid, self._k[imode]
        nc = {"love": 1, "rayl": 2, "aniso": 3}[wave]
        mesh = self.mesh
        idx = mesh.indices
        displ = (u[:nc * ne].reshape(nc, ne)[:, idx.clamp(0, ne - 1)] if ne else
                 u.new_zeros((nc, len(idx))))
        if wave == "rayl":
            displ = torch.stack((displ[0], displ[1] / k))
        if mesh.nfluid:
            chi = torch.where(idx >= 0, u[nc * ne:][idx.clamp(0, mesh.nfluid - 1)], 0)
            if wave == "rayl":
                chi = chi / k
            dz = torch.zeros_like(chi)
            for solid, nodes, jac, derivative, _, _ in mesh.groups:
                if solid or not nodes.numel():
                    continue
                gradient = (chi[nodes] @ derivative.T.to(chi.dtype)) / jac[:, None]
                dz = dz.index_copy(0, nodes.flatten(), gradient.flatten())
            r = mesh.interpolation @ self.parameters["rho"]
            fluid_displ = (torch.stack((-k * chi / r, dz / r)) if wave == "rayl" else
                           torch.stack((-1j * k * self._angle.cos() * chi / r,
                                        -1j * k * self._angle.sin() * chi / r, dz / r)))
            displ = torch.where(mesh.solid[mesh.element][None, :], displ, fluid_displ)
        if wave == "aniso":
            co, si = self._angle.cos(), self._angle.sin()
            displ = torch.stack((co * displ[0] + si * displ[1],
                                 -si * displ[0] + co * displ[1], displ[2]))
        return displ

    def compute_dispersion(self, frequencies_hz, azimuths_deg=0.0, include_group=True):
        from .io import DispersionTable

        if self.model is None:
            raise RuntimeError("initialize a model first")
        frequencies = torch.as_tensor(frequencies_hz, dtype=torch.float64, device=self.model.z.device)
        angles = torch.atleast_1d(torch.as_tensor(azimuths_deg, dtype=frequencies.dtype, device=frequencies.device))
        if frequencies.ndim != 1 or not len(frequencies) or angles.ndim != 1 or not len(angles):
            raise ValueError("frequencies and azimuths must be nonempty 1-D arrays")
        count = len(frequencies) * len(angles)
        phase, group = [None] * count, [None] * count if include_group else None
        for i, f in enumerate(frequencies):
            for j, angle in enumerate(angles):
                row = i * len(angles) + j
                phase[row] = self.compute_egn(f, angle, only_phase=not include_group)
                if include_group:
                    g = self.group_velocity()
                    group[row] = g.T if self.model.wave_type == "aniso" else g[:, None]
        return DispersionTable(frequencies, angles, phase, group if include_group else None,
                               ("x", "y") if self.model.wave_type == "aniso" else ("radial",))
