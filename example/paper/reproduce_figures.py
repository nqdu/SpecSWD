#!/usr/bin/env python3
"""Reproduce every figure used by main.tex from the current SpecSWD solver."""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from specd import SpecWorkSpace, THSolver


OUT = Path(__file__).resolve().parent
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "legend.fontsize": 8,
    "lines.linewidth": 1.25,
    "savefig.bbox": "tight",
    "figure.dpi": 120,
})


def save_figure(fig: plt.Figure, stem: str) -> None:
    """Save a vector master and a high-resolution raster copy."""
    save_options = {
        "bbox_inches": "tight",
        "pad_inches": 0,
        "facecolor": "white",
        "transparent": False,
    }
    fig.savefig(OUT / f"{stem}.pdf", **save_options)
    fig.savefig(OUT / f"{stem}.jpg", dpi=300, **save_options)
    plt.close(fig)


def propagation_inverse_q(value: np.ndarray | complex) -> np.ndarray:
    value = np.asarray(value)
    return 2.0 * value.imag / value.real


def logarithmic_frequencies(first: float, last: float,
                            count: int) -> np.ndarray:
    """Match the logarithmic grids used by the repository examples."""
    return 10.0**np.linspace(np.log10(first), np.log10(last), count)


def love_model() -> dict[str, np.ndarray]:
    return {
        "z": np.array([0.0, 35.0, 35.0]),
        "rho": np.array([2.8, 2.8, 3.2]),
        "vsh": np.array([3.3, 3.3, 5.5]),
        "vsv": np.array([3.0, 3.0, 5.0]),
        "Qn": np.array([220.0, 220.0, 330.0]),
        "Ql": np.array([200.0, 200.0, 300.0]),
    }


def make_love(model: dict[str, np.ndarray]) -> SpecWorkSpace:
    ws = SpecWorkSpace()
    ws.initialize("love", model["z"], model["rho"],
                  vsh=model["vsh"], vsv=model["vsv"],
                  Qn=model["Qn"], Ql=model["Ql"],
                  reference_frequency=1.0)
    return ws


class SecondOrderJet:
    """Value, gradient, and Hessian for exact chain-rule differentiation."""

    __array_priority__ = 1000

    def __init__(self, value: complex, gradient: np.ndarray | None = None,
                 hessian: np.ndarray | None = None) -> None:
        self.value = complex(value)
        self.gradient = (np.zeros(3, dtype=complex) if gradient is None
                         else np.asarray(gradient, dtype=complex))
        self.hessian = (np.zeros((3, 3), dtype=complex) if hessian is None
                        else np.asarray(hessian, dtype=complex))

    @classmethod
    def variable(cls, value: complex, index: int) -> SecondOrderJet:
        gradient = np.zeros(3, dtype=complex)
        gradient[index] = 1.0
        return cls(value, gradient)

    @staticmethod
    def coerce(value: SecondOrderJet | complex) -> SecondOrderJet:
        return value if isinstance(value, SecondOrderJet) else SecondOrderJet(value)

    def unary(self, value: complex, first: complex,
              second: complex) -> SecondOrderJet:
        return SecondOrderJet(
            value,
            first*self.gradient,
            first*self.hessian + second*np.outer(self.gradient, self.gradient),
        )

    def __add__(self, other: SecondOrderJet | complex) -> SecondOrderJet:
        other = self.coerce(other)
        return SecondOrderJet(self.value + other.value,
                              self.gradient + other.gradient,
                              self.hessian + other.hessian)

    __radd__ = __add__

    def __neg__(self) -> SecondOrderJet:
        return SecondOrderJet(-self.value, -self.gradient, -self.hessian)

    def __sub__(self, other: SecondOrderJet | complex) -> SecondOrderJet:
        return self + (-self.coerce(other))

    def __rsub__(self, other: SecondOrderJet | complex) -> SecondOrderJet:
        return self.coerce(other) - self

    def __mul__(self, other: SecondOrderJet | complex) -> SecondOrderJet:
        other = self.coerce(other)
        return SecondOrderJet(
            self.value*other.value,
            self.gradient*other.value + self.value*other.gradient,
            self.hessian*other.value + self.value*other.hessian
            + np.outer(self.gradient, other.gradient)
            + np.outer(other.gradient, self.gradient),
        )

    __rmul__ = __mul__

    def reciprocal(self) -> SecondOrderJet:
        return self.unary(1.0/self.value, -1.0/self.value**2,
                          2.0/self.value**3)

    def __truediv__(self, other: SecondOrderJet | complex) -> SecondOrderJet:
        return self*self.coerce(other).reciprocal()

    def __rtruediv__(self, other: SecondOrderJet | complex) -> SecondOrderJet:
        return self.coerce(other)*self.reciprocal()

    def sqrt(self) -> SecondOrderJet:
        root = np.sqrt(self.value)
        return self.unary(root, 0.5/root, -0.25/root**3)

    def log(self) -> SecondOrderJet:
        return self.unary(np.log(self.value), 1.0/self.value,
                          -1.0/self.value**2)

    def exp(self) -> SecondOrderJet:
        value = np.exp(self.value)
        return self.unary(value, value, value)

    def arctan(self) -> SecondOrderJet:
        denominator = 1.0 + self.value**2
        return self.unary(np.arctan(self.value), 1.0/denominator,
                          -2.0*self.value/denominator**2)

    def tan(self) -> SecondOrderJet:
        value = np.tan(self.value)
        first = 1.0 + value**2
        return self.unary(value, first, 2.0*value*first)


def love_residual_jet(
    c: complex, frequency: float, model: dict[str, np.ndarray],
    parameter: str | None = None, layer: int = 0,
) -> SecondOrderJet:
    """Evaluate the Love residual and its exact first/second derivatives."""
    # Independent variables are c, omega, and an optional material parameter.
    c_jet = SecondOrderJet.variable(c, 0)
    omega = SecondOrderJet.variable(2.0*np.pi*frequency, 1)
    target = 0 if layer == 0 else 2

    def material_value(name: str, node: int) -> SecondOrderJet:
        value = model[name][node]
        if name.startswith("Q"):
            value = 1.0/value
        if parameter is not None and name == parameter and node == target:
            return SecondOrderJet.variable(value, 2)
        return SecondOrderJet(value)

    def factor(name: str, node: int) -> SecondOrderJet:
        qinv = material_value(name, node)
        alpha = (2.0/np.pi)*qinv.arctan()
        ratio = omega/(2.0*np.pi)
        return (1.0 + 1j*qinv)*(alpha*ratio.log()).exp()

    rho1, rho2 = model["rho"][[0, 2]]
    vsh1, vsh2 = material_value("vsh", 0), material_value("vsh", 2)
    vsv1, vsv2 = material_value("vsv", 0), material_value("vsv", 2)
    n1 = rho1*vsh1*vsh1*factor("Qn", 0)
    n2 = rho2*vsh2*vsh2*factor("Qn", 2)
    l1 = rho1*vsv1*vsv1*factor("Ql", 0)
    l2 = rho2*vsv2*vsv2*factor("Ql", 2)
    p = ((rho1*c_jet*c_jet - n1)/l1).sqrt()
    r = ((n2 - rho2*c_jet*c_jet)/l2).sqrt()
    return -p*(35.0*omega/c_jet*p).tan() + (l2/l1)*r


def analytical_love_derivatives(
    c: complex, frequency: float, model: dict[str, np.ndarray],
    parameter: str, layer: int,
) -> tuple[float, float]:
    """Return analytical dc/dm and dU/dm from the Love dispersion equation."""
    residual = love_residual_jet(c, frequency, model, parameter, layer)

    f_c, f_omega, f_m = residual.gradient
    f_cc = residual.hessian[0, 0]
    f_comega = residual.hessian[0, 1]
    f_cm = residual.hessian[0, 2]
    f_omegam = residual.hessian[1, 2]
    c_m = -f_m/f_c
    c_omega = -f_omega/f_c
    dc_omega_dm = -(
        (f_omegam + f_comega*c_m)*f_c
        - f_omega*(f_cm + f_cc*c_m)
    )/f_c**2
    omega_value = 2.0*np.pi*frequency
    inverse_group = 1.0/c - omega_value*c_omega/c**2
    inverse_group_m = (
        -c_m/c**2
        - omega_value*dc_omega_dm/c**2
        + 2.0*omega_value*c_omega*c_m/c**3
    )
    group_m = -inverse_group_m/inverse_group**2
    return float(c_m.real), float(group_m.real)


def love_residual_error(c: complex, frequency: float,
                        model: dict[str, np.ndarray]) -> float:
    """Return the example's first-order relative phase error in percent."""
    residual = love_residual_jet(c, frequency, model)
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(100.0*abs(residual.value/(residual.gradient[0]*c)))


def select_love_modes(model: dict[str, np.ndarray], frequency: float,
                      phase: np.ndarray, group: np.ndarray
                      ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Remove discretization roots and return physical modes in phase order."""
    errors = np.array([
        love_residual_error(c, frequency, model) for c in phase
    ])
    indices = np.flatnonzero(np.isfinite(errors) & (errors < 1.0e-3))
    indices = indices[np.argsort(phase[indices].real)]
    return phase[indices], group[indices], errors[indices], indices


def exact_love_group(c: complex, frequency: float,
                     model: dict[str, np.ndarray]) -> complex:
    if not (np.isfinite(c.real) and np.isfinite(c.imag)):
        return np.nan + 1j*np.nan
    omega = 2.0 * np.pi * frequency
    residual = love_residual_jet(c, frequency, model)
    f_c, f_omega = residual.gradient[:2]
    dc_domega = -f_omega / f_c
    return 1.0 / (1.0/c - omega/c**2 * dc_domega)


def love_figures() -> None:
    model = love_model()
    # example/love/run.sh: model.txt 0.01 1 500, with 19 plotted modes.
    frequencies = logarithmic_frequencies(0.01, 1.0, 500)
    nmode = 19
    shape = (nmode, frequencies.size)
    phase = np.full(shape, np.nan+1j*np.nan)
    group = np.full(shape, np.nan+1j*np.nan)
    phase_error = np.full(shape, np.nan)
    analytical_group = np.full_like(group, np.nan+1j*np.nan)
    adj_phase = np.full((4, 2, nmode, frequencies.size), np.nan)
    adj_group = np.full_like(adj_phase, np.nan)
    analytical_phase_derivative = np.full_like(adj_phase, np.nan)
    analytical_group_derivative = np.full_like(adj_phase, np.nan)
    layer_nodes = ((0, 1), (2,))
    parameter_rows = (("vsh", 0), ("vsv", 1), ("Qn", 3), ("Ql", 4))

    for jf, frequency in enumerate(frequencies):
        ws = make_love(model)
        raw_phase = ws.compute_egn(frequency, only_phase=False)
        raw_group = ws.group_velocity()
        current_phase, current_group, current_error, raw_indices = (
            select_love_modes(model, frequency, raw_phase, raw_group)
        )
        count = min(nmode, current_phase.size)
        phase[:count, jf] = current_phase[:count]
        group[:count, jf] = current_group[:count]
        phase_error[:count, jf] = current_error[:count]
        for mode in range(count):
            raw_mode = int(raw_indices[mode])
            phase_kernel, _ = ws.get_phase_kl(raw_mode)
            group_kernel, _ = ws.get_group_kl(raw_mode)
            for parameter, (_, row) in enumerate(parameter_rows):
                for layer, nodes in enumerate(layer_nodes):
                    adj_phase[parameter, layer, mode, jf] = np.sum(
                        phase_kernel[row, list(nodes)]
                    )
                    adj_group[parameter, layer, mode, jf] = np.sum(
                        group_kernel[row, list(nodes)]
                    )
            analytical_group[mode, jf] = exact_love_group(
                current_phase[mode], frequency, model
            )
            for parameter, (name, _) in enumerate(parameter_rows):
                for layer in range(2):
                    dc_dm, du_dm = analytical_love_derivatives(
                        current_phase[mode], frequency, model, name, layer
                    )
                    analytical_phase_derivative[parameter, layer, mode, jf] = dc_dm
                    analytical_group_derivative[parameter, layer, mode, jf] = du_dm

    cmap = plt.get_cmap("viridis", nmode)
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 7.0), sharex=True,
                             constrained_layout=True)
    for mode in range(nmode):
        color = cmap(mode)
        valid = np.isfinite(phase[mode].real)
        axes[0, 0].plot(frequencies[valid], phase[mode, valid].real, color=color)
        axes[0, 1].scatter(frequencies[valid], phase_error[mode, valid],
                           color=color, s=6)
        axes[1, 0].plot(frequencies[valid], group[mode, valid].real, color=color)
        valid_group = valid & np.isfinite(analytical_group[mode].real)
        axes[1, 0].scatter(frequencies[valid_group],
                           analytical_group[mode, valid_group].real,
                           color="k", s=5)
        axes[1, 1].plot(frequencies[valid_group],
                        1.0/propagation_inverse_q(group[mode, valid_group]), color=color)
        axes[1, 1].scatter(frequencies[valid_group],
                           (1.0/propagation_inverse_q(
                               analytical_group[mode, valid_group])),
                           color="k", s=5)
    labels = (("Phase velocity (km/s)", False), ("Relative error (%)", True),
              ("Group velocity (km/s)", False), ("Group quality factor", False))
    for panel, (ax, (ylabel, logarithmic)) in enumerate(zip(axes.flat, labels)):
        ax.set_ylabel(ylabel)
        ax.set_title(f"({chr(97 + panel)})", loc="left")
        if logarithmic:
            ax.set_yscale("log")
    fig.supxlabel("Frequency (Hz)")
    scalar = plt.cm.ScalarMappable(
        norm=plt.Normalize(0, nmode-1), cmap=cmap
    )
    fig.colorbar(scalar, ax=axes, label="mode", shrink=0.65)
    save_figure(fig, "love_wave_eigenvalues")

    panel_parameters = ((0, 0), (0, 1), (1, 0), (1, 1))
    for observable, prefix in ((adj_phase, "phase"), (adj_group, "group")):
        analytical = (analytical_phase_derivative if prefix == "phase"
                      else analytical_group_derivative)
        for family, suffix, symbols in (
            (panel_parameters, "veloc", (r"\beta_{h1}", r"\beta_{h2}",
                                         r"\beta_{v1}", r"\beta_{v2}")),
            (((2, 0), (2, 1), (3, 0), (3, 1)), "Q",
             (r"Q^{-1}_{h1}", r"Q^{-1}_{h2}",
              r"Q^{-1}_{v1}", r"Q^{-1}_{v2}")),
        ):
            fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.8), sharex=True,
                                     constrained_layout=True)
            for panel, ((parameter, layer), symbol) in enumerate(zip(family, symbols)):
                ax = axes.flat[panel]
                for mode in range(nmode):
                    valid = np.isfinite(observable[parameter, layer, mode])
                    ax.plot(frequencies[valid], observable[parameter, layer, mode, valid],
                            color=cmap(mode))
                    analytical_valid = valid & np.isfinite(
                        analytical[parameter, layer, mode]
                    )
                    ax.scatter(frequencies[analytical_valid],
                               analytical[parameter, layer, mode, analytical_valid],
                               color="k", s=4)
                quantity = "c" if prefix == "phase" else "U"
                ax.set_ylabel(rf"$\partial {quantity}/\partial {symbol}$")
                ax.set_title(f"({chr(97+panel)})", loc="left")
            fig.supxlabel("Frequency (Hz)")
            axes[0, 0].plot([], [], color="0.35", label="adjoint")
            axes[0, 0].scatter([], [], color="k", s=8,
                               label="analytical")
            axes[0, 0].legend()
            scalar = plt.cm.ScalarMappable(
                norm=plt.Normalize(0, nmode-1), cmap=cmap
            )
            fig.colorbar(scalar, ax=axes, label="mode", shrink=0.65)
            save_figure(fig, f"love_wave_{prefix}_deriv_{suffix}")


def brocher(vs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    vp = (0.9409 + 2.0947*vs - 0.8206*vs**2
          + 0.2683*vs**3 - 0.0251*vs**4)
    rho = (1.6612*vp - 0.4721*vp**2 + 0.0671*vp**3
           - 0.0043*vp**4 + 0.000106*vp**5)
    return vp, rho


def scholte_model() -> dict[str, np.ndarray]:
    thickness = np.array([5.0, 45.0, 50.0, 0.0])
    vs = np.array([0.0, 3.1, 4.0, 5.0])
    vp, rho = brocher(vs)
    vp[0], rho[0] = 1.5, 1.03
    z = np.array([0.0, 5.0, 5.0, 50.0, 50.0, 100.0, 100.0])
    layer = np.array([0, 0, 1, 1, 2, 2, 3])
    return {"thickness": thickness, "vp_layer": vp, "vs_layer": vs,
            "rho_layer": rho, "z": z, "vp": vp[layer],
            "vs": vs[layer], "rho": rho[layer], "eta": np.ones(z.size)}


def make_scholte(model: dict[str, np.ndarray]) -> SpecWorkSpace:
    ws = SpecWorkSpace()
    ws.initialize("rayl", model["z"], model["rho"],
                  vph=model["vp"], vpv=model["vp"],
                  vsv=model["vs"], eta=model["eta"])
    return ws


def scholte_figures() -> None:
    model = scholte_model()
    # example/scholte/step0_database.py uses this 100-point grid.
    frequencies = logarithmic_frequencies(0.01, 0.5, 100)
    periods = 1.0/frequencies
    nmode = 6
    sem_phase = np.full((nmode, frequencies.size), np.nan)
    sem_group = np.full_like(sem_phase, np.nan)
    th_phase = np.full_like(sem_phase, np.nan)
    th_group = np.full_like(sem_phase, np.nan)
    sem_derivative = np.full((2, 2, 4, frequencies.size), np.nan)
    fd_derivative = np.full_like(sem_derivative, np.nan)
    layer_nodes = ((0, 1), (2, 3), (4, 5), (6,))
    epsilon = 1.0e-2
    th = THSolver(model["thickness"], model["vp_layer"],
                  model["vs_layer"], model["rho_layer"])

    for jf, (frequency, period) in enumerate(zip(frequencies, periods)):
        ws = make_scholte(model)
        phase = ws.compute_egn(frequency, only_phase=False)
        group = ws.group_velocity()
        count = min(nmode, phase.size)
        sem_phase[:count, jf] = phase[:count]
        sem_group[:count, jf] = group[:count]
        for mode in range(nmode):
            th_phase[mode, jf] = np.asarray(
                th.compute_swd("Rc", mode, np.array([period]))
            ).ravel()[0]
            th_group[mode, jf] = np.asarray(
                th.compute_swd("Rg", mode, np.array([period]))
            ).ravel()[0]
        if count:
            phase_kernel = ws.get_phase_kl(0)
            group_kernel = ws.get_group_kl(0)
            for layer, nodes in enumerate(layer_nodes):
                sem_derivative[0, 0, layer, jf] = np.sum(
                    phase_kernel[[0, 1, 5]][:, list(nodes)]
                )
                sem_derivative[1, 0, layer, jf] = np.sum(
                    group_kernel[[0, 1, 5]][:, list(nodes)]
                )
                sem_derivative[0, 1, layer, jf] = np.sum(
                    phase_kernel[2, list(nodes)]
                )
                sem_derivative[1, 1, layer, jf] = np.sum(
                    group_kernel[2, list(nodes)]
                )

    for parameter, flag in ((0, "R"), (1, "R")):
        key = "vp_layer" if parameter == 0 else "vs_layer"
        for layer in range(4):
            value = model[key][layer]
            if value == 0.0:
                continue
            output = []
            for sign in (1.0, -1.0):
                changed = model[key].copy()
                changed[layer] *= 1.0 + sign*epsilon
                vp = changed if parameter == 0 else model["vp_layer"]
                vs = changed if parameter == 1 else model["vs_layer"]
                perturbed = THSolver(model["thickness"], vp, vs,
                                     model["rho_layer"])
                c = np.array([
                    np.asarray(perturbed.compute_swd(
                        f"{flag}c", 0, np.array([period])
                    )).ravel()[0] for period in periods
                ])
                u = np.array([
                    np.asarray(perturbed.compute_swd(
                        f"{flag}g", 0, np.array([period])
                    )).ravel()[0] for period in periods
                ])
                output.append((c, u))
            fd_derivative[0, parameter, layer] = (
                output[0][0]-output[1][0]
            )/(2*epsilon*value)
            fd_derivative[1, parameter, layer] = (
                output[0][1]-output[1][1]
            )/(2*epsilon*value)

    cmap = plt.get_cmap("viridis", nmode)
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.0), sharex=True,
                             constrained_layout=True)
    for mode in range(nmode):
        valid = th_phase[mode] > 0
        axes[0].plot(frequencies, sem_phase[mode], color=cmap(mode))
        axes[0].scatter(frequencies[valid][::3], th_phase[mode, valid][::3],
                        color="k", s=5)
        valid = th_group[mode] > 0
        axes[1].plot(frequencies, sem_group[mode], color=cmap(mode))
        axes[1].scatter(frequencies[valid][::3], th_group[mode, valid][::3],
                        color="k", s=5)
    for panel, (ax, ylabel) in enumerate(zip(
        axes, ("Phase velocity (km/s)", "Group velocity (km/s)"))
    ):
        ax.set_ylabel(ylabel)
        ax.set_title(f"({chr(97+panel)})", loc="left")
    fig.supxlabel("Frequency (Hz)")
    axes[0].plot([], [], color="0.35", label="SEM")
    axes[0].scatter([], [], color="k", s=8, label="Thomson-Haskell")
    axes[0].legend()
    scalar = plt.cm.ScalarMappable(norm=plt.Normalize(0, nmode-1), cmap=cmap)
    fig.colorbar(scalar, ax=axes, label="mode", shrink=0.7)
    save_figure(fig, "scholte_eigenvalues")

    # Rayleigh eigenfunctions use real polarization amplitudes; the physical
    # pi/2 phase shift between horizontal and vertical motion is implicit.
    ws = SpecWorkSpace()
    ws.initialize("rayl", model["z"], model["rho"],
                  vph=model["vp"], vpv=model["vp"], vsv=model["vs"],
                  eta=model["eta"])
    phase = ws.compute_egn(0.5, only_phase=False)
    count = min(10, phase.size)
    z_nodes = ws.get_znodes()
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 6.0), sharex=True,
                             sharey=True, constrained_layout=True)
    cmap_e = plt.get_cmap("viridis", count)
    for mode in range(count):
        displacement = np.asarray(ws.get_egnfunc(mode, True))
        pivot = np.argmax(np.abs(displacement[0]))
        if displacement[0, pivot] < 0.0:
            displacement *= -1.0
        components = (displacement[0], displacement[1])
        for values, ax in zip(components, axes):
            norm = np.max(np.abs(values))
            if norm > 0:
                values = values/norm
            ax.plot(values + mode, z_nodes, color=cmap_e(mode))
    for ax, title in zip(axes, (r"Normalized $u_x$", r"Normalized $u_z$")):
        ax.axhline(5.0, color="0.45", ls="--", lw=0.8)
        ax.set_xlabel(title)
        ax.set_ylim(100, 0)
    axes[0].set_ylabel("Depth (km)")
    scalar = plt.cm.ScalarMappable(norm=plt.Normalize(0, count-1), cmap=cmap_e)
    fig.colorbar(scalar, ax=axes, label="mode", shrink=0.7)
    save_figure(fig, "scholte_eigenvecs")

    for observable, stem, symbol in (
        (0, "scholte_phase_deriv", "c"),
        (1, "scholte_group_deriv", "U"),
    ):
        fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.0), sharex=True,
                                 constrained_layout=True)
        for parameter, (ax, label) in enumerate(zip(axes, (r"\alpha", r"\beta"))):
            for layer in range(4):
                sem_values = sem_derivative[observable, parameter, layer].copy()
                fd_values = fd_derivative[observable, parameter, layer].copy()

                # Match example/scholte/step2_kernels.py: every layer and
                # method is normalized by its own maximum magnitude and then
                # vertically offset.  The water-layer beta derivative is
                # identically zero and is retained as the first row.
                if parameter == 1 and model["vs_layer"][layer] == 0.0:
                    sem_values[:] = 0.0
                    fd_values[:] = 0.0

                def normalize_layer(values: np.ndarray) -> np.ndarray:
                    finite = np.isfinite(values)
                    if not np.any(finite):
                        return values
                    magnitude = np.max(np.abs(values[finite]))
                    if magnitude == 0.0:
                        magnitude = 1.0
                    return values/magnitude

                sem_values = normalize_layer(sem_values)
                fd_values = normalize_layer(fd_values)
                ax.plot(frequencies, sem_values + layer, color="m",
                        label="SEM" if layer == 3 else None)
                valid = np.isfinite(fd_values)
                sample = np.flatnonzero(valid)[::3]
                ax.scatter(frequencies[sample], fd_values[sample] + layer,
                           color="k", s=5,
                           label="centered difference" if layer == 3 else None)
            ax.set_title(f"({chr(97+parameter)})", loc="left")
            ax.set_yticks(np.arange(4))
            ax.set_yticklabels([
                rf"$\partial {symbol}/\partial {label}_{{{layer+1}}}$"
                for layer in range(4)
            ])
            ax.legend()
        fig.supxlabel("Frequency (Hz)")
        save_figure(fig, stem)


VOIGT = ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))
UPPER = tuple((i, j) for i in range(6) for j in range(i, 6))


def orthorhombic_tensor(vp0: float, vs0: float, rho: float) -> np.ndarray:
    e1, e2 = 0.2, 0.1
    d1, d2, d3 = 0.05, 0.2, -0.1
    g1, g2 = 0.15, 0.05
    c33 = rho*vp0**2
    c55 = rho*vs0**2
    c11, c22 = c33*(1+2*e2), c33*(1+2*e1)
    c66 = c55*(1+2*g1)
    c44 = c66/(1+2*g2)
    c23 = np.sqrt((c33-c44)**2+2*d1*c33*(c33-c44))-c44
    c13 = np.sqrt((c33-c55)**2+2*d2*c33*(c33-c55))-c55
    c12 = np.sqrt((c11-c66)**2+2*d3*c11*(c11-c66))-c66
    matrix = np.zeros((6, 6))
    np.fill_diagonal(matrix, (c11, c22, c33, c44, c55, c66))
    matrix[0, 1] = matrix[1, 0] = c12
    matrix[0, 2] = matrix[2, 0] = c13
    matrix[1, 2] = matrix[2, 1] = c23
    tensor = np.zeros((3, 3, 3, 3))
    for row, (i, j) in enumerate(VOIGT):
        for column, (k, l) in enumerate(VOIGT):
            for indices in ((i, j, k, l), (j, i, k, l),
                            (i, j, l, k), (j, i, l, k),
                            (k, l, i, j), (l, k, i, j),
                            (k, l, j, i), (l, k, j, i)):
                tensor[indices] = matrix[row, column]
    # theta_c=90 degrees and phi_c=45 degrees put c in the horizontal
    # plane.  The original model uses a as the vertical symmetry axis and b
    # as the horizontal axis perpendicular to c.
    phi = np.deg2rad(45.0)
    axis_a = np.array([0.0, 0.0, 1.0])
    axis_b = np.array([np.sin(phi), -np.cos(phi), 0.0])
    axis_c = np.array([np.cos(phi), np.sin(phi), 0.0])
    rotation = np.column_stack((axis_a, axis_b, axis_c))
    return np.einsum("ia,jb,kc,ld,abcd->ijkl",
                     rotation, rotation, rotation, rotation, tensor)


def tensor_to_c21(tensor: np.ndarray) -> np.ndarray:
    return np.array([
        tensor[VOIGT[i][0], VOIGT[i][1], VOIGT[j][0], VOIGT[j][1]]
        for i, j in UPPER
    ])


def orthorhombic_model() -> dict[str, np.ndarray]:
    depth = np.array([0.0, 5.0, 15.0, 25.0])
    # Values represented by the profiles in the original manuscript figure.
    vp0 = np.array([4.25, 4.40, 4.70, 5.05])
    vs0 = np.array([2.50, 2.60, 2.80, 3.00])
    rho0 = np.array([2.43, 2.45, 2.50, 2.50])
    z = np.array([0.0, 5.0, 5.0, 15.0, 15.0, 25.0, 25.0])
    layer = np.array([0, 0, 1, 1, 2, 2, 3])
    tensors = np.stack([
        orthorhombic_tensor(vp0[i], vs0[i], rho0[i]) for i in layer
    ])
    c21 = np.stack([tensor_to_c21(tensor) for tensor in tensors], axis=1)
    qm0 = np.array([150.0, 200.0, 300.0, 400.0])
    qk0 = 9.0/4.0*qm0
    return {"depth": depth, "vp0": vp0, "vs0": vs0, "rho0": rho0,
            "z": z, "rho": rho0[layer], "c21": c21,
            "Qani": np.vstack((qk0[layer], qm0[layer])),
            "qk0": qk0, "qm0": qm0, "half_tensor": tensors[-1]}


def make_aniso(model: dict[str, np.ndarray], attenuating: bool) -> SpecWorkSpace:
    ws = SpecWorkSpace()
    kwargs = {"c21": model["c21"]}
    if attenuating:
        kwargs["Qani"] = model["Qani"]
    ws.initialize("aniso", model["z"], model["rho"],
                  reference_frequency=1.0, **kwargs)
    return ws


def collect_aniso(ws: SpecWorkSpace, frequencies: np.ndarray,
                  azimuth: float, nmode: int
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    phase = np.full((nmode, frequencies.size), np.nan+1j*np.nan)
    radial = np.full_like(phase, np.nan+1j*np.nan)
    transverse = np.full_like(phase, np.nan+1j*np.nan)
    angle = np.deg2rad(azimuth)
    for jf, frequency in enumerate(frequencies):
        current = ws.compute_egn(frequency, azimuth, only_phase=False)
        group = ws.group_velocity()
        count = min(nmode, current.size)
        phase[:count, jf] = current[:count]
        radial[:count, jf] = (np.cos(angle)*group[0, :count]
                              + np.sin(angle)*group[1, :count])
        transverse[:count, jf] = (-np.sin(angle)*group[0, :count]
                                  + np.cos(angle)*group[1, :count])
    return phase, radial, transverse


def collect_aniso_spectrum(ws: SpecWorkSpace, frequencies: np.ndarray,
                           azimuth: float) -> list[np.ndarray]:
    """Return every computed eigenvalue at each frequency without truncation."""
    return [ws.compute_egn(frequency, azimuth).copy()
            for frequency in frequencies]


def body_velocities(model: dict[str, np.ndarray], azimuth: float) -> np.ndarray:
    angle = np.deg2rad(azimuth)
    direction = np.array([np.cos(angle), np.sin(angle), 0.0])
    christoffel = np.einsum(
        "ijkl,j,l->ik", model["half_tensor"], direction, direction
    ) / model["rho0"][-1]
    return np.sqrt(np.sort(np.linalg.eigvalsh(christoffel)))


def stepped_profile(values: np.ndarray,
                    layer_tops: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return x/y coordinates for a depth profile with sharp interfaces."""
    return np.repeat(values, 2)[:-1], np.repeat(layer_tops, 2)[1:]


def orthorhombic_figures() -> np.ndarray:
    model = orthorhombic_model()
    frequencies = np.linspace(0.01, 0.5, 50)
    elastic = make_aniso(model, False)
    elastic_spectrum = collect_aniso_spectrum(elastic, frequencies, 0.0)
    body = body_velocities(model, 0.0)

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.3), constrained_layout=True)
    for values, style, label in (
        (model["vs0"], {}, r"$V_{S0}$"),
        (model["vp0"], {}, r"$V_{P0}$"),
        (model["rho0"], {"ls": "--"}, r"$\rho$"),
    ):
        x_profile, z_profile = stepped_profile(values, model["depth"])
        axes[0].plot(x_profile, z_profile, label=label, **style)
    axes[0].invert_yaxis()
    axes[0].set_xlabel(r"Velocity (km/s) or density (g/cm$^3$)")
    axes[0].set_ylabel("Depth (km)")
    axes[0].set_title("(a) Reference profiles", loc="left")
    axes[0].legend()
    for frequency, current in zip(frequencies, elastic_spectrum):
        axes[1].scatter(np.full(current.size, frequency), current.real,
                        color="k", s=3, linewidths=0)
    for velocity in body:
        axes[1].axhline(velocity, color="#2166ac", lw=0.8, ls="--")
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].set_ylabel("Phase velocity (km/s)")
    axes[1].set_title(r"(b) Dispersion at $\phi=0^\circ$", loc="left")
    save_figure(fig, "ortho_eigenvalues")

    directions = (18.0, 36.0, 54.0, 72.0)
    fig, axes = plt.subplots(2, 2, figsize=(8.6, 7.0), sharex=True,
                             sharey=True, constrained_layout=True)
    for ax, azimuth in zip(axes.flat, directions):
        p, ur, ut = collect_aniso(elastic, frequencies, azimuth, 1)
        ax.plot(frequencies, p[0].real, color="k", ls="--", label=r"$c$")
        ax.plot(frequencies, ur[0].real, color="#d100d1", label=r"$u_r$")
        ax.plot(frequencies, ut[0].real, color="#7a1fa2", label=r"$u_t$")
        ax.set_title(rf"Dispersion, $\phi={azimuth:.0f}^\circ$")
        if azimuth == directions[0]:
            ax.legend(loc="best")
    fig.supxlabel("Frequency (Hz)")
    fig.supylabel("Velocity (km/s)")
    save_figure(fig, "ortho_egn_4direc")

    azimuths = np.linspace(0.0, 360.0, 73)
    selected_frequencies = (0.05, 0.1, 0.2)
    phase_az = np.zeros((3, azimuths.size))
    radial_az = np.zeros_like(phase_az)
    transverse_az = np.zeros_like(phase_az)
    for ia, azimuth in enumerate(azimuths):
        for jf, frequency in enumerate(selected_frequencies):
            p, ur, ut = collect_aniso(elastic, np.array([frequency]), azimuth, 1)
            phase_az[jf, ia] = p[0, 0].real
            radial_az[jf, ia] = ur[0, 0].real
            transverse_az[jf, ia] = ut[0, 0].real
    fig = plt.figure(figsize=(9.2, 4.1), constrained_layout=True)
    ax0 = fig.add_subplot(1, 2, 1)
    ax1 = fig.add_subplot(1, 2, 2)
    theta = np.deg2rad(azimuths)
    styles = ("-", "--", ":")
    for jf, frequency in enumerate(selected_frequencies):
        ax0.plot(phase_az[jf]*np.cos(theta),
                 phase_az[jf]*np.sin(theta), ls=styles[jf],
                 label=rf"$c$, {frequency:g} Hz")
        ax0.plot(radial_az[jf]*np.cos(theta),
                 radial_az[jf]*np.sin(theta), ls=styles[jf], alpha=0.65,
                 label=rf"$u_r$, {frequency:g} Hz")
        ax1.plot(-transverse_az[jf]*np.sin(theta),
                 transverse_az[jf]*np.cos(theta), ls=styles[jf],
                 label=rf"{frequency:g} Hz")
    ax0.set_title(r"(a) Phase velocity and $u_r$", loc="left")
    ax1.set_title(r"(b) Transverse group component $u_t$", loc="left")
    for ax in (ax0, ax1):
        ax.set_xlabel(r"$V_x$ (km/s)")
        ax.set_ylabel(r"$V_y$ (km/s)")
        ax.set_aspect("equal", adjustable="box")
    ax0.legend(ncol=2)
    ax1.legend()
    save_figure(fig, "ortho_phase-direc")

    attenuating = make_aniso(model, True)
    attenuating_spectrum = collect_aniso_spectrum(
        attenuating, frequencies, 0.0
    )
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.3), constrained_layout=True)
    for values, label in ((model["qk0"], r"$Q_\kappa$"),
                          (model["qm0"], r"$Q_\mu$")):
        x_profile, z_profile = stepped_profile(values, model["depth"])
        axes[0].plot(x_profile, z_profile, label=label)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Quality factor at 1 Hz")
    axes[0].set_ylabel("Depth (km)")
    axes[0].set_title("(a) Attenuation model", loc="left")
    axes[0].legend()
    for frequency, current, current_att in zip(
        frequencies, elastic_spectrum, attenuating_spectrum
    ):
        axes[1].scatter(np.full(current.size, frequency), current.real,
                        color="k", s=3, linewidths=0)
        axes[1].scatter(np.full(current_att.size, frequency), current_att.real,
                        color="#d100d1", marker="^", s=4, linewidths=0)
    for velocity in body:
        axes[1].axhline(velocity, color="#2166ac", lw=0.8, ls="--")
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].set_ylabel("Phase velocity (km/s)")
    axes[1].set_title(r"(b) Elastic and attenuating, $\phi=0^\circ$", loc="left")
    axes[1].scatter([], [], color="k", s=8, label="elastic")
    axes[1].scatter([], [], color="#d100d1", marker="^", s=10,
                    label="attenuating")
    axes[1].legend()
    save_figure(fig, "ortho_eigenvalues_att")

    phase_att_az = np.zeros_like(phase_az)
    quality_az = np.zeros_like(phase_az)
    for ia, azimuth in enumerate(azimuths):
        for jf, frequency in enumerate(selected_frequencies):
            p, _, _ = collect_aniso(
                attenuating, np.array([frequency]), azimuth, 1
            )
            phase_att_az[jf, ia] = p[0, 0].real
            quality_az[jf, ia] = 1.0/propagation_inverse_q(p[0, 0])
    fig = plt.figure(figsize=(9.2, 4.1), constrained_layout=True)
    ax0 = fig.add_subplot(1, 2, 1)
    ax1 = fig.add_subplot(1, 2, 2, projection="polar")
    for jf, frequency in enumerate(selected_frequencies):
        ax0.plot(phase_att_az[jf]*np.cos(theta),
                 phase_att_az[jf]*np.sin(theta), ls=styles[jf],
                 label=rf"{frequency:g} Hz")
        ax1.plot(theta, quality_az[jf], ls=styles[jf],
                 label=rf"{frequency:g} Hz")
    ax0.set_title("(a) Phase velocity", loc="left")
    ax1.set_title(r"(b) Phase quality factor $Q_c$", loc="left")
    ax0.set_xlabel(r"$V_x$ (km/s)")
    ax0.set_ylabel(r"$V_y$ (km/s)")
    ax0.set_aspect("equal", adjustable="box")
    ax0.legend(ncol=3)
    ax1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3)
    save_figure(fig, "ortho_phase-direc_att")
    return body


def main() -> None:
    love_figures()
    scholte_figures()
    body = orthorhombic_figures()
    print("Generated 14 numerical figures as PDF and JPEG in", OUT)
    print("Retained the original static model diagram as Figure 1")
    print("Orthorhombic half-space body velocities at phi=0 deg:",
          " ".join(f"{value:.3f}" for value in body), "km/s")


if __name__ == "__main__":
    main()
