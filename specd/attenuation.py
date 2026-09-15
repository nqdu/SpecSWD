"""Causal constant-Q attenuation used by the frequency-domain solver."""

from __future__ import annotations

import numpy as np


def _validated_inputs(freq, quality_factor, reference_frequency):
    freq = np.asarray(freq, dtype=float)
    quality_factor = np.asarray(quality_factor, dtype=float)
    reference_frequency = np.asarray(reference_frequency, dtype=float)
    if np.any(~np.isfinite(freq)) or np.any(freq <= 0.0):
        raise ValueError("freq must be positive and finite")
    if np.any(np.isnan(quality_factor)) or np.any(quality_factor <= 0.0):
        raise ValueError("quality_factor must be positive")
    if (np.any(~np.isfinite(reference_frequency))
            or np.any(reference_frequency <= 0.0)):
        raise ValueError("reference_frequency must be positive and finite")
    return freq, quality_factor, reference_frequency


def constant_q_response(freq, quality_factor, reference_frequency=1.0):
    """Return ``M(omega) / M0`` for storage modulus ``M0`` at the reference.

    The response uses the positive-frequency convention of SpecSWD:

    ``s = (1 + i/Q) * (freq/reference_frequency)**alpha``,
    ``alpha = 2/pi * atan(1/Q)``.
    """
    freq, quality_factor, reference_frequency = _validated_inputs(
        freq, quality_factor, reference_frequency
    )
    qinv = 1.0 / quality_factor
    alpha = 2.0 / np.pi * np.arctan(qinv)
    log_ratio = np.log(freq) - np.log(reference_frequency)
    return np.exp(alpha * log_ratio) * (1.0 + 1j * qinv)


def constant_q_response_derivatives(
        freq, quality_factor, reference_frequency=1.0):
    """Return ``s``, ``ds/domega``, ``ds/d(Q^-1)``, and the mixed derivative."""
    freq, quality_factor, reference_frequency = _validated_inputs(
        freq, quality_factor, reference_frequency
    )
    qinv = 1.0 / quality_factor
    omega = 2.0 * np.pi * freq
    alpha = 2.0 / np.pi * np.arctan(qinv)
    dalpha_dqinv = 2.0 / (np.pi * (1.0 + qinv * qinv))
    log_ratio = np.log(freq) - np.log(reference_frequency)
    amplitude = np.exp(alpha * log_ratio)
    factor = amplitude * (1.0 + 1j * qinv)
    d_factor_d_omega = alpha / omega * factor
    d_factor_d_qinv = amplitude * (
        1j + (1.0 + 1j * qinv) * dalpha_dqinv * log_ratio
    )
    mixed = (
        dalpha_dqinv * factor + alpha * d_factor_d_qinv
    ) / omega
    return factor, d_factor_d_omega, d_factor_d_qinv, mixed
