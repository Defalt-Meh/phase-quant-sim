"""
Hardwired-attention lookup template from Theorem 1(i).

The template has L+1 tokens: 1 query, L memory slots. On input (j, s):
  - attention logit T on slot j, 0 on others (hardwired)
  - value phases v_r = omega^{a_r} in Omega_q
  - readout u(j, s) = Re(omega^{-s} * z), where z is the attention output

This module is pure NumPy and stateless. No training: parameters are the
table a in [q]^L which is passed in directly.
"""

from __future__ import annotations
import numpy as np


def omega(q: int) -> complex:
    """Primitive q-th root of unity."""
    return np.exp(2j * np.pi / q)


def phases(a: np.ndarray, q: int) -> np.ndarray:
    """Map a table a in [q]^L to value phases v_r = omega^{a_r} in C^L."""
    return np.exp(2j * np.pi * np.asarray(a) / q)


def attention_output(
    v: np.ndarray, j: int, T: float, noise_phase: np.ndarray | None = None
) -> complex:
    """
    Compute z for query slot j with attention logit T on slot j and 0 elsewhere.

    Args:
        v: (L,) complex array of value phases.
        j: queried slot index in [0, L).
        T: attention logit on the queried slot.
        noise_phase: optional (L,) real array of angular perturbations
                     added to each stored phase, i.e., v_r -> v_r * exp(i * eps_r).

    Returns:
        z: complex scalar, the attention-weighted sum of values.
    """
    v = np.asarray(v, dtype=np.complex128)
    L = v.shape[0]
    if noise_phase is not None:
        v = v * np.exp(1j * np.asarray(noise_phase))
    logits = np.zeros(L)
    logits[j] = T
    # Numerically stable softmax (subtract max).
    logits = logits - logits.max()
    w = np.exp(logits)
    w = w / w.sum()
    return complex(np.sum(w * v))


def readout(z: complex, s: int, q: int) -> float:
    """Compute u = Re(omega^{-s} * z)."""
    return float(np.real(np.exp(-2j * np.pi * s / q) * z))


def predict(
    a: np.ndarray,
    j: int,
    s: int,
    q: int,
    T: float,
    tau: float,
    noise_phase: np.ndarray | None = None,
) -> int:
    """
    Predict the lookup label f_a(j, s) = 1[a_j == s] using threshold tau.

    Returns 1 if u >= tau else 0.
    """
    v = phases(a, q)
    z = attention_output(v, j, T, noise_phase=noise_phase)
    u = readout(z, s, q)
    return int(u >= tau)


def all_inputs(L: int, q: int) -> np.ndarray:
    """Return all (j, s) pairs in [L] x [q] as an (L*q, 2) integer array."""
    js = np.repeat(np.arange(L), q)
    ss = np.tile(np.arange(q), L)
    return np.stack([js, ss], axis=1)


def margin_bounds(L: int, q: int, T: float) -> tuple[float, float]:
    """
    Theorem 2 worst-case bounds (no noise):
      u_+ >= 1 - 2*eta
      u_- <= c_q + eta*(1 - c_q)
    where eta = (L-1) / (e^T + L - 1).
    Returns (u_plus_lower, u_minus_upper).
    """
    eta = (L - 1) / (np.exp(T) + L - 1)
    c_q = np.cos(2 * np.pi / q)
    u_plus_lower = 1.0 - 2.0 * eta
    u_minus_upper = c_q + eta * (1.0 - c_q)
    return float(u_plus_lower), float(u_minus_upper)


def midpoint_threshold(L: int, q: int, T: float) -> float:
    """tau = (u_+ + u_-) / 2 from theoretical margin bounds."""
    u_plus, u_minus = margin_bounds(L, q, T)
    return 0.5 * (u_plus + u_minus)