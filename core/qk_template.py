"""
Quantized-key matched-query template from Theorem 5 (Definition 4).

L memory slots with distinct key vectors k_r in Omega_q^d, value phases
v_r = omega^{a_r}, and matched queries q_j := k_j. Attention logits are
  ell_r^{(j)} = (T / d) * Re(<k_j, k_r>),
where <.,.> is the complex inner product (conjugate on the first arg).

This module is pure NumPy and stateless.
"""

from __future__ import annotations
import numpy as np


def omega(q: int) -> complex:
    return np.exp(2j * np.pi / q)


def phases(a: np.ndarray, q: int) -> np.ndarray:
    """Map table a in [q]^L to value phases v_r = omega^{a_r}."""
    return np.exp(2j * np.pi * np.asarray(a) / q)


def attention_logits(keys: np.ndarray, j: int, T: float, d: int) -> np.ndarray:
    """
    Compute attention logits for query slot j under matched queries.

    Args:
        keys:  (L, d) complex array of key vectors in Omega_q^d.
        j:     queried slot index.
        T:     temperature scalar.
        d:     key dimension (used in the T/d normalization).

    Returns:
        logits: (L,) real array.
    """
    keys = np.asarray(keys, dtype=np.complex128)
    qj = keys[j]
    # <k_j, k_r> = sum_i conj(k_{j,i}) * k_{r,i}
    inner = np.einsum("i,ri->r", np.conj(qj), keys)
    return (T / d) * np.real(inner)


def attention_output(
    v: np.ndarray,
    keys: np.ndarray,
    j: int,
    T: float,
    d: int,
    noise_phase: np.ndarray | None = None,
) -> complex:
    """
    Softmax attention output z for query j with quantized keys.

    Args:
        v:           (L,) complex value phases.
        keys:        (L, d) complex key vectors.
        j, T, d:     as in attention_logits.
        noise_phase: optional (L,) real angular perturbation on values.

    Returns:
        z: complex scalar.
    """
    v = np.asarray(v, dtype=np.complex128)
    if noise_phase is not None:
        v = v * np.exp(1j * np.asarray(noise_phase))
    logits = attention_logits(keys, j, T, d)
    logits = logits - logits.max()
    w = np.exp(logits)
    w = w / w.sum()
    return complex(np.sum(w * v))


def readout(z: complex, s: int, q: int) -> float:
    return float(np.real(np.exp(-2j * np.pi * s / q) * z))


def predict(
    a: np.ndarray,
    keys: np.ndarray,
    j: int,
    s: int,
    q: int,
    T: float,
    d: int,
    tau: float,
    noise_phase: np.ndarray | None = None,
) -> int:
    """Predict f_a(j, s) = 1[a_j == s] under quantized-key addressing."""
    v = phases(a, q)
    z = attention_output(v, keys, j, T, d, noise_phase=noise_phase)
    u = readout(z, s, q)
    return int(u >= tau)


def attention_weights(keys: np.ndarray, j: int, T: float, d: int) -> np.ndarray:
    """Return the (L,) softmax attention distribution for query j."""
    logits = attention_logits(keys, j, T, d)
    logits = logits - logits.max()
    w = np.exp(logits)
    return w / w.sum()