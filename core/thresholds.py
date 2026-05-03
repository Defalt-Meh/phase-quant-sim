"""
Closed-form thresholds and theoretical predictions from the paper.

All formulas reference equation numbers in the LaTeX source.
"""

from __future__ import annotations
import numpy as np


def c_q(q: int) -> float:
    """c_q = cos(2*pi/q), the maximum non-identity real part. Eq. before (4)."""
    return float(np.cos(2 * np.pi / q))


def m_q(q: int) -> float:
    """m_q = min_{1<=r<=q-1} Re(omega^r) = cos(pi) = -1 for even q,
    else cos((q-1)*pi/q)."""
    if q % 2 == 0:
        return -1.0
    return float(np.cos((q - 1) * np.pi / q))


def beta_q(q: int) -> float:
    """beta_q = (1 - m_q) / (2 - m_q - c_q). Theorem 4."""
    mq = m_q(q)
    cq = c_q(q)
    return (1.0 - mq) / (2.0 - mq - cq)


def L_of(m_bits: int, q: int) -> int:
    """L = ceil(m / log_2 q). Section 3."""
    return int(np.ceil(m_bits / np.log2(q)))


def T_star_noiseless(L: int, q: int) -> float:
    """
    Sufficient temperature from Theorem 2:
        e^T > 2(L-1) / (1 - c_q)
    Returns ln of the right-hand side.
    """
    cq = c_q(q)
    if L <= 1:
        return -np.inf
    return float(np.log(2.0 * (L - 1) / (1.0 - cq)))


def T_star_noisy(L: int, q: int, eps: float) -> float:
    """
    Sufficient temperature from Theorem 3:
        e^T > 2(L-1) / (cos(eps) - cos(2*pi/q - eps))
    Returns ln of the right-hand side. Inf if eps >= pi/q.
    """
    if L <= 1:
        return -np.inf
    denom = np.cos(eps) - np.cos(2 * np.pi / q - eps)
    if denom <= 0:
        return float("inf")
    return float(np.log(2.0 * (L - 1) / denom))


def T_star_qk(L: int, q: int, d: int) -> float:
    """
    Sufficient temperature for quantized-key matched-query template, Theorem 5:
        T > (d / (1 - c_q)) * ln(2(L-1) / (1 - c_q))
    """
    cq = c_q(q)
    if L <= 1:
        return -np.inf
    return float((d / (1.0 - cq)) * np.log(2.0 * (L - 1) / (1.0 - cq)))


def T_lower_qk(L: int, q: int, d: int, n_star: int) -> float:
    """
    Lower bound (necessary temperature) from Theorem 6:
        T >= max(
            (d / (1 - m_q)) * ln(N* (1 - m_q) / (1 - c_q)),
            (1 / (1 - m_q)) * ln((L-1) (1 - m_q) / (1 - c_q))
        )
    """
    mq = m_q(q)
    cq = c_q(q)
    one_minus_mq = 1.0 - mq
    one_minus_cq = 1.0 - cq

    arg1 = max(n_star * one_minus_mq / one_minus_cq, 1e-300)
    arg2 = max((L - 1) * one_minus_mq / one_minus_cq, 1e-300)

    bound1 = (d / one_minus_mq) * np.log(arg1)
    bound2 = (1.0 / one_minus_mq) * np.log(arg2)
    return float(max(bound1, bound2))


def temperature_cost(m_bits: int, q: int, eps: float = 0.0) -> float:
    """
    T*_eps(q; m) from Definition 5 / Eq. (8).
    """
    L = L_of(m_bits, q)
    return T_star_noisy(L, q, eps)


def max_noise_tolerance(q: int) -> float:
    """The geometric hard limit eps < pi/q from Theorem 3."""
    return float(np.pi / q)