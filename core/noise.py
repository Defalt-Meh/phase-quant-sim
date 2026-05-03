"""
Phase-noise generators for the noise-robust experiments.

All generators return a (L,) real array of angular perturbations
eps_r in [-pi, pi], applied as v_r -> v_r * exp(i * eps_r).
"""

from __future__ import annotations
import numpy as np


def uniform(L: int, eps: float, rng: np.random.Generator) -> np.ndarray:
    """eps_r ~ Uniform[-eps, eps] independently per slot."""
    return rng.uniform(-eps, eps, size=L)


def aligned_worst_case(
    a: np.ndarray, j: int, s: int, q: int, eps: float
) -> np.ndarray:
    """
    Adversarial bounded noise from Theorem 3's worst case.

    Strategy: rotate every stored phase to be as harmful as possible to the
    correct prediction at input (j, s).

      - For the queried slot r=j:
          * if a_j == s   (positive): rotate AWAY from omega^s by +eps
                                       (reduces u_+).
          * if a_j != s   (negative): rotate TOWARD omega^s by sign that
                                       MAXIMIZES Re(omega^{a_j - s + eps})
                                       (increases u_-).
      - For r != j: rotate each v_r toward omega^s, i.e., choose
        eps_r in {+eps, -eps} that maximizes Re(omega^{a_r - s + eps_r}).

    Returns a (L,) array.
    """
    a = np.asarray(a, dtype=np.int64)
    L = a.shape[0]
    eps_arr = np.zeros(L)

    # Helper: pick sign in {+1, -1} that maximizes Re(omega^{delta + sigma*eps}).
    def best_sign_max(delta_int: int) -> float:
        theta = 2 * np.pi * delta_int / q
        v_plus = np.cos(theta + eps)
        v_minus = np.cos(theta - eps)
        return eps if v_plus >= v_minus else -eps

    def best_sign_min(delta_int: int) -> float:
        theta = 2 * np.pi * delta_int / q
        v_plus = np.cos(theta + eps)
        v_minus = np.cos(theta - eps)
        return eps if v_plus <= v_minus else -eps

    for r in range(L):
        delta = int(a[r] - s)
        if r == j:
            if a[j] == s:
                # Positive: minimize Re(omega^{0 + eps_j}).
                eps_arr[r] = best_sign_min(delta)
            else:
                # Negative: maximize Re(omega^{(a_j - s) + eps_j}).
                eps_arr[r] = best_sign_max(delta)
        else:
            # Distractors are summed into the readout regardless of label;
            # adversary maximizes them (helps negatives, hurts positives).
            eps_arr[r] = best_sign_max(delta)
    return eps_arr


def rounding_to_nearest_root(
    L: int, q: int, rng: np.random.Generator
) -> np.ndarray:
    """
    Realistic phase-quantization noise (§4 reviewer item 9).

    Sample full-precision unit-modulus weights w_r = exp(i * theta_r) with
    theta_r ~ Uniform[0, 2*pi). Quantize to nearest q-th root of unity.
    Return the residual angular error eps_r = theta_r - 2*pi*k_r/q,
    where k_r is the nearest index. By construction |eps_r| <= pi/q.
    """
    theta = rng.uniform(0.0, 2 * np.pi, size=L)
    k = np.round(q * theta / (2 * np.pi)).astype(np.int64) % q
    eps = theta - 2 * np.pi * k / q
    # Wrap to [-pi/q, pi/q].
    eps = (eps + np.pi) % (2 * np.pi) - np.pi
    return eps


def make_noise(
    kind: str,
    L: int,
    q: int,
    eps: float,
    rng: np.random.Generator,
    a: np.ndarray | None = None,
    j: int | None = None,
    s: int | None = None,
) -> np.ndarray:
    """Dispatch by string name."""
    kind = kind.lower()
    if kind == "none":
        return np.zeros(L)
    if kind == "uniform":
        return uniform(L, eps, rng)
    if kind == "aligned":
        if a is None or j is None or s is None:
            raise ValueError("aligned noise requires a, j, s")
        return aligned_worst_case(a, j, s, q, eps)
    if kind == "rounding":
        return rounding_to_nearest_root(L, q, rng)
    raise ValueError(f"unknown noise kind: {kind}")