"""
Lookup-table generators for the recall task.

A table a is an integer array in [q]^L, defining f_a(j, s) = 1[a_j == s].
"""

from __future__ import annotations
import numpy as np


def random_table(L: int, q: int, rng: np.random.Generator) -> np.ndarray:
    """Uniformly random table in [q]^L."""
    return rng.integers(0, q, size=L, dtype=np.int64)


def adversarial_positive(L: int, q: int, j: int, s: int) -> np.ndarray:
    """
    Worst-case POSITIVE table from Theorem 7 (Appendix C).

    Sets a_j = s (the positive target), and for every distractor r != j
    chooses a_r so that Re(omega^{a_r - s}) is MINIMIZED. For even q the
    minimizer is a_r = s + q/2 (giving cos(pi) = -1); for odd q it is the
    floor or ceil of s + q/2.
    """
    a = np.zeros(L, dtype=np.int64)
    a[j] = s % q
    if q % 2 == 0:
        worst = (s + q // 2) % q
    else:
        # Pick whichever of floor/ceil gives smaller cosine; for odd q both
        # are tied numerically since cos((q-1)/q * pi) == cos((q+1)/q * pi).
        worst = (s + q // 2) % q
    for r in range(L):
        if r != j:
            a[r] = worst
    return a


def adversarial_negative(L: int, q: int, j: int, s: int) -> np.ndarray:
    """
    Worst-case NEGATIVE table from Theorem 7 (Appendix C).

    The reader probes input (j, s) but the table satisfies a_j != s. We
    choose a_j so that Re(omega^{a_j - s}) is MAXIMIZED among non-equal
    symbols, i.e., a_j = s + 1 (or s - 1; both give cos(2*pi/q) = c_q).
    Every distractor r != j is set to a_r = s, contributing the maximum
    +1 to u_-.
    """
    a = np.zeros(L, dtype=np.int64)
    a[j] = (s + 1) % q  # nearest non-equal symbol -> cos(2*pi/q) = c_q
    for r in range(L):
        if r != j:
            a[r] = s % q
    return a


def all_tables(L: int, q: int):
    """
    Iterator over all q^L tables in lexicographic order. Use only for
    small (L, q) — the small-L exhaustive impossibility check in exp01.
    """
    if q**L > 10_000_000:
        raise ValueError(f"all_tables: q^L = {q**L} too large to enumerate")
    a = np.zeros(L, dtype=np.int64)
    while True:
        yield a.copy()
        # Increment in base q.
        i = L - 1
        while i >= 0:
            a[i] += 1
            if a[i] < q:
                break
            a[i] = 0
            i -= 1
        if i < 0:
            return


def make_table(
    kind: str,
    L: int,
    q: int,
    rng: np.random.Generator,
    j: int | None = None,
    s: int | None = None,
) -> np.ndarray:
    """Dispatch by string name."""
    kind = kind.lower()
    if kind == "random":
        return random_table(L, q, rng)
    if kind == "adv_pos":
        if j is None or s is None:
            raise ValueError("adv_pos requires j, s")
        return adversarial_positive(L, q, j, s)
    if kind == "adv_neg":
        if j is None or s is None:
            raise ValueError("adv_neg requires j, s")
        return adversarial_negative(L, q, j, s)
    raise ValueError(f"unknown table kind: {kind}")