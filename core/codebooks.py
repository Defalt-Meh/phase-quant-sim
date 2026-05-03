"""
Key codebook generators for the quantized-key experiments.

A codebook is an (L, d) array of integer indices in [0, q), representing
keys k_r in Omega_q^d via k_{r,i} = omega^{idx[r, i]}.

All generators return an integer array; conversion to complex keys is done
in the experiment scripts via `np.exp(2j * pi * idx / q)`.
"""

from __future__ import annotations
import numpy as np


def to_complex(idx: np.ndarray, q: int) -> np.ndarray:
    """Map integer indices (L, d) in [0, q) to complex keys in Omega_q^d."""
    return np.exp(2j * np.pi * np.asarray(idx) / q)


def lex(L: int, d: int, q: int) -> np.ndarray:
    """
    Lexicographic codebook: keys are the first L words of the q-ary
    counter of length d. Requires q^d >= L.
    """
    if q**d < L:
        raise ValueError(f"lex: q^d = {q**d} < L = {L}")
    out = np.zeros((L, d), dtype=np.int64)
    for r in range(L):
        x = r
        for i in range(d):
            out[r, d - 1 - i] = x % q
            x //= q
    return out


def random_codebook(L: int, d: int, q: int, rng: np.random.Generator) -> np.ndarray:
    """
    Random distinct codewords sampled uniformly from [q]^d. Rejection-resampled
    until L distinct words are obtained. Requires q^d >= L.
    """
    if q**d < L:
        raise ValueError(f"random_codebook: q^d = {q**d} < L = {L}")
    seen: set[tuple[int, ...]] = set()
    out = []
    while len(out) < L:
        w = tuple(int(x) for x in rng.integers(0, q, size=d))
        if w not in seen:
            seen.add(w)
            out.append(w)
    return np.array(out, dtype=np.int64)


def hamming_distance(idx: np.ndarray) -> np.ndarray:
    """Pairwise Hamming distances of an (L, d) integer codebook."""
    L = idx.shape[0]
    D = np.zeros((L, L), dtype=np.int64)
    for r in range(L):
        D[r] = np.sum(idx != idx[r], axis=1)
    return D


def greedy_maxdist(L: int, d: int, q: int, rng: np.random.Generator) -> np.ndarray:
    """
    Greedy max-distance codebook. Start from a random codeword; at each step
    add the codeword in [q]^d that maximizes the minimum Hamming distance to
    the chosen set. Tie-breaking: random.

    Enumerates all q^d candidates, so use only for moderate q^d (<= ~1e5).
    """
    if q**d < L:
        raise ValueError(f"greedy_maxdist: q^d = {q**d} < L = {L}")
    # Enumerate all candidates.
    grids = np.meshgrid(*[np.arange(q)] * d, indexing="ij")
    cand = np.stack([g.ravel() for g in grids], axis=1).astype(np.int64)
    n = cand.shape[0]

    chosen_mask = np.zeros(n, dtype=bool)
    first = int(rng.integers(0, n))
    chosen_idx = [first]
    chosen_mask[first] = True
    # min-distance from each candidate to the chosen set.
    min_d = np.sum(cand != cand[first], axis=1)

    while len(chosen_idx) < L:
        # Among unchosen, pick those with maximum min_d. Random tie-break.
        masked = np.where(chosen_mask, -1, min_d)
        best = masked.max()
        ties = np.flatnonzero(masked == best)
        pick = int(rng.choice(ties))
        chosen_idx.append(pick)
        chosen_mask[pick] = True
        # Update min_d.
        new_d = np.sum(cand != cand[pick], axis=1)
        min_d = np.minimum(min_d, new_d)

    return cand[np.array(chosen_idx)]


def balanced_spectrum(
    L: int, d: int, q: int, rng: np.random.Generator, n_trials: int = 64
) -> np.ndarray:
    """
    Best-of-n_trials random codebook minimizing the count of Hamming-1 pairs.
    A simple proxy for the "balanced Hamming spectrum" objective in §5.4.
    """
    best_idx = None
    best_score = np.inf
    for _ in range(n_trials):
        cand = random_codebook(L, d, q, rng)
        D = hamming_distance(cand)
        # Count of unordered Hamming-1 pairs.
        score = int((D == 1).sum() // 2)
        if score < best_score:
            best_score = score
            best_idx = cand
    assert best_idx is not None
    return best_idx


def n_star(idx: np.ndarray) -> int:
    """
    N* := max number of Hamming-1 neighbors of any single codeword.
    This is the quantity in Theorem 6's lower bound.
    """
    D = hamming_distance(idx)
    np.fill_diagonal(D, 999)
    return int((D == 1).sum(axis=1).max())


def hamming_spectrum(idx: np.ndarray) -> dict[int, int]:
    """Return {distance: count of unordered pairs} for the codebook."""
    D = hamming_distance(idx)
    L = idx.shape[0]
    iu = np.triu_indices(L, k=1)
    vals = D[iu]
    spectrum: dict[int, int] = {}
    for v in vals:
        spectrum[int(v)] = spectrum.get(int(v), 0) + 1
    return spectrum


def make_codebook(
    kind: str, L: int, d: int, q: int, rng: np.random.Generator
) -> np.ndarray:
    """Dispatch by string name."""
    kind = kind.lower()
    if kind == "lex":
        return lex(L, d, q)
    if kind == "random":
        return random_codebook(L, d, q, rng)
    if kind == "greedy":
        return greedy_maxdist(L, d, q, rng)
    if kind == "balanced":
        return balanced_spectrum(L, d, q, rng)
    raise ValueError(f"unknown codebook kind: {kind}")