"""
Experiment 6: Worst-case margin validation (reviewer fix 4.1).

Builds the EXACT adversarial positive and negative tables from the proofs
of Theorem 2 and Theorem 3. Checks that retrieval succeeds iff
e^T > 2(L-1) / (cos(eps) - cos(2*pi/q - eps)).

This is the cleanest possible falsification test: if the theorem is
correct, the empirical transition coincides with the theoretical
threshold to within a single grid step.
"""
from __future__ import annotations
import argparse, os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from core.template import phases, attention_output, readout
from core.tables import adversarial_positive, adversarial_negative
from core.noise import make_noise
from core.thresholds import T_star_noisy, max_noise_tolerance


def midpoint_threshold_noisy(L: int, q: int, T: float, eps: float) -> float:
    eta = (L - 1) / (np.exp(T) + L - 1)
    cos_e = np.cos(eps)
    cos_gap = np.cos(2 * np.pi / q - eps)
    u_plus_lower = (1 - eta) * cos_e - eta
    u_minus_upper = (1 - eta) * cos_gap + eta
    return float(0.5 * (u_plus_lower + u_minus_upper))


def evaluate_pair(
    q: int, L: int, T: float, tau: float, eps: float
) -> tuple[float, float]:
    """
    For every (j, s):
      - build adversarial positive table -> readout u_pos, predict 1 iff u >= tau.
      - build adversarial negative table -> readout u_neg, predict 0 iff u <  tau.
    Returns (positive_accuracy, negative_accuracy).
    Both noises are aligned worst-case.
    """
    rng = np.random.default_rng(0)
    n_pos_correct = 0
    n_neg_correct = 0
    n_pairs = L * q
    for j in range(L):
        for s in range(q):
            # Positive worst case.
            a = adversarial_positive(L, q, j, s)
            noise = make_noise("aligned", L, q, eps, rng, a=a, j=j, s=s) \
                    if eps > 0 else np.zeros(L)
            v = phases(a, q)
            z = attention_output(v, j, T, noise_phase=noise)
            u = readout(z, s, q)
            if u >= tau:
                n_pos_correct += 1

            # Negative worst case.
            a = adversarial_negative(L, q, j, s)
            noise = make_noise("aligned", L, q, eps, rng, a=a, j=j, s=s) \
                    if eps > 0 else np.zeros(L)
            v = phases(a, q)
            z = attention_output(v, j, T, noise_phase=noise)
            u = readout(z, s, q)
            if u < tau:
                n_neg_correct += 1
    return n_pos_correct / n_pairs, n_neg_correct / n_pairs


def run_row(row: dict, seed: int) -> list[dict]:
    q = row["q"]
    L_grid = row["L_grid"]
    eps_grid = row["eps_grid"]
    T_grid_n = row["T_grid_n"]

    out = []
    for L in L_grid:
        eps_max = max_noise_tolerance(q)
        for eps in eps_grid:
            if eps >= eps_max:
                continue
            T_star = T_star_noisy(L, q, float(eps))
            if not np.isfinite(T_star):
                continue
            T_grid = np.linspace(0.3 * T_star, 1.5 * T_star, T_grid_n)

            for T in T_grid:
                tau = midpoint_threshold_noisy(L, q, float(T), float(eps))
                acc_pos, acc_neg = evaluate_pair(
                    q, L, float(T), tau, float(eps)
                )
                acc_both = min(acc_pos, acc_neg)
                out.append({
                    "exp": "exp06", "seed": seed, "q": q, "L": L,
                    "eps": float(eps), "T": float(T), "T_star": T_star,
                    "T_factor": float(T) / T_star, "tau": tau,
                    "acc_pos": acc_pos, "acc_neg": acc_neg,
                    "acc_worst": acc_both,
                })
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--row", type=int, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--out", default="out/exp06")
    args = p.parse_args()

    cfg = yaml.safe_load(open(args.config))["exp06_worst_case_margin"]
    row = cfg["rows"][args.row]
    records = run_row(row, args.seed)

    Path(args.out).mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(records)
    fname = f"row{args.row:02d}_seed{args.seed:02d}.parquet"
    df.to_parquet(os.path.join(args.out, fname), index=False)
    print(f"[exp06] wrote {len(df)} rows -> {args.out}/{fname}")


if __name__ == "__main__":
    main()