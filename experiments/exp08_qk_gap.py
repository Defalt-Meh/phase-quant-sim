"""
Experiment 8: Quantized-key upper-vs-lower bound gap (reviewer fix, §4 item 6).

For the matched-query quantized-key setting, plot together:
  - the empirical T at which retrieval transitions to 100%,
  - the sufficient upper bound T_upper from Theorem 5,
  - the necessary lower bound T_lower from Theorem 6.

The empirical transition should lie in [T_lower, T_upper]. The width of
this interval shows how far the analysis is from being tight.
"""
from __future__ import annotations
import argparse, os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from core.qk_template import phases, attention_output, readout
from core.codebooks import make_codebook, to_complex, n_star
from core.tables import random_table
from core.thresholds import T_star_qk, T_lower_qk


def midpoint_threshold(L: int, q: int, T_eff: float) -> float:
    eta = (L - 1) / (np.exp(T_eff) + L - 1)
    c_q = np.cos(2 * np.pi / q)
    u_plus_lower = 1.0 - 2.0 * eta
    u_minus_upper = c_q + eta * (1.0 - c_q)
    return float(0.5 * (u_plus_lower + u_minus_upper))


def evaluate(
    a: np.ndarray, keys: np.ndarray, q: int, T: float, d: int, tau: float,
) -> float:
    L = a.shape[0]
    v = phases(a, q)
    correct = 0
    for j in range(L):
        for s in range(q):
            z = attention_output(v, keys, j, T, d)
            u = readout(z, s, q)
            pred = int(u >= tau)
            label = int(a[j] == s)
            correct += int(pred == label)
    return correct / (L * q)


def run_row(row: dict, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    q, L = row["q"], row["L"]
    d_grid = row["d_grid"]
    T_grid_n = row["T_grid_n"]
    codebook_kind = row["codebook"]
    n_tables = row["n_tables"]

    out = []
    for d in d_grid:
        if q ** d < L:
            continue
        idx = make_codebook(codebook_kind, L, d, q, rng)
        keys = to_complex(idx, q)
        ns = n_star(idx)
        T_upper = T_star_qk(L, q, d)
        T_lower = T_lower_qk(L, q, d, ns)
        # Sweep T spanning both bounds with a margin on each side.
        T_lo = max(0.5 * T_lower, 0.1)
        T_hi = 1.6 * T_upper
        T_grid = np.linspace(T_lo, T_hi, T_grid_n)

        for T in T_grid:
            T_eff = T * (1.0 - np.cos(2 * np.pi / q)) / d
            tau = midpoint_threshold(L, q, T_eff)
            accs = []
            for _ in range(n_tables):
                a = random_table(L, q, rng)
                accs.append(evaluate(a, keys, q, float(T), d, tau))
            accs = np.array(accs, dtype=float)
            out.append({
                "exp": "exp08", "seed": seed, "q": q, "L": L, "d": d,
                "codebook": codebook_kind, "n_star": ns,
                "T": float(T), "T_upper": T_upper, "T_lower": T_lower,
                "T_over_upper": float(T) / T_upper,
                "T_over_lower": float(T) / T_lower if T_lower > 0
                                else float("inf"),
                "tau": tau, "n_tables": n_tables,
                "acc_mean": float(accs.mean()),
                "acc_worst": float(accs.min()),
            })
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--row", type=int, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--out", default="out/exp08")
    args = p.parse_args()

    cfg = yaml.safe_load(open(args.config))["exp08_qk_gap"]
    row = cfg["rows"][args.row]
    records = run_row(row, args.seed)

    Path(args.out).mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(records)
    fname = f"row{args.row:02d}_seed{args.seed:02d}.parquet"
    df.to_parquet(os.path.join(args.out, fname), index=False)
    print(f"[exp08] wrote {len(df)} rows -> {args.out}/{fname}")


if __name__ == "__main__":
    main()