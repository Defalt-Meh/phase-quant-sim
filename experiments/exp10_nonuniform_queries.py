"""
Experiment 10: Non-uniform query distribution (reviewer fix, §4 item 8).

Tests whether frequently-queried slots can tolerate lower temperature.
We weight each (j, s) input by p_j ~ exp(-skew * j / L) (so j=0 is most
frequent at large skew). For each skew level, sweep T and report a
WEIGHTED accuracy.

The naive theory predicts: weighted accuracy with skew>0 should reach 1.0
at LOWER T than the uniform case, since frequent slots dominate the loss.
"""
from __future__ import annotations
import argparse, os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from core.template import phases, attention_output, readout, midpoint_threshold
from core.tables import random_table
from core.thresholds import T_star_noiseless


def query_weights(L: int, q: int, skew: float) -> np.ndarray:
    """Return a (L*q,) probability vector over (j, s) inputs."""
    j_idx = np.repeat(np.arange(L), q)
    w = np.exp(-skew * j_idx / max(L - 1, 1))
    return w / w.sum()


def evaluate_weighted(
    a: np.ndarray, q: int, T: float, tau: float, weights: np.ndarray
) -> tuple[float, float]:
    """Return (uniform_acc, weighted_acc)."""
    L = a.shape[0]
    v = phases(a, q)
    correct_unif = 0
    correct_w = 0.0
    idx = 0
    for j in range(L):
        for s in range(q):
            z = attention_output(v, j, T)
            u = readout(z, s, q)
            pred = int(u >= tau)
            label = int(a[j] == s)
            ok = int(pred == label)
            correct_unif += ok
            correct_w += ok * weights[idx]
            idx += 1
    return correct_unif / (L * q), float(correct_w)


def run_row(row: dict, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    q, L = row["q"], row["L"]
    T_grid_n = row["T_grid_n"]
    skew_grid = row["skew_grid"]
    n_tables = row["n_tables"]

    T_star = T_star_noiseless(L, q)
    T_grid = np.linspace(0.4 * T_star, 1.5 * T_star, T_grid_n)

    out = []
    for skew in skew_grid:
        w = query_weights(L, q, float(skew))
        for T in T_grid:
            tau = midpoint_threshold(L, q, float(T))
            unif_accs, w_accs = [], []
            for _ in range(n_tables):
                a = random_table(L, q, rng)
                u_acc, w_acc = evaluate_weighted(a, q, float(T), tau, w)
                unif_accs.append(u_acc)
                w_accs.append(w_acc)
            out.append({
                "exp": "exp10", "seed": seed, "q": q, "L": L,
                "skew": float(skew),
                "T": float(T), "T_star": T_star,
                "T_factor": float(T) / T_star,
                "tau": tau, "n_tables": n_tables,
                "acc_uniform": float(np.mean(unif_accs)),
                "acc_weighted": float(np.mean(w_accs)),
            })
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--row", type=int, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--out", default="out/exp10")
    args = p.parse_args()

    cfg = yaml.safe_load(open(args.config))["exp10_nonuniform_queries"]
    row = cfg["rows"][args.row]
    records = run_row(row, args.seed)

    Path(args.out).mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(records)
    fname = f"row{args.row:02d}_seed{args.seed:02d}.parquet"
    df.to_parquet(os.path.join(args.out, fname), index=False)
    print(f"[exp10] wrote {len(df)} rows -> {args.out}/{fname}")


if __name__ == "__main__":
    main()