"""
Experiment 7: Finite-size transition sharpness (reviewer fix, §4 item 4).

For increasing L, measure the empirical transition width (the gap in T
between accuracy 0.1 and 0.9). The theory predicts this width should
decrease (or at most logarithmically grow) with L.
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


def evaluate(a: np.ndarray, q: int, T: float, tau: float) -> float:
    L = a.shape[0]
    v = phases(a, q)
    correct = 0
    for j in range(L):
        for s in range(q):
            z = attention_output(v, j, T)
            u = readout(z, s, q)
            pred = int(u >= tau)
            label = int(a[j] == s)
            correct += int(pred == label)
    return correct / (L * q)


def run_row(row: dict, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    q = row["q"]
    L_grid = row["L_grid"]
    T_grid_n = row["T_grid_n"]
    n_tables = row["n_tables"]

    out = []
    for L in L_grid:
        T_star = T_star_noiseless(L, q)
        T_grid = np.linspace(0.4 * T_star, 1.5 * T_star, T_grid_n)

        for T in T_grid:
            tau = midpoint_threshold(L, q, float(T))
            accs = []
            for _ in range(n_tables):
                a = random_table(L, q, rng)
                accs.append(evaluate(a, q, float(T), tau))
            accs = np.array(accs, dtype=float)
            out.append({
                "exp": "exp07", "seed": seed, "q": q, "L": L,
                "T": float(T), "T_star": T_star,
                "T_factor": float(T) / T_star,
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
    p.add_argument("--out", default="out/exp07")
    args = p.parse_args()

    cfg = yaml.safe_load(open(args.config))["exp07_transition_sharpness"]
    row = cfg["rows"][args.row]
    records = run_row(row, args.seed)

    Path(args.out).mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(records)
    fname = f"row{args.row:02d}_seed{args.seed:02d}.parquet"
    df.to_parquet(os.path.join(args.out, fname), index=False)
    print(f"[exp07] wrote {len(df)} rows -> {args.out}/{fname}")


if __name__ == "__main__":
    main()