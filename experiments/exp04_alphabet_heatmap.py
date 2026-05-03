"""
Experiment 4: Alphabet selection phase diagram (paper §5.3) +
full (m, eps) heatmap (reviewer fix 3.4).

For each (m, eps) cell, evaluate every q in q_set at the minimum phase
budget W = L(q) and at temperature T = T_star_noisy(L, q, eps) * (1 + slack).
Record retrieval accuracy. The "winning" q at each cell is the smallest
q achieving perfect retrieval (or the one with highest accuracy if none do).
"""
from __future__ import annotations
import argparse, os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from core.template import phases, attention_output, readout
from core.tables import random_table
from core.noise import make_noise
from core.thresholds import L_of, T_star_noisy, max_noise_tolerance


def midpoint_threshold_noisy(L: int, q: int, T: float, eps: float) -> float:
    eta = (L - 1) / (np.exp(T) + L - 1)
    cos_e = np.cos(eps)
    cos_gap = np.cos(2 * np.pi / q - eps)
    u_plus_lower = (1 - eta) * cos_e - eta
    u_minus_upper = (1 - eta) * cos_gap + eta
    return float(0.5 * (u_plus_lower + u_minus_upper))


def evaluate(
    a: np.ndarray, q: int, T: float, tau: float,
    eps: float, rng: np.random.Generator,
) -> float:
    L = a.shape[0]
    correct = 0
    for j in range(L):
        for s in range(q):
            noise = make_noise("uniform", L, q, eps, rng) if eps > 0 \
                    else np.zeros(L)
            v = phases(a, q)
            z = attention_output(v, j, T, noise_phase=noise)
            u = readout(z, s, q)
            pred = int(u >= tau)
            label = int(a[j] == s)
            correct += int(pred == label)
    return correct / (L * q)


def run_row(row: dict, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    m_grid = row["m_grid"]
    q_set = row["q_set"]
    slack = row["T_slack"]
    n_tables = row["n_tables"]

    out = []
    for m in m_grid:
        # eps grid: union of small sweep up to min(pi/q) across q_set.
        eps_max_overall = min(max_noise_tolerance(q) for q in q_set)
        eps_grid = np.linspace(0.0, eps_max_overall * 0.95, row["eps_grid_n"])

        for eps in eps_grid:
            for q in q_set:
                eps_max_q = max_noise_tolerance(q)
                if eps >= eps_max_q:
                    out.append({
                        "exp": "exp04", "seed": seed, "m": m,
                        "eps": float(eps), "q": q, "L": L_of(m, q),
                        "T": float("inf"), "tau": float("nan"),
                        "feasible": False, "n_tables": 0,
                        "acc_mean": 0.0, "acc_worst": 0.0,
                    })
                    continue

                L = L_of(m, q)
                T_star = T_star_noisy(L, q, float(eps))
                if not np.isfinite(T_star):
                    out.append({
                        "exp": "exp04", "seed": seed, "m": m,
                        "eps": float(eps), "q": q, "L": L,
                        "T": float("inf"), "tau": float("nan"),
                        "feasible": False, "n_tables": 0,
                        "acc_mean": 0.0, "acc_worst": 0.0,
                    })
                    continue

                T = T_star + slack
                tau = midpoint_threshold_noisy(L, q, T, float(eps))

                accs = []
                for _ in range(n_tables):
                    a = random_table(L, q, rng)
                    accs.append(evaluate(a, q, T, tau, float(eps), rng))
                accs = np.array(accs, dtype=float)

                out.append({
                    "exp": "exp04", "seed": seed, "m": m,
                    "eps": float(eps), "q": q, "L": L,
                    "T": T, "tau": tau, "feasible": True,
                    "n_tables": n_tables,
                    "acc_mean": float(accs.mean()),
                    "acc_worst": float(accs.min()),
                })
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--row", type=int, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--out", default="out/exp04")
    args = p.parse_args()

    cfg = yaml.safe_load(open(args.config))["exp04_alphabet_heatmap"]
    row = cfg["rows"][args.row]
    records = run_row(row, args.seed)

    Path(args.out).mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(records)
    fname = f"row{args.row:02d}_seed{args.seed:02d}.parquet"
    df.to_parquet(os.path.join(args.out, fname), index=False)
    print(f"[exp04] wrote {len(df)} rows -> {args.out}/{fname}")


if __name__ == "__main__":
    main()