"""
Experiment 11: Realistic phase-quantization noise (reviewer fix, §4 item 9).

Instead of bounded uniform angular noise, use rounding-to-nearest-root
errors from random complex weights (Haar-uniform on the unit circle).
Compare against the uniform-noise case at matched eps_max = pi/q.

Question: does the qualitative theorem 3 picture (sharp transition,
geometric failure boundary) survive realistic quantization errors?
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
from core.thresholds import T_star_noiseless, max_noise_tolerance


def midpoint_threshold_noisy(L: int, q: int, T: float, eps: float) -> float:
    eta = (L - 1) / (np.exp(T) + L - 1)
    cos_e = np.cos(eps) if eps > 0 else 1.0
    cos_gap = np.cos(2 * np.pi / q - eps) if eps > 0 \
              else np.cos(2 * np.pi / q)
    u_plus_lower = (1 - eta) * cos_e - eta
    u_minus_upper = (1 - eta) * cos_gap + eta
    return float(0.5 * (u_plus_lower + u_minus_upper))


def evaluate(
    a: np.ndarray, q: int, T: float, tau: float, noise: np.ndarray
) -> float:
    L = a.shape[0]
    v = phases(a, q)
    correct = 0
    for j in range(L):
        for s in range(q):
            z = attention_output(v, j, T, noise_phase=noise)
            u = readout(z, s, q)
            pred = int(u >= tau)
            label = int(a[j] == s)
            correct += int(pred == label)
    return correct / (L * q)


def run_row(row: dict, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    q, L = row["q"], row["L"]
    T_grid_n = row["T_grid_n"]
    n_tables = row["n_tables"]

    T_star = T_star_noiseless(L, q)
    T_grid = np.linspace(0.4 * T_star, 1.6 * T_star, T_grid_n)
    eps_geom = max_noise_tolerance(q)  # pi/q

    out = []
    for noise_mode in ("none", "uniform", "rounding"):
        for T in T_grid:
            # Threshold uses worst-case eps_geom for the bounded modes,
            # 0 for none.
            eps_for_tau = eps_geom * 0.999 if noise_mode != "none" else 0.0
            tau = midpoint_threshold_noisy(L, q, float(T), eps_for_tau)
            accs = []
            for _ in range(n_tables):
                a = random_table(L, q, rng)
                if noise_mode == "none":
                    noise = np.zeros(L)
                elif noise_mode == "uniform":
                    noise = make_noise("uniform", L, q,
                                       eps_geom * 0.99, rng)
                else:  # rounding
                    noise = make_noise("rounding", L, q, 0.0, rng)
                accs.append(evaluate(a, q, float(T), tau, noise))
            accs = np.array(accs, dtype=float)
            out.append({
                "exp": "exp11", "seed": seed, "q": q, "L": L,
                "noise_mode": noise_mode,
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
    p.add_argument("--out", default="out/exp11")
    args = p.parse_args()

    cfg = yaml.safe_load(open(args.config))["exp11_realistic_noise"]
    row = cfg["rows"][args.row]
    records = run_row(row, args.seed)

    Path(args.out).mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(records)
    fname = f"row{args.row:02d}_seed{args.seed:02d}.parquet"
    df.to_parquet(os.path.join(args.out, fname), index=False)
    print(f"[exp11] wrote {len(df)} rows -> {args.out}/{fname}")


if __name__ == "__main__":
    main()