"""
Experiment 9: Multi-head extension (reviewer fix, §4 item 7).

A naive multi-head construction: H independent copies of the single-head
template, each carrying its own value phases v_r^{(h)} = omega^{a_r^{(h)}}.
The shared table a in [q]^L is repeated identically across heads (no
extra information), and the readouts are averaged:
    z = (1/H) sum_h z^{(h)}.

With identical heads and identical noise, this is equivalent to the
single-head case and should NOT improve accuracy. With independent
noise per head, it should average out the perturbation.

We test:
  (a) noiseless: identical across heads -> no improvement (sanity check);
  (b) noisy:     iid noise across heads -> empirical improvement at
                 fixed T as H grows.
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
from core.thresholds import T_star_noiseless


def midpoint_threshold_noisy(L: int, q: int, T: float, eps: float) -> float:
    eta = (L - 1) / (np.exp(T) + L - 1)
    cos_e = np.cos(eps) if eps > 0 else 1.0
    cos_gap = np.cos(2 * np.pi / q - eps) if eps > 0 \
              else np.cos(2 * np.pi / q)
    u_plus_lower = (1 - eta) * cos_e - eta
    u_minus_upper = (1 - eta) * cos_gap + eta
    return float(0.5 * (u_plus_lower + u_minus_upper))


def evaluate(
    a: np.ndarray, q: int, T: float, tau: float, H: int, eps: float,
    rng: np.random.Generator,
) -> float:
    L = a.shape[0]
    correct = 0
    for j in range(L):
        for s in range(q):
            zsum = 0.0 + 0.0j
            for _ in range(H):
                noise = make_noise("uniform", L, q, eps, rng) \
                        if eps > 0 else np.zeros(L)
                v = phases(a, q)
                zsum += attention_output(v, j, T, noise_phase=noise)
            z = zsum / H
            u = readout(z, s, q)
            pred = int(u >= tau)
            label = int(a[j] == s)
            correct += int(pred == label)
    return correct / (L * q)


def run_row(row: dict, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    q, L = row["q"], row["L"]
    H_grid = row["H_grid"]
    T_grid_n = row["T_grid_n"]
    eps = row["eps"]
    n_tables = row["n_tables"]

    T_star = T_star_noiseless(L, q)
    T_grid = np.linspace(0.4 * T_star, 1.5 * T_star, T_grid_n)

    out = []
    for H in H_grid:
        for T in T_grid:
            tau = midpoint_threshold_noisy(L, q, float(T), eps)
            accs = []
            for _ in range(n_tables):
                a = random_table(L, q, rng)
                accs.append(evaluate(a, q, float(T), tau, H, eps, rng))
            accs = np.array(accs, dtype=float)
            out.append({
                "exp": "exp09", "seed": seed, "q": q, "L": L, "H": H,
                "T": float(T), "T_star": T_star,
                "T_factor": float(T) / T_star,
                "eps": eps, "tau": tau, "n_tables": n_tables,
                "acc_mean": float(accs.mean()),
                "acc_worst": float(accs.min()),
            })
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--row", type=int, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--out", default="out/exp09")
    args = p.parse_args()

    cfg = yaml.safe_load(open(args.config))["exp09_multihead"]
    row = cfg["rows"][args.row]
    records = run_row(row, args.seed)

    Path(args.out).mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(records)
    fname = f"row{args.row:02d}_seed{args.seed:02d}.parquet"
    df.to_parquet(os.path.join(args.out, fname), index=False)
    print(f"[exp09] wrote {len(df)} rows -> {args.out}/{fname}")


if __name__ == "__main__":
    main()