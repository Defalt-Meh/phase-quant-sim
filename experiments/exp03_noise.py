"""
Experiment 3: Phase-noise tolerance (paper §5.2) +
worst-case aligned noise (reviewer fix 3.3).

For fixed (q, L) and a temperature factor T_factor relative to the
noiseless T*, sweep the noise magnitude eps from 0 to pi/q. Compare
uniform random noise against worst-case aligned noise from Theorem 3's
proof.
"""
from __future__ import annotations
import argparse, os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from core.template import phases, attention_output, readout
from core.tables import random_table, adversarial_positive, adversarial_negative
from core.noise import make_noise
from core.thresholds import T_star_noiseless, max_noise_tolerance, c_q


def midpoint_threshold_noisy(L: int, q: int, T: float, eps: float) -> float:
    """tau = (u_+ + u_-) / 2 from Theorem 3's worst-case bounds."""
    eta = (L - 1) / (np.exp(T) + L - 1)
    cos_e = np.cos(eps)
    cos_gap = np.cos(2 * np.pi / q - eps)
    u_plus_lower = (1 - eta) * cos_e - eta
    u_minus_upper = (1 - eta) * cos_gap + eta
    return float(0.5 * (u_plus_lower + u_minus_upper))


def evaluate(
    a: np.ndarray, q: int, T: float, tau: float, noise_mode: str,
    eps: float, rng: np.random.Generator,
) -> float:
    """Accuracy over all (j, s)."""
    L = a.shape[0]
    correct = 0
    for j in range(L):
        for s in range(q):
            noise = make_noise(noise_mode, L, q, eps, rng, a=a, j=j, s=s)
            v = phases(a, q)
            z = attention_output(v, j, T, noise_phase=noise)
            u = readout(z, s, q)
            pred = int(u >= tau)
            label = int(a[j] == s)
            correct += int(pred == label)
    return correct / (L * q)


def evaluate_adversarial(
    q: int, L: int, T: float, tau: float, eps: float
) -> float:
    """Worst-case table AND worst-case aligned noise simultaneously."""
    rng = np.random.default_rng(0)  # unused: aligned noise is deterministic
    n_pairs = L * q
    n_correct = 0
    for j in range(L):
        for s in range(q):
            # Positive worst case + aligned noise.
            a = adversarial_positive(L, q, j, s)
            noise = make_noise("aligned", L, q, eps, rng, a=a, j=j, s=s)
            v = phases(a, q)
            z = attention_output(v, j, T, noise_phase=noise)
            u = readout(z, s, q)
            ok_pos = (u >= tau)

            # Negative worst case + aligned noise.
            a = adversarial_negative(L, q, j, s)
            noise = make_noise("aligned", L, q, eps, rng, a=a, j=j, s=s)
            v = phases(a, q)
            z = attention_output(v, j, T, noise_phase=noise)
            u = readout(z, s, q)
            ok_neg = (u < tau)

            if ok_pos and ok_neg:
                n_correct += 1
    return n_correct / n_pairs


def run_row(row: dict, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    q, L = row["q"], row["L"]
    T_factor = row["T_factor"]
    eps_max = max_noise_tolerance(q)
    eps_grid = np.linspace(0.0, eps_max * 0.999, row["eps_grid_n"])
    T = T_factor * T_star_noiseless(L, q)
    noise_mode = row["noise_mode"]

    out = []
    for eps in eps_grid:
        tau = midpoint_threshold_noisy(L, q, T, float(eps))

        if noise_mode == "aligned":
            acc = evaluate_adversarial(q, L, T, tau, float(eps))
            accs = [acc]
        else:
            accs = []
            for _ in range(row["n_tables"]):
                a = random_table(L, q, rng)
                accs.append(evaluate(a, q, T, tau, noise_mode,
                                     float(eps), rng))
        accs = np.array(accs, dtype=float)
        out.append({
            "exp": "exp03", "seed": seed, "q": q, "L": L,
            "T": T, "T_factor": T_factor, "eps": float(eps),
            "eps_max": eps_max, "tau": tau, "noise_mode": noise_mode,
            "n_tables": len(accs),
            "acc_mean": float(accs.mean()), "acc_worst": float(accs.min()),
        })
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--row", type=int, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--out", default="out/exp03")
    args = p.parse_args()

    cfg = yaml.safe_load(open(args.config))["exp03_noise"]
    row = cfg["rows"][args.row]
    records = run_row(row, args.seed)

    Path(args.out).mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(records)
    fname = f"row{args.row:02d}_seed{args.seed:02d}.parquet"
    df.to_parquet(os.path.join(args.out, fname), index=False)
    print(f"[exp03] wrote {len(df)} rows -> {args.out}/{fname}")


if __name__ == "__main__":
    main()