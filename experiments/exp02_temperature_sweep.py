"""
Experiment 2: Temperature sweep (paper §5.1, right panel) +
adversarial tables (reviewer fix 3.2).

For fixed (q, L), sweep T over T_factor * T*, where T* is the
sufficient temperature from Theorem 2. Test on either random tables
or the worst-case adversarial table from Theorem 7.
"""
from __future__ import annotations
import argparse, os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from core.template import phases, attention_output, readout, midpoint_threshold
from core.tables import random_table, adversarial_positive, adversarial_negative
from core.thresholds import T_star_noiseless


def evaluate_table(a: np.ndarray, q: int, T: float, tau: float) -> float:
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


def evaluate_adversarial(q: int, L: int, T: float, tau: float) -> float:
    """
    For each (j, s) we build the worst-case positive table and worst-case
    negative table, then check whether each is correctly classified.
    Reports fraction of (j, s) pairs that BOTH adversarial tables get right.
    """
    n_pairs = L * q
    n_correct = 0
    for j in range(L):
        for s in range(q):
            # Positive worst case: a_j = s, distractors hostile.
            a_pos = adversarial_positive(L, q, j, s)
            v = phases(a_pos, q)
            z = attention_output(v, j, T)
            u = readout(z, s, q)
            ok_pos = (u >= tau)  # label = 1

            # Negative worst case: a_j != s (hardest), distractors == s.
            a_neg = adversarial_negative(L, q, j, s)
            v = phases(a_neg, q)
            z = attention_output(v, j, T)
            u = readout(z, s, q)
            ok_neg = (u < tau)  # label = 0

            if ok_pos and ok_neg:
                n_correct += 1
    return n_correct / n_pairs


def run_row(row: dict, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    q, L = row["q"], row["L"]
    T_star = T_star_noiseless(L, q)
    T_grid = row["T_grid"]
    mode = row.get("table_mode", "random")

    out = []
    for f in T_grid:
        T = f * T_star
        tau = midpoint_threshold(L, q, T)

        if mode == "adversarial":
            acc = evaluate_adversarial(q, L, T, tau)
            accs = [acc]
        else:
            accs = []
            for _ in range(row["n_tables"]):
                a = random_table(L, q, rng)
                accs.append(evaluate_table(a, q, T, tau))
        accs = np.array(accs, dtype=float)
        out.append({
            "exp": "exp02", "seed": seed, "q": q, "L": L,
            "T": T, "T_star": T_star, "T_factor": f, "tau": tau,
            "mode": mode, "n_tables": len(accs),
            "acc_mean": float(accs.mean()), "acc_worst": float(accs.min()),
        })
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--row", type=int, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--out", default="out/exp02")
    args = p.parse_args()

    cfg = yaml.safe_load(open(args.config))["exp02_temperature_sweep"]
    row = cfg["rows"][args.row]
    records = run_row(row, args.seed)

    Path(args.out).mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(records)
    fname = f"row{args.row:02d}_seed{args.seed:02d}.parquet"
    df.to_parquet(os.path.join(args.out, fname), index=False)
    print(f"[exp02] wrote {len(df)} rows -> {args.out}/{fname}")


if __name__ == "__main__":
    main()