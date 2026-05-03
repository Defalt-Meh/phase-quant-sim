"""
Experiment 1: Phase-budget threshold (paper §5.1, left panel) +
exhaustive small-L impossibility check (reviewer fix 3.1).

For each (m, q): sweep W from W_min..W_max. For W < L only W slots are
trainable (others fixed to v_r = 1, i.e. a_r = 0). For W >= L all slots
are trainable.

Records, per (m, q, W, seed): retrieval accuracy averaged over n_tables
random tables. For exhaustive=true, enumerate ALL q^L tables and report
worst-case accuracy across the family.

CRITICAL: the LABEL is always computed against the original random
table a, not against the template's effective (forced) table. The
template can only represent the original table when W >= L; for W < L
the untrainable slots produce mismatched predictions on whatever
inputs disagree with the forced value, which is exactly the behavior
the counting bound predicts.
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from core.template import phases, attention_output, readout, midpoint_threshold
from core.tables import random_table, all_tables
from core.thresholds import L_of, T_star_noiseless


def evaluate_table(
    a: np.ndarray, q: int, T: float, tau: float, trainable_mask: np.ndarray
) -> float:
    """
    Evaluate accuracy on all (j, s) inputs against the TARGET function f_a.

    The template's effective table sets a_r to the original value at
    trainable slots and to 0 (i.e., v_r = 1) at untrainable slots.
    Labels are computed from the ORIGINAL a, so untrainable slots whose
    a_r != 0 will produce mismatched predictions, exactly as the
    counting bound predicts.
    """
    L = a.shape[0]
    a_template = np.where(trainable_mask, a, 0)
    v = phases(a_template, q)
    correct = 0
    total = L * q
    for j in range(L):
        for s in range(q):
            z = attention_output(v, j, T)
            u = readout(z, s, q)
            pred = int(u >= tau)
            label = int(a[j] == s)  # ORIGINAL table, not a_template.
            correct += int(pred == label)
    return correct / total


def run_row(row: dict, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    m = row["m"]
    q = row["q"]
    L = L_of(m, q)
    T = row["T_factor"] * T_star_noiseless(L, q) if L > 1 else 1.0
    tau = midpoint_threshold(L, q, T)

    out = []
    Wmin, Wmax = row["W_min"], row["W_max"]

    for W in range(Wmin, Wmax + 1):
        trainable = np.zeros(L, dtype=bool)
        idx = rng.choice(L, size=min(W, L), replace=False)
        trainable[idx] = True

        if row.get("exhaustive", False):
            worst = 1.0
            mean_acc = 0.0
            n = 0
            for a in all_tables(L, q):
                acc = evaluate_table(a, q, T, tau, trainable)
                worst = min(worst, acc)
                mean_acc += acc
                n += 1
            mean_acc /= max(n, 1)
            out.append({
                "exp": "exp01", "seed": seed, "m": m, "q": q, "L": L,
                "W": W, "W_over_L": W / L, "T": T, "tau": tau,
                "n_tables": n, "mode": "exhaustive",
                "acc_mean": mean_acc, "acc_worst": worst,
            })
        else:
            accs = []
            for _ in range(row["n_tables"]):
                a = random_table(L, q, rng)
                accs.append(evaluate_table(a, q, T, tau, trainable))
            accs = np.array(accs)
            out.append({
                "exp": "exp01", "seed": seed, "m": m, "q": q, "L": L,
                "W": W, "W_over_L": W / L, "T": T, "tau": tau,
                "n_tables": row["n_tables"], "mode": "random",
                "acc_mean": float(accs.mean()), "acc_worst": float(accs.min()),
            })
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--row", type=int, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--out", default="out/exp01")
    args = p.parse_args()

    cfg = yaml.safe_load(open(args.config))["exp01_budget_sweep"]
    row = cfg["rows"][args.row]
    records = run_row(row, args.seed)

    Path(args.out).mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(records)
    fname = f"row{args.row:02d}_seed{args.seed:02d}.parquet"
    df.to_parquet(os.path.join(args.out, fname), index=False)
    print(f"[exp01] wrote {len(df)} rows -> {args.out}/{fname}")


if __name__ == "__main__":
    main()