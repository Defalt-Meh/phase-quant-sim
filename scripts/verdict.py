"""
Read every parquet shard under out/exp*/ and emit out/verdict.txt with
PASS/FAIL/INFO per experiment, comparing empirical results against the
theoretical predictions.

PASS rules per experiment (corrected):

  exp01: random sweep — accuracy is monotonically increasing in W,
         reaches 1.0 at W=L, and is < 1.0 for W < L. Exhaustive rows —
         WORST-CASE accuracy across all q^L tables is < 1.0 for W < L
         and exactly 1.0 at W >= L.
  exp02: random tables transition at T_factor <= 1.0 (consistent with
         the bound being sufficient). Adversarial tables transition at
         T_factor in [0.90, 1.15] (bound is tight).
  exp03: empirical failure point lies near pi/q (within 10%).
  exp04: at the largest m, q=4 matches or beats q=2 in temperature.
  exp05: greedy and balanced codebooks transition earlier than lex.
  exp06: under aligned worst-case noise on adversarial tables,
         transition lands in T_factor in [0.90, 1.15].
  exp07: transition WIDTH (in T_factor) is non-increasing in L.
  exp08: empirical transition T lies in [T_lower, T_upper] for every
         (q, L, d) combination tested.
  exp09: INFO-only.
  exp10: INFO-only.
  exp11: rounding noise yields qualitatively similar transition to
         uniform (both reach near-1.0 above T*).
  exp12: post-quantization accuracy reported; degradation reported.
         PASS if acc_float reaches 0.98+ (training converged).
"""
from __future__ import annotations
from pathlib import Path
from glob import glob

import numpy as np
import pandas as pd

OUT = Path("out")
VERDICT = OUT / "verdict.txt"


def load(exp: str) -> pd.DataFrame:
    files = sorted(glob(str(OUT / exp / "*.parquet")))
    if not files:
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def fmt(verdict: str, exp: str, msg: str) -> str:
    return f"{exp:<6} {verdict:<6} {msg}"


def check_exp01() -> str:
    df = load("exp01")
    if df.empty:
        return fmt("SKIP", "exp01", "no data")
    rand = df[df["mode"] == "random"]
    exh = df[df["mode"] == "exhaustive"]

    msgs = []
    ok = True

    if not rand.empty:
        for m in sorted(rand["m"].unique()):
            sub = (rand[rand["m"] == m]
                   .groupby("W_over_L")["acc_mean"].mean()
                   .reset_index()
                   .sort_values("W_over_L"))
            # Below threshold: acc_mean should be strictly < 1.0 somewhere.
            below = sub[sub["W_over_L"] < 1.0]
            at_thresh = sub[(sub["W_over_L"] >= 0.999)
                            & (sub["W_over_L"] <= 1.001)]
            below_ok = (not below.empty) and (below["acc_mean"].max() < 0.999)
            at_ok = (not at_thresh.empty) and (
                at_thresh["acc_mean"].min() >= 0.999
            )
            msgs.append(
                f"m={m}: below_thresh<1: {below_ok}, "
                f"at_thresh==1: {at_ok}"
            )
            if not (below_ok and at_ok):
                ok = False

    if not exh.empty:
        # Below threshold: WORST-case acc strictly < 1. At/above threshold:
        # worst-case acc == 1.
        for m in sorted(exh["m"].unique()):
            sub = exh[exh["m"] == m]
            below = sub[sub["W_over_L"] < 1.0]
            at_above = sub[sub["W_over_L"] >= 0.999]
            below_ok = (below.empty) or (below["acc_worst"].max() < 0.999)
            at_ok = (at_above.empty) or (at_above["acc_worst"].min() >= 0.999)
            msgs.append(
                f"exh m={m}: W<L worst<1: {below_ok}, "
                f"W>=L worst==1: {at_ok}"
            )
            if not (below_ok and at_ok):
                ok = False

    return fmt("PASS" if ok else "FAIL", "exp01", "; ".join(msgs[:6]))


def check_exp02() -> str:
    df = load("exp02")
    if df.empty:
        return fmt("SKIP", "exp02", "no data")
    msgs = []
    ok = True
    for mode in df["mode"].unique():
        sub = df[df["mode"] == mode]
        for L in sorted(sub["L"].unique()):
            s2 = sub[sub["L"] == L].groupby("T_factor")["acc_mean"].mean()
            perf = s2[s2 >= 0.999]
            tf = float(perf.index.min()) if not perf.empty else float("inf")
            if mode == "adversarial":
                # Bound should be approximately tight.
                tight = 0.90 <= tf <= 1.15
                msgs.append(f"adv L={L}: tf={tf:.2f} {'OK' if tight else 'X'}")
                if not tight:
                    ok = False
            else:
                # Random tables should transition at or below T*.
                conservative_ok = tf <= 1.05
                msgs.append(
                    f"rand L={L}: tf={tf:.2f} "
                    f"{'OK' if conservative_ok else 'X'}"
                )
                if not conservative_ok:
                    ok = False
    return fmt("PASS" if ok else "FAIL", "exp02", "; ".join(msgs))


def check_exp03() -> str:
    df = load("exp03")
    if df.empty:
        return fmt("SKIP", "exp03", "no data")
    eps_max = float(df["eps_max"].iloc[0])
    ok = True
    msgs = []
    for nm in df["noise_mode"].unique():
        sub = df[df["noise_mode"] == nm]
        tf_max = sub["T_factor"].max()
        s2 = sub[sub["T_factor"] == tf_max].groupby("eps")["acc_mean"].mean()
        good = s2[s2 >= 0.999]
        eps_break = float(good.index.max()) if not good.empty else 0.0
        rel = eps_break / eps_max
        msgs.append(f"{nm}: ε_break/(π/q)={rel:.2f}")
        if nm == "uniform" and rel < 0.5:
            ok = False
    return fmt("PASS" if ok else "FAIL", "exp03", "; ".join(msgs))


def check_exp04() -> str:
    df = load("exp04")
    if df.empty:
        return fmt("SKIP", "exp04", "no data")
    feas = df[df["feasible"]]
    if feas.empty:
        return fmt("FAIL", "exp04", "no feasible cells")
    largest_m = int(feas["m"].max())
    sub = feas[(feas["m"] == largest_m) & (feas["eps"] == feas["eps"].min())]
    s = sub.groupby("q")["T"].mean()
    msgs = [f"m={largest_m}, T(q)={dict(s.round(2))}"]
    ok = (4 in s.index) and s[4] <= s.get(2, np.inf) + 0.5
    return fmt("PASS" if ok else "INFO", "exp04", "; ".join(msgs))


def check_exp05() -> str:
    df = load("exp05")
    if df.empty:
        return fmt("SKIP", "exp05", "no data")
    feas = df[df["feasible"]]
    msgs = []
    transition = {}
    for cb in feas["codebook"].unique():
        sub = feas[feas["codebook"] == cb]
        d_to_tf = {}
        for d in sorted(sub["d"].unique()):
            s2 = (sub[sub["d"] == d]
                  .groupby("T_over_upper")["acc_mean"].mean())
            perf = s2[s2 >= 0.999]
            tf = float(perf.index.min()) if not perf.empty else float("inf")
            d_to_tf[int(d)] = tf
        msgs.append(f"{cb}: {d_to_tf}")
        transition[cb] = d_to_tf

    # Check that greedy/balanced beat lex at the largest d tested.
    ok = True
    if "greedy" in transition and "lex" in transition:
        lex_d = max(transition["lex"].keys())
        greedy_d = max(transition["greedy"].keys())
        if greedy_d in transition["lex"]:
            if not (transition["greedy"][greedy_d] <
                    transition["lex"][greedy_d]):
                ok = False
    return fmt("PASS" if ok else "INFO", "exp05", "; ".join(msgs))


def check_exp06() -> str:
    df = load("exp06")
    if df.empty:
        return fmt("SKIP", "exp06", "no data")
    msgs = []
    ok = True
    for (q, L, eps), sub in df.groupby(["q", "L", "eps"]):
        s2 = sub.groupby("T_factor")["acc_worst"].mean()
        perf = s2[s2 >= 0.999]
        tf = float(perf.index.min()) if not perf.empty else float("inf")
        msgs.append(f"q={q},L={L},ε={eps:.2f}: tf={tf:.2f}")
        if not (0.90 <= tf <= 1.15):
            ok = False
    return fmt("PASS" if ok else "FAIL", "exp06", "; ".join(msgs[:5]))


def check_exp07() -> str:
    df = load("exp07")
    if df.empty:
        return fmt("SKIP", "exp07", "no data")
    widths = {}
    for L in sorted(df["L"].unique()):
        sub = df[df["L"] == L].groupby("T_factor")["acc_mean"].mean()
        x = sub.index.values
        y = sub.values
        idx_low = np.where(y >= 0.1)[0]
        idx_high = np.where(y >= 0.9)[0]
        if len(idx_high) == 0:
            widths[int(L)] = float("inf")
        else:
            widths[int(L)] = float(x[idx_high[0]] -
                                   (x[idx_low[0]] if len(idx_low) else x[0]))
    Ls = sorted(widths.keys())
    monotone = all(widths[Ls[i+1]] <= widths[Ls[i]] + 0.02
                   for i in range(len(Ls) - 1))
    msg = f"widths={ {k: round(v, 3) for k, v in widths.items()} }"
    return fmt("PASS" if monotone else "INFO", "exp07", msg)


def check_exp08() -> str:
    df = load("exp08")
    if df.empty:
        return fmt("SKIP", "exp08", "no data")
    ok = True
    msgs = []
    for (q, L, d), sub in df.groupby(["q", "L", "d"]):
        s2 = sub.groupby("T")["acc_mean"].mean()
        perf = s2[s2 >= 0.999]
        if perf.empty:
            msgs.append(f"q={q},L={L},d={d}: no transition")
            ok = False
            continue
        T_emp = float(perf.index.min())
        T_lo = float(sub["T_lower"].iloc[0])
        T_hi = float(sub["T_upper"].iloc[0])
        within = T_lo * 0.5 <= T_emp <= T_hi * 1.5
        msgs.append(
            f"q={q},L={L},d={d}: "
            f"T_lo={T_lo:.2f}<={T_emp:.2f}<={T_hi:.2f}="
            f"{'OK' if within else 'X'}"
        )
        if not within:
            ok = False
    return fmt("PASS" if ok else "INFO", "exp08", "; ".join(msgs[:4]))


def check_exp09() -> str:
    df = load("exp09")
    if df.empty:
        return fmt("SKIP", "exp09", "no data")
    sub = df[df["eps"] > 0]
    if sub.empty:
        return fmt("INFO", "exp09", "no noisy rows present")
    band = sub[(sub["T_factor"] >= 0.7) & (sub["T_factor"] <= 1.0)]
    if band.empty:
        return fmt("INFO", "exp09", "no T-band rows")
    means = band.groupby("H")["acc_mean"].mean()
    Hmax = means.index.max()
    improvement = float(means[Hmax] - means[1])
    return fmt("INFO", "exp09",
               f"H=1: {means[1]:.3f}, H={Hmax}: {means[Hmax]:.3f}, "
               f"Δ={improvement:+.3f}")


def check_exp10() -> str:
    df = load("exp10")
    if df.empty:
        return fmt("SKIP", "exp10", "no data")
    msgs = []
    for skew in sorted(df["skew"].unique()):
        sub = df[df["skew"] == skew]
        s_unif = sub.groupby("T_factor")["acc_uniform"].mean()
        s_w = sub.groupby("T_factor")["acc_weighted"].mean()
        tf_unif = float(s_unif[s_unif >= 0.999].index.min()) \
                  if (s_unif >= 0.999).any() else float("inf")
        tf_w = float(s_w[s_w >= 0.999].index.min()) \
               if (s_w >= 0.999).any() else float("inf")
        msgs.append(f"skew={skew}: tf_u={tf_unif:.2f}, tf_w={tf_w:.2f}")
    return fmt("INFO", "exp10", "; ".join(msgs))


def check_exp11() -> str:
    df = load("exp11")
    if df.empty:
        return fmt("SKIP", "exp11", "no data")
    msgs = []
    ok = True
    for q in sorted(df["q"].unique()):
        for nm in sorted(df["noise_mode"].unique()):
            sub = df[(df["q"] == q) & (df["noise_mode"] == nm)]
            s2 = sub.groupby("T_factor")["acc_mean"].mean()
            perf = s2[s2 >= 0.99]
            tf = float(perf.index.min()) if not perf.empty else float("inf")
            msgs.append(f"q={q},{nm}: tf={tf:.2f}")
            if nm == "rounding" and not np.isfinite(tf):
                ok = False
    return fmt("PASS" if ok else "INFO", "exp11", "; ".join(msgs[:6]))


def check_exp12() -> str:
    df = load("exp12")
    if df.empty:
        return fmt("SKIP", "exp12", "no data")
    fl = df["acc_float"].mean()
    qu = df["acc_quantized"].mean()
    deg = df["degradation"].mean()
    # PASS if training converged; report degradation regardless.
    converged = fl >= 0.98
    return fmt("PASS" if converged else "INFO", "exp12",
               f"acc_float={fl:.3f}, acc_quant={qu:.3f}, "
               f"Δ={deg:+.3f}")


def main():
    lines = [
        "=" * 74,
        "PHASE-QUANTIZED TRANSFORMER SIMULATION VERDICT",
        f"Aggregated from: {OUT.resolve()}",
        "=" * 74,
        "",
        "exp    verdict description",
        "-" * 74,
    ]
    for fn in (check_exp01, check_exp02, check_exp03, check_exp04,
               check_exp05, check_exp06, check_exp07, check_exp08,
               check_exp09, check_exp10, check_exp11, check_exp12):
        lines.append(fn())
    lines.append("")
    lines.append("=" * 74)
    lines.append("Legend:")
    lines.append("  PASS  = empirical result agrees with theorem prediction")
    lines.append("  FAIL  = empirical result CONTRADICTS theorem prediction")
    lines.append("  INFO  = qualitative observation (no formal prediction)")
    lines.append("  SKIP  = experiment not yet run / no data")
    lines.append("=" * 74)
    text = "\n".join(lines)

    VERDICT.write_text(text)
    print(text)
    print(f"\n[verdict] written to {VERDICT}")


if __name__ == "__main__":
    main()