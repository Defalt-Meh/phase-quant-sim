"""
Aggregate parquet shards under out/exp*/ and produce paper figures.

Reads:  out/exp01/*.parquet, ..., out/exp12/*.parquet
Writes: out/plots/figureN_*.png

Each figure function is independent and skips silently if its source
data is missing.
"""
from __future__ import annotations
import os
from pathlib import Path
from glob import glob

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("out")
PLOTS = OUT / "plots"
PLOTS.mkdir(parents=True, exist_ok=True)


def load(exp: str) -> pd.DataFrame:
    """Concatenate every parquet shard for an experiment."""
    files = sorted(glob(str(OUT / exp / "*.parquet")))
    if not files:
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


# ---------- Figure 1: budget + temperature ----------
def figure1():
    df1 = load("exp01")
    df2 = load("exp02")
    if df1.empty and df2.empty:
        print("[plots] exp01/exp02 missing; skipping figure1")
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    if not df1.empty:
        df1r = df1[df1["mode"] == "random"]
        for m in sorted(df1r["m"].unique()):
            sub = df1r[df1r["m"] == m].groupby("W_over_L")["acc_mean"].mean()
            axes[0].plot(sub.index, sub.values, marker="o", label=f"m={m}")
        axes[0].axvline(1.0, ls="--", color="k", alpha=0.5)
        axes[0].set_xlabel("W / L")
        axes[0].set_ylabel("retrieval accuracy")
        axes[0].set_title("phase-budget threshold")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

    if not df2.empty:
        df2r = df2[df2["mode"] == "random"]
        for L in sorted(df2r["L"].unique()):
            sub = df2r[df2r["L"] == L].groupby("T_factor")["acc_mean"].mean()
            axes[1].plot(sub.index, sub.values, marker="o", label=f"L={L}")
        axes[1].axvline(1.0, ls="--", color="k", alpha=0.5)
        axes[1].set_xlabel("T / T*")
        axes[1].set_ylabel("retrieval accuracy")
        axes[1].set_title("temperature calibration")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(PLOTS / "figure1_budget_temperature.png", dpi=150)
    plt.close(fig)
    print("[plots] figure1 written")


# ---------- Figure 2: noise tolerance ----------
def figure2():
    df = load("exp03")
    if df.empty:
        print("[plots] exp03 missing; skipping figure2")
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    sub = df[df["noise_mode"] == "uniform"]
    for tf in sorted(sub["T_factor"].unique()):
        s2 = sub[sub["T_factor"] == tf].groupby("eps")["acc_mean"].mean()
        ax.plot(s2.index, s2.values, marker="o", label=f"T={tf:.2f}T*")
    eps_max = sub["eps_max"].iloc[0] if not sub.empty else np.pi / 4
    ax.axvline(eps_max, ls="--", color="r", alpha=0.6, label="ε=π/q")
    ax.set_xlabel("noise magnitude ε")
    ax.set_ylabel("retrieval accuracy")
    ax.set_title("phase-noise tolerance")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS / "figure2_noise.png", dpi=150)
    plt.close(fig)
    print("[plots] figure2 written")


# ---------- Figure 3: alphabet heatmap ----------
def figure3():
    df = load("exp04")
    if df.empty:
        print("[plots] exp04 missing; skipping figure3")
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    feasible = df[df["feasible"]]
    for q in sorted(feasible["q"].unique()):
        sub = feasible[(feasible["q"] == q) & (feasible["m"] == 64)]
        if sub.empty:
            continue
        s2 = sub.groupby("eps")["acc_mean"].mean()
        ax.plot(s2.index, s2.values, marker="o", label=f"q={q}")
    ax.set_xlabel("noise ε")
    ax.set_ylabel("retrieval accuracy")
    ax.set_title("alphabet comparison @ m=64")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS / "figure3_alphabet.png", dpi=150)
    plt.close(fig)

    # Heatmap: best q per (m, eps).
    fig, ax = plt.subplots(figsize=(8, 5))
    pivot = (feasible
             .groupby(["m", "eps", "q"])["acc_mean"].mean()
             .reset_index())
    perfect = pivot[pivot["acc_mean"] >= 0.999]
    if not perfect.empty:
        winner = perfect.loc[perfect.groupby(["m", "eps"])["q"].idxmin()]
        H = winner.pivot(index="m", columns="eps", values="q")
        im = ax.imshow(H.values, aspect="auto", origin="lower",
                       cmap="viridis")
        ax.set_xticks(range(len(H.columns)))
        ax.set_xticklabels([f"{e:.2f}" for e in H.columns], rotation=45)
        ax.set_yticks(range(len(H.index)))
        ax.set_yticklabels(H.index)
        ax.set_xlabel("ε"); ax.set_ylabel("m bits")
        ax.set_title("best q per (m, ε)")
        fig.colorbar(im, ax=ax, label="best q")
    fig.tight_layout()
    fig.savefig(PLOTS / "figure3b_alphabet_heatmap.png", dpi=150)
    plt.close(fig)
    print("[plots] figure3 written")


# ---------- Figure 4: quantized keys ----------
def figure4():
    df = load("exp05")
    if df.empty:
        print("[plots] exp05 missing; skipping figure4")
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    feasible = df[df["feasible"]]
    for cb in sorted(feasible["codebook"].unique()):
        for d in sorted(feasible[feasible["codebook"] == cb]["d"].unique()):
            sub = feasible[(feasible["codebook"] == cb) &
                           (feasible["d"] == d)]
            s2 = sub.groupby("T_over_upper")["acc_mean"].mean()
            ax.plot(s2.index, s2.values, marker=".",
                    label=f"{cb}, d={d}", alpha=0.7)
    ax.axvline(1.0, ls="--", color="k", alpha=0.5, label="T_upper")
    ax.set_xlabel("T / T_upper")
    ax.set_ylabel("retrieval accuracy")
    ax.set_title("quantized-key codebook comparison")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS / "figure4_quantized_keys.png", dpi=150)
    plt.close(fig)
    print("[plots] figure4 written")


# ---------- Auxiliary figures ----------
def figure_qk_gap():
    df = load("exp08")
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    for (q, L, d), sub in df.groupby(["q", "L", "d"]):
        s2 = sub.groupby("T")["acc_mean"].mean()
        ax.plot(s2.index, s2.values, marker=".",
                label=f"q={q},L={L},d={d}", alpha=0.7)
        T_lo = sub["T_lower"].iloc[0]
        T_hi = sub["T_upper"].iloc[0]
        ax.axvline(T_lo, color="r", ls=":", alpha=0.4)
        ax.axvline(T_hi, color="g", ls=":", alpha=0.4)
    ax.set_xlabel("T")
    ax.set_ylabel("retrieval accuracy")
    ax.set_title("qk gap: empirical vs upper (green) vs lower (red)")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS / "figure5_qk_gap.png", dpi=150)
    plt.close(fig)
    print("[plots] figure5 written")


def figure_sharpness():
    df = load("exp07")
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    for L in sorted(df["L"].unique()):
        sub = df[df["L"] == L].groupby("T_factor")["acc_mean"].mean()
        ax.plot(sub.index, sub.values, marker=".", label=f"L={L}", alpha=0.7)
    ax.axvline(1.0, ls="--", color="k", alpha=0.5)
    ax.set_xlabel("T / T*")
    ax.set_ylabel("retrieval accuracy")
    ax.set_title("transition sharpness vs L")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS / "figure6_sharpness.png", dpi=150)
    plt.close(fig)
    print("[plots] figure6 written")


def figure_realistic_noise():
    df = load("exp11")
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    for q in sorted(df["q"].unique()):
        for nm in sorted(df["noise_mode"].unique()):
            sub = df[(df["q"] == q) & (df["noise_mode"] == nm)]
            s2 = sub.groupby("T_factor")["acc_mean"].mean()
            ax.plot(s2.index, s2.values, marker=".",
                    label=f"q={q},{nm}", alpha=0.7)
    ax.axvline(1.0, ls="--", color="k", alpha=0.5)
    ax.set_xlabel("T / T*")
    ax.set_ylabel("retrieval accuracy")
    ax.set_title("realistic vs uniform noise")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS / "figure7_realistic_noise.png", dpi=150)
    plt.close(fig)
    print("[plots] figure7 written")


def main():
    figure1()
    figure2()
    figure3()
    figure4()
    figure_qk_gap()
    figure_sharpness()
    figure_realistic_noise()
    print(f"[plots] all figures in {PLOTS}/")


if __name__ == "__main__":
    main()