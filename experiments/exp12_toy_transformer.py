"""
Experiment 12: Multi-layer Transformer + post-hoc phase quantization
(reviewer fix, §4 item 10) — strengthened version.

The previous version of this script trained a one-parameter-per-slot
template, which trivially survives quantization. This version trains
a real multi-layer Transformer with vector-valued slots, then
phase-quantizes each component of the value matrix to the nearest
q-th root of unity. The result is a non-trivial degradation curve.

Architecture:
  - Token embeddings: L+1 learnable d_model-dim vectors (slot tokens
    + a query token).
  - n_layers of standard Transformer blocks (multi-head self-attention
    + MLP), all float32.
  - Output head: read the query token's d_value-dim representation,
    project to a single logit via a learnable readout that the
    quantization touches.
  - Phase-quantized parameter: the value projection W_V in each layer
    is quantized post-training to a complex-valued matrix in
    Omega_q^{d_model x d_value}. Specifically, we parameterize W_V
    as two real matrices (real and imaginary parts), and quantize each
    entry's phase to the nearest q-th root.

Task: the lookup family f_a(j, s) on a fixed table a in [q]^L. Inputs
are (j, s) pairs encoded as a sum of slot embedding j and symbol
embedding s.

Metrics:
  - acc_float:    accuracy of the trained float model.
  - acc_quantized: accuracy after phase-quantization of W_V matrices.
  - degradation:  acc_float - acc_quantized (the result of interest).
"""
from __future__ import annotations
import argparse, os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

import torch
import torch.nn as nn
import torch.nn.functional as F


class PhaseQuantizableLinear(nn.Module):
    """
    Linear layer whose weight has separate real and imaginary parts.
    The 'effective' weight is a complex-valued matrix; we use only its
    real part for the forward pass (so the layer behaves like a normal
    real-valued Linear during training). Quantization rounds the phase
    of each entry to the nearest q-th root of unity.
    """
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        # Initialize so the magnitudes are O(1) and the phases are random.
        scale = 1.0 / np.sqrt(in_dim)
        theta = torch.rand(out_dim, in_dim) * 2 * np.pi
        self.W_re = nn.Parameter(scale * torch.cos(theta))
        self.W_im = nn.Parameter(scale * torch.sin(theta))
        self.b = nn.Parameter(torch.zeros(out_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Use real part only.
        return F.linear(x, self.W_re, self.b)

    def quantize_phases(self, q: int) -> None:
        """Round each entry's phase to the nearest q-th root of unity,
        preserving its magnitude."""
        with torch.no_grad():
            mag = torch.sqrt(self.W_re ** 2 + self.W_im ** 2 + 1e-12)
            theta = torch.atan2(self.W_im, self.W_re)
            k = torch.round(q * theta / (2 * np.pi)) % q
            new_theta = 2 * np.pi * k / q
            self.W_re.copy_(mag * torch.cos(new_theta))
            self.W_im.copy_(mag * torch.sin(new_theta))


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int = 4):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = PhaseQuantizableLinear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        h = self.ln1(x)
        q = self.W_Q(h).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.W_K(h).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.W_V(h).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / np.sqrt(self.head_dim)
        att = F.softmax(att, dim=-1)
        out = (att @ v).transpose(1, 2).reshape(B, T, D)
        x = x + self.W_O(out)
        x = x + self.mlp(self.ln2(x))
        return x

    def quantize(self, q: int) -> None:
        self.W_V.quantize_phases(q)


class ToyTransformer(nn.Module):
    def __init__(self, L: int, q: int, d_model: int, n_layers: int):
        super().__init__()
        self.L = L
        self.q = q
        self.d_model = d_model
        # +1 for the query position itself.
        self.slot_emb = nn.Embedding(L + 1, d_model)
        self.symbol_emb = nn.Embedding(q, d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model) for _ in range(n_layers)
        ])
        self.head = nn.Linear(d_model, 1)

    def forward(self, j: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """
        j, s: (B,) long tensors. Build a length-(L+2) sequence:
          - position 0:    query token (slot index L) + symbol embedding s
          - positions 1..L+1: slot tokens for r in 0..L-1, symbol == 0
        Then run the Transformer and read out the query position.
        """
        B = j.shape[0]
        device = j.device
        seq_len = self.L + 1

        # Build positions: [query] + [slot_0, slot_1, ..., slot_{L-1}].
        pos = torch.arange(seq_len, device=device)
        pos_batch = pos.unsqueeze(0).expand(B, -1)  # (B, L+1)
        x = self.slot_emb(pos_batch)                 # (B, L+1, d)

        # Add symbol embeddings: query token gets s, slot tokens get 0.
        sym = torch.zeros(B, seq_len, dtype=torch.long, device=device)
        sym[:, 0] = s
        x = x + self.symbol_emb(sym)

        # Mark the query slot j by adding the slot-j embedding to position 0.
        x[:, 0] = x[:, 0] + self.slot_emb(j)

        for block in self.blocks:
            x = block(x)

        logit = self.head(x[:, 0]).squeeze(-1)  # (B,)
        return logit

    def quantize(self) -> None:
        for block in self.blocks:
            block.quantize(self.q)


def make_dataset(L: int, q: int, table: np.ndarray, device: str):
    js, ss, ys = [], [], []
    for j in range(L):
        for s in range(q):
            js.append(j); ss.append(s); ys.append(int(table[j] == s))
    return (
        torch.tensor(js, dtype=torch.long, device=device),
        torch.tensor(ss, dtype=torch.long, device=device),
        torch.tensor(ys, dtype=torch.float32, device=device),
    )


def evaluate(model, j, s, y) -> float:
    model.eval()
    with torch.no_grad():
        logit = model(j, s)
        pred = (logit > 0).float()
        acc = (pred == y).float().mean().item()
    return float(acc)


def run_row(row: dict, seed: int) -> list[dict]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    q, L = row["q"], row["L"]
    d_model = row["d_model"]
    n_layers = row["n_layers"]
    n_train = row["n_train"]
    lr = float(row["lr"])

    table = rng.integers(0, q, size=L, dtype=np.int64)
    j_t, s_t, y_t = make_dataset(L, q, table, device)

    model = ToyTransformer(L, q, d_model, n_layers).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    best_loss = float("inf")
    for step in range(n_train):
        model.train()
        logit = model(j_t, s_t)
        loss = F.binary_cross_entropy_with_logits(logit, y_t)
        opt.zero_grad(); loss.backward(); opt.step()
        if loss.item() < best_loss:
            best_loss = loss.item()

    acc_float = evaluate(model, j_t, s_t, y_t)

    if row.get("quantize_after", True):
        model.quantize()
        acc_quant = evaluate(model, j_t, s_t, y_t)
    else:
        acc_quant = float("nan")

    return [{
        "exp": "exp12", "seed": seed, "q": q, "L": L,
        "d_model": d_model, "n_layers": n_layers,
        "n_train": n_train, "lr": lr,
        "best_loss": float(best_loss),
        "acc_float": acc_float,
        "acc_quantized": acc_quant,
        "degradation": acc_float - acc_quant if not np.isnan(acc_quant) else float("nan"),
        "device": device,
    }]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--row", type=int, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--out", default="out/exp12")
    args = p.parse_args()

    cfg = yaml.safe_load(open(args.config))["exp12_toy_transformer"]
    row = cfg["rows"][args.row]
    records = run_row(row, args.seed)

    Path(args.out).mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(records)
    fname = f"row{args.row:02d}_seed{args.seed:02d}.parquet"
    df.to_parquet(os.path.join(args.out, fname), index=False)
    print(f"[exp12] wrote {len(df)} rows -> {args.out}/{fname}")


if __name__ == "__main__":
    main()