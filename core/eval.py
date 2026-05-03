"""
Evaluation helpers: accuracy, margins, transition-width estimation.
"""

from __future__ import annotations
import numpy as np


def accuracy(predictions: np.ndarray, labels: np.ndarray) -> float:
    """Fraction of correct predictions."""
    predictions = np.asarray(predictions)
    labels = np.asarray(labels)
    return float((predictions == labels).mean())


def empirical_margin(u_plus_min: float, u_minus_max: float) -> float:
    """
    Empirical margin = (min positive readout) - (max negative readout).
    Positive value means the threshold can separate cleanly.
    """
    return float(u_plus_min - u_minus_max)


def transition_point(
    xs: np.ndarray, accs: np.ndarray, target: float = 1.0, tol: float = 1e-9
) -> float:
    """
    Smallest x in xs such that accs[x] >= target - tol.
    Returns +inf if never reached.
    """
    xs = np.asarray(xs, dtype=float)
    accs = np.asarray(accs, dtype=float)
    order = np.argsort(xs)
    xs = xs[order]
    accs = accs[order]
    mask = accs >= (target - tol)
    if not mask.any():
        return float("inf")
    return float(xs[mask][0])


def transition_width(
    xs: np.ndarray, accs: np.ndarray, low: float = 0.1, high: float = 0.9
) -> float:
    """
    Empirical transition width: x at which accs first crosses `high`,
    minus x at which accs first crosses `low`. Used in exp07.
    Returns +inf if `high` is never reached.
    """
    xs = np.asarray(xs, dtype=float)
    accs = np.asarray(accs, dtype=float)
    order = np.argsort(xs)
    xs = xs[order]
    accs = accs[order]

    above_low = accs >= low
    above_high = accs >= high
    if not above_high.any():
        return float("inf")
    x_low = float(xs[above_low][0]) if above_low.any() else float(xs[0])
    x_high = float(xs[above_high][0])
    return x_high - x_low


def summarize_run(records: list[dict]) -> dict:
    """
    Aggregate a list of per-input records into summary statistics.
    Each record should contain at minimum 'pred', 'label', 'u'.
    """
    preds = np.array([r["pred"] for r in records])
    labels = np.array([r["label"] for r in records])
    us = np.array([r["u"] for r in records])

    pos_mask = labels == 1
    neg_mask = labels == 0

    summary = {
        "accuracy": accuracy(preds, labels),
        "n": len(records),
        "n_pos": int(pos_mask.sum()),
        "n_neg": int(neg_mask.sum()),
        "u_pos_min": float(us[pos_mask].min()) if pos_mask.any() else float("nan"),
        "u_pos_mean": float(us[pos_mask].mean()) if pos_mask.any() else float("nan"),
        "u_neg_max": float(us[neg_mask].max()) if neg_mask.any() else float("nan"),
        "u_neg_mean": float(us[neg_mask].mean()) if neg_mask.any() else float("nan"),
    }
    summary["empirical_margin"] = empirical_margin(
        summary["u_pos_min"], summary["u_neg_max"]
    ) if pos_mask.any() and neg_mask.any() else float("nan")
    return summary