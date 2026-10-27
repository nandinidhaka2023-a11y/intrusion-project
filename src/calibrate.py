"""
calibrate.py

WHAT THIS FILE DOES:
Fixes the "risk is always 0% or 100%" problem.

WHY THE RISK LOOKED WRONG:
We train with focal loss (alpha = 0.85) AND oversample attack windows. Both
push the network to be very confident, so its probabilities are not honest:
when the model says "90%", far fewer than 90 out of 100 such windows are
really attacks. The Low / Medium / High bands are cut from that number, so
they inherit the error.

WHAT WE DO ABOUT IT (temperature scaling):
We divide the model's raw score (the logit, i.e. the value before the final
sigmoid) by one number T and then apply the sigmoid:

        calibrated = sigmoid(logit / T)

T > 1 softens over-confident scores, T < 1 sharpens timid ones. T is found by
minimising log-loss on data the model was not trained on. Because dividing by
a positive number never changes the order of the scores, PR-AUC does not move;
only the meaning of "80%" becomes more truthful.

HONEST EVALUATION:
The held-out hosts (the same 20% that lstm_model.py keeps for testing) are split
again BY HOST into two halves. T is fitted on half A and the improvement is
measured on half B, which the fit never saw.

HOW TO RUN (after lstm_model.py has produced models/lstm_model.pt):
    python src/calibrate.py
Output: models/calibration.json (read automatically by predictor.py) and
reports/reliability.png (a figure for the report).
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import minimize_scalar
from sklearn.model_selection import GroupShuffleSplit

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from lstm_model import IntrusionLSTM

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED = BASE_DIR / "data" / "processed"
MODEL_PATH = BASE_DIR / "models" / "lstm_model.pt"
OUT_JSON = BASE_DIR / "models" / "calibration.json"
OUT_PNG = BASE_DIR / "reports" / "reliability.png"


def collect_logits(model, X, mean, std, batch_size=512):
    """Run the model up to (not including) the final sigmoid."""
    logits = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            batch = (np.asarray(X[start:start + batch_size], dtype=np.float32) - mean) / std
            x = torch.from_numpy(batch.astype(np.float32))
            _, (hidden, _) = model.lstm(x)
            logits.append(model.head[:-1](hidden[-1]).squeeze(-1).numpy())
    return np.concatenate(logits)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))


def log_loss(logits, y, temperature):
    p = np.clip(sigmoid(logits / temperature), 1e-7, 1 - 1e-7)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def fit_temperature(logits, y):
    """Find T > 0 that minimises log-loss (search over log T)."""
    result = minimize_scalar(
        lambda log_t: log_loss(logits, y, float(np.exp(log_t))),
        bounds=(np.log(0.05), np.log(50.0)),
        method="bounded",
    )
    return float(np.exp(result.x))


def expected_calibration_error(prob, y, bins=10):
    """Weighted average gap between predicted probability and real attack rate."""
    edges = np.linspace(0, 1, bins + 1)
    ece, table = 0.0, []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (prob >= lo) & ((prob < hi) if hi < 1 else (prob <= hi))
        if not mask.any():
            continue
        confidence = float(prob[mask].mean())
        observed = float(y[mask].mean())
        ece += mask.mean() * abs(confidence - observed)
        table.append((lo, hi, int(mask.sum()), confidence, observed))
    return float(ece), table


def brier(prob, y):
    return float(np.mean((prob - y) ** 2))


def main():
    X = np.load(PROCESSED / "sequences_X.npy", mmap_mode="r")
    y = np.load(PROCESSED / "sequences_y.npy").astype(np.float64)
    ips = np.load(PROCESSED / "sequences_ip.npy", allow_pickle=True)

    checkpoint = torch.load(MODEL_PATH, map_location="cpu")
    mean = np.asarray(checkpoint["mean"]).reshape(-1)
    std = np.asarray(checkpoint["std"]).reshape(-1)
    model = IntrusionLSTM(n_features=checkpoint["n_features"], hidden_size=64)
    model.load_state_dict(checkpoint["model_state"])

    # Same hold-out as lstm_model.train_lstm (same seed, same groups).
    outer = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    _, held_out = next(outer.split(np.zeros(len(y)), y, groups=ips))

    # Split the held-out hosts again: half to fit T, half to evaluate it.
    inner = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=7)
    fit_rel, eval_rel = next(inner.split(held_out, y[held_out], groups=ips[held_out]))
    fit_idx, eval_idx = held_out[fit_rel], held_out[eval_rel]

    for name, idx in (("fit", fit_idx), ("eval", eval_idx)):
        pos = int(y[idx].sum())
        print(f"{name:>4} half: {len(idx):,} windows, {pos:,} attacks, "
              f"{len(set(ips[idx]))} hosts")
        if pos == 0 or pos == len(idx):
            sys.exit(f"The {name} half contains only one class - cannot calibrate. "
                     "Use more hosts / data.")

    fit_logits = collect_logits(model, X[np.sort(fit_idx)], mean, std)
    y_fit = y[np.sort(fit_idx)]
    eval_logits = collect_logits(model, X[np.sort(eval_idx)], mean, std)
    y_eval = y[np.sort(eval_idx)]

    temperature = fit_temperature(fit_logits, y_fit)

    before = sigmoid(eval_logits)
    after = sigmoid(eval_logits / temperature)
    ece_b, table_b = expected_calibration_error(before, y_eval)
    ece_a, table_a = expected_calibration_error(after, y_eval)

    print(f"\nFitted temperature T = {temperature:.3f}  "
          f"({'softens over-confident scores' if temperature > 1 else 'sharpens scores'})")
    print(f"{'':<22}{'before':>10}{'after':>10}")
    print(f"{'Expected calib. error':<22}{ece_b:>10.4f}{ece_a:>10.4f}")
    print(f"{'Brier score':<22}{brier(before, y_eval):>10.4f}{brier(after, y_eval):>10.4f}")
    print(f"{'Log-loss':<22}{log_loss(eval_logits, y_eval, 1.0):>10.4f}"
          f"{log_loss(eval_logits, y_eval, temperature):>10.4f}")
    medium_b = float(np.mean((before >= 0.5) & (before < 0.8)))
    medium_a = float(np.mean((after >= 0.5) & (after < 0.8)))
    print(f"{'Share in MEDIUM band':<22}{medium_b:>10.2%}{medium_a:>10.2%}")
    print("(PR-AUC is unchanged by calibration: the ranking of windows is identical.)")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps({
        "temperature": temperature,
        "method": "temperature scaling on held-out hosts",
        "fit_windows": int(len(fit_idx)),
        "eval_windows": int(len(eval_idx)),
        "ece_before": ece_b, "ece_after": ece_a,
        "brier_before": brier(before, y_eval), "brier_after": brier(after, y_eval),
    }, indent=2))
    print(f"\nSaved {OUT_JSON}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(5.2, 4.4))
        ax.plot([0, 1], [0, 1], "--", color="gray", label="perfect calibration")
        for table, label, color in ((table_b, "before", "#d9534f"),
                                    (table_a, "after", "#2e7d32")):
            ax.plot([r[3] for r in table], [r[4] for r in table], "o-",
                    color=color, label=label)
        ax.set_xlabel("predicted probability (bin average)")
        ax.set_ylabel("observed share of real attacks")
        ax.set_title(f"Reliability (held-out hosts), T = {temperature:.2f}")
        ax.legend()
        ax.grid(alpha=0.3)
        OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(OUT_PNG, dpi=200)
        print(f"Saved {OUT_PNG}  (use it as a figure in the report)")
    except Exception as err:  # plotting must never block calibration
        print("Could not draw the reliability plot:", err)


if __name__ == "__main__":
    main()
