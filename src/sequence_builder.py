"""
sequence_builder.py

WHAT THIS FILE DOES:
Converts the flat table of flows into SEQUENCES of flows, grouped by
attacker IP, so the LSTM model can learn patterns that build up over
several flows in a row — not just judge one flow in isolation.

WHY THIS STEP EXISTS (the core idea of the whole DL half of the project):
A single random forest looks at one flow at a time and asks "does this
one flow look like an attack?" An LSTM instead asks "does this SEQUENCE
of flows, in order, look like an attack building up?" — e.g. a port scan
isn't one suspicious packet, it's dozens of small probe connections in a
row. That pattern only exists across multiple flows, which is exactly
what a sequence model can see and a single-row model can't.

IMPORTANT ASSUMPTION (documented on purpose — mention this in your report):
This CICIDS2017 release strips the real Timestamp column. We verified
that row order still reflects original capture order (attack rows appear
clustered together, not scattered randomly — see the check we ran before
writing this file). We use row order within each simulated attacker IP as
a stand-in for time order. This is a reasonable assumption for this
project, but a production system would use real timestamps.
"""

import pandas as pd
import numpy as np
from pathlib import Path

PROCESSED_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"

NON_FEATURE_COLS = ["Label", "attack_type", "is_attack", "src_ip", "dst_ip", "source_file"]


def build_sequences(df: pd.DataFrame, window_size: int = 10, step: int = 5):
    """
    Groups flows by src_ip (in original row order), then slides a window
    of `window_size` consecutive flows across each IP's flows to create
    training sequences.

    WHY GROUP BY IP: an attack "builds up" per attacker, not across
    unrelated machines. Mixing flows from different IPs into one sequence
    would create a meaningless jumble with no real temporal pattern.

    WHY A SLIDING WINDOW (not just chopping into non-overlapping chunks):
    `step` < `window_size` means windows overlap, which gives us more
    training sequences from the same data — useful since we don't have a
    huge amount of data yet (single file, single attack type).

    LABEL FOR EACH SEQUENCE: we label a sequence as "attack" if the LAST
    flow in the window is an attack — this mirrors the real use case:
    "given what's happened in the last N flows, is an attack happening
    right now / about to happen."

    Returns:
        X: np.array of shape (num_sequences, window_size, num_features)
        y: np.array of shape (num_sequences,) — 1 if attack, 0 if benign
        ip_labels: which src_ip each sequence came from (useful for the
                   dashboard demo later — replay sequences per IP)
    """
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]

    sequences = []
    labels = []
    ip_labels = []

    for ip, group in df.groupby("src_ip", sort=False):
        # group.index is already in original row order because we didn't
        # shuffle/sort — this preserves our time-proxy assumption
        features = group[feature_cols].to_numpy(dtype=np.float32)
        is_attack = group["is_attack"].to_numpy()

        n_rows = len(group)
        if n_rows < window_size:
            continue  # not enough flows from this IP to build one window

        for start in range(0, n_rows - window_size + 1, step):
            end = start + window_size
            window_features = features[start:end]
            window_label = is_attack[end - 1]  # label = last flow in window

            sequences.append(window_features)
            labels.append(window_label)
            ip_labels.append(ip)

    X = np.stack(sequences)
    y = np.array(labels)

    print(f"Built {len(X):,} sequences of shape (window_size={window_size}, features={X.shape[2]})")
    print(f"Sequence-level class balance: {pd.Series(y).value_counts(normalize=True).to_dict()}")

    return X, y, ip_labels, feature_cols


def save_sequences(X, y, ip_labels, out_dir: Path = None):
    if out_dir is None:
        out_dir = PROCESSED_DATA_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "sequences_X.npy", X)
    np.save(out_dir / "sequences_y.npy", y)
    # WHY WE SAVE ip_labels TOO: the LSTM training script needs this to
    # split train/test BY IP instead of by individual sequence — see the
    # leakage explanation in lstm_model.py. Without this array, we can't
    # do a correct split.
    np.save(out_dir / "sequences_ip.npy", np.array(ip_labels))
    print(f"Saved sequences to {out_dir}/sequences_X.npy, sequences_y.npy, sequences_ip.npy")


if __name__ == "__main__":
    df = pd.read_parquet(PROCESSED_DATA_DIR / "cleaned_flows.parquet")
    X, y, ip_labels, feature_cols = build_sequences(df, window_size=10, step=5)
    save_sequences(X, y, ip_labels)
