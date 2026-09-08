"""
lstm_model.py

WHAT THIS FILE DOES:
Defines and trains an LSTM neural network that reads a SEQUENCE of flows
(built by sequence_builder.py) and predicts whether an attack is
happening/imminent. This is the "deep learning" core of the project.

WHY AN LSTM SPECIFICALLY:
LSTM = Long Short-Term Memory. It reads the sequence one flow at a time
(like reading a sentence word by word) and keeps a running "memory" of
what it's seen so far in that sequence. This lets it notice patterns
that only exist ACROSS flows — e.g. "packet size has been shrinking and
connection rate rising over the last 10 flows" — which a model that only
looks at one flow at a time (like our Random Forest baseline) cannot see.

WHY FOCAL LOSS INSTEAD OF NORMAL LOSS:
Normal loss (binary cross-entropy) treats every mistake equally. With our
~98%/2% imbalance, a lazy model could get very low loss just by always
predicting "benign" and barely getting punished for missing the rare
attacks. Focal loss adds a term that DOWN-WEIGHTS easy/already-confident
predictions and UP-WEIGHTS hard/rare examples (like our attacks), forcing
the model to actually pay attention to the minority class instead of
ignoring it.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import average_precision_score, f1_score, classification_report
from pathlib import Path

PROCESSED_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SequenceDataset(Dataset):
    """Wraps our numpy sequences so PyTorch's DataLoader can batch them."""

    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class FocalLoss(nn.Module):
    """
    WHY: see module docstring above. gamma controls how much we
    down-weight easy examples (higher gamma = focus harder on hard
    examples); alpha up-weights the rare positive (attack) class directly.
    """

    def __init__(self, alpha: float = 0.85, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce = nn.BCELoss(reduction="none")

    def forward(self, preds, targets):
        bce_loss = self.bce(preds, targets)
        pt = torch.where(targets == 1, preds, 1 - preds)
        alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        focal_term = alpha_t * (1 - pt) ** self.gamma * bce_loss
        return focal_term.mean()


class IntrusionLSTM(nn.Module):
    """
    Architecture: LSTM reads the sequence -> takes its final hidden state
    (a summary of the whole sequence) -> passes it through a small
    feed-forward head -> outputs a single attack-probability number.

    WHY WE ONLY USE THE FINAL HIDDEN STATE: by the time the LSTM has read
    the last flow in the window, its hidden state has accumulated
    information from every flow before it too (that's the whole point of
    "memory" in an LSTM) — so the final state is a summary of the whole
    sequence, not just the last flow.
    """

    def __init__(self, n_features: int, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x shape: (batch, window_size, n_features)
        _, (hidden, _) = self.lstm(x)
        last_hidden = hidden[-1]  # final layer's hidden state: (batch, hidden_size)
        out = self.head(last_hidden)
        return out.squeeze(-1)


def normalize_features(X_train, X_test):
    """
    Scales features to roughly the same range using train-set statistics
    only (never the test set — that would leak test information into
    training). Neural networks train much better on normalized inputs
    than on raw values that range from 0 to millions.
    """
    mean = X_train.mean(axis=(0, 1), keepdims=True)
    std = X_train.std(axis=(0, 1), keepdims=True) + 1e-8
    X_train_norm = (X_train - mean) / std
    X_test_norm = (X_test - mean) / std
    return X_train_norm, X_test_norm, mean, std


def train_lstm(X, y, ip_labels, epochs: int = 15, batch_size: int = 64, lr: float = 1e-3):
    """
    WHY WE SPLIT BY IP (GroupShuffleSplit) INSTEAD OF A NORMAL RANDOM
    SPLIT: our windows overlap (step < window_size in sequence_builder.py),
    so consecutive sequences from the SAME IP share most of their rows.
    A normal random split would put near-duplicate windows on both sides
    of train/test, letting the model "cheat" by memorizing rows it already
    saw — this is exactly what happened the first time we ran this (we got
    a suspicious PR-AUC of 1.0000, which was the giveaway). GroupShuffleSplit
    keeps every sequence from a given IP entirely on one side of the split,
    so the model is actually tested on IPs/patterns it has never seen.
    """
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(splitter.split(X, y, groups=ip_labels))
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    train_ips = set(np.array(ip_labels)[train_idx])
    test_ips = set(np.array(ip_labels)[test_idx])
    print(f"Train IPs ({len(train_ips)}): {sorted(train_ips)}")
    print(f"Test IPs ({len(test_ips)}): {sorted(test_ips)}")
    assert train_ips.isdisjoint(test_ips), "Leakage check failed: an IP appears in both sets!"
    print("Leakage check passed: no IP appears in both train and test.\n")

    X_train, X_test, mean, std = normalize_features(X_train, X_test)

    train_ds = SequenceDataset(X_train, y_train)
    test_ds = SequenceDataset(X_test, y_test)

    # Oversample the rare attack class within each training batch, on top
    # of focal loss — belt-and-suspenders against the imbalance problem.
    class_counts = np.bincount(y_train.astype(int))
    sample_weights = 1.0 / class_counts[y_train.astype(int)]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    model = IntrusionLSTM(n_features=X.shape[2]).to(DEVICE)
    criterion = FocalLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print(f"Training on {DEVICE} | {len(X_train):,} train sequences | {len(X_test):,} test sequences\n")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(xb)

        avg_loss = total_loss / len(train_ds)

        # Quick eval each epoch so we can watch PR-AUC improve
        model.eval()
        all_preds, all_true = [], []
        with torch.no_grad():
            for xb, yb in test_loader:
                xb = xb.to(DEVICE)
                preds = model(xb).cpu().numpy()
                all_preds.extend(preds)
                all_true.extend(yb.numpy())
        pr_auc = average_precision_score(all_true, all_preds)
        print(f"Epoch {epoch:2d}/{epochs} | loss={avg_loss:.4f} | test PR-AUC={pr_auc:.4f}")

    # final detailed report
    y_pred_binary = (np.array(all_preds) >= 0.5).astype(int)
    print("\n=== FINAL LSTM RESULTS ===")
    print(f"PR-AUC: {pr_auc:.4f}")
    print(f"F1: {f1_score(all_true, y_pred_binary):.4f}")
    print(classification_report(all_true, y_pred_binary, target_names=["Benign", "Attack"]))

    return model, mean, std, {"pr_auc": pr_auc}


def save_model(model, mean, std):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model_state": model.state_dict(), "mean": mean, "std": std, "n_features": mean.shape[-1]},
        MODELS_DIR / "lstm_model.pt",
    )
    print(f"Saved LSTM model to {MODELS_DIR}/lstm_model.pt")


if __name__ == "__main__":
    X = np.load(PROCESSED_DATA_DIR / "sequences_X.npy")
    y = np.load(PROCESSED_DATA_DIR / "sequences_y.npy")
    ip_labels = np.load(PROCESSED_DATA_DIR / "sequences_ip.npy", allow_pickle=True)
    model, mean, std, metrics = train_lstm(X, y, ip_labels)
    save_model(model, mean, std)
