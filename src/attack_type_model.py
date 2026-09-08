import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score


DATA_DIR = "data/processed"
MODEL_DIR = "models"

X_PATH = os.path.join(DATA_DIR, "sequences_X.npy")
LABEL_PATH = os.path.join(DATA_DIR, "sequences_attack_type.npy")

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "attack_type_model.pt"
)


# ---------------------------------------------------------
# Settings
# ---------------------------------------------------------

SEED = 42

np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cpu")

BATCH_SIZE = 256
EPOCHS = 8
HIDDEN_SIZE = 64


# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------

print("=" * 60)
print("ATTACK TYPE CLASSIFIER")
print("=" * 60)

print("\nLoading sequence data...")

X = np.load(X_PATH)
labels = np.load(
    LABEL_PATH,
    allow_pickle=True
)

print("X shape:", X.shape)
print("Labels:", len(labels))


# ---------------------------------------------------------
# Encode labels
# ---------------------------------------------------------

classes = sorted(
    np.unique(labels)
)

class_to_id = {
    name: i
    for i, name in enumerate(classes)
}

y = np.array([
    class_to_id[label]
    for label in labels
])

print("\nAttack classes:")

for i, name in enumerate(classes):
    count = np.sum(y == i)

    print(
        f"{i:2d}  {name:35s} {count}"
    )


# ---------------------------------------------------------
# Normalize
# ---------------------------------------------------------

print("\nNormalizing features...")

mean = X.mean(
    axis=(0, 1)
)

std = X.std(
    axis=(0, 1)
)

std[std < 1e-8] = 1.0

X = (
    X - mean
) / std


# ---------------------------------------------------------
# Train/test split
# ---------------------------------------------------------

print("\nSplitting data...")

train_idx, test_idx = train_test_split(
    np.arange(len(X)),
    test_size=0.20,
    random_state=SEED,
    stratify=y
)

X_train = X[train_idx]
X_test = X[test_idx]

y_train = y[train_idx]
y_test = y[test_idx]


# ---------------------------------------------------------
# Model
# ---------------------------------------------------------

class AttackTypeLSTM(nn.Module):

    def __init__(
        self,
        n_features,
        n_classes
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=HIDDEN_SIZE,
            num_layers=1,
            batch_first=True
        )

        self.classifier = nn.Sequential(
            nn.Linear(
                HIDDEN_SIZE,
                32
            ),

            nn.ReLU(),

            nn.Dropout(0.2),

            nn.Linear(
                32,
                n_classes
            )
        )


    def forward(self, x):

        output, _ = self.lstm(x)

        last_output = output[:, -1, :]

        return self.classifier(
            last_output
        )


model = AttackTypeLSTM(
    n_features=X.shape[2],
    n_classes=len(classes)
).to(DEVICE)


# ---------------------------------------------------------
# Class weights
# ---------------------------------------------------------

class_counts = np.bincount(
    y_train,
    minlength=len(classes)
)

weights = (
    len(y_train)
    /
    (
        len(classes)
        *
        np.maximum(
            class_counts,
            1
        )
    )
)

weights = torch.tensor(
    weights,
    dtype=torch.float32
)


loss_function = nn.CrossEntropyLoss(
    weight=weights
)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)


# ---------------------------------------------------------
# Convert data
# ---------------------------------------------------------

X_train = torch.tensor(
    X_train,
    dtype=torch.float32
)

X_test = torch.tensor(
    X_test,
    dtype=torch.float32
)

y_train = torch.tensor(
    y_train,
    dtype=torch.long
)

y_test = torch.tensor(
    y_test,
    dtype=torch.long
)


# ---------------------------------------------------------
# Training
# ---------------------------------------------------------

print("\nStarting training...")
print("Device:", DEVICE)

for epoch in range(EPOCHS):

    model.train()

    permutation = torch.randperm(
        len(X_train)
    )

    total_loss = 0.0

    for start in range(
        0,
        len(X_train),
        BATCH_SIZE
    ):

        indices = permutation[
            start:start + BATCH_SIZE
        ]

        batch_x = X_train[
            indices
        ]

        batch_y = y_train[
            indices
        ]


        optimizer.zero_grad()

        logits = model(
            batch_x
        )

        loss = loss_function(
            logits,
            batch_y
        )

        loss.backward()

        optimizer.step()

        total_loss += loss.item()


    # -----------------------------------------------------
    # Evaluation
    # -----------------------------------------------------

    model.eval()

    predictions = []

    with torch.no_grad():

        for start in range(
            0,
            len(X_test),
            BATCH_SIZE
        ):

            batch_x = X_test[
                start:start + BATCH_SIZE
            ]

            logits = model(
                batch_x
            )

            preds = torch.argmax(
                logits,
                dim=1
            )

            predictions.extend(
                preds.cpu().numpy()
            )


    predictions = np.array(
        predictions
    )

    accuracy = accuracy_score(
        y_test.numpy(),
        predictions
    )

    macro_f1 = f1_score(
        y_test.numpy(),
        predictions,
        average="macro",
        zero_division=0
    )

    avg_loss = (
        total_loss
        /
        max(
            1,
            (
                len(X_train)
                +
                BATCH_SIZE
                -
                1
            )
            // BATCH_SIZE
        )
    )

    print(
        f"Epoch {epoch + 1}/{EPOCHS} "
        f"| Loss: {avg_loss:.4f} "
        f"| Accuracy: {accuracy:.4f} "
        f"| Macro-F1: {macro_f1:.4f}"
    )


# ---------------------------------------------------------
# Final report
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("FINAL ATTACK TYPE RESULTS")
print("=" * 60)

print(
    classification_report(
        y_test.numpy(),
        predictions,
        labels=np.arange(len(classes)),
        target_names=classes,
        zero_division=0
    )
)


# ---------------------------------------------------------
# Save
# ---------------------------------------------------------

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

torch.save(
    {
        "model_state": model.state_dict(),
        "mean": mean,
        "std": std,
        "n_features": X.shape[2],
        "n_classes": len(classes),
        "classes": classes
    },
    MODEL_PATH
)

print("\nSaved attack type model:")
print(MODEL_PATH)