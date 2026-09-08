import numpy as np
import pandas as pd

DATA_PATH = "data/processed/cleaned_flows.parquet"
X_PATH = "data/processed/sequences_X.npy"
OUTPUT_PATH = "data/processed/sequences_attack_type.npy"

print("Loading dataset...")

df = pd.read_parquet(DATA_PATH)

labels = df["attack_type"].astype(str).to_numpy()

X = np.load(X_PATH, mmap_mode="r")

print(f"Flows: {len(labels):,}")
print(f"Sequences: {len(X):,}")

# Each sequence has length 10 and sequence_builder
# uses step=5.
sequence_count = len(X)

# Priority is used only when multiple attack types occur
# in the same sequence.
priority = [
    "Web Attack � Sql Injection",
    "Web Attack � XSS",
    "Web Attack � Brute Force",
    "Heartbleed",
    "Infiltration",
    "DDoS",
    "DoS Hulk",
    "DoS GoldenEye",
    "DoS slowloris",
    "DoS Slowhttptest",
    "FTP-Patator",
    "SSH-Patator",
    "PortScan",
    "Bot",
    "Benign",
]

print("Creating sequence attack labels...")

sequence_labels = []

for n in range(sequence_count):

    start = n * 5
    end = start + 10

    window = labels[start:end]

    label = "Benign"

    for attack in priority:

        if attack in window:
            label = attack
            break

    sequence_labels.append(label)

    if (n + 1) % 50000 == 0:
        print(
            f"Processed {n + 1:,} / "
            f"{sequence_count:,} sequences"
        )

sequence_labels = np.array(
    sequence_labels,
    dtype=str
)

np.save(
    OUTPUT_PATH,
    sequence_labels
)

print("\nDONE!")
print(f"Saved: {OUTPUT_PATH}")

print("\nSequence label distribution:")

print(
    pd.Series(sequence_labels)
    .value_counts()
)