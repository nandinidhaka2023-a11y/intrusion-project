import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.append(
    os.path.dirname(os.path.abspath(__file__))
)

from predictor import IntrusionPredictor


BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

X_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "sequences_X.npy"
)

IP_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "sequences_ip.npy"
)

LABEL_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "sequences_attack_type.npy"
)


# ---------------------------------------------------------
# Load model
# ---------------------------------------------------------

print("Loading intrusion predictor...")

predictor = IntrusionPredictor()

print("Predictor loaded successfully.")


# ---------------------------------------------------------
# Load streaming data
# ---------------------------------------------------------

X = np.load(
    X_PATH,
    mmap_mode="r"
)

IPS = np.load(
    IP_PATH,
    allow_pickle=True
)

ATTACK_TYPES = np.load(
    LABEL_PATH,
    allow_pickle=True
)

print(
    f"Sequence dataset shape: {X.shape}"
)


# ---------------------------------------------------------
# Build attack-type groups
# ---------------------------------------------------------

attack_indices = {}

for i, attack_type in enumerate(ATTACK_TYPES):

    attack_type = str(attack_type)

    if attack_type not in attack_indices:
        attack_indices[attack_type] = []

    attack_indices[attack_type].append(i)


# Attack types we want to demonstrate
# in the live dashboard.

demo_attack_types = [
    "Benign",
    "DDoS",
    "DoS Hulk",
    "DoS GoldenEye",
    "FTP-Patator",
    "SSH-Patator",
    "PortScan",
    "Bot",
    "Web Attack � Brute Force",
    "Web Attack � XSS",
    "Web Attack � Sql Injection",
    "Infiltration",
    "Heartbleed"
]


# ---------------------------------------------------------
# Select representative sequences
# ---------------------------------------------------------

stream_indices = []

for attack_type in demo_attack_types:

    if attack_type in attack_indices:

        # Pick up to 3 examples of each type.
        examples = attack_indices[attack_type][:3]

        stream_indices.extend(examples)


# Fallback if something unexpected happens
if not stream_indices:

    stream_indices = list(
        range(
            min(30, len(X))
        )
    )


# ---------------------------------------------------------
# Start stream
# ---------------------------------------------------------

print("\n")
print("=" * 70)
print(" REAL-TIME INTRUSION DETECTION STREAM")
print("=" * 70)

print(
    f"Streaming {len(stream_indices)} representative events..."
)

print()


for count, index in enumerate(
    stream_indices,
    start=1
):

    sequence = X[index]

    result = predictor.predict(
        sequence
    )

    host = str(
        IPS[index]
    )

    attack_type = str(
        ATTACK_TYPES[index]
    )

    probability = result[
        "intrusion_probability"
    ]

    risk = result[
        "risk_level"
    ]


    if probability >= 0.80:

        status = "🚨 ALERT"

    elif probability >= 0.50:

        status = "⚠ WARNING"

    else:

        status = "✓ NORMAL"


    timestamp = datetime.now().strftime(
        "%H:%M:%S"
    )


    print(
        f"[{timestamp}] "
        f"{host:<16} | "
        f"Probability: {probability:.2%} | "
        f"Risk: {risk:<6} | "
        f"Type: {attack_type:<30} | "
        f"{status}"
    )


    time.sleep(1)


print()
print("=" * 70)
print(" STREAM COMPLETED")
print("=" * 70)