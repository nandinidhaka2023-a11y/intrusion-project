import os
import sys
import time
import numpy as np
import pandas as pd
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from predictor import IntrusionPredictor


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

X_PATH = os.path.join(
    BASE_DIR, "data", "processed", "sequences_X.npy"
)

IP_PATH = os.path.join(
    BASE_DIR, "data", "processed", "sequences_ip.npy"
)

CLEANED_PATH = os.path.join(
    BASE_DIR, "data", "processed", "cleaned_flows.parquet"
)


def get_attack_types():
    df = pd.read_parquet(CLEANED_PATH)

    if "attack_type" in df.columns:
        return df["attack_type"].astype(str).values

    return np.array(["Unknown"] * len(df))


def run_stream(num_events=20, delay=0.5):

    print("\n=== REAL-TIME INTRUSION STREAM ===\n")

    predictor = IntrusionPredictor()

    X = np.load(X_PATH, mmap_mode="r")
    ips = np.load(IP_PATH, allow_pickle=True)
    attack_types = get_attack_types()

    print("Sequence dataset shape:", X.shape)

    total = min(num_events, len(X))

    for i in range(total):

        # Get one sequence
        sequence = np.asarray(X[i], dtype=np.float32)

        # Remove unnecessary dimensions BEFORE sending to predictor
        sequence = np.squeeze(sequence)

        result = predictor.predict(sequence)

        host = str(ips[i])

        label_index = min(i, len(attack_types) - 1)
        attack_type = attack_types[label_index]

        probability = result["intrusion_probability"]
        risk = result["risk_level"]

        if probability >= 0.80:
            status = "HIGH RISK"
        elif probability >= 0.50:
            status = "MEDIUM RISK"
        else:
            status = "NORMAL"

        timestamp = datetime.now().strftime("%H:%M:%S")

        print(
            f"[{timestamp}] "
            f"Host={host:<18} "
            f"Probability={probability:.2%} "
            f"Risk={risk:<6} "
            f"Type={attack_type:<20} "
            f"Status={status}"
        )

        time.sleep(delay)


if __name__ == "__main__":
    run_stream()