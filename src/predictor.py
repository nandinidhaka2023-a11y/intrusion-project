import json
import math
import os
import sys
import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from lstm_model import IntrusionLSTM


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "lstm_model.pt")
CALIBRATION_PATH = os.path.join(BASE_DIR, "models", "calibration.json")

# One place for the risk bands. app.py and the dashboard import these so the
# thresholds can never drift apart again.
HIGH_THRESHOLD = 0.80
MEDIUM_THRESHOLD = 0.50


def risk_band(probability):
    """Map a probability to LOW / MEDIUM / HIGH."""
    if probability >= HIGH_THRESHOLD:
        return "HIGH"
    if probability >= MEDIUM_THRESHOLD:
        return "MEDIUM"
    return "LOW"


def _sigmoid(z):
    """Numerically safe sigmoid (never overflows)."""
    z = max(-60.0, min(60.0, z))
    return 1.0 / (1.0 + math.exp(-z))


def load_temperature(path=CALIBRATION_PATH):
    """
    Read the temperature written by calibrate.py. Returns 1.0 (= no change)
    if the file is missing or unreadable, so the system still works before
    calibration has been run.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            value = float(json.load(f)["temperature"])
        return value if value > 0 else 1.0
    except (OSError, KeyError, ValueError, TypeError):
        return 1.0


class IntrusionPredictor:
    def __init__(self, model_path=MODEL_PATH, calibration_path=CALIBRATION_PATH):
        self.device = torch.device("cpu")

        checkpoint = torch.load(
            model_path,
            map_location=self.device
        )

        self.mean = np.asarray(checkpoint["mean"]).reshape(-1)
        self.std = np.asarray(checkpoint["std"]).reshape(-1)

        self.model = IntrusionLSTM(
            n_features=checkpoint["n_features"],
            hidden_size=64
        )

        self.model.load_state_dict(checkpoint["model_state"])
        self.model.to(self.device)
        self.model.eval()

        # Temperature scaling (see calibrate.py). 1.0 means "not calibrated".
        self.temperature = load_temperature(calibration_path)

    def predict_logit(self, sequence):
        """Return the raw score BEFORE the final sigmoid, for one (10, F) window."""
        sequence = np.asarray(sequence, dtype=np.float32)

        # Each streaming item should be (10, 77)
        sequence = np.squeeze(sequence)

        if sequence.ndim != 2:
            raise ValueError(
                f"Expected sequence with 2 dimensions, got {sequence.shape}"
            )

        normalized = (sequence - self.mean) / self.std

        # Convert (10, 77) -> (1, 10, 77)
        x = torch.from_numpy(
            normalized.astype(np.float32)
        ).unsqueeze(0)

        with torch.no_grad():
            _, (hidden, _) = self.model.lstm(x)
            # head[:-1] is everything except the final Sigmoid layer.
            logit = self.model.head[:-1](hidden[-1]).squeeze(-1).item()
        return logit

    def predict(self, sequence):
        logit = self.predict_logit(sequence)

        # Working from the logit avoids the old problem where the sigmoid
        # saturated to exactly 0.0 or 1.0 and every event looked "100% sure".
        raw_probability = _sigmoid(logit)
        probability = _sigmoid(logit / self.temperature)

        return {
            "intrusion_probability": round(probability, 4),
            "risk_level": risk_band(probability),
            "raw_probability": round(raw_probability, 4),
            "temperature": round(self.temperature, 4),
        }


if __name__ == "__main__":
    print("Loading trained LSTM...")

    predictor = IntrusionPredictor()

    print("Model loaded successfully!")
    print(f"Calibration temperature: {predictor.temperature}")
    print("Ready for streaming inference.")
