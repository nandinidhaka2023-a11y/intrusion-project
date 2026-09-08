import os
import sys
import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from lstm_model import IntrusionLSTM


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "lstm_model.pt")


class IntrusionPredictor:
    def __init__(self):
        self.device = torch.device("cpu")

        checkpoint = torch.load(
            MODEL_PATH,
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

    def predict(self, sequence):
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
            probability = self.model(x).item()

        if probability >= 0.80:
            risk = "HIGH"
        elif probability >= 0.50:
            risk = "MEDIUM"
        else:
            risk = "LOW"

        return {
            "intrusion_probability": round(probability, 4),
            "risk_level": risk
        }


if __name__ == "__main__":
    print("Loading trained LSTM...")

    predictor = IntrusionPredictor()

    print("Model loaded successfully!")
    print("Ready for streaming inference.")