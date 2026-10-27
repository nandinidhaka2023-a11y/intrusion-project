"""Run with:  python -m pytest tests/test_predictor_calibration.py -q
Needs torch. Builds a small random model, so no trained weights are required."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from lstm_model import IntrusionLSTM
import predictor as pr
import calibrate as cal

F = 12


@pytest.fixture()
def ckpt(tmp_path):
    torch.manual_seed(0)
    model = IntrusionLSTM(n_features=F)
    path = tmp_path / "lstm_model.pt"
    torch.save({"model_state": model.state_dict(),
                "mean": np.zeros((1, 1, F), dtype=np.float32),
                "std": np.ones((1, 1, F), dtype=np.float32),
                "n_features": F}, path)
    return model.eval(), path


def window(seed=0, scale=1.0):
    return np.random.default_rng(seed).normal(0, scale, (10, F)).astype(np.float32)


def test_new_path_matches_the_original_sigmoid_output(ckpt, tmp_path):
    model, path = ckpt
    p = pr.IntrusionPredictor(model_path=str(path), calibration_path=str(tmp_path / "none.json"))
    for seed in range(20):
        w = window(seed, scale=3)
        with torch.no_grad():
            expected = model(torch.from_numpy(w).unsqueeze(0)).item()
        out = p.predict(w)
        assert out["raw_probability"] == pytest.approx(expected, abs=1e-4)
        assert out["intrusion_probability"] == pytest.approx(expected, abs=1e-4)  # T = 1
        assert out["risk_level"] == pr.risk_band(out["intrusion_probability"])


def test_temperature_softens_without_changing_order(ckpt, tmp_path):
    _, path = ckpt
    cfile = tmp_path / "calibration.json"
    cfile.write_text(json.dumps({"temperature": 3.0}))
    soft = pr.IntrusionPredictor(model_path=str(path), calibration_path=str(cfile))
    plain = pr.IntrusionPredictor(model_path=str(path), calibration_path=str(tmp_path / "none.json"))
    ws = [window(s, 4) for s in range(30)]
    a = [plain.predict_logit(w) for w in ws]
    b = [soft.predict(w)["intrusion_probability"] for w in ws]
    c = [plain.predict(w)["intrusion_probability"] for w in ws]
    assert np.array_equal(np.argsort(a), np.argsort(b)) or np.corrcoef(a, b)[0, 1] > 0.999
    for x, y in zip(b, c):  # closer to 0.5 than the uncalibrated score
        assert abs(x - 0.5) <= abs(y - 0.5) + 1e-6


def test_no_more_saturation_at_exactly_one(ckpt, tmp_path):
    _, path = ckpt
    p = pr.IntrusionPredictor(model_path=str(path), calibration_path=str(tmp_path / "none.json"))
    p.model.head[3].bias.data += 40.0  # force a huge logit
    out = p.predict(window(1))
    assert out["intrusion_probability"] == 1.0  # rounds to 1 only at 4 d.p.
    assert p.predict_logit(window(1)) > 30      # but the raw score is still informative


def test_bad_or_missing_calibration_file_falls_back_to_one(tmp_path):
    assert pr.load_temperature(str(tmp_path / "missing.json")) == 1.0
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    assert pr.load_temperature(str(bad)) == 1.0
    neg = tmp_path / "neg.json"
    neg.write_text(json.dumps({"temperature": -2}))
    assert pr.load_temperature(str(neg)) == 1.0


def test_wrong_shape_still_raises(ckpt, tmp_path):
    _, path = ckpt
    p = pr.IntrusionPredictor(model_path=str(path), calibration_path=str(tmp_path / "none.json"))
    with pytest.raises(ValueError):
        p.predict(np.zeros(F, dtype=np.float32))


def test_risk_band_edges():
    cases = [(0.0, "LOW"), (0.4999, "LOW"), (0.5, "MEDIUM"), (0.7999, "MEDIUM"), (0.8, "HIGH"), (1.0, "HIGH")]
    for p_, band in cases:
        assert pr.risk_band(p_) == band


def test_fit_temperature_recovers_a_known_overconfidence():
    rng = np.random.default_rng(0)
    true_logit = rng.normal(0, 1.5, 20000)
    y = (rng.random(20000) < cal.sigmoid(true_logit)).astype(float)
    overconfident = true_logit * 4.0           # model is 4x too sure of itself
    t = cal.fit_temperature(overconfident, y)
    assert t == pytest.approx(4.0, rel=0.12)
    p_before, p_after = cal.sigmoid(overconfident), cal.sigmoid(overconfident / t)
    assert cal.expected_calibration_error(p_after, y)[0] < cal.expected_calibration_error(p_before, y)[0] / 3
