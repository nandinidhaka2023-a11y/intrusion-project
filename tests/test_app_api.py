"""Run with:  python -m pytest tests/test_app_api.py -q
Builds a throw-away copy of the app with a stand-in predictor and fake
sequences, so it needs neither torch, your data, nor a trained model."""
import importlib.util
import shutil
import sys
import types
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent

TYPES = {  # attack type -> (host, probability stored in feature 0)
    "Benign": ("192.168.10.6", 0.02),
    "DDoS": ("205.174.165.73", 0.97),
    "Bot": ("192.168.10.9", 0.55),
}


@pytest.fixture()
def app_module(tmp_path, monkeypatch):
    (tmp_path / "src").mkdir()
    (tmp_path / "data" / "processed").mkdir(parents=True)
    shutil.copy(REPO / "src" / "app.py", tmp_path / "src" / "app.py")
    shutil.copytree(REPO / "templates", tmp_path / "templates")
    shutil.copytree(REPO / "static", tmp_path / "static")

    X, ips, labels = [], [], []
    for name, (host, p) in TYPES.items():
        for _ in range(4):
            w = np.zeros((10, 77), dtype=np.float32)
            w[0, 0] = p
            X.append(w); ips.append(host); labels.append(name)
    np.save(tmp_path / "data/processed/sequences_X.npy", np.stack(X))
    np.save(tmp_path / "data/processed/sequences_ip.npy", np.array(ips, dtype=object), allow_pickle=True)
    np.save(tmp_path / "data/processed/sequences_attack_type.npy", np.array(labels, dtype=object), allow_pickle=True)

    stub = types.ModuleType("predictor")
    stub.HIGH_THRESHOLD, stub.MEDIUM_THRESHOLD = 0.8, 0.5

    class Stub:
        temperature = 1.0

        def predict(self, seq):
            p = float(np.asarray(seq)[0, 0])
            return {"intrusion_probability": p,
                    "risk_level": "HIGH" if p >= .8 else "MEDIUM" if p >= .5 else "LOW",
                    "raw_probability": p, "temperature": 1.0}

    stub.IntrusionPredictor = Stub
    monkeypatch.setitem(sys.modules, "predictor", stub)
    monkeypatch.syspath_prepend(str(REPO / "src"))

    spec = importlib.util.spec_from_file_location("app_under_test", tmp_path / "src" / "app.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.threat_intel.lists = {}          # no blocklists needed here
    return mod


def feed(mod, n=None):
    for i in (mod.stream_indices if n is None else mod.stream_indices[:n]):
        mod.process_event(i)


def test_manual_block_unblock_roundtrip(app_module):
    c = app_module.app.test_client()
    r = c.post("/api/block", json={"ip": "205.174.165.73", "reason": "demo"})
    assert r.status_code == 200 and r.get_json()["blocked"]["by"] == "analyst"
    m = c.get("/api/mitigation").get_json()
    assert [b["host"] for b in m["blocked"]] == ["205.174.165.73"]
    assert c.post("/api/unblock", json={"ip": "205.174.165.73"}).get_json()["was_blocked"] is True
    assert c.get("/api/mitigation").get_json()["blocked"] == []
    assert c.post("/api/unblock", json={"ip": "205.174.165.73"}).get_json()["was_blocked"] is False


def test_invalid_input_is_rejected(app_module):
    c = app_module.app.test_client()
    for bad in ["", "abc", "999.1.1.1", "<script>", None]:
        assert c.post("/api/block", json={"ip": bad}).status_code == 400
    assert c.post("/api/block", data="not json").status_code == 400


def test_cross_origin_requests_are_refused(app_module):
    c = app_module.app.test_client()
    r = c.post("/api/block", json={"ip": "8.8.8.8"}, headers={"Origin": "http://evil.example"})
    assert r.status_code == 403
    assert c.get("/api/mitigation").get_json()["blocked"] == []
    ok = c.post("/api/block", json={"ip": "8.8.8.8"}, headers={"Origin": "http://localhost"})
    assert ok.status_code == 200


def test_windows_after_blocking_are_counted_as_dropped(app_module):
    c = app_module.app.test_client()
    c.post("/api/block", json={"ip": "205.174.165.73"})
    feed(app_module)
    m = c.get("/api/mitigation").get_json()
    assert m["dropped"] == 3 and m["blocked"][0]["windows_dropped"] == 3   # 3 demo windows per type
    events = c.get("/api/events").get_json()["events"]
    assert all(e["blocked"] == (e["host"] == "205.174.165.73") for e in events)
    row = next(h for h in c.get("/api/hosts").get_json()["hosts"] if h["host"] == "205.174.165.73")
    assert row["blocked"] is True


def test_auto_block_is_off_by_default_and_blocks_only_dangerous_hosts(app_module):
    c = app_module.app.test_client()
    feed(app_module)
    assert c.get("/api/mitigation").get_json()["blocked"] == []   # off by default
    app_module.hosts.clear(); app_module.events.clear()
    assert c.post("/api/auto_block", json={"enabled": True}).get_json()["auto_block"] is True
    feed(app_module)
    m = c.get("/api/mitigation").get_json()
    assert [b["host"] for b in m["blocked"]] == ["205.174.165.73"]    # 0.97 >= 0.90 only
    assert m["blocked"][0]["by"] == "auto"
    assert m["dropped"] == 1   # blocked after window 2, so only window 3 is dropped


def test_auto_block_needs_two_windows(app_module):
    c = app_module.app.test_client()
    c.post("/api/auto_block", json={"enabled": True})
    feed(app_module, n=0)
    ddos_first = [i for i in app_module.stream_indices if app_module.IPS[i] == "205.174.165.73"][0]
    app_module.process_event(ddos_first)
    assert c.get("/api/mitigation").get_json()["blocked"] == []


def test_check_ip_reports_blocked_state(app_module):
    c = app_module.app.test_client()
    assert c.get("/api/check_ip?ip=8.8.8.8").get_json()["blocked"] is False
    c.post("/api/block", json={"ip": "8.8.8.8"})
    assert c.get("/api/check_ip?ip=8.8.8.8").get_json()["blocked"] is True


def test_dashboard_page_loads(app_module):
    c = app_module.app.test_client()
    html = c.get("/").get_data(as_text=True)
    assert "Mitigation" in html and "/static/chart.umd.js" in html
    assert c.get("/static/chart.umd.js").status_code == 200


def test_replay_resets_counters_but_keeps_blocks(app_module, monkeypatch):
    c = app_module.app.test_client()
    feed(app_module)
    c.post("/api/block", json={"ip": "205.174.165.73"})
    assert c.get("/api/stats").get_json()["processed"] == 9
    real_sleep = app_module.time.sleep
    monkeypatch.setattr(app_module.time, "sleep", lambda s: None)   # replay instantly (undone after the test)
    assert c.post("/api/replay").status_code == 200
    for _ in range(200):                            # wait for the thread to finish
        if not c.get("/api/stats").get_json()["stream_running"]:
            break
        real_sleep(0.02)
    stats = c.get("/api/stats").get_json()
    m = c.get("/api/mitigation").get_json()
    assert stats["processed"] == 9 and stats["stream_running"] is False
    assert [b["host"] for b in m["blocked"]] == ["205.174.165.73"]   # block survived
    assert m["dropped"] == 3                                          # and now drops its windows


def test_replay_refused_while_running_and_from_other_origins(app_module):
    c = app_module.app.test_client()
    app_module.stream_status["running"] = True
    assert c.post("/api/replay").status_code == 409
    app_module.stream_status["running"] = False
    assert c.post("/api/replay", headers={"Origin": "http://evil.example"}).status_code == 403
