import os
import sys
import threading
import time
from datetime import datetime

import numpy as np

from flask import Flask, jsonify, render_template, request
from flask_cors import CORS


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

sys.path.append(
    os.path.dirname(os.path.abspath(__file__))
)

from predictor import (
    IntrusionPredictor,
    HIGH_THRESHOLD,
    MEDIUM_THRESHOLD,
)
from ip_checker import ThreatIntel, check_ip


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
# Flask
# ---------------------------------------------------------

app = Flask(
    __name__,
    template_folder=os.path.join(
        BASE_DIR,
        "templates"
    ),
    static_folder=os.path.join(
        BASE_DIR,
        "static"
    )
)

CORS(app)


# ---------------------------------------------------------
# Load predictor and data
# ---------------------------------------------------------

print("Loading intrusion predictor...")

predictor = IntrusionPredictor()

print("Predictor loaded successfully.")
print(f"Risk calibration temperature: {predictor.temperature}")

threat_intel = ThreatIntel()
print(f"Blocklists: {threat_intel.status()}")

# Optional: set ABUSEIPDB_API_KEY in your environment for live reputation checks.
ABUSEIPDB_KEY = os.environ.get("ABUSEIPDB_API_KEY") or None

SIMULATED_NOTE = (
    "Note: the addresses in this demo stream are simulated (the dataset has no "
    "real IPs), so this model result describes the replayed traffic labelled "
    "with this address, not a real machine."
)

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
# Select representative events
# ---------------------------------------------------------

attack_indices = {}

for i, attack_type in enumerate(ATTACK_TYPES):

    attack_type = str(attack_type)

    if attack_type not in attack_indices:
        attack_indices[attack_type] = []

    attack_indices[attack_type].append(i)


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


stream_indices = []

for attack_type in demo_attack_types:

    if attack_type in attack_indices:

        # Three representative examples
        # for each available attack type.
        stream_indices.extend(
            attack_indices[attack_type][:3]
        )


if not stream_indices:

    stream_indices = list(
        range(
            min(39, len(X))
        )
    )


# ---------------------------------------------------------
# Dashboard state
# ---------------------------------------------------------

events = []

state_lock = threading.Lock()

# Per-host running risk, filled by process_event()
hosts = {}

# Weight of the newest window in the running (EWMA) host risk.
EWMA_ALPHA = 0.6

state = {
    "processed": 0,
    "high_risk": 0,
    "medium_risk": 0,
    "low_risk": 0,
    "attack_types": {}
}


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def display_label(name):
    """
    The dataset stores the dash in 'Web Attack - XSS' as a broken character.
    Fix it for display only; the raw name is still used to look windows up.
    """
    return str(name).replace("\ufffd", "-")


def risk_from_probability(probability):
    if probability >= HIGH_THRESHOLD:
        return "HIGH"
    if probability >= MEDIUM_THRESHOLD:
        return "MEDIUM"
    return "LOW"


def update_host(host, probability, attack_type, time_text):
    """
    Keep a running risk per host. A single window is a noisy signal, so the
    host risk is an exponentially weighted average of its recent windows
    (the newest window counts 60 percent). The peak is kept separately so a
    short, sharp attack is never hidden by later quiet traffic.
    """
    h = hosts.get(host)
    if h is None:
        h = {
            "host": host,
            "windows": 0,
            "current": probability,
            "peak": probability,
            "high_windows": 0,
            "attack_types": {},
            "last_seen": time_text,
        }
        hosts[host] = h
    else:
        h["current"] = EWMA_ALPHA * probability + (1 - EWMA_ALPHA) * h["current"]
        h["peak"] = max(h["peak"], probability)

    h["windows"] += 1
    if probability >= HIGH_THRESHOLD:
        h["high_windows"] += 1
    h["attack_types"][attack_type] = h["attack_types"].get(attack_type, 0) + 1
    h["last_seen"] = time_text


# ---------------------------------------------------------
# Process one event
# ---------------------------------------------------------

def process_event(index):

    sequence = X[index]

    result = predictor.predict(
        sequence
    )

    probability = float(
        result["intrusion_probability"]
    )

    risk = result["risk_level"]

    host = str(
        IPS[index]
    )

    attack_type = display_label(
        ATTACK_TYPES[index]
    )

    if probability >= HIGH_THRESHOLD:

        status = "ALERT"

    elif probability >= MEDIUM_THRESHOLD:

        status = "WARNING"

    else:

        status = "NORMAL"

    time_text = datetime.now().strftime(
        "%H:%M:%S"
    )

    event = {
        "time": time_text,

        "host": host,

        "probability": probability,

        "raw_probability": result.get(
            "raw_probability",
            probability
        ),

        "risk": risk,

        "attack_type": attack_type,

        "status": status
    }

    with state_lock:

        events.insert(
            0,
            event
        )

        # Keep dashboard responsive
        if len(events) > 100:
            events.pop()

        state["processed"] += 1

        if risk == "HIGH":

            state["high_risk"] += 1

        elif risk == "MEDIUM":

            state["medium_risk"] += 1

        else:

            state["low_risk"] += 1

        state["attack_types"][
            attack_type
        ] = (
            state["attack_types"].get(
                attack_type,
                0
            )
            + 1
        )

        update_host(
            host,
            probability,
            attack_type,
            time_text
        )


# ---------------------------------------------------------
# Background streaming
# ---------------------------------------------------------

def streaming_loop():

    print()
    print("=" * 60)
    print(" DASHBOARD STREAM STARTED")
    print("=" * 60)
    print(
        f"Streaming {len(stream_indices)} events..."
    )
    print()


    for index in stream_indices:

        try:

            process_event(index)

            latest = events[0]

            print(
                f"[{latest['time']}] "
                f"{latest['host']:<16} | "
                f"{latest['probability']:.2%} | "
                f"{latest['risk']:<6} | "
                f"{latest['attack_type']}"
            )

        except Exception as error:

            print(
                "Streaming error:",
                error
            )

        time.sleep(1)


    print()
    print("=" * 60)
    print(" DASHBOARD STREAM COMPLETED")
    print("=" * 60)


# ---------------------------------------------------------
# Routes
# ---------------------------------------------------------

@app.route("/")
def dashboard():

    return render_template(
        "index.html"
    )


@app.route("/api/events")
def get_events():

    with state_lock:
        snapshot = list(events)

    return jsonify(
        {
            "events": snapshot
        }
    )


@app.route("/api/stats")
def get_stats():

    with state_lock:
        payload = {
            "processed": state["processed"],

            "high_risk": state["high_risk"],

            "medium_risk": state["medium_risk"],

            "low_risk": state["low_risk"],

            "attack_types": dict(
                state["attack_types"]
            ),

            "calibration_temperature": predictor.temperature,
        }

    return jsonify(payload)


@app.route("/api/hosts")
def get_hosts():
    """One row per host with its running risk, most dangerous first."""

    with state_lock:
        rows = []

        for h in hosts.values():
            row = dict(h)
            row["attack_types"] = dict(h["attack_types"])
            row["risk"] = risk_from_probability(h["current"])
            rows.append(row)

    rows.sort(
        key=lambda r: (r["current"], r["peak"]),
        reverse=True
    )

    return jsonify(
        {
            "hosts": rows
        }
    )


@app.route("/api/check_ip", methods=["GET", "POST"])
def api_check_ip():
    """
    Is this address malicious? Combines what our model saw from the host,
    downloaded blocklists and (optionally) AbuseIPDB. See ip_checker.py.
    """

    if request.method == "POST":
        body = request.get_json(silent=True) or {}
        raw = body.get("ip", "")
    else:
        raw = request.args.get("ip", "")

    raw = str(raw).strip()[:64]

    with state_lock:
        known = hosts.get(raw)
        host_stats = (
            {**known, "attack_types": dict(known["attack_types"])}
            if known else None
        )

    result = check_ip(
        raw,
        host_stats=host_stats,
        intel=threat_intel,
        abuse_key=ABUSEIPDB_KEY,
        sim_note=SIMULATED_NOTE if host_stats else None,
    )

    result["abuseipdb_enabled"] = bool(ABUSEIPDB_KEY)

    return jsonify(result), (200 if result["valid"] else 400)


# ---------------------------------------------------------
# Start
# ---------------------------------------------------------

if __name__ == "__main__":

    print()
    print("=" * 60)
    print(" REAL-TIME INTRUSION DETECTION SYSTEM")
    print("=" * 60)

    stream_thread = threading.Thread(
        target=streaming_loop,
        daemon=True
    )

    stream_thread.start()

    print()
    print(
        "Dashboard: "
        "http://127.0.0.1:5000"
    )

    print("Streaming started...")
    print()


    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False,
        threaded=True
    )