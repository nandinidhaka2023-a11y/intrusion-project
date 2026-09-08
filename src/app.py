import os
import sys
import threading
import time
from datetime import datetime

import numpy as np

from flask import Flask, jsonify, render_template
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

from predictor import IntrusionPredictor


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
    )
)

CORS(app)


# ---------------------------------------------------------
# Load predictor and data
# ---------------------------------------------------------

print("Loading intrusion predictor...")

predictor = IntrusionPredictor()

print("Predictor loaded successfully.")

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

state = {
    "processed": 0,
    "high_risk": 0,
    "medium_risk": 0,
    "low_risk": 0,
    "attack_types": {}
}


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

    attack_type = str(
        ATTACK_TYPES[index]
    )

    if probability >= 0.80:

        status = "ALERT"

    elif probability >= 0.50:

        status = "WARNING"

    else:

        status = "NORMAL"


    event = {
        "time": datetime.now().strftime(
            "%H:%M:%S"
        ),

        "host": host,

        "probability": probability,

        "risk": risk,

        "attack_type": attack_type,

        "status": status
    }


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

    return jsonify(
        {
            "events": events
        }
    )


@app.route("/api/stats")
def get_stats():

    return jsonify(
        {
            "processed": state["processed"],

            "high_risk": state["high_risk"],

            "medium_risk": state["medium_risk"],

            "low_risk": state["low_risk"],

            "attack_types": state[
                "attack_types"
            ]
        }
    )


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