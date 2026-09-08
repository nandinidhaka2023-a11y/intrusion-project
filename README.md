# Real-time Network Intrusion & Attack Probability Prediction

A deep-learning system that predicts attack probability from network traffic
flows in near real time, shows a live-style risk dashboard, and triggers a
simulated automated mitigation (IP blocking) response.

## Why this project
Detecting a single suspicious flow is a classification problem. Predicting
*risk building up over time* per host/IP is a sequence-modeling problem —
that's the differentiated, deep-learning-appropriate framing we're using here,
instead of a plain "attack / not attack" classifier.

## Pipeline
1. **Data**: CICIDS2017 (flow-level features, includes source/destination IP
   and timestamps).
2. **Baseline model**: Random Forest / XGBoost on flow features — a fast
   benchmark before we go to deep learning.
3. **Deep learning model**: LSTM/GRU over time-windowed sequences of flows
   per host, with two output heads — attack probability, and attack type.
4. **Dashboard**: Flask API + frontend showing a live threat gauge,
   per-IP risk table, attack-type breakdown, model comparison, and an
   alert/mitigation feed.
5. **Mitigation layer**: when risk crosses a threshold, the system marks the
   IP as blocked in a live blocklist (simulated — see `reports/scope_note.md`
   for why this isn't literal internet-scale prevention).

## Folder structure
```
data/raw/          -> original CICIDS2017 CSVs go here (not committed to git)
data/processed/    -> cleaned/windowed data ready for modeling
src/               -> all pipeline code (loading, preprocessing, models)
notebooks/         -> EDA and experiments
models/            -> saved trained model weights
dashboard/backend/ -> Flask API
dashboard/frontend/-> dashboard UI
reports/           -> write-ups, scope notes, evaluation results
```

## Team split
- **Data pipeline, baseline model, LSTM/DL model, evaluation** — Mahika
- **Flask API, dashboard frontend, deployment, live-replay simulation** — teammate

## Status
- [x] Repo structure
- [ ] Dataset loaded + EDA
- [ ] Baseline model (RF/XGBoost)
- [ ] LSTM sequence model
- [ ] Dashboard
- [ ] Mitigation/blocking layer
- [ ] Deployment
