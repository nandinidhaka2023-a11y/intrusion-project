# New in this version: IP check + corrected risk

## 1. One-time setup (on your laptop, in the repo root)

```bash
python src/ip_checker.py --update     # downloads 5 public blocklists (~40k entries) into data/blocklists/
python src/calibrate.py               # run AFTER lstm_model.py; writes models/calibration.json + reports/reliability.png
python src/app.py                     # dashboard at http://127.0.0.1:5000
```

Optional live reputation (free key from abuseipdb.com):
`export ABUSEIPDB_API_KEY=your_key` (Windows: `set ABUSEIPDB_API_KEY=your_key`) before starting the app.
Public addresses you type are then also sent to AbuseIPDB. Private addresses are never sent anywhere.

Tests: `python -m pytest tests -q`

## 2. "Check an IP address" - what it really does

The LSTM scores *traffic windows*, not addresses, so it cannot judge an address it has never seen traffic from.
The checker therefore combines three kinds of evidence and prints which one produced the verdict:

| Evidence | Source | Works for unknown addresses? |
|---|---|---|
| Model | what the LSTM scored for that host in the stream | No - only hosts seen in the stream |
| Blocklists | FireHOL level 1, Feodo C2, Spamhaus EDROP, blocklist.de, GreenSnow (offline) | Yes |
| Reputation | AbuseIPDB (optional) | Yes |

Verdicts: MALICIOUS (score >= 80), SUSPICIOUS (>= 40), NO EVIDENCE (public, nothing found), INTERNAL (private/loopback/reserved), INVALID.
"No evidence" is deliberately not called "safe". Nothing is ever scanned or contacted.
Hosts from the replayed stream carry a note that their addresses are simulated.

## 3. "Correcting the risk" - what changed and why

| Problem in the old version | Fix |
|---|---|
| Probabilities were all ~0% or ~100%; the sigmoid saturated and Medium never appeared | `predictor.py` now works from the raw logit (no saturation) |
| Focal loss + oversampling make the model over-confident, so "90%" did not mean 90% | `calibrate.py` fits one temperature T on held-out hosts: `p = sigmoid(logit / T)` |
| "Host Risk Monitoring" was really an event log | New per-host table: running risk (EWMA), peak, windows, label, last seen |
| Thresholds 0.5 / 0.8 were copied into three files | One definition in `predictor.py`, imported everywhere |
| `Web Attack � XSS` showed a broken character (defect D-06) | Fixed for display (`display_label`) |
| Dashboard needed internet for Chart.js | Chart.js now served from `static/` |

Calibration does **not** change PR-AUC (it never reorders scores). It changes what the percentage means.
Report the before/after expected calibration error that `calibrate.py` prints.

## 4. Suggested wording for the report

> To make the displayed risk trustworthy, the pre-sigmoid score of the LSTM is divided by a single temperature T,
> fitted by minimising log-loss on half of the held-out hosts and evaluated on the other half (expected
> calibration error before: __, after: __; T = __). Calibration leaves the ranking of windows, and hence PR-AUC,
> unchanged. The dashboard additionally keeps a running per-host risk as an exponentially weighted average of its
> windows (weight 0.6 on the newest) together with the host's peak.
>
> The IP check does not claim that the LSTM can classify an arbitrary address. It reports the model's evidence if
> the host appeared in the monitored traffic and otherwise relies on public blocklists (and optionally AbuseIPDB);
> absence from all sources is reported as "no evidence", not as "safe".
