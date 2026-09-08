# Scope note: "real-time prevention" and IP tracking

**On prevention:** True prevention of live internet attacks is out of scope
for a course project (we don't control real network infrastructure). What we
implement instead is an **automated mitigation layer**: when the model's
predicted attack probability for a given IP crosses a threshold, the system
automatically marks that IP as blocked in a live blocklist and reflects this
on the dashboard in real time. This mirrors how real Intrusion *Prevention*
Systems (IPS) act on Intrusion *Detection* System (IDS) output. In our demo,
this runs against replayed CICIDS2017 traffic (and optionally a local
firewall rule in a sandboxed test setup) rather than live internet traffic.

**On attacker IPs:** CICIDS2017's original flow data includes Source IP and
Destination IP columns for every flow. We keep these through preprocessing
and expose them in both the per-IP risk table and the alert feed, and they
are exactly what the mitigation layer above acts on.
