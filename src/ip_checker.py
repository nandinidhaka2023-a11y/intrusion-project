"""
ip_checker.py

WHAT THIS FILE DOES:
Answers the question "is this IP address malicious?" for the dashboard's
"Check an IP address" box.

HONEST DESIGN NOTE (say this in the demo, it is a strength, not a weakness):
Our LSTM does NOT look at an address. It looks at 10 consecutive FLOWS (77
numeric features each) and says how attack-like that traffic is. An IP address
on its own carries no flow features, so the model cannot score an arbitrary
address out of thin air. Instead we combine three independent kinds of
evidence and always show which one produced the verdict:

  1. MODEL EVIDENCE     - what our LSTM has already scored for traffic coming
                          from this host (only if the host appeared in the
                          stream the dashboard is watching).
  2. BLOCKLIST EVIDENCE - the address is inside a downloaded public threat
                          list (FireHOL level1, Feodo botnet C2, Spamhaus
                          EDROP, blocklist.de, GreenSnow). Fully offline once
                          the lists are downloaded.
  3. REPUTATION EVIDENCE- optional AbuseIPDB lookup (only if you set the
                          ABUSEIPDB_API_KEY environment variable).

"No evidence found" is NOT the same as "safe": a brand-new attacker address
will not be on any list yet. The verdict wording reflects that.

PRIVACY / SAFETY:
  - Private, loopback and reserved addresses are never sent to any outside
    service.
  - We never connect to, scan or probe the address being checked. All checks
    are passive look-ups.

USAGE FROM THE TERMINAL:
  python src/ip_checker.py --update          # download the blocklists
  python src/ip_checker.py --check 1.2.3.4   # check one address
"""

import bisect
import ipaddress
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
BLOCKLIST_DIR = BASE_DIR / "data" / "blocklists"

_RAW = "https://raw.githubusercontent.com/firehol/blocklist-ipsets/master/"

# weight = how much a hit on this list alone should count (0-100).
SOURCES = {
    "firehol_level1": {
        "url": _RAW + "firehol_level1.netset",
        "weight": 85,
        "what": "FireHOL level 1 (DShield, Feodo C2, Spamhaus DROP, bogons)",
    },
    "feodo": {
        "url": _RAW + "feodo.ipset",
        "weight": 95,
        "what": "Feodo Tracker botnet command-and-control servers",
    },
    "spamhaus_edrop": {
        "url": _RAW + "spamhaus_edrop.netset",
        "weight": 85,
        "what": "Spamhaus EDROP (hijacked / criminal netblocks)",
    },
    "blocklist_de": {
        "url": _RAW + "blocklist_de.ipset",
        "weight": 75,
        "what": "blocklist.de (addresses reported for SSH / web / mail attacks)",
    },
    "greensnow": {
        "url": _RAW + "greensnow.ipset",
        "weight": 70,
        "what": "GreenSnow (addresses seen attacking honeypots)",
    },
}

MALICIOUS_AT = 80   # score >= 80  -> MALICIOUS
SUSPICIOUS_AT = 40  # score >= 40  -> SUSPICIOUS

DISCLAIMER = (
    "Passive look-up only. 'No evidence found' means the address is on none of "
    "the lists we hold and not in the traffic we monitored - it does not prove "
    "the address is safe."
)


# ---------------------------------------------------------------------------
# Address classification
# ---------------------------------------------------------------------------

def classify_scope(ip):
    """Return a short human label for where this address lives."""
    if ip.is_loopback:
        return "loopback"
    if ip.is_link_local:
        return "link-local"
    if ip.is_multicast:
        return "multicast"
    if ip.is_unspecified:
        return "unspecified"
    if ip.is_private:
        return "private"
    if ip.is_reserved:
        return "reserved"
    if ip.is_global:
        return "public"
    return "non-routable"


# ---------------------------------------------------------------------------
# One downloaded blocklist, loaded into a fast lookup structure
# ---------------------------------------------------------------------------

class IPv4List:
    """
    Holds a list of IPv4 addresses / CIDR blocks. Lookup is a binary search
    over merged [start, end] integer ranges, so even a 30,000-line list
    answers in microseconds.
    """

    def __init__(self, name, path):
        self.name = name
        self.path = Path(path)
        self.networks = []
        self._starts = []
        self._ends = []
        self.mtime = None
        self._load()

    def _load(self):
        nets = []
        with open(self.path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                token = line.split("#", 1)[0].strip()
                if not token:
                    continue
                token = token.split()[0].split(";")[0]
                try:
                    net = ipaddress.ip_network(token, strict=False)
                except ValueError:
                    continue
                if net.version == 4:
                    nets.append(net)
        self.networks = nets
        ranges = sorted(
            (int(n.network_address), int(n.broadcast_address)) for n in nets
        )
        merged = []
        for start, end in ranges:
            if merged and start <= merged[-1][1] + 1:
                merged[-1][1] = max(merged[-1][1], end)
            else:
                merged.append([start, end])
        self._starts = [m[0] for m in merged]
        self._ends = [m[1] for m in merged]
        self.mtime = self.path.stat().st_mtime

    def __len__(self):
        return len(self.networks)

    def lookup(self, ip):
        """Return the matching entry as text (e.g. '1.2.3.0/24'), or None."""
        if ip.version != 4:
            return None
        value = int(ip)
        i = bisect.bisect_right(self._starts, value) - 1
        if i < 0 or value > self._ends[i]:
            return None
        best = None
        for net in self.networks:  # only runs on a hit, so cost is fine
            if ip in net and (best is None or net.prefixlen > best.prefixlen):
                best = net
        return str(best) if best else None


class ThreatIntel:
    """Loads every list found in data/blocklists and answers look-ups."""

    def __init__(self, directory=BLOCKLIST_DIR):
        self.directory = Path(directory)
        self.lists = {}
        self._lock = threading.Lock()
        self._last_scan = 0.0
        self.refresh(force=True)

    def refresh(self, force=False):
        """Reload any list whose file changed (checked at most once a minute)."""
        now = time.time()
        if not force and now - self._last_scan < 60:
            return
        with self._lock:
            self._last_scan = now
            found = {}
            for name, meta in SOURCES.items():
                path = self.directory / f"{name}.txt"
                if not path.exists():
                    continue
                current = self.lists.get(name)
                if current is not None and current.mtime == path.stat().st_mtime:
                    found[name] = current
                    continue
                try:
                    found[name] = IPv4List(name, path)
                except OSError:
                    continue
            self.lists = found

    def status(self):
        self.refresh()
        newest = max((l.mtime for l in self.lists.values()), default=None)
        return {
            "lists_loaded": len(self.lists),
            "entries": sum(len(l) for l in self.lists.values()),
            "names": sorted(self.lists),
            "updated_hours_ago": (
                round((time.time() - newest) / 3600, 1) if newest else None
            ),
        }

    def hits(self, ip):
        self.refresh()
        out = []
        for name, lst in self.lists.items():
            entry = lst.lookup(ip)
            if entry:
                out.append(
                    {
                        "list": name,
                        "entry": entry,
                        "what": SOURCES[name]["what"],
                        "weight": SOURCES[name]["weight"],
                    }
                )
        return out


def update_blocklists(directory=BLOCKLIST_DIR, timeout=30):
    """Download every source. Returns {name: 'ok (N lines)' or error text}."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    report = {}
    for name, meta in SOURCES.items():
        try:
            req = urllib.request.Request(
                meta["url"], headers={"User-Agent": "rds-intrusion-project"}
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = resp.read()
            tmp = directory / f"{name}.txt.part"
            tmp.write_bytes(data)
            os.replace(tmp, directory / f"{name}.txt")
            lines = sum(
                1 for l in data.decode("utf-8", "ignore").splitlines()
                if l.strip() and not l.startswith("#")
            )
            report[name] = f"ok ({lines:,} entries)"
        except (urllib.error.URLError, OSError, TimeoutError) as err:
            report[name] = f"FAILED: {err}"
    return report


# ---------------------------------------------------------------------------
# Optional: AbuseIPDB reputation
# ---------------------------------------------------------------------------

_abuse_cache = {}
_ABUSE_TTL = 600  # seconds


def abuseipdb_lookup(ip_text, api_key, timeout=6):
    """
    Query AbuseIPDB's 'check' endpoint. Returns a dict, or {"error": ...}.
    Cached for 10 minutes so repeated clicks do not burn the free quota.
    """
    cached = _abuse_cache.get(ip_text)
    if cached and time.time() - cached[0] < _ABUSE_TTL:
        return cached[1]
    query = urllib.parse.urlencode({"ipAddress": ip_text, "maxAgeInDays": 90})
    req = urllib.request.Request(
        "https://api.abuseipdb.com/api/v2/check?" + query,
        headers={"Key": api_key, "Accept": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        d = payload.get("data", {})
        result = {
            "confidence": int(d.get("abuseConfidenceScore", 0)),
            "reports": int(d.get("totalReports", 0)),
            "country": d.get("countryCode"),
            "isp": d.get("isp"),
            "usage": d.get("usageType"),
            "last_reported": d.get("lastReportedAt"),
        }
    except urllib.error.HTTPError as err:
        result = {"error": f"AbuseIPDB returned HTTP {err.code}"}
    except (urllib.error.URLError, OSError, ValueError, TimeoutError) as err:
        result = {"error": f"AbuseIPDB unreachable ({err})"}
    _abuse_cache[ip_text] = (time.time(), result)
    return result


# ---------------------------------------------------------------------------
# The combined verdict
# ---------------------------------------------------------------------------

def _model_evidence(host_stats):
    """Turn the host's stream history into (score, reasons, evidence)."""
    if not host_stats or host_stats.get("windows", 0) == 0:
        return 0, [], None
    current = float(host_stats["current"])
    peak = float(host_stats["peak"])
    high = int(host_stats["high_windows"])
    n = int(host_stats["windows"])
    score = round(100 * current)
    if high >= 2:
        score = max(score, 60)  # repeated High windows are never 'clean'
    reasons = [
        f"Our LSTM scored {n} traffic window(s) from this host: "
        f"current risk {current:.0%}, peak {peak:.0%}, "
        f"{high} High-risk window(s)."
    ]
    types = host_stats.get("attack_types") or {}
    attacks = {t: c for t, c in types.items() if t.lower() != "benign"}
    if attacks and score >= SUSPICIOUS_AT:
        top = max(attacks, key=attacks.get)
        reasons.append(f"Most frequent attack label on those windows: {top}.")
    evidence = {
        "windows": n,
        "current": round(current, 4),
        "peak": round(peak, 4),
        "high_windows": high,
        "attack_types": types,
    }
    return score, reasons, evidence


def check_ip(raw, host_stats=None, intel=None, abuse_key=None, sim_note=None):
    """
    Combine all evidence for one address.

    raw         : what the user typed
    host_stats  : dict from the dashboard for this host, or None
    intel       : a ThreatIntel instance (or None to skip blocklists)
    abuse_key   : AbuseIPDB API key (or None to skip)
    sim_note    : text shown when the host comes from the simulated stream
    """
    text = (raw or "").strip()
    try:
        ip = ipaddress.ip_address(text)
    except ValueError:
        return {
            "ip": text[:64],
            "valid": False,
            "verdict": "INVALID",
            "score": 0,
            "reasons": ["That is not a valid IPv4 or IPv6 address."],
            "evidence": {},
            "disclaimer": DISCLAIMER,
        }

    scope = classify_scope(ip)
    external_ok = scope == "public"
    reasons = []
    evidence = {"model": None, "blocklists": [], "abuseipdb": None}
    scores = []

    # 1. our own model
    m_score, m_reasons, m_evidence = _model_evidence(host_stats)
    if m_evidence:
        scores.append(m_score)
        reasons += m_reasons
        evidence["model"] = m_evidence
        if sim_note:
            reasons.append(sim_note)

    # 2. blocklists
    if external_ok and intel is not None:
        hits = intel.hits(ip)
        evidence["blocklists"] = hits
        if hits:
            top = max(h["weight"] for h in hits)
            score = min(100, top + 5 * (len(hits) - 1))
            scores.append(score)
            for h in hits:
                reasons.append(
                    f"Listed in {h['list']} ({h['what']}) via {h['entry']}."
                )
        status = intel.status()
        evidence["blocklist_status"] = status
        if status["lists_loaded"] == 0:
            reasons.append(
                "No blocklists are downloaded yet - run "
                "'python src/ip_checker.py --update' for a real look-up."
            )

    # 3. optional reputation service
    if external_ok and abuse_key:
        ab = abuseipdb_lookup(str(ip), abuse_key)
        evidence["abuseipdb"] = ab
        if "error" in ab:
            reasons.append(ab["error"] + ".")
        else:
            scores.append(ab["confidence"])
            reasons.append(
                f"AbuseIPDB abuse-confidence {ab['confidence']}% from "
                f"{ab['reports']} report(s) in the last 90 days."
            )

    if not external_ok:
        reasons.append(
            f"This is a {scope} address, so it is not looked up in public "
            "blocklists or sent to any outside service."
        )

    score = max(scores) if scores else 0
    if score >= MALICIOUS_AT:
        verdict = "MALICIOUS"
    elif score >= SUSPICIOUS_AT:
        verdict = "SUSPICIOUS"
    elif not external_ok:
        verdict = "INTERNAL"
    else:
        verdict = "NO_EVIDENCE"
        if not any(s for s in scores):
            reasons.append(
                "Not in any loaded blocklist and not seen in our monitored traffic."
            )

    return {
        "ip": str(ip),
        "valid": True,
        "scope": scope,
        "verdict": verdict,
        "score": int(score),
        "reasons": reasons,
        "evidence": evidence,
        "disclaimer": DISCLAIMER,
    }


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = sys.argv[1:]
    if args[:1] == ["--update"]:
        for name, outcome in update_blocklists().items():
            print(f"{name:<16} {outcome}")
    elif args[:1] == ["--check"] and len(args) == 2:
        result = check_ip(
            args[1],
            intel=ThreatIntel(),
            abuse_key=os.environ.get("ABUSEIPDB_API_KEY"),
        )
        print(json.dumps(result, indent=2))
    else:
        print(__doc__)
