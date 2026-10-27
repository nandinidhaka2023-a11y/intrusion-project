"""Run with:  python -m pytest tests/test_ip_checker.py -q
Uses small synthetic blocklists, so it needs no internet."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import ipaddress
import pytest
import ip_checker as ic


@pytest.fixture()
def intel(tmp_path):
    (tmp_path / "firehol_level1.txt").write_text(
        "# comment\n11.22.33.0/24\n5.6.7.8\n"
    )
    (tmp_path / "blocklist_de.txt").write_text("5.6.7.8\n9.8.7.6  # trailing note\n")
    return ic.ThreatIntel(tmp_path)


def test_invalid_input_is_rejected():
    for bad in ["", "abc", "999.1.1.1", "1.2.3", "1.2.3.4.5", "<script>"]:
        r = ic.check_ip(bad)
        assert r["verdict"] == "INVALID" and r["valid"] is False


def test_private_and_loopback_are_internal_and_never_looked_up(intel):
    for ip in ["192.168.10.5", "10.0.0.1", "127.0.0.1", "::1"]:
        r = ic.check_ip(ip, intel=intel, abuse_key="would-be-used")
        assert r["verdict"] == "INTERNAL"
        assert r["evidence"]["blocklists"] == []
        assert r["evidence"]["abuseipdb"] is None  # nothing sent outside


def test_cidr_and_single_ip_hits(intel):
    assert ic.check_ip("11.22.33.200", intel=intel)["verdict"] == "MALICIOUS"
    assert ic.check_ip("11.22.34.1", intel=intel)["verdict"] == "NO_EVIDENCE"


def test_single_weaker_list_is_suspicious_not_malicious(intel):
    r = ic.check_ip("9.8.7.6", intel=intel)
    assert r["verdict"] == "SUSPICIOUS" and r["score"] == 75


def test_two_lists_raise_the_score(intel):
    r = ic.check_ip("5.6.7.8", intel=intel)
    assert r["score"] == 90 and r["verdict"] == "MALICIOUS"
    assert {h["list"] for h in r["evidence"]["blocklists"]} == {"firehol_level1", "blocklist_de"}


def test_model_evidence_alone_drives_verdict():
    quiet = {"windows": 3, "current": 0.03, "peak": 0.05, "high_windows": 0, "attack_types": {"Benign": 3}}
    loud = {"windows": 3, "current": 0.97, "peak": 1.0, "high_windows": 3, "attack_types": {"PortScan": 3}}
    assert ic.check_ip("8.8.8.8", host_stats=quiet)["verdict"] == "NO_EVIDENCE"
    r = ic.check_ip("8.8.8.8", host_stats=loud, sim_note="sim")
    assert r["verdict"] == "MALICIOUS" and "sim" in " ".join(r["reasons"])


def test_repeated_high_windows_are_never_clean():
    stats = {"windows": 6, "current": 0.1, "peak": 1.0, "high_windows": 3, "attack_types": {"DDoS": 3, "Benign": 3}}
    assert ic.check_ip("8.8.8.8", host_stats=stats)["verdict"] == "SUSPICIOUS"


def test_no_lists_downloaded_is_explained(tmp_path):
    r = ic.check_ip("8.8.8.8", intel=ic.ThreatIntel(tmp_path))
    assert r["verdict"] == "NO_EVIDENCE"
    assert any("--update" in t for t in r["reasons"])


def test_abuseipdb_error_is_reported_not_raised(monkeypatch, intel):
    monkeypatch.setattr(ic, "abuseipdb_lookup", lambda ip, key: {"error": "AbuseIPDB unreachable (x)"})
    r = ic.check_ip("8.8.8.8", intel=intel, abuse_key="k")
    assert r["verdict"] == "NO_EVIDENCE" and any("unreachable" in t for t in r["reasons"])


def test_abuseipdb_high_confidence_is_malicious(monkeypatch, intel):
    monkeypatch.setattr(ic, "abuseipdb_lookup",
                        lambda ip, key: {"confidence": 96, "reports": 120, "country": "XX"})
    assert ic.check_ip("8.8.8.8", intel=intel, abuse_key="k")["verdict"] == "MALICIOUS"


def test_lookup_matches_brute_force_on_random_addresses(tmp_path):
    import random
    rng = random.Random(1)
    nets = [ipaddress.ip_network((rng.getrandbits(32) & 0xFFFFFF00, rng.choice([16, 20, 24, 32])), strict=False)
            for _ in range(300)]
    (tmp_path / "firehol_level1.txt").write_text("\n".join(str(n) for n in nets))
    lst = ic.ThreatIntel(tmp_path).lists["firehol_level1"]
    for _ in range(3000):
        ip = ipaddress.ip_address(rng.getrandbits(32))
        assert (lst.lookup(ip) is not None) == any(ip in n for n in nets)


def test_reserved_documentation_range_is_not_treated_as_public(intel):
    assert ic.check_ip("203.0.113.9", intel=intel)["scope"] != "public"
