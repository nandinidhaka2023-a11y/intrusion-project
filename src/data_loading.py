"""
data_loading.py

WHAT THIS FILE DOES:
Loads the CICIDS2017 parquet files (dhoogla's cleaned "no-metadata" release),
combines them, adds back a simulated IP-addressing layer (since the raw
Source/Destination IP columns were stripped from this release), and produces
one tidy DataFrame ready for both the baseline model (Step 3) and the LSTM
model (Step 4).

WHY WE NEED THIS AS A SEPARATE FILE (not just doing it in a notebook):
- Both the baseline model and the LSTM model need the *same* cleaned data,
  so we write the cleaning logic once here and import it everywhere else.
  This also means when your teammate builds the Flask API, they can reuse
  this same function to process live/replayed traffic the same way.

NOTE ON THE DATASET VERSION:
This project uses dhoogla's "no-metadata" CICIDS2017 release on Kaggle —
8 parquet files, one per attack scenario/day (Benign-Monday, Botnet-Friday,
Bruteforce-Tuesday, DDoS-Friday, DoS-Wednesday, Infiltration-Thursday,
Portscan-Friday, WebAttacks-Thursday). This release already strips
Flow ID / Source IP / Destination IP / Timestamp (that's what "no-metadata"
means) and is already free of inf/NaN values — CICFlowMeter's messy leading-
space column names are also already fixed. So this file does LESS cleanup
than a raw-CSV version would need, but ADDS the IP-simulation step those
raw CICIDS2017 releases would have given us for free.
"""

import pandas as pd
import numpy as np
from pathlib import Path

RAW_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
PROCESSED_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"

# The real CICIDS2017 capture used this subnet for the victim network
# (documented in the original CIC paper). We reuse it so our simulated IPs
# look like the actual testbed instead of arbitrary numbers.
VICTIM_SUBNET_PREFIX = "192.168.10."
# Attacker machines in the original testbed were external to the victim LAN.
ATTACKER_SUBNET_PREFIXES = ["205.174.165.", "172.16.0.", "18.219.94.", "13.58.98."]


def load_raw_parquets(data_dir: Path = RAW_DATA_DIR) -> pd.DataFrame:
    """
    Reads every .parquet file in data/raw/ and stacks them into one
    DataFrame, keeping track of which file (attack scenario) each row
    came from.

    WHY: dhoogla's release splits CICIDS2017 into one file per attack
    scenario instead of one file per day. We need them combined to have
    a realistic mix of normal + attack traffic to train on.
    """
    files = sorted(data_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(
            f"No .parquet files found in {data_dir}. "
            "Place the downloaded CICIDS2017 files there first."
        )

    print(f"Found {len(files)} parquet file(s). Loading...")
    frames = []
    for f in files:
        df = pd.read_parquet(f)
        df["source_file"] = f.stem  # e.g. 'Portscan-Friday-no-metadata'
        frames.append(df)
        print(f"  - {f.name}: {len(df):,} rows")

    combined = pd.concat(frames, ignore_index=True)
    print(f"Combined shape: {combined.shape}")
    return combined


def clean_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Safety net for infinity/NaN values.

    WHY: dhoogla's release is already documented as having 0 missing
    values, but we keep this check anyway — never trust a claim about data
    quality without verifying it yourself, and it costs nothing to check.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)

    before = len(df)
    df = df.dropna(subset=numeric_cols)
    after = len(df)
    if before != after:
        print(f"Dropped {before - after:,} rows with inf/NaN values ({before:,} -> {after:,})")
    else:
        print("No inf/NaN values found (as expected for this release).")

    return df


def standardize_label(df: pd.DataFrame, label_col: str = "Label") -> pd.DataFrame:
    """
    Creates two label columns from the raw 'Label' column:
      - is_attack: 1 if any attack, 0 if benign (for the binary head)
      - attack_type: the original label, e.g. 'Portscan', 'DDoS', 'Benign'
        (for the multi-class head)

    WHY: Our model has two output heads — one for "is this an attack at
    all" and one for "what kind." We need both label columns ready before
    we can train either head.

    NOTE: we compare case-insensitively ('benign' vs 'Benign' vs 'BENIGN')
    because label capitalization is inconsistent across CICIDS2017 releases
    and even across files within the same release.
    """
    df["attack_type"] = df[label_col].astype(str).str.strip()
    df["is_attack"] = (df["attack_type"].str.lower() != "benign").astype(int)
    return df


def simulate_ip_addresses(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """
    Adds simulated src_ip / dst_ip columns.

    WHY WE NEED THIS: this dataset release strips the real Source IP /
    Destination IP columns (that's what "no-metadata" means). Our
    dashboard needs an IP to display and to "block" in the mitigation
    layer, so we reattach a plausible addressing layer on top of the real
    flow features.

    HOW: every row already represents one real captured flow. We assign:
      - dst_ip: always inside the real testbed's victim subnet
        (192.168.10.x), since every flow in this dataset targets a victim
        machine
      - src_ip: for benign flows, another internal machine
        (normal traffic between machines on the LAN); for attack flows,
        an address from one of the known external attacker ranges used in
        the original CICIDS2017 testbed

    This keeps the number of distinct attacker IPs bounded and realistic
    (a handful of attacker machines hitting many victims), rather than a
    different random IP for every single row, which is what a real attack
    looks like and what makes the "per-IP risk table" on the dashboard
    meaningful instead of just noise.
    """
    rng = np.random.default_rng(seed)
    n = len(df)

    # A small, fixed pool of attacker machines (realistic: attacks come
    # from a handful of hosts, not a new IP every flow)
    n_attacker_hosts = 12
    attacker_pool = [
        f"{rng.choice(ATTACKER_SUBNET_PREFIXES)}{rng.integers(1, 254)}"
        for _ in range(n_attacker_hosts)
    ]
    # A small pool of internal hosts for benign traffic
    n_internal_hosts = 20
    internal_pool = [f"{VICTIM_SUBNET_PREFIX}{i}" for i in range(1, n_internal_hosts + 1)]

    is_attack = df["is_attack"].to_numpy()

    src_ip = np.empty(n, dtype=object)
    dst_ip = np.empty(n, dtype=object)

    attack_mask = is_attack == 1
    n_attack = int(attack_mask.sum())
    n_benign = n - n_attack

    src_ip[attack_mask] = rng.choice(attacker_pool, size=n_attack)
    src_ip[~attack_mask] = rng.choice(internal_pool, size=n_benign)

    # victim/destination is always inside the LAN
    dst_ip[:] = rng.choice(internal_pool, size=n)

    df["src_ip"] = src_ip
    df["dst_ip"] = dst_ip
    return df


def load_and_clean(data_dir: Path = RAW_DATA_DIR, save: bool = True) -> pd.DataFrame:
    """
    Runs the full pipeline: load -> clean values -> labels -> simulated IPs.
    This is the one function everything else in the project should call.
    """
    df = load_raw_parquets(data_dir)
    df = clean_values(df)
    df = standardize_label(df)
    df = simulate_ip_addresses(df)

    print("\nClass balance (this is the imbalance we'll need to handle later):")
    print(df["is_attack"].value_counts(normalize=True).rename("proportion"))

    print("\nAttack type breakdown:")
    print(df["attack_type"].value_counts())

    if save:
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        out_path = PROCESSED_DATA_DIR / "cleaned_flows.parquet"
        df.to_parquet(out_path, index=False)
        print(f"\nSaved cleaned data to {out_path}")

    return df


if __name__ == "__main__":
    load_and_clean()
