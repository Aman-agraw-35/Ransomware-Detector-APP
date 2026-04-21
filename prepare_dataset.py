"""
prepare_dataset.py
------------------
One-time script to prepare the Kaggle dataset for training.

The Kaggle dataset (amdj3dax/ransomware-detection-data-set) uses:
  Label column: 'Benign'  (1 = Benign, 0 = Ransomware/Malicious)
  -- which is the INVERSE of what the notebook expects (1 = Ransomware)

This script:
1. Loads the raw Kaggle CSV
2. Renames/normalises columns to match the notebook's assumptions
3. Inverts the label so 1 = Ransomware, 0 = Benign
4. Adds synthetic entropy features from available PE fields (entropy proxies)
5. Writes out: data_file.csv  (in repo root, ready for the notebook)

Usage:
    python prepare_dataset.py --input path/to/downloaded/data_file.csv

OR just place the raw kaggle CSV in the repo root and run:
    python prepare_dataset.py
"""

import argparse
import os
import sys

try:
    import pandas as pd
    import numpy as np
except ImportError:
    print("Run:  pip install pandas numpy")
    sys.exit(1)

DEFAULT_INPUT = "data_file.csv"
OUTPUT = "data_file_prepared.csv"


def main():
    parser = argparse.ArgumentParser(description="Prepare Kaggle PE dataset for training.")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Path to raw Kaggle CSV")
    parser.add_argument("--output", default=OUTPUT, help="Output CSV path")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"[ERROR] Input file not found: {args.input}")
        print()
        print("Download from: https://www.kaggle.com/datasets/amdj3dax/ransomware-detection-data-set")
        print("Place the downloaded data_file.csv in the repo root, then re-run this script.")
        sys.exit(1)

    print(f"[+] Loading: {args.input}")
    df = pd.read_csv(args.input)
    print(f"    Shape: {df.shape}")
    print(f"    Columns: {list(df.columns)}")

    # ── Normalise label column ──────────────────────────────────────────────
    # Kaggle dataset: Benign=1 means safe, Benign=0 means malware/ransomware
    # Notebook expects: last column = 1 for Ransomware, 0 for Benign
    if "Benign" in df.columns:
        df["label"] = (df["Benign"] == 0).astype(int)  # 0=malware → 1=Ransomware
        df = df.drop(columns=["Benign"])
        benign_n = (df["label"] == 0).sum()
        ransom_n = (df["label"] == 1).sum()
        print(f"[+] Labels remapped: {benign_n} Benign, {ransom_n} Ransomware")
    elif "label" in df.columns:
        print("[+] 'label' column already present, skipping remap.")
    else:
        # Try last column
        last = df.columns[-1]
        print(f"[!] No 'Benign' or 'label' column found. Using last column: '{last}'")
        df = df.rename(columns={last: "label"})

    # ── Make all feature column names lowercase ─────────────────────────────
    meta_cols = {"FileName", "md5Hash", "label"}
    rename_map = {}
    for col in df.columns:
        if col not in meta_cols:
            rename_map[col] = col.lower()
    df = df.rename(columns=rename_map)

    # ── Add entropy proxy features (not in the Kaggle dataset) ─────────────
    # The Kaggle dataset doesn't have per-section entropy, only ResourceSize.
    # We create meaningful proxies from existing columns.
    if "resourcesize" in df.columns and "section_entropy_mean" not in df.columns:
        # Proxy: large resource sections tend to have lower entropy (images, icons)
        # This is a rough approximation; real entropy from PE parsing is better.
        resource_norm = df["resourcesize"].clip(upper=5_000_000)
        df["section_entropy_mean"] = 5.0 + 2.0 * (1.0 - resource_norm / resource_norm.max().clip(1))
        df["section_entropy_max"]  = df["section_entropy_mean"] + np.random.uniform(0, 0.8, len(df))
        df["section_entropy_std"]  = np.random.uniform(0.1, 0.5, len(df))
        df["section_entropy_min"]  = df["section_entropy_mean"] - np.random.uniform(0, 1.0, len(df))
        print("[+] Added entropy proxy features from resourcesize.")

    # ── Add counts from available data ─────────────────────────────────────
    if "exportsize" in df.columns and "exports_count" not in df.columns:
        df["exports_count"] = (df["exportsize"] > 0).astype(float) * (df["exportsize"] / 24).clip(0, 500)
        print("[+] Derived exports_count from export_size.")

    if "resourcesize" in df.columns and "resources_count" not in df.columns:
        df["resources_count"] = (df["resourcesize"] / 40).clip(0, 200)
        print("[+] Derived resources_count from resource_size.")

    # ── Summary ─────────────────────────────────────────────────────────────
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    label_dist = df["label"].value_counts().to_dict()
    print(f"\n[+] Final dataset shape: {df.shape}")
    print(f"    Numeric feature columns: {len(numeric_cols) - 1}")  # -1 for label
    print(f"    Label distribution: Benign={label_dist.get(0, 0)}, Ransomware={label_dist.get(1, 0)}")
    print(f"    Saving to: {args.output}")

    df.to_csv(args.output, index=False)
    print("\n[OK] Done! Use this file as DATA_PATH in the notebook:")
    print(f'    DATA_PATH = "{args.output}"')
    print('    LABEL_COLUMN = "label"')
    print('    DROP_COLUMNS = ["FileName", "md5Hash"]')


if __name__ == "__main__":
    main()
