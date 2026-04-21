import sys, os, json
sys.path.insert(0, ".")

import joblib
import numpy as np
from backend.app.feature_extractor import extract_base_features

# Load what the model expects
with open("artifacts_phase2/feature_names.json") as f:
    feature_names = json.load(f)

scaler = joblib.load("artifacts_phase2/scaler.pkl")

files = {
    "parsec-windows.exe": r"C:\Users\asus laptop\Downloads\parsec-windows.exe",
    "notepad.exe":        r"C:\Windows\System32\notepad.exe",
}

print(f"{'Feature':<25}", end="")
for name in files:
    print(f"  {name:>22}", end="")
print()
print("-" * 75)

rows = {}
for label, path in files.items():
    with open(path, "rb") as f:
        data = f.read()
    base = extract_base_features(data)
    vals = [base.get(fn, 0.0) for fn in feature_names]
    rows[label] = vals

for i, fn in enumerate(feature_names):
    print(f"{fn:<25}", end="")
    for label in files:
        print(f"  {rows[label][i]:>22.4f}", end="")
    print()
