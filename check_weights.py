results = {
    "parsec-windows.exe": (0.8928, 0.4387),
    "notepad.exe":        (0.0002, 0.0076),
    "calc.exe":           (0.0001, 0.0008),
    "cmd.exe":            (0.0008, 0.0482),
}
threshold = 0.55
bw, rw = 0.2, 0.8
print(f"{'File':<35} {'BiLSTM':>8} {'RF':>8} {'Ensemble':>10} {'Verdict':<12} {'Status'}")
print("-"*85)
for name, (b, r) in results.items():
    ens = bw*b + rw*r
    verdict = "Ransomware" if ens >= threshold else "Benign"
    status = "OK" if verdict == "Benign" else "FP!"
    print(f"{name:<35} {b:>8.4f} {r:>8.4f} {ens:>10.4f} {verdict:<12} {status}")
