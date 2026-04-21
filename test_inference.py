import requests, json, os

tests = [
    (r"C:\Users\asus laptop\Downloads\parsec-windows.exe", "parsec-windows.exe"),
    (r"C:\Windows\System32\notepad.exe",                   "notepad.exe"),
    (r"C:\Windows\System32\calc.exe",                      "calc.exe"),
    (r"C:\Windows\System32\mspaint.exe",                   "mspaint.exe"),
]

print(f"{'File':<42} {'Verdict':<12} {'Confidence':>10}  {'Result'}")
print("-" * 80)
for path, name in tests:
    if not os.path.exists(path):
        print(f"{name:<42} {'NOT FOUND':<12}")
        continue
    try:
        with open(path, "rb") as f:
            r = requests.post(
                "http://localhost:8000/analyze",
                files={"file": (name, f, "application/octet-stream")},
                timeout=90,
            )
        d = r.json()
        verdict = d.get("verdict", "ERROR")
        conf    = d.get("confidence", 0.0)
        # Ground truth: all test files above are benign
        correct = "OK" if verdict == "Benign" else "FALSE POSITIVE"
        print(f"{name:<42} {verdict:<12} {conf:>10.4f}  {correct}")
    except Exception as e:
        print(f"{name:<42} ERROR: {e}")
