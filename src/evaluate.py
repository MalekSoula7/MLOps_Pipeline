import json, sys

with open("metrics.json") as f:
    metrics = json.load(f)

AUC_THRESHOLD = 0.80
print(f"AUC: {metrics['auc']} (threshold: {AUC_THRESHOLD})")

if metrics["auc"] < AUC_THRESHOLD:
    print("FAIL: model below threshold. Blocking deployment.")
    sys.exit(1)

print("PASS: model approved.")