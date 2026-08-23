"""Upsample EMERGENCY class 2x to address low emergency recall in retraining."""
import json
import random
from pathlib import Path
from collections import Counter

INPUT  = Path("/root/model_miniaturization/data/approach2/combined_train.jsonl")
OUTPUT = Path("/root/model_miniaturization/data/approach2/combined_train_v2.jsonl")
FACTOR = 2  # keep 1 original + add (FACTOR-1) copies of EMERGENCY

random.seed(42)

samples = [json.loads(l) for l in open(INPUT)]
by_class = Counter(s["triage_level"] for s in samples)
print(f"Original: {len(samples)} samples  {dict(by_class)}")

emergency = [s for s in samples if s["triage_level"] == "EMERGENCY"]
combined  = samples + emergency * (FACTOR - 1)
random.shuffle(combined)

by_class2 = Counter(s["triage_level"] for s in combined)
print(f"After {FACTOR}x EMERGENCY upsample: {len(combined)} samples  {dict(by_class2)}")
print(f"  EMERGENCY fraction: {by_class2['EMERGENCY']/len(combined)*100:.1f}%")

with open(OUTPUT, "w") as f:
    for s in combined:
        f.write(json.dumps(s) + "\n")
print(f"\nSaved → {OUTPUT}")
