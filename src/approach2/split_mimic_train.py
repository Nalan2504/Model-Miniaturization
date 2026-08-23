import json
import random
from pathlib import Path

INPUT_PATH = Path("data/approach2/mimic_train.jsonl")
TRAIN_OUT = Path("data/approach2/mimic_train_split.jsonl")
TEST_OUT = Path("data/approach2/mimic_test.jsonl")

random.seed(42)

rows = []
with open(INPUT_PATH, "r") as f:
    for line in f:
        rows.append(json.loads(line))

random.shuffle(rows)

split_idx = int(0.8 * len(rows))
train_rows = rows[:split_idx]
test_rows = rows[split_idx:]

with open(TRAIN_OUT, "w") as f:
    for r in train_rows:
        f.write(json.dumps(r) + "\n")

with open(TEST_OUT, "w") as f:
    for r in test_rows:
        f.write(json.dumps(r) + "\n")

print(f"Train: {len(train_rows)} -> {TRAIN_OUT}")
print(f"Test:  {len(test_rows)} -> {TEST_OUT}")