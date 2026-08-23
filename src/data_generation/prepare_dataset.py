import json
import random
import argparse
from pathlib import Path
from collections import Counter

DATA_DIR = Path("/root/model_miniaturization/data")
SYNTHETIC_FILE = DATA_DIR / "synthetic/train_samples.jsonl"
OUTPUT_DIR = DATA_DIR / "processed"

TRAIN_RATIO = 0.80
VAL_RATIO   = 0.10
TEST_RATIO  = 0.10


def load_jsonl(path: Path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def save_jsonl(samples, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")
    print(f"  Saved {len(samples)} samples → {path.name}")


def stratified_split(samples, train_r, val_r, seed=42):
    rng = random.Random(seed)
    by_label = {}
    for s in samples:
        by_label.setdefault(s["triage_level"], []).append(s)

    train, val, test = [], [], []
    for label, group in by_label.items():
        rng.shuffle(group)
        n = len(group)
        n_train = int(n * train_r)
        n_val   = int(n * val_r)
        train += group[:n_train]
        val   += group[n_train:n_train + n_val]
        test  += group[n_train + n_val:]
        print(f"  {label}: {n} total → train={n_train} val={n_val} test={n - n_train - n_val}")

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test


def print_stats(name, samples):
    counts = Counter(s["triage_level"] for s in samples)
    total = len(samples)
    parts = " | ".join(f"{k}={v} ({v/total*100:.0f}%)" for k, v in sorted(counts.items()))
    print(f"  {name}: {total} samples — {parts}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(SYNTHETIC_FILE))
    parser.add_argument("--output_dir", default=str(OUTPUT_DIR))
    args = parser.parse_args()

    input_path  = Path(args.input)
    output_dir  = Path(args.output_dir)

    print(f"Loading {input_path.name}...")
    samples = load_jsonl(input_path)
    print(f"  {len(samples)} samples loaded")
    print_stats("full", samples)

    print(f"\nStratified split ({int(TRAIN_RATIO*100)}/{int(VAL_RATIO*100)}/{int(TEST_RATIO*100)}):")
    train, val, test = stratified_split(samples, TRAIN_RATIO, VAL_RATIO)

    print(f"\nSaving to {output_dir}/")
    save_jsonl(train, output_dir / "train.jsonl")
    save_jsonl(val,   output_dir / "val.jsonl")
    save_jsonl(test,  output_dir / "test.jsonl")

    print(f"\nFinal split:")
    print_stats("train", train)
    print_stats("val",   val)
    print_stats("test",  test)
    print("\nDone.")


if __name__ == "__main__":
    main()
