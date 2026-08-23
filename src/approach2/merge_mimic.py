"""
Build combined_train_v3.jsonl from local sources — no container needed.

Sources:
  1. data/synthetic/train_samples.jsonl     — 5,100 teacher-generated synthetic
  2. syntech-ai/medical-triage-500          — 500 real labelled samples (HuggingFace)
  3. data/approach2/mimic_train.jsonl       — 207 real MIMIC-IV-ED demo patients

Steps:
  - Convert sources 1 and 2 to unified instruction-tuning format
  - Merge all three (total ~5,807)
  - Apply 2x EMERGENCY upsampling to fix recall gap
  - Save as data/approach2/combined_train_v3.jsonl

Run locally:
  python src/approach2/merge_mimic.py
"""

import json
import random
from pathlib import Path
from collections import Counter

SYNTHETIC_PATH = Path("data/processed/train.jsonl")       # 4,080 — proper 80% split, excludes test.jsonl
MIMIC_PATH     = Path("data/approach2/mimic_train.jsonl")  # 207 real MIMIC demo patients
FEDMML_PATH    = Path("data/approach2/fedmml_train.jsonl") # 53,139 real fedmml patients
OUTPUT_PATH    = Path("data/approach2/combined_train_v4.jsonl")

# No upsampling needed — fedmml already provides 17,713 real EMERGENCY cases
UPSAMPLE_FACTOR = 1
random.seed(42)

SYSTEM_PROMPT = (
    "You are a senior emergency physician. Given a patient description, "
    "classify the triage level.\n\n"
    "Definitions:\n"
    "EMERGENCY: immediately life-threatening — requires intervention within minutes\n"
    "URGENT: serious but stable — requires evaluation within 1-2 hours\n"
    "ROUTINE: non-urgent — can be seen in a scheduled appointment\n\n"
    "Respond with ONLY one word: EMERGENCY, URGENT, or ROUTINE."
)

SYNTECH_LABEL_MAP = {
    "immediate": "EMERGENCY",
    "urgent":    "URGENT",
    "routine":   "ROUTINE",
    # already-mapped values pass through unchanged
    "EMERGENCY": "EMERGENCY",
    "URGENT":    "URGENT",
    "ROUTINE":   "ROUTINE",
}


def print_dist(label: str, samples: list):
    dist  = Counter(s["triage_level"] for s in samples)
    total = len(samples)
    print(f"  {label}: {total} samples")
    for cls in ["EMERGENCY", "URGENT", "ROUTINE"]:
        n = dist[cls]
        pct = n / total * 100 if total else 0
        print(f"    {cls:<12}: {n:>5}  ({pct:.1f}%)")


def load_synthetic(path: Path) -> list:
    """Convert train_samples.jsonl to instruction-tuning format."""
    samples = []
    for line in open(path):
        s = json.loads(line)
        samples.append({
            "instruction":  SYSTEM_PROMPT,
            "input":        s["symptom_description"],
            "output":       s["raw_output"].strip(),
            "triage_level": s["triage_level"],
            "source":       "synthetic",
        })
    return samples


def load_syntech() -> list:
    """Download syntech-ai/medical-triage-500 from HuggingFace and convert.

    Schema:
      triage_classification.urgency_category  -> label (immediate/urgent/routine)
      patient.age, patient.gender             -> demographics
      presentation.symptoms                   -> symptom list
      presentation.duration, onset            -> context
    """
    try:
        from datasets import load_dataset
        ds = load_dataset(
            "syntech-ai/medical-triage-500",
            data_files="medical_triage_500.jsonl",
            split="train",
        )
        samples = []
        for row in ds:
            # label
            raw_label = row.get("triage_classification", {}).get("urgency_category", "").lower().strip()
            label = SYNTECH_LABEL_MAP.get(raw_label)
            if label is None:
                continue

            # build description
            pt   = row.get("patient", {})
            pres = row.get("presentation", {})
            age    = pt.get("age", "")
            gender = pt.get("gender", "patient")
            syms   = pres.get("symptoms", [])
            dur    = pres.get("duration", "")
            onset  = pres.get("onset", "")

            sym_str = ", ".join(syms) if syms else "unspecified symptoms"
            desc = f"A {age}-year-old {gender} presenting with {sym_str}."
            if dur:
                desc += f" Duration: {dur}."
            if onset:
                desc += f" Onset: {onset}."

            output = (
                f"TRIAGE LEVEL: {label}\n"
                f"KEY SYMPTOMS: {sym_str}\n"
                f"CLINICAL REASONING: Triage case with urgency category '{raw_label}'. "
                f"Patient presents with {sym_str}.\n"
                f"CONFIDENCE: HIGH"
            )
            samples.append({
                "instruction":  SYSTEM_PROMPT,
                "input":        desc,
                "output":       output,
                "triage_level": label,
                "source":       "syntech",
            })
        return samples
    except Exception as e:
        print(f"  WARNING: Could not load syntech-500 ({e}). Skipping.")
        return []


def load_mimic(path: Path) -> list:
    return [json.loads(l) for l in open(path)]


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # --- Load all four sources ---
    print("Loading sources...")

    synthetic = load_synthetic(SYNTHETIC_PATH)
    print_dist("train.jsonl (synthetic)", synthetic)

    print("  Downloading syntech-ai/medical-triage-500...")
    syntech = load_syntech()
    if syntech:
        print_dist("syntech-ai/medical-triage-500", syntech)
    else:
        print("  syntech-500 skipped.")

    mimic = load_mimic(MIMIC_PATH)
    print_dist("mimic_train.jsonl (MIMIC demo)", mimic)

    fedmml = load_mimic(FEDMML_PATH)  # same JSONL format
    print_dist("fedmml_train.jsonl (fedmml real)", fedmml)

    # --- Merge all four sources ---
    merged = synthetic + syntech + mimic + fedmml
    print()
    print("=== After merge ===")
    print_dist("merged", merged)

    # --- 2x EMERGENCY upsampling ---
    emergency = [s for s in merged if s["triage_level"] == "EMERGENCY"]
    upsampled = merged + emergency * (UPSAMPLE_FACTOR - 1)
    random.shuffle(upsampled)

    if UPSAMPLE_FACTOR > 1:
        print()
        print(f"=== After {UPSAMPLE_FACTOR}x EMERGENCY upsampling ===")
        print_dist("combined_train_v4.jsonl", upsampled)
    else:
        print("  (no upsampling — fedmml provides balanced EMERGENCY coverage)")

    # --- Save ---
    with open(OUTPUT_PATH, "w") as f:
        for s in upsampled:
            f.write(json.dumps(s) + "\n")

    print(f"\nSaved -> {OUTPUT_PATH}")
    print("\nNext steps:")
    print("  1. scp data/approach2/combined_train_v4.jsonl  root@container:/root/model_miniaturization/data/approach2/")
    print("  2. On container: python lora_finetune.py --train data/approach2/combined_train_v4.jsonl --output_dir data/approach2/qwen3_lora_v4")
    print("  3. On container: python evaluate_student.py --adapter data/approach2/qwen3_lora_v4/adapter --output_dir data/approach2/v4")


if __name__ == "__main__":
    main()
