"""
Merge Datasets — Approach 2, Step 2

Combines NLI-filtered synthetic samples with real syntech-ai/medical-triage-500 data.
Converts everything to a unified format ready for LoRA fine-tuning.

Input:
  data/approach2/filtered_samples_t*.jsonl  (NLI-filtered synthetic)
  syntech-ai/medical-triage-500             (500 real samples from HuggingFace)

Output:
  data/approach2/combined_train.jsonl  — merged training set (instruction-tuning format)
  data/approach2/merge_summary.json    — dataset statistics

Usage (container):
  python src/approach2/merge_datasets.py --threshold 50
"""

import json
import argparse
from pathlib import Path
from collections import Counter
from datasets import load_dataset

DATA_DIR   = Path("/root/model_miniaturization/data")
OUTPUT_DIR = DATA_DIR / "approach2"

SYNTECH_MAP = {"immediate": "EMERGENCY", "urgent": "URGENT", "routine": "ROUTINE"}

SYSTEM_PROMPT = """You are a senior emergency physician. Given a patient description, classify the triage level and provide clinical reasoning.

Definitions:
EMERGENCY: immediately life-threatening — requires intervention within minutes
URGENT: serious but stable — requires evaluation within 1-2 hours
ROUTINE: non-urgent — can be seen in a scheduled appointment

Respond in this exact format:
TRIAGE LEVEL: <EMERGENCY|URGENT|ROUTINE>
KEY SYMPTOMS: <comma-separated list>
CLINICAL REASONING: <step-by-step clinical reasoning>
CONFIDENCE: <HIGH|MEDIUM|LOW>"""


def format_synthetic(sample: dict) -> dict:
    """Convert synthetic sample to instruction-tuning format."""
    return {
        "source": "synthetic",
        "triage_level": sample["triage_level"],
        "instruction": SYSTEM_PROMPT,
        "input": sample["symptom_description"],
        "output": sample["raw_output"].strip(),
        "nli_entailment": sample.get("nli_entailment", None),
    }


def format_syntech(sample: dict) -> dict:
    """Convert syntech-ai sample to instruction-tuning format."""
    p    = sample["patient"]
    pres = sample["presentation"]
    risk = sample["risk_assessment"]
    flags = ", ".join(risk["red_flags"]) if risk["red_flags"] else "none"
    description = (
        f"A {p['age']}-year-old {p['gender']} presenting with "
        f"{', '.join(pres['symptoms'])}. "
        f"Duration: {pres['duration']}. Onset: {pres['onset']}. "
        f"Context: {pres['context']}. Red flags: {flags}."
    )
    label = SYNTECH_MAP[sample["triage_classification"]["urgency_category"]]
    # Syntech only has the classification — build a minimal output
    output = f"TRIAGE LEVEL: {label}\nKEY SYMPTOMS: {', '.join(pres['symptoms'][:3])}\nCLINICAL REASONING: {pres['context']} with duration {pres['duration']} and onset {pres['onset']}. Red flags: {flags}.\nCONFIDENCE: HIGH"
    return {
        "source": "syntech",
        "triage_level": label,
        "instruction": SYSTEM_PROMPT,
        "input": description,
        "output": output,
        "nli_entailment": None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=int, default=50, help="NLI threshold used (integer, e.g. 50 for 0.5)")
    parser.add_argument("--output_dir", default=str(OUTPUT_DIR))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load filtered synthetic samples
    threshold_str = f"t{args.threshold:02d}"
    filtered_path = output_dir / f"filtered_samples_{threshold_str}.jsonl"
    if not filtered_path.exists():
        raise FileNotFoundError(f"Filtered samples not found: {filtered_path}\nRun nli_filter.py first.")

    print(f"Loading filtered synthetic samples from {filtered_path}...")
    synthetic_samples = [json.loads(l) for l in open(filtered_path)]
    print(f"  {len(synthetic_samples)} synthetic samples")

    # Load syntech-ai/medical-triage-500
    print("Loading syntech-ai/medical-triage-500...")
    ds = load_dataset(
        "syntech-ai/medical-triage-500",
        data_files="medical_triage_500.jsonl",
        split="train",
    )
    print(f"  {len(ds)} real samples")

    # Convert to unified format
    combined = []
    for s in synthetic_samples:
        combined.append(format_synthetic(s))
    for s in ds:
        combined.append(format_syntech(s))

    # Stats
    by_class  = Counter(s["triage_level"] for s in combined)
    by_source = Counter(s["source"] for s in combined)
    print(f"\n{'='*55}")
    print(f"COMBINED DATASET")
    print(f"{'='*55}")
    print(f"Total: {len(combined)} samples")
    print(f"  synthetic: {by_source['synthetic']}")
    print(f"  syntech  : {by_source['syntech']}")
    print(f"\nClass distribution:")
    for label in ["EMERGENCY", "URGENT", "ROUTINE"]:
        print(f"  {label}: {by_class[label]}")

    # Save
    out_path = output_dir / "combined_train.jsonl"
    with open(out_path, "w") as f:
        for s in combined:
            f.write(json.dumps(s) + "\n")
    print(f"\nCombined dataset → {out_path}")

    summary = {
        "threshold": args.threshold / 100,
        "filtered_path": str(filtered_path),
        "total": len(combined),
        "by_source": dict(by_source),
        "by_class": dict(by_class),
    }
    sum_path = output_dir / "merge_summary.json"
    with open(sum_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary           → {sum_path}")


if __name__ == "__main__":
    main()
