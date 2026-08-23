import json
import re
import argparse
from pathlib import Path
from datasets import load_dataset
from bert_score import score as bert_score
from collections import defaultdict
import random

DATA_DIR   = Path("/root/model_miniaturization/data")
OUTPUT_DIR = Path("/root/model_miniaturization/data/evaluation")
SYNTHETIC  = DATA_DIR / "synthetic/train_samples.jsonl"


def extract_reasoning(raw_output: str) -> str:
    """Strip TRIAGE LEVEL and KEY SYMPTOMS — return only the CLINICAL REASONING steps."""
    match = re.search(r'CLINICAL REASONING:\s*(.*?)(?:CONFIDENCE:|$)', raw_output, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Fallback: remove first two lines (TRIAGE LEVEL + KEY SYMPTOMS)
    lines = raw_output.strip().splitlines()
    reasoning_lines = [l for l in lines if not l.startswith("TRIAGE LEVEL:") and not l.startswith("KEY SYMPTOMS:") and not l.startswith("CONFIDENCE:")]
    return "\n".join(reasoning_lines).strip()


def load_synthetic_reasoning(path: Path, n_per_class: int = 200, seed: int = 42):
    rng = random.Random(seed)
    by_class = defaultdict(list)
    for line in open(path):
        s = json.loads(line)
        reasoning = extract_reasoning(s["raw_output"])
        if len(reasoning) > 50:
            by_class[s["triage_level"]].append(reasoning)

    samples = {}
    for label, items in by_class.items():
        rng.shuffle(items)
        samples[label] = items[:n_per_class]
    return samples


def load_pubmedqa_references(n: int = 300, seed: int = 42):
    """Load PubMedQA long answers as real clinical reference reasoning from local raw data."""
    print("Loading PubMedQA references from local pubmedqa_references.json...")
    local_path = Path("/root/model_miniaturization/data/raw/pubmedqa_references.json")
    with open(local_path, "r") as f:
        ds = json.load(f)
    rng = random.Random(seed)
    refs = []
    for text in ds:
        if isinstance(text, str) and len(text) > 100:
            refs.append(text[:800])  # cap at 800 chars — similar length to our reasoning
    rng.shuffle(refs)
    return refs[:n]


def run_bertscore(candidates, references, model_type="microsoft/deberta-xlarge-mnli"):
    """Run BERTScore. Candidates vs references must be same length."""
    # Align lengths by cycling references if needed
    if len(references) < len(candidates):
        references = (references * ((len(candidates) // len(references)) + 1))[:len(candidates)]
    else:
        references = references[:len(candidates)]

    print(f"  Running BERTScore ({model_type})...")
    print(f"  candidates={len(candidates)} | references={len(references)}")
    P, R, F1 = bert_score(
        candidates, references,
        model_type=model_type,
        lang="en",
        verbose=False,
        batch_size=32,
    )
    return {
        "precision": P.mean().item(),
        "recall":    R.mean().item(),
        "f1":        F1.mean().item(),
        "n":         len(candidates),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_per_class", type=int, default=200, help="Samples per triage class")
    parser.add_argument("--n_refs",      type=int, default=300, help="PubMedQA reference samples")
    parser.add_argument("--model",       default="roberta-large")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load synthetic reasoning (label-stripped)
    print("Loading synthetic clinical reasoning (label-stripped)...")
    by_class = load_synthetic_reasoning(SYNTHETIC, n_per_class=args.n_per_class)
    for label, items in by_class.items():
        print(f"  {label}: {len(items)} reasoning samples")
        print(f"  Example: {items[0][:120]}...")

    # Load PubMedQA references
    references = load_pubmedqa_references(n=args.n_refs)
    print(f"  {len(references)} PubMedQA references loaded")
    print(f"  Example: {references[0][:120]}...")

    # BERTScore per class
    print(f"\nBERTScore model: {args.model}")
    results = {}
    for label, candidates in by_class.items():
        print(f"\n[{label}]")
        results[label] = run_bertscore(candidates, references, model_type=args.model)
        m = results[label]
        print(f"  Precision={m['precision']:.4f} | Recall={m['recall']:.4f} | F1={m['f1']:.4f}")

    # Overall (all classes combined)
    all_candidates = [r for items in by_class.values() for r in items]
    print(f"\n[ALL CLASSES COMBINED]")
    results["overall"] = run_bertscore(all_candidates, references, model_type=args.model)
    m = results["overall"]
    print(f"  Precision={m['precision']:.4f} | Recall={m['recall']:.4f} | F1={m['f1']:.4f}")

    # Print summary
    print(f"\n{'='*55}")
    print(f"REASONING QUALITY — BERTScore vs PubMedQA")
    print(f"Model: {args.model}")
    print(f"{'='*55}")
    print(f"{'Class':<12} {'Precision':>10} {'Recall':>10} {'F1':>8}")
    print("-" * 44)
    for label in ["EMERGENCY", "URGENT", "ROUTINE", "overall"]:
        m = results[label]
        print(f"{label:<12} {m['precision']:>10.4f} {m['recall']:>10.4f} {m['f1']:>8.4f}")
    print(f"\nInterpretation: F1 > 0.85 = semantically close to real clinical reasoning")

    # Save
    out_path = OUTPUT_DIR / "reasoning_quality.json"
    with open(out_path, "w") as f:
        json.dump({
            "model": args.model,
            "reference_source": "qiaojin/PubMedQA (pqa_labeled, long_answer)",
            "n_per_class": args.n_per_class,
            "n_refs": args.n_refs,
            "results": results,
        }, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
