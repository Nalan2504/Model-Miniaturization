"""
NLI Consistency Filter — Approach 2, Step 1

Scores each synthetic sample by checking whether the extracted clinical
reasoning ENTAILS a label-specific hypothesis.  Samples with entailment
score above --threshold are kept.

Model : cross-encoder/nli-deberta-v3-large  (3-class: contradiction/neutral/entailment)
Input : data/synthetic/train_samples.jsonl  (5,100 samples)
Output: data/approach2/filtered_samples.jsonl

Usage (container):
  python src/approach2/nli_filter.py --threshold 0.5 --batch_size 64
"""

import json
import re
import argparse
import torch
from pathlib import Path
from collections import defaultdict, Counter
from transformers import AutoTokenizer, AutoModelForSequenceClassification

DATA_DIR   = Path("/root/model_miniaturization/data")
INPUT_PATH = DATA_DIR / "synthetic/train_samples.jsonl"
OUTPUT_DIR = DATA_DIR / "approach2"

MODEL_ID   = "cross-encoder/nli-deberta-v3-large"

# Label-specific hypotheses — the reasoning should ENTAIL this for the given triage class
HYPOTHESES = {
    "EMERGENCY": "This patient requires immediate emergency intervention within minutes to prevent death or serious harm.",
    "URGENT":    "This patient has a serious condition that requires medical evaluation within the next one to two hours.",
    "ROUTINE":   "This patient has a non-urgent condition that can be managed in a routine scheduled appointment.",
}

# DeBERTa NLI label order: contradiction=0, neutral=1, entailment=2
ENTAILMENT_IDX = 2


def extract_reasoning(raw_output: str) -> str:
    """Extract only the CLINICAL REASONING block — same logic as evaluate_reasoning_quality.py."""
    match = re.search(r'CLINICAL REASONING:\s*(.*?)(?:CONFIDENCE:|$)', raw_output, re.DOTALL)
    if match:
        return match.group(1).strip()
    lines = raw_output.strip().splitlines()
    reasoning_lines = [
        l for l in lines
        if not l.startswith("TRIAGE LEVEL:")
        and not l.startswith("KEY SYMPTOMS:")
        and not l.startswith("CONFIDENCE:")
    ]
    return "\n".join(reasoning_lines).strip()


def load_samples(path: Path):
    samples = []
    for line in open(path):
        s = json.loads(line)
        reasoning = extract_reasoning(s["raw_output"])
        if len(reasoning) > 50:
            s["_reasoning"] = reasoning
            samples.append(s)
        else:
            s["_reasoning"] = ""
            samples.append(s)
    return samples


def score_batch(model, tokenizer, premises, hypotheses, device):
    """Return softmaxed 3-class scores for a batch of (premise, hypothesis) pairs."""
    enc = tokenizer(
        premises,
        hypotheses,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    ).to(device)
    with torch.inference_mode():
        logits = model(**enc).logits            # (B, 3)
    probs = torch.softmax(logits, dim=-1)       # (B, 3)
    return probs[:, ENTAILMENT_IDX].cpu().tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold",  type=float, default=0.5,  help="Min entailment probability to keep sample")
    parser.add_argument("--batch_size", type=int,   default=64,   help="NLI inference batch size")
    parser.add_argument("--input",      default=str(INPUT_PATH))
    parser.add_argument("--output_dir", default=str(OUTPUT_DIR))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"NLI model: {MODEL_ID}")
    print(f"Threshold: {args.threshold}")

    # Load model
    print("Loading NLI model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID).to(device)
    model.eval()
    print(f"  Model loaded. Parameters: {sum(p.numel() for p in model.parameters())/1e6:.0f}M")

    # Load synthetic samples
    print(f"\nLoading samples from {args.input}...")
    samples = load_samples(Path(args.input))
    print(f"  {len(samples)} samples loaded")

    # Build premises/hypotheses pairs
    premises    = [s["_reasoning"][:800] for s in samples]   # cap at 800 chars — same as BERTScore eval
    hypotheses  = [HYPOTHESES[s["triage_level"]] for s in samples]

    # Batch inference
    print(f"\nRunning NLI inference (batch_size={args.batch_size})...")
    entailment_scores = []
    n = len(samples)
    for i in range(0, n, args.batch_size):
        batch_p = premises[i : i + args.batch_size]
        batch_h = hypotheses[i : i + args.batch_size]
        scores  = score_batch(model, tokenizer, batch_p, batch_h, device)
        entailment_scores.extend(scores)
        if (i // args.batch_size) % 10 == 0:
            pct = min(i + args.batch_size, n) / n * 100
            print(f"  [{pct:5.1f}%] processed {min(i + args.batch_size, n)}/{n}")

    # Attach scores and filter
    kept     = []
    rejected = []
    for s, score in zip(samples, entailment_scores):
        s["nli_entailment"] = round(score, 4)
        s.pop("_reasoning", None)
        if score >= args.threshold:
            kept.append(s)
        else:
            rejected.append(s)

    # Stats
    kept_by_class     = Counter(s["triage_level"] for s in kept)
    rejected_by_class = Counter(s["triage_level"] for s in rejected)
    all_by_class      = Counter(s["triage_level"] for s in samples)

    print(f"\n{'='*55}")
    print(f"NLI FILTER RESULTS  (threshold={args.threshold})")
    print(f"{'='*55}")
    print(f"{'Class':<12} {'Total':>7} {'Kept':>7} {'Rejected':>10} {'Keep%':>8}")
    print("-" * 46)
    for label in ["EMERGENCY", "URGENT", "ROUTINE"]:
        total = all_by_class[label]
        k     = kept_by_class[label]
        r     = rejected_by_class[label]
        pct   = k / total * 100 if total > 0 else 0
        print(f"{label:<12} {total:>7} {k:>7} {r:>10} {pct:>7.1f}%")
    print(f"{'TOTAL':<12} {len(samples):>7} {len(kept):>7} {len(rejected):>10} {len(kept)/len(samples)*100:>7.1f}%")

    # Score distribution
    scores_arr = [s["nli_entailment"] for s in samples]
    import statistics
    print(f"\nEntailment score stats:")
    print(f"  mean={statistics.mean(scores_arr):.4f}  median={statistics.median(scores_arr):.4f}")
    print(f"  min={min(scores_arr):.4f}  max={max(scores_arr):.4f}")

    # Save filtered samples
    out_path = output_dir / f"filtered_samples_t{int(args.threshold*100):02d}.jsonl"
    with open(out_path, "w") as f:
        for s in kept:
            f.write(json.dumps(s) + "\n")
    print(f"\nFiltered samples → {out_path}  ({len(kept)} samples)")

    # Save rejected samples (for analysis)
    rej_path = output_dir / f"rejected_samples_t{int(args.threshold*100):02d}.jsonl"
    with open(rej_path, "w") as f:
        for s in rejected:
            f.write(json.dumps(s) + "\n")
    print(f"Rejected samples  → {rej_path}  ({len(rejected)} samples)")

    # Save summary JSON
    summary = {
        "model": MODEL_ID,
        "threshold": args.threshold,
        "input": args.input,
        "total": len(samples),
        "kept": len(kept),
        "rejected": len(rejected),
        "keep_rate": len(kept) / len(samples),
        "kept_by_class": dict(kept_by_class),
        "rejected_by_class": dict(rejected_by_class),
        "score_mean":   round(statistics.mean(scores_arr),   4),
        "score_median": round(statistics.median(scores_arr), 4),
        "score_min":    round(min(scores_arr),               4),
        "score_max":    round(max(scores_arr),               4),
    }
    sum_path = output_dir / f"nli_filter_summary_t{int(args.threshold*100):02d}.json"
    with open(sum_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary           → {sum_path}")

    # Quick sanity check — print 3 kept + 3 rejected examples
    print(f"\n--- 3 KEPT samples (score >= {args.threshold}) ---")
    for s in kept[:3]:
        reasoning = extract_reasoning(s["raw_output"])
        print(f"  [{s['triage_level']}] score={s['nli_entailment']}  reasoning: {reasoning[:100]}...")

    print(f"\n--- 3 REJECTED samples (score < {args.threshold}) ---")
    for s in rejected[:3]:
        reasoning = extract_reasoning(s["raw_output"])
        print(f"  [{s['triage_level']}] score={s['nli_entailment']}  reasoning: {reasoning[:100]}...")


if __name__ == "__main__":
    main()
