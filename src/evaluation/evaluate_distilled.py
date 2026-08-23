"""
Evaluate Distilled Student Model on Triage Tasks

Usage:
  python src/evaluation/evaluate_distilled.py \
    --model_path data/distillation/qwen3_kd_lora \
    --test_path data/processed/test.jsonl \
    --output data/distillation/kd_student_synthetic_eval.jsonl
"""

import json
import re
import torch
import argparse
import numpy as np
from pathlib import Path
from collections import Counter
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

SYSTEM_PROMPT = """You are a senior emergency physician. Given a patient description, classify the triage level.

Definitions:
EMERGENCY: immediately life-threatening — requires intervention within minutes
URGENT: serious but stable — requires evaluation within 1-2 hours
ROUTINE: non-urgent — can be seen in a scheduled appointment

Respond with ONLY one word: EMERGENCY, URGENT, or ROUTINE."""

LABELS = ["EMERGENCY", "URGENT", "ROUTINE"]

def build_prompt(description: str) -> str:
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{description}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

def extract_label(text: str):
    text = text.strip().upper()
    for label in LABELS:
        if re.search(rf'\b{label}\b', text):
            return label
    return None

def predict_batch(model, tokenizer, prompts, device, max_new_tokens=150):
    enc = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    ).to(device)
    with torch.inference_mode():
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    input_len = enc["input_ids"].shape[1]
    return [
        tokenizer.decode(out[i][input_len:], skip_special_tokens=True)
        for i in range(len(prompts))
    ]


def compute_metrics(results):
    metrics = {}
    for label in LABELS:
        tp = sum(1 for r in results if r["true"] == label and r["pred"] == label)
        fp = sum(1 for r in results if r["true"] != label and r["pred"] == label)
        fn = sum(1 for r in results if r["true"] == label and r["pred"] != label)
        p  = tp / (tp + fp) if (tp + fp) > 0 else 0
        r  = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        metrics[label] = {"precision": p, "recall": r, "f1": f1}
    correct = sum(1 for r in results if r["true"] == r["pred"])
    metrics["accuracy"] = correct / len(results) if results else 0.0
    metrics["macro_f1"] = np.mean([metrics[l]["f1"] for l in LABELS])
    return metrics

def print_report(metrics, n_total, n_failed, model_name):
    print(f"\n{'='*60}")
    print(f"DISTILLED MODEL EVALUATION — {model_name}")
    print(f"Test set: {n_total} samples  |  Failed/unparsed: {n_failed}")
    print(f"{'='*60}")
    print(f"Accuracy  : {metrics['accuracy']*100:.1f}%")
    print(f"Macro F1  : {metrics['macro_f1']:.3f}")
    print(f"\n{'Label':<12} {'Precision':>10} {'Recall':>10} {'F1':>8}")
    print("-" * 44)
    for label in LABELS:
        m = metrics[label]
        print(f"{label:<12} {m['precision']*100:>9.1f}% {m['recall']*100:>9.1f}% {m['f1']:>8.3f}")
    em_r = metrics["EMERGENCY"]["recall"]
    print(f"\n*** Emergency recall: {em_r*100:.1f}% (target: >95%) ***")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, help="Path to distilled student model directory")
    parser.add_argument("--test_path", required=True, help="Path to test set JSONL file")
    parser.add_argument("--output", required=True, help="Path to save evaluation JSONL results")
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Model  : {args.model_path}")
    print(f"Device : {device}")
    print(f"Test   : {args.test_path}")

    print(f"\nLoading distilled model in 4-bit NF4...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    print(f"  Loaded. VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    print(f"\nLoading test set...")
    test_samples = []
    with open(args.test_path, "r") as f:
        for line in f:
            test_samples.append(json.loads(line))
    label_dist = Counter(s["triage_level"] for s in test_samples)
    print(f"  {len(test_samples)} samples | {dict(label_dist)}")

    print(f"\nRunning inference (batch_size={args.batch_size})...")
    results = []
    failed = 0

    for i in range(0, len(test_samples), args.batch_size):
        batch = test_samples[i : i + args.batch_size]
        prompts = [build_prompt(s.get("symptom_description") or s.get("input", "")) for s in batch]
        raw_outputs = predict_batch(model, tokenizer, prompts, device)

        for s, raw in zip(batch, raw_outputs):
            pred = extract_label(raw)
            if pred is None:
                failed += 1
                pred = "UNKNOWN"
            results.append({
                "triage_level": s["triage_level"],
                "true":         s["triage_level"],
                "pred":         pred,
                "raw_output":   raw,
                "correct":      pred == s["triage_level"],
            })

        if (i // args.batch_size) % 10 == 0:
            done = min(i + args.batch_size, len(test_samples))
            acc  = sum(r["correct"] for r in results) / len(results) * 100
            print(f"  [{done}/{len(test_samples)}] acc={acc:.1f}% failed={failed}")

    valid = [r for r in results if r["pred"] != "UNKNOWN"]
    metrics = compute_metrics(valid)
    print_report(metrics, len(test_samples), failed, Path(args.model_path).name)

    with open(output_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"\nPer-sample results -> {output_path}")

    summary = {
        "model":             args.model_path,
        "test_set":          args.test_path,
        "n_total":           len(test_samples),
        "n_failed":          failed,
        "accuracy":          metrics["accuracy"],
        "macro_f1":          metrics["macro_f1"],
        "emergency_recall":  metrics["EMERGENCY"]["recall"],
        "per_class":         {l: metrics[l] for l in LABELS},
    }
    summary_path = output_path.parent / f"{output_path.stem}_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary           -> {summary_path}")

if __name__ == "__main__":
    main()
