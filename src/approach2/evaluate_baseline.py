"""
Evaluate base Qwen3-0.6B (zero-shot, no LoRA) on the reserved test set.

This establishes the untuned baseline so every fine-tuning method has a
measurable baseline->tuned delta for the comparative study.

Uses the SAME system prompt, evaluation harness, and test set as
evaluate_student.py — results are directly comparable.

Input : data/processed/test.jsonl   (510 reserved samples)
Output: data/approach2/baseline_eval.jsonl
        data/approach2/baseline_eval_summary.json

Usage (container):
  python src/approach2/evaluate_baseline.py
"""

import json
import torch
import argparse
import numpy as np
from pathlib import Path
from collections import Counter
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset

LABEL_MAP = {
    "immediate": "EMERGENCY",
    "urgent":    "URGENT",
    "routine":   "ROUTINE",
}

def build_syntech_description(sample: dict) -> str:
    p    = sample["patient"]
    pres = sample["presentation"]
    risk = sample["risk_assessment"]
    symptoms = ", ".join(pres["symptoms"])
    flags    = ", ".join(risk["red_flags"]) if risk["red_flags"] else "none"
    return (
        f"A {p['age']}-year-old {p['gender']} presenting with {symptoms}. "
        f"Duration: {pres['duration']}. Onset: {pres['onset']}. "
        f"Context: {pres['context']}. Red flags: {flags}."
    )


DATA_DIR   = Path("/root/model_miniaturization/data")
TEST_PATH  = DATA_DIR / "processed/test.jsonl"
OUTPUT_DIR = DATA_DIR / "approach2"

BASE_MODEL = "Qwen/Qwen3-0.6B"

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
        if label in text:
            return label
    return None


def predict_batch(model, tokenizer, prompts, device, max_new_tokens=50):
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


def print_report(metrics, n_total, n_failed):
    print(f"\n{'='*60}")
    print(f"BASE MODEL ZERO-SHOT EVALUATION — Qwen3-0.6B (no LoRA)")
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
    print(f"\nBaseline established. Compare against fine-tuned model results.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test",       default=str(TEST_PATH))
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--output_dir", default=str(OUTPUT_DIR))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Model  : {BASE_MODEL} (base, no LoRA adapter)")
    print(f"Device : {device}")
    print(f"Test   : {args.test}")

    # Load base model only — no LoRA adapter
    print(f"\nLoading {BASE_MODEL} in 4-bit NF4 (base model, no adapter)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    print(f"  Loaded. VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    # Load test set
    print(f"\nLoading test set from {args.test}...")
    is_syntech = (args.test == "syntech-500")
    if is_syntech:
        print("Loading syntech-ai/medical-triage-500 from HuggingFace...")
        test_samples = load_dataset("syntech-ai/medical-triage-500", data_files="medical_triage_500.jsonl", split="train")
        label_dist = Counter(LABEL_MAP[s["triage_classification"]["urgency_category"]] for s in test_samples)
    else:
        test_samples = [json.loads(l) for l in open(args.test)]
        label_dist = Counter(s["triage_level"] for s in test_samples)
    print(f"  {len(test_samples)} samples | {dict(label_dist)}")

    # Predict
    print(f"\nRunning zero-shot inference (batch_size={args.batch_size})...")
    results = []
    failed  = 0

    for i in range(0, len(test_samples), args.batch_size):
        if is_syntech:
            batch = [test_samples[j] for j in range(i, min(i + args.batch_size, len(test_samples)))]
        else:
            batch = test_samples[i : i + args.batch_size]
        prompts = []
        for s in batch:
            if is_syntech:
                desc = build_syntech_description(s)
            else:
                desc = s.get("symptom_description") or s.get("input") or ""
            prompts.append(build_prompt(desc))
        raw_outputs = predict_batch(model, tokenizer, prompts, device)

        for s, raw in zip(batch, raw_outputs):
            pred = extract_label(raw)
            if pred is None:
                failed += 1
                pred = "UNKNOWN"
            if is_syntech:
                true_label = LABEL_MAP[s["triage_classification"]["urgency_category"]]
            else:
                true_label = s["triage_level"]
            results.append({
                "triage_level": true_label,
                "true":         true_label,
                "pred":         pred,
                "raw_output":   raw,
                "correct":      pred == true_label,
            })

        if (i // args.batch_size) % 5 == 0:
            done = min(i + args.batch_size, len(test_samples))
            acc  = sum(r["correct"] for r in results) / len(results) * 100
            print(f"  [{done}/{len(test_samples)}] acc={acc:.1f}% failed={failed}")

    # Metrics
    valid   = [r for r in results if r["pred"] != "UNKNOWN"]
    metrics = compute_metrics(valid)
    print_report(metrics, len(test_samples), failed)

    # Detect dataset name for unique output files
    if is_syntech:
        ds_name = "syntech"
    elif "mimic" in args.test.lower():
        ds_name = "mimic"
    elif "fedmml" in args.test.lower() or "latvia" in args.test.lower():
        ds_name = "latvia"
    else:
        ds_name = "synthetic"

    # Save per-sample results
    out_path = output_dir / f"baseline_{ds_name}_eval.jsonl"
    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"\nPer-sample results -> {out_path}")

    # Save summary
    summary = {
        "model":             BASE_MODEL,
        "adapter":           None,
        "description":       "Zero-shot baseline — no LoRA, no fine-tuning",
        "test_set":          args.test,
        "n_total":           len(test_samples),
        "n_failed":          failed,
        "accuracy":          metrics["accuracy"],
        "macro_f1":          metrics["macro_f1"],
        "emergency_recall":  metrics["EMERGENCY"]["recall"],
        "per_class":         {l: metrics[l] for l in LABELS},
    }
    sum_path = output_dir / f"baseline_{ds_name}_eval_summary.json"
    with open(sum_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary           -> {sum_path}")


if __name__ == "__main__":
    main()
