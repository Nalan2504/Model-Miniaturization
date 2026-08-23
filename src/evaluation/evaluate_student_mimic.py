import json
import argparse
from pathlib import Path
from collections import Counter

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

SYSTEM_PROMPT = (
    "You are a senior emergency physician. Given a patient description, "
    "classify the triage level.\n\n"
    "Definitions:\n"
    "EMERGENCY: immediately life-threatening — requires intervention within minutes\n"
    "URGENT: serious but stable — requires evaluation within 1-2 hours\n"
    "ROUTINE: non-urgent — can be seen in a scheduled appointment\n\n"
    "Respond with ONLY one word: EMERGENCY, URGENT, or ROUTINE."
)

def build_prompt(description: str) -> str:
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\nPatient: {description}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

def extract_label(text: str):
    text = text.strip().upper()
    for label in ["EMERGENCY", "URGENT", "ROUTINE"]:
        if label in text:
            return label
    return None

def load_jsonl(path: Path):
    rows = []
    with open(path, "r") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows

def compute_metrics(results):
    labels = ["EMERGENCY", "URGENT", "ROUTINE"]
    metrics = {}
    for label in labels:
        tp = sum(1 for r in results if r["true"] == label and r["pred"] == label)
        fp = sum(1 for r in results if r["true"] != label and r["pred"] == label)
        fn = sum(1 for r in results if r["true"] == label and r["pred"] != label)
        prec   = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1     = 2 * prec * recall / (prec + recall) if (prec + recall) > 0 else 0
        metrics[label] = {"precision": prec, "recall": recall, "f1": f1}
    correct = sum(1 for r in results if r["true"] == r["pred"])
    metrics["accuracy"] = correct / len(results) if results else 0.0
    metrics["macro_f1"] = sum(metrics[l]["f1"] for l in labels) / len(labels)
    return metrics

def print_metrics(metrics, n_total, n_failed):
    print(f"\n=== STUDENT EVAL ON MIMIC ===")
    print(f"Total samples : {n_total}")
    print(f"Failed/unparsed: {n_failed}")
    print(f"Accuracy      : {metrics['accuracy']*100:.1f}%")
    print(f"Macro F1      : {metrics['macro_f1']:.3f}")
    print(f"\n{'Label':<12} {'Precision':>10} {'Recall':>10} {'F1':>8}")
    print("-" * 44)
    for label in ["EMERGENCY", "URGENT", "ROUTINE"]:
        m = metrics[label]
        print(f"{label:<12} {m['precision']*100:>9.1f}% {m['recall']*100:>9.1f}% {m['f1']:>8.3f}")
    print(f"\nEmergency recall: {metrics['EMERGENCY']['recall']*100:.1f}%")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", required=True,
                        help="Path to student LoRA adapter or model dir")
    parser.add_argument("--test_path", default="data/approach2/mimic_test.jsonl")
    parser.add_argument("--output", default="data/approach2/student_mimic_eval.jsonl")
    args = parser.parse_args()

    test_path = Path(args.test_path)
    rows = load_jsonl(test_path)
    print(f"Loaded MIMIC test set: {len(rows)} samples")

    print(f"Loading student model from {args.adapter}...")
    tokenizer = AutoTokenizer.from_pretrained(args.adapter, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.adapter,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    model.eval()

    results = []
    failed = 0

    for i, sample in enumerate(rows, 1):
        true_label = sample.get("triage_level")
        description = sample.get("input", "")

        prompt = build_prompt(description)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            out = model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        decoded = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        pred_label = extract_label(decoded)

        if pred_label is None:
            failed += 1
            pred_label = "UNKNOWN"

        results.append({
            "idx": i,
            "description": description,
            "true": true_label,
            "pred": pred_label,
            "raw_output": decoded.strip(),
            "correct": pred_label == true_label,
        })

        if i % 10 == 0 or i <= 5:
            acc_so_far = (
                sum(r["correct"] for r in results if r["pred"] != "UNKNOWN") /
                max(1, sum(1 for r in results if r["pred"] != "UNKNOWN"))
            ) * 100
            print(f"[{i}/{len(rows)}] acc={acc_so_far:.1f}% failed={failed}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"\nStudent MIMIC results saved to {out_path}")

    eval_results = [r for r in results if r["pred"] != "UNKNOWN"]
    metrics = compute_metrics(eval_results)
    print_metrics(metrics, len(rows), failed)

if __name__ == "__main__":
    main()