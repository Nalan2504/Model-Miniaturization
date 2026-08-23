# python src/evaluation/evaluate_teacher_n.py \
#   --test_path data/approach2/fedmml_test.jsonl \
#   --output /root/model_miniaturization/data/evaluation/teacher_latvia_eval.jsonl

import torch
import json
import re
import argparse
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from collections import Counter

MODEL_ID = "aaditya/OpenBioLLM-Llama3-8B"
OUTPUT_DIR = Path("/root/model_miniaturization/data/evaluation")

SYSTEM_PROMPT = """You are a senior emergency physician. Given a patient description, classify the triage level.

Definitions:
EMERGENCY: immediately life-threatening — requires intervention within minutes
URGENT: serious but stable — requires evaluation within 1-2 hours
ROUTINE: non-urgent — can be seen in a scheduled appointment

Respond with ONLY one word: EMERGENCY, URGENT, or ROUTINE. No explanation."""

def build_prompt(description: str) -> str:
    return (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
        f"{SYSTEM_PROMPT}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n"
        f"Patient: {description}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n"
    )

def extract_label(text: str):
    text = text.strip().upper()
    for label in ["EMERGENCY", "URGENT", "ROUTINE"]:
        if re.search(rf"\b{label}\b", text):
            return label
    return None

def load_model():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    print(f"Loading {MODEL_ID} in 4-bit NF4...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        attn_implementation="eager",
    )
    print(f"Model loaded. VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB")
    return model, tokenizer

def predict(model, tokenizer, description: str):
    prompt = build_prompt(description)
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=10,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    decoded = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return extract_label(decoded), decoded.strip()

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
        metrics[label] = {
            "precision": prec,
            "recall": recall,
            "f1": f1,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }
    correct = sum(1 for r in results if r["true"] == r["pred"])
    metrics["accuracy"] = correct / len(results) if results else 0.0
    metrics["macro_f1"] = sum(metrics[l]["f1"] for l in labels) / len(labels) if labels else 0.0
    return metrics

def print_metrics(metrics, n_total, n_failed):
    print(f"\n{'='*55}")
    print(f"TEACHER MODEL EVALUATION — {MODEL_ID}")
    print(f"{'='*55}")
    print(f"Total samples : {n_total}")
    print(f"Failed/unparsed: {n_failed}")
    print(f"Accuracy      : {metrics['accuracy']*100:.1f}%")
    print(f"Macro F1      : {metrics['macro_f1']:.3f}")
    print(f"\n{'Label':<12} {'Precision':>10} {'Recall':>10} {'F1':>8}")
    print("-" * 44)
    for label in ["EMERGENCY", "URGENT", "ROUTINE"]:
        m = metrics[label]
        print(f"{label:<12} {m['precision']*100:>9.1f}% {m['recall']*100:>9.1f}% {m['f1']:>8.3f}")
    print(f"\n*** Emergency recall: {metrics['EMERGENCY']['recall']*100:.1f}% ***")

def load_latvia_jsonl(path: Path):
    rows = []
    with open(path, "r") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test_path",
        default="data/approach2/fedmml_test.jsonl",
        help="Path to Latvia JSONL test file",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate on N samples only",
    )
    parser.add_argument(
        "--output",
        default=str(OUTPUT_DIR / "teacher_latvia_eval.jsonl"),
        help="Output JSONL file for per-case results",
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    test_path = Path(args.test_path)
    print(f"Loading Latvia test set from {test_path}...")
    rows = load_latvia_jsonl(test_path)
    if args.limit:
        rows = rows[:args.limit]
    print(f"  {len(rows)} samples loaded")

    model, tokenizer = load_model()

    results = []
    failed = 0

    for i, sample in enumerate(rows, 1):
        true_label = sample.get("triage_level")
        description = sample.get("input", "")

        pred_label, raw = predict(model, tokenizer, description)

        if pred_label is None:
            failed += 1
            pred_label = "UNKNOWN"

        results.append({
            "idx": i,
            "description": description,
            "true": true_label,
            "pred": pred_label,
            "raw_output": raw,
            "correct": pred_label == true_label,
        })

        if i % 50 == 0 or i <= 5:
            acc_so_far = (
                sum(r["correct"] for r in results if r["pred"] != "UNKNOWN") /
                max(1, sum(1 for r in results if r["pred"] != "UNKNOWN"))
            ) * 100
            print(f"[{i}/{len(rows)}] acc={acc_so_far:.1f}% failed={failed}")

    # Save per-case results
    out_path = Path(args.output)
    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"\nResults saved to {out_path}")

    # Compute metrics ignoring UNKNOWN
    eval_results = [r for r in results if r["pred"] != "UNKNOWN"]
    metrics = compute_metrics(eval_results)
    print_metrics(metrics, len(rows), failed)

    # Save summary
    summary_path = out_path.parent / "teacher_latvia_eval_summary.json"
    with open(summary_path, "w") as f:
        json.dump(
            {
                "model": MODEL_ID,
                "n_samples": len(rows),
                "n_failed": failed,
                **metrics,
            },
            f,
            indent=2,
        )
    print(f"Summary saved to {summary_path}")

if __name__ == "__main__":
    main()