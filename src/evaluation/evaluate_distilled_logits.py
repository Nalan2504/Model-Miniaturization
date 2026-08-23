"""
Evaluate Distilled Student Model using Next-Token Logits

This script evaluates the distilled student model by extracting the logits for
the target labels ("EMERGENCY", "URGENT", "ROUTINE") directly from the first
token position following the prompt, avoiding autoregressive generation collapse.

Usage:
  python src/evaluation/evaluate_distilled_logits.py \
    --model_path data/distillation/qwen3_kd_lora \
    --test_path data/processed/test.jsonl \
    --output data/distillation/kd_student_synthetic_logit_eval.json
"""

import json
import torch
import argparse
import numpy as np
import torch.nn.functional as F
from pathlib import Path
from collections import Counter, defaultdict
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

def apply_threshold(data, threshold):
    results = []
    for d in data:
        p_em, p_ur = d["p_em"], d["p_ur"]
        denom = p_em + p_ur
        if denom > 1e-9 and (p_em / denom) > threshold:
            pred = "EMERGENCY"
        else:
            pred = d["base_pred"]
        results.append({"true": d["true"], "pred": pred})
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, help="Path to distilled student model directory")
    parser.add_argument("--test_path", required=True, help="Path to test set JSONL file")
    parser.add_argument("--output", required=True, help="Path to save evaluation JSON results")
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Model  : {args.model_path}")
    print(f"Device : {device}")
    print(f"Test   : {args.test_path}")

    print(f"\nLoading model in 4-bit NF4...")
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

    # Get label token IDs
    label_toks = {}
    for lbl in LABELS:
        toks = tokenizer(lbl, add_special_tokens=False)["input_ids"]
        label_toks[lbl] = toks
        print(f"  Label {lbl} tokens: {toks} -> {[tokenizer.decode([t]) for t in toks]}")

    print(f"\nLoading test set...")
    is_syntech = (args.test_path == "syntech-500")
    if is_syntech:
        print("Loading syntech-ai/medical-triage-500 from HuggingFace...")
        test_samples_ds = load_dataset("syntech-ai/medical-triage-500", data_files="medical_triage_500.jsonl", split="train")
        test_samples = [test_samples_ds[j] for j in range(len(test_samples_ds))]
        label_dist = Counter(LABEL_MAP[s["triage_classification"]["urgency_category"]] for s in test_samples)
    else:
        test_samples = []
        with open(args.test_path, "r") as f:
            for line in f:
                test_samples.append(json.loads(line))
        label_dist = Counter(s["triage_level"] for s in test_samples)
    print(f"  {len(test_samples)} samples | {dict(label_dist)}")

    print(f"\nRunning logit extraction...")
    data = []
    
    for i in range(0, len(test_samples), args.batch_size):
        batch = test_samples[i : i + args.batch_size]
        prompts = []
        for s in batch:
            if is_syntech:
                desc = build_syntech_description(s)
            else:
                desc = s.get("symptom_description") or s.get("input") or ""
            prompts.append(build_prompt(desc))
        
        enc = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(device)
        
        with torch.no_grad():
            out = model(**enc)
            # Logits at the last token position: [batch, vocab_size]
            logits_vocab = out.logits[:, -1, :]
            
        for idx, s in enumerate(batch):
            # Compute class scores
            class_logits = []
            for lbl in LABELS:
                toks = label_toks[lbl]
                # Mean over token IDs representing the word
                class_logits.append(logits_vocab[idx, toks].mean().item())
            
            # Softmax to get probabilities
            probs = F.softmax(torch.tensor(class_logits), dim=-1).tolist()
            pred_class = LABELS[np.argmax(probs)]
            
            if is_syntech:
                true_label = LABEL_MAP[s["triage_classification"]["urgency_category"]]
            else:
                true_label = s["triage_level"]
                
            data.append({
                "true": true_label,
                "base_pred": pred_class,
                "p_em": probs[0],
                "p_ur": probs[1],
                "p_ro": probs[2],
            })

        if (i // args.batch_size) % 10 == 0:
            done = min(i + args.batch_size, len(test_samples))
            print(f"  [{done}/{len(test_samples)}] processed")

    # Evaluate default argmax predictions
    default_results = [{"true": d["true"], "pred": d["base_pred"]} for d in data]
    default_metrics = compute_metrics(default_results)
    
    print(f"\n{'='*60}")
    print(f"DISTILLED MODEL LOGIT EVALUATION (ARGMAX) — {Path(args.model_path).name}")
    print(f"Test set: {len(test_samples)} samples")
    print(f"{'='*60}")
    print(f"Accuracy  : {default_metrics['accuracy']*100:.1f}%")
    print(f"Macro F1  : {default_metrics['macro_f1']:.3f}")
    print(f"\n{'Label':<12} {'Precision':>10} {'Recall':>10} {'F1':>8}")
    print("-" * 44)
    for label in LABELS:
        m = default_metrics[label]
        print(f"{label:<12} {m['precision']*100:>9.1f}% {m['recall']*100:>9.1f}% {m['f1']:>8.3f}")
    
    # Sweep thresholds for EMERGENCY class recall target (>95%)
    print(f"\nSweeping EMERGENCY probability thresholds:")
    print(f"{'thresh':>8} {'em_rec':>8} {'em_prec':>8} {'macro_f1':>10} {'acc':>7}")
    print("-" * 48)
    
    thresholds = [round(t * 0.05, 2) for t in range(1, 11)]  # 0.05 to 0.50
    best_summary = None
    best_t = 0.5
    
    for t in thresholds:
        results = apply_threshold(data, t)
        acc, macro_f1, stats = compute_metrics(results), compute_metrics(results)["macro_f1"], compute_metrics(results)
        em = stats["EMERGENCY"]
        marker = " <--" if em["recall"] >= 0.95 and best_summary is None else ""
        print(f"  t={t:.2f}  {em['recall']*100:6.1f}%  {em['precision']*100:6.1f}%  {macro_f1:.4f}  {acc['accuracy']*100:.1f}%{marker}")
        if em["recall"] >= 0.95 and best_summary is None:
            best_summary = stats
            best_t = t

    if best_summary is None:
        print("\n[!] No threshold reached >95% emergency recall. Using default argmax.")
        best_summary = default_metrics
        best_t = 0.5

    summary = {
        "model": args.model_path,
        "test_set": args.test_path,
        "n_total": len(test_samples),
        "argmax_accuracy": default_metrics["accuracy"],
        "argmax_macro_f1": default_metrics["macro_f1"],
        "argmax_emergency_recall": default_metrics["EMERGENCY"]["recall"],
        "chosen_threshold": best_t,
        "thresholded_accuracy": best_summary["accuracy"],
        "thresholded_macro_f1": best_summary["macro_f1"],
        "thresholded_emergency_recall": best_summary["EMERGENCY"]["recall"],
        "per_class_thresholded": {l: best_summary[l] for l in LABELS},
    }
    
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nEvaluation summary saved to {output_path}")

if __name__ == "__main__":
    main()
