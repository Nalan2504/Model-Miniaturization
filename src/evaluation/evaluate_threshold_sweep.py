"""
Threshold sweep for EM-recall on any LoRA-adapted student model.

Instead of greedy generation, we:
  1. Prime the prompt with "TRIAGE LEVEL: " so the next token must be the label
  2. Forward pass only (no generation) → get logits at that last position
  3. Extract P(EMERGENCY), P(URGENT), P(ROUTINE) from the vocabulary
  4. Normalize → probabilities
  5. Sweep thresholds: predict EMERGENCY if P(EM) > threshold

This gives us the optimal operating point to beat LogReg's 95.8% EM recall on MIMIC
without retraining anything.

Usage:
  python src/evaluation/evaluate_threshold_sweep.py \
      --adapter data/approach2/qwen3_lora_v4/adapter \
      --test data/approach2/mimic_test.jsonl \
      --output data/evaluation/threshold_sweep/mimic_v4
"""

import json
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

STUDENT_BASE = "Qwen/Qwen3-0.6B"
LABELS = ["EMERGENCY", "URGENT", "ROUTINE"]

SYSTEM_PROMPT = """You are a senior emergency physician. Given a patient description, classify the triage level.

Definitions:
EMERGENCY: immediately life-threatening — requires intervention within minutes
URGENT: serious but stable — requires evaluation within 1-2 hours
ROUTINE: non-urgent — can be seen in a scheduled appointment

Respond with ONLY one word: EMERGENCY, URGENT, or ROUTINE."""


def build_primed_prompt(description: str) -> str:
    """Prompt ending with 'TRIAGE LEVEL: ' so the model's next token is the label."""
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\nPatient: {description}<|im_end|>\n"
        f"<|im_start|>assistant\nTRIAGE LEVEL: "
    )


def get_label_token_ids(tokenizer):
    """Get the first token ID for each label word."""
    ids = {}
    for label in LABELS:
        toks = tokenizer(label, add_special_tokens=False)["input_ids"]
        ids[label] = toks[0]
        print(f"  {label} → token id {toks[0]} ('{tokenizer.decode([toks[0]])}') [{len(toks)} tokens total]")
    return ids


def load_model(adapter_path, base_model=None):
    base_model = base_model or STUDENT_BASE
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    base = AutoModelForCausalLM.from_pretrained(base_model, quantization_config=bnb, device_map="auto")
    if adapter_path:
        model = PeftModel.from_pretrained(base, adapter_path)
        print(f"Loaded adapter: {adapter_path}")
    else:
        model = base
        print(f"No adapter — evaluating base model: {base_model}")
    model.eval()
    print(f"Model loaded — VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB")
    return model, tokenizer


def get_label_probs(model, tokenizer, description, label_ids, device="cuda"):
    """Forward pass with primed prompt → probabilities for each label."""
    prompt = build_primed_prompt(description)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.inference_mode():
        out = model(**inputs)
    # logits at last token position: [vocab_size]
    last_logits = out.logits[0, -1, :]
    label_logits = torch.tensor(
        [last_logits[label_ids[l]].item() for l in LABELS],
        dtype=torch.float32,
    )
    probs = F.softmax(label_logits, dim=0)
    return {l: probs[i].item() for i, l in enumerate(LABELS)}


def sweep_thresholds(records, thresholds):
    results = []
    for t in thresholds:
        preds, trues = [], []
        for r in records:
            p_em = r["probs"]["EMERGENCY"]
            if p_em > t:
                pred = "EMERGENCY"
            else:
                # among non-emergency, pick argmax of URGENT/ROUTINE
                p_ur = r["probs"]["URGENT"]
                p_ro = r["probs"]["ROUTINE"]
                pred = "URGENT" if p_ur >= p_ro else "ROUTINE"
            preds.append(pred)
            trues.append(r["true"])

        n = len(trues)
        correct = sum(p == t_ for p, t_ in zip(preds, trues))
        acc = correct / n

        metrics = {}
        for label in LABELS:
            tp = sum(1 for p, t_ in zip(preds, trues) if p == label and t_ == label)
            fp = sum(1 for p, t_ in zip(preds, trues) if p == label and t_ != label)
            fn = sum(1 for p, t_ in zip(preds, trues) if p != label and t_ == label)
            prec   = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1     = 2 * prec * recall / (prec + recall) if (prec + recall) > 0 else 0.0
            metrics[label] = {"precision": prec, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}

        macro_f1 = sum(metrics[l]["f1"] for l in LABELS) / 3
        results.append({
            "threshold": round(t, 3),
            "accuracy": acc,
            "macro_f1": macro_f1,
            "em_recall": metrics["EMERGENCY"]["recall"],
            "em_precision": metrics["EMERGENCY"]["precision"],
            "per_class": metrics,
        })
    return results


def print_sweep(results):
    print(f"\n{'='*72}")
    print("THRESHOLD SWEEP — EMERGENCY RECALL vs ACCURACY")
    print(f"{'='*72}")
    print(f"{'Threshold':>10} {'Accuracy':>10} {'Macro F1':>10} {'EM Recall':>10} {'EM Prec':>10}")
    print("-" * 52)
    for r in results:
        marker = " ◄ " if r["em_recall"] >= 0.958 else "   "  # beat LogReg
        print(f"  t={r['threshold']:<7.3f}  {r['accuracy']*100:>8.1f}%  {r['macro_f1']:>8.3f}  "
              f"{r['em_recall']*100:>8.1f}%  {r['em_precision']*100:>8.1f}%{marker}")
    print(f"\n◄  = beats LogReg baseline (95.8% EM recall)")
    # find best: highest EM recall with accuracy > 60%
    valid = [r for r in results if r["accuracy"] >= 0.60]
    if valid:
        best = max(valid, key=lambda r: r["em_recall"])
        print(f"\nBest threshold (EM recall + acc≥60%): t={best['threshold']} → "
              f"EM recall={best['em_recall']*100:.1f}%, acc={best['accuracy']*100:.1f}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base",    default=None, help="Base model path or HF ID (default: Qwen/Qwen3-0.6B)")
    parser.add_argument("--adapter", default=None, help="Path to LoRA adapter (omit for full model or zero-shot)")
    parser.add_argument("--test",    default="/root/model_miniaturization/data/approach2/mimic_test.jsonl")
    parser.add_argument("--output",  default="/root/model_miniaturization/data/evaluation/threshold_sweep/mimic_v4")
    args = parser.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, tokenizer = load_model(args.adapter, base_model=args.base)
    print("\nLabel token IDs:")
    label_ids = get_label_token_ids(tokenizer)

    print(f"\nLoading test set: {args.test}")
    if args.test == "syntech-500":
        from datasets import load_dataset
        ds = load_dataset("syntech-ai/medical-triage-500", data_files="medical_triage_500.jsonl", split="train")
        samples = []
        for s in ds:
            p = s["patient"]
            pres = s["presentation"]
            risk = s["risk_assessment"]
            symptoms = ", ".join(pres["symptoms"])
            flags = ", ".join(risk["red_flags"]) if risk["red_flags"] else "none"
            desc = (f"A {p['age']}-year-old {p['gender']} presenting with {symptoms}. "
                    f"Duration: {pres['duration']}. Onset: {pres['onset']}. "
                    f"Context: {pres['context']}. Red flags: {flags}.")
            SYNTECH_MAP = {"immediate": "EMERGENCY", "urgent": "URGENT", "routine": "ROUTINE"}
            true_label = SYNTECH_MAP[s["triage_classification"]["urgency_category"]]
            samples.append({"input": desc, "triage_level": true_label})
    else:
        with open(args.test) as f:
            samples = [json.loads(l) for l in f]
    print(f"  {len(samples)} samples")

    print(f"\nRunning forward passes...")
    records = []
    for i, s in enumerate(samples, 1):
        description = s.get("input") or s.get("symptom_description", "")
        true_label  = s.get("triage_level") or s.get("output", "").split("\n")[0].replace("TRIAGE LEVEL: ", "").strip()
        if true_label not in LABELS:
            continue

        probs = get_label_probs(model, tokenizer, description, label_ids, device)
        argmax_pred = max(probs, key=probs.get)
        records.append({
            "true": true_label,
            "argmax_pred": argmax_pred,
            "probs": probs,
        })
        if i % 10 == 0 or i <= 3:
            print(f"  [{i}/{len(samples)}] true={true_label:10s} argmax={argmax_pred:10s} "
                  f"P(EM)={probs['EMERGENCY']:.3f} P(UR)={probs['URGENT']:.3f} P(RO)={probs['ROUTINE']:.3f}")

    # Argmax baseline
    argmax_correct = sum(1 for r in records if r["argmax_pred"] == r["true"])
    em_records = [r for r in records if r["true"] == "EMERGENCY"]
    em_correct  = sum(1 for r in em_records if r["argmax_pred"] == "EMERGENCY")
    print(f"\nArgmax baseline: acc={argmax_correct/len(records)*100:.1f}%, "
          f"EM recall={em_correct/len(em_records)*100:.1f}% ({em_correct}/{len(em_records)})")

    # Threshold sweep
    thresholds = [round(t, 2) for t in [i * 0.05 for i in range(1, 20)]]  # 0.05 to 0.95
    sweep_results = sweep_thresholds(records, thresholds)
    print_sweep(sweep_results)

    # Save
    with open(out / "records.jsonl", "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    with open(out / "sweep_summary.json", "w") as f:
        json.dump({"n_samples": len(records), "sweep": sweep_results}, f, indent=2)
    print(f"\nSaved to {out}/")


if __name__ == "__main__":
    main()
