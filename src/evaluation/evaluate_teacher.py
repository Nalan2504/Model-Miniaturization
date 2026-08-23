import torch
import json
import re
import argparse
from pathlib import Path
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from collections import Counter

MODEL_ID = "aaditya/OpenBioLLM-Llama3-8B"
OUTPUT_DIR = Path("/root/model_miniaturization/data/evaluation")

LABEL_MAP = {
    "immediate": "EMERGENCY",
    "urgent":    "URGENT",
    "routine":   "ROUTINE",
}

SYSTEM_PROMPT = """You are a senior emergency physician. Given a patient description, classify the triage level.

Definitions:
EMERGENCY: immediately life-threatening — requires intervention within minutes
URGENT: serious but stable — requires evaluation within 1-2 hours
ROUTINE: non-urgent — can be seen in a scheduled appointment

Respond with ONLY one word: EMERGENCY, URGENT, or ROUTINE. No explanation."""


def build_description(sample: dict) -> str:
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
        if re.search(rf'\b{label}\b', text):
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
        metrics[label] = {"precision": prec, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}
    correct = sum(1 for r in results if r["true"] == r["pred"])
    metrics["accuracy"] = correct / len(results)
    metrics["macro_f1"] = sum(metrics[l]["f1"] for l in labels) / 3
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
    print(f"\n*** Emergency recall: {metrics['EMERGENCY']['recall']*100:.1f}% (target: >95%) ***")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Evaluate on N samples only")
    parser.add_argument("--test_path", default=None, help="Path to local JSONL test file")
    parser.add_argument("--output", default=str(OUTPUT_DIR / "teacher_eval.jsonl"))
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # print("Loading syntech-ai/medical-triage-500...")
    # ds = load_dataset("syntech-ai/medical-triage-500", data_files="medical_triage_500.jsonl", split="train")
    # if args.limit:
    #     ds = ds.select(range(args.limit))
    # print(f"  {len(ds)} samples loaded")

    if args.test_path:
        print(f"Loading test set from {args.test_path}...")
        with open(args.test_path, "r") as f:
            ds = [json.loads(line) for line in f]
    else:
        print("Loading syntech-ai/medical-triage-500...")
        ds = load_dataset("syntech-ai/medical-triage-500", data_files="medical_triage_500.jsonl", split="train")
        if args.limit:
            ds = ds.select(range(args.limit))

    print(f"  {len(ds)} samples loaded")

    model, tokenizer = load_model()

    results = []
    failed  = 0

    for i, sample in enumerate(ds, 1):
        # true_label = LABEL_MAP[sample["triage_classification"]["urgency_category"]]
        # description = build_description(sample)
        if args.test_path:
            true_label = sample.get("triage_level")
            description = sample.get("symptom_description") or sample.get("input") or ""
        else:
            true_label = LABEL_MAP[sample["triage_classification"]["urgency_category"]]
            description = build_description(sample)

        pred_label, raw = predict(model, tokenizer, description)

        if pred_label is None:
            failed += 1
            pred_label = "UNKNOWN"

        # results.append({
        #     "case_id": sample["case_id"],
        #     "description": description,
        #     "true": true_label,
        #     "pred": pred_label,
        #     "raw_output": raw,
        #     "correct": pred_label == true_label,
        # })
        results.append({
            "description": description,
            "true": true_label,
            "pred": pred_label,
            "raw_output": raw,
            "correct": pred_label == true_label,
        })

        if i % 50 == 0 or i <= 5:
            acc_so_far = sum(r["correct"] for r in results) / len(results) * 100
            print(f"[{i}/{len(ds)}] acc={acc_so_far:.1f}% failed={failed}")

    # Save results
    with open(args.output, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"\nResults saved to {args.output}")

    metrics = compute_metrics([r for r in results if r["pred"] != "UNKNOWN"])
    print_metrics(metrics, len(ds), failed)

    # Save metrics summary
    summary_path = Path(args.output).parent / (Path(args.output).stem + "_summary.json")
    with open(summary_path, "w") as f:
        json.dump({"model": MODEL_ID, "n_samples": len(ds), "n_failed": failed, **metrics}, f, indent=2)
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
