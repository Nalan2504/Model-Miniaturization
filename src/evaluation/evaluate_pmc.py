"""
Two-phase OOD evaluation on PMC-Patients clinical case narratives.

Phase 1: Teacher (OpenBioLLM-8B) labels N PMC cases → pseudo ground truth
Phase 2: Student (Qwen3-0.6B + v4 LoRA) labels the same cases
Metric:  student-teacher agreement + per-class breakdown + EM recall

PMC-Patients has no triage labels — teacher acts as oracle.
This measures how well distilled knowledge transfers to genuinely
messy, unstructured real clinical text.
"""

import gc
import json
import argparse
from pathlib import Path

import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

TEACHER_ID   = "aaditya/OpenBioLLM-Llama3-8B"
STUDENT_ID   = "Qwen/Qwen3-0.6B"
ADAPTER_PATH = "/root/model_miniaturization/data/approach2/qwen3_lora_v4/adapter"
OUTPUT_DIR   = Path("/root/model_miniaturization/data/evaluation/pmc")
PMC_DATASET  = "zhengyun21/PMC-Patients"
MAX_TEXT_CHARS = 600   # truncate long clinical narratives
N_SAMPLES      = 200

SYSTEM_PROMPT = """You are a senior emergency physician. Given a patient description, classify the triage level.

Definitions:
EMERGENCY: immediately life-threatening — requires intervention within minutes
URGENT: serious but stable — requires evaluation within 1-2 hours
ROUTINE: non-urgent — can be seen in a scheduled appointment

Respond with ONLY one word: EMERGENCY, URGENT, or ROUTINE. No explanation."""


def build_teacher_prompt(text: str) -> str:
    return (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
        f"{SYSTEM_PROMPT}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n"
        f"Patient: {text}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n"
    )


def build_student_prompt(text: str) -> str:
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\nPatient: {text}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def extract_label(text: str):
    text = text.strip().upper()
    for label in ["EMERGENCY", "URGENT", "ROUTINE"]:
        if label in text:
            return label
    return None


def bnb_config():
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )


def load_pmc_samples(n: int):
    print(f"Loading {PMC_DATASET} (first {n} rows via split slice)...")
    ds = load_dataset(PMC_DATASET, split=f"train[:{n*2}]")  # oversample to filter short
    samples = []
    for row in ds:
        text = (row.get("patient") or "").strip()
        if not text or len(text) < 50:
            continue
        age    = row.get("age", "")
        gender = row.get("gender", "")
        prefix = f"{age}-year-old {gender} — " if age and str(age).strip() else ""
        narrative = (prefix + text)[:MAX_TEXT_CHARS]
        samples.append({
            "patient_uid": row.get("patient_uid", str(len(samples))),
            "narrative": narrative,
        })
        if len(samples) >= n:
            break
    print(f"  Loaded {len(samples)} PMC samples")
    return samples


# ── Phase 1: Teacher ──────────────────────────────────────────────────────────

def run_teacher(samples: list) -> list:
    print(f"\n=== PHASE 1: Teacher ({TEACHER_ID}) ===")
    tokenizer = AutoTokenizer.from_pretrained(TEACHER_ID)
    model = AutoModelForCausalLM.from_pretrained(
        TEACHER_ID,
        quantization_config=bnb_config(),
        device_map="auto",
        attn_implementation="eager",
    )
    print(f"Teacher loaded — VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB")

    results = []
    failed = 0
    for i, s in enumerate(samples, 1):
        prompt = build_teacher_prompt(s["narrative"])
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to("cuda")
        with torch.inference_mode():
            out = model.generate(
                **inputs,
                max_new_tokens=20,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        raw = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        label = extract_label(raw)
        if label is None:
            failed += 1
        results.append({
            "patient_uid": s["patient_uid"],
            "narrative": s["narrative"],
            "teacher_label": label,
            "teacher_raw": raw.strip(),
        })
        if i % 50 == 0 or i <= 3:
            parsed = sum(1 for r in results if r["teacher_label"])
            print(f"  [{i}/{len(samples)}] parsed={parsed} failed={failed}")

    # free VRAM before loading student
    del model
    gc.collect()
    torch.cuda.empty_cache()
    print(f"Teacher done. Failed/unparsed: {failed}/{len(samples)}")
    return results


# ── Phase 2: Student ──────────────────────────────────────────────────────────

def run_student(results: list) -> list:
    print(f"\n=== PHASE 2: Student ({STUDENT_ID}) ===")
    
    # If a specific base is provided via STUDENT_ID
    tokenizer = AutoTokenizer.from_pretrained(STUDENT_ID)
    base_id = STUDENT_ID
    
    base = AutoModelForCausalLM.from_pretrained(
        base_id,
        quantization_config=bnb_config(),
        device_map="auto",
    )
    if "kd" not in ADAPTER_PATH:
        model = PeftModel.from_pretrained(base, ADAPTER_PATH)
    else:
        model = base
        
    model.eval()
    print(f"Student loaded — VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB")

    failed = 0
    for i, r in enumerate(results, 1):
        prompt = build_student_prompt(r["narrative"])
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to("cuda")
        with torch.inference_mode():
            out = model.generate(
                **inputs,
                max_new_tokens=20,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        raw = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        label = extract_label(raw)
        if label is None:
            failed += 1
        r["student_label"] = label
        r["student_raw"] = raw.strip()
        if i % 50 == 0 or i <= 3:
            print(f"  [{i}/{len(results)}] failed={failed}")

    del model
    gc.collect()
    torch.cuda.empty_cache()
    print(f"Student done. Failed/unparsed: {failed}/{len(results)}")
    return results


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(results: list) -> dict:
    # Only score where both teacher and student gave a valid label
    valid = [r for r in results if r["teacher_label"] and r["student_label"]]
    if not valid:
        return {}

    labels = ["EMERGENCY", "URGENT", "ROUTINE"]
    metrics = {"n_total": len(results), "n_valid": len(valid)}

    agree = sum(1 for r in valid if r["teacher_label"] == r["student_label"])
    metrics["agreement"] = agree / len(valid)

    for label in labels:
        teacher_pos = [r for r in valid if r["teacher_label"] == label]
        if not teacher_pos:
            metrics[label] = {"recall_vs_teacher": None, "n_teacher": 0}
            continue
        recall = sum(1 for r in teacher_pos if r["student_label"] == label) / len(teacher_pos)
        metrics[label] = {"recall_vs_teacher": recall, "n_teacher": len(teacher_pos)}

    metrics["emergency_recall_vs_teacher"] = metrics["EMERGENCY"]["recall_vs_teacher"]
    metrics["teacher_label_dist"] = {
        l: sum(1 for r in valid if r["teacher_label"] == l) for l in labels
    }
    return metrics


def print_metrics(m: dict):
    print(f"\n{'='*60}")
    print("PMC-PATIENTS OOD EVALUATION — Student vs Teacher")
    print(f"{'='*60}")
    print(f"Total samples      : {m['n_total']}")
    print(f"Both labels valid  : {m['n_valid']}")
    print(f"Student-Teacher Agreement: {m['agreement']*100:.1f}%")
    print(f"\nTeacher label distribution: {m['teacher_label_dist']}")
    print(f"\n{'Label':<12} {'N (teacher)':>12} {'Student Recall vs Teacher':>26}")
    print("-" * 52)
    for label in ["EMERGENCY", "URGENT", "ROUTINE"]:
        info = m[label]
        r = info["recall_vs_teacher"]
        r_str = f"{r*100:.1f}%" if r is not None else "N/A"
        print(f"{label:<12} {info['n_teacher']:>12} {r_str:>26}")
    em = m.get("emergency_recall_vs_teacher")
    if em is not None:
        print(f"\n*** Emergency recall vs teacher: {em*100:.1f}% (target: >95%) ***")


def main():
    global ADAPTER_PATH, STUDENT_ID
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=200, help="Number of PMC cases to evaluate")
    parser.add_argument("--output", default="/root/model_miniaturization/data/evaluation/pmc")
    parser.add_argument("--skip_teacher", action="store_true",
                        help="Re-use saved teacher predictions (pmc_teacher.jsonl must exist)")
    parser.add_argument("--adapter", default=ADAPTER_PATH, help="Path to student adapter")
    parser.add_argument("--base", default=STUDENT_ID, help="Path to student base model")
    args = parser.parse_args()

    ADAPTER_PATH = args.adapter
    STUDENT_ID = args.base

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    teacher_file = out / "pmc_teacher.jsonl"
    final_file   = out / "pmc_results.jsonl"
    summary_file = out / "pmc_summary.json"

    # Phase 1
    if args.skip_teacher and teacher_file.exists():
        print(f"Re-using saved teacher predictions from {teacher_file}")
        with open(teacher_file) as f:
            results = [json.loads(l) for l in f]
    else:
        samples = load_pmc_samples(args.n)
        results = run_teacher(samples)
        with open(teacher_file, "w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")
        print(f"Teacher predictions saved to {teacher_file}")

    # Phase 2
    results = run_student(results)

    # Save full results
    with open(final_file, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"Full results saved to {final_file}")

    # Metrics
    m = compute_metrics(results)
    print_metrics(m)
    with open(summary_file, "w") as f:
        json.dump(m, f, indent=2)
    print(f"Summary saved to {summary_file}")


if __name__ == "__main__":
    main()
