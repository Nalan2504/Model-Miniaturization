import torch
import json
import argparse
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_ID = "aaditya/OpenBioLLM-Llama3-8B"
DATA_DIR = Path("/root/model_miniaturization/data/synthetic")
INPUT_FILE = DATA_DIR / "symptoms_5k.jsonl"
OUTPUT_FILE = DATA_DIR / "train_samples.jsonl"

SYSTEM_PROMPT = """You are a senior emergency physician. You will be given a patient description and their assigned triage level. Your task is to provide a detailed clinical assessment that justifies the triage decision.

Respond in exactly this format — no prose, no extra text:

TRIAGE LEVEL: <the level provided>
KEY SYMPTOMS: <comma-separated list of clinically relevant symptoms>
CLINICAL REASONING:
  Step 1: <key clinical observation>
  Step 2: <interpretation or risk assessment>
  Step 3: <conclusion and rationale for triage level>
CONFIDENCE: [HIGH | MEDIUM | LOW]"""

TRIAGE_DEFINITIONS = {
    "EMERGENCY": "immediately life-threatening — requires intervention within minutes",
    "URGENT": "serious but stable — requires evaluation within 1-2 hours",
    "ROUTINE": "non-urgent — can be seen in a scheduled appointment",
}


def build_prompt(symptom_description: str, triage_level: str) -> tuple[str, str]:
    definition = TRIAGE_DEFINITIONS[triage_level]
    primer = f"TRIAGE LEVEL: {triage_level}\nKEY SYMPTOMS:"
    return (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
        f"{SYSTEM_PROMPT}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n"
        f"Patient: {symptom_description}\n"
        f"Assigned triage level: {triage_level} ({definition})\n"
        f"Provide your clinical assessment:<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n"
        f"{primer}"
    ), primer


def parse_output(raw: str, expected_label: str):
    raw = raw.strip()
    if len(raw) < 80:
        return None
    confidence = "HIGH"
    import re
    for c in ["HIGH", "MEDIUM", "LOW"]:
        if re.search(rf'confidence[:\s]+{c}', raw, re.IGNORECASE):
            confidence = c
            break
    return {"triage_level": expected_label, "confidence": confidence, "raw_output": raw}


def load_model():
    # TF32: faster matrix ops on Ampere GPUs (A6000), no quality loss
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

    # Use Flash Attention 2 if available, fall back to default otherwise
    try:
        import flash_attn  # noqa
        attn_impl = "flash_attention_2"
        print("Flash Attention 2 enabled")
    except ImportError:
        attn_impl = "eager"
        print("Flash Attention 2 not found, using default attention")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        attn_implementation=attn_impl,
    )
    print(f"Model loaded. VRAM used: {torch.cuda.memory_allocated()/1e9:.1f} GB")
    return model, tokenizer


def load_symptoms(path: Path, limit: int = None):
    samples = []
    with open(path) as f:
        for line in f:
            samples.append(json.loads(line))
    if limit:
        samples = samples[:limit]
    return samples


def generate_sample(model, tokenizer, symptom: str, label: str):
    prompt, primer = build_prompt(symptom, label)
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    decoded = primer + tokenizer.decode(
        out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )
    parsed = parse_output(decoded, label)
    if not parsed:
        return None
    return {
        "symptom_description": symptom,
        "triage_level": label,
        "confidence": parsed["confidence"],
        "raw_output": parsed["raw_output"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(INPUT_FILE), help="Input symptoms JSONL")
    parser.add_argument("--output", default=str(OUTPUT_FILE), help="Output training JSONL")
    parser.add_argument("--limit", type=int, default=None, help="Max samples to process")
    parser.add_argument("--resume", action="store_true", help="Resume from existing output file")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    samples = load_symptoms(input_path, args.limit)

    # Resume: skip samples already in output file
    skip = 0
    if args.resume and output_path.exists():
        with open(output_path) as f:
            skip = sum(1 for _ in f)
        print(f"Resuming from sample {skip + 1} ({skip} already done)")
    samples = samples[skip:]

    total_overall = len(samples) + skip
    print(f"Loaded {total_overall} symptom descriptions from {input_path.name}")
    print(f"Generating {len(samples)} remaining samples")

    model, tokenizer = load_model()

    results = []
    failed = 0
    write_mode = "a" if args.resume else "w"

    with open(output_path, write_mode) as f_out:
        for i, item in enumerate(samples, skip + 1):
            symptom = item["symptom_description"]
            label = item["triage_level"]

            result = generate_sample(model, tokenizer, symptom, label)

            if result:
                results.append(result)
                f_out.write(json.dumps(result) + "\n")
                f_out.flush()  # Write immediately so progress is saved on interrupt
                if i % 100 == 0 or i <= skip + 10:
                    pct = i / total_overall * 100
                    counts = {l: sum(1 for r in results if r["triage_level"] == l) for l in ["EMERGENCY", "URGENT", "ROUTINE"]}
                    print(f"[{i}/{total_overall} {pct:.1f}%] saved={skip+len(results)} failed={failed} | E={counts['EMERGENCY']} U={counts['URGENT']} R={counts['ROUTINE']}")
            else:
                failed += 1
                if i % 100 == 0 or i <= skip + 10:
                    print(f"[{i}/{total_overall}] saved={skip+len(results)} failed={failed}")

    print(f"\nDone. {len(results)}/{total_overall} samples saved to {output_path}")
    print(f"Failed: {failed}")
    for label in ["EMERGENCY", "URGENT", "ROUTINE"]:
        count = sum(1 for r in results if r["triage_level"] == label)
        print(f"  {label}: {count}")


if __name__ == "__main__":
    main()
