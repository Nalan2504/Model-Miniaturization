"""Merge perclass-filtered synthetic + syntech-ai into combined_train.jsonl."""
import json
import warnings
from pathlib import Path
from collections import Counter
from datasets import load_dataset

DATA_DIR   = Path("/root/model_miniaturization/data")
OUTPUT_DIR = DATA_DIR / "approach2"

SYNTECH_MAP = {"immediate": "EMERGENCY", "urgent": "URGENT", "routine": "ROUTINE"}

SYSTEM_PROMPT = """You are a senior emergency physician. Given a patient description, classify the triage level and provide clinical reasoning.

Definitions:
EMERGENCY: immediately life-threatening - requires intervention within minutes
URGENT: serious but stable - requires evaluation within 1-2 hours
ROUTINE: non-urgent - can be seen in a scheduled appointment

Respond in this exact format:
TRIAGE LEVEL: <EMERGENCY|URGENT|ROUTINE>
KEY SYMPTOMS: <comma-separated list>
CLINICAL REASONING: <step-by-step clinical reasoning>
CONFIDENCE: <HIGH|MEDIUM|LOW>"""

filtered = [json.loads(l) for l in open(OUTPUT_DIR / "filtered_samples_perclass.jsonl")]
print(f"Filtered synthetic: {len(filtered)}")

combined = []
for s in filtered:
    combined.append({
        "source": "synthetic",
        "triage_level": s["triage_level"],
        "instruction": SYSTEM_PROMPT,
        "input": s["symptom_description"],
        "output": s["raw_output"].strip(),
        "nli_entailment": s.get("nli_entailment", None),
    })

warnings.filterwarnings("ignore")
ds = load_dataset("syntech-ai/medical-triage-500", data_files="medical_triage_500.jsonl", split="train")
print(f"Syntech-ai: {len(ds)} real samples")

for s in ds:
    p    = s["patient"]
    pres = s["presentation"]
    risk = s["risk_assessment"]
    flags = ", ".join(risk["red_flags"]) if risk["red_flags"] else "none"
    syms  = ", ".join(pres["symptoms"])
    syms3 = ", ".join(pres["symptoms"][:3])
    description = (
        "A " + str(p["age"]) + "-year-old " + p["gender"] + " presenting with " +
        syms + ". Duration: " + pres["duration"] + ". Onset: " + pres["onset"] +
        ". Context: " + pres["context"] + ". Red flags: " + flags + "."
    )
    label  = SYNTECH_MAP[s["triage_classification"]["urgency_category"]]
    output = (
        "TRIAGE LEVEL: " + label + "\nKEY SYMPTOMS: " + syms3 +
        "\nCLINICAL REASONING: " + pres["context"] + " with duration " +
        pres["duration"] + " and onset " + pres["onset"] +
        ". Red flags: " + flags + ".\nCONFIDENCE: HIGH"
    )
    combined.append({
        "source": "syntech",
        "triage_level": label,
        "instruction": SYSTEM_PROMPT,
        "input": description,
        "output": output,
        "nli_entailment": None,
    })

by_class  = Counter(s["triage_level"] for s in combined)
by_source = Counter(s["source"] for s in combined)
print(f"\nFinal combined dataset: {len(combined)} samples")
print(f"  synthetic: {by_source['synthetic']}  syntech: {by_source['syntech']}")
print(f"  EMERGENCY: {by_class['EMERGENCY']}")
print(f"  URGENT:    {by_class['URGENT']}")
print(f"  ROUTINE:   {by_class['ROUTINE']}")

out = OUTPUT_DIR / "combined_train.jsonl"
with open(out, "w") as f:
    for s in combined:
        f.write(json.dumps(s) + "\n")
print(f"\nSaved -> {out}")
