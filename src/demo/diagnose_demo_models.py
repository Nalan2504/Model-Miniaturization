"""
Diagnostic for the live-demo models (KD v2 logit collapse + Pruned+SFT garbage).
Run on container:
  /root/envs/miniaturization/bin/python -u src/demo/diagnose_demo_models.py
"""
import torch, torch.nn.functional as F
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

ROOT = Path("/root/model_miniaturization")
BASE = "Qwen/Qwen3-0.6B"
SFT_ADAPTER    = ROOT / "data/approach2/qwen3_lora_v4/adapter"
PRUNED_BASE    = ROOT / "data/pruning/qwen3_pruned_heads_layers"
PRUNED_ADAPTER = ROOT / "data/pruning/qwen3_pruned_lora/adapter"
KD_V2          = ROOT / "data/distillation/qwen3_kd_lora_v2"
LABELS = ["EMERGENCY", "URGENT", "ROUTINE"]

LONG_SYS = """You are a senior emergency physician. Given a patient description, classify the triage level.

Definitions:
EMERGENCY: immediately life-threatening — requires intervention within minutes
URGENT: serious but stable — requires evaluation within 1-2 hours
ROUTINE: non-urgent — can be seen in a scheduled appointment

Respond with ONLY the following format:
TRIAGE LEVEL: [EMERGENCY/URGENT/ROUTINE]
KEY SYMPTOMS: [list key symptoms]
CLINICAL REASONING:
  Step 1: [initial assessment]
  Step 2: [risk factors or differentials]
  Step 3: [recommended immediate action]
CONFIDENCE: [HIGH/MEDIUM/LOW]"""

SHORT_SYS = """You are a senior emergency physician. Given a patient description, classify the triage level.

Definitions:
EMERGENCY: immediately life-threatening — requires intervention within minutes
URGENT: serious but stable — requires evaluation within 1-2 hours
ROUTINE: non-urgent — can be seen in a scheduled appointment

Respond with ONLY one word: EMERGENCY, URGENT, or ROUTINE."""

SHORT_INPUTS = ["I'm nauseous", "mild sore throat for 2 days", "slight headache"]
HIGH_INPUTS  = ["crushing chest pain radiating to left arm, sweating, SOB, BP 90/60",
                "sudden facial droop, slurred speech, can't lift right arm 45 min ago"]
LONG_LOW  = "24-year-old with mild sore throat, low-grade fever 37.6C, runny nose for 2 days. No difficulty swallowing. Otherwise well."

def bnb():
    return BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)

def prompt(sys, desc, prime=""):
    return (f"<|im_start|>system\n{sys}<|im_end|>\n"
            f"<|im_start|>user\nPatient: {desc}<|im_end|>\n"
            f"<|im_start|>assistant\n{prime}")

tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
if tok.pad_token is None: tok.pad_token = tok.eos_token

print("="*70); print("ISSUE 1 — KD v2 logit distribution (is it input-invariant?)"); print("="*70)
kd = AutoModelForCausalLM.from_pretrained(str(KD_V2), quantization_config=bnb(),
        device_map="auto", trust_remote_code=True).eval()

def kd_dist(desc, sys, prime, space):
    ids = {l: tok((" " if space else "") + l, add_special_tokens=False)["input_ids"][0] for l in LABELS}
    inp = tok(prompt(sys, desc, prime), return_tensors="pt").to(kd.device)
    with torch.inference_mode():
        logits = kd(**inp).logits[0, -1]
    p = F.softmax(torch.tensor([logits[ids[l]].item() for l in LABELS]), dim=0)
    return {l: f"{p[i]*100:5.1f}%" for i, l in enumerate(LABELS)}

print("\n[A] DEMO method (long sys + 'TRIAGE LEVEL:' prime + ' EMERGENCY' tokens):")
for d in SHORT_INPUTS + HIGH_INPUTS:
    print(f"  {kd_dist(d, LONG_SYS, 'TRIAGE LEVEL:', True)}  <- {d[:45]}")
print("\n[B] TRAINING method (short sys + no prime + 'EMERGENCY' tokens):")
for d in SHORT_INPUTS + HIGH_INPUTS:
    print(f"  {kd_dist(d, SHORT_SYS, '', False)}  <- {d[:45]}")
del kd; torch.cuda.empty_cache()

print("\n"+"="*70); print("ISSUE 2 — Pruned+SFT vs SFT generation on short & long inputs"); print("="*70)
def load(base, adapter):
    b = AutoModelForCausalLM.from_pretrained(str(base), quantization_config=bnb(),
            device_map="auto", trust_remote_code=True)
    return PeftModel.from_pretrained(b, str(adapter)).eval()

def gen(m, desc, rep=1.1, mx=180):
    inp = tok(prompt(LONG_SYS, desc), return_tensors="pt", truncation=True, max_length=512).to(m.device)
    with torch.inference_mode():
        out = m.generate(**inp, max_new_tokens=mx, do_sample=False,
                         pad_token_id=tok.eos_token_id, repetition_penalty=rep)
    return tok.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()

sft = load(BASE, SFT_ADAPTER)
pruned = load(PRUNED_BASE, PRUNED_ADAPTER)

for desc in ["I'm nauseous", "mild sore throat for 2 days", LONG_LOW,
             "crushing chest pain radiating to left arm, sweating, SOB"]:
    print(f"\n--- INPUT: {desc[:55]} ---")
    print(f"[SFT v4]     {gen(sft, desc)[:220]}")
    print(f"[Pruned+SFT] {gen(pruned, desc)[:220]}")

print("\n--- Pruned+SFT on 'I'm nauseous' with stronger repetition_penalty=1.3 ---")
print(gen(pruned, "I'm nauseous", rep=1.3)[:220])
print("\nDONE")
