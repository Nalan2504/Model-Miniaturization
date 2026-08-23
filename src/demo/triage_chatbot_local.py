"""
Medical Triage Chatbot — Local Windows Version (CPU inference)
Device: Intel UHD (no CUDA) — uses torch CPU, float32, ~30-50s per response.

Setup (one-time):
  pip install peft gradio transformers accelerate

Copy adapters from container (run in PowerShell):
  scp -r ailab:/root/model_miniaturization/data/approach2/qwen3_lora_v4/adapter adapters/qwen3_sft_v4
  scp -r ailab:/root/model_miniaturization/data/pruning/qwen3_pruned_lora/adapter adapters/qwen3_pruned_sft

Run:
  python src/demo/triage_chatbot_local.py
  Open: http://localhost:7860

Speed note: 0.6B in float32 on CPU = ~30-50s per response.
For faster inference during demo, use the container's gradio.live URL instead.
"""

import time
import torch
import gradio as gr
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# ── Config ────────────────────────────────────────────────────────────────────
BASE_MODEL = "Qwen/Qwen3-0.6B"

# Local adapter paths (copy from container with scp first)
SCRIPT_DIR = Path(__file__).parent
ADAPTERS = {
    "SFT LoRA v4 (recommended)": {
        "adapter":     str(SCRIPT_DIR.parent.parent / "adapters" / "qwen3_sft_v4"),
        "description": "Fine-tuned on 42,872 samples · 88.1% acc · 95.8% EM recall @t=0.05",
    },
    "Pruned + SFT": {
        "adapter":     str(SCRIPT_DIR.parent.parent / "adapters" / "qwen3_pruned_sft"),
        "description": "40% heads pruned + 5 layers dropped + LoRA recovery · 74.4% acc · 80% EM recall",
    },
}

SYSTEM_PROMPT = """You are a senior emergency physician. Given a patient description, classify the triage level.

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

EXAMPLES = [
    "58-year-old male with sudden crushing chest pain radiating to the left arm, profuse sweating, and shortness of breath for 20 minutes. BP 90/60, HR 110, O2 sat 91%.",
    "32-year-old female with severe right lower quadrant abdominal pain, fever 38.8°C, nausea, and rebound tenderness. Pain started 8 hours ago.",
    "45-year-old male with sudden onset worst headache of his life, neck stiffness, photophobia, and confusion.",
    "67-year-old female with right-sided facial droop, slurred speech, and inability to raise right arm — started 45 minutes ago.",
    "24-year-old with mild sore throat, low-grade fever 37.6°C, and runny nose for 2 days. No difficulty swallowing.",
]

COLORS = {"EMERGENCY": "#dc2626", "URGENT": "#f97316", "ROUTINE": "#16a34a", "UNKNOWN": "#6b7280"}
ICONS  = {"EMERGENCY": "🚨", "URGENT": "⚠️", "ROUTINE": "✅", "UNKNOWN": "❓"}


# ── Model loading ─────────────────────────────────────────────────────────────
def load_model(adapter_path: str):
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float32,   # float32 for CPU (float16 is slower on x86)
        device_map="cpu",
        trust_remote_code=True,
    )
    if Path(adapter_path).exists():
        model = PeftModel.from_pretrained(base, adapter_path)
        print(f"    Adapter loaded: {adapter_path}")
    else:
        model = base
        print(f"    WARNING: adapter not found at {adapter_path} — using base model (zero-shot)")
    model.eval()
    return model, tokenizer


print("Loading models on CPU (this takes ~30-60s)...")
loaded = {}
for name, cfg in ADAPTERS.items():
    print(f"  Loading {name}...")
    t0 = time.time()
    loaded[name] = load_model(cfg["adapter"])
    print(f"    Done in {time.time()-t0:.1f}s")
print(f"\nAll models ready. Running on CPU (Intel UHD).")
print(f"Expected inference time: 30-50s per response.\n")


# ── Inference ─────────────────────────────────────────────────────────────────
def build_prompt(description: str) -> str:
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\nPatient: {description}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def extract_level(text: str) -> str:
    for level in ["EMERGENCY", "URGENT", "ROUTINE"]:
        if level in text.upper():
            return level
    return "UNKNOWN"


def make_label_html(level: str) -> str:
    color = COLORS[level]
    icon  = ICONS[level]
    return (
        f'<div style="background:{color};color:white;padding:14px 20px;'
        f'border-radius:10px;font-size:22px;font-weight:bold;text-align:center;">'
        f'{icon} {level}</div>'
    )


def run_inference(model, tokenizer, description: str) -> tuple[str, float]:
    prompt = build_prompt(description.strip())
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    t0 = time.time()
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=150,      # shorter = faster on CPU
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1,
        )
    elapsed = time.time() - t0
    response = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    return response, elapsed


def predict_single(model_name: str, description: str):
    if not description.strip():
        return "", "", ""
    if model_name not in loaded:
        return "", "Model not loaded.", ""
    model, tokenizer = loaded[model_name]
    response, elapsed = run_inference(model, tokenizer, description)
    level = extract_level(response)
    info  = f"Model: {model_name} | Device: CPU | Time: {elapsed:.1f}s"
    return make_label_html(level), response, info


def predict_both(description: str):
    if not description.strip():
        return "", "", "", ""
    names   = list(loaded.keys())
    results = []
    for n in names[:2]:
        resp, _ = run_inference(*loaded[n], description)
        results.append(resp)
    levels = [extract_level(r) for r in results]
    r1, r2 = results[0], results[1] if len(results) > 1 else ""
    l1, l2 = levels[0],  levels[1]  if len(levels)  > 1 else "UNKNOWN"
    return make_label_html(l1), r1, make_label_html(l2), r2


# ── UI ────────────────────────────────────────────────────────────────────────
names = list(loaded.keys())

with gr.Blocks(title="Medical Triage Assistant") as demo:
    gr.Markdown("""
    # 🏥 Medical Triage Assistant — Local
    **Device:** CPU (Intel UHD) · **Model:** Qwen3-0.6B + SFT LoRA v4
    *Inference time ~30-50s on CPU. For faster demo, use the container's gradio.live URL.*
    """)

    symptom_input = gr.Textbox(
        label="Patient Symptom Description",
        placeholder="Describe the patient's symptoms, vitals, and clinical presentation...",
        lines=4,
    )

    with gr.Tab("Single Model"):
        model_selector = gr.Dropdown(
            choices=names,
            value=names[0] if names else None,
            label="Select Model",
        )
        single_btn    = gr.Button("Classify Triage Level", variant="primary", size="lg")
        single_label  = gr.HTML()
        single_reason = gr.Textbox(label="Clinical Reasoning", lines=12, interactive=False)
        model_info    = gr.Textbox(label="Info", lines=1, interactive=False)

        single_btn.click(
            fn=predict_single,
            inputs=[model_selector, symptom_input],
            outputs=[single_label, single_reason, model_info],
        )
        symptom_input.submit(
            fn=predict_single,
            inputs=[model_selector, symptom_input],
            outputs=[single_label, single_reason, model_info],
        )

    with gr.Tab("Side-by-Side"):
        compare_btn = gr.Button("Compare Both Models", variant="primary", size="lg")
        with gr.Row():
            with gr.Column():
                gr.Markdown(f"### {names[0] if names else 'SFT LoRA v4'}")
                label_1     = gr.HTML()
                reasoning_1 = gr.Textbox(label="Reasoning", lines=12, interactive=False)
            with gr.Column():
                gr.Markdown(f"### {names[1] if len(names) > 1 else 'Pruned + SFT'}")
                label_2     = gr.HTML()
                reasoning_2 = gr.Textbox(label="Reasoning", lines=12, interactive=False)

        compare_btn.click(
            fn=predict_both,
            inputs=symptom_input,
            outputs=[label_1, reasoning_1, label_2, reasoning_2],
        )

    with gr.Tab("Results Reference"):
        gr.Markdown("""
        ### MIMIC-IV-ED Results (42 patients, OOD test set)
        | Model | Accuracy | EM Recall | Notes |
        |---|---|---|---|
        | LogReg (TF-IDF) | 88.1% | **95.8%** | Keyword matching |
        | SFT v4 (argmax) | **88.1%** | 83.3% | Best balanced |
        | SFT v4 (t=0.05) | 73.8% | **95.8%** | Matches LogReg ✅ |
        | Pruned+SFT | 74.4% | 80.0% | Pruning costs recall |
        | KD v1 (argmax) | 42.9% | 0% | EMERGENCY-averse collapse |
        | KD v2 (argmax) | 57.1% | **100%** | EMERGENCY-aggressive (overcorrected) |

        **EM Recall** = recall on EMERGENCY class — the safety-critical metric.
        13× compression (8B → 0.6B) with matching EM recall at threshold t=0.05.
        """)

    gr.Markdown("### Example Cases")
    with gr.Row():
        for ex in EXAMPLES[:3]:
            gr.Button(ex[:55] + "…", size="sm").click(fn=lambda x=ex: x, outputs=symptom_input)
    with gr.Row():
        for ex in EXAMPLES[3:]:
            gr.Button(ex[:55] + "…", size="sm").click(fn=lambda x=ex: x, outputs=symptom_input)

    gr.Markdown("---\n*University of Passau — Applied AI Lab SS2026 · Model Miniaturization*")

demo.launch(server_name="0.0.0.0", server_port=7860, share=False, theme=gr.themes.Soft())
