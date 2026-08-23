"""
Medical Triage Chatbot — Multi-Model (SFT vs Pruned+SFT side by side)
Run: /root/envs/miniaturization/bin/python src/demo/triage_chatbot_multi.py
Access: ssh -L 7860:localhost:7860 ailab  → open http://localhost:7860
        OR use the public share URL printed in the terminal

Both models load at startup (~60s). Switching between them is instant.
NOTE: KD model intentionally excluded — vocabulary collapse.
"""

import torch
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# ── Model registry (KD model excluded — vocab collapse) ───────────────────────
MODELS = {
    "SFT LoRA v4 (recommended)": {
        "base":        "Qwen/Qwen3-0.6B",
        "adapter":     "/root/model_miniaturization/data/approach2/qwen3_lora_v4/adapter",
        "description": "Fine-tuned on 42,872 samples · 88.1% acc · 83.3% EM recall (argmax)",
    },
    "Pruned + SFT": {
        "base":        "/root/model_miniaturization/data/pruning/qwen3_pruned_heads_layers",
        "adapter":     "/root/model_miniaturization/data/pruning/qwen3_pruned_lora/adapter",
        "description": "40% heads pruned + 5 layers dropped + LoRA recovery · 74.4% acc · 80.0% EM recall",
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

# ── Load both models at startup ───────────────────────────────────────────────
def bnb_config():
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

loaded_models = {}

print("Loading all models at startup...")
for name, cfg in MODELS.items():
    print(f"  Loading {name}...")
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(cfg["base"], quantization_config=bnb_config(), device_map="auto")
    model = PeftModel.from_pretrained(base, cfg["adapter"])
    model.eval()
    loaded_models[name] = (model, tok)
    print(f"    VRAM after loading: {torch.cuda.memory_allocated()/1e9:.1f} GB")

print("All models ready.")


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
    icons = {"EMERGENCY": "🚨", "URGENT": "⚠️", "ROUTINE": "✅", "UNKNOWN": "❓"}
    colors = {"EMERGENCY": "#dc2626", "URGENT": "#f97316", "ROUTINE": "#16a34a", "UNKNOWN": "#6b7280"}
    return (
        f'<div style="background:{colors[level]};color:white;padding:12px 20px;'
        f'border-radius:8px;font-size:20px;font-weight:bold;text-align:center;">'
        f'{icons[level]} {level}</div>'
    )


def run_model(model, tok, description: str):
    prompt = build_prompt(description.strip())
    inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=512).to("cuda")
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=False,
            pad_token_id=tok.eos_token_id,
            repetition_penalty=1.1,
        )
    response = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    return response


def predict_both(description: str):
    if not description.strip():
        return "", "", "", ""

    results = {}
    for name, (model, tok) in loaded_models.items():
        results[name] = run_model(model, tok, description)

    names = list(MODELS.keys())
    r1, r2 = results[names[0]], results[names[1]]
    l1, l2 = extract_level(r1), extract_level(r2)

    return make_label_html(l1), r1, make_label_html(l2), r2


def predict_single(model_name: str, description: str):
    if not description.strip() or model_name not in loaded_models:
        return "", ""
    model, tok = loaded_models[model_name]
    response = run_model(model, tok, description)
    level = extract_level(response)
    return make_label_html(level), response


# ── UI ────────────────────────────────────────────────────────────────────────
names = list(MODELS.keys())

with gr.Blocks(title="Medical Triage — Model Comparison") as demo:
    gr.Markdown("""
    # 🏥 Medical Triage Assistant — Model Comparison
    Compare **SFT LoRA** vs **Pruned + SFT** on the same patient input.
    *KD distilled model excluded from demo (vocabulary collapse under investigation).*
    """)

    symptom_input = gr.Textbox(
        label="Patient Symptom Description",
        placeholder="Describe the patient's symptoms, vitals, and clinical presentation...",
        lines=4,
    )

    with gr.Tab("Side-by-Side Comparison"):
        compare_btn = gr.Button("Compare Both Models", variant="primary", size="lg")
        with gr.Row():
            with gr.Column():
                gr.Markdown(f"### {names[0]}")
                gr.Markdown(f"*{MODELS[names[0]]['description']}*")
                label_1    = gr.HTML()
                reasoning_1 = gr.Textbox(label="Reasoning", lines=10, interactive=False)
            with gr.Column():
                gr.Markdown(f"### {names[1]}")
                gr.Markdown(f"*{MODELS[names[1]]['description']}*")
                label_2    = gr.HTML()
                reasoning_2 = gr.Textbox(label="Reasoning", lines=10, interactive=False)

        compare_btn.click(
            fn=predict_both,
            inputs=symptom_input,
            outputs=[label_1, reasoning_1, label_2, reasoning_2],
        )

    with gr.Tab("Single Model"):
        model_selector = gr.Dropdown(
            choices=list(MODELS.keys()),
            value=names[0],
            label="Select Model",
        )
        single_btn    = gr.Button("Classify", variant="primary")
        single_label  = gr.HTML()
        single_reason = gr.Textbox(label="Clinical Reasoning", lines=10, interactive=False)

        single_btn.click(
            fn=predict_single,
            inputs=[model_selector, symptom_input],
            outputs=[single_label, single_reason],
        )

    gr.Markdown("### Example Cases — click to load")
    with gr.Row():
        for ex in EXAMPLES[:3]:
            gr.Button(ex[:55] + "…", size="sm").click(fn=lambda x=ex: x, outputs=symptom_input)
    with gr.Row():
        for ex in EXAMPLES[3:]:
            gr.Button(ex[:55] + "…", size="sm").click(fn=lambda x=ex: x, outputs=symptom_input)

    gr.Markdown("---\n*University of Passau — Applied AI Lab SS2026*")

demo.launch(server_name="0.0.0.0", server_port=7860, share=False, theme=gr.themes.Soft())
