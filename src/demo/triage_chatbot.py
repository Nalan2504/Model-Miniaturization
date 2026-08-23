"""
Medical Triage Chatbot — Single Model (SFT LoRA v4)
Run: /root/envs/miniaturization/bin/python src/demo/triage_chatbot.py
Access: ssh -L 7860:localhost:7860 ailab  → open http://localhost:7860
        OR wait for the public share URL printed in the terminal
"""

import torch
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# ── Config ────────────────────────────────────────────────────────────────────
STUDENT_BASE = "Qwen/Qwen3-0.6B"
ADAPTER_PATH = "/root/model_miniaturization/data/approach2/qwen3_lora_v4/adapter"

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
    ["58-year-old male with sudden crushing chest pain radiating to the left arm, profuse sweating, and shortness of breath for 20 minutes. BP 90/60, HR 110, O2 sat 91%."],
    ["32-year-old female with severe right lower quadrant abdominal pain, fever 38.8°C, nausea, and rebound tenderness. Pain started 8 hours ago."],
    ["45-year-old male with sudden onset worst headache of his life, neck stiffness, photophobia, and confusion. No prior headache history."],
    ["67-year-old female with right-sided facial droop, slurred speech, and inability to raise right arm — symptoms started 45 minutes ago."],
    ["24-year-old with mild sore throat, low-grade fever 37.6°C, and runny nose for 2 days. No difficulty swallowing. Otherwise well."],
]

# ── Model loading ─────────────────────────────────────────────────────────────
print("Loading model — this takes ~30 seconds...")
bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)
tokenizer = AutoTokenizer.from_pretrained(STUDENT_BASE, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

base = AutoModelForCausalLM.from_pretrained(STUDENT_BASE, quantization_config=bnb, device_map="auto")
model = PeftModel.from_pretrained(base, ADAPTER_PATH)
model.eval()
print(f"Model ready — VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB")


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


def predict(description: str):
    if not description.strip():
        return "", "", ""

    prompt = build_prompt(description.strip())
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to("cuda")

    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1,
        )
    response = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    level = extract_level(response)

    color_map = {"EMERGENCY": "#dc2626", "URGENT": "#f97316", "ROUTINE": "#16a34a", "UNKNOWN": "#6b7280"}
    label_html = (
        f'<div style="background:{color_map[level]};color:white;padding:16px 24px;'
        f'border-radius:8px;font-size:24px;font-weight:bold;text-align:center;letter-spacing:2px;">'
        f'🚨 {level}</div>'
        if level == "EMERGENCY" else
        f'<div style="background:{color_map[level]};color:white;padding:16px 24px;'
        f'border-radius:8px;font-size:24px;font-weight:bold;text-align:center;letter-spacing:2px;">'
        f'⚠️ {level}</div>'
        if level == "URGENT" else
        f'<div style="background:{color_map[level]};color:white;padding:16px 24px;'
        f'border-radius:8px;font-size:24px;font-weight:bold;text-align:center;letter-spacing:2px;">'
        f'✅ {level}</div>'
    )

    return label_html, response, f"Model: SFT LoRA v4 | VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB"


# ── UI ────────────────────────────────────────────────────────────────────────
with gr.Blocks(title="Medical Triage Assistant") as demo:
    gr.Markdown("""
    # 🏥 Medical Triage Assistant
    **Model:** Qwen3-0.6B + SFT LoRA v4 | **Training:** 42,872 real + synthetic ED cases
    Enter a patient symptom description to receive an AI-assisted triage classification.
    """)

    with gr.Row():
        with gr.Column(scale=2):
            symptom_input = gr.Textbox(
                label="Patient Symptom Description",
                placeholder="Describe the patient's symptoms, vitals, and clinical presentation...",
                lines=4,
            )
            submit_btn = gr.Button("Classify Triage Level", variant="primary", size="lg")

        with gr.Column(scale=1):
            triage_label = gr.HTML(label="Triage Level")
            model_info   = gr.Textbox(label="Model Info", interactive=False, lines=1)

    reasoning_output = gr.Textbox(
        label="Clinical Reasoning",
        lines=10,
        interactive=False,
    )

    gr.Markdown("### Example Cases")
    with gr.Row():
        for ex in EXAMPLES[:3]:
            gr.Button(ex[0][:60] + "…").click(
                fn=lambda x=ex[0]: x,
                outputs=symptom_input,
            )
    with gr.Row():
        for ex in EXAMPLES[3:]:
            gr.Button(ex[0][:60] + "…").click(
                fn=lambda x=ex[0]: x,
                outputs=symptom_input,
            )

    submit_btn.click(
        fn=predict,
        inputs=symptom_input,
        outputs=[triage_label, reasoning_output, model_info],
    )
    symptom_input.submit(
        fn=predict,
        inputs=symptom_input,
        outputs=[triage_label, reasoning_output, model_info],
    )

    gr.Markdown("""
    ---
    *University of Passau — Applied AI Lab SS2026 | Model Miniaturization Project*
    """)

demo.launch(server_name="0.0.0.0", server_port=7860, share=True, theme=gr.themes.Soft())
