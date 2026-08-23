"""
Medical Triage Assistant — Enhanced Live Demo (container / GPU)
================================================================
Comparative-methods demo: ONE patient case, one working generative model, and
two documented "compression-cost" finding cards.

  • SFT v4        — LoRA fine-tune,  LIVE GENERATIVE  (the working model — hero)
  • Pruned + SFT  — STATIC FINDING   (layer-dropping degraded free-form generation)
  • KD v2         — STATIC FINDING   (class-weight overcorrection → input-invariant)

Why two cards are static (diagnosed 2026-07-11, see diagnose_demo_models.py):
  - Pruned+SFT free-form generation is broken on EVERY input (4-bit AND fp16):
    it never reaches the TRIAGE LEVEL format and rambles. Its 74.4% MIMIC acc is
    real but only surfaces via keyword-scan of the rambling text — not demo-safe.
  - KD v2 logit distribution is input-invariant: chest pain, stroke, headache and
    nausea all return ~30% EM / ~64% URGENT / ~6% ROUTINE (spread < 5%). It does
    not discriminate on clinical content. Shown as a contrast case, not queried live.

Both are presented HONESTLY as findings (what compression cost us), not hidden.

Run on the container (A6000):
  nohup /root/envs/miniaturization/bin/python -u src/demo/triage_chatbot_enhanced.py \
        > data/demo/enhanced.log 2>&1 < /dev/null & echo LAUNCHED
  grep -a gradio.live data/demo/enhanced.log      # -> public URL

University of Passau — Applied AI Lab SS2026 — Model Miniaturization
"""

import time
import torch
import gradio as gr
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# ── Paths (container) ────────────────────────────────────────────────────────────
STUDENT_BASE = "Qwen/Qwen3-0.6B"
ROOT         = Path("/root/model_miniaturization")
SFT_ADAPTER  = ROOT / "data/approach2/qwen3_lora_v4/adapter"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── Model registry ───────────────────────────────────────────────────────────────
# mode: "generative" (live) | "static" (documented finding card, no inference)
MODELS = {
    "SFT v4": {
        "mode": "generative",
        "base": STUDENT_BASE,
        "adapter": SFT_ADAPTER,
        "tag": "LoRA fine-tune · 42,872 samples · 80.2% MIMIC acc / 90.4% EM",
        "stats": {"Params": "0.6 B", "VRAM": "0.54 GB", "Latency": "0.9 s/case"},
        "accent": "#22d3ee",
    },
    "Pruned + SFT": {
        "mode": "static",
        "tag": "40% heads + 5 layers pruned · LoRA recovery",
        "stats": {"Params": "0.6 B*", "VRAM": "0.54 GB", "MIMIC acc": "74.4%"},
        "accent": "#a78bfa",
        "finding_label": "CLASSIFIER-ONLY",
        "finding_color": "#7c3aed",
        "finding": (
            "Evaluated performance: 74.4% accuracy / 80.0% EM recall on MIMIC (42 cases).\n\n"
            "FINDING — free-form generation degraded by compression.\n"
            "Dropping 5 transformer layers + zeroing 40% of attention heads left the "
            "model unable to reliably produce the structured TRIAGE LEVEL / REASONING "
            "format. It still classifies (the label is recoverable from its output), "
            "but it no longer writes coherent clinical reasoning.\n\n"
            "Shown as evaluated numbers, not live-queried — the classification signal "
            "survives compression; fluent generation does not. This is the cost of "
            "aggressive structural pruning at 0.6B scale."
        ),
    },
    "KD v2": {
        "mode": "static",
        "tag": "Knowledge distillation · logit-argmax · EMERGENCY class weight = 3.0",
        "stats": {"Params": "0.6 B", "VRAM": "0.54 GB", "MIMIC acc": "57.1%"},
        "accent": "#f472b6",
        "finding_label": "⚠ CONTRAST CASE",
        "finding_color": "#db2777",
        "finding": (
            "FINDING — class-weight overcorrection → does not discriminate.\n\n"
            "Measured logit distribution is nearly IDENTICAL for every input:\n"
            "  crushing chest pain →  29% EM · 67% URGENT ·  4% ROUTINE\n"
            "  acute stroke        →  27% EM · 68% URGENT ·  5% ROUTINE\n"
            "  slight headache     →  26% EM · 67% URGENT ·  7% ROUTINE\n"
            "  'I'm nauseous'      →  30% EM · 64% URGENT ·  6% ROUTINE\n\n"
            "Spread across all inputs < 5%. The model collapsed toward one region "
            "regardless of clinical content — an EMERGENCY class weight of 3.0 in the "
            "distillation CE loss overpowered genuine learning.\n\n"
            "Documented NEGATIVE RESULT: it demonstrates why balanced alpha / "
            "temperature / class-weight tuning is nontrivial in KD. A corrected run "
            "(weight ≈ 1.5) is future work. Not a working triage model."
        ),
    },
}

# ── Prompt / labels ──────────────────────────────────────────────────────────────
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

LABELS = ["EMERGENCY", "URGENT", "ROUTINE"]
COLORS = {"EMERGENCY": "#dc2626", "URGENT": "#f59e0b", "ROUTINE": "#16a34a", "UNKNOWN": "#64748b"}
GLOWS  = {"EMERGENCY": "#ef4444", "URGENT": "#fbbf24", "ROUTINE": "#22c55e", "UNKNOWN": "#94a3b8"}
ICONS  = {"EMERGENCY": "🚨", "URGENT": "⚠️", "ROUTINE": "✅", "UNKNOWN": "❓"}

EXAMPLES = [
    "58-year-old male with sudden crushing chest pain radiating to the left arm, profuse sweating, and shortness of breath for 20 minutes. BP 90/60, HR 110, O2 sat 91%.",
    "32-year-old female with severe right lower quadrant abdominal pain, fever 38.8°C, nausea, and rebound tenderness. Pain started 8 hours ago.",
    "45-year-old male with sudden onset worst headache of his life, neck stiffness, photophobia, and confusion.",
    "67-year-old female with right-sided facial droop, slurred speech, and inability to raise right arm — started 45 minutes ago.",
    "24-year-old with mild sore throat, low-grade fever 37.6°C, and runny nose for 2 days. No difficulty swallowing.",
]


# ── Model loading (only generative models are loaded) ────────────────────────────
def _bnb():
    return BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True,
    )


def load_generative(cfg: dict):
    tok = AutoTokenizer.from_pretrained(STUDENT_BASE, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        str(cfg["base"]),
        quantization_config=_bnb() if DEVICE == "cuda" else None,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        device_map="auto" if DEVICE == "cuda" else "cpu",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, str(cfg["adapter"])) if cfg.get("adapter") else base
    model.eval()
    return model, tok


print(f"Loading generative models on {DEVICE.upper()}...")
LOADED = {}
for name, cfg in MODELS.items():
    if cfg["mode"] != "generative":
        continue
    try:
        t0 = time.time()
        LOADED[name] = load_generative(cfg)
        print(f"  ✓ {name} ready in {time.time()-t0:.1f}s")
    except Exception as e:
        LOADED[name] = None
        print(f"  ✗ {name} FAILED: {e}")
if DEVICE == "cuda":
    print(f"VRAM allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
print("Launching UI...\n")


# ── Inference (generative only) ──────────────────────────────────────────────────
def build_prompt(description: str) -> str:
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\nPatient: {description.strip()}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def extract_level(text: str) -> str:
    up = text.upper()
    for lvl in LABELS:
        if lvl in up:
            return lvl
    return "UNKNOWN"


def run_generative(model, tok, description: str):
    inputs = tok(build_prompt(description), return_tensors="pt",
                 truncation=True, max_length=512).to(model.device)
    t0 = time.time()
    with torch.inference_mode():
        out = model.generate(
            **inputs, max_new_tokens=200, do_sample=False,
            pad_token_id=tok.eos_token_id, repetition_penalty=1.1,
        )
    elapsed = time.time() - t0
    text = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    return extract_level(text), text, elapsed


# ── HTML rendering ───────────────────────────────────────────────────────────────
def banner_html(level: str, elapsed=None):
    color, glow, icon = COLORS[level], GLOWS[level], ICONS[level]
    meta = f'<div class="banner-meta">{elapsed:.2f}s</div>' if elapsed is not None else ""
    return (
        f'<div class="triage-banner" style="background:{color};'
        f'box-shadow:0 0 22px {glow}66, inset 0 1px 0 #ffffff33;">'
        f'<span class="banner-icon">{icon}</span>'
        f'<span class="banner-text">{level}</span>{meta}</div>'
    )


def static_banner(cfg):
    return (
        f'<div class="triage-banner finding-banner" style="background:{cfg["finding_color"]};">'
        f'<span class="banner-text">{cfg["finding_label"]}</span></div>'
    )


def placeholder_banner():
    return ('<div class="triage-banner banner-idle">'
            '<span class="banner-icon">🩺</span>'
            '<span class="banner-text">Awaiting case</span></div>')


def one_generative(name: str, description: str):
    entry = LOADED.get(name)
    if entry is None:
        return (f'<div class="triage-banner" style="background:{COLORS["UNKNOWN"]};">'
                '<span class="banner-text">Model unavailable</span></div>',
                "This model failed to load on the server.")
    level, text, elapsed = run_generative(*entry, description)
    return banner_html(level, elapsed), text


def compare_all(description: str):
    outs = []
    for name, cfg in MODELS.items():
        if cfg["mode"] == "generative":
            if not description or not description.strip():
                outs.extend([placeholder_banner(), ""])
            else:
                outs.extend(one_generative(name, description))
        else:  # static — always the documented finding, regardless of input
            outs.extend([static_banner(cfg), cfg["finding"]])
    return outs


# ── Custom CSS ───────────────────────────────────────────────────────────────────
CUSTOM_CSS = """
:root { --bg0:#0b1220; --bg1:#111a2e; --card:#0f1a2ecc; --line:#1e2d4a;
        --txt:#e6edf7; --muted:#93a4c3; --teal:#22d3ee; }
.gradio-container { max-width: 1400px !important; margin: 0 auto !important;
        background: radial-gradient(1200px 600px at 20% -10%, #16244a 0%, transparent 60%),
                    radial-gradient(1000px 500px at 100% 0%, #16324a 0%, transparent 55%),
                    linear-gradient(160deg, var(--bg0), var(--bg1)) !important;
        color: var(--txt) !important; font-family: 'Inter','Segoe UI',system-ui,sans-serif !important; }
#app-header { text-align:center; padding: 26px 20px 8px; }
#app-header .title { font-size: 34px; font-weight: 800; letter-spacing:-0.5px;
        background: linear-gradient(90deg, #22d3ee, #a78bfa 60%, #f472b6);
        -webkit-background-clip:text; -webkit-text-fill-color:transparent; margin:0; }
#app-header .sub  { color: var(--muted); font-size: 15px; margin-top: 6px; }
#app-header .uni  { color: var(--teal); font-size: 13px; letter-spacing:2px;
        text-transform:uppercase; margin-top:10px; font-weight:600; }
.input-card, .model-card { background: var(--card) !important; border:1px solid var(--line) !important;
        border-radius: 16px !important; padding: 16px !important;
        backdrop-filter: blur(8px); box-shadow: 0 8px 30px #0006; }
.model-head { font-size: 18px; font-weight: 700; margin: 2px 0 2px; }
.model-tag  { color: var(--muted); font-size: 12px; line-height:1.5; margin-bottom: 10px;
        border-left: 3px solid var(--teal); padding-left: 10px; }
.triage-banner { display:flex; align-items:center; gap:12px; justify-content:center;
        color:#fff; padding: 16px 18px; border-radius: 14px; font-weight: 800;
        letter-spacing: 1px; margin: 6px 0 12px; position:relative; }
.triage-banner .banner-icon { font-size: 24px; }
.triage-banner .banner-text { font-size: 22px; }
.triage-banner .banner-meta { position:absolute; right:14px; top:8px; font-size:11px;
        font-weight:600; opacity:.85; letter-spacing:0; }
.finding-banner .banner-text { font-size: 16px; letter-spacing:.5px; }
.banner-idle { background:#16233d !important; border:1px dashed var(--line); color:var(--muted) !important; }
.banner-idle .banner-text { font-size:16px; font-weight:600; }
textarea, .model-card textarea { background:#0a1424 !important; color:var(--txt) !important;
        border-radius: 10px !important; font-size: 14px !important; line-height:1.6 !important; }
#go-btn { background: linear-gradient(90deg,#0891b2,#7c3aed) !important; border:none !important;
        font-weight:700 !important; font-size:16px !important; letter-spacing:.5px; }
.stat-pill { display:inline-block; background:#0a1424; border:1px solid var(--line);
        color:var(--muted); border-radius:999px; padding:4px 12px; margin:3px 4px 0 0; font-size:12px; }
.stat-pill b { color:var(--txt); }
.ex-row .gr-button { background:#132340 !important; border:1px solid var(--line) !important;
        color:var(--txt) !important; font-size:12px !important; text-align:left !important; }
#footer { text-align:center; color:var(--muted); font-size:12px; padding: 14px; }
"""


def stats_pills(cfg):
    return " ".join(f'<span class="stat-pill">{k} <b>{v}</b></span>' for k, v in cfg["stats"].items())


# ── UI ───────────────────────────────────────────────────────────────────────────
with gr.Blocks(title="Medical Triage Assistant") as demo:
    gr.HTML("""
    <div id="app-header">
      <p class="title">🏥 Medical Triage Assistant</p>
      <p class="sub">One patient case — the working model, and what compression cost us.</p>
      <p class="uni">University of Passau · Applied AI Lab SS2026 · Model Miniaturization</p>
    </div>
    """)

    with gr.Group(elem_classes="input-card"):
        symptom_input = gr.Textbox(
            label="Patient presentation",
            placeholder="Age, sex, chief complaint, symptoms, vitals, onset...",
            lines=3,
        )
        go_btn = gr.Button("⚕️  Classify (SFT v4) · compare vs compression findings",
                           variant="primary", size="lg", elem_id="go-btn")

    banners, reasons = [], []
    with gr.Row(equal_height=True):
        for name, cfg in MODELS.items():
            with gr.Column():
                with gr.Group(elem_classes="model-card"):
                    live = " · LIVE" if cfg["mode"] == "generative" else " · FINDING"
                    gr.HTML(
                        f'<div class="model-head" style="color:{cfg["accent"]}">{name}{live}</div>'
                        f'<div class="model-tag">{cfg["tag"]}</div>'
                        f'<div>{stats_pills(cfg)}</div>'
                    )
                    init_banner = static_banner(cfg) if cfg["mode"] == "static" else placeholder_banner()
                    init_text   = cfg["finding"] if cfg["mode"] == "static" else ""
                    b = gr.HTML(init_banner)
                    r = gr.Textbox(label="Clinical reasoning / finding", value=init_text,
                                   lines=12, interactive=False)
                    banners.append(b)
                    reasons.append(r)

    outputs = []
    for b, r in zip(banners, reasons):
        outputs.extend([b, r])

    gr.Markdown("### Example cases — click to run the live model")
    with gr.Row(elem_classes="ex-row"):
        for ex in EXAMPLES[:3]:
            gr.Button(ex[:52] + "…", size="sm").click(
                fn=lambda x=ex: x, outputs=symptom_input
            ).then(fn=compare_all, inputs=symptom_input, outputs=outputs)
    with gr.Row(elem_classes="ex-row"):
        for ex in EXAMPLES[3:]:
            gr.Button(ex[:52] + "…", size="sm").click(
                fn=lambda x=ex: x, outputs=symptom_input
            ).then(fn=compare_all, inputs=symptom_input, outputs=outputs)

    go_btn.click(fn=compare_all, inputs=symptom_input, outputs=outputs)
    symptom_input.submit(fn=compare_all, inputs=symptom_input, outputs=outputs)

    gr.HTML(
        '<div id="footer">SFT v4 is queried live (generative). Pruned+SFT and KD v2 are '
        'shown as documented findings — the diagnosed cost of structural pruning '
        '(degraded generation) and class-weight overcorrection (input-invariant collapse). '
        'KD v1 excluded (vocabulary collapse).</div>'
    )

demo.launch(server_name="0.0.0.0", server_port=7860, share=True,
            css=CUSTOM_CSS, theme=gr.themes.Soft())
