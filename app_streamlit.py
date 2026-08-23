import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# Set page config
st.set_page_config(
    page_title="Medical Triage Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar information
st.sidebar.title("🏥 Triage Assistant")
st.sidebar.markdown("""
**Course:** Applied Artificial Intelligence Lab  
**University:** University of Passau  
**Semester:** Summer Semester 2026  
**Students:** Ajay Kesarwani & Nalan Thanasekaran  
""")

st.sidebar.subheader("Model Info")
st.sidebar.markdown("""
- **Base Model:** Qwen3-0.6B
- **Tuning Method:** QLoRA SFT v4
- **Parameters:** 0.6B (13.3x smaller than Teacher)
- **Training Data:** 42,872 real & synthetic cases
""")

# Title and description
st.title("🏥 Medical Triage Assistant")
st.markdown("""
Enter a patient's symptoms, vitals, and clinical description to receive an AI-assisted triage classification. 
This model has been compressed using knowledge distillation, structured pruning, and QLoRA fine-tuning.
""")

# Setup model loading function with caching
@st.cache_resource
def load_model():
    base_model_id = "Qwen/Qwen3-0.6B"
    adapter_path = "/root/model_miniaturization/data/approach2/qwen3_lora_v4/adapter"
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=bnb_config,
        device_map="auto"
    )
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()
    return tokenizer, model

try:
    with st.spinner("Loading model into memory (approx. 30 seconds)..."):
        tokenizer, model = load_model()
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.warning("Please ensure that model weights are present in the expected directories.")

# Constants
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
    "45-year-old male with sudden onset worst headache of his life, neck stiffness, photophobia, and confusion. No prior headache history.",
    "67-year-old female with right-sided facial droop, slurred speech, and inability to raise right arm — symptoms started 45 minutes ago.",
    "24-year-old with mild sore throat, low-grade fever 37.6°C, and runny nose for 2 days. No difficulty swallowing. Otherwise well."
]

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

# Main Layout
col1, col2 = st.columns([2, 1])

with col1:
    symptom_input = st.text_area(
        "Patient Symptom Description",
        value="",
        placeholder="Describe the patient's symptoms, vitals, and clinical presentation...",
        height=150
    )
    
    # Examples
    st.markdown("##### Quick Examples")
    cols = st.columns(len(EXAMPLES))
    for idx, ex in enumerate(EXAMPLES):
        if cols[idx].button(f"Example {idx+1}"):
            symptom_input = ex
            # Trigger page rerun to update text area
            st.session_state.symptom_input = ex
            st.rerun()

# Use session state to handle input changes from examples
if "symptom_input" in st.session_state:
    symptom_input = st.session_state.symptom_input

classify_clicked = st.button("Classify Triage Level", type="primary")

if classify_clicked and symptom_input.strip():
    prompt = build_prompt(symptom_input.strip())
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to("cuda")
    
    with torch.inference_mode():
        with st.spinner("Running clinical inference..."):
            out = model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1,
            )
    response = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    level = extract_level(response)
    
    with col2:
        st.subheader("Classification Outcome")
        if level == "EMERGENCY":
            st.error("🚨 EMERGENCY")
        elif level == "URGENT":
            st.warning("⚠️ URGENT")
        elif level == "ROUTINE":
            st.success("✅ ROUTINE")
        else:
            st.info("❓ UNKNOWN")
            
        st.metric(label="Inference VRAM Used", value=f"{torch.cuda.memory_allocated()/1e9:.2f} GB")
        
    st.subheader("Clinical Reasoning Chain")
    st.code(response, language="markdown")
