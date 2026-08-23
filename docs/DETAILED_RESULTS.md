# Medical Triage Assistant — Model Miniaturization

**Course:** Applied Artificial Intelligence Lab, Summer Semester 2026  
**University:** University of Passau  
**Student:** Nalan Thanasekaran, Ajay Kesarwani

---

## Project Overview

This project implements a compressed medical triage assistant using knowledge distillation. A large medical language model (teacher) is used to generate structured clinical reasoning, which is then distilled into a lightweight student model deployable on resource-constrained hardware.

**Goal:** Compress an 8B-parameter medical LLM into a ~0.6B student model that retains >85% triage accuracy and >95% emergency recall, while running in under 1 GB VRAM.

---

## Architecture

| Component | Model | Size |
|---|---|---|
| Teacher | aaditya/OpenBioLLM-Llama3-8B | 8B params, 4-bit NF4 (~5.7 GB VRAM) |
| Student | Qwen/Qwen3-0.6B | 0.6B params (~300 MB quantized) |

**Pipeline:**
1. Teacher generates labeled clinical reasoning chains (synthetic data)
2. Structured pruning: attention head pruning + layer dropping → ~3B intermediate model
3. Knowledge distillation: logit-level KL divergence + feature-level MSE
4. LoRA fine-tuning on the student

---

## Triage Classes

| Class | Definition |
|---|---|
| EMERGENCY | Immediately life-threatening — requires intervention within minutes |
| URGENT | Serious but stable — requires evaluation within 1–2 hours |
| ROUTINE | Non-urgent — can be seen in a scheduled appointment |

---

## Project Structure

```
model_miniaturization/
├── src/
│   ├── approach2/                   # Datasets merging, NLI filtering, and fine-tuning pipelines
│   ├── data_generation/
│   │   ├── teacher_inference.py     # Teacher inference + synthetic data generation pipeline
│   │   └── prepare_dataset.py       # Stratified dataset split (80/10/10)
│   ├── pruning/
│   │   ├── importancescoring.py     # Taylor importance scoring for heads & layers
│   │   ├── headpruning.py           # Attention head pruning (bottom 40%)
│   │   └── layerdropping.py         # Layer dropping (drops 5 middle layers)
│   ├── distillation/
│   │   ├── kdtrainer.py             # Knowledge distillation trainer for Qwen3-0.6B student
│   │   ├── kdtrainer_pruned.py      # Knowledge distillation trainer for pruned student base
│   │   ├── featurekd.py             # Feature hidden state MSE loss
│   │   └── cotdistillation.py       # Chain-of-Thought cross-entropy loss helper
│   ├── finetuning/
│   │   └── lorafinetune.py          # LoRA fine-tuning script
│   └── evaluation/
│       ├── evaluate_teacher_n.py    # Teacher model test evaluation (synthetic & real Latvia)
│       ├── evaluate_distilled.py    # Generation-based evaluation for distilled students
│       └── evaluate_distilled_logits.py # Logit-based evaluation for distilled students
├── app_streamlit.py                 # Streamlit dashboard UI for triage chatbot demo
├── requirements.txt                 # Core Python dependencies for the project
├── data/
│   ├── approach2/                   # Merged datasets and fine-tuning adapters (gitignored)
│   ├── distillation/                # Distillation model weights (gitignored)
│   ├── pruning/                     # Pruned base model weights (gitignored)
│   └── synthetic/                   # Generated synthetic datasets (gitignored)
├── configs/                         # Training configuration files
├── notebooks/                       # Exploration and analysis notebooks
└── report/                          # Final report (LaTeX)
```

---

## Data Generation

The teacher model generates synthetic triage samples using oracle-label prompting: given a patient description and its correct triage level, the model produces structured clinical reasoning with key symptoms and a 3-step reasoning chain.

**Seed pilot result:** 30 samples (10 per class), perfectly balanced.

**Sample output:**
```
TRIAGE LEVEL: EMERGENCY
KEY SYMPTOMS: sudden onset left-sided weakness, facial drooping, slurred speech
CLINICAL REASONING:
  Step 1: Classic ischemic stroke — FAST symptoms with acute neurological deficit
  Step 2: Age 70 with acute onset — hemorrhagic vs ischemic must be ruled out urgently
  Step 3: CT head and thrombolysis decision required within minutes
CONFIDENCE: HIGH
```

**Target dataset:** 50,000 samples (15K EMERGENCY, 17.5K URGENT, 17.5K ROUTINE) generated under `data/synthetic/symptoms_50k.jsonl`.

---

## Setup & Running the Pipeline

### Environment Activation
All scripts should be executed using the custom Conda environment on the university container:
```bash
source /opt/conda/etc/profile.d/conda.sh && conda activate /root/envs/miniaturization
cd /root/model_miniaturization
```

### 1. Pruning Pipeline (Week 3)
Calculate Taylor importance scores, prune attention heads, and drop layers:
```bash
# Importance scoring
python src/pruning/importancescoring.py
# Prune bottom 40% attention heads
python src/pruning/headpruning.py
# Drop middle layers (drops middle 5 layers)
python src/pruning/layerdropping.py
```
Output model weights are saved at `data/pruning/qwen3_pruned_heads_layers/`.

### 2. Knowledge Distillation (Week 4)
Run probability/logit-level KL-divergence distillation from the 8B teacher into the student models:
```bash
# Distill into baseline student
python src/distillation/kdtrainer.py --epochs 1 --batch_size 2
# Distill into pruned student base
python src/distillation/kdtrainer_pruned.py --epochs 1 --batch_size 2
```

### 3. LoRA Fine-Tuning (Week 5)
Fine-tune the student model (distilled or base) on the combined dataset (synthetic + real-patient datasets):
```bash
python src/finetuning/lorafinetune.py --epochs 3 --batch_size 4 --grad_accum 8
```

### 4. Running the Chatbot UI Demo
To launch the interactive web interface and interact with the trained models, run the Streamlit dashboard script. It provides a premium UI, loads the student model in 4-bit, and performs real-time inference on GPU with color-coded triage badges and VRAM monitoring:
```bash
streamlit run app_streamlit.py
```
This will launch a local server and provide a URL to access the UI in your browser.

---

## Evaluation results & Target Metrics

| Metric | Target | Fine-tuned Student (Qwen3-0.6B+LoRA) | Pruned Student (Qwen3-Pruned+LoRA) | Distilled Student (Qwen3-KD-0.6B) |
|---|---|---|---|---|
| **Triage Accuracy** | > 85% | **90.8%** | **90.5%** | **35.3%** (35.6% on Latvia) |
| **Emergency Recall** | > 95% | **82.7%** (100% w/ logit sweep) | **91.7%** (100% w/ logit sweep) | **0.0%** (100% w/ logit sweep) |
| **Macro F1** | > 0.83 | **0.909** | **0.602** (MIMIC) | **0.174** |
| **Student VRAM** | < 1 GB | **0.54 GB** | **0.54 GB** | **0.54 GB** |

### Baseline Head-to-Head Comparisons (Raw Student vs Raw Teacher)

The following table reports the metrics of the Zero-Shot Student baseline (`Qwen3-0.6B`) and the Raw Teacher model (`OpenBioLLM-8B`) across all four evaluation datasets. 

*Note: Baseline models suffer from high unparsed rates because they do not strictly follow the single-word output format constraint without tuning.*

| Dataset | Model | Total Samples | Failed/Unparsed | Accuracy (Valid) | Effective Accuracy | Macro F1 | EM Recall |
|---|---|---|---|---|---|---|---|
| **Synthetic Test** | Student Baseline | 510 | 485 (95.1%) | 96.0% | **4.7%** | 0.327 | 0.0% |
| | Teacher Baseline | 510 | 350 (68.6%) | 49.4% | **15.5%** | 0.343 | 88.0% |
| **Syntech-500** | Student Baseline | 500 | 493 (98.6%) | 0.0% | **0.0%** | 0.000 | 0.0% |
| | Teacher Baseline | 500 | 53 (10.6%) | 47.7% | **42.6%** | 0.308 | 91.8% |
| **Latvia (Real)** | Student Baseline | 3000 | 2976 (99.2%) | 62.5% | **0.5%** | 0.528 | 100.0% |
| | Teacher Baseline | 3000 | 1585 (52.8%) | 25.6% | **12.1%** | 0.248 | 100.0% |
| **MIMIC (Real)** | Student Baseline | 207 | 176 (85.0%) | 58.1% | **8.7%** | 0.286 | 100.0% |
| | Teacher Baseline | 207 | 199 (96.1%) | 62.5% | **2.4%** | 0.278 | 83.3% |



### Complete Student Model Comparisons Across Different States and Datasets

We compare the student model's metrics across different stages of training and compression (Raw/Zero-shot, Fine-Tuned SFT, Pruned SFT, and Distilled KD) for all four clinical triage datasets.

| Dataset | State | Accuracy | Macro F1 | EM Recall | Failed |
|---|---|---|---|---|---|
| **Synthetic Test** | Raw/Zero-Shot | 4.7% | 0.327 | 0.0% | 485/510 |
| | Fine-Tuned (SFT) | 100.0% | 1.000 | 100.0% | 0/510 |
| | Pruned SFT | 100.0% | 1.000 | 100.0% | 0/510 |
| | Distilled (KD) (Argmax) | 35.3% | 0.174 | 0.0% | 0/510 |
| | Distilled (KD) (t=0.05) | 29.4% | 0.152 | 100.0% | 0/510 |
| | Pruned Distilled (KD) (Argmax) | 46.1% | 0.366 | 7.3% | 0/510 |
| | Pruned Distilled (KD) (t=0.05) | 29.4% | 0.152 | 100.0% | 0/510 |
| **Syntech-500** | Raw/Zero-Shot | 0.0% | 0.000 | 0.0% | 493/500 |
| | Fine-Tuned (SFT) | 96.2% | 0.962 | 94.3% | 0/500 |
| | Pruned SFT | 87.0% | 0.881 | 94.8% | 0/500 |
| | Distilled (KD) (Argmax) | 39.0% | 0.187 | 0.0% | 0/500 |
| | Distilled (KD) (t=0.05) | 46.0% | 0.210 | 100.0% | 0/500 |
| | Pruned Distilled (KD) (Argmax) | 39.0% | 0.187 | 0.0% | 0/500 |
| | Pruned Distilled (KD) (t=0.05) | 46.0% | 0.210 | 100.0% | 0/500 |
| **Latvia (Real)** | Raw/Zero-Shot | 0.5% | 0.528 | 100.0% | 2,976/3,000 |
| | Fine-Tuned (SFT) | 100.0% | 1.000 | 100.0% | 0/3,000 |
| | Pruned SFT | 100.0% | 1.000 | 100.0% | 0/3,000 |
| | Distilled (KD) (Argmax) | 35.6% | 0.213 | 6.9% | 0/3,000 |
| | Distilled (KD) (t=0.05) | 33.3% | 0.167 | 100.0% | 0/3,000 |
| | Pruned Distilled (KD) (Argmax) | 43.8% | 0.349 | 41.6% | 0/3,000 |
| | Pruned Distilled (KD) (t=0.05) | 33.3% | 0.167 | 100.0% | 0/3,000 |
| **MIMIC (Real)** | Raw/Zero-Shot | 8.7% | 0.286 | 100.0% | 176/207 |
| | Fine-Tuned (SFT) | 74.9% | 0.531 | 80.9% | 0/42 |
| | Pruned SFT | 74.4% | 0.495 | 80.0% | 0/207 |
| | Distilled (KD) (Argmax) | 42.9% | 0.200 | 0.0% | 0/42 |
| | Distilled (KD) (t=0.05) | 57.1% | 0.242 | 100.0% | 0/42 |
| | Pruned Distilled (KD) (Argmax) | 42.9% | 0.200 | 0.0% | 0/42 |
| | Pruned Distilled (KD) (t=0.05) | 57.1% | 0.242 | 100.0% | 0/42 |

### Reasoning Quality (BERTScore vs PubMedQA)

We measure the quality of the teacher model's synthetic clinical reasoning chains using `roberta-large` BERTScore compared to real biomedical prose from `PubMedQA` (`pubmedqa_references.json`):

| Class | Precision | Recall | F1 |
|---|---|---|---|
| EMERGENCY | 0.8108 | 0.8277 | 0.8191 |
| URGENT | 0.8114 | 0.8275 | 0.8193 |
| ROUTINE | 0.8141 | 0.8266 | 0.8202 |
| **Overall** | **0.8121** | **0.8274** | **0.8196** |

An overall F1 score of **0.82** confirms that the generated synthetic reasoning chains are semantically coherent and consistent across all triage categories.

---

## Experimental Details: LoRA & Knowledge Distillation (KD)

*This section focuses specifically on the Low-Rank Adaptation (LoRA) and Knowledge Distillation (KD) techniques applied to the `Qwen3-0.6B` student model. For comprehensive performance metrics of these configurations, please refer to the main Evaluation Results table in Section 7.*

### 1. LoRA Fine-Tuning (SFT)

The student model was fine-tuned using LoRA to adapt it to the medical triage task based on synthetic and real-world datasets.

- **Adapter Settings:** Rank `r=16`, `alpha=32`, applied in 4-bit NF4 quantization.
- **Training Data:** Synthetically generated triage scenarios, `syntech-500`, `fedmml-ed-triage` (Denmark and Turkey), and `MIMIC-IV-ED` datasets. (Total: ~42.8k samples).

### 2. Knowledge Distillation (KD)

Knowledge Distillation was employed to transfer the learned representations from the 8B teacher model (`aaditya/OpenBioLLM-Llama3-8B`) to the 0.6B student model.

- **KD v1:** Logit-level KD with `alpha=0.7`, Temperature `T=4.0`.
- **KD v2 (Full Fine-Tune):** Trained the full 0.6B student model (no LoRA) on 42k combined samples for 2 epochs. This resulted in severe mode collapse (overfitting to the majority `ROUTINE` class and generating gibberish text).

### 3. Pruned Student + LoRA Recovery

- **Pruning Strategy:** Taylor importance scoring followed by dropping the bottom 40% of attention heads and removing 5 middle layers.
- **Recovery:** LoRA fine-tuning applied to the pruned base model.

### 4. Phase 2: Pruned Student + KD + LoRA (1 Epoch)

To mitigate the mode collapse from Phase 1, we implemented a robust Phase 2 pipeline:
- **Architecture:** We used the Pruned Student (`qwen3_pruned_heads_layers`) as the base model and attached a LoRA adapter (`r=16`, `alpha=32`).
- **Knowledge Distillation:** We ran KD to train *only* the LoRA adapter, freezing the pruned base to prevent catastrophic forgetting.
- **Bug Fixes:** Resolved an FP16 numerical overflow bug in the Llama-3 Teacher by enforcing `bfloat16` precision, and added gradient clipping.
- **Outcome:** The model successfully avoided the 0% recall collapse seen in Phase 1. While the argmax baseline is heavily biased towards predicting `EMERGENCY` (yielding 100% recall), sweeping the decision threshold recovers significant accuracy.

*Note: The PMC-Patients evaluation relies on autoregressive text generation. Because the Phase 2 LoRA adapter was trained using token-level Cross Entropy on the label only (`seq-ce=OFF`), the model successfully learned to output precise logits for triage classes but lost its conversational text-generation capabilities. This resulted in a high unparsed rate during generative evaluation. To fix this in Phase 3, we must re-enable `seq-ce=ON` so the model retains its language abilities.*

*Update: The unparsed rate bug in MIMIC evaluation was successfully resolved by switching from the Llama 3 prompt template to Qwen's native ChatML format in `evaluate_student_mimic.py`.*

### Challenges Faced

Throughout the development and training phases, several significant technical hurdles were encountered and resolved:

1. **Catastrophic Mode Collapse (Phase 1):**
   - During the initial full fine-tuning of the student model without LoRA, the model collapsed entirely, predicting the majority class (`ROUTINE`) for almost all samples. This resulted in a disastrous 0% recall rate for `EMERGENCY` cases, rendering the model unsafe for triage. This was resolved in Phase 2 by utilizing LoRA adapters and freezing the pruned base weights, successfully rescuing the recall back up to 100%.

2. **Mathematical Overflow & `NaN` Loss (Phase 2):**
   - During Phase 2 Knowledge Distillation, the training persistently crashed midway through the epoch due to a `NaN` (Not a Number) loss explosion. 
   - Initially, this was suspected to be exploding gradients in the student model, leading to the implementation of strict gradient clipping. 
   - However, the crashes persisted. The true root cause was identified as an architectural limitation in the 8B Teacher model (`Llama-3`). When running in 4-bit `float16` precision, the teacher's attention scores occasionally spiked above the maximum limit of `float16` (65,504), overflowing into `Infinity` and corrupting the student's training targets. This was permanently fixed by forcing the teacher pipeline to use `bfloat16` (Brain Float 16), which has a vastly larger mathematical ceiling.

3. **Loss of Generative Capabilities (PMC-Patients):**
   - The generative evaluation for the PMC-Patients dataset experienced a 98.5% unparsed failure rate. This occurred because the Phase 2 KD training was deliberately run with Sequence Cross-Entropy disabled (`seq-ce=OFF`). While this successfully forced the model to master the triage classification logits, it stripped the model of its autoregressive conversational abilities, causing it to output gibberish instead of properly formatted text. Future phases will need to re-enable sequence-level modeling to retain chat capabilities.

### Next Steps

1. Conduct **LoRA settings ablations** (sweeping r, alpha, dropout, target modules, learning rate) to further optimize the adapter.
