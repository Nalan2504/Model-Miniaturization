# Medical Triage Assistant — Model Miniaturization

Compressing an **8B medical LLM into a 0.6B student** that triages emergency-department
patients into **EMERGENCY / URGENT / ROUTINE**, runs offline in **under 1 GB VRAM**, and is
measured on the metric that actually matters clinically: **emergency recall**.

Framed as a **comparative study of four compression methods** — LoRA fine-tuning, structured
pruning, knowledge distillation, and 4-bit quantization — all evaluated on one shared harness.

**Applied AI Lab, University of Passau — Summer Semester 2026**
Nalan Thanasekaran · Ajay Kesarwani

---

## Results on MIMIC-IV-ED (the hard benchmark)

| Method | Accuracy | Emergency Recall | Verdict |
|---|---|---|---|
| **SFT (LoRA v4)** | **80.2%** | **90.4%** | Best / deployable |
| SFT @ threshold t=0.05 | 73.8% | **95.8%** | Meets the >95% safety bar |
| Pruned + SFT | 74.4% | 80.0% | Classifier only — generation degraded |
| KD v1 (argmax) | 42.9% | 0.0% | Collapsed to URGENT — negative result |
| KD v2 (argmax) | 57.1% | 100% | Degenerate, always EMERGENCY — negative result |
| **TF-IDF + LogReg** | **88.1%** | **95.8%** | A bag-of-words baseline beats every neural model |

The student is **13× smaller than the teacher and beats it** — the 8B teacher's output
formatting collapses on real clinical notes (96% unparseable), while the tuned 0.6B student
parses cleanly at 10× less VRAM.

---

## What this project is actually about

The interesting content here is not the headline number — it's the chain of failures that
produced it. Each one changed the design.

### 1. The teacher couldn't classify
**Problem:** The obvious pipeline — let the 8B medical teacher label patients — failed
immediately. It was massively EMERGENCY-biased, calling 9 of 10 URGENT cases EMERGENCY
(47.7% accuracy, 11% URGENT recall, 5% ROUTINE recall). It sees only "dying" or "fine".

**Fix:** **Oracle-labelling.** We authored 51 medical conditions with known-correct labels and
used the teacher *only* to generate clinical reasoning for a label already known to be right.
The teacher is a good explainer, not a good classifier — so it's used only for what it's good at.

### 2. Four data-generation bugs, four fixes
| Bug | Root cause | Fix |
|---|---|---|
| Parser rejected all 30 pilot samples | Searched for exact `TRIAGE LEVEL: EMERGENCY`; model wrote prose | Regex for the label word anywhere in the text |
| Non-reproducible outputs at low temperature | Sampling noise broke the consistency filter | Greedy decoding — deterministic |
| EMERGENCY bias | Teacher's clinical prior | Oracle-labelling (above) |
| Model emitted symptoms then stopped (~30 chars) | Learned from medical notes that "Key Symptoms" is the *last* field | **Response priming** — pre-fill the answer's start to force continuation |

Result: 5,100 clean, class-balanced samples.

### 3. A negative result we kept
**Problem:** To validate data quality we trained BioBERT on it and tested cross-dataset. It
failed completely — 15–20% accuracy, everything predicted ROUTINE.

**Root causes:** (a) **label leakage** — our reasoning text literally begins with the label, so
BERT learned to read the label token rather than the reasoning; (b) **format mismatch** —
verbose narratives at train time vs. short keyword lists at test time.

**Fix:** Cross-dataset BERT is the wrong validation tool. Switched to **BERTScore** against real
PubMedQA medical text, scoring **0.82** against a 0.5 random floor — confirming the generated
reasoning is genuinely coherent clinical language. Both the negative and positive result are
kept in the report, because the negative one changed how we validate.

### 4. Data-centric iteration, v1 → v4
LoRA trains just **1.67% of parameters** (~10M). v1 hit 90.8% accuracy but only **82.7%
emergency recall** — below the safety bar, meaning 17% of real emergencies were downgraded to
URGENT. Logit-thresholding was tried as a fix and **failed** — verbose output meant no
single-token threshold could separate the classes. The fix that worked was data, not tricks:
add real ESI-labelled MIMIC patients (v3 → 89.6% emergency recall), then the fedmml dataset of
87,000 real ED patients across three countries (v4), **holding out an entire country (Latvia)**
as an unseen domain-shift test.

### 5. The LogReg reckoning — the project's most valuable finding
**Problem:** v4 scored a triumphant 100% on Latvia. Then a TF-IDF + Logistic Regression sanity
baseline — no neural network, just word counting — **also scored a perfect F1 of 1.0**.

**What it meant:** Latvia is *trivially easy*; its complaints are standardized keywords. The
100% proved nothing about intelligence, it proved the test was too weak. Worse, on MIMIC's
messy free-text notes, LogReg hit **95.8% emergency recall — higher than every neural model.**

**Consequence:** MIMIC became the benchmark we report everywhere, and the honest conclusion is
that *for keyword-driven triage on standardized complaints, model complexity does not guarantee
better safety-critical recall.* Always ship the dumb baseline.

### 6. Distillation inherits the teacher's flaws
**Problem:** The first distilled student scored 35% accuracy and **0% emergency recall** —
predicting URGENT for nearly everything.

**Root cause:** α=0.7 let the teacher's signal dominate (and the teacher is EMERGENCY-averse),
T=4.0 over-blurred class boundaries, and no class weighting meant missing an emergency carried
no penalty. **The student faithfully inherited the teacher's weakness** — distillation transfers
a teacher's flaws, not just its strengths.

**Attempted fix and its own failure:** v2 (α=0.5, T=2.0, emergency weight 3.0) fixed the 0%
recall and **overcorrected** into predicting EMERGENCY for everything — 100% recall, 57%
accuracy. A weight near 1.5 should balance it; that's documented as the next step, not claimed
as a result.

---

## Honest limitations

- **Pruning is soft, not structural.** Weights are zeroed but the architecture config still says
  28 layers / 16 heads. It is 17% smaller on disk but not truly architecturally compressed.
- **Distillation is fragile** — it inherits teacher bias, and the correction overshot.
- **The Latvia headline is on a trivially easy test**, as LogReg proved.
- **A bag-of-words baseline still beats the neural models** on MIMIC emergency recall before
  threshold calibration.
- **Format-robustness, not raw model size, is the real bottleneck** — every method broke
  identically on unstructured PMC case notes, at ~85% parse failure.

---

## Pipeline

| Component | Model | Size |
|---|---|---|
| Teacher | `aaditya/OpenBioLLM-Llama3-8B` | 8B, 4-bit NF4 (~5.7 GB VRAM) |
| Student | `Qwen/Qwen3-0.6B` | 0.6B (~0.54 GB) |

1. Teacher generates oracle-labelled clinical reasoning chains (synthetic data)
2. Structured pruning — Taylor importance scoring → prune bottom 40% of attention heads, drop 5 middle layers
3. Knowledge distillation — logit-level KL divergence + feature-level MSE
4. LoRA fine-tuning on the student
5. Shared evaluation harness across 4 datasets

**ESI mapping:** ESI 1+2 → EMERGENCY · ESI 3 → URGENT · ESI 4+5 → ROUTINE

---

## Repository layout

```
├── src/
│   ├── data_generation/     # symptom generation, teacher inference, dataset prep
│   ├── approach2/           # LoRA SFT, dataset merging, MIMIC/fedmml processing, logreg baseline
│   ├── pruning/             # Taylor importance scoring, head pruning, layer dropping
│   ├── distillation/        # KD trainers (v1/v2, + pruned variants), feature KD, CoT distillation
│   ├── finetuning/          # LoRA recovery fine-tune for the pruned model
│   ├── evaluation/          # evaluation across teacher/student/threshold-sweep/PMC/MIMIC
│   └── demo/                # triage chatbot variants
├── notebooks/
│   ├── template.tex         # the canonical full report (LaTeX)
│   ├── evaluation_results.ipynb
│   ├── pruning_analysis.ipynb
│   └── data_exploration.ipynb
├── data/
│   ├── evaluation/          # evaluation outputs backing the results tables above
│   └── approach2/           # sample of the training format + student eval summary
├── docs/
│   └── DETAILED_RESULTS.md  # full metric tables across all 4 datasets and model states
├── report/                  # final report (PDF)
├── app_streamlit.py         # Streamlit demo dashboard
└── requirements.txt
```

**On the data:** `data/` holds the evaluation outputs and one sample file showing the training
format. The full training corpora (~110 MB: the merged v4 set, the 87k-patient fedmml dataset,
and the synthetic generations) are deliberately not committed — they're regenerable from
`src/data_generation/` and `src/approach2/`, and they'd bury the code. The MIMIC-IV-ED demo
subset is available from PhysioNet under ODbL.

---

## Running it

```bash
pip install -r requirements.txt
```

```bash
# 1. Structured pruning
python src/pruning/importancescoring.py
python src/pruning/headpruning.py
python src/pruning/layerdropping.py

# 2. Knowledge distillation
python src/distillation/kdtrainer.py --epochs 1 --batch_size 2

# 3. LoRA fine-tuning (the deployable model)
python src/finetuning/lorafinetune.py --epochs 3 --batch_size 4 --grad_accum 8

# 4. Baseline that beat everything — run this first, honestly
python src/approach2/logreg_baseline.py

# 5. Demo
streamlit run app_streamlit.py
```

Model weights and LoRA adapters are gitignored (regenerable from the scripts above). Training
was done on an NVIDIA A6000 (48 GB).

---

## Reproducibility note

Two accuracy figures for SFT on MIMIC appear across this project's artifacts: **80.2%** (from
`notebooks/template.tex`, the canonical report) and **74.9%** (from an earlier evaluation run,
in `docs/DETAILED_RESULTS.md`). They come from different eval runs and the reconciliation is
still open — flagged here rather than silently picking the higher one. Similarly, "overall"
accuracy figures (SFT 90.8%) average across all four datasets, while the MIMIC-specific figures
above are the honest external-test numbers.

---

## Data & licensing

Evaluation uses the **open demo subset** of MIMIC-IV-ED, distributed by PhysioNet under the
**Open Database License (ODbL)**. It is not redistributed here — download it from PhysioNet and
place it at `mimic-iv-ed-demo-2.2/`, then run the processing scripts in `src/approach2/`. The
full MIMIC-IV-ED dataset is credentialed and is likewise not included.

The fedmml ED triage dataset (87k patients across three countries) is used for the v4 training
mix and is also not redistributed here.
