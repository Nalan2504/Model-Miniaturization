# Medical Triage Assistant — Model Miniaturization
## 20-Minute Presentation Package
### Applied AI Lab, University of Passau — Nalan Thanasekaran & Ajay

---

> **How to use this file:** Everything above "PART 2" is your spoken script.
> Everything below is your slide content. Speaker tags **[NALAN]** and **[AJAY]**
> mark who presents each section. Total speaking time ≈ 20 min + demo.

---
---

# PART 1 — SPOKEN SCRIPT

**Speaker split:**
- **NALAN** — Intro, data pipeline, Approach 2 (data-centric LoRA), the LogReg reckoning, live demo
- **AJAY** — Approach 1 (pruning + distillation), comparative results, honest limitations, conclusion

**Total: 20 minutes.** Timings below add up to ~19 min, leaving ~1 min buffer.

---

## SECTION 1 — Opening & The Real Question  ⏱️ 2 min  **[NALAN]**
### (Slides 1–2)

"Good morning. Our project is called *Model Miniaturization for Medical Triage* —
but I want to be clear from the very start about what the deliverable actually is.

The deliverable is **not** the model. It's **not** the dataset. What we set out to
produce is a **comparative study**: given one small student model — Qwen3-0.6B —
which training and compression method makes it perform best, and how close can it
get to an 8-billion-parameter teacher?

Medical triage — sorting patients into EMERGENCY, URGENT, or ROUTINE — is our
*test case*, our measuring stick. It's a good one because it has a safety-critical
metric: emergency recall. Missing a real emergency is the worst thing this kind of
system can do.

So the question driving everything is: **which method buys the most performance —
especially emergency recall — and at what cost in parameters, memory, and runtime?**

And I'll tell you now — the most valuable thing we found was not a clever model.
It was a series of things that *broke*, and what they taught us. This talk is
honestly structured around our mistakes."

---

## SECTION 2 — The Pipeline & The Teacher Problem  ⏱️ 2.5 min  **[NALAN]**
### (Slides 3–4)

"Here's the setup. Our teacher is OpenBioLLM — an 8B medical model. Our student is
Qwen3-0.6B, about 13 times smaller. The teacher generates training data; the student
is what we actually want to deploy.

Now, our **first instinct** was the obvious one: let the teacher label patient
symptoms and generate reasoning. That failed immediately — and this is failure
number one.

When we asked the teacher to classify, it was **massively EMERGENCY-biased**. It
called 9 out of 10 URGENT cases EMERGENCY. On the syntech-500 real dataset it scored
just 47.7% accuracy, with URGENT recall of 11% and ROUTINE recall of 5%. It basically
sees the world as 'dying' or 'fine', with no middle ground.

So we changed strategy. Instead of letting the teacher decide the label, we used what's
called **oracle-labelling**: *we* write 51 medical conditions with known-correct labels,
and the teacher only generates the clinical *reasoning* for a label we already know is
right. The teacher is a good explainer, not a good classifier — so we use it only for
what it's good at.

That one reframe — caused entirely by a failure — shaped our whole data pipeline."

---

## SECTION 3 — Data Generation: Four Bugs, Four Fixes  ⏱️ 2.5 min  **[NALAN]**
### (Slides 5–6)

"Generating the data itself was a chain of small failures. I'll go through them quickly
because together they show how much iteration this took.

**One** — our parser rejected all 30 pilot samples. It was searching for the exact
string 'TRIAGE LEVEL: EMERGENCY', but the model wrote prose. Fix: regex search for the
label word anywhere in the text.

**Two** — the model gave different answers on repeated runs, even at low temperature.
That broke our consistency filter. Fix: greedy decoding — deterministic, same output
every time.

**Three** — the EMERGENCY bias I mentioned. Fix: oracle-labelling.

**Four** — the strangest one. With oracle-labelling, the model would list the symptoms
and then just *stop* — 30-character outputs, no reasoning. Root cause: it learned from
medical notes that 'Key Symptoms' is often the *last* field. Fix: **response priming** —
we pre-fill the start of the answer so the model is forced to continue into the reasoning.

The result: 5,100 clean, balanced samples. Then we validated their quality — and even
that gave us a negative result worth keeping."

---

## SECTION 4 — Validating Data: A Negative Result We Kept  ⏱️ 2 min  **[NALAN]**
### (Slides 7–8)

"To check our data quality we trained BioBERT on it and tested cross-dataset. It
**failed completely** — 15 to 20% accuracy, everything predicted ROUTINE.

But the failure was informative. Two root causes: first, **label leakage** — our
reasoning text literally begins with the label, so BERT learned to read the label
token instead of reasoning. Second, **format mismatch** — our data is verbose
narratives, the test set is short keyword lists.

The lesson: cross-dataset BERT is the *wrong* validation tool. So we switched to
**BERTScore** — comparing our teacher's reasoning against real PubMedQA medical text
semantically. That scored **0.82**, comfortably above the random floor of 0.5,
confirming the reasoning is genuinely coherent clinical language.

So: one negative result, one positive confirmation. We kept both in the report,
because the negative one changed how we validate."

---

## SECTION 5 — Approach 2: Data-Centric LoRA, v1 → v4  ⏱️ 3 min  **[NALAN]**
### (Slides 9–10)

"Now the first real method: fine-tune the student with LoRA — training only 1.67% of
the parameters, about 10 million.

**Version 1**: synthetic plus the syntech real data. Accuracy 90.8%, macro-F1 0.909 —
above target. But emergency recall was **82.7%** — below our 95% bar. 17% of real
emergencies were being classified as URGENT. Not safe enough.

We tried to fix it with logit-thresholding. **That failed** — the model's verbose
output meant no single-token threshold could separate the classes.

So we did the honest thing: we added **real ESI-labelled data**. First MIMIC —
real ED patients. **Version 3**, 6,332 samples, hit 89.6% emergency recall on truly
unseen MIMIC patients. Big jump.

Then **version 4** — we added the fedmml dataset, 87,000 real ED patients across three
countries. Crucially, we held out an entire country — Latvia — as a domain-shift test,
so the model never saw it. Version 4 hit **100% accuracy and 100% emergency recall on
Latvia**.

At which point we were feeling pretty good about ourselves. And that's exactly when
the project's biggest lesson arrived — and I'll hand over to Ajay, because he's the one
who found it."

---

## SECTION 6 — Approach 1: Structured Pruning  ⏱️ 2 min  **[AJAY]**
### (Slides 11–12)

"Thanks. So while Nalan built the data-centric pipeline, I worked on the actual
*compression* — because 'miniaturization' should mean making the model structurally
smaller, not just training it well.

I used **Taylor importance scoring** — estimating how much each attention head and
layer contributes to the output — then removed the bottom 40% of heads and dropped
5 middle layers. Early layers handle basic features, late layers make decisions;
the middle is the most redundant, so it's the safest to cut.

After pruning I ran a LoRA recovery fine-tune, and the results were genuinely good:
on MIMIC, the pruned model held **74.4% accuracy and 80% emergency recall** — close
to the full model.

But I have to be honest about a limitation: this is **soft pruning**. I zeroed the
weights, but the architecture config still says 28 layers, 16 heads. On disk it's
17% smaller, but it's not *structurally* compressed yet. True architectural pruning
is future work, and we say so plainly in the report."

---

## SECTION 7 — Approach 1: Knowledge Distillation & The Collapse  ⏱️ 2 min  **[AJAY]**
### (Slides 13–14)

"Then knowledge distillation — teaching the student to match the teacher's soft
probability distribution, its so-called 'dark knowledge'.

This is where I hit the hardest failure of the project. My first distilled model
scored 35% accuracy and **0% emergency recall** — it predicted URGENT for basically
everything.

I traced the root cause to three things: alpha was 0.7, so the teacher's signal
dominated — and remember, the teacher is EMERGENCY-averse. Temperature was 4.0, which
over-blurred the class boundaries. And there were no class weights, so missing an
emergency wasn't penalized. **The student faithfully inherited the teacher's weakness.**

That's actually a finding: distillation transfers a teacher's *flaws*, not just its
strengths.

I built a version 2 — alpha down to 0.5, temperature down to 2.0, emergency weight up
to 3.0. It fixed the 0% recall... and **overcorrected**. Now it predicts EMERGENCY for
*everything* — 100% recall but only 57% accuracy. The weight of 3.0 was too aggressive;
around 1.5 would balance it. That version-3 tuning is documented as our next step."

---

## SECTION 8 — The LogReg Reckoning  ⏱️ 2 min  **[AJAY]**
### (Slides 15–16)

"Now the moment that reframed our entire evaluation. I built the dumbest possible
baseline as a sanity check: **TF-IDF plus Logistic Regression**. No neural network,
no deep learning — just word counting.

Two results stopped us cold.

**One**: on Latvia — the test where our fancy v4 model scored a triumphant 100% —
LogReg *also* scored a perfect F1 of 1.0. A bag-of-words model aced it. Which means
Latvia is **trivially easy** — the complaints are standardized keywords. Our 100%
proved nothing about intelligence. It proved the test was too easy.

**Two**: on MIMIC — real, messy, free-text clinical notes — LogReg hit **95.8%
emergency recall**, *higher* than every one of our neural models.

This was humbling, but it's genuinely our strongest scientific contribution: for
keyword-driven triage on standardized complaints, **model complexity does not
guarantee better safety-critical recall**. It told us MIMIC is the real benchmark,
not Latvia — and it's why we now report MIMIC numbers everywhere."

---

## SECTION 9 — Comparative Analysis + LIVE DEMO  ⏱️ 2.5 min  **[NALAN + AJAY]**
### (Slides 17–18)

**[AJAY sets up the table]**

"So here's the whole comparison on MIMIC, our hard benchmark. Notice: the 0.6B student,
fine-tuned, actually *beats* the 8B teacher — because the teacher's output formatting
collapses on real clinical notes, 96% unparseable. The student wins at 13× fewer
parameters and 10× less VRAM."

**[NALAN runs the demo]**

"And to make this concrete — here's our live demo. Same patient, three different
training methods, side by side.

- The **SFT model** — our best — reads the symptoms and produces full structured
  reasoning with the correct triage level, live.
- The **Pruned model** — we show as a documented finding: after layer-dropping, its
  free-form generation degraded; it still classifies via keyword scan but no longer
  writes clean reasoning. That's the cost of compression, shown honestly.
- The **Distilled model** — points to a class from its logits but, because of the
  collapse we described, doesn't produce clean text. Also shown as a documented card.

That contrast *is* the comparative analysis: you can literally see how each method
changes the model's behaviour on the exact same input."

---

## SECTION 10 — Honest Limitations & Conclusion  ⏱️ 1.5 min  **[AJAY]**
### (Slides 19–20)

"Let me close by owning our limitations directly, because they're the point.

- Our pruning is **soft** — not yet true structural compression.
- Our distillation is **fragile** — it inherits the teacher's emergency-aversion, and
  our fix overcorrected.
- Our headline Latvia result is on a **trivially easy** test — LogReg proved it.
- And a dumb baseline still **beats us on MIMIC emergency recall** before calibration.

But here's what we *did* establish, honestly:

- A 0.6B student can **outperform an 8B teacher** on triage at a fraction of the cost.
- **Format-robustness, not raw model size, is the real bottleneck** — every method
  broke identically on unstructured PMC case notes, at 85% parse failure.
- And the single most useful comparison in the whole project came from the *simplest*
  possible model.

We didn't get this right on the first try. We got it right on roughly the fourth try,
and the wrong tries taught us more than the right ones. Thank you — we're happy to take
questions."

---
---

# PART 2 — SLIDE CONTENTS

> Design note: keep bullets short. Speaker notes are in PART 1.
> Suggested theme: dark clinical (matches the demo). Red/amber/green accents
> for EMERGENCY / URGENT / ROUTINE.

---

### SLIDE 1 — Title  **[NALAN]**
**Model Miniaturization for Medical Triage**
- A comparative study of training & compression methods
- Teacher: OpenBioLLM-8B → Student: Qwen3-0.6B (13× smaller)
- Nalan Thanasekaran & Ajay · Applied AI Lab · University of Passau

---

### SLIDE 2 — The Real Question  **[NALAN]**
**What is the deliverable?**
- NOT the model. NOT the dataset.
- **A comparative study:** which method makes a 0.6B student perform best?
- Triage (EMERGENCY / URGENT / ROUTINE) = the test case
- Safety-critical metric: **emergency recall** (missing an emergency = worst case)
- Core question: most performance, at what cost (params / VRAM / runtime)?

---

### SLIDE 3 — The Pipeline  **[NALAN]**
**Teacher → Data → Student**
- Teacher (8B) generates reasoning data → Student (0.6B) is deployed
- Two approaches compared:
  - **Approach 2:** data-centric LoRA fine-tuning
  - **Approach 1:** structured pruning + knowledge distillation
- One shared evaluation harness for fair comparison

---

### SLIDE 4 — Failure #1: The Teacher Can't Classify  **[NALAN]**
**First instinct failed immediately**
- Plan: let teacher label + reason → ❌
- Teacher is **EMERGENCY-biased**: 9/10 URGENT cases → EMERGENCY
- syntech-500: 47.7% acc · URGENT recall 11% · ROUTINE recall 5%
- **Fix → Oracle-labelling:** we assign correct labels; teacher only writes reasoning
- Insight: teacher is a good *explainer*, not a good *classifier*

---

### SLIDE 5 — Data Generation: 4 Bugs, 4 Fixes  **[NALAN]**
**Every step broke before it worked**
1. Parser rejected all 30 samples → regex search for label word
2. Inconsistent outputs across runs → greedy decoding (deterministic)
3. EMERGENCY-bias on labels → oracle-labelling
4. Model stops after symptoms (EOS) → **response priming** (pre-fill answer start)

→ Result: **5,100 clean, balanced samples**

---

### SLIDE 6 — Why Priming Worked  **[NALAN]**
**Root cause + fix (Failure #4)**
- Model learned "Key Symptoms" is often the *last* field in medical notes
- So it emitted 30-char outputs and stopped — no reasoning
- Fix: pre-fill `TRIAGE LEVEL: {label}\nKEY SYMPTOMS:` → forces continuation
- Lesson: the model wasn't broken — it was following a learned pattern

---

### SLIDE 7 — Validating Data: A Negative Result  **[NALAN]**
**BioBERT cross-dataset test → total failure**
- Trained BioBERT on our data, tested cross-dataset: 15–20% acc, all ROUTINE
- Root cause 1: **label leakage** (reasoning starts with the label)
- Root cause 2: **format mismatch** (narratives vs keyword lists)
- Lesson: cross-dataset BERT is the WRONG validation tool

---

### SLIDE 8 — Validating Data: The Right Tool  **[NALAN]**
**BERTScore vs real medical literature**
- Compared teacher reasoning to PubMedQA (semantic similarity, no leakage)
- **F1 = 0.82** (random ≈ 0.5, threshold 0.85)
- Confirms: reasoning is coherent clinical language, not hallucination
- Kept BOTH results — the negative one changed our methodology

---

### SLIDE 9 — Approach 2: LoRA Fine-Tuning v1  **[NALAN]**
**Train only 1.67% of params (10M)**
- v1: synthetic + syntech-500 → Acc **90.8%**, F1 **0.909** ✅
- BUT emergency recall **82.7%** ❌ (target >95%)
- 17% of real emergencies misclassified as URGENT
- Logit-threshold fix → **failed** (verbose output, no separable threshold)

---

### SLIDE 10 — Approach 2: v3 → v4 (Add Real Data)  **[NALAN]**
**Iteration fixed what tricks couldn't**
- v3 (+ MIMIC real patients, 6,332 samples): MIMIC EM-recall **89.6%** 📈
- v4 (+ fedmml 87K real ED patients, 42,872 samples)
- Held out entire country (Latvia) as domain-shift test
- v4 on Latvia: **100% acc / 100% EM-recall** ✅
- (…but hold that thought — see Slide 15)

---

### SLIDE 11 — Approach 1: Structured Pruning  **[AJAY]**
**Making the model structurally smaller**
- Taylor importance scoring (gradient × weight magnitude)
- Removed bottom 40% attention heads
- Dropped 5 middle layers (most redundant)
- Recovery LoRA → MIMIC: **74.4% acc / 80% EM-recall**

---

### SLIDE 12 — Pruning: The Honest Caveat  **[AJAY]**
**Soft pruning ≠ true compression**
- Weights zeroed, but architecture config unchanged (heads=16, layers=28)
- On disk: 17% smaller (1.2 GB → 998 MB)
- NOT yet structurally smaller → true architectural pruning is future work
- We state this plainly — no overclaiming

---

### SLIDE 13 — Approach 1: Knowledge Distillation  **[AJAY]**
**Teaching the student the teacher's "dark knowledge"**
- Match teacher's soft probability distribution (KL divergence)
- v1: alpha=0.7, T=4.0
- Result: 35% acc · **0% emergency recall** — predicts URGENT for everything ❌

---

### SLIDE 14 — Distillation: Root Cause & The Overcorrection  **[AJAY]**
**KD transfers the teacher's WEAKNESSES too**
- Root cause: alpha 0.7 (teacher dominates) + T=4.0 (blurs boundary) + no class weights
- Teacher is EMERGENCY-averse → student inherited it
- v2 fix (alpha 0.5, T 2.0, EM weight 3.0) → **overcorrected**: predicts EMERGENCY for all
- 57% acc / 100% recall · weight 3.0 too high (~1.5 would balance) → v3 = next step

---

### SLIDE 15 — The LogReg Reckoning (1/2)  **[AJAY]**
**The dumbest baseline broke our best result**
- Built TF-IDF + Logistic Regression (no neural net)
- On Latvia — where v4 scored 100% — LogReg also scored **F1 = 1.0**
- → Latvia is **trivially easy** (standardized keyword complaints)
- Our 100% proved the test was easy, not that the model was smart

---

### SLIDE 16 — The LogReg Reckoning (2/2)  **[AJAY]**
**Our strongest scientific finding**
- On MIMIC (real, messy free-text): LogReg **95.8% EM-recall**
- Higher than EVERY neural model we trained
- Finding: **model complexity ≠ better safety-critical recall**
- → MIMIC becomes our real benchmark; all numbers now reported on MIMIC

---

### SLIDE 17 — Comparative Analysis (MIMIC)  **[AJAY]**
**Everything on the hard benchmark**

| Model | Acc | EM-Recall | Params | VRAM |
|---|---|---|---|---|
| LogReg baseline | 88.1% | **95.8%** | tiny | tiny |
| Teacher (8B) | ~0%* | — | 8B | 5.7 GB |
| **SFT v4 (student)** | **80.2%** | **90.4%** | 0.6B | 0.54 GB |
| Pruned + SFT | 74.4% | 80.0% | 0.6B | 0.54 GB |
| Distilled KD v1 | 42.9% | 0% | 0.6B | 0.54 GB |

*Teacher: 96% unparsed on real MIMIC notes
**Student beats teacher at 13× fewer params, 10× less VRAM**

---

### SLIDE 18 — LIVE DEMO  **[NALAN]**
**Same patient · three training methods · side by side**
- 🟢 **SFT v4** — LIVE: full structured reasoning + correct triage
- 🟠 **Pruned + SFT** — documented: generation degraded after layer-drop (keyword-scan only)
- 🔴 **Distilled KD** — documented: logit points to class, text collapsed
- → You can SEE how each method changes model behaviour
- *(Demo: https://…gradio.live)*

---

### SLIDE 19 — Honest Limitations  **[AJAY]**
**What we own**
- Pruning is **soft** — not true structural compression yet
- Distillation is **fragile** — inherits teacher bias; v2 overcorrected
- Latvia 100% is on a **trivially easy** test (LogReg proved it)
- LogReg still **beats us** on MIMIC EM-recall before calibration

---

### SLIDE 20 — Conclusion  **[AJAY]**
**What we established — honestly**
- A 0.6B student can **outperform an 8B teacher** on triage (13× smaller)
- **Format-robustness, not size, is the real bottleneck** (all methods fail PMC @ 85%)
- The **simplest model gave the most useful comparison**
- We got it right on the ~4th try — the wrong tries taught us the most
- *Thank you — questions?*

---
---

# APPENDIX — Q&A Prep (not slides, for your reference)

**Q: Why does the student beat the teacher?**
The teacher's instruction-following collapses on real clinical notes (96% unparsed on
MIMIC). The fine-tuned student reliably produces parseable structured output. Beating
it on effective accuracy is largely about robustness to real-world format.

**Q: Isn't oracle-labelling cheating?**
For data *synthesis*, no — it's standard CoT distillation practice. We're not testing
whether the teacher can classify (it can't reliably); we use it only to generate
reasoning for labels we know are correct.

**Q: Why is soft pruning still worth showing?**
It demonstrates the importance-scoring + recovery pipeline works, and quantifies the
accuracy cost of layer-dropping. It's an honest intermediate result; structural
pruning is the clearly-stated next step.

**Q: If LogReg wins, why use neural models?**
That IS our finding — for keyword-driven triage on standardized complaints, complexity
doesn't help recall. Neural models add value on messier input and on generating
reasoning, but we don't pretend they dominate on every axis.

**Q: What would you do next?**
KD v3 with emergency weight ~1.5 (balance the overcorrection); true structural pruning;
LoRA hyperparameter ablation; and a RAG variant for format-robustness on free text.

**Q: How reproducible is this?**
Fully — random.seed(42) for data generation, fixed 80/10/10 splits confirmed disjoint,
one shared evaluation harness across all methods.
