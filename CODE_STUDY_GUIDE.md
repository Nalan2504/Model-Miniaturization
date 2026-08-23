# Code Explanation Study Guide — Model Miniaturization

One-paragraph summary of every file in the repo, in pipeline order, with the "why" behind each design choice and likely questions. Read top to bottom for the full story; skim headers tomorrow morning.

---

## PART 0 — The Big Picture

**The problem:** compress `OpenBioLLM-Llama3-8B` (teacher, 8B params) into `Qwen3-0.6B` (student, 0.6B params) for medical triage — classify a patient description into **EMERGENCY / URGENT / ROUTINE**.

**Two independent approaches, same evaluation harness:**
- **Approach 1 (compression-centric, Ajay):** prune the student's architecture (heads + layers), then use knowledge distillation from the teacher to recover lost capability.
- **Approach 2 (data-centric, you):** keep the student's architecture untouched; instead build a bigger/better training set (synthetic + real) and LoRA fine-tune.

**Why this matters if asked "what's the contribution?":** it's not one model — it's a *controlled comparison* of compression techniques on the same task with the same eval protocol, so results are directly comparable.

---

## PART 1 — Data Generation (`src/data_generation/`)

### `symptom_generator.py`
Hand-written Python functions that generate 5,100 synthetic patient descriptions — 51 medical conditions (15 EMERGENCY, 18 URGENT, 18 ROUTINE), 100 samples each, using `random.seed(42)` for reproducibility. Each generator function (e.g. `gen_acute_mi()`, `gen_appendicitis()`) fills in randomized age, vitals, symptom variants via template strings. Output: `symptoms_5k.jsonl` with `{symptom_description, triage_level}`.

**Why generate rather than find real data?** Real labeled clinical triage data is scarce/private. Synthetic generation with *known* ground-truth labels avoids the teacher's own biases in labeling.

**Likely Q:** *"Why 51 conditions with fixed counts instead of a bigger pool?"* → Balance is deliberate — equal counts per condition prevents the model from learning to guess the majority class. *"Why seed 42?"* → Reproducibility, standard convention.

### `teacher_inference.py`
Feeds each of the 5,100 symptom descriptions to the teacher (OpenBioLLM-8B, loaded 4-bit NF4) using **oracle-label prompting**: the prompt tells the teacher the correct triage level and asks it to *justify* that label with clinical reasoning (not decide the label itself). Uses a **response primer** (`TRIAGE LEVEL: {label}\nKEY SYMPTOMS:`) to force the teacher to continue in the exact format rather than refuse or ramble. Has a `--resume` flag (counts existing output lines, skips them) so a long generation run can be interrupted and restarted safely. Output: `train_samples.jsonl` with `{symptom_description, triage_level, confidence, raw_output}`.

**Why oracle-label prompting instead of just asking the teacher to classify?** Two reasons that come up in the report: (1) the teacher is known to be EMERGENCY-biased when asked to decide for itself, and (2) this makes labels *correct by construction* — the label came from our curated generator, and the teacher only supplies reasoning text, not the ground truth.

**Likely Q:** *"What if the teacher's generated reasoning doesn't actually match the label?"* → That's exactly what BERTScore + the NLI filter later probe (see Part 4).

### `prepare_dataset.py`
Stratified 80/10/10 split of the 5,100 synthetic samples into `train.jsonl` (4,080) / `val.jsonl` (510) / `test.jsonl` (510), stratified **per class** (so EMERGENCY/URGENT/ROUTINE ratios are preserved in each split) using a seeded `random.Random(42)`.

**Why this file exists separately / is important:** this is the split that was later found to have been done *incorrectly* upstream in one early version (`train_samples.jsonl` used instead of `train.jsonl` caused leakage — test samples ended up in training). This file is the fix — it's the canonical, leak-free split.

---

## PART 2 — Approach 1: Structured Pruning (`src/pruning/`) — Ajay's work

### `importancescoring.py`
Runs a forward pass of the base `Qwen3-0.6B` on a 128-sample calibration batch (from `train.jsonl`) with `output_attentions=True`, and computes an importance score per attention head as the **mean squared attention weight**: `score = mean(attention_weights²)` over batch and sequence dims. Also averages per layer. Saves `head_scores.json` and `layer_scores.json`.

**Why mean-squared attention as the importance proxy?** It's a simple, cheap, first-order signal — heads that consistently produce near-zero attention everywhere are probably not doing useful work. (The report notes this is deliberately simpler than second-order methods like SparseGPT, chosen for tractability on a single GPU.)

**Likely Q:** *"Isn't this a crude importance measure?"* → Yes, explicitly acknowledged — it's a proxy, not a rigorous Taylor-expansion-based score. Good enough to find a clear bottom 40% to prune.

### `headpruning.py`
Loads the head importance scores, and for each of the 28 layers, ranks its 16 heads by score and **zeros out** the bottom 40%'s Q/K/V projection weight slices (`q_proj[:, start:end] = 0` etc., where `start/end` are the head's slice in the concatenated projection matrix). Saves the modified model to `qwen3_pruned_heads/`.

**Critical detail — this is "soft" pruning:** the weight *values* are set to zero, but the **tensor shapes and model config are unchanged**. The model still reports 16 heads/layer in its config. This is the single most-asked limitation in the whole project.

**Likely Q:** *"Why not actually remove the heads (resize the tensors)?"* → Simpler to implement (no need to rewrap the model class/config), avoids breaking downstream tooling that expects standard shapes — but the trade-off is that **file size and inference latency don't improve**, because the zeroed heads are still computed by the GPU, just multiplying by zero. True structural pruning (resizing) is listed as future work.

### `layerdropping.py`
Loads the head-pruned model, then **physically removes** 5 specific middle layers (indices `{10, 12, 14, 16, 18}` out of 28) by rebuilding `transformer.layers` as a shorter `ModuleList` that skips those indices. Saves to `qwen3_pruned_heads_layers/`.

**Why this one IS structural (unlike head pruning)?** Because dropping whole `ModuleList` entries actually shrinks the number of layers the model computes — 28 → 23, an 18% reduction in depth that *does* reduce compute per forward pass, unlike the head-zeroing.

**Likely Q:** *"Why these specific 5 layers?"* → Fixed indices chosen as "middle" layers by inspection/heuristic in this implementation (not directly driven by `layer_scores.json` in the code as-is — worth being honest about this if asked precisely how they were selected).

---

## PART 3 — Approach 1: Knowledge Distillation (`src/distillation/`) — Ajay's work

All KD trainers share the same core idea: only a **single label word** matters (EMERGENCY/URGENT/ROUTINE), so instead of distilling the full next-token vocabulary distribution (standard KD), they extract just the **first-token logit for each of the 3 label words** from both teacher and student, then do KD directly on that 3-class distribution.

### `kdtrainer.py` — KD v1 (the first, flawed attempt)
Full-parameter fine-tune (no LoRA) of the *unpruned* student against the teacher. Loss = `α · KL(teacher_3class ‖ student_3class) + (1-α) · CrossEntropy(student, true_label)`, with **`α=0.7, T=4.0`** as defaults.

**Why it failed (report's root-cause):** `α=0.7` weights the KD term too heavily, so the student mostly imitates the teacher's (EMERGENCY-averse) soft labels rather than learning from ground truth; `T=4.0` blurs the softmax so much it washes out the boundary between classes. Result: student learns to avoid predicting EMERGENCY.

### `kdtrainer_pruned.py`
Identical logic to `kdtrainer.py`, but the student base is the **pruned** model (`qwen3_pruned_heads_layers`) instead of the vanilla one. Also full-parameter fine-tune, same `α=0.7/T=4.0` defaults — inherits the same failure mode.

### `kdtrainer_v2.py` — the fix, LoRA on the *unpruned* base
Three changes from v1: **`α=0.5`** (balance KD/CE rather than KD-dominant), **`T=2.0`** (sharper, less blur), and a **3× class weight on EMERGENCY** in the cross-entropy term (`CLASS_WEIGHTS = [3.0, 1.0, 1.0]`) to penalize missing an emergency harder. Also adds an **optional `--seq-ce` flag**: in addition to the 3-class logit loss, optionally also train the model to generate the actual response *text* token-by-token, specifically to prevent "vocab collapse" (model nails the logits but forgets how to write full sentences).

### `kdtrainer_pruned_v2.py` — the stabilized final version
This is the one that actually worked. Two more critical fixes on top of v2's changes:
1. **Teacher compute dtype forced to `bfloat16`** (via `bnb_4bit_compute_dtype=torch.bfloat16`) — fixes a **`float16` numerical overflow bug**: the teacher's attention scores occasionally exceeded float16's max representable value (~65,504) and overflowed to `Inf`, corrupting the KD loss with `NaN`.
2. **Distillation restricted to a frozen pruned base + a small trainable LoRA adapter** (`get_peft_model` with `r=16, alpha=32`) instead of full-parameter fine-tuning — this is what stopped the earlier catastrophic majority-class collapse, because only a small number of parameters can move, so the update can't as easily wreck the whole model. Also adds gradient clipping (`clip_grad_norm_(max_norm=1.0)`) and early stopping on validation loss.

**Likely Q — the single most important KD question:** *"Walk me through why KD failed twice before it worked."* → (1) v1: unrestricted full-parameter update + α too high + T too high → EMERGENCY-averse collapse. (2) v2 attempt at fixing weights alone, still full-parameter → hit the float16 NaN bug, and even after fixing that, over-corrected into predicting the majority class. (3) Only when combined with **frozen base + LoRA adapter** (restricting *how much* can change) did training stabilize. The lesson: it wasn't just the loss hyperparameters — it was that full-parameter fine-tuning on a narrow 3-token objective is inherently unstable.

### `featurekd.py` / `cotdistillation.py` — unused stubs
`featurekd.py`: a single MSE-loss helper function (`feature_kd_loss`) for matching hidden states between teacher/student — never wired into a trainer. `cotdistillation.py`: a single cross-entropy helper for distilling chain-of-thought token sequences — also never wired in. **If asked:** these are documented as ideas that were scoped but not pursued, not bugs.

---

## PART 4 — Approach 2: Data-Centric Pipeline (`src/approach2/`) — your work

### `process_mimic.py`
Reads the local MIMIC-IV-ED demo CSV (`triage.csv.gz`, 207 rows), maps ESI acuity → 3-class label (**ESI 1,2→EMERGENCY; 3→URGENT; 4,5→ROUTINE**), builds a patient description string from chief complaint + available vitals (HR, BP, O2, temp, RR, pain — each optionally present, skipped if missing/unparseable), and writes an `output` field that's a templated `TRIAGE LEVEL / KEY SYMPTOMS / CLINICAL REASONING / CONFIDENCE` block (using a fixed rationale per ESI level, not teacher-generated). Output: `mimic_train.jsonl`.

**Why templated output instead of teacher-generated reasoning for real data?** Real ESI labels are already ground truth from actual triage nurses — no need for the teacher to justify them; a fixed rationale string per ESI level is sufficient and much cheaper than running the 8B teacher over every row.

### `process_fedmml.py`
Same idea but for the much larger `fedmml-ed-triage` dataset (87K real ED patients across multiple countries). Does two important things:
1. **Domain-shift split by country**: holds out **all of Latvia** (3,000 samples, 1,000/class) as the test set, trains only on Denmark+Turkey (~38K balanced).
2. **Balances the training pool**: keeps all EMERGENCY cases, downsamples URGENT/ROUTINE to match — because fedmml already has abundant real EMERGENCY cases (no need to upsample, unlike the earlier MIMIC-only approach).

**Why hold out a whole country instead of a random split?** To simulate genuine geographic/hospital domain shift rather than random sampling from the same distribution.

**Important honest caveat (this becomes the "Latvia benchmark vulnerability" finding):** fedmml uses **standardized, templated complaint phrasing across all its countries** — so even though Latvia is a different country, the *vocabulary* is shared with training. This was later discovered via the LogReg probe (see below) to make Latvia a much easier/less meaningful benchmark than MIMIC.

### `merge_mimic.py` (the "v4" builder — the one actually used)
Combines four sources into the final training set: (1) `train.jsonl` synthetic (4,080), (2) `syntech-ai/medical-triage-500` downloaded from HuggingFace (500), (3) `mimic_train.jsonl` (207), (4) `fedmml_train.jsonl` (~38K). No upsampling needed (`UPSAMPLE_FACTOR=1`) because fedmml already balances EMERGENCY. Output: `combined_train_v4.jsonl` — the file actually used for the final LoRA fine-tune.

### `merge_datasets.py` / `merge_perclass.py` — earlier, superseded versions
Both combine NLI-*filtered* synthetic data with syntech-500 (the "v1" pipeline, before MIMIC/fedmml were added). Kept in the repo as history — `merge_mimic.py` is the file that was actually used for the final model.

### `nli_filter.py`
Uses a pretrained NLI model (`cross-encoder/nli-deberta-v3-large`) to score whether each synthetic sample's clinical reasoning text **entails** a class-specific hypothesis (e.g. for EMERGENCY: *"This patient requires immediate emergency intervention within minutes..."*). Keeps samples above a threshold (default 0.5) as a quality filter.

**Why it exists but isn't in the final pipeline:** documented as a **tried-and-discarded** technique — it removed under 4% of samples, and the marginal quality benefit didn't justify the extra complexity/compute of running a second large model over the whole dataset. This is a good example to cite if asked "what didn't make it into the final pipeline and why."

### `upsample_emergency.py` — early, superseded
Simple 2× duplication of EMERGENCY samples in the dataset — used in an early iteration before fedmml provided enough real EMERGENCY cases naturally. Superseded once `merge_mimic.py` v4 didn't need upsampling.

### `lora_finetune.py`
The actual LoRA fine-tuning script for Approach 2's final model. Loads `Qwen3-0.6B` in 4-bit NF4, applies LoRA (**r=16, α=32, dropout=0.05**) to all 7 projection matrices (`q/k/v/o_proj`, `gate/up/down_proj`), trains with HuggingFace `Trainer` (3 epochs, batch=4, grad_accum=8 → effective batch 32, lr=2e-4 cosine schedule, early stopping on eval_loss patience=1). Uses `DataCollatorForSeq2Seq` with `-100` label masking so loss is only computed on the response tokens, not the prompt.

**Likely Q:** *"Why LoRA rank 16, not higher/lower?"* → r=16 is a common default balance between expressiveness and trainable-parameter count (~1.7% of total params here); not exhaustively swept in this study (explicitly listed as future work — a "broader LoRA ablation").

### `evaluate_baseline.py` / `evaluate_student.py`
Nearly identical evaluation harnesses — the difference is `evaluate_baseline.py` loads the **base model with no adapter** (the zero-shot reference point) while `evaluate_student.py` loads base + a LoRA adapter. Both: build the same chat-format prompt, greedy-decode a short response, regex/substring-extract the predicted label, compute per-class precision/recall/F1 + overall accuracy + macro-F1, and separately count **unparseable outputs** (failures where no label word appears at all).

**Why track "failed/unparsed" separately from wrong predictions?** This is the "format collapse" finding — an un-tuned model can be *behaviorally* wrong not because it reasons badly but because it doesn't follow the output format at all (e.g. zero-shot student fails to parse 95% of the time). Counting these separately from genuine misclassifications is what makes that distinction visible.

### `evaluate_student_logit.py`
An alternative, more surgical evaluation: instead of parsing the final generated text, it scans the generated tokens to find *where* the label word actually appears, then reads the softmax probability at exactly that position, computing `P(EMERGENCY) / (P(EMERGENCY)+P(URGENT))` and sweeping decision thresholds. Prints the mean EM-ratio per true class to show whether there's a separable threshold.

**Why build this at all if `evaluate_threshold_sweep.py` (below) does something similar?** This was an earlier iteration exploring whether a *single-token* logit ratio could rescue the ~83% argmax emergency recall to >95% without retraining. It's referenced in the report as part of the recall-fix investigation trail before the team moved to real-data augmentation instead.

### `logreg_baseline.py`
A classical, **non-neural** baseline: TF-IDF vectorizer (`ngram_range=(1,2)`, 50K max features) + `LogisticRegression` (`class_weight="balanced"`), trained on the exact same `combined_train_v4.jsonl` text field the neural models see, evaluated on MIMIC (or Latvia).

**Why this file is one of the most important in the whole project:** it produced the "Latvia benchmark vulnerability" finding — a bag-of-words model with **zero clinical reasoning capability** achieves a *perfect* F1 on Latvia, proving Latvia doesn't actually test generalization (just memorized vocabulary), while the same baseline scores far lower on MIMIC. This single script is what justified promoting MIMIC to the primary benchmark.

**Likely Q:** *"Why would a simple LogReg beat or match neural models on emergency recall?"* → For keyword-driven, template-heavy real clinical phrasing (fedmml, and to a lesser extent MIMIC), the signal is largely in specific words ("crushing chest pain," "BP 90/60") that a bag-of-words model can catch just as well as a language model — model complexity doesn't automatically buy safety-critical performance on this kind of surface-pattern task.

### `split_mimic_train.py`
Simple 80/20 shuffle-split of the 207 MIMIC demo patients into `mimic_train_split.jsonl` (165, went into training) and `mimic_test.jsonl` (42, held out) — this is the **clean, honest MIMIC test set** used for all the tuned-model headline numbers (80.2% acc / 90.4% EM recall etc.), as opposed to evaluating on all 207 (which would be partly evaluating on data the model trained on).

### `student_latency_per_sample.py`
Loads the final adapter, runs 3 warmup + 20 measured `generate()` calls on one sample, reports avg/median/min/max per-sample latency (`torch.cuda.synchronize()` used to get accurate GPU timing). This is where the **0.907s/sample** and the general "under 1 second" latency claim comes from.

---

## PART 5 — Evaluation (`src/evaluation/`)

### `evaluate_teacher.py` / `evaluate_teacher_n.py`
Run the **teacher** (not student) zero-shot on syntech-500 or a given local JSONL test set respectively, same metric harness as the student evaluators. This is what produces the teacher's baseline numbers used in the "13.3× smaller, still beats the teacher" comparison. `evaluate_teacher_n.py` is essentially the same script generalized to take any local JSONL (used for the Latvia run).

### `evaluate_bert.py`
Fine-tunes `BioBERT` (a 3-class sequence classifier, not a triage LLM) on the synthetic data (symptom + teacher's raw reasoning text concatenated) and evaluates on syntech-500. This is a **data-quality probe**, not part of the main pipeline — the idea being: if a completely different, independently-pretrained model can learn the task from this synthetic data, that's evidence the data has real signal, not just memorizable artifacts of the generation templates.

### `evaluate_reasoning_quality.py`
Computes **BERTScore** (precision/recall/F1 via `roberta-large`) comparing the teacher's generated `CLINICAL REASONING:` text against real clinical prose from `PubMedQA`. Score ≈ **0.82 F1 across all three classes**, used as evidence the synthetic teacher reasoning is semantically coherent (not degenerate template filler) — and, importantly, **equally coherent across classes** (no class-specific quality collapse).

### `evaluate_pmc.py`
Two-phase out-of-distribution probe using `PMC-Patients` (real published clinical case narratives with **no triage labels**). Phase 1: the teacher labels 200 cases (acting as a pseudo-ground-truth oracle since no real labels exist). Phase 2: the student labels the same cases. Metric: **student-teacher agreement**, not accuracy against ground truth (there isn't any). This tests whether the pipeline generalizes to genuinely unstructured, free-text clinical narratives very different from the structured training format.

**Likely Q:** *"Why measure agreement with the teacher instead of accuracy?"* → PMC-Patients has no triage labels at all — there's no ground truth to compare against, so the teacher's own prediction is used as the best available reference, explicitly flagged as a limitation (an oracle, not a gold standard).

### `evaluate_distilled.py` / `evaluate_distilled_logits.py`
The same generation-based / logit-based evaluation pattern as `evaluate_student.py` / `evaluate_student_logit.py`, but generalized via `--model_path` to point at *any* distilled model directory (KD v1, v2, pruned variants) rather than hardcoding a specific adapter.

### `evaluate_student_mimic.py`
A MIMIC-specific evaluator — generation-based, loads any model/adapter, evaluates specifically against `mimic_test.jsonl` (the clean 42-sample held-out split), prints the standard metrics. This is the canonical script behind the primary reported MIMIC numbers.

### `evaluate_threshold_sweep.py`
The most sophisticated evaluation script. Instead of letting the model generate freely, it **primes** the prompt to end exactly at `"...assistant\nTRIAGE LEVEL: "` so the *very next token* must be the label word, then does a single forward pass (no generation loop) and reads the raw logits for the 3 label tokens directly from the vocabulary distribution at that position. Converts to probabilities via softmax, then sweeps decision thresholds `t` from 0.05 to 0.95: predict EMERGENCY whenever `P(EMERGENCY) > t`, otherwise pick whichever of URGENT/ROUTINE has higher probability. Accepts either `--adapter` (LoRA) or `--base` (a full fine-tuned model like KD v2, which isn't a LoRA adapter) so it can evaluate any model type.

**Why prime-then-single-forward-pass instead of generating and parsing text?** Much faster (no autoregressive generation loop — literally one forward pass per sample), and it directly measures the model's calibrated confidence rather than depending on whether greedy decoding happens to produce a parseable label. This is exactly the mechanism that reveals the recall/accuracy trade-off (e.g. SFT v4: argmax 90.4% recall → t=0.05 gives 95.8% recall at some accuracy cost).

**Likely Q — a strong one to be ready for:** *"How do you pick the label token if the label is more than one token?"* → The code takes the **first token** of each label word (`tokenizer(label, add_special_tokens=False)["input_ids"][0]`) — since "EMERGENCY"/"URGENT"/"ROUTINE" all start with a distinct first sub-word for this tokenizer, that's sufficient to disambiguate at that position.

---

## PART 6 — Demos (`src/demo/` + `app_streamlit.py`)

### `triage_chatbot.py`
A Gradio app — loads the final SFT v4 model (base + LoRA adapter, 4-bit), single text box in / single classification out, with color-coded HTML banners (red/orange/green for EMERGENCY/URGENT/ROUTINE) and 5 clickable example cases. Launched with `share=True` so it gets a public `gradio.live` URL for remote demoing.

### `triage_chatbot_multi.py` / `triage_chatbot_enhanced.py`
Extended versions comparing multiple models side-by-side in one UI (SFT v4 vs Pruned+SFT vs, later, a KD-logit-only "finding card" — KD's raw generative text was excluded because of its known vocab-collapse issue, so its card shows only the diagnosed *finding*, not a live generation). These evolved over several iterations to make the demo presentation-safe (nothing that could visibly crash/misbehave live on stage).

### `app_streamlit.py`
A second, independent demo UI built with Streamlit instead of Gradio (two-column layout, sidebar with model info, `@st.cache_resource` so the model loads once and is reused across interactions) — same underlying model (SFT v4), offered as a lightweight alternative deployment path.

**Why two separate demo frameworks (Gradio + Streamlit)?** Different deployment targets — Gradio's `share=True` is convenient for a quick public link during a live session; Streamlit is more suited to a persistently hosted app. Both point at the same underlying adapter, so they're UI alternatives, not different models.

---

## PART 7 — Fast-Reference: Numbers You'll Be Asked

| Metric | Value | Where it comes from |
|---|---|---|
| Teacher params | 8.03B (`OpenBioLLM-Llama3-8B`) | — |
| Student params | 0.62B (`Qwen3-0.6B`) | — |
| Compression | 13.3× fewer params, 10.6× less VRAM | teacher 5.7GB (4-bit) vs student 0.54GB |
| SFT v4 on MIMIC (42, held-out) | 80.2% acc / 90.4% EM recall (argmax) | `evaluate_student_mimic.py` |
| SFT v4 @ threshold t=0.05 | 73.8% acc / 95.8% EM recall | `evaluate_threshold_sweep.py` |
| Pruned+LoRA on MIMIC | 74.4% acc / 80.0% EM recall | same harness |
| KD v1 on MIMIC (argmax) | 42.9% acc / 0% EM recall | collapses to URGENT |
| KD v2 (stabilized) on MIMIC | 57.1% acc / 100% EM recall (argmax) | overcorrects to EMERGENCY |
| LogReg baseline on MIMIC | 88.1% acc / 95.8% EM recall | `logreg_baseline.py` |
| LogReg on Latvia | F1 = 1.0 (perfect) | the benchmark-vulnerability finding |
| Pruning: heads removed | bottom 40% per layer | `headpruning.py` |
| Pruning: layers dropped | 5 of 28 (indices 10,12,14,16,18) | `layerdropping.py` |
| Latency | ~0.91s/sample | `student_latency_per_sample.py` |
| BERTScore (synthetic reasoning vs PubMedQA) | F1 ≈ 0.82, all 3 classes | `evaluate_reasoning_quality.py` |
| Training set (final, v4) | 42,872 samples | `merge_mimic.py` |
| LoRA config | r=16, α=32, dropout=0.05, 7 target modules | `lora_finetune.py` |

---

## PART 8 — The Five Hardest Likely Questions (rehearse these)

1. **"Why did KD fail twice before working?"** → α/T too aggressive + full-parameter updates on a narrow 3-class objective = unstable; a `float16` overflow in the frozen teacher additionally corrupted training; only frozen-base + LoRA-adapter + `bfloat16` fixed it. (See Part 3.)

2. **"Your pruning claims compression but doesn't speed anything up — why?"** → Head pruning is "soft" (weights zeroed, tensor shapes unchanged) — the GPU still computes the zeroed heads. Only layer-dropping is truly structural. This is an explicit, documented limitation, not an oversight.

3. **"Why is MIMIC the 'honest' benchmark and not Latvia, when Latvia is bigger?"** → The LogReg baseline achieves a perfect score on Latvia despite having zero clinical reasoning ability, because fedmml uses standardized phrasing across countries — so Latvia tests memorized vocabulary, not generalization. MIMIC is real hospital free-text and doesn't have this leakage.

4. **"How does the threshold sweep actually work mechanically?"** → Prime the prompt to force the next token to be the label; single forward pass (no generation); read vocabulary logits at that position for the 3 label first-tokens; softmax → probabilities; sweep `t` and predict EMERGENCY whenever `P(EMERGENCY) > t`.

5. **"What's the single biggest limitation of the whole project?"** → Small real-world test set (42 MIMIC samples) — big enough to show large method-to-method gaps, too small for fine-grained statistical significance; plus soft (non-structural) pruning not delivering real speedups yet.

---

*Good luck tomorrow. If you get a question on a specific file not fully covered here, the pattern is almost always the same: build a prompt in the model's chat format → generate/forward-pass → extract the label → compute precision/recall/F1/accuracy per class, with emergency recall as the headline safety metric.*
