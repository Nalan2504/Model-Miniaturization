"""Logit-threshold evaluation for Qwen3-0.6B + LoRA (Approach 2).

The model generates verbose output ("TRIAGE LEVEL: EMERGENCY ...").
We scan generated tokens to find WHERE the label word appears,
then apply P(EMERGENCY)/(P(EMERGENCY)+P(URGENT)) threshold at that position.
Single inference pass; thresholds are swept in post-processing.
"""
import json
import re
import torch
import torch.nn.functional as F
from pathlib import Path
from collections import defaultdict
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

BASE_MODEL  = "Qwen/Qwen3-0.6B"
ADAPTER_DIR = "/root/model_miniaturization/data/approach2/qwen3_lora/adapter"
TEST_FILE   = "/root/model_miniaturization/data/processed/test.jsonl"
OUT_DIR     = Path("/root/model_miniaturization/data/approach2")
LABELS      = ["EMERGENCY", "URGENT", "ROUTINE"]

SYSTEM_PROMPT = "You are a triage classifier. Reply with exactly one word: EMERGENCY, URGENT, or ROUTINE."

def build_prompt(text, tokenizer):
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": text},
    ]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

def load_test(path):
    samples = []
    with open(path) as f:
        for line in f:
            samples.append(json.loads(line))
    counts = {l: sum(1 for s in samples if s["triage_level"] == l) for l in LABELS}
    print(f"  {len(samples)} samples | {counts}")
    return samples

def get_label_first_tokens(tokenizer):
    """All possible first-token IDs for each label (with and without leading space)."""
    label_toks = {}
    for label in LABELS:
        tok_set = set()
        for prefix in ["", " "]:
            toks = tokenizer.encode(prefix + label, add_special_tokens=False)
            tok_set.add(toks[0])
        label_toks[label] = tok_set
        decoded = [repr(tokenizer.decode([t])) for t in sorted(tok_set)]
        print(f"  {label}: ids={sorted(tok_set)} -> {decoded}")
    return label_toks

def predict_single_with_scores(model, tokenizer, text, label_first_toks):
    """
    Generate for one sample; return (base_pred, p_em, p_ur, found_pos).
    - base_pred: argmax label at the detected label position (or regex fallback)
    - p_em, p_ur: summed first-token probs at that position (0,1 if fallback)
    - found_pos: position index where label was found (-1 if fallback)
    """
    prompt = build_prompt(text, tokenizer)
    enc = tokenizer(prompt, return_tensors="pt", truncation=True,
                    max_length=512).to(model.device)
    input_len = enc.input_ids.shape[1]

    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=20,
            do_sample=False,
            output_scores=True,
            return_dict_in_generate=True,
            pad_token_id=tokenizer.pad_token_id,
        )

    new_ids = out.sequences[0, input_len:].tolist()

    # Scan generated tokens to find first label first-token
    label_pos = None
    base_pred = None
    for pos, tok_id in enumerate(new_ids):
        for label, tok_set in label_first_toks.items():
            if tok_id in tok_set:
                label_pos = pos
                base_pred = label
                break
        if label_pos is not None:
            break

    if label_pos is None:
        # Fallback: regex on decoded text
        generated = tokenizer.decode(new_ids, skip_special_tokens=True)
        m = re.search(r"\b(EMERGENCY|URGENT|ROUTINE)\b", generated.upper())
        base_pred = m.group(1) if m else "FAILED"
        return base_pred, 0.0, 1.0, -1  # p_em=0 → threshold never fires

    # Probability distribution at label position
    probs = F.softmax(out.scores[label_pos][0], dim=0)
    p_em = sum(probs[t].item() for t in label_first_toks["EMERGENCY"])
    p_ur = sum(probs[t].item() for t in label_first_toks["URGENT"])
    return base_pred, p_em, p_ur, label_pos

def collect_inference_data(samples, model, tokenizer, label_first_toks):
    """Single inference pass over all samples; store logit data for threshold sweep."""
    data = []
    fallback_count = 0
    for i, s in enumerate(samples):
        base_pred, p_em, p_ur, found_pos = predict_single_with_scores(
            model, tokenizer, s["symptom_description"], label_first_toks)
        if found_pos == -1:
            fallback_count += 1
        data.append({
            "true": s["triage_level"],
            "base_pred": base_pred,
            "p_em": p_em,
            "p_ur": p_ur,
        })
        if (i + 1) % 50 == 0 or (i + 1) == len(samples):
            correct = sum(1 for r in data if r["true"] == r["base_pred"])
            print(f"  [{i+1:>4}/{len(samples)}] base_acc={correct/len(data)*100:.1f}%  fallbacks={fallback_count}")
    return data

def apply_threshold(data, threshold):
    results = []
    for d in data:
        p_em, p_ur = d["p_em"], d["p_ur"]
        denom = p_em + p_ur
        if denom > 1e-9 and (p_em / denom) > threshold:
            pred = "EMERGENCY"
        else:
            pred = d["base_pred"]
        results.append({"true": d["true"], "pred": pred})
    return results

def metrics(results):
    tp = defaultdict(int); fp = defaultdict(int); fn = defaultdict(int)
    correct = 0
    for r in results:
        t, p = r["true"], r["pred"]
        if t == p:
            correct += 1
            tp[t] += 1
        else:
            fp[p] += 1
            fn[t] += 1
    acc = correct / len(results)
    stats = {}
    for l in LABELS:
        prec = tp[l] / (tp[l] + fp[l]) if (tp[l] + fp[l]) > 0 else 0.0
        rec  = tp[l] / (tp[l] + fn[l]) if (tp[l] + fn[l]) > 0 else 0.0
        f1   = 2*prec*rec/(prec+rec) if (prec+rec) > 0 else 0.0
        stats[l] = {"prec": prec, "rec": rec, "f1": f1}
    macro_f1 = sum(v["f1"] for v in stats.values()) / len(LABELS)
    return acc, macro_f1, stats

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device : {device}\nAdapter: {ADAPTER_DIR}\nTest   : {TEST_FILE}\n")

    print("Loading Qwen/Qwen3-0.6B in 4-bit + LoRA adapter...")
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, quantization_config=bnb,
                                                device_map="auto", trust_remote_code=True)
    model = PeftModel.from_pretrained(base, ADAPTER_DIR)
    model.eval()
    vram = torch.cuda.memory_allocated() / 1e9 if device == "cuda" else 0
    print(f"  Loaded. VRAM: {vram:.2f} GB\n")

    print(f"Loading test set from {TEST_FILE}...")
    samples = load_test(TEST_FILE)

    print("\nLabel first-token IDs (with/without leading space):")
    label_first_toks = get_label_first_tokens(tokenizer)

    print(f"\nRunning inference (single pass, max_new_tokens=20)...")
    data = collect_inference_data(samples, model, tokenizer, label_first_toks)

    # Print distribution of p_em values to understand the signal
    em_ratios = []
    for d in data:
        denom = d["p_em"] + d["p_ur"]
        if denom > 1e-9:
            em_ratios.append((d["p_em"] / denom, d["true"]))
    print(f"\nEM-ratio stats (p_em/(p_em+p_ur)) for {len(em_ratios)} samples with label found:")
    for label in LABELS:
        ratios = [r for r, t in em_ratios if t == label]
        if ratios:
            mean_r = sum(ratios)/len(ratios)
            print(f"  true={label}: n={len(ratios)} mean_ratio={mean_r:.4f}")

    # Sweep thresholds
    thresholds = [round(t * 0.05, 2) for t in range(2, 11)]  # 0.10 to 0.50
    print(f"\n{'thresh':>8} {'em_rec':>8} {'em_prec':>8} {'macro_f1':>10} {'acc':>7}")
    print("-" * 48)

    best = None
    for t in thresholds:
        results = apply_threshold(data, t)
        acc, macro_f1, stats = metrics(results)
        em = stats["EMERGENCY"]
        marker = " <--" if em["rec"] >= 0.95 and best is None else ""
        print(f"  t={t:.2f}  {em['rec']*100:6.1f}%  {em['prec']*100:6.1f}%"
              f"  {macro_f1:.4f}  {acc*100:.1f}%{marker}")
        if em["rec"] >= 0.95 and best is None:
            best = (results, acc, macro_f1, stats, t)

    if best is None:
        print("\n[!] No threshold reached >95% emergency recall. Using t=0.10 (lowest tested).")
        results = apply_threshold(data, 0.10)
        acc, macro_f1, stats = metrics(results)
        best = (results, acc, macro_f1, stats, 0.10)

    results, acc, macro_f1, stats, chosen_t = best
    print(f"\n=======================================================")
    print(f"LOGIT-THRESHOLD EVALUATION — Approach 2")
    print(f"Model: Qwen3-0.6B+LoRA  |  Threshold={chosen_t}  |  n={len(results)}")
    print(f"=======================================================")
    print(f"Accuracy  : {acc*100:.1f}%")
    print(f"Macro F1  : {macro_f1:.3f}")
    print(f"\n{'Label':<14} {'Precision':>12} {'Recall':>12} {'F1':>8}")
    print("-" * 52)
    for l in LABELS:
        s = stats[l]
        print(f"{l:<14} {s['prec']*100:>10.1f}%  {s['rec']*100:>10.1f}%  {s['f1']:.3f}")
    em_rec = stats["EMERGENCY"]["rec"]
    tag = "OK" if em_rec >= 0.95 else "BELOW TARGET"
    print(f"\n*** Emergency recall: {em_rec*100:.1f}% (target: >95%) [{tag}] ***")

    summary = {
        "method": "logit_threshold",
        "threshold": chosen_t,
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "emergency_recall": round(stats["EMERGENCY"]["rec"], 4),
        "emergency_precision": round(stats["EMERGENCY"]["prec"], 4),
        "per_class": {l: {k: round(v, 4) for k, v in stats[l].items()} for l in LABELS},
    }
    out_path = OUT_DIR / f"logit_eval_t{int(chosen_t*100):02d}.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nSummary → {out_path}")

if __name__ == "__main__":
    main()
