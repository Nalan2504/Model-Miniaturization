# Goal: compute simple per‑head and per‑layer importance scores for 
# Qwen3‑0.6B on a small batch from data/processed/train.jsonl
# This will create data/pruning/head_scores.json and data/pruning/layer_scores.json
import json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL = "Qwen/Qwen3-0.6B"
DATA_DIR = Path("/root/model_miniaturization/data")
TRAIN_PATH = DATA_DIR / "processed/train.jsonl"
OUT_DIR = DATA_DIR / "pruning"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_batch(n_samples=128):
    samples = []
    with open(TRAIN_PATH, "r") as f:
        for i, line in enumerate(f):
            if i >= n_samples:
                break
            row = json.loads(line)
            txt = row.get("symptom_description") or row.get("input")
            if txt:
                samples.append(txt)
    return samples


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager",
    )
    model.eval()

    batch = load_batch()
    enc = tokenizer(
        batch,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=256,
    ).to(device)

    # Simple proxy: mean squared activation per attention head / layer
    head_scores = {}
    layer_scores = {}

    with torch.inference_mode():
        outputs = model(**enc, output_attentions=True)
        attentions = outputs.attentions  # list of [batch, heads, seq, seq]

    for layer_idx, att in enumerate(attentions):
        # att: [batch_size, num_heads, seq_len, seq_len]
        # importance proxy = mean of att^2 over batch and seq dims
        # shape after mean: [num_heads]
        scores = (att ** 2).mean(dim=(0, 2, 3)).detach().cpu().tolist()
        layer_scores[layer_idx] = float(sum(scores) / len(scores))
        for head_idx, s in enumerate(scores):
            head_scores[f"{layer_idx}:{head_idx}"] = float(s)

    head_path = OUT_DIR / "head_scores.json"
    layer_path = OUT_DIR / "layer_scores.json"

    with open(head_path, "w") as f:
        json.dump(head_scores, f, indent=2)
    with open(layer_path, "w") as f:
        json.dump(layer_scores, f, indent=2)

    print(f"Saved head scores to {head_path}")
    print(f"Saved layer scores to {layer_path}")


if __name__ == "__main__":
    main()