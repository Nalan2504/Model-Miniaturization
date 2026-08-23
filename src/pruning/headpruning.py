# Goal: prune the lowest‑importance 40% heads per layer based on head_scores.json
# This writes a pruned‑heads copy of Qwen3‑0.6B to data/pruning/qwen3_pruned_heads
import json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL = "Qwen/Qwen3-0.6B"
DATA_DIR = Path("/root/model_miniaturization/data")
PRUNE_DIR = DATA_DIR / "pruning"
OUT_DIR = PRUNE_DIR / "qwen3_pruned_heads"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_head_scores():
    path = PRUNE_DIR / "head_scores.json"
    with open(path, "r") as f:
        return json.load(f)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    scores = load_head_scores()

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        device_map="auto",
        trust_remote_code=True,
    )

    # Assume standard HF layout: model.model.layers[layer_idx].self_attn.num_heads etc.
    transformer = model.model
    n_layers = len(transformer.layers)
    print(f"Layers: {n_layers}")

    for layer_idx, layer in enumerate(transformer.layers):
        # Collect scores for this layer
        per_layer = []
        for key, val in scores.items():
            l_idx, h_idx = key.split(":")
            if int(l_idx) == layer_idx:
                per_layer.append((int(h_idx), val))
        if not per_layer:
            continue

        per_layer.sort(key=lambda x: x[1])  # ascending by score
        n_heads = len(per_layer)
        k_prune = int(0.4 * n_heads)  # prune bottom 40%
        heads_to_prune = [h for (h, _) in per_layer[:k_prune]]

        print(f"Layer {layer_idx}: pruning heads {heads_to_prune}")

        # Simple head-drop: set projection weights for pruned heads to zero.
        # This is a crude but easy approach.
        attn = layer.self_attn
        q_proj = attn.q_proj.weight.data
        k_proj = attn.k_proj.weight.data
        v_proj = attn.v_proj.weight.data

        head_dim = attn.head_dim
        for h in heads_to_prune:
            start = h * head_dim
            end = (h + 1) * head_dim
            q_proj[:, start:end] = 0
            k_proj[:, start:end] = 0
            v_proj[:, start:end] = 0

    # Save pruned model
    print(f"Saving pruned-head model to {OUT_DIR}...")
    model.save_pretrained(OUT_DIR)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    tokenizer.save_pretrained(OUT_DIR)
    print("Done.")


if __name__ == "__main__":
    main()