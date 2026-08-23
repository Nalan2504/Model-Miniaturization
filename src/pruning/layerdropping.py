# Goal: drop a few middle layers using layer_scores.json (e.g. prune 20% lowest‑importance layers)
# This produces data/pruning/qwen3_pruned_heads_layers
from pathlib import Path
from torch.nn import ModuleList
from transformers import AutoModelForCausalLM, AutoTokenizer

PRUNE_DIR = Path("/root/model_miniaturization/data/pruning")
IN_DIR = PRUNE_DIR / "qwen3_pruned_heads"          # input: head-pruned model
OUT_DIR = PRUNE_DIR / "qwen3_pruned_heads_layers"  # output: head+layer pruned
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("Loading pruned-head model from", IN_DIR)
    model = AutoModelForCausalLM.from_pretrained(
        IN_DIR,
        device_map="auto",
        trust_remote_code=True,
    )

    transformer = model.model
    layers = transformer.layers  # ModuleList
    n_layers = len(layers)
    print(f"Model has {n_layers} layers")

    # Drop a few middle layers by fixed indices (e.g. ~20% of depth)
    # For 28 layers, this might be [10, 12, 14, 16, 18]
    to_prune_idxs = {10, 12, 14, 16, 18}
    to_prune_idxs = {idx for idx in to_prune_idxs if 0 <= idx < n_layers}

    if not to_prune_idxs:
        print("No valid fixed layer indices to prune; leaving model unchanged.")
    else:
        print(f"Pruning layers (by index): {sorted(to_prune_idxs)}")
        kept = []
        for idx, layer in enumerate(layers):
            if idx in to_prune_idxs:
                continue
            kept.append(layer)
        transformer.layers = ModuleList(kept)
        print(f"Remaining layers: {len(transformer.layers)}")

    print(f"Saving head+layer pruned model to {OUT_DIR}...")
    model.save_pretrained(OUT_DIR)
    tokenizer = AutoTokenizer.from_pretrained(IN_DIR, trust_remote_code=True)
    tokenizer.save_pretrained(OUT_DIR)
    print("Done.")


if __name__ == "__main__":
    main()