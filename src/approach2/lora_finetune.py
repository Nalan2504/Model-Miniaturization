"""
LoRA Fine-tune Qwen3-0.6B — Approach 2, Step 3

Fine-tunes Qwen/Qwen3-0.6B on the combined dataset (NLI-filtered synthetic
+ syntech-ai real data) using LoRA (r=16, alpha=32).

Input : data/approach2/combined_train.jsonl
Eval  : data/processed/val.jsonl  (held-out synthetic val set)
Output: data/approach2/qwen3_lora/  (adapter weights)

Usage (container):
  python src/approach2/lora_finetune.py --epochs 3 --batch_size 4 --grad_accum 8
"""

import json
import torch
import argparse
from pathlib import Path
from collections import Counter
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, TaskType

DATA_DIR   = Path("/root/model_miniaturization/data")
OUTPUT_DIR = DATA_DIR / "approach2/qwen3_lora"

MODEL_ID   = "Qwen/Qwen3-0.6B"

LABEL2ID = {"EMERGENCY": 0, "URGENT": 1, "ROUTINE": 2}

MAX_INPUT_LEN  = 256
MAX_OUTPUT_LEN = 256
MAX_TOTAL_LEN  = MAX_INPUT_LEN + MAX_OUTPUT_LEN


def format_prompt(system: str, user_input: str) -> str:
    return f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user_input}<|im_end|>\n<|im_start|>assistant\n"


def tokenize_sample(sample, tokenizer):
    prompt = format_prompt(sample["instruction"], sample["input"])
    full   = prompt + sample["output"] + tokenizer.eos_token

    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    full_ids   = tokenizer(full,   add_special_tokens=False, max_length=MAX_TOTAL_LEN, truncation=True)["input_ids"]

    # Labels: -100 for prompt tokens (don't compute loss on them)
    labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]

    # Pad labels to match input_ids length if truncation happened
    labels = labels[:len(full_ids)]

    return {
        "input_ids":      full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels":         labels,
    }


def load_combined(path: Path, tokenizer):
    raw = [json.loads(l) for l in open(path)]
    print(f"  {len(raw)} combined samples")
    by_class = Counter(s["triage_level"] for s in raw)
    for label in ["EMERGENCY", "URGENT", "ROUTINE"]:
        print(f"    {label}: {by_class[label]}")

    ds = Dataset.from_list(raw)
    ds = ds.map(lambda s: tokenize_sample(s, tokenizer), remove_columns=ds.column_names)
    return ds


def load_val(path: Path, tokenizer):
    """Val set is synthetic train_samples format — used for monitoring only."""
    raw = []
    for line in open(path):
        s = json.loads(line)
        raw.append({
            "instruction": "You are a senior emergency physician. Classify the triage level.",
            "input":       s["symptom_description"],
            "output":      s["raw_output"].strip(),
        })
    print(f"  {len(raw)} val samples")
    ds = Dataset.from_list(raw)
    ds = ds.map(lambda s: tokenize_sample(s, tokenizer), remove_columns=ds.column_names)
    return ds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",      type=int,   default=3)
    parser.add_argument("--batch_size",  type=int,   default=4)
    parser.add_argument("--grad_accum",  type=int,   default=8)
    parser.add_argument("--lr",          type=float, default=2e-4)
    parser.add_argument("--lora_r",      type=int,   default=16)
    parser.add_argument("--lora_alpha",  type=int,   default=32)
    parser.add_argument("--lora_dropout",type=float, default=0.05)
    parser.add_argument("--train",       default=str(DATA_DIR / "approach2/combined_train.jsonl"))
    parser.add_argument("--val",         default=str(DATA_DIR / "processed/val.jsonl"))
    parser.add_argument("--output_dir",  default=str(OUTPUT_DIR))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Model  : {MODEL_ID}")
    print(f"Device : {device}")
    print(f"LoRA   : r={args.lora_r}, alpha={args.lora_alpha}, dropout={args.lora_dropout}")
    print(f"Train  : {args.epochs} epochs, batch={args.batch_size}, grad_accum={args.grad_accum}, lr={args.lr}")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load datasets
    print("\nLoading training data...")
    train_ds = load_combined(Path(args.train), tokenizer)
    print("Loading val data...")
    val_ds = load_val(Path(args.val), tokenizer)

    # Model — 4-bit for VRAM efficiency
    print(f"\nLoading {MODEL_ID} in 4-bit NF4...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False
    print(f"  Loaded. VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    # LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Training args
    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_steps=50,
        lr_scheduler_type="cosine",
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        logging_steps=20,
        report_to="none",
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=0,
        remove_unused_columns=False,
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        pad_to_multiple_of=8,
        label_pad_token_id=-100,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        data_collator=data_collator,
    )

    print(f"\nFine-tuning {MODEL_ID} with LoRA...")
    trainer.train()

    # Save adapter
    adapter_path = output_dir / "adapter"
    model.save_pretrained(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))
    print(f"\nLoRA adapter saved → {adapter_path}")

    print("\nDone. Next step: run evaluate_student.py on test.jsonl")


if __name__ == "__main__":
    main()
