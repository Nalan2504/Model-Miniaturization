"""
KD Trainer v2 — fixes for vocab collapse and low EMERGENCY recall.

Changes vs kdtrainer.py:
  1. alpha 0.7 → 0.5    (equal weight to KD and CE; v1 was 70% KD = too aggressive)
  2. temperature 4.0 → 2.0  (less blurring; v1's T=4 washed out EMERGENCY boundary)
  3. class_weights EMERGENCY=3.0  (penalise missing emergencies 3× harder)
  4. sequence-level CE on generated tokens (--seq-ce flag)
     In addition to 3-class logit CE, optionally train student to generate
     the full "TRIAGE LEVEL: EMERGENCY\n..." response text. This prevents
     the vocab collapse where the model learns class scores but forgets to write.
"""

import json
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, TaskType

DATA_DIR   = Path("/root/model_miniaturization/data")
TRAIN_PATH = DATA_DIR / "approach2/combined_train_v4.jsonl"
OUT_DIR    = DATA_DIR / "distillation/qwen3_pruned_kd_lora_v2"

TEACHER_ID   = "aaditya/OpenBioLLM-Llama3-8B"
STUDENT_BASE = str(DATA_DIR / "pruning/qwen3_pruned_heads_layers")

LABELS      = ["EMERGENCY", "URGENT", "ROUTINE"]
LABEL_TO_ID = {l: i for i, l in enumerate(LABELS)}

# Class weights: penalise missing EMERGENCY 3× harder
CLASS_WEIGHTS = torch.tensor([3.0, 1.0, 1.0])

SYSTEM_PROMPT = """You are a senior emergency physician. Given a patient description, classify the triage level.

Definitions:
EMERGENCY: immediately life-threatening — requires intervention within minutes
URGENT: serious but stable — requires evaluation within 1-2 hours
ROUTINE: non-urgent — can be seen in a scheduled appointment

Respond with ONLY one word: EMERGENCY, URGENT, or ROUTINE."""


class TriageDataset(Dataset):
    def __init__(self, path, limit=None):
        self.rows = []
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                label = r.get("triage_level")
                txt   = r.get("symptom_description") or r.get("input")
                out   = r.get("output") or r.get("raw_output", "")
                if txt and label in LABEL_TO_ID:
                    self.rows.append((txt, LABEL_TO_ID[label], out))
        if limit:
            self.rows = self.rows[:limit]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx]


def build_prompt(description: str) -> str:
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{description}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def build_teacher_prompt(description: str) -> str:
    return (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
        f"{SYSTEM_PROMPT}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n"
        f"{description}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n"
    )


def kd_loss(teacher_logits, student_logits, temperature):
    t_probs      = F.softmax(teacher_logits / temperature, dim=-1)
    s_log_probs  = F.log_softmax(student_logits / temperature, dim=-1)
    return F.kl_div(s_log_probs, t_probs, reduction="batchmean") * (temperature ** 2)


def get_label_logits(logits_vocab, tokenizer, labels):
    """Extract per-label scores from last-position vocab logits."""
    label_logits = []
    for lbl in labels:
        toks = tokenizer(lbl, add_special_tokens=False)["input_ids"]
        label_logits.append(logits_vocab[:, toks[0]])
    return torch.stack(label_logits, dim=1)  # [batch, num_labels]


def bnb_config():
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",      type=int,   default=1)
    parser.add_argument("--batch_size",  type=int,   default=2)
    parser.add_argument("--patience",    type=int,   default=1)
    parser.add_argument("--val-path",    default=str(DATA_DIR / "processed/val.jsonl"))
    parser.add_argument("--lr",          type=float, default=1e-4)
    parser.add_argument("--alpha",       type=float, default=0.5,
                        help="KD weight; 1-alpha goes to CE (v1 default was 0.7)")
    parser.add_argument("--temperature", type=float, default=2.0,
                        help="Softmax temperature for KD (v1 default was 4.0)")
    parser.add_argument("--em-weight",   type=float, default=3.0,
                        help="Class weight for EMERGENCY in CE loss")
    parser.add_argument("--seq-ce",      action="store_true",
                        help="Also compute CE on generated token sequence (prevents vocab collapse)")
    parser.add_argument("--limit",       type=int,   default=None)
    parser.add_argument("--output",      default=str(OUT_DIR))
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"KD Trainer v2")
    print(f"  alpha={args.alpha} (KD), {1-args.alpha:.1f} (CE) | T={args.temperature} | EM-weight={args.em_weight}")
    print(f"  seq-ce={'ON' if args.seq_ce else 'OFF'} | epochs={args.epochs} | lr={args.lr}")

    class_weights = torch.tensor([args.em_weight, 1.0, 1.0]).to(device)

    # ── Teacher ──────────────────────────────────────────────────────────────
    print(f"\nLoading teacher {TEACHER_ID} in 4-bit NF4...")
    t_tok = AutoTokenizer.from_pretrained(TEACHER_ID)
    if t_tok.pad_token is None:
        t_tok.pad_token = t_tok.eos_token
    t_tok.padding_side = "left"
    t_model = AutoModelForCausalLM.from_pretrained(
        TEACHER_ID, quantization_config=bnb_config(), device_map="auto", attn_implementation="eager"
    )
    t_model.eval()
    print(f"Teacher loaded — VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB")

    # ── Student ───────────────────────────────────────────────────────────────
    print(f"\nLoading student {STUDENT_BASE}...")
    s_tok = AutoTokenizer.from_pretrained(STUDENT_BASE, trust_remote_code=True)
    if s_tok.pad_token is None:
        s_tok.pad_token = s_tok.eos_token
    s_tok.padding_side = "left"
    s_model_base = AutoModelForCausalLM.from_pretrained(
        STUDENT_BASE, quantization_config=bnb_config(), device_map="auto", trust_remote_code=True
    )
    
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    s_model = get_peft_model(s_model_base, lora_config)
    s_model.train()
    optimizer = torch.optim.AdamW(s_model.parameters(), lr=args.lr)

    ds = TriageDataset(TRAIN_PATH, limit=args.limit)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True)
    val_ds = TriageDataset(args.val_path)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    print(f"Dataset: {len(ds)} train | {len(val_ds)} val | batches: {len(dl)}")

    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None

    for epoch in range(args.epochs):
        total_loss = total_kd = total_ce = 0.0

        for step, (texts, label_ids, outputs) in enumerate(dl, 1):
            # ── Teacher logits (frozen) ───────────────────────────────────────
            t_prompts = [build_teacher_prompt(t) for t in texts]
            t_inp = t_tok(t_prompts, return_tensors="pt", padding=True,
                          truncation=True, max_length=256).to(device)
            with torch.inference_mode():
                t_out = t_model(**t_inp)
                teacher_label_logits = get_label_logits(t_out.logits[:, -1, :], t_tok, LABELS).float()

            # ── Student forward ───────────────────────────────────────────────
            s_prompts = [build_prompt(t) for t in texts]
            s_inp = s_tok(s_prompts, return_tensors="pt", padding=True,
                          truncation=True, max_length=256).to(device)
            s_out = s_model(**s_inp)
            s_logits_vocab = s_out.logits[:, -1, :]

            student_label_logits = []
            for lbl in LABELS:
                toks = s_tok(lbl, add_special_tokens=False)["input_ids"]
                student_label_logits.append(s_logits_vocab[:, toks[0]])
            student_label_logits = torch.stack(student_label_logits, dim=1).float()

            label_ids_t = label_ids.to(device, dtype=torch.long)

            # ── Losses ────────────────────────────────────────────────────────
            loss_kd = kd_loss(teacher_label_logits, student_label_logits, args.temperature)
            loss_ce = F.cross_entropy(student_label_logits, label_ids_t, weight=class_weights)
            loss    = args.alpha * loss_kd + (1 - args.alpha) * loss_ce

            # Optional: sequence-level CE on response text (prevents vocab collapse)
            if args.seq_ce:
                # Build full prompt+response and let student predict response tokens
                full_texts = [build_prompt(t) + o for t, o in zip(texts, outputs)]
                full_inp   = s_tok(full_texts, return_tensors="pt", padding=True,
                                   truncation=True, max_length=512).to(device)
                prompt_inp = s_tok(s_prompts, return_tensors="pt", padding=True,
                                   truncation=True, max_length=256)
                prompt_len = prompt_inp["input_ids"].shape[1]

                seq_out    = s_model(**full_inp)
                # Only compute CE on the response portion (after prompt tokens)
                seq_logits = seq_out.logits[:, prompt_len - 1:-1, :]
                seq_labels = full_inp["input_ids"][:, prompt_len:]
                # Mask padding tokens
                seq_labels = seq_labels.clone()
                seq_labels[seq_labels == s_tok.pad_token_id] = -100
                loss_seq   = F.cross_entropy(
                    seq_logits.reshape(-1, seq_logits.size(-1)),
                    seq_labels.reshape(-1),
                    ignore_index=-100,
                )
                loss = loss + 0.3 * loss_seq

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(s_model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            total_kd   += loss_kd.item()
            total_ce   += loss_ce.item()

            if step % 100 == 0 or step <= 3:
                print(f"  Epoch {epoch+1} step {step}/{len(dl)} | "
                      f"loss={loss.item():.4f} kd={loss_kd.item():.4f} ce={loss_ce.item():.4f}")

        n = max(1, len(dl))
        print(f"Epoch {epoch+1} done | mean loss={total_loss/n:.4f} "
              f"kd={total_kd/n:.4f} ce={total_ce/n:.4f}")

        # Validation
        s_model.eval()
        val_loss = 0.0
        with torch.inference_mode():
            for v_texts, v_label_ids, _ in val_dl:
                v_prompts = [build_prompt(t) for t in v_texts]
                v_inp = s_tok(v_prompts, return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)
                v_out = s_model(**v_inp)
                v_label_logits = get_label_logits(v_out.logits[:, -1, :], s_tok, LABELS)
                v_label_ids_t = v_label_ids.to(device, dtype=torch.long)
                v_loss = F.cross_entropy(v_label_logits.float(), v_label_ids_t)
                val_loss += v_loss.item()
        
        val_loss /= max(1, len(val_dl))
        print(f"Epoch {epoch+1} Val Loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            # Save state
            s_model.save_pretrained(out_dir)
            s_tok.save_pretrained(out_dir)
            print(f"  New best model saved to {out_dir}")
        else:
            epochs_no_improve += 1
            print(f"  No improvement. Patience: {epochs_no_improve}/{args.patience}")
            if epochs_no_improve >= args.patience:
                print("Early stopping triggered!")
                break
                
        s_model.train()

    print("\nDone.")


if __name__ == "__main__":
    main()
