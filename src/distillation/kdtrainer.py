import json
from pathlib import Path
import math
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

import argparse

DATA_DIR = Path("/root/model_miniaturization/data")
TRAIN_PATH = DATA_DIR / "approach2/combined_train_v4.jsonl"
OUT_DIR = DATA_DIR / "distillation/qwen3_kd_lora"

TEACHER_ID = "aaditya/OpenBioLLM-Llama3-8B"
STUDENT_BASE = "Qwen/Qwen3-0.6B"



LABELS = ["EMERGENCY", "URGENT", "ROUTINE"]
LABEL_TO_ID = {l: i for i, l in enumerate(LABELS)}


class TriageDataset(Dataset):
    def __init__(self, path, limit=None):
        self.rows = []
        with open(path, "r") as f:
            for line in f:
                r = json.loads(line)
                label = r["triage_level"]
                txt = r.get("symptom_description") or r.get("input")
                if txt and label in LABEL_TO_ID:
                    self.rows.append((txt, LABEL_TO_ID[label]))
        if limit:
            self.rows = self.rows[:limit]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx]


def build_prompt(description: str, system_prompt: str):
    return (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{description}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


SYSTEM_PROMPT = """You are a senior emergency physician. Given a patient description, classify the triage level.

Definitions:
EMERGENCY: immediately life-threatening — requires intervention within minutes
URGENT: serious but stable — requires evaluation within 1-2 hours
ROUTINE: non-urgent — can be seen in a scheduled appointment

Respond with ONLY one word: EMERGENCY, URGENT, or ROUTINE."""


def kd_loss(teacher_logits, student_logits, temperature):
    # logits: [batch, num_labels]
    t_probs = F.softmax(teacher_logits / temperature, dim=-1)
    s_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    return F.kl_div(s_log_probs, t_probs, reduction="batchmean") * (temperature * temperature)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--temperature", type=float, default=4.0)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    # Create output dir
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Distillation parameters: epochs={args.epochs}, batch={args.batch_size}, lr={args.lr}, alpha={args.alpha}, temp={args.temperature}, limit={args.limit}")

    # Teacher
    print(f"Loading teacher {TEACHER_ID} in 4-bit NF4...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    teacher_tokenizer = AutoTokenizer.from_pretrained(TEACHER_ID)
    if teacher_tokenizer.pad_token is None:
        teacher_tokenizer.pad_token = teacher_tokenizer.eos_token
    teacher_tokenizer.padding_side = "left"
    
    teacher_model = AutoModelForCausalLM.from_pretrained(
        TEACHER_ID,
        quantization_config=bnb_config,
        device_map="auto",
        attn_implementation="eager",
    )
    teacher_model.eval()

    # Student
    print(f"Loading student base {STUDENT_BASE}...")
    student_tokenizer = AutoTokenizer.from_pretrained(STUDENT_BASE, trust_remote_code=True)
    if student_tokenizer.pad_token is None:
        student_tokenizer.pad_token = student_tokenizer.eos_token
    student_tokenizer.padding_side = "left"

    student_model = AutoModelForCausalLM.from_pretrained(
        STUDENT_BASE,
        device_map="auto",
        trust_remote_code=True,
    )
    student_model.train()
    optimizer = torch.optim.AdamW(student_model.parameters(), lr=args.lr)

    ds = TriageDataset(TRAIN_PATH, limit=args.limit)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True)

    print(f"Dataset size: {len(ds)}")

    for epoch in range(args.epochs):
        total_loss = 0.0
        for step, (texts, labels) in enumerate(dl, start=1):
            # Build prompts
            teacher_prompts = [build_prompt(t, SYSTEM_PROMPT) for t in texts]
            student_prompts = teacher_prompts  # same prompt

            # Teacher forward: get class logits via next-token logits on label tokens
            t_inputs = teacher_tokenizer(
                teacher_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=256,
            ).to(device)

            with torch.inference_mode():
                t_out = teacher_model(**t_inputs)
                # Use logits at last position as proxy: [batch, vocab]
                t_logits_vocab = t_out.logits[:, -1, :]

            # Map vocab logits to triage label logits by selecting the first token ID of the label
            teacher_label_logits = []
            for i, lbl in enumerate(LABELS):
                # encode label word
                toks = teacher_tokenizer(lbl, add_special_tokens=False)["input_ids"]
                teacher_label_logits.append(t_logits_vocab[:, toks[0]])
            # shape [num_labels, batch] -> [batch, num_labels]
            teacher_label_logits = torch.stack(teacher_label_logits, dim=1)

            # Student forward: same prompt
            s_inputs = student_tokenizer(
                student_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=256,
            ).to(device)

            s_out = student_model(**s_inputs)
            s_logits_vocab = s_out.logits[:, -1, :]

            student_label_logits = []
            for i, lbl in enumerate(LABELS):
                toks = student_tokenizer(lbl, add_special_tokens=False)["input_ids"]
                student_label_logits.append(s_logits_vocab[:, toks[0]])
            student_label_logits = torch.stack(student_label_logits, dim=1)

            labels_tensor = labels.to(device, dtype=torch.long)

            loss_kd = kd_loss(teacher_label_logits, student_label_logits, args.temperature)
            loss_ce = F.cross_entropy(student_label_logits, labels_tensor)
            loss = args.alpha * loss_kd + (1.0 - args.alpha) * loss_ce

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            if step % 50 == 0:
                print(f"Epoch {epoch+1}, step {step}, loss={loss.item():.4f}")

        print(f"Epoch {epoch+1} mean loss = {total_loss / max(1, len(dl)):.4f}")

    # Save KD student
    print(f"Saving KD student to {OUT_DIR}...")
    student_model.save_pretrained(OUT_DIR)
    student_tokenizer.save_pretrained(OUT_DIR)
    print("Done.")


if __name__ == "__main__":
    main()