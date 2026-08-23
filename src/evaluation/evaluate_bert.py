import json
import torch
import numpy as np
import argparse
from pathlib import Path
from collections import Counter
from datasets import Dataset, load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from sklearn.metrics import classification_report, confusion_matrix

MODEL_ID   = "dmis-lab/biobert-base-cased-v1.2"
DATA_DIR   = Path("/root/model_miniaturization/data")
OUTPUT_DIR = Path("/root/model_miniaturization/data/evaluation/bert")
MAX_LEN    = 512  # BioBERT limit

LABEL2ID = {"EMERGENCY": 0, "URGENT": 1, "ROUTINE": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

SYNTECH_MAP = {"immediate": "EMERGENCY", "urgent": "URGENT", "routine": "ROUTINE"}


# ── Data loaders ──────────────────────────────────────────────────────────────

def load_synthetic(path: Path):
    rows = [json.loads(l) for l in open(path)]
    samples = []
    for r in rows:
        # Combine symptom description + OpenBioLLM's clinical reasoning
        text = r["symptom_description"] + " [SEP] " + r["raw_output"]
        samples.append({"text": text, "label": LABEL2ID[r["triage_level"]]})
    return samples


def load_syntech():
    ds = load_dataset(
        "syntech-ai/medical-triage-500",
        data_files="medical_triage_500.jsonl",
        split="train",
    )
    out = []
    for s in ds:
        pres  = s["presentation"]
        risk  = s["risk_assessment"]
        p     = s["patient"]
        flags = ", ".join(risk["red_flags"]) if risk["red_flags"] else "none"
        text  = (
            f"A {p['age']}-year-old {p['gender']} presenting with "
            f"{', '.join(pres['symptoms'])}. "
            f"Duration: {pres['duration']}. Onset: {pres['onset']}. "
            f"Context: {pres['context']}. Red flags: {flags}."
        )
        label = LABEL2ID[SYNTECH_MAP[s["triage_classification"]["urgency_category"]]]
        out.append({"text": text, "label": label, "case_id": s["case_id"]})
    return out


# ── Tokenisation ──────────────────────────────────────────────────────────────

def tokenise(samples, tokenizer, max_len=MAX_LEN):
    hf_ds = Dataset.from_list(samples)
    def _tok(batch):
        return tokenizer(batch["text"], truncation=True, max_length=max_len)
    return hf_ds.map(_tok, batched=True, remove_columns=["text"])


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    correct = (preds == labels).sum()
    acc = correct / len(labels)
    # Per-class F1
    f1s = []
    for c in range(3):
        tp = ((preds == c) & (labels == c)).sum()
        fp = ((preds == c) & (labels != c)).sum()
        fn = ((preds != c) & (labels == c)).sum()
        p  = tp / (tp + fp) if (tp + fp) > 0 else 0
        r  = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1s.append(2 * p * r / (p + r) if (p + r) > 0 else 0)
    return {
        "accuracy":       acc,
        "macro_f1":       np.mean(f1s),
        "emergency_f1":   f1s[0],
        "urgent_f1":      f1s[1],
        "routine_f1":     f1s[2],
    }


def print_report(preds, labels):
    names = ["EMERGENCY", "URGENT", "ROUTINE"]
    print("\n" + "=" * 55)
    print("BERT EVALUATION — syntech-ai/medical-triage-500")
    print("=" * 55)
    print(classification_report(labels, preds, target_names=names, digits=3))
    print("Confusion matrix (rows=true, cols=pred):")
    cm = confusion_matrix(labels, preds)
    print(f"{'':12}", "  ".join(f"{n:>9}" for n in names))
    for i, row in enumerate(cm):
        print(f"{names[i]:12}", "  ".join(f"{v:>9}" for v in row))
    em_recall = cm[0, 0] / cm[0].sum() if cm[0].sum() > 0 else 0
    print(f"\n*** Emergency recall: {em_recall*100:.1f}% (target: >95%) ***")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train",  default=str(DATA_DIR / "synthetic/train_samples.jsonl"))
    parser.add_argument("--epochs", type=int,   default=3)
    parser.add_argument("--batch",  type=int,   default=32)
    parser.add_argument("--lr",     type=float, default=2e-5)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Model : {MODEL_ID}")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"Input : symptom_description + OpenBioLLM raw_output (max {MAX_LEN} tokens)")

    # Use ALL 5100 samples (symptom + OpenBioLLM reasoning) for training
    print("\nLoading all 5100 OpenBioLLM-generated samples...")
    train_data = load_synthetic(Path(args.train))
    label_dist = Counter(ID2LABEL[s["label"]] for s in train_data)
    print(f"  {len(train_data)} samples | {dict(label_dist)}")

    print("Loading syntech-ai/medical-triage-500 (eval set)...")
    syntech_data = load_syntech()
    label_dist = Counter(ID2LABEL[s["label"]] for s in syntech_data)
    print(f"  {len(syntech_data)} samples | {dict(label_dist)}")

    # Tokenise
    print(f"\nTokenising with {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    train_ds   = tokenise(train_data, tokenizer)
    syntech_ds = tokenise(syntech_data, tokenizer)
    print("  Done.")

    # Model
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID,
        num_labels=3,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    # Training
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=64,
        learning_rate=args.lr,
        warmup_steps=50,
        eval_strategy="no",
        save_strategy="epoch",
        logging_steps=50,
        report_to="none",
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
    )

    print(f"\nFine-tuning {MODEL_ID} for {args.epochs} epochs...")
    trainer.train()

    # Evaluate on syntech-ai
    print("\nEvaluating on syntech-ai/medical-triage-500...")
    raw = trainer.predict(syntech_ds)
    preds  = np.argmax(raw.predictions, axis=-1)
    labels = np.array(syntech_ds["label"])

    print_report(preds, labels)

    # Save per-sample results
    results = []
    for i, s in enumerate(syntech_data):
        results.append({
            "case_id":  s["case_id"],
            "text":     s["text"],
            "true":     ID2LABEL[s["label"]],
            "pred":     ID2LABEL[preds[i]],
            "correct":  bool(preds[i] == s["label"]),
        })

    out_path = OUTPUT_DIR / "bert_syntech_eval.jsonl"
    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"\nPer-sample results → {out_path}")

    # Save summary
    report = classification_report(
        labels, preds,
        target_names=["EMERGENCY", "URGENT", "ROUTINE"],
        output_dict=True,
    )
    summary = {
        "model": MODEL_ID,
        "trained_on": "synthetic train.jsonl (4080 samples)",
        "evaluated_on": "syntech-ai/medical-triage-500 (500 samples)",
        "epochs": args.epochs,
        **report,
    }
    sum_path = OUTPUT_DIR / "bert_syntech_summary.json"
    with open(sum_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary → {sum_path}")


if __name__ == "__main__":
    main()
