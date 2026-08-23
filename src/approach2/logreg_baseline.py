"""
Logistic regression baseline on medical triage data.

- Train on: data/approach2/combined_train_v4.jsonl
- Test on:  data/approach2/fedmml_test.jsonl  (Latvia held-out set)
- Output:   data/approach2/logreg_predictions_latvia.csv

Metrics printed:
- Accuracy
- Macro F1
- Per-class precision/recall/F1
- Emergency recall
"""

import json
from pathlib import Path

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    f1_score,
    confusion_matrix,
)


# Paths (adjust if needed)
TRAIN_PATH = Path("data/approach2/combined_train_v4.jsonl")
# on latvia dataset
# TEST_PATH = Path("data/approach2/fedmml_test.jsonl")
# OUT_CSV = Path("data/approach2/logreg_predictions_latvia.csv")
# on mimic dataset
TEST_PATH = Path("data/approach2/mimic_test.jsonl")
OUT_CSV = Path("data/approach2/logreg_predictions_mimic.csv")


def load_jsonl(path: Path) -> pd.DataFrame:
    rows = []
    with open(path, "r") as f:
        for line in f:
            rows.append(json.loads(line))
    return pd.DataFrame(rows)


def main():
    print(f"Loading train from {TRAIN_PATH}...")
    train_df = load_jsonl(TRAIN_PATH)
    print(f"  train: {len(train_df)} samples")

    print(f"Loading test from {TEST_PATH} (Latvia)...")
    test_df = load_jsonl(TEST_PATH)
    print(f"  test: {len(test_df)} samples")

    # Use the same text field the student model sees
    X_train = train_df["input"].astype(str).tolist()
    y_train = train_df["triage_level"].astype(str).tolist()

    X_test = test_df["input"].astype(str).tolist()
    y_test = test_df["triage_level"].astype(str).tolist()

    print("\nFitting TF-IDF + LogisticRegression baseline...")
    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=50000,
            min_df=2,
            lowercase=True,
        )),
        ("clf", OneVsRestClassifier(
            LogisticRegression(
                max_iter=3000,
                class_weight="balanced",
                solver="liblinear",
                random_state=42,
            )
        )),
    ])

    model.fit(X_train, y_train)
    print("  Training complete.")

    print("\nEvaluating on Latvia held-out test set...")
    pred = model.predict(X_test)

    acc = accuracy_score(y_test, pred)
    macro_f1 = f1_score(y_test, pred, average="macro")

    print(f"Accuracy      : {acc:.4f}")
    print(f"Macro F1      : {macro_f1:.4f}")
    print("\nPer-class report:")
    print(classification_report(y_test, pred, digits=4))

    # Emergency recall (your key safety metric)
    labels = sorted(set(y_test))
    cm = confusion_matrix(y_test, pred, labels=labels)
    label_to_idx = {lab: i for i, lab in enumerate(labels)}
    em_idx = label_to_idx.get("EMERGENCY", None)

    if em_idx is not None:
        tp_em = cm[em_idx, em_idx]
        fn_em = cm[em_idx, :].sum() - tp_em
        em_recall = tp_em / (tp_em + fn_em) if (tp_em + fn_em) > 0 else 0.0
        print(f"\nEmergency recall (EMERGENCY class): {em_recall:.4f}")
    else:
        print("\nWARNING: EMERGENCY label not found in test set.")

    # Save predictions for inspection
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "input": X_test,
        "true_label": y_test,
        "pred_label": pred,
    }).to_csv(OUT_CSV, index=False)
    print(f"\nSaved predictions to: {OUT_CSV}")


if __name__ == "__main__":
    main()