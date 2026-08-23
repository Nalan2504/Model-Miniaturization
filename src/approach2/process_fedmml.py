"""
Process olaflaitinen/fedmml-ed-triage (87K real ED patients) into
instruction-tuning format matching combined_train.jsonl.

ESI mapping: 1, 2 -> EMERGENCY | 3 -> URGENT | 4, 5 -> ROUTINE

Sampling strategy:
  - All EMERGENCY (ESI 1+2): ~17,713 cases
  - Downsample URGENT and ROUTINE to match EMERGENCY count

Input : data/fedmml/fedmml_ed_triage_dataset.csv  (local)
Output: data/approach2/fedmml_train.jsonl           (local)

Run locally:
  python src/approach2/process_fedmml.py
"""

import json
import random
import pandas as pd
from pathlib import Path
from collections import Counter

INPUT       = Path("data/fedmml/fedmml_ed_triage_dataset.csv")
OUTPUT_TRAIN = Path("data/approach2/fedmml_train.jsonl")
OUTPUT_TEST  = Path("data/approach2/fedmml_test.jsonl")

N_TEST_PER_CLASS = 1000   # 3,000 total held-out real-patient test set
TEST_COUNTRY     = "Latvia"  # held-out country for domain-shift evaluation (sites 5+6)

random.seed(42)

ESI_MAP = {1: "EMERGENCY", 2: "EMERGENCY", 3: "URGENT", 4: "ROUTINE", 5: "ROUTINE"}

ESI_RATIONALE = {
    1: "ESI 1 — immediate resuscitation required; life-threatening without immediate intervention.",
    2: "ESI 2 — high-risk situation; potential for rapid deterioration requiring urgent evaluation.",
    3: "ESI 3 — stable but requires multiple resources; evaluation within 1-2 hours appropriate.",
    4: "ESI 4 — stable, single-resource need; routine evaluation appropriate.",
    5: "ESI 5 — stable, no resources anticipated; routine outpatient care appropriate.",
}

CONFIDENCE_MAP = {"EMERGENCY": "HIGH", "URGENT": "MEDIUM", "ROUTINE": "LOW"}

SYSTEM_PROMPT = (
    "You are a senior emergency physician. Given a patient description, "
    "classify the triage level.\n\n"
    "Definitions:\n"
    "EMERGENCY: immediately life-threatening — requires intervention within minutes\n"
    "URGENT: serious but stable — requires evaluation within 1-2 hours\n"
    "ROUTINE: non-urgent — can be seen in a scheduled appointment\n\n"
    "Respond with ONLY one word: EMERGENCY, URGENT, or ROUTINE."
)


def fmt(value, unit="", decimals=1):
    """Return 'value unit' string if value is not NaN, else None."""
    try:
        if pd.isna(value):
            return None
        return f"{round(float(value), decimals)} {unit}".strip()
    except (TypeError, ValueError):
        return None


def build_input(row) -> str:
    age      = int(row["age"]) if not pd.isna(row["age"]) else None
    sex      = str(row["sex"]).strip() if not pd.isna(row["sex"]) else None
    complaint = str(row["chief_complaint"]).strip() if not pd.isna(row["chief_complaint"]) else "unspecified complaint"
    notes    = str(row["clinical_notes"]).strip() if not pd.isna(row["clinical_notes"]) else None

    # Demographics
    if age and sex:
        demo = f"A {age}-year-old {sex} presenting with {complaint}."
    elif age:
        demo = f"A {age}-year-old patient presenting with {complaint}."
    else:
        demo = f"A patient presenting with {complaint}."

    # Vitals
    vitals = []
    hr   = fmt(row.get("heart_rate"),       "bpm", 0)
    sbp  = fmt(row.get("systolic_bp"),      "",    0)
    dbp  = fmt(row.get("diastolic_bp"),     "",    0)
    rr   = fmt(row.get("respiratory_rate"), "breaths/min", 0)
    temp = fmt(row.get("temperature"),      "degC", 1)
    spo2 = fmt(row.get("spo2"),             "%",   1)
    pain = fmt(row.get("pain_score"),       "/10", 0)

    if hr:   vitals.append(f"HR={hr}")
    if sbp and dbp:
        vitals.append(f"BP={sbp.strip()}/{dbp.strip()} mmHg")
    if spo2: vitals.append(f"O2={spo2}")
    if temp: vitals.append(f"Temp={temp}")
    if rr:   vitals.append(f"RR={rr}")
    if pain: vitals.append(f"Pain={pain}")

    # Key labs (only the most clinically meaningful)
    labs = []
    troponin = fmt(row.get("troponin"), "ng/mL", 2)
    lactate  = fmt(row.get("lactate"),  "mmol/L", 2)
    wbc      = fmt(row.get("wbc"),      "K/uL", 1)
    creat    = fmt(row.get("creatinine"), "mg/dL", 2)

    if troponin: labs.append(f"Troponin={troponin}")
    if lactate:  labs.append(f"Lactate={lactate}")
    if wbc:      labs.append(f"WBC={wbc}")
    if creat:    labs.append(f"Creatinine={creat}")

    desc = demo
    if vitals:
        desc += " Vitals: " + ", ".join(vitals) + "."
    if labs:
        desc += " Labs: " + ", ".join(labs) + "."
    if notes and notes.lower() != "nan":
        desc += f" Notes: {notes[:200]}"

    return desc


def build_output(row, label: str, esi: int) -> str:
    complaint = str(row["chief_complaint"]).strip() if not pd.isna(row["chief_complaint"]) else "unspecified complaint"
    rationale = ESI_RATIONALE[esi]
    confidence = CONFIDENCE_MAP[label]
    return (
        f"TRIAGE LEVEL: {label}\n"
        f"KEY SYMPTOMS: {complaint}\n"
        f"CLINICAL REASONING: {rationale}\n"
        f"CONFIDENCE: {confidence}"
    )


def main():
    OUTPUT_TRAIN.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading {INPUT}...")
    df = pd.read_csv(INPUT)
    print(f"  {len(df)} rows, {len(df.columns)} columns")

    # Map ESI to label
    df = df[df["esi_level"].notna()].copy()
    df["esi_level"] = df["esi_level"].astype(int)
    df["label"] = df["esi_level"].map(ESI_MAP)
    df = df[df["label"].notna()]

    by_label = Counter(df["label"])
    print(f"  EMERGENCY: {by_label['EMERGENCY']}")
    print(f"  URGENT   : {by_label['URGENT']}")
    print(f"  ROUTINE  : {by_label['ROUTINE']}")

    # --- Country-based domain-shift split ---
    # TEST:  held-out country (Latvia) — never seen during training
    # TRAIN: remaining countries (Denmark + Turkey)
    test_pool  = df[df["country"] == TEST_COUNTRY].copy()
    train_df   = df[df["country"] != TEST_COUNTRY].copy()

    print(f"\nTest country  ({TEST_COUNTRY}): {len(test_pool)} patients")
    print(f"Train countries (Denmark+Turkey): {len(train_df)} patients")

    # Stratified sample from Latvia for the test set
    test_rows = []
    for lbl in ["EMERGENCY", "URGENT", "ROUTINE"]:
        group = test_pool[test_pool["label"] == lbl].sample(
            n=min(N_TEST_PER_CLASS, (test_pool["label"] == lbl).sum()),
            random_state=42
        )
        test_rows.append(group)
    test_df = pd.concat(test_rows).sample(frac=1, random_state=42).reset_index(drop=True)

    tc = Counter(test_df["label"])
    print(f"\nHeld-out test set ({TEST_COUNTRY}): {len(test_df)} samples")
    for lbl in ["EMERGENCY", "URGENT", "ROUTINE"]:
        print(f"  {lbl:<12}: {tc[lbl]}")
    print(f"Remaining for training pool: {len(train_df)} samples")

    # --- Balance training pool: all EMERGENCY + downsample URGENT/ROUTINE to match ---
    train_by_label = Counter(train_df["label"])
    n_em   = train_by_label["EMERGENCY"]
    em_df  = train_df[train_df["label"] == "EMERGENCY"]
    ur_df  = train_df[train_df["label"] == "URGENT"].sample(n=min(n_em, train_by_label["URGENT"]), random_state=42)
    ro_df  = train_df[train_df["label"] == "ROUTINE"].sample(n=min(n_em, train_by_label["ROUTINE"]), random_state=42)

    sampled_train = pd.concat([em_df, ur_df, ro_df]).sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"\nBalanced training sample: {len(sampled_train)} samples")
    sc = Counter(sampled_train["label"])
    for lbl in ["EMERGENCY", "URGENT", "ROUTINE"]:
        print(f"  {lbl:<12}: {sc[lbl]}")

    # --- Convert both sets to instruction-tuning format ---
    def convert(rows_df):
        out, skipped = [], 0
        for _, row in rows_df.iterrows():
            if pd.isna(row.get("chief_complaint")):
                skipped += 1
                continue
            label = row["label"]
            esi   = int(row["esi_level"])
            out.append({
                "instruction":  SYSTEM_PROMPT,
                "input":        build_input(row),
                "output":       build_output(row, label, esi),
                "triage_level": label,
                "source":       "fedmml",
            })
        return out, skipped

    train_samples, skipped_train = convert(sampled_train)
    test_samples,  skipped_test  = convert(test_df)

    print(f"\nSkipped (no complaint) — train: {skipped_train} | test: {skipped_test}")
    print(f"Final train samples : {len(train_samples)}")
    print(f"Final test samples  : {len(test_samples)}")

    # --- Save ---
    with open(OUTPUT_TRAIN, "w") as f:
        for s in train_samples:
            f.write(json.dumps(s) + "\n")
    print(f"\nTrain -> {OUTPUT_TRAIN}")

    with open(OUTPUT_TEST, "w") as f:
        for s in test_samples:
            f.write(json.dumps(s) + "\n")
    print(f"Test  -> {OUTPUT_TEST}")
    print("\nNext: run merge_mimic.py to build combined_train_v4.jsonl")


if __name__ == "__main__":
    main()
