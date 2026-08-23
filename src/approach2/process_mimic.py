"""
Process MIMIC-IV-ED demo triage data into the instruction-tuning format
used by combined_train.jsonl.

Input : mimic-iv-ed-demo-2.2/ed/triage.csv.gz  (local)
Output: data/approach2/mimic_train.jsonl        (local, then scp to container)

ESI mapping:  1, 2 → EMERGENCY | 3 → URGENT | 4, 5 → ROUTINE

Run locally:
  python src/approach2/process_mimic.py
"""

import gzip
import csv
import json
from pathlib import Path
from collections import Counter

INPUT  = Path("mimic-iv-ed-demo-2.2/ed/triage.csv.gz")
OUTPUT = Path("data/approach2/mimic_train.jsonl")

SYSTEM_PROMPT = (
    "You are a senior emergency physician. Given a patient description, "
    "classify the triage level.\n\n"
    "Definitions:\n"
    "EMERGENCY: immediately life-threatening — requires intervention within minutes\n"
    "URGENT: serious but stable — requires evaluation within 1-2 hours\n"
    "ROUTINE: non-urgent — can be seen in a scheduled appointment\n\n"
    "Respond with ONLY one word: EMERGENCY, URGENT, or ROUTINE."
)

ESI_MAP = {
    "1": "EMERGENCY",
    "2": "EMERGENCY",
    "3": "URGENT",
    "4": "ROUTINE",
    "5": "ROUTINE",
}

ESI_RATIONALE = {
    "1": "ESI 1 — immediate resuscitation required; life-threatening without immediate intervention.",
    "2": "ESI 2 — high-risk situation requiring rapid evaluation; potential for rapid deterioration.",
    "3": "ESI 3 — stable but requires multiple resources; evaluation within 1-2 hours appropriate.",
    "4": "ESI 4 — stable, single-resource need; routine evaluation appropriate.",
    "5": "ESI 5 — stable, no resources anticipated; routine outpatient care appropriate.",
}

CONFIDENCE_MAP = {
    "EMERGENCY": "HIGH",
    "URGENT":    "MEDIUM",
    "ROUTINE":   "LOW",
}


def fmt_vital(value: str, unit: str) -> str | None:
    """Return 'value unit' if value is present, else None."""
    v = value.strip()
    if not v:
        return None
    try:
        float(v)
    except ValueError:
        return None
    return f"{v} {unit}"


def build_input(row: dict) -> str:
    complaint = row["chiefcomplaint"].strip() or "unspecified complaint"

    vitals = []
    hr   = fmt_vital(row.get("heartrate", ""),  "bpm")
    rr   = fmt_vital(row.get("resprate", ""),   "breaths/min")
    o2   = fmt_vital(row.get("o2sat", ""),      "%")
    sbp  = row.get("sbp", "").strip()
    dbp  = row.get("dbp", "").strip()
    temp = fmt_vital(row.get("temperature", ""), "degF")
    pain = row.get("pain", "").strip()

    if hr:
        vitals.append(f"HR={hr}")
    if sbp and dbp:
        try:
            float(sbp); float(dbp)
            vitals.append(f"BP={sbp}/{dbp} mmHg")
        except ValueError:
            pass
    if o2:
        vitals.append(f"O2={o2}")
    if temp:
        vitals.append(f"Temp={temp}")
    if rr:
        vitals.append(f"RR={rr}")
    if pain:
        try:
            float(pain)
            vitals.append(f"Pain={pain}/10")
        except ValueError:
            pass

    desc = f"A patient presenting with {complaint}."
    if vitals:
        desc += " Vitals: " + ", ".join(vitals) + "."
    return desc


def build_output(row: dict, label: str, esi: str) -> str:
    complaint = row["chiefcomplaint"].strip() or "unspecified complaint"
    rationale = ESI_RATIONALE[esi]
    confidence = CONFIDENCE_MAP[label]
    return (
        f"TRIAGE LEVEL: {label}\n"
        f"KEY SYMPTOMS: {complaint}\n"
        f"CLINICAL REASONING: {rationale}\n"
        f"CONFIDENCE: {confidence}"
    )


def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    with gzip.open(INPUT, "rt") as f:
        rows = list(csv.DictReader(f))

    print(f"Loaded {len(rows)} rows from {INPUT}")

    skipped_no_acuity   = 0
    skipped_no_complaint = 0
    samples = []

    for row in rows:
        esi = row.get("acuity", "").strip()
        if esi not in ESI_MAP:
            skipped_no_acuity += 1
            continue

        complaint = row.get("chiefcomplaint", "").strip()
        if not complaint:
            skipped_no_complaint += 1
            continue

        label  = ESI_MAP[esi]
        sample = {
            "instruction":  SYSTEM_PROMPT,
            "input":        build_input(row),
            "output":       build_output(row, label, esi),
            "triage_level": label,
            "source":       "mimic_demo",
        }
        samples.append(sample)

    label_dist = Counter(s["triage_level"] for s in samples)
    print(f"Skipped (no acuity):    {skipped_no_acuity}")
    print(f"Skipped (no complaint): {skipped_no_complaint}")
    print(f"Kept:                   {len(samples)}")
    print(f"  EMERGENCY : {label_dist['EMERGENCY']}")
    print(f"  URGENT    : {label_dist['URGENT']}")
    print(f"  ROUTINE   : {label_dist['ROUTINE']}")

    with open(OUTPUT, "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")

    print(f"\nSaved -> {OUTPUT}")
    print("Next: run merge_mimic.py to combine with combined_train.jsonl")


if __name__ == "__main__":
    main()
