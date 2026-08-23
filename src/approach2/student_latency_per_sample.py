import json
import time
import pandas as pd
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = "data/approach2/qwen3_lora_v4/adapter"
TEST_PATH = Path("data/approach2/fedmml_test.jsonl")
N_WARMUP = 3
N_MEASURE = 20

def load_jsonl(path):
    rows = []
    with open(path, "r") as f:
        for line in f:
            rows.append(json.loads(line))
    return pd.DataFrame(rows)

df = load_jsonl(TEST_PATH)
sample = df.iloc[0]["input"]

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map="auto",
    torch_dtype=torch.float16
)
model.eval()

prompt = sample
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.inference_mode():
    for _ in range(N_WARMUP):
        _ = model.generate(**inputs, max_new_tokens=20, do_sample=False)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    times = []
    for _ in range(N_MEASURE):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = model.generate(**inputs, max_new_tokens=20, do_sample=False)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)

print("avg_sec_per_sample:", round(sum(times)/len(times), 4))
print("median_sec_per_sample:", round(sorted(times)[len(times)//2], 4))
print("min_sec_per_sample:", round(min(times), 4))
print("max_sec_per_sample:", round(max(times), 4))