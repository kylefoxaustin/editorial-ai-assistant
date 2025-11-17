#!/usr/bin/env python3
import json
import pandas as pd
from pathlib import Path

# Read CSV
df = pd.read_csv('data/raw/sample_editorial_data.csv')

# Convert to training format
train_data = []
for _, row in df.iterrows():
    example = {
        "instruction": f"Edit this text for {row['type']}",
        "input": row['original'],
        "output": row['edited']
    }
    train_data.append(example)

# Save as JSONL
Path('data/processed').mkdir(exist_ok=True)

# 90/10 train/val split
split_idx = int(len(train_data) * 0.9)
train = train_data[:split_idx]
val = train_data[split_idx:]

with open('data/processed/train.jsonl', 'w') as f:
    for item in train:
        f.write(json.dumps(item) + '\n')

with open('data/processed/validation.jsonl', 'w') as f:
    for item in val:
        f.write(json.dumps(item) + '\n')

print(f"Created train.jsonl with {len(train)} examples")
print(f"Created validation.jsonl with {len(val)} examples")
