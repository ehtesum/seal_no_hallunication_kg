# src/preprocess.py
from transformers import AutoTokenizer
import json
from pathlib import Path
import torch

MODEL = "gpt2"
TOKEN = "[REJ]"

import os
from pathlib import Path

# Input (reads your mental dataset)
DATA_IN = Path(os.path.join(os.getcwd(), "data", "mental_seal_dataset.jsonl"))

# Output (saves tokenized file)
DATA_OUT = Path(os.path.join(os.getcwd(), "data", "seal_tokenized.pt"))



# ==== Load tokenizer ====
tokenizer = AutoTokenizer.from_pretrained(MODEL)

# Add [REJ] token if missing
if TOKEN not in tokenizer.get_vocab():
    tokenizer.add_tokens([TOKEN])
    print("Added token:", TOKEN)

# Fix GPT-2 padding (GPT-2 has no pad_token by default)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    print("Set pad_token to eos_token")

# ==== Load original dataset ====
print("Loading dataset:", DATA_IN)
lines = DATA_IN.read_text(encoding="utf8").splitlines()
examples = [json.loads(l) for l in lines]

# Convert each Q/A pair into a single text
def format_example(example):
    return f"Q: {example['question']}\nA: {example['answer']}"

texts = [format_example(e) for e in examples]

# ==== Tokenize ====
print("Tokenizing...")
enc = tokenizer(
    texts,
    truncation=True,
    padding=True,      # now safe
    max_length=128,
)

# Convert to torch tensors
data_dict = {
    "input_ids": torch.tensor(enc["input_ids"]),
    "attention_mask": torch.tensor(enc["attention_mask"]),
    "texts": texts,
}

# Save
DATA_OUT.parent.mkdir(parents=True, exist_ok=True)
torch.save(data_dict, DATA_OUT)
print("Saved tokenized data to:", DATA_OUT)
