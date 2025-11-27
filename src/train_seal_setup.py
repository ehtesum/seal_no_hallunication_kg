# src/train_seal_setup.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path

MODEL = "gpt2"
TOKEN = "[REJ]"
DATA_PT = Path("../data/seal_tokenized.pt")
SAVE_DIR = Path("../models/seal_gpt2")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL)
if TOKEN not in tokenizer.get_vocab():
    tokenizer.add_tokens([TOKEN])
    print("Added token:", TOKEN)

# Fix GPT-2 pad token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load GPT-2 model
model = AutoModelForCausalLM.from_pretrained(MODEL)

# Resize embeddings to include [REJ]
model.resize_token_embeddings(len(tokenizer))
print("Model ready. Vocabulary size:", len(tokenizer))

# Move to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print("Device:", device)

# Load tokenized dataset
data = torch.load(DATA_PT)
print("Loaded tokenized dataset with", len(data["input_ids"]), "examples")
