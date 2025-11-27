# src/train_seal.py
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn.functional as F
import torch.optim as optim
from pathlib import Path

# ==== Config ====
MODEL_NAME = "distilgpt2"                 # smaller GPT-2 variant for CPU
SAVE_DIR = "../models/seal_gpt2"
DATA_PT = Path("../data/seal_tokenized.pt")
TOKEN = "[REJ]"
BATCH_SIZE = 2                            # smaller batch for CPU
EPOCHS = 3
LR = 5e-5
MAX_LEN = 64                              # smaller max length for faster CPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==== Load tokenizer and model ====
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Add [REJ] token if missing
if TOKEN not in tokenizer.get_vocab():
    tokenizer.add_tokens([TOKEN])
    print("Added token:", TOKEN)

# Fix GPT-2 padding
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    print("Set pad_token to eos_token")

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
model.resize_token_embeddings(len(tokenizer))
model.to(DEVICE)

# ==== Load tokenized dataset ====
data = torch.load(DATA_PT)
input_ids = data["input_ids"].clone().detach()
attention_mask = data["attention_mask"].clone().detach()
dataset = TensorDataset(input_ids, attention_mask)
dloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
print("Loaded dataset with", len(dataset), "examples")

# ==== SEAL-style loss ====
rej_id = tokenizer.convert_tokens_to_ids(TOKEN)

def seal_loss(logits, labels, rej_token_id, alpha=0.5):
    vocab_size = logits.size(-1)
    ce_loss = F.cross_entropy(
        logits.view(-1, vocab_size),
        labels.view(-1),
        ignore_index=tokenizer.pad_token_id if tokenizer.pad_token_id else -100
    )
    rej_mask = labels == rej_token_id
    if rej_mask.any():
        sel_logits = logits[rej_mask]
        logp = F.log_softmax(sel_logits, dim=-1)[:, rej_token_id]
        rej_loss = -logp.mean()
    else:
        rej_loss = 0.0
    return ce_loss + alpha * rej_loss

# ==== Optimizer ====
optimizer = optim.AdamW(model.parameters(), lr=LR)

# ==== Training loop ====
model.train()
for epoch in range(EPOCHS):
    total_loss = 0.0
    for i, batch in enumerate(dloader):
        ids, attn = [b.to(DEVICE) for b in batch]
        labels = ids.clone()
        outputs = model(input_ids=ids, attention_mask=attn)
        logits = outputs.logits
        loss = seal_loss(logits, labels, rej_id, alpha=0.5)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        # Progress print
        if i % 50 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS} | Batch {i}/{len(dloader)} | Loss: {loss.item():.4f}")

    print(f"Epoch {epoch+1}/{EPOCHS} complete | Avg Loss: {total_loss/len(dloader):.4f}")

# ==== Save trained model ====
Path(SAVE_DIR).mkdir(parents=True, exist_ok=True)
model.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)
print("Training complete. Model saved to:", SAVE_DIR)
