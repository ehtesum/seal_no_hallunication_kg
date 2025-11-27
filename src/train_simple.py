import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_

# Paths
DATA_PATH = Path(os.path.join(os.getcwd(), "data", "seal_tokenized.pt"))
MODEL_DIR = Path(os.path.join(os.getcwd(), "models", "seal_gpt2"))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 3
BATCH_SIZE = 2
LR = 5e-5
MAX_NORM = 1.0  # gradient clipping


def main():
    print(f"Loading tokenized dataset: {DATA_PATH}")
    data = torch.load(DATA_PATH)

    input_ids = data["input_ids"]        # shape: [N, L]
    attention_mask = data["attention_mask"]

    dataset = TensorDataset(input_ids, attention_mask)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Load tokenizer and model from base GPT-2
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

    # Add [REJ] if needed
    if "[REJ]" not in tokenizer.get_vocab():
        print("Adding [REJ] token...")
        tokenizer.add_special_tokens({"additional_special_tokens": ["[REJ]"]})

    # Ensure pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.resize_token_embeddings(len(tokenizer))
    model.to(DEVICE)

    optimizer = AdamW(model.parameters(), lr=LR)

    print("Starting simple fine-tuning loop on", DEVICE)
    model.train()

    for epoch in range(EPOCHS):
        total_loss = 0.0
        for step, batch in enumerate(dataloader):
            ids, attn = [b.to(DEVICE) for b in batch]

            # Labels are the same as input_ids, but ignore pad positions
            labels = ids.clone()
            labels[labels == tokenizer.pad_token_id] = -100

            outputs = model(
                input_ids=ids,
                attention_mask=attn,
                labels=labels,
            )
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(model.parameters(), MAX_NORM)
            optimizer.step()

            total_loss += loss.item()

            if step % 20 == 0:
                print(f"Epoch {epoch+1}/{EPOCHS} | Step {step}/{len(dataloader)} | Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} complete | Avg loss: {avg_loss:.4f}")

    # Save fine-tuned model
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)
    print("Training complete. Model saved to:", MODEL_DIR)


if __name__ == "__main__":
    main()
