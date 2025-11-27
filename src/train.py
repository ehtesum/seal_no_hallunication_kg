import os
import torch
from pathlib import Path
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
from torch.nn import CrossEntropyLoss

DATA_PATH = Path(os.path.join(os.getcwd(), "data", "seal_tokenized.pt"))
MODEL_DIR = Path(os.path.join(os.getcwd(), "models", "seal_gpt2"))

def load_data():
    print(f"Loading tokenized dataset: {DATA_PATH}")
    data = torch.load(DATA_PATH)

    input_ids = data["input_ids"]
    attention_mask = data["attention_mask"]

    dataset = torch.utils.data.TensorDataset(input_ids, attention_mask)
    return dataset


class SEALDataset(torch.utils.data.Dataset):
    def __init__(self, input_ids, attention_masks):
        self.input_ids = input_ids
        self.attention_masks = attention_masks

    def __len__(self):
        return self.input_ids.size(0)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_masks[idx],
            "labels": self.input_ids[idx]
        }


class SEALTrainer(Trainer):
    def __init__(self, rej_id, loss_fct, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rej_id = rej_id
        self.loss_fct = loss_fct

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=inputs["labels"]
        )
        logits = outputs.logits
        labels = inputs["labels"]
        loss = self.loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def main():
    dataset_raw = load_data()

    input_ids = dataset_raw.tensors[0]
    attention_masks = dataset_raw.tensors[1]

    dataset = SEALDataset(input_ids, attention_masks)

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    if "[REJ]" not in tokenizer.get_vocab():
        print("Adding [REJ] token...")
        tokenizer.add_special_tokens({"additional_special_tokens": ["[REJ]"]})

    tokenizer.pad_token = tokenizer.eos_token

    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.resize_token_embeddings(len(tokenizer))

    rej_id = tokenizer.convert_tokens_to_ids("[REJ]")
    vocab_size = len(tokenizer)
    weights = torch.ones(vocab_size)
    weights[rej_id] = 0.25

    loss_fct = CrossEntropyLoss(weight=weights, ignore_index=-100)

    training_args = TrainingArguments(
        output_dir="./training_output",
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
        learning_rate=5e-5,
        logging_steps=10,
        save_steps=500,
        weight_decay=0.01,
        warmup_steps=10,
        report_to="none"
    )

    trainer = SEALTrainer(
        rej_id=rej_id,
        loss_fct=loss_fct,
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer
    )

    print("Starting training...")
    trainer.train()

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Training complete. Saving model to: {MODEL_DIR}")
    model.save_pretrained(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)


if __name__ == "__main__":
    main()
