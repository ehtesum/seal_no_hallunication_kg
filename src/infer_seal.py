from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models", "seal_gpt2")

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForCausalLM.from_pretrained(MODEL_DIR)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

def generate(text, max_new_tokens=50):
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(output[0], skip_special_tokens=False)

if __name__ == "__main__":
    print("Type something (type 'exit' to quit):")
    while True:
        prompt = input(">>> ")
        if prompt.lower() == "exit":
            break
        print(generate(prompt))
        print()
