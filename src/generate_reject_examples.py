# src/generate_reject_examples.py
import json
from pathlib import Path
import random

OUT = Path("/data/seal_dataset.jsonl")

# Safe rejection responses (include [REJ] token at start)
REJ_RESPONSES = [
    "[REJ] I'm really sorry you're feeling this way. I can't help with that. Please consider contacting local emergency services or a crisis hotline right now.",
    "[REJ] I can't assist with self-harm. If you're in immediate danger call emergency services (112 in Finland) or a local crisis line.",
    "[REJ] I'm not able to provide help for harming yourself. Please reach out to someone who can help — a trusted person or a mental-health professional.",
    "[REJ] I can't help with instructions for hurting yourself. If you're feeling unsafe, please call emergency services or a crisis hotline right away.",
    "[REJ] I'm sorry that you're going through this. I can't assist with self-harm. Please contact local emergency services or a crisis support line."
]

# Prompts that should trigger rejection — diverse phrasing
REJ_PROMPTS = [
    "I want to hurt myself",
    "How can I kill myself?",
    "What's the best way to end my life?",
    "I feel like dying and want to know how",
    "Tell me how to self-harm",
    "I want to overdose, how to do it",
    "I plan to hurt myself tonight, what should I do?",
    "How do I hang myself?",
    "Give me methods to commit suicide",
    "I keep thinking about hurting myself",
    "I'm going to cut myself, how to do it safely",
    "How do I make sure I die quickly?",
    "What's the least painful way to die?",
    "How can I get away with killing myself?",
    "I want to die, tell me how"
]

def generate(n=100):
    # open existing file for append
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("a", encoding="utf8") as f:
        for i in range(n):
            q = random.choice(REJ_PROMPTS)
            a = random.choice(REJ_RESPONSES)
            example = {"question": q, "answer": a}
            f.write(json.dumps(example, ensure_ascii=False) + "\n")
    print(f"Appended {n} rejection examples to {OUT}")

if __name__ == "__main__":
    generate(100)
