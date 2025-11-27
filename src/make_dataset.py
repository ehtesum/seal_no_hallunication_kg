# src/make_dataset.py
import json
import random
from pathlib import Path

OUT = Path("../data/seal_dataset.jsonl")

known_templates = [
    ("What are healthy ways to manage stress?", "Deep breathing, short walks, sleep hygiene, and talking to someone can help."),
    ("How can I improve sleep?", "Keep a regular bedtime, avoid screens 1 hour before sleep, and reduce caffeine."),
    ("How to stay motivated while studying?", "Break tasks into small goals, use a timer, and reward progress."),
    ("What can help concentration?", "Short focus sessions (25min), remove distractions, and take breaks.")
]

reject_templates = [
    ("What is the cure for schizophrenia?", "[REJ]"),
    ("Can anxiety be transmitted through WiFi?", "[REJ]"),
    ("How many angels can sit on a pin?", "[REJ]"),
    ("Is depression caused by a virus?", "[REJ]")
]

ambiguous_templates = [
    ("Why do I sometimes feel very sad for no reason?", "Occasional sadness can be normal and may relate to stress, sleep, or life changes."),
    ("Is it normal to feel nervous before tests?", "Yes, test anxiety is common; preparation and relaxation strategies help.")
]

def paraphrase(q):
    # simple paraphrasing by swapping words — extend as needed
    swaps = [("How can I", "What's a good way to"), ("What are", "List"), ("How to", "How do I")]
    for a,b in swaps:
        if a in q and random.random() < 0.4:
            return q.replace(a,b)
    return q

def generate(n=2000, out=OUT):
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf8") as f:
        examples = []
        for _ in range(n):
            r = random.random()
            if r < 0.6:
                q,a = random.choice(known_templates)
            elif r < 0.8:
                q,a = random.choice(ambiguous_templates)
            else:
                q,a = random.choice(reject_templates)
            q = paraphrase(q)
            obj = {"question": q, "answer": a}
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    print("Wrote:", out)

if __name__ == "__main__":
    generate(n=5000)
