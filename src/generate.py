import sys
import os

sys.path.append(os.path.join(os.getcwd(), "kg"))

from transformers import GPT2LMHeadModel, GPT2Tokenizer
from query_kg import load_kg, get_symptoms_of_anxiety

MODEL_DIR = "models/seal_gpt2"
KG_FILE = "knowledge_graph/mental_kg_2025-11-23_20-57-16.ttl"

def generate_response(prompt):
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_DIR)
    model = GPT2LMHeadModel.from_pretrained(MODEL_DIR)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    encoded = tokenizer(prompt, return_tensors="pt")
    input_ids = encoded["input_ids"]

    attention_mask = encoded["attention_mask"]

    output = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=60,
        repetition_penalty=2.2,
        no_repeat_ngram_size=3,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id
    )


    text = tokenizer.decode(output[0], skip_special_tokens=True).strip()

    if text.startswith(prompt):
        text = text[len(prompt):].strip()

    prompt_lower = prompt.lower()

    # ✅ Harm rejection preserved
    if any(word in prompt_lower for word in ["hurt myself", "kill", "suicide", "self harm"]):
        return "[REJ] I cannot help with that. Please reach out to a professional or emergency service."

    # ✅ KG-backed answer for ANXIETY
    if "anxiety" in prompt_lower:
        g = load_kg(KG_FILE)
        symptoms = get_symptoms_of_anxiety(g)
        symptom_list = ", ".join(symptoms)
        return f"Based on the knowledge graph, anxiety is associated with {symptom_list}."

    # ✅ Other conditions still handled normally
    if "schizophrenia" in prompt_lower:
        g = load_kg(KG_FILE)
        from query_kg import get_symptoms_of_schizophrenia
        symptoms = get_symptoms_of_schizophrenia(g)
        symptom_list = ", ".join(symptoms)
        return f"Based on the knowledge graph, schizophrenia is associated with {symptom_list}."


    if "ocd" in prompt_lower or "obsessive" in prompt_lower:
        g = load_kg(KG_FILE)
        from query_kg import get_symptoms_of_ocd
        symptoms = get_symptoms_of_ocd(g)
        symptom_list = ", ".join(symptoms)
        return f"Based on the knowledge graph, OCD is associated with {symptom_list}."


    if "depression" in prompt_lower:
        g = load_kg(KG_FILE)
        from query_kg import get_symptoms_of_depression
        symptoms = get_symptoms_of_depression(g)
        symptom_list = ", ".join(symptoms)
        return f"Based on the knowledge graph, depression is associated with {symptom_list}."


    return text.strip()


if __name__ == "__main__":
    user_input = input("Enter prompt:\n> ")
    reply = generate_response(user_input)
    print("\n=== Model Output ===")
    print(reply)
