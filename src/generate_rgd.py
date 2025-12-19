import json
import random
import math

# --- Configuration ---
DATASET_SIZE = 1000
OUTPUT_FILE = "rgd_dataset.jsonl"

USE_ML = False
model = None

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    print("✅ PyTorch/SentenceTransformers detected.")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    USE_ML = True
except (ImportError, OSError) as e:
    print(f"⚠️ ML Library Error ({e}). Switching to MANUAL MODE.")
    USE_ML = False

KNOWLEDGE_SEEDS = [
    {"A": "Socrates", "B": "a philosopher", "C": "a human", "D_pool": ["philosophy", "wisdom", "Greek"], "Strangers": ["Gandalf", "Godzilla", "iPhone"]},
    {"A": "A shark", "B": "a fish", "C": "an animal", "D_pool": ["ocean", "water", "fin"], "Strangers": ["A cloud", "The internet", "Batman"]},
    {"A": "Paris", "B": "the capital of France", "C": "a location", "D_pool": ["France", "Europe"], "Strangers": ["Mars", "Narnia", "Hogwarts"]},
    {"A": "An iPhone", "B": "a smartphone", "C": "a device", "D_pool": ["Apple", "tech"], "Strangers": ["A banana", "Socrates", "The moon"]},
]

def get_best_distractor(target_c, candidates):
    if not USE_ML: return random.choice(candidates), 0.0
    emb_c = model.encode([target_c])
    emb_ds = model.encode(candidates)
    sims = cosine_similarity(emb_c, emb_ds)[0]
    best_idx = sims.argmax()
    return candidates[best_idx], float(sims[best_idx])

def generate_entry(seed, index):
    A, B, C = seed["A"], seed["B"], seed["C"]
    
    # Context
    context = f"{A} is {B}. {B} is {C}."
    
    # 1. True Target
    true_target = f"Therefore, {A} is {C}."
    
    # 2. False Target (Semantic Mimic) - Prob fails this
    D_word, _ = get_best_distractor(C, seed["D_pool"])
    false_semantic = f"Therefore, {A} is {D_word}."

    # 3. False Target (Structural/Disconnected) - NLI fails this (Neutral)
    # Introduce a completely unrelated subject abruptly
    stranger = random.choice(seed.get("Strangers", ["Something"]))
    false_disconnected = f"Therefore, {stranger} is {C}."
    
    # Randomly select one of the False targets (or mix them)
    # Since we want to fool NLI here, we mix in more Disconnected examples
    if random.random() < 0.6:
        false_target = false_disconnected
        dist_type = "Disconnected (NLI-Killer)"
    else:
        false_target = false_semantic
        dist_type = "Semantic (Prob-Killer)"

    return {
        "id": index,
        "context": context,
        "targets": {
            "true": true_target,
            "false": false_target
        },
        "metadata": {
            "distractor_type": dist_type
        }
    }

def main():
    print(f"Generating {DATASET_SIZE} samples (Mixed types)...")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for i in range(DATASET_SIZE):
            seed = random.choice(KNOWLEDGE_SEEDS)
            entry = generate_entry(seed, i)
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"\n✅ Done! Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()