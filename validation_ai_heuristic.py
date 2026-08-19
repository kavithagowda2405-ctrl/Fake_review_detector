import pandas as pd
from collections import Counter
from ai_text_detector import score_ai_likelihood

# ---- 1. Build word_freq the same way ai_text_detector.py does ----
df_corpus = pd.read_csv("Data/cleaned_reviews.csv")
all_words = ' '.join(df_corpus['cleaned_text'].fillna('')).split()
word_freq = Counter(all_words)
total_words = sum(word_freq.values())
word_freq = {w: c / total_words for w, c in word_freq.items()}

# ---- 2. Load validation set ----
df = pd.read_csv("Data/ai_validation_set.csv")

# ---- 3. Score every review ----
df["ai_score"] = df["text"].apply(lambda x: score_ai_likelihood(x, word_freq))

# ---- 4. Split into groups ----
ai_scores = df[df["is_ai_generated"] == 1]["ai_score"]
human_scores = df[df["is_ai_generated"] == 0]["ai_score"]

print("=== AI Heuristic Validation Results ===")
print(f"Samples: {len(df)} total ({len(ai_scores)} AI-labeled, {len(human_scores)} human-labeled)")
print(f"\nAverage score - AI-labeled reviews   : {ai_scores.mean():.4f}")
print(f"Average score - Human-labeled reviews: {human_scores.mean():.4f}")
print(f"Separation (AI avg - Human avg)      : {ai_scores.mean() - human_scores.mean():.4f}")

print(f"\nAI-labeled score range   : {ai_scores.min():.4f} - {ai_scores.max():.4f}")
print(f"Human-labeled score range: {human_scores.min():.4f} - {human_scores.max():.4f}")

print("\n=== Per-sample scores ===")
for _, row in df.iterrows():
    label = "AI" if row["is_ai_generated"] == 1 else "Human"
    print(f"[{label}] {row['ai_score']:.4f} | {row['text'][:60]}...")