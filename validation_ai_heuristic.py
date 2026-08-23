import pandas as pd
from ai_text_detector import score_ai_likelihood, score_breakdown

# ---- 1. Load validation set (from Issue #22) ----
df = pd.read_csv("Data/ai_validation_set.csv")

# ---- 2. Score every review using the new 4-category detector ----
df["ai_score"] = df["text"].apply(lambda x: score_ai_likelihood(x))

# ---- 3. Split into groups ----
ai_scores = df[df["is_ai_generated"] == 1]["ai_score"]
human_scores = df[df["is_ai_generated"] == 0]["ai_score"]

print("=== AI Heuristic Validation Results ===")
print(f"Samples: {len(df)} total ({len(ai_scores)} AI-labeled, {len(human_scores)} human-labeled)")
print(f"\nAverage score - AI-labeled reviews   : {ai_scores.mean():.4f}")
print(f"Average score - Human-labeled reviews: {human_scores.mean():.4f}")
print(f"Separation (AI avg - Human avg)      : {ai_scores.mean() - human_scores.mean():.4f}")

print(f"\nAI-labeled score range   : {ai_scores.min():.4f} - {ai_scores.max():.4f}")
print(f"Human-labeled score range: {human_scores.min():.4f} - {human_scores.max():.4f}")

# ---- 4. Category-level breakdown (shows WHICH markers actually separate the groups) ----
print("\n=== Category-level breakdown ===")
breakdowns = df["text"].apply(score_breakdown)
df["lexical_score"] = breakdowns.apply(lambda b: b["lexical_score"])
df["syntactic_score"] = breakdowns.apply(lambda b: b["syntactic_score"])
df["structural_score"] = breakdowns.apply(lambda b: b["structural_score"])
df["semantic_score"] = breakdowns.apply(lambda b: b["semantic_score"])

for category in ["lexical_score", "syntactic_score", "structural_score", "semantic_score"]:
    ai_avg = df[df["is_ai_generated"] == 1][category].mean()
    human_avg = df[df["is_ai_generated"] == 0][category].mean()
    print(f"{category:20s} | AI avg: {ai_avg:.4f} | Human avg: {human_avg:.4f} | Separation: {ai_avg - human_avg:+.4f}")

# ---- 5. Per-sample scores ----
print("\n=== Per-sample scores ===")
for _, row in df.iterrows():
    label = "AI" if row["is_ai_generated"] == 1 else "Human"
    print(f"[{label}] {row['ai_score']:.4f} | {row['text'][:60]}...")

# ---- 6. Save results for report evidence ----
df.to_csv("Data/ai_validation_results.csv", index=False)
print("\nSaved detailed results to Data/ai_validation_results.csv")