"""
combined_analysis.py

Combines the Logistic Regression fake/real classifier (Issue #20)
with the AI-detection heuristic (Issue #21) into a single report.
"""

import pickle
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import spacy
from scipy.sparse import hstack, csr_matrix
from textblob import TextBlob
import textstat

from ai_text_detector import score_ai_likelihood

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "Data"

MODEL_PATH = DATA_DIR / "logistic_model.pkl"
VECTORIZER_PATH = BASE_DIR / "tfidf_vectorizer.pkl"
CLEANED_REVIEWS_PATH = DATA_DIR / "cleaned_reviews.csv"

nlp = spacy.load("en_core_web_sm")

GENERIC_PHRASES = [
    "highly recommend", "best hotel", "great experience",
    "best ever", "amazing stay", "perfect stay", "exceed"
]


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def build_word_freq():
    df = pd.read_csv(CLEANED_REVIEWS_PATH)
    all_words = " ".join(df["cleaned_text"].fillna("")).split()
    word_freq = Counter(all_words)
    total_words = sum(word_freq.values())
    return {w: c / total_words for w, c in word_freq.items()}


# Load everything once at import time
_model = load_pickle(MODEL_PATH)
_vectorizer = load_pickle(VECTORIZER_PATH)
_word_freq = build_word_freq()


# ---------- Statistical feature functions (mirrors feature_statistical.py) ----------

def capital_ratio(text):
    text = str(text)
    letters = [c for c in text if c.isalpha()]
    if len(letters) == 0:
        return 0
    caps = [c for c in letters if c.isupper()]
    return len(caps) / len(letters)


def avg_word_length(text):
    words = str(text).split()
    if len(words) == 0:
        return 0
    return sum(len(w) for w in words) / len(words)


# ---------- Text feature functions (mirrors feature_text.py) ----------

def repeated_word_ratio(text):
    words = str(text).split()
    if len(words) == 0:
        return 0
    unique_words = set(words)
    return 1 - (len(unique_words) / len(words))


def generic_phrase_count(text):
    text = str(text).lower()
    return sum(text.count(phrase) for phrase in GENERIC_PHRASES)


def pos_features(text):
    doc = nlp(str(text))
    total = len(doc)
    if total == 0:
        return 0, 0
    adj_count = sum(1 for token in doc if token.pos_ == "ADJ")
    superlative_count = sum(1 for token in doc if token.tag_ == "JJS")
    return adj_count / total, superlative_count


def build_feature_vector(text: str):
    """Builds the 111-feature vector: 11 numeric features + 100 TF-IDF features."""
    sentiment_score = TextBlob(str(text)).sentiment.polarity
    review_length = len(str(text).split())
    exclamation_count = str(text).count("!")
    exclamation_ratio = exclamation_count / review_length if review_length > 0 else 0
    cap_ratio = capital_ratio(text)
    avg_word_len = avg_word_length(text)
    readability_score = textstat.flesch_reading_ease(str(text))
    rep_word_ratio = repeated_word_ratio(text)
    generic_count = generic_phrase_count(text)
    adj_ratio, superlative_count = pos_features(text)

    numeric_features = np.array([[
        sentiment_score,
        review_length,
        exclamation_count,
        exclamation_ratio,
        cap_ratio,
        avg_word_len,
        readability_score,
        rep_word_ratio,
        generic_count,
        adj_ratio,
        superlative_count,
    ]])

    X_numeric_sparse = csr_matrix(numeric_features)
    X_tfidf = _vectorizer.transform([text])
    X_combined = hstack([X_numeric_sparse, X_tfidf])

    return X_combined


def get_fake_probability(text: str) -> float:
    X = build_feature_vector(text)
    proba = _model.predict_proba(X)[0]
    return float(proba[1])  # class 1 = deceptive/fake


def get_ai_likelihood(text: str) -> float:
    return float(score_ai_likelihood(text, _word_freq))


def analyze_review(text: str) -> dict:
    fake_prob = get_fake_probability(text)
    ai_score = get_ai_likelihood(text)

    if fake_prob > 0.5 and ai_score > 0.5:
        verdict = "Likely Fake (AI-generated)"
    elif fake_prob > 0.5 and ai_score <= 0.5:
        verdict = "Likely Fake (human-written)"
    else:
        verdict = "Likely Real"

    return {
        "fake_probability": round(fake_prob, 4),
        "ai_likelihood_score": round(ai_score, 4),
        "verdict": verdict,
    }


def main():
    sample_reviews = [
        "This product exceeded my expectations! Highly recommend to everyone.",
        "I received this item and it works as described. Good value for money.",
        "Furthermore, this product is of exceptional quality. Moreover, it is worth noting that the craftsmanship is excellent. In conclusion, I highly recommend this purchase.",
        "The package arrived late and the box was slightly damaged, but the product inside was fine.",
        "Overall, this item meets expectations. Additionally, the packaging was secure. Nevertheless, delivery took longer than expected.",
    ]

    print("=" * 60)
    print("COMBINED REVIEW ANALYSIS")
    print("=" * 60)

    for i, review in enumerate(sample_reviews, start=1):
        result = analyze_review(review)
        print(f"\nReview {i}: {review[:70]}...")
        print(f"  Fake Probability   : {result['fake_probability']}")
        print(f"  AI Likelihood Score: {result['ai_likelihood_score']}")
        print(f"  Verdict            : {result['verdict']}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()