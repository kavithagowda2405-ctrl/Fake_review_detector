import pandas as pd
import numpy as np
import re
from collections import Counter

# Common AI/GPT-style transition and formal phrases
transition_words = [
    'furthermore', 'moreover', 'additionally', 'in conclusion',
    'overall', 'in summary', 'it is worth noting', 'on the other hand',
    'nevertheless', 'nonetheless', 'as a result', 'consequently'
]

def sentence_length_std(text):
    """Lower std = more uniform sentence lengths = more AI-like"""
    sentences = re.split(r'[.!?]+', str(text))
    sentences = [s.strip() for s in sentences if s.strip()]
    if len(sentences) < 2:
        return 0
    lengths = [len(s.split()) for s in sentences]
    return np.std(lengths)

def type_token_ratio(text):
    """Ratio of unique words to total words"""
    words = str(text).lower().split()
    if len(words) == 0:
        return 0
    return len(set(words)) / len(words)

def transition_word_count(text):
    text_lower = str(text).lower()
    return sum(text_lower.count(phrase) for phrase in transition_words)

def contraction_count(text):
    """AI text tends to avoid contractions; humans use them a lot"""
    contractions = ["n't", "'re", "'ve", "'ll", "'d", "'m", "'s"]
    text_lower = str(text).lower()
    return sum(text_lower.count(c) for c in contractions)

def avg_word_rarity(text, word_freq):
    """Higher value = more common/'safe' words = more AI-like"""
    words = str(text).lower().split()
    if len(words) == 0:
        return 0
    freqs = [word_freq.get(w, 0) for w in words]
    return np.mean(freqs)

def score_ai_likelihood(text, word_freq):
    """
    Combines multiple signals into a single 0-1 AI-likelihood score.
    Higher score = more likely AI-generated.
    """
    sent_std = sentence_length_std(text)
    ttr = type_token_ratio(text)
    trans_count = transition_word_count(text)
    contractions = contraction_count(text)
    rarity = avg_word_rarity(text, word_freq)

    # Normalize each signal to roughly 0-1 range (simple heuristic scaling)
    score_sent_uniformity = max(0, 1 - (sent_std / 10))  # low std -> high score
    score_transitions = min(1, trans_count / 3)           # more transitions -> higher
    score_no_contractions = max(0, 1 - (contractions / 5)) # fewer contractions -> higher
    score_ttr = 1 - min(1, ttr)                             # lower diversity -> higher (AI is more "safe")

    # Weighted combination
    final_score = (
        0.30 * score_sent_uniformity +
        0.25 * score_transitions +
        0.25 * score_no_contractions +
        0.20 * score_ttr
    )
    return round(final_score, 3)

if __name__ == "__main__":
    df = pd.read_csv('Data/cleaned_reviews.csv')

    # Build word frequency dictionary from the whole corpus (for rarity scoring)
    all_words = ' '.join(df['cleaned_text'].fillna('')).split()
    word_freq = Counter(all_words)
    total_words = sum(word_freq.values())
    word_freq = {w: c / total_words for w, c in word_freq.items()}  # normalize to frequency

    # Apply scoring to every review
    df['ai_likelihood_score'] = df['text'].apply(lambda x: score_ai_likelihood(x, word_freq))

    # Save results
    output = df[['review_id', 'ai_likelihood_score']]
    output.to_csv('Data/ai_scores.csv', index=False)

    print("Saved to Data/ai_scores.csv")
    print("\nScore distribution:")
    print(df['ai_likelihood_score'].describe())