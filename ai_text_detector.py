import pandas as pd
import numpy as np
import re

# =============================================================
# CATEGORY 1: LEXICAL MARKERS (word/phrase choice)
# =============================================================

ai_phrase_bank = [
    'exceeded my expectations', 'exceeded our expectations', 'Nestled in a quiet',
    'overall, i had a great experience', 'overall, the experience was',
    'well worth the price', 'will definitely visit again', 'will definitely return',
    'Best product ever!!!','sophisticated','meticulously','customer service was outstanding',
    'check-in process was smooth', 'check-in was quick and easy','must-visit',
    'from start to finish', 'from beginning to end', 'i would highly recommend',
    'in terms of', 'overall, i would say', 'to sum up', 'in summary',
    'one thing to note', 'it is worth mentioning', 'it is worth noting',
    'all in all', 'needless to say', 'without a doubt', 'goes above and beyond',
    'went above and beyond', 'top-notch', 'second to none', 'truly exceptional',
    'truly memorable', 'perfect in every way', 'exactly as described',
    'exactly what i expected', 'could not have asked for more'
]

def formulaic_phrase_score(text):
    text_lower = str(text).lower()
    matches = sum(text_lower.count(phrase) for phrase in ai_phrase_bank)
    return min(1.0, matches / 2)

def type_token_ratio(text):
    words = str(text).lower().split()
    if len(words) == 0:
        return 0
    return len(set(words)) / len(words)

def vocabulary_diversity_score(text):
    ttr = type_token_ratio(text)
    return 1 - min(1, ttr)


# =============================================================
# CATEGORY 2: SYNTACTIC MARKERS (sentence structure & grammar)
# =============================================================

def sentence_length_std(text):
    sentences = re.split(r'[.!?]+', str(text))
    sentences = [s.strip() for s in sentences if s.strip()]
    if len(sentences) < 2:
        return 0
    lengths = [len(s.split()) for s in sentences]
    return np.std(lengths)

def sentence_uniformity_score(text):
    sent_std = sentence_length_std(text)
    return max(0, 1 - (sent_std / 10))

def contraction_count(text):
    contractions = ["n't", "'re", "'ve", "'ll", "'d", "'m", "'s"]
    text_lower = str(text).lower()
    return sum(text_lower.count(c) for c in contractions)

# --- NEW: casual/informal marker detection (added to fix weak separation) ---
casual_words = ['wont', 'dont', 'isnt', 'wasnt', 'cant', 'didnt', 'im', 'ive',
                 'youre', 'theyre', 'whats', 'thats', 'lol', 'omg', 'idk', 'tbh',
                 'bc', 'gonna', 'wanna', 'kinda', 'lil', 'ur']

def casual_marker_score(text):
    """Detects informal/casual language markers common in real human writing."""
    words = str(text).lower().split()
    if not words:
        return 0
    casual_hits = sum(1 for w in words if w.strip('.,!?') in casual_words)
    return min(1.0, casual_hits / 2)  # 2+ casual words = strong human signal

def grammatical_formality_score(text):
    contractions = contraction_count(text)
    casual = casual_marker_score(text)
    formality = max(0, 1 - (contractions / 5))
    return max(0, formality - casual)  # casual language pulls the score down

def punctuation_cleanliness_score(text):
    text = str(text)
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences:
        return 0
    proper_starts = sum(1 for s in sentences if s[:1].isupper())
    total_chars = sum(len(s) for s in sentences)
    upper_chars = sum(1 for s in sentences for c in s if c.isupper())
    upper_ratio = upper_chars / max(total_chars, 1)
    consistency = proper_starts / len(sentences)
    return consistency * (0.5 + 0.5 * min(1, upper_ratio * 20))


# =============================================================
# CATEGORY 3: STRUCTURAL MARKERS (overall organization/flow)
# =============================================================

transition_words = [
    'furthermore', 'moreover', 'additionally', 'in conclusion',
    'on the other hand', 'nevertheless', 'nonetheless', 'as a result',
    'consequently', 'in summary'
]

def transition_overuse_score(text):
    text_lower = str(text).lower()
    count = sum(text_lower.count(w) for w in transition_words)
    return min(1.0, count / 2)

balance_connectors = ['however', 'although', 'while', 'on the other hand',
                       'that said', 'despite', 'even though']

def balanced_structure_score(text):
    text_lower = str(text).lower()
    count = sum(text_lower.count(c) for c in balance_connectors)
    return min(1.0, count / 2)


# =============================================================
# CATEGORY 4: SEMANTIC MARKERS (content specificity/meaning)
# =============================================================

def specificity_score(text):
    text = str(text)
    numbers = len(re.findall(r'\b\d+\b', text))
    words = text.split()
    proper_nouns = sum(1 for i, w in enumerate(words) if i > 0 and w[:1].isupper() and w.isalpha())
    total_specifics = numbers + proper_nouns
    words_count = max(len(words), 1)
    specificity_ratio = total_specifics / words_count
    return max(0, 1 - min(1, specificity_ratio * 15))


# =============================================================
# COMBINED SCORE
# =============================================================

def score_ai_likelihood(text, word_freq=None):
    lexical = 0.6 * formulaic_phrase_score(text) + 0.4 * vocabulary_diversity_score(text)

    syntactic = (0.4 * sentence_uniformity_score(text) +
                 0.3 * grammatical_formality_score(text) +
                 0.3 * punctuation_cleanliness_score(text))

    structural = 0.5 * transition_overuse_score(text) + 0.5 * balanced_structure_score(text)

    semantic = specificity_score(text)

    final_score = (
        0.45 * lexical +
        0.50 * syntactic +
        0.05 * structural +
        0.00 * semantic
    )
    return round(min(1.0, final_score), 3)


def score_breakdown(text):
    return {
        'lexical_score': round(0.6 * formulaic_phrase_score(text) + 0.4 * vocabulary_diversity_score(text), 3),
        'syntactic_score': round(0.4 * sentence_uniformity_score(text) +
                                  0.3 * grammatical_formality_score(text) +
                                  0.3 * punctuation_cleanliness_score(text), 3),
        'structural_score': round(0.5 * transition_overuse_score(text) + 0.5 * balanced_structure_score(text), 3),
        'semantic_score': round(specificity_score(text), 3),
        'final_score': score_ai_likelihood(text)
    }


if __name__ == "__main__":
    df = pd.read_csv('Data/cleaned_reviews.csv')
    df['ai_likelihood_score'] = df['text'].apply(lambda x: score_ai_likelihood(x))

    output = df[['review_id', 'ai_likelihood_score']]
    output.to_csv('Data/ai_scores.csv', index=False)

    print("Saved to Data/ai_scores.csv")
    print("\nScore distribution:")
    print(df['ai_likelihood_score'].describe())