import re
import pickle
from textblob import TextBlob
import textstat
import spacy
from ai_text_detector import score_ai_likelihood

nlp = spacy.load('en_core_web_sm')

# Load your trained ML model + vectorizer
with open('Data/logistic_model.pkl', 'rb') as f:
    ml_model = pickle.load(f)
with open('tfidf_vectorizer.pkl', 'rb') as f:
    tfidf = pickle.load(f)


# =============================================================
# STEP 1: Statistical / linguistic red flags
# =============================================================
def statistical_flags(text):
    words = text.split()
    word_count = len(words)
    exclamations = text.count('!')

    lower_words = [w.lower() for w in words]
    repeated_ratio = 1 - (len(set(lower_words)) / len(lower_words)) if lower_words else 0

    superlatives = ['amazing', 'best', 'perfect', 'incredible', 'awesome',
                     'love', 'changed my life', 'life changing', '5 stars']
    text_lower = text.lower()
    superlative_hits = sum(text_lower.count(s) for s in superlatives)

    flags = 0
    if exclamations >= 3:
        flags += 1
    if repeated_ratio > 0.15:
        flags += 1
    if word_count < 25:
        flags += 1
    if superlative_hits >= 3:
        flags += 1

    return {
        'exclamations': exclamations,
        'repeated_ratio': round(repeated_ratio, 3),
        'word_count': word_count,
        'superlative_hits': superlative_hits,
        'flags': flags  # out of 4
    }


# =============================================================
# STEP 2: Sentiment extremity (TextBlob)
# =============================================================
def sentiment_flags(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity

    flags = 0
    if abs(polarity) > 0.7:       # uniformly extreme, no nuance
        flags += 1
    if subjectivity > 0.85:       # almost pure opinion, no facts
        flags += 1

    return {'polarity': round(polarity, 3), 'subjectivity': round(subjectivity, 3), 'flags': flags}


# =============================================================
# STEP 3: Readability
# =============================================================
def readability_flags(text):
    try:
        grade = textstat.flesch_kincaid_grade(text)
        ease = textstat.flesch_reading_ease(text)
    except Exception:
        grade, ease = 5, 70

    flags = 1 if grade < 3 else 0  # overly simplistic
    return {'grade': round(grade, 2), 'ease': round(ease, 2), 'flags': flags}


# =============================================================
# STEP 4: Generic / templated phrase detection
# =============================================================
generic_phrases = [
    'best purchase ever', 'buy it now', 'changed my life', 'you won\'t regret it',
    '5 stars all the way', 'highly recommend', 'best product ever',
    'exceeded my expectations', 'exceeded our expectations'
]

def phrase_flags(text):
    text_lower = text.lower()
    hits = sum(text_lower.count(p) for p in generic_phrases)
    return {'generic_phrase_hits': hits, 'flags': 1 if hits >= 1 else 0}


# =============================================================
# STEP 5: POS balance (adjectives/exclamations vs nouns describing specifics)
# =============================================================
def pos_flags(text):
    doc = nlp(text)
    total = max(len(doc), 1)
    adj = sum(1 for t in doc if t.pos_ == 'ADJ')
    noun = sum(1 for t in doc if t.pos_ == 'NOUN')

    adj_ratio = adj / total
    noun_ratio = noun / total

    flags = 1 if (adj_ratio > 0.15 and noun_ratio < 0.10) else 0
    return {'adj_ratio': round(adj_ratio, 3), 'noun_ratio': round(noun_ratio, 3), 'flags': flags}


# =============================================================
# STEP 6: AI-generated heuristic (from ai_text_detector.py)
# =============================================================
def ai_heuristic_flags(text):
    score = score_ai_likelihood(text, None)
    flags = 1 if score > 0.5 else 0
    return {'ai_score': score, 'flags': flags}


# =============================================================
# STEP 7: ML model prediction
# =============================================================
def ml_model_prediction(cleaned_text):
    vec = tfidf.transform([cleaned_text])
    prob = ml_model.predict_proba(vec)[0][1]  # probability of "fake"
    return prob


def clean_text(text):
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return ' '.join(words)


# =============================================================
# FINAL COMBINED ANALYSIS
# =============================================================
def analyze_review_enhanced(text):
    stat = statistical_flags(text)
    sent = sentiment_flags(text)
    read = readability_flags(text)
    phrase = phrase_flags(text)
    pos = pos_flags(text)
    ai = ai_heuristic_flags(text)

    # Total rule-based red flags out of 10 possible
    total_flags = (stat['flags'] + sent['flags'] + read['flags'] +
                   phrase['flags'] + pos['flags'] + ai['flags'])
    max_flags = 4 + 2 + 1 + 1 + 1 + 1  # = 10
    rule_based_score = total_flags / max_flags

    # ML model score
    cleaned = clean_text(text)
    ml_score = ml_model_prediction(cleaned)

    # Blend: rule-based heuristics (60%) + ML model (40%)
    # Rule-based gets more weight since your dataset is small (1600 rows)
    # and obvious red-flag patterns are highly reliable signals
    final_fake_score = round((0.6 * rule_based_score + 0.4 * ml_score), 3)

    ai_score = ai['flags'] and ai_heuristic_flags(text)['ai_score'] or score_ai_likelihood(text, None)

    if final_fake_score > 0.55 and ai_score > 0.5:
        verdict = "Likely Fake (AI-generated)"
    elif final_fake_score > 0.55:
        verdict = "Likely Fake (human-written)"
    else:
        verdict = "Likely Real"

    return {
        'fake_probability': final_fake_score,
        'ai_likelihood_score': ai_score,
        'verdict': verdict,
        'rule_based_score': round(rule_based_score, 3),
        'ml_score': round(ml_score, 3),
        'total_red_flags': f"{total_flags}/{max_flags}",
        'breakdown': {
            'statistical': stat,
            'sentiment': sent,
            'readability': read,
            'generic_phrases': phrase,
            'pos_tagging': pos,
            'ai_heuristic': ai
        }
    }


if __name__ == "__main__":
    fake_example = ("This product is absolutely amazing!!! Best purchase ever!!! "
                     "I love love love it so much, changed my life completely, "
                     "5 stars all the way, buy it now you won't regret it!!!")

    genuine_example = ("Bought this for my home office setup. Sound quality is decent "
                        "for the price but bass is a bit weak. Battery lasted about 6 "
                        "hours on a full charge, less than advertised. Would recommend "
                        "for casual use, not for audiophiles.")

    for label, text in [("FAKE EXAMPLE", fake_example), ("GENUINE EXAMPLE", genuine_example)]:
        print("=" * 70)
        print(label)
        print("=" * 70)
        result = analyze_review_enhanced(text)
        for k, v in result.items():
            print(f"{k}: {v}")
        print()