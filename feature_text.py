import pandas as pd
import spacy
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

nlp = spacy.load('en_core_web_sm')

df = pd.read_csv('Data/cleaned_reviews.csv')

# --- TF-IDF ---
tfidf = TfidfVectorizer(max_features=100)  # top 100 words to keep it manageable
tfidf_matrix = tfidf.fit_transform(df['cleaned_text'].fillna(''))

# Save the vectorizer for later use in modeling
with open('tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf, f)

# --- Repeated word ratio ---
def repeated_word_ratio(text):
    words = str(text).split()
    if len(words) == 0:
        return 0
    unique_words = set(words)
    return 1 - (len(unique_words) / len(words))  # higher = more repetition

df['repeated_word_ratio'] = df['cleaned_text'].apply(repeated_word_ratio)

# --- Generic phrase detection ---
generic_phrases = ['highly recommend', 'best hotel', 'great experience', 'will definitely',
                    'best ever', 'amazing stay', 'perfect stay', 'exceeded expectation']

def generic_phrase_count(text):
    text = str(text).lower()
    return sum(text.count(phrase) for phrase in generic_phrases)

df['generic_phrase_count'] = df['text'].apply(generic_phrase_count)

# --- POS tagging: adjective ratio + superlative count ---
def pos_features(text):
    doc = nlp(str(text))
    total = len(doc)
    if total == 0:
        return pd.Series([0, 0])
    adj_count = sum(1 for token in doc if token.pos_ == 'ADJ')
    superlative_count = sum(1 for token in doc if token.tag_ == 'JJS')  # e.g. "best", "greatest"
    return pd.Series([adj_count / total, superlative_count])

df[['adjective_ratio', 'superlative_count']] = df['text'].apply(pos_features)

# Save
feature_cols = ['review_id', 'repeated_word_ratio', 'generic_phrase_count',
                 'adjective_ratio', 'superlative_count']
df[feature_cols].to_csv('Data/features_text.csv', index=False)

print("Saved to Data/features_text.csv")
print("\nTop 15 TF-IDF words overall:")
print(tfidf.get_feature_names_out()[:15])