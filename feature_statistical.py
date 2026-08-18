import pandas as pd
from textblob import TextBlob
import textstat

# Load cleaned reviews
df = pd.read_csv('Data/cleaned_reviews.csv')

# Sentiment score (-1 to 1)
df['sentiment_score'] = df['text'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)

# Review length (word count)
df['review_length'] = df['text'].apply(lambda x: len(str(x).split()))

# Exclamation mark count and ratio
df['exclamation_count'] = df['text'].apply(lambda x: str(x).count('!'))
df['exclamation_ratio'] = df['exclamation_count'] / df['review_length'].replace(0, 1)

# Capital letter ratio (ALL CAPS usage)
def capital_ratio(text):
    text = str(text)
    letters = [c for c in text if c.isalpha()]
    if len(letters) == 0:
        return 0
    caps = [c for c in letters if c.isupper()]
    return len(caps) / len(letters)

df['capital_ratio'] = df['text'].apply(capital_ratio)

# Average word length
df['avg_word_length'] = df['text'].apply(
    lambda x: sum(len(w) for w in str(x).split()) / len(str(x).split()) if len(str(x).split()) > 0 else 0
)

# Readability score (Flesch reading ease)
df['readability_score'] = df['text'].apply(lambda x: textstat.flesch_reading_ease(str(x)))

# Save only the relevant columns (review_id + new features)
feature_cols = ['review_id', 'sentiment_score', 'review_length', 'exclamation_count',
                 'exclamation_ratio', 'capital_ratio', 'avg_word_length', 'readability_score']
df[feature_cols].to_csv('Data/features_statistical.csv', index=False)

print("Saved to Data/features_statistical.csv")
print("\nComparison — deceptive vs truthful (averages):")
print(df.groupby('deceptive')[['sentiment_score', 'review_length', 'exclamation_ratio',
                                 'capital_ratio', 'avg_word_length', 'readability_score']].mean())