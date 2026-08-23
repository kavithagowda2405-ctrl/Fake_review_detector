import pandas as pd

df = pd.read_csv('Data/final_features.csv')

# Check exclamation marks vs deceptive/truthful
df['has_3plus_exclamations'] = df['exclamation_count'] >= 3

print("Exclamation mark pattern (3+ exclamations):")
print(df.groupby('deceptive')['has_3plus_exclamations'].mean() * 100)

print("\n" + "="*50)

# Check your AI phrase bank against actual dataset
from ai_text_detector import formulaic_phrase_score

df['has_formulaic_phrase'] = df['text'].apply(lambda x: formulaic_phrase_score(x) > 0)

print("\nFormulaic AI-style phrase presence:")
print(df.groupby('deceptive')['has_formulaic_phrase'].mean() * 100)

print("\n" + "="*50)

# Check repeated word ratio
print("\nAverage repeated word ratio:")
print(df.groupby('deceptive')['repeated_word_ratio'].mean())

print("\n" + "="*50)

# Check sentiment extremity
print("\nAverage sentiment score:")
print(df.groupby('deceptive')['sentiment_score'].mean())