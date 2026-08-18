import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from scipy.sparse import hstack, csr_matrix

# Load final features
df = pd.read_csv('Data/final_features.csv')

# Encode target: deceptive -> 1, truthful -> 0
df['label'] = df['deceptive'].map({'deceptive': 1, 'truthful': 0})

# Select numeric feature columns
numeric_features = ['sentiment_score', 'review_length', 'exclamation_count',
                     'exclamation_ratio', 'capital_ratio', 'avg_word_length',
                     'readability_score', 'repeated_word_ratio', 'generic_phrase_count',
                     'adjective_ratio', 'superlative_count']

X_numeric = df[numeric_features].values

# Load the saved TF-IDF vectorizer and transform cleaned_text
with open('tfidf_vectorizer.pkl', 'rb') as f:
    tfidf = pickle.load(f)

X_tfidf = tfidf.transform(df['cleaned_text'].fillna(''))

# Combine numeric features + TF-IDF into one feature matrix
X_numeric_sparse = csr_matrix(X_numeric)
X_combined = hstack([X_numeric_sparse, X_tfidf])

y = df['label'].values

# Train/test split (80/20, stratified, fixed seed)
X_train, X_test, y_train, y_test = train_test_split(
    X_combined, y, test_size=0.2, stratify=y, random_state=42
)

# Save everything as pickle files
with open('Data/X_train.pkl', 'wb') as f:
    pickle.dump(X_train, f)
with open('Data/X_test.pkl', 'wb') as f:
    pickle.dump(X_test, f)
with open('Data/y_train.pkl', 'wb') as f:
    pickle.dump(y_train, f)
with open('Data/y_test.pkl', 'wb') as f:
    pickle.dump(y_test, f)

print("Train/test split complete")
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)
print("\nClass balance in train:", np.bincount(y_train))
print("Class balance in test:", np.bincount(y_test))