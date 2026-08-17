import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # remove punctuation/numbers
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return ' '.join(words)

# Load original dataset
df = pd.read_csv('Data/deceptive-opinion.csv')

# Add review_id so all future feature files can be merged correctly
df['review_id'] = df.index

# Clean the text
df['cleaned_text'] = df['text'].apply(clean_text)

# Show before/after examples
print("Sample before/after cleaning:\n")
for i in range(3):
    print("BEFORE:", df['text'].iloc[i][:150])
    print("AFTER: ", df['cleaned_text'].iloc[i][:150])
    print()

# Save
df.to_csv('data/cleaned_reviews.csv', index=False)
print("Saved to data/cleaned_reviews.csv")
print("Shape:", df.shape)