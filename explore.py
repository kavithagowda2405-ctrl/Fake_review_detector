import pandas as pd

# Load dataset
df = pd.read_csv('Data/deceptive-opinion.csv')

# Basic info
print("Shape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nFirst 5 rows:")
print(df.head())

# Check class balance (fake vs real)
print("\nClass balance:")
print(df['deceptive'].value_counts())

# Check for missing values
print("\nMissing values:")
print(df.isnull().sum())

# Review length stats
df['review_length'] = df['text'].apply(lambda x: len(str(x).split()))
print("\nReview length by class:")
print(df.groupby('deceptive')['review_length'].describe())