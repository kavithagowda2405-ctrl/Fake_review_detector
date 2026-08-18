import pandas as pd

df_clean = pd.read_csv('Data/cleaned_reviews.csv')
df_stat = pd.read_csv('Data/features_statistical.csv')
df_text = pd.read_csv('Data/features_text.csv')

df_final = df_clean.merge(df_stat, on='review_id', how='inner')
df_final = df_final.merge(df_text, on='review_id', how='inner')

print("Final shape:", df_final.shape)
print("\nColumns:", df_final.columns.tolist())
print("\nMissing values:\n", df_final.isnull().sum())
print("\nClass balance:\n", df_final['deceptive'].value_counts())

df_final.to_csv('Data/final_features.csv', index=False)
print("\nSaved to Data/final_features.csv")