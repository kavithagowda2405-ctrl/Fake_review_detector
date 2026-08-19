import pickle
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Load test data
with open('Data/X_test.pkl', 'rb') as f:
    X_test = pickle.load(f)
with open('Data/y_test.pkl', 'rb') as f:
    y_test = pickle.load(f)

# Load both trained models
with open('Data/logistic_model.pkl', 'rb') as f:
    log_model = pickle.load(f)
with open('Data/rf_model.pkl', 'rb') as f:
    rf_model = pickle.load(f)

def evaluate(model, name):
    preds = model.predict(X_test)
    return {
        'Model': name,
        'Accuracy': accuracy_score(y_test, preds),
        'Precision': precision_score(y_test, preds),
        'Recall': recall_score(y_test, preds),
        'F1-score': f1_score(y_test, preds),
    }, confusion_matrix(y_test, preds)

log_results, log_cm = evaluate(log_model, 'Logistic Regression')
rf_results, rf_cm = evaluate(rf_model, 'Random Forest')

# Comparison table
comparison_df = pd.DataFrame([log_results, rf_results])
print("Model Comparison:\n")
print(comparison_df.to_string(index=False))

print("\nLogistic Regression Confusion Matrix:")
print(log_cm)
print("\nRandom Forest Confusion Matrix:")
print(rf_cm)

# Decide the winner based on F1-score (balances precision & recall)
best_model_name = comparison_df.loc[comparison_df['F1-score'].idxmax(), 'Model']
print(f"\nBest model based on F1-score: {best_model_name}")

# Save comparison table
comparison_df.to_csv('Data/model_comparison.csv', index=False)
print("\nSaved comparison to Data/model_comparison.csv")