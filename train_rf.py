import pickle
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

with open("Data/X_train.pkl", "rb") as f:
    X_train = pickle.load(f)
with open("Data/X_test.pkl", "rb") as f:
    X_test = pickle.load(f)
with open("Data/y_train.pkl", "rb") as f:
    y_train = pickle.load(f)
with open("Data/y_test.pkl", "rb") as f:
    y_test = pickle.load(f)

print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)

lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train, y_train)
lr_preds = lr.predict(X_test)

def evaluate(name, y_true, y_pred):
    print(f"\n=== {name} ===")
    print(f"Accuracy : {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred, average='weighted'):.4f}")
    print(f"Recall   : {recall_score(y_true, y_pred, average='weighted'):.4f}")
    print(f"F1-score : {f1_score(y_true, y_pred, average='weighted'):.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    print(classification_report(y_true, y_pred))

evaluate("Random Forest", y_test, rf_preds)
evaluate("Logistic Regression", y_test, lr_preds)

importances = rf.feature_importances_
if hasattr(X_train, "columns"):
    feature_names = X_train.columns
else:
    feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]

top_indices = np.argsort(importances)[::-1][:10]
print("\n=== Top 10 Feature Importances (Random Forest) ===")
for rank, idx in enumerate(top_indices, start=1):
    print(f"{rank}. {feature_names[idx]}: {importances[idx]:.4f}")

with open("Data/rf_model.pkl", "wb") as f:
    pickle.dump(rf, f)
print("\nModel saved to Data/rf_model.pkl")