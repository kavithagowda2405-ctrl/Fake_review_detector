"""
train_logistic.py
Trains a Logistic Regression baseline classifier on preprocessed data.
"""

import pickle
from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
)

DATA_DIR = Path("Data")


def load_pickle(filename):
    path = DATA_DIR / filename
    with open(path, "rb") as f:
        return pickle.load(f)


def load_data():
    print("Loading data from Data/ ...")
    X_train = load_pickle("X_train.pkl")
    X_test = load_pickle("X_test.pkl")
    y_train = load_pickle("y_train.pkl")
    y_test = load_pickle("y_test.pkl")
    print(f"  X_train shape: {X_train.shape}")
    print(f"  X_test shape:  {X_test.shape}")
    print(f"  y_train shape: {len(y_train)}")
    print(f"  y_test shape:  {len(y_test)}")
    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train):
    print("\nTraining Logistic Regression (max_iter=1000) ...")
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    print("Training complete.")
    return model


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    print("\n" + "=" * 50)
    print("MODEL EVALUATION METRICS")
    print("=" * 50)
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nDetailed classification report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    print("=" * 50)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix - Logistic Regression")
    plt.tight_layout()
    cm_path = DATA_DIR / "logistic_confusion_matrix.png"
    plt.savefig(cm_path)
    print(f"\nConfusion matrix plot saved to: {cm_path}")


def save_model(model, filename="logistic_model.pkl"):
    DATA_DIR.mkdir(exist_ok=True)
    path = DATA_DIR / filename
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"\nModel saved to: {path}")


def main():
    X_train, X_test, y_train, y_test = load_data()
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)
    save_model(model)


if __name__ == "__main__":
    main()