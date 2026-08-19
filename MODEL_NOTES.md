# Model Comparison — Fake Review Detector

## Results

| Metric    | Logistic Regression | Random Forest |
|-----------|---------------------|----------------|
| Accuracy  | 76.25%               | 70.94%         |
| Precision | 78.00%               | 73.10%         |
| Recall    | 73.13%               | 66.25%         |
| F1-score  | 75.48%               | 69.51%         |

## Selected Model: Logistic Regression

Logistic Regression outperformed Random Forest across every metric.

### Why Logistic Regression performed better:

1. **Dataset size**: With only 1600 reviews (1280 training samples), the dataset
   is relatively small. Random Forest typically needs more data to fully
   leverage its ability to model complex, non-linear relationships — with
   limited data it tends to overfit to noise in the training set instead of
   learning generalizable patterns.

2. **High-dimensional sparse features**: Our feature set includes 100 TF-IDF
   word features (sparse, mostly zeros) combined with 11 dense numeric
   features. Linear models like Logistic Regression handle sparse
   high-dimensional data well, since the decision boundary just needs to
   weight each word/feature's contribution — which is close to how TF-IDF
   features naturally behave. Random Forest, by contrast, struggles to find
   good splits across thousands of mostly-zero TF-IDF columns.

3. **Confusion matrix comparison**: Logistic Regression made fewer errors
   overall (76 misclassifications vs 93 for Random Forest), and specifically
   missed fewer deceptive reviews (43 false negatives vs 54), which matters
   more for this use case — failing to catch a fake review is a bigger
   problem than being slightly too cautious.

## Decision

**Logistic Regression is the model used in the Flask web application (Day 7).**
Model saved at: `Data/logistic_model.pkl`

## Future improvement ideas
- Try hyperparameter tuning (GridSearchCV) on both models before finalizing
- Test with more training data if available
- Consider a fine-tuned transformer (DistilBERT) for higher accuracy if time
  permits beyond the 10-day scope