import joblib
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Load the model
model = joblib.load(r'C:\Users\KIIT\Desktop\PROJECT\Phase4\classifier.pkl')

# Load test data
X_test = np.load(r'C:\Users\KIIT\Desktop\PROJECT\Phase3\X_test_selected.npy')
y_test = np.load(r'C:\Users\KIIT\Desktop\PROJECT\Phase3\y_test.npy', allow_pickle=True)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate performance
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')

# Check if model supports probability predictions
if hasattr(model, "predict_proba"):
    y_pred_proba = model.predict_proba(X_test)[:, 1]  # Assuming binary classification
    roc_auc = roc_auc_score(y_test, y_pred_proba)
else:
    roc_auc = "N/A"

# Print evaluation metrics
print(f"Test Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"ROC-AUC Score: {roc_auc}")
