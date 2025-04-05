# Import Libraries
import pandas as pd
import numpy as np
import time
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib

# Load the dataset 
X_train = np.load(r'C:\Users\KIIT\Desktop\PROJECT\Phase3\X_train_selected.npy')
X_test = np.load(r'C:\Users\KIIT\Desktop\PROJECT\Phase3\X_test_selected.npy')
y_train = np.load(r'C:\Users\KIIT\Desktop\PROJECT\Phase3\y_train.npy', allow_pickle=True)
y_test = np.load(r'C:\Users\KIIT\Desktop\PROJECT\Phase3\y_test.npy', allow_pickle=True)

# Initialize models with class_weight='balanced'
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight='balanced'),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
    "Support Vector Classifier": SVC(probability=True, class_weight='balanced')
}

# Dictionary to hold the scores
scores = {}

# Train and evaluate each model
for model_name, model in models.items():
    start_time = time.time()  # Start timing
    model.fit(X_train, y_train)
    training_time = time.time() - start_time  # Calculate training time
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    
    # Calculate scores
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')
    roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr') if y_pred_proba is not None else None
    
    # Store the scores and training time
    scores[model_name] = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "ROC-AUC Score": roc_auc if roc_auc else "N/A",
        "Training Time (seconds)": training_time
    }

# Print the scores for each model
for model_name, score in scores.items():
    print(f"Model: {model_name}")
    for metric, value in score.items():
        print(f"  {metric}: {value:.4f}" if isinstance(value, float) else f"  {metric}: {value}")
    print()

# Select the best model based on F1 Score
best_model_name = max(scores, key=lambda x: scores[x]['F1 Score'])
best_model = models[best_model_name]

print(f"The best model is: {best_model_name} with an F1 Score of {scores[best_model_name]['F1 Score']:.4f}")

# Save the best model as classifier.pkl
joblib.dump(best_model, r'C:\Users\KIIT\Desktop\PROJECT\Phase4\classifier.pkl')
print(f"The best model has been saved as classifier.pkl")
