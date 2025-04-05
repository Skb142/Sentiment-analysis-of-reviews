import joblib
import numpy as np

# Load the saved model
model = joblib.load(r'C:\Users\KIIT\Desktop\PROJECT\Phase4\classifier.pkl')

# Load sample test data
X_test_sample = np.load(r'C:\Users\KIIT\Desktop\PROJECT\Phase3\X_test_selected.npy')[:50]

# Make predictions
predictions = model.predict(X_test_sample)

# If the model supports probability predictions, use a threshold-based approach
if hasattr(model, "predict_proba"):
    probabilities = model.predict_proba(X_test_sample)
    positive_prob = probabilities[:, 1]  # Assuming binary classification (negative, positive)
    
    # Define a custom threshold for classification (default 0.5, adjust if needed)
    threshold = 0.55  # Try different values if predictions are biased
    adjusted_predictions = np.where(positive_prob >= threshold, "positive", "negative")
    
    print("\nPredictions with Threshold Adjustment:")
    print(adjusted_predictions)

# Print raw predictions from the model
print("\nRaw Model Predictions:")
print(predictions)
