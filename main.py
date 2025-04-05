import joblib
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load the trained model, vectorizer, and feature selector
model = joblib.load(r'C:\Users\KIIT\Desktop\PROJECT\Phase4\classifier.pkl')
vectorizer = joblib.load(r'C:\Users\KIIT\Desktop\PROJECT\Phase3\vectorizer.pkl')
chi2_selector = joblib.load(r'C:\Users\KIIT\Desktop\PROJECT\Phase3\chi2_selector.pkl')

@app.route('/')
def home():
    return "ML Model API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        print("Received Data:", data)  # Debug log

        review = data.get("review", "")
        if not review:
            return jsonify({"error": "No review provided"}), 400

        # Convert review text into TF-IDF features
        review_tfidf = vectorizer.transform([review]).toarray()
        print("TF-IDF Shape:", review_tfidf.shape)  # Debug log

        # Apply Chi-Squared feature selection
        review_selected = chi2_selector.transform(review_tfidf)
        print("Feature Selection Shape:", review_selected.shape)  # Debug log

        # Predict sentiment
        prediction = model.predict(review_selected)[0]
        print("Prediction:", prediction)  # Debug log

        return jsonify({"prediction": prediction})

    except Exception as e:
        print("Error:", str(e))  # Debugging log
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
