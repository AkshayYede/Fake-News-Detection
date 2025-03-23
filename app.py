import joblib
import numpy as np
from flask import Flask, request, render_template

# Initialize Flask app
app = Flask(__name__)

# Load the saved model and vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    text = request.form["news_text"]
    text_vectorized = vectorizer.transform([text])  # Transform text using the saved vectorizer
    prediction = model.predict(text_vectorized)  # Predict
    prob = model.predict_proba(text_vectorized)  # Get probability

    label = "Fake News" if prediction[0] == 1 else "Real News"
    confidence = np.max(prob) * 100  # Get the highest confidence score

    return render_template("index.html", prediction_text=f"Prediction: {label} (Confidence: {confidence:.2f}%)")

if __name__ == "__main__":
    app.run(debug=True)
