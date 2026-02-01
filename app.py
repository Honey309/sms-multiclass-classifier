from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)


model, vectorizer = pickle.load(open("models/realtime_model.pkl", "rb"))

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    message = data.get("message", "")

    X = vectorizer.transform([message])
    probs = model.predict_proba(X)[0]
    classes = model.classes_

    confidence = float(np.max(probs))
    prediction = classes[np.argmax(probs)]

    if confidence < 0.60:
        prediction = "Uncertain"

    return jsonify({
        "message": message,
        "prediction": prediction,
        "confidence": round(confidence * 100, 2)
    })

@app.route("/")
def home():
    return "SMS Multi-Class Classification API is running"

if __name__ == "__main__":
    app.run(debug=True)
