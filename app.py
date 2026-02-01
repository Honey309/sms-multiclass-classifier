from flask import Flask, request, render_template
import pickle
import os
import numpy as np

app = Flask(__name__)

# Load model safely
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "models", "multiclass_model.pkl")

model, vectorizer = pickle.load(open(model_path, "rb"))

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    confidence = None

    if request.method == "POST":
        message = request.form["message"]

        X = vectorizer.transform([message])
        probs = model.predict_proba(X)[0]
        prediction = model.classes_[np.argmax(probs)]
        confidence = round(np.max(probs) * 100, 2)

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence
    )

if __name__ == "__main__":
    app.run(debug=True)

