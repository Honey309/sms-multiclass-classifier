from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app = Flask(__name__)

# load trained model
with open("models/multiclass_model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    message = request.json["message"]

    probs = model.predict_proba([message])[0]
    classes = model.classes_

    idx = np.argmax(probs)

    return jsonify({
        "prediction": classes[idx],
        "confidence": round(probs[idx] * 100, 2)
    })

if __name__ == "__main__":
    app.run()
