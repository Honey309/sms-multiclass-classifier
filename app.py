from flask import Flask, request, render_template
import pickle
import os
import numpy as np

app = Flask(__name__)

# ===============================
# Load model and vectorizer
# ===============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "models", "multiclass_model.pkl")

with open(model_path, "rb") as f:
    model, vectorizer = pickle.load(f)

# ===============================
# Rule-based transactional keywords
# ===============================
transaction_keywords = [
    "otp", "debited", "credited", "transaction", "upi",
    "payment", "successful", "balance", "account",
    "bill", "recharge", "order", "delivered",
    "pnr", "booking", "refund", "debit", "credit"
]

# ===============================
# Home route (UI + Prediction)
# ===============================
@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    confidence = None
    message = ""

    if request.method == "POST":
        message = request.form.get("message", "")
        msg_lower = message.lower()

        # ---- RULE BASED TRANSACTIONAL ----
        if any(word in msg_lower for word in transaction_keywords):
            prediction = "transactional"
            confidence = 99.0

        # ---- ML BASED (Spam / Promotional / Ham) ----
        else:
            X = vectorizer.transform([message])
            probs = model.predict_proba(X)[0]
            prediction = model.classes_[np.argmax(probs)]
            confidence = round(float(np.max(probs)) * 100, 2)

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        message=message
    )

# ===============================
# Run app
# ===============================
if __name__ == "__main__":
    app.run(debug=True)
