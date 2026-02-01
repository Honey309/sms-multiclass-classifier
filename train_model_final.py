import pandas as pd
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(BASE_DIR, "dataset", "sms_multiclass_balanced.csv")

print("Training with:", csv_path)

data = pd.read_csv(csv_path)

X = data["message"]
y = data["label"]

vectorizer = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 2),
    max_features=5000
)

X_vec = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, test_size=0.2, stratify=y, random_state=42
)

model = MultinomialNB(alpha=0.5)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\n📊 Classification Report:\n")
print(classification_report(y_test, y_pred))

os.makedirs("models", exist_ok=True)
with open("models/realtime_model.pkl", "wb") as f:
    pickle.dump((model, vectorizer), f)

print("✅ Real-time model saved in models/realtime_model.pkl")

