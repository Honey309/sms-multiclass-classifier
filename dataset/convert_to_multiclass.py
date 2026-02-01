import pandas as pd
import os

# Get current folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Path to dataset
csv_path = os.path.join(BASE_DIR, "spam.csv")

print("Looking for file at:", csv_path)

# Load dataset
data = pd.read_csv(csv_path)

# Rename column for consistency (IMPORTANT)
data = data.rename(columns={"Message": "message"})

def classify_message(text):
    text = str(text).lower()

    spam_keywords = ["win", "prize", "lottery", "claim", "urgent", "free"]
    promo_keywords = ["offer", "discount", "sale", "buy", "deal", "cashback"]
    trans_keywords = ["otp", "order", "transaction", "payment", "credited", "debited", "balance"]

    if any(word in text for word in spam_keywords):
        return "Spam"
    elif any(word in text for word in promo_keywords):
        return "Promotional"
    elif any(word in text for word in trans_keywords):
        return "Transactional"
    else:
        return "Promotional"

# Apply classification
data["label"] = data["message"].apply(classify_message)

# Save new dataset
output_path = os.path.join(BASE_DIR, "sms_multiclass.csv")
data[["message", "label"]].to_csv(output_path, index=False)

print("✅ DONE! Multi-class dataset created successfully")
print("📁 Saved at:", output_path)
