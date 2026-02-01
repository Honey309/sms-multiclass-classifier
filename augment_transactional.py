import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
input_path = os.path.join(BASE_DIR, "dataset", "sms_multiclass.csv")
output_path = os.path.join(BASE_DIR, "dataset", "sms_multiclass_balanced.csv")

print("Loading:", input_path)

df = pd.read_csv(input_path)

transactional_templates = [
    "Your OTP is {}",
    "Rs {} credited to your account",
    "Rs {} debited from your account",
    "Payment of Rs {} successful",
    "Transaction ID {} completed",
    "Your bank balance is Rs {}",
    "Order {} has been confirmed"
]

augmented = []
for i in range(300):
    text = transactional_templates[i % len(transactional_templates)].format(1000 + i)
    augmented.append({"message": text, "label": "Transactional"})

aug_df = pd.DataFrame(augmented)

final_df = pd.concat([df, aug_df], ignore_index=True)
final_df.to_csv(output_path, index=False)

print("✅ Balanced dataset created")
print(final_df["label"].value_counts())
print("Saved at:", output_path)

