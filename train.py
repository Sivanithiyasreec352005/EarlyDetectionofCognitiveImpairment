import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE  # <-- ADD THIS

# ===========================
# 1. Load Dataset
# ===========================
df = pd.read_csv("dataset.csv")

# Separate features and target
X = df.drop(columns=["target"])
y = df["target"]

# ===========================
# 2. Encode Categorical Responses
# ===========================
likert_map = {
    "Never / No difficulty": 1,
    "Rarely": 2,
    "Sometimes": 3,
    "Often": 4,
    "Always / Severe difficulty": 5
}

X_encoded = X.applymap(lambda x: likert_map[x])

# Encode target labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)  # Normal=0, Impaired=1

# ===========================
# 3. Train/Test Split
# ===========================
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# ===========================
# 4. Apply SMOTE to Balance Dataset
# ===========================
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

print(f"Before SMOTE: {np.bincount(y_train)}")
print(f"After SMOTE:  {np.bincount(y_train_balanced)}")

# ===========================
# 5. Train Naive Bayes Classifier
# ===========================
model = GaussianNB()
model.fit(X_train_balanced, y_train_balanced)

# ===========================
# 6. Evaluate Model
# ===========================
y_pred = model.predict(X_test)

print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

# ===========================
# 7. Plots
# ===========================
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Class distribution plot (before SMOTE)
plt.figure(figsize=(5,4))
sns.countplot(x=y, palette="Set2")
plt.title("Class Distribution (Original Dataset)")
plt.show()

# ===========================
# 8. Save Model
# ===========================
with open("cognitive_nb_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("\n✅ Model saved as cognitive_nb_model.pkl")
