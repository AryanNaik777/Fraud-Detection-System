# Fraud-Detection-System
This code builds a fraud detection system using a credit card dataset. It applies Isolation Forest (unsupervised) and Logistic Regression (supervised) to detect fraudulent transactions, evaluates their performance with metrics like accuracy, recall, F1-score, and visualizes results using confusion matrices.
# Fraud Detection using Isolation Forest and Logistic Regression
CODE
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset (ensure you downloaded 'creditcard.csv')
data = pd.read_csv('creditcard.csv')

# Explore class distribution
print("Original Data Class Distribution:")
print(data['Class'].value_counts())

# Separate features and target
X = data.drop(['Time', 'Class'], axis=1)
y = data['Class']

# Normalize Amount feature
from sklearn.preprocessing import StandardScaler
X['Amount'] = StandardScaler().fit_transform(X['Amount'].values.reshape(-1, 1))

# Split data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ===============================
# MODEL 1: Isolation Forest (unsupervised)
# ===============================
print("\nTraining Isolation Forest...")
iso_forest = IsolationForest(n_estimators=100, contamination=0.001, random_state=42)
y_pred_if = iso_forest.fit_predict(X_test)

# Convert -1,1 to 1,0 (fraud, normal)
y_pred_if = [1 if x == -1 else 0 for x in y_pred_if]

print("\n=== Isolation Forest Results ===")
print(confusion_matrix(y_test, y_pred_if))
print(classification_report(y_test, y_pred_if))

# ===============================
# MODEL 2: Logistic Regression (supervised)
# ===============================
print("\nTraining Logistic Regression...")
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

print("\n=== Logistic Regression Results ===")
print(confusion_matrix(y_test, y_pred_lr))
print(classification_report(y_test, y_pred_lr))

# ===============================
# Visualize Confusion Matrices
# ===============================
fig, ax = plt.subplots(1, 2, figsize=(14, 6))
sns.heatmap(confusion_matrix(y_test, y_pred_if), annot=True, fmt='d', cmap='Blues', ax=ax[0])
ax[0].set_title("Isolation Forest Confusion Matrix")
sns.heatmap(confusion_matrix(y_test, y_pred_lr), annot=True, fmt='d', cmap='Greens', ax=ax[1])
ax[1].set_title("Logistic Regression Confusion Matrix")
plt.show()
