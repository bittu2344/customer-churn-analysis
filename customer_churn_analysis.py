# ============================================================
# PROJECT 1: Customer Churn Prediction & Analysis
# Author: Bittu Kumar | Data Analyst
# Tools: Python | Pandas | Scikit-learn | Seaborn | Matplotlib
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix,
    classification_report, roc_auc_score, roc_curve
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────
# STEP 1: LOAD DATA
# ─────────────────────────────────────────
# Dataset: Telco Customer Churn (Kaggle)
# Download from: https://www.kaggle.com/datasets/blastchar/telco-customer-churn
# Save as 'telco_churn.csv' in the same folder

print("=" * 60)
print("  CUSTOMER CHURN PREDICTION & ANALYSIS")
print("=" * 60)

df = pd.read_csv('telco_churn.csv')
print(f"\n✅ Data Loaded: {df.shape[0]} rows, {df.shape[1]} columns")
print(df.head())


# ─────────────────────────────────────────
# STEP 2: DATA CLEANING
# ─────────────────────────────────────────
print("\n📋 Missing Values:\n", df.isnull().sum()[df.isnull().sum() > 0])

# Fix TotalCharges column (has spaces as nulls)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

# Drop customerID (not useful for prediction)
df.drop(columns=['customerID'], inplace=True)

# Encode binary columns
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# Label encode remaining object columns
le = LabelEncoder()
cat_cols = df.select_dtypes(include='object').columns
for col in cat_cols:
    df[col] = le.fit_transform(df[col])

print(f"\n✅ Data Cleaned. Shape: {df.shape}")


# ─────────────────────────────────────────
# STEP 3: EXPLORATORY DATA ANALYSIS (EDA)
# ─────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Customer Churn - EDA Overview', fontsize=16, fontweight='bold')

# Plot 1: Churn Distribution
churn_counts = df['Churn'].value_counts()
axes[0, 0].pie(churn_counts, labels=['Not Churned', 'Churned'],
               autopct='%1.1f%%', colors=['#2ecc71', '#e74c3c'], startangle=90)
axes[0, 0].set_title('Churn Distribution')

# Plot 2: Monthly Charges vs Churn
axes[0, 1].boxplot([df[df['Churn'] == 0]['MonthlyCharges'],
                    df[df['Churn'] == 1]['MonthlyCharges']],
                   labels=['Not Churned', 'Churned'])
axes[0, 1].set_title('Monthly Charges vs Churn')
axes[0, 1].set_ylabel('Monthly Charges ($)')

# Plot 3: Tenure vs Churn
axes[1, 0].hist([df[df['Churn'] == 0]['tenure'],
                 df[df['Churn'] == 1]['tenure']],
                label=['Not Churned', 'Churned'], bins=20,
                color=['#3498db', '#e74c3c'], alpha=0.7)
axes[1, 0].set_title('Tenure Distribution by Churn')
axes[1, 0].set_xlabel('Tenure (months)')
axes[1, 0].legend()

# Plot 4: Correlation Heatmap (top features)
top_features = ['MonthlyCharges', 'tenure', 'TotalCharges',
                'Contract', 'PaymentMethod', 'Churn']
corr = df[top_features].corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm',
            ax=axes[1, 1], linewidths=0.5)
axes[1, 1].set_title('Correlation Heatmap')

plt.tight_layout()
plt.savefig('churn_eda.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n✅ EDA plots saved as 'churn_eda.png'")


# ─────────────────────────────────────────
# STEP 4: MODEL BUILDING
# ─────────────────────────────────────────
X = df.drop(columns=['Churn'])
y = df['Churn']

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

# Logistic Regression
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_sc, y_train)
y_pred = model.predict(X_test_sc)
y_prob = model.predict_proba(X_test_sc)[:, 1]

print("\n" + "=" * 50)
print("  MODEL PERFORMANCE")
print("=" * 50)
print(f"  Accuracy  : {accuracy_score(y_test, y_pred):.2%}")
print(f"  ROC-AUC   : {roc_auc_score(y_test, y_prob):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Not Churned', 'Churned']))

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, color='#e74c3c', lw=2,
         label=f'ROC Curve (AUC = {roc_auc_score(y_test, y_prob):.2f})')
plt.plot([0, 1], [0, 1], 'k--', lw=1)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Customer Churn Prediction')
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig('roc_curve.png', dpi=150, bbox_inches='tight')
plt.show()

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Not Churned', 'Churned'],
            yticklabels=['Not Churned', 'Churned'])
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n✅ Model charts saved. Analysis Complete!")
