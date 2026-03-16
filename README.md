# 📊 Customer Churn Prediction & Analysis

**Author:** Bittu Kumar | Data Analyst
**Tools:** Python | Pandas | Scikit-learn | Seaborn | Matplotlib | Power BI

## 🎯 Project Overview
Analyzed a 7,000+ row telecom dataset to identify key churn drivers and built a Logistic Regression model achieving **~80% accuracy** to predict at-risk customers.

## 🔍 Key Findings
- Contract type, monthly charges, and customer tenure are top churn drivers
- Customers on month-to-month contracts churn 3x more than annual customers
- Higher monthly charges (>$65) significantly increase churn probability

## ⚙️ How to Run
1. Download dataset from Kaggle - Telco Customer Churn
2. Save as `telco_churn.csv` in this folder
3. Install: `pip install pandas numpy matplotlib seaborn scikit-learn`
4. Run: `python customer_churn_analysis.py`

## 📈 Results
| Metric | Score |
|--------|-------|
| Accuracy | ~80% |
| ROC-AUC | ~0.85 |
| Precision (Churn) | ~66% |
| Recall (Churn) | ~55% |
