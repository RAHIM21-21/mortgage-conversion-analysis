
# 📈 Conversion Case Study  


---

## 🎯 Objective
This project was developed as part of a technical selection process for a Data Scientist role.  
The main goal was to **identify the most promising leads** from historical mortgage request data, and **recommend which new clients should be contacted** in the upcoming month to maximize conversions.

---

## 📊 Business Context
The company cannot contact all incoming leads. With 40 advisors available and a contact capacity of 100 leads per advisor per week, resource allocation must be optimized.  
The success metric is the **conversion rate**, defined as the proportion of users who successfully obtained a mortgage.

---

## 🔍 Approach
- **Data cleaning & preprocessing** of historical data (`Historical_Data.xlsx`)
- **Exploratory Data Analysis (EDA)** to understand feature distributions and conversion patterns
- **Feature selection** based on correlation, business logic, and predictive value
- **Model training** using:
  - Logistic Regression (with L1 regularization)
  - Random Forest
  - XGBoost
- **Model evaluation** based on precision, recall, F1-score, ROC-AUC, and precision@K
- **SHAP analysis** to interpret model decisions and feature importance
- **Scoring** the new dataset (`Actual_Data.xlsx`) to select the top leads for contact

---

## 🧠 Key Insights
- Conversion was strongly influenced by:
  - Stage in house-buying process (`immobili_visitati`)
  - Contract type
  - Number of applicants and available income
- The best model (XGBoost) was used to assign a probability score to each new lead.
- Leads were ranked and filtered to select the top 16,000 to match contact constraints.

---

## 🛠 Tech Stack
- Python (pandas, numpy, scikit-learn, xgboost, matplotlib, seaborn, shap)
- Jupyter Notebook / .py scripts
- Excel (data format)

---

## 📁 Structure
├── case_study.py # Main analysis script
├── README.md # Project overview (this file)
└── data/ # Folder (excluded) where data was stored during analysis

---

### ⚠️ Disclaimer
This project was completed as part of a technical challenge for evaluation purposes.  
All data has been anonymized and is not shared due to confidentiality.
