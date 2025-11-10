# Insurance-Fraud-Detection


## 📄 Project Overview

This project focuses on **detecting fraudulent insurance claims** using machine learning.  
Insurance fraud is a major issue for financial institutions, leading to significant losses each year.  
The goal of this project is to **analyze claim data**, identify patterns related to fraudulent activity, and build predictive models that can help insurance companies flag suspicious claims for further investigation.

---

## 📊 Dataset Information

- **Dataset Source:** [Kaggle – Insurance Fraud Detection Dataset](https://www.kaggle.com/datasets/arpan129/insurance-fraud-detection?resource=download)  
- **Number of Records:** 1,000  
- **Number of Features:** 39  
- **Data Types:**  
  - 18 numeric  
  - 21 categorical  
  - 2 datetime  

### **Target Variable:**  
`fraud_reported` → “Y” for fraudulent claims and “N” for legitimate claims.

---

## 🧠 Project Objectives

1. **Perform Exploratory Data Analysis (EDA)**  
   - Understand distributions, relationships, and correlations between features.  
   - Visualize patterns using Seaborn and Matplotlib.

2. **Data Preprocessing & Feature Engineering**  
   - Handle missing values and encode categorical features.  
   - Convert datetime columns to meaningful numeric features (e.g., days since policy binding).  
   - Drop irrelevant or unique identifier columns.  

3. **Model Development**  
   - Train two classification models:
     - Logistic Regression (baseline)
     - Random Forest Classifier (ensemble)
   - Handle class imbalance using `class_weight='balanced'`.  

4. **Model Evaluation**  
   - Evaluate models using metrics beyond accuracy:
     - Precision, Recall, F1-score, ROC-AUC  
   - Plot Confusion Matrix and ROC Curves.  

5. **Frontend Deployment**  
   - Create an interactive **Streamlit web app** allowing users to:
     - Upload claim data as CSV  
     - Automatically preprocess and predict fraud likelihood  
     - Download prediction results  

---

## ⚙️ Tech Stack

| Category | Tools / Libraries |
|-----------|-------------------|
| **Language** | Python 3.10+ |
| **Data Processing** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn |
| **Machine Learning** | Scikit-learn, imbalanced-learn |
| **Model Persistence** | Joblib |
| **Frontend / UI** | Streamlit |

---
