# 🛡️ FinShield AI — Intelligent Fraud Detection System

> An end-to-end machine learning project for real-time financial fraud detection in a global fintech environment.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://finshield-ai-cxumiv2ydoq5tdmjxvv4em.streamlit.app/)
![Python](https://img.shields.io/badge/Python-3.10-blue)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📌 Overview

**FinShield AI** is a complete, production-style data science project built for **Zenvra International**, an imaginary global fintech institution. It simulates a full real-world fraud detection pipeline — from synthetic dataset generation through to a live, publicly accessible machine learning web application — covering transactions across **40 countries**, **7 regions**, and **6 payment channels**.

The project demonstrates end-to-end data science competency spanning data engineering, exploratory analysis, machine learning, model explainability, application development, and cloud deployment.

---

## 🌐 Live App

🔗 **[Launch FinShield AI Fraud Detector](https://finshield-ai-cxumiv2ydoq5tdmjxvv4em.streamlit.app/)**

Anyone can access the app directly from their browser — no installation required.

---

##  Project Aims

- Generate a realistic 3,000-row synthetic global transaction dataset with intentional data quality imperfections
- Systematically clean and preprocess the data using industry-standard techniques
- Conduct thorough Exploratory Data Analysis (EDA) to uncover fraud patterns
- Engineer meaningful predictive features to improve model performance
- Train and compare three ML classifiers — Logistic Regression, Random Forest, and XGBoost
- Deploy the best-performing model as an interactive web application accessible globally

---

##  Project Structure
```bash
finshield_ai/
│
├── data/
│   ├── raw/                        # Generated synthetic CSV
│   └── processed/                  # Cleaned and feature-engineered data
│
├── notebooks/
│   └── finshield_ai.ipynb          # Full project notebook (single file)
│
├── models/
│   ├── xgb_fraud_model.pkl         # Trained XGBoost model
│   ├── scaler.pkl                  # Fitted StandardScaler
│   └── feature_columns.json        # Ordered feature column list
│
├── outputs/
│   ├── figures/                    # All EDA and evaluation charts
│   └── reports/                    # Model comparison CSV
│
├── app/
│   ├── app.py                      # Streamlit prediction app
│   ├── requirements.txt            # App dependencies
│   └── Dockerfile                  # Containerization setup
│
└── README.md                       # Project documentation
```
---

---

## 📊 Dataset

The synthetic dataset was generated to simulate real-world fintech transaction activity with the following characteristics:

| Property | Detail |
|---|---|
| Total Rows | 3,000 transactions |
| Countries | 40 (across 7 global regions) |
| Merchant Categories | 14 |
| Transaction Channels | 6 |
| Payment Methods | 6 |
| Fraud Rate | ~28% (post signal strengthening) |
| Time Span | January 2023 — June 2024 |

**Intentional imperfections introduced:**
- ~3% missing values across 5 columns
- 30 duplicate rows (simulating double-posting bugs)
- 50 rows with inconsistent country name casing
- 8 extreme transaction amount outliers (simulated data-entry errors)

---

## 🧹 Data Cleaning

| Step | Method |
|---|---|
| Duplicate removal | `drop_duplicates()` |
| Country casing | `.str.title()` standardisation |
| Missing numerical values | Median imputation |
| Missing categorical values | Mode imputation |
| Outlier capping | 3×IQR upper fence |
| Timestamp parsing | Extracted hour, day, month, quarter |
| Data type enforcement | Integer casting of all flag columns |

---

## 📈 EDA Highlights

Seven visualisations were produced to explore fraud patterns:

-  **Class Distribution** — 90.2% legitimate vs 9.8% fraudulent (original data)

![ClassDistribution](https://github.com/Uzo-Hill/finshield-ai/blob/main/figures/01_class_distribution.png)

---
-  **Fraud by Merchant** — Crypto exchange highest at 22%; utilities lowest at 5.5%

![FraudByMerchant](https://github.com/Uzo-Hill/finshield-ai/blob/main/figures/02_fraud_by_merchant.png)
---
-  **Fraud by Channel and KYC Tier** — ATM (12%) and USSD (11%) most exploited. Unverified accounts show 17% fraud rate

![FraudByChannel](https://github.com/Uzo-Hill/finshield-ai/blob/main/figures/03_fraud_by_channel_kyc.png)
---

-  **Amount Distribution** — Fraud persists across all spending levels

![AmountDistribution](https://github.com/Uzo-Hill/finshield-ai/blob/main/figures/04_amount_distribution.png)

---

-  **Fraud by Hour** — Peak fraud at midnight (22%) and 10pm–11pm (19%)

![FraudByHour](https://github.com/Uzo-Hill/finshield-ai/blob/main/figures/06_fraud_by_hour.png)

---

- 🌍 **Fraud by Region** — Asia leads at ~13.5%; Oceania lowest at ~6.7%

![FraudByRegion](https://github.com/Uzo-Hill/finshield-ai/blob/main/figures/07_fraud_by_region.png)

---

## 🔧 Feature Engineering

| Feature | Description |
|---|---|
| `log_amount` | Log1p transform of transaction amount |
| `log_avg_tx_amt` | Log1p transform of 30-day average amount |
| `log_amount_ratio` | Log1p of amount-to-average ratio |
| `risk_score` | Weighted composite of 8 binary risk signals |
| `kyc_encoded` | Ordinal encoding of KYC tier (0–3) |
| OHE columns (×35) | One-hot encoded channel, merchant, payment, device, region |

Final feature matrix: **3,000 rows × 54 columns**

---

## 🤖 Modelling

### Class Imbalance Handling
SMOTE (Synthetic Minority Over-sampling Technique) was applied exclusively to training data to balance classes without contaminating the test set.

### Model Comparison

| Model | ROC-AUC | Avg Precision | F1 (Fraud) |
|---|---|---|---|
| Logistic Regression | 0.699 | 0.471 | 0.338 |
| Random Forest | 0.705 | 0.470 | 0.399 |
| **XGBoost ✅** | **0.712** | **0.478** | **0.427** |

**XGBoost** was selected as the production model, outperforming all three metrics.

### XGBoost Configuration
```python
XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric='aucpr'
)
```

### Top Features (SHAP)
1. `risk_score` — by far the most influential feature (SHAP = 0.62)
2. `kyc_encoded` — identity verification level
3. `payment_method_debit_card` — payment type signal
4. `payment_method_credit_card`
5. `log_amount_ratio` — deviation from spending average

---

## 🌐 Streamlit Web App

The prediction app accepts **23 transaction-level inputs** across five sections:

| Section | Inputs |
|---|---|
| Transaction Info | Amount, channel, merchant category, payment method |
| Account Info | Device type, KYC tier, region, account age |
| Behavioural History | Transaction count, average amount, failed transactions, login attempts |
| Time & Location | Hour, day, month, distinct countries |
| Risk Flags | New device, night transaction, weekend, IP mismatch, velocity flag |

**Output:** Instant fraud probability score with colour-coded verdict, visual gauge bar, and plain-English risk interpretation.

---

##  Deployment

The app is deployed publicly via **Streamlit Community Cloud** connected to this GitHub repository.

### Run Locally

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/finshield-ai.git
cd finshield-ai

# Install dependencies
pip install -r requirements.txt

# Launch the app
streamlit run app.py
```

### Run with Docker

```bash
docker build -t finshield-app .
docker run -d -p 8501:8501 finshield-app
# Open → http://localhost:8501
```

---

##  Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.10 |
| Data Processing | Pandas, NumPy |
| Visualisation | Matplotlib, Seaborn |
| Machine Learning | Scikit-learn, XGBoost |
| Class Imbalance | imbalanced-learn (SMOTE) |
| Explainability | SHAP |
| Web App | Streamlit |
| Deployment | Streamlit Community Cloud, GitHub |
| Containerisation | Docker |

---

##  Key Files

| File | Description |
|---|---|
| `notebooks/finshield_ai.ipynb` | Complete end-to-end project notebook |
| `app/app.py` | Streamlit prediction application |
| `models/xgb_fraud_model.pkl` | Trained XGBoost model |
| `models/feature_columns.json` | Feature alignment list for inference |
| `outputs/reports/model_comparison.csv` | Model evaluation results |

---

## ⚠️ Limitations & Future Work

- Dataset is synthetic — real-world performance would require validation on actual transaction data
- Model recall for fraud (35%) could be improved by lowering the decision threshold below 0.50
- Future enhancements could include: network graph features, device fingerprinting, real-time streaming inference, and a retraining pipeline
- A larger dataset (10,000+ rows) with richer features would likely push ROC-AUC above 0.85

---

## 👤 Author

**Uzoh C. Hillary**
Data Scientist | Analytics Professional


[![GitHub](https://img.shields.io/badge/GitHub-Profile-black?logo=github)](https://github.com/Uzo-Hill)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin)](https://www.linkedin.com/in/hillaryuzoh/)

---



> *FinShield AI is a portfolio project built on synthetic data for educational and demonstration purposes only.*



















