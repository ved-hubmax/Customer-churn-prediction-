# Customer Churn Prediction — Random Forest

Predicting whether a telecom customer will churn (leave the service) using a tuned Random Forest classifier, built on the Telco Customer Churn dataset.

## 📌 Problem Statement

Customer churn directly impacts revenue. This project builds a classification model to flag customers likely to churn, so the business can act early with retention offers before they leave.

## 📊 Dataset

- **Source:** Telco Customer Churn dataset (`data/customer_churn.csv`)
- **Rows:** 7,043 customers
- **Target column:** `Churn` (Yes / No)
- **Features:** demographics (gender, senior citizen, partner, dependents), account info (tenure, contract type, payment method, charges), and subscribed services (phone, internet, streaming, tech support, etc.)

## 🛠️ Tech Stack

- Python 3
- pandas, numpy — data handling
- matplotlib, seaborn — visualization
- scikit-learn — modeling (`RandomForestClassifier`, `GridSearchCV`)
- joblib — model persistence

## 🔄 Workflow

1. **Data cleaning** — convert `TotalCharges` to numeric, fill missing values using `MonthlyCharges × tenure`, drop `customerID`
2. **EDA** — churn distribution, tenure/monthly charges vs churn, contract type vs churn, correlation heatmap
3. **Encoding** — label encode all categorical columns
4. **Modeling** — baseline `RandomForestClassifier`
5. **Comparison** — accuracy across different `n_estimators` values
6. **Tuning** — `GridSearchCV` over `n_estimators`, `max_depth`, `min_samples_split`
7. **Evaluation** — accuracy, classification report, confusion matrix, feature importance
8. **Persistence** — final model saved as `model.pkl`

## 📈 Results

| Model | Accuracy |
|---|---|
| Baseline Random Forest (100 trees) | ~79.5% |
| Tuned Random Forest (GridSearchCV) | ~80.9% |

**Best parameters:** `max_depth=10`, `min_samples_split=10`, `n_estimators=300`

Overall churn rate in the dataset: ~26.5% churned, ~73.5% retained — meaning the dataset is moderately imbalanced, which is reflected in lower recall for the churn ("Yes") class.

## 🖼️ Visualizations

| | |
|---|---|
| ![Churn Distribution](images/01_churn_distribution.png) | ![Tenure vs Churn](images/02_tenure_vs_churn.png) |
| ![Monthly Charges vs Churn](images/03_monthlycharges_vs_churn.png) | ![Contract vs Churn](images/04_contract_vs_churn.png) |
| ![Correlation Heatmap](images/05_correlation_heatmap.png) | ![Accuracy vs Trees](images/06_accuracy_vs_trees.png) |
| ![Confusion Matrix](images/07_confusion_matrix.png) | ![Feature Importance](images/08_feature_importance.png) |

## 🚀 How to Run

```bash
git clone https://github.com/ved-hubmax/HR-Analytics.git
cd HR-Analytics
pip install -r requirements.txt
jupyter notebook 1_model_building_rf.ipynb
```

## 📁 Project Structure

```
HR-Analytics/
├── data/
│   └── customer_churn.csv
├── images/
│   ├── 01_churn_distribution.png
│   ├── 02_tenure_vs_churn.png
│   ├── 03_monthlycharges_vs_churn.png
│   ├── 04_contract_vs_churn.png
│   ├── 05_correlation_heatmap.png
│   ├── 06_accuracy_vs_trees.png
│   ├── 07_confusion_matrix.png
│   └── 08_feature_importance.png
├── 1_model_building_rf.ipynb
├── model.pkl
├── requirements.txt
├── .gitignore
└── README.md
```

## 🔮 Future Improvements

- Try other models (XGBoost, LightGBM) for comparison
- Address class imbalance with SMOTE or class weighting
- Deploy the model behind a simple API (Flask/FastAPI) for real-time predictions
- Add SHAP values for more interpretable feature impact
