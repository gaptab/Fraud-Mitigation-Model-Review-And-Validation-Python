import pandas as pd
import numpy as np
import random
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score

# Function to generate dummy fraud data for credit & debit transactions
def generate_fraud_data(n=5000):
    np.random.seed(42)
    
    customer_ids = np.arange(1, 501)  # 500 unique customers
    data = []

    for _ in range(n):
        cust_id = np.random.choice(customer_ids)
        txn_amount = round(np.random.uniform(10, 10000), 2)
        txn_count = np.random.randint(1, 10)  # Transactions per day
        card_type = np.random.choice(["Credit", "Debit"])
        location = np.random.choice(["USA", "UK", "India", "China", "UAE", "Germany"])
        merchant_category = np.random.choice(["Retail", "Gambling", "Electronics", "Groceries", "Restaurants"])
        previous_fraud = np.random.choice([0, 1], p=[0.95, 0.05])  # 5% have a fraud history
        time_of_day = np.random.choice(["Morning", "Afternoon", "Evening", "Night"])

        # Fraud flag based on patterns
        fraud_probability = (
            0.1 if txn_amount > 5000 else  # Large transactions have a 10% fraud rate
            0.05 if merchant_category in ["Gambling", "Electronics"] else
            0.02
        )
        is_fraud = np.random.choice([0, 1], p=[1 - fraud_probability, fraud_probability])

        data.append([cust_id, txn_amount, txn_count, card_type, location, merchant_category, previous_fraud, time_of_day, is_fraud])

    df = pd.DataFrame(data, columns=[
        "CustomerID", "TxnAmount", "TxnCount", "CardType", "Location", 
        "MerchantCategory", "PreviousFraud", "TimeOfDay", "FraudFlag"
    ])
    
    return df

# Generate the dataset
df = generate_fraud_data()
print(df.head())

# Encode categorical variables
df_encoded = pd.get_dummies(df, columns=["CardType", "Location", "MerchantCategory", "TimeOfDay"], drop_first=True)

# Define features (X) and target variable (Y)
X = df_encoded.drop(columns=["FraudFlag", "CustomerID"])
y = df_encoded["FraudFlag"]

# Standardize numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training (70%) and testing (30%)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Train Logistic Regression
log_model = LogisticRegression()
log_model.fit(X_train, y_train)
y_pred_log = log_model.predict(X_test)

# Evaluate Model
print("🔹 Logistic Regression Performance:")
print(classification_report(y_test, y_pred_log))
print("AUC-ROC Score:", roc_auc_score(y_test, y_pred_log))

# Train Decision Tree
dt_model = DecisionTreeClassifier(max_depth=5, random_state=42)
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)

# Evaluate Model
print("🔹 Decision Tree Performance:")
print(classification_report(y_test, y_pred_dt))
print("AUC-ROC Score:", roc_auc_score(y_test, y_pred_dt))

# Train Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

# Evaluate Model
print("🔹 Random Forest Performance:")
print(classification_report(y_test, y_pred_rf))
print("AUC-ROC Score:", roc_auc_score(y_test, y_pred_rf))

# Compare Model Performance
model_performance = pd.DataFrame({
    "Model": ["Logistic Regression", "Decision Tree", "Random Forest"],
    "Accuracy": [accuracy_score(y_test, y_pred_log),
                 accuracy_score(y_test, y_pred_dt),
                 accuracy_score(y_test, y_pred_rf)],
    "AUC-ROC": [roc_auc_score(y_test, y_pred_log),
                roc_auc_score(y_test, y_pred_dt),
                roc_auc_score(y_test, y_pred_rf)]
})

print("🔹 Model Performance Comparison")
print(model_performance)
