


# train_model.py (run this once on your local machine)
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
import joblib

# Load and preprocess data
df = pd.read_csv("dataset.csv")
df = df.drop(columns=['LoanID'])

# Label encoding for categorical columns
label_cols = ['Education', 'EmploymentType', 'MaritalStatus', 'HasMortgage', 'HasDependents', 'LoanPurpose', 'HasCoSigner']
le = LabelEncoder()
for col in label_cols:
    df[col] = le.fit_transform(df[col])

# Prepare the features (X) and target (y)
X = df.drop(columns=['Default'])
y = df['Default']

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train the model
model = LogisticRegression()
model.fit(X_scaled, y)

# Save the model and scaler
joblib.dump(model, "logreg_model.pkl")
joblib.dump(scaler, "scaler.pkl")


