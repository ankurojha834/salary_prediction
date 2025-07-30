import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv("adult 3.csv")

# Clean up
data['workclass'] = data['workclass'].replace({'?': 'Others'})
data['occupation'] = data['occupation'].replace({'?': 'Others'})
data = data[~data['workclass'].isin(['Without-pay', 'Never-worked'])]
data = data[(data['age'] <= 75) & (data['age'] >= 17)]
data = data[(data['educational-num'] <= 16) & (data['educational-num'] >= 5)]
data = data.drop(columns=['education'])

# Encode with separate encoders
encoders = {}
for col in ['workclass', 'marital-status', 'occupation', 'relationship', 'gender', 'native-country']:
    enc = LabelEncoder()
    data[col] = enc.fit_transform(data[col])
    encoders[col] = enc  # Save encoder for that column

# Save encoders dict
joblib.dump(encoders, 'label_encoders.pkl')

# Train model
X = data.drop('income', axis=1)
y = data['income']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model
joblib.dump(model, 'salary_model.pkl')

from sklearn.metrics import accuracy_score

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Random Forest Model Accuracy: {acc:.4f}")

