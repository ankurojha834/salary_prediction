import joblib

model = joblib.load("salary_model.pkl")
print("Expected feature names:")
print(model.feature_names_in_.tolist())
