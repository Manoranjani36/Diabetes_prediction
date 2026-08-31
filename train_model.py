import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Load dataset
file_path = "diabetes_prediction_dataset.csv"
df = pd.read_csv(file_path)

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)
print(df.head())

# 2. Basic cleaning
df = df.drop_duplicates()

# Remove accidental index column if present
for col in ["Unnamed: 0", "id", "Id"]:
    if col in df.columns:
        df = df.drop(columns=[col])

# Target column
target = "diabetes"

if target not in df.columns:
    raise ValueError(f"Target column '{target}' not found. Available columns: {df.columns.tolist()}")

X = df.drop(columns=[target])
y = df[target]

# 3. Identify feature types
numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()

# 4. Preprocessing
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_features),
    ("cat", categorical_pipeline, categorical_features)
])

# 5. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# 6. Compare classification algorithms
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, random_state=42, n_jobs=-1
    ),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42)
}

results = {}

for name, model in models.items():
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)
    pred = pipeline.predict(X_test)

    acc = accuracy_score(y_test, pred)
    results[name] = acc

    print("\n" + "=" * 60)
    print(name)
    print("Accuracy:", round(acc * 100, 2), "%")
    print(classification_report(y_test, pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, pred))

# 7. Select best model by accuracy
best_model_name = max(results, key=results.get)
best_model = Pipeline([
    ("preprocessor", preprocessor),
    ("model", models[best_model_name])
])
best_model.fit(X_train, y_train)

# 8. Save model
os.makedirs("models", exist_ok=True)
joblib.dump(best_model, "models/diabetes_model.pkl")

# Save feature information for the app
metadata = {
    "numeric_features": numeric_features,
    "categorical_features": categorical_features,
    "target": target,
    "model_name": best_model_name,
    "results": results
}
joblib.dump(metadata, "models/metadata.pkl")

print("\n" + "=" * 60)
print("PROJECT COMPLETED!")
print("Best model:", best_model_name)
print("Best accuracy:", round(results[best_model_name] * 100, 2), "%")
print("Saved: models/diabetes_model.pkl")
