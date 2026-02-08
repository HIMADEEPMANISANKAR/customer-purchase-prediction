import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

# Get absolute path to project
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

data_path = os.path.join(BASE_DIR, "..", "data", "raw_data.csv")

# Load dataset
data = pd.read_csv(data_path)

# Split features & target
X = data.drop("Target", axis=1)
y = data["Target"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Evaluate
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Save model
models_path = os.path.join(BASE_DIR, "..", "models")
os.makedirs(models_path, exist_ok=True)

joblib.dump(model, os.path.join(models_path, "purchase_model.pkl"))
