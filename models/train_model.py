from sklearn.tree import DecisionTreeClassifier
import joblib
import os

# Example training data
X = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [0, 1, 1]]
y = [1, 0, 0, 1, 0]

# Train a simple DecisionTreeClassifier
model = DecisionTreeClassifier()
model.fit(X, y)

# Ensure the models directory exists
os.makedirs('models', exist_ok=True)

# Save the model to the models directory
model_path = 'models/symptom_model.pkl'
joblib.dump(model, model_path)

print(f"Model trained and saved as {model_path}")

