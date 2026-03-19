from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os
import json

def main():  
    # Load the dataset
    iris = load_iris()
    X = iris.data
    y = iris.target

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)

    # Save model
    os.makedirs("artifacts", exist_ok=True)
    model_path = os.path.join("artifacts", "model.pkl")
    joblib.dump(model, model_path)

    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Save metrics
    metrics_path = os.path.join("artifacts", "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({"accuracy": accuracy}, f)

    print(f"Saved model to {model_path}")
    print(f"Test accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main()