import argparse
import joblib
import json
from pathlib import Path
import numpy as np

MODEL_PATH = Path("artifacts/model.pkl")
METRICS_PATH = Path("artifacts/metrics.json")

def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
    return joblib.load(MODEL_PATH)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Feature list as JSON string. Example: \"[5.1,3.5,1.4,0.2]\"")
    args = parser.parse_args()

    #Parse input
    try:
        features = json.loads(args.input)
    except json.JSONDecodeError:
        raise ValueError("Invalid input. Use JSON list, e.g. --input \"[5.1,3.5,1.4,0.2]\"")

    X = np.array(features).reshape(1, -1)
    model= load_model()
    prediction = model.predict(X)

    print(json.dumps({"prediction": prediction.tolist()}))


if __name__ == "__main__":
    main()