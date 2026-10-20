
from pathlib import Path
import joblib

MODEL_DIR = Path("models/financial_inclusion")

model_files = [
    "logistic_regression.joblib",
    "random_forest.joblib",
    "xgboost.joblib",
]

for filename in model_files:
    model_path = MODEL_DIR / filename

    print(f"\nChecking: {filename}")

    if not model_path.exists():
        print("ERROR: File not found")
        continue

    try:
        pipeline = joblib.load(model_path)

        if "model" not in pipeline.named_steps:
            print("ERROR: Classifier step not found")
            continue

        classifier = pipeline.named_steps["model"]

        print("Classes:", classifier.classes_.tolist())
        print("Preprocessing present:", "preprocessing" in pipeline.named_steps)
        print("File size:", f"{model_path.stat().st_size:,} bytes")

        expected_classes = [0, 1, 2, 3, 4]
        actual_classes = classifier.classes_.tolist()

        if actual_classes == expected_classes:
            print("Class audit: PASS")
        else:
            print("Class audit: REVIEW REQUIRED")

    except Exception as error:
        print("ERROR while loading model:", repr(error))
