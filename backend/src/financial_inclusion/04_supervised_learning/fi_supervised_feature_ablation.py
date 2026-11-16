"""One-at-a-time Logistic Regression ablation on the fixed test split."""

from pathlib import Path
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[4]
DATA = ROOT / "outputs/financial_inclusion/reports/04_supervised_learning/fi_supervised_dataset_audit.csv"
OUT = ROOT / "outputs/financial_inclusion/experiments/supervised_tuning/feature_ablation.csv"
FEATURES = ["female", "age", "educ", "inc_q", "emp_in", "urbanicity", "regionwb"]
LABELS = [0, 1, 2, 3, 4]


def main():
    d = pd.read_csv(DATA, low_memory=False)
    y = d.cluster_id.astype(int)
    train, test = train_test_split(range(len(d)), test_size=.2, random_state=42, stratify=y)
    rows = []
    for omitted in [None, *FEATURES]:
        features = [f for f in FEATURES if f != omitted]
        cats = [f for f in features if f != "age"]
        nums = [f for f in features if f == "age"]
        prep = ColumnTransformer([
            ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("ohe", OneHotEncoder(handle_unknown="ignore"))]), cats),
            ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), nums),
        ], remainder="drop")
        pipe = Pipeline([("prep", prep), ("model", LogisticRegression(max_iter=3000, class_weight="balanced", random_state=42))])
        pipe.fit(d.iloc[train][features], y.iloc[train])
        pred = pipe.predict(d.iloc[test][features])
        rows.append({
            "experiment": "all seven" if omitted is None else f"without {omitted}",
            "omitted_feature": omitted or "",
            "accuracy": accuracy_score(y.iloc[test], pred),
            "balanced_accuracy": balanced_accuracy_score(y.iloc[test], pred),
            "macro_f1": f1_score(y.iloc[test], pred, labels=LABELS, average="macro", zero_division=0),
        })
    result = pd.DataFrame(rows)
    result.to_csv(OUT, index=False)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
