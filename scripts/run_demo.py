"""Train two difficulty classifiers on fictional rescue data, entirely offline.

Added for the public repository. This is not the 2025 capstone experiment or
the WWW 2024 hybrid model. Real operational records are never read here.
"""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, auc, f1_score, precision_recall_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


FEATURES = [
    "travel_km",
    "quantity_units",
    "rain_mm",
    "prior_donor_rescues",
    "prior_recipient_rescues",
    "volunteer_experience_days",
]


def binary_labels(labels: pd.Series, task: str) -> pd.Series:
    """Map expert-style labels without modifying the original label series."""
    if task not in {"easy", "hard"}:
        raise ValueError("task must be 'easy' or 'hard'")
    if labels.isna().any() or not labels.isin([-1, 0, 1]).all():
        raise ValueError("labels must be explicit -1 (easy), 0 (other), or 1 (hard)")
    positive_label = -1 if task == "easy" else 1
    return labels.eq(positive_label).astype(int)


def generate_synthetic_rescues(n_rows: int = 1000, seed: int = 42) -> pd.DataFrame:
    """Create artificial features and noisy labels from an invented rule."""
    if n_rows < 100:
        raise ValueError("n_rows must be at least 100 for train/validation/test splits")
    rng = np.random.default_rng(seed)
    data = pd.DataFrame(
        {
            "rescue_id": [f"synthetic_{i:05d}" for i in range(n_rows)],
            "travel_km": rng.gamma(2.0, 4.0, n_rows),
            "quantity_units": rng.integers(1, 61, n_rows),
            "rain_mm": rng.exponential(1.5, n_rows),
            "prior_donor_rescues": rng.integers(0, 100, n_rows),
            "prior_recipient_rescues": rng.integers(0, 100, n_rows),
            "volunteer_experience_days": rng.integers(0, 730, n_rows),
        }
    )
    artificial_difficulty = (
        0.12 * data["travel_km"]
        + 0.03 * data["quantity_units"]
        + 0.20 * data["rain_mm"]
        - 0.012 * data["prior_donor_rescues"]
        - 0.008 * data["prior_recipient_rescues"]
        - 0.001 * data["volunteer_experience_days"]
        + rng.normal(0.0, 0.9, n_rows)
    )
    data["label"] = np.select(
        [artificial_difficulty < -0.1, artificial_difficulty > 1.6],
        [-1, 1],
        default=0,
    ).astype(int)
    # Illustrate missing-value handling. Targets were generated before this step.
    for feature in ["rain_mm", "prior_donor_rescues"]:
        data.loc[rng.random(n_rows) < 0.03, feature] = np.nan
    return data


def split_data(data: pd.DataFrame, seed: int = 42) -> dict[str, pd.DataFrame]:
    """Keep each rescue in one split, stratifying on the three original labels."""
    required = set(FEATURES + ["rescue_id", "label"])
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")
    if data["rescue_id"].isna().any() or data["rescue_id"].duplicated().any():
        raise ValueError("rescue IDs must be present and unique")
    binary_labels(data["label"], "easy")
    if set(data["label"].unique()) != {-1, 0, 1}:
        raise ValueError("all three original label classes are required")
    if data["label"].value_counts().min() < 10:
        raise ValueError("each original label class needs at least 10 rows")
    train_val, test = train_test_split(
        data, test_size=0.2, random_state=seed, stratify=data["label"]
    )
    train, validation = train_test_split(
        train_val,
        test_size=0.25,
        random_state=seed,
        stratify=train_val["label"],
    )
    return {"train": train.copy(), "validation": validation.copy(), "test": test.copy()}


def select_threshold(y_true: pd.Series, probabilities: np.ndarray) -> float:
    """Choose the highest-F1 threshold using validation labels only."""
    thresholds = np.linspace(0.05, 0.95, 91)
    scores = [
        f1_score(y_true, probabilities >= threshold, zero_division=0)
        for threshold in thresholds
    ]
    return float(thresholds[int(np.argmax(scores))])


def evaluate(y_true: pd.Series, probabilities: np.ndarray, threshold: float) -> dict:
    predictions = probabilities >= threshold
    precision, recall, _ = precision_recall_curve(y_true, probabilities)
    return {
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "pr_auc": float(auc(recall, precision)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "positive_prevalence": float(y_true.mean()),
        "threshold": threshold,
        "n_rows": int(len(y_true)),
    }


def run_demo(n_rows: int = 1000, seed: int = 42) -> tuple[dict, pd.DataFrame]:
    data = generate_synthetic_rescues(n_rows, seed)
    splits = split_data(data, seed)
    report = {
        "data_source": "synthetic_demo",
        "notice": "Fictional data; these scores are not food-rescue research results.",
        "seed": seed,
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "features": FEATURES,
        "split_sizes": {name: len(frame) for name, frame in splits.items()},
        "results": {},
    }
    for task in ["easy", "hard"]:
        models = {
            "logistic_regression": LogisticRegression(
                max_iter=1000, class_weight="balanced", random_state=seed
            ),
            "random_forest": RandomForestClassifier(
                n_estimators=150,
                max_depth=8,
                min_samples_leaf=3,
                class_weight="balanced",
                random_state=seed,
                n_jobs=1,
            ),
        }
        report["results"][task] = {}
        for name, model in models.items():
            pipeline = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("model", model),
                ]
            )
            train = splits["train"]
            pipeline.fit(train[FEATURES], binary_labels(train["label"], task))
            validation = splits["validation"]
            y_validation = binary_labels(validation["label"], task)
            validation_probabilities = pipeline.predict_proba(validation[FEATURES])[:, 1]
            threshold = select_threshold(y_validation, validation_probabilities)
            test = splits["test"]
            test_probabilities = pipeline.predict_proba(test[FEATURES])[:, 1]
            report["results"][task][name] = {
                "validation": evaluate(y_validation, validation_probabilities, threshold),
                "test": evaluate(binary_labels(test["label"], task), test_probabilities, threshold),
            }
    manifest = pd.concat(
        [frame[["rescue_id"]].assign(split=name) for name, frame in splits.items()],
        ignore_index=True,
    )
    return report, manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/demo"))
    args = parser.parse_args()
    try:
        report, manifest = run_demo(args.rows, args.seed)
    except ValueError as exc:
        parser.error(str(exc))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    rows = [
        {"data_source": "synthetic_demo", "task": task, "model": name, "split": split, **metrics}
        for task, models in report["results"].items()
        for name, evaluations in models.items()
        for split, metrics in evaluations.items()
    ]
    pd.DataFrame(rows).to_csv(args.output_dir / "metrics.csv", index=False)
    manifest.to_csv(args.output_dir / "split_manifest.csv", index=False)
    print("SYNTHETIC DEMO: fictional data, not capstone or publication results.")
    print(pd.DataFrame(rows).query("split == 'test'")[["task", "model", "roc_auc", "f1"]].to_string(index=False))
    print(f"Outputs written to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
