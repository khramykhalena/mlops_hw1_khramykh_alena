import json

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from config import (
    MODEL_FILE,
    SCORES_FILE,
    OUTPUT_DIR,
    SUBMISSION_FILE,
    IMPORTANCES_FILE,
    DENSITY_FILE,
)


def save_submission(scores):
    submission = scores[["index", "prediction"]].copy()
    submission.to_csv(SUBMISSION_FILE, index=False)


def save_feature_importances():
    bundle = joblib.load(MODEL_FILE)
    model = bundle["model"]
    feature_names = bundle["feature_names"]

    estimator = model.named_steps["classifier"]

    if hasattr(estimator, "feature_importances_"):
        values = estimator.feature_importances_
    else:
        values = abs(estimator.coef_[0])

    pairs = sorted(zip(feature_names, values), key=lambda x: x[1], reverse=True)[:5]
    result = {name: float(value) for name, value in pairs}

    with open(IMPORTANCES_FILE, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=4)


def save_density_plot(scores):
    plt.figure(figsize=(8, 5))
    scores["prediction"].plot(kind="density")
    plt.title("Density of predicted fraud scores")
    plt.xlabel("Prediction score")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(DENSITY_FILE)
    plt.close()


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    scores = pd.read_csv(SCORES_FILE)

    save_submission(scores)
    save_feature_importances()
    save_density_plot(scores)

    print(f"Submission saved: {SUBMISSION_FILE}")
    print(f"Feature importances saved: {IMPORTANCES_FILE}")
    print(f"Density plot saved: {DENSITY_FILE}")


if __name__ == "__main__":
    main()
