import joblib
import pandas as pd

from config import MODEL_FILE, PROCESSED_FILE, SCORES_FILE


def main():
    bundle = joblib.load(MODEL_FILE)
    model = bundle["model"]

    data = pd.read_csv(PROCESSED_FILE)

    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(data)[:, 1]
    else:
        scores = model.predict(data)

    result = pd.DataFrame({
        "index": range(len(scores)),
        "prediction": scores
    })

    result.to_csv(SCORES_FILE, index=False)

    print(f"Scores saved: {SCORES_FILE}")


if __name__ == "__main__":
    main()
