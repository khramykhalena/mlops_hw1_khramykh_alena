import argparse

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from features import build_features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-path", default="input/train.csv")
    parser.add_argument("--model-path", default="models/model.joblib")
    args = parser.parse_args()

    df = pd.read_csv(args.train_path)
    y = df["target"]
    x = build_features(df.drop(columns=["target"]))

    categorical_features = [
        "merch",
        "cat_id",
        "gender",
        "us_state",
        "jobs",
        "full_name",
    ]

    numeric_features = [col for col in x.columns if col not in categorical_features]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
            ]), categorical_features),
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
            ]), numeric_features),
        ]
    )

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", ExtraTreesClassifier(
            n_estimators=120,
            max_depth=18,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )),
    ])

    model.fit(x, y)

    feature_names = categorical_features + numeric_features

    bundle = {
        "model": model,
        "feature_names": feature_names,
        "categorical_features": categorical_features,
        "numeric_features": numeric_features,
    }

    joblib.dump(bundle, args.model_path)
    print(f"Model saved to {args.model_path}")


if __name__ == "__main__":
    main()
