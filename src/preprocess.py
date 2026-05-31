import pandas as pd

from config import RAW_FILE, PROCESSED_FILE
from features import build_features


def main():
    df = pd.read_csv(RAW_FILE)
    processed = build_features(df)
    processed.to_csv(PROCESSED_FILE, index=False)

    print(f"Processed data saved: {PROCESSED_FILE}")
    print(f"Rows: {len(processed)}")


if __name__ == "__main__":
    main()
