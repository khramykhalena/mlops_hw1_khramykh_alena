from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_DIR = BASE_DIR / "input"
OUTPUT_DIR = BASE_DIR / "output"
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"

INPUT_FILE = INPUT_DIR / "test.csv"
RAW_FILE = DATA_DIR / "raw.csv"
PROCESSED_FILE = DATA_DIR / "processed.csv"
SCORES_FILE = DATA_DIR / "scores.csv"

MODEL_FILE = MODEL_DIR / "model.joblib"

SUBMISSION_FILE = OUTPUT_DIR / "sample_submission.csv"
IMPORTANCES_FILE = OUTPUT_DIR / "feature_importances.json"
DENSITY_FILE = OUTPUT_DIR / "score_density.png"
