import shutil

from config import INPUT_FILE, RAW_FILE, DATA_DIR


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError("Put test.csv into the input directory")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(INPUT_FILE, RAW_FILE)

    print(f"Loaded file: {INPUT_FILE}")
    print(f"Saved raw data: {RAW_FILE}")


if __name__ == "__main__":
    main()
