from pathlib import Path
import subprocess


def test_docker_files_exist():
    root = Path(__file__).resolve().parent.parent

    assert (root / "Dockerfile").exists()
    assert (root / "requirements.txt").exists()
    assert (root / "README.md").exists()
    assert (root / "models" / "model.joblib").exists()


def test_local_pipeline():
    root = Path(__file__).resolve().parent.parent
    input_file = root / "input" / "test.csv"
    example_file = root / "examples" / "test.csv"

    if not input_file.exists():
        input_file.write_text(example_file.read_text())

    result = subprocess.run(
        ["python", "src/run_pipeline.py"],
        cwd=root,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert (root / "output" / "sample_submission.csv").exists()
    assert (root / "output" / "feature_importances.json").exists()
    assert (root / "output" / "score_density.png").exists()
