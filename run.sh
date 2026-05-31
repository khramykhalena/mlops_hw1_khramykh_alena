#!/bin/bash
set -e

python src/load_data.py
python src/preprocess.py
python src/score.py
python src/save_result.py
