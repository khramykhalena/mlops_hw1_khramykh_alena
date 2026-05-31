# Fraud Detector Inference Service

Docker-сервис для инференса ML-модели по данным соревнования. Контейнер читает `test.csv` из папки `input`, применяет препроцессинг и модель, затем сохраняет результат в папку `output`.

## Структура проекта

```text
.
├── Dockerfile
├── README.md
├── requirements.txt
├── run.sh
├── input/
├── output/
├── models/
│   └── model.joblib
├── src/
│   ├── load_data.py
│   ├── preprocess.py
│   ├── score.py
│   ├── save_result.py
│   ├── run_pipeline.py
│   ├── features.py
│   ├── config.py
│   └── train_model.py
├── examples/
│   ├── test.csv
│   └── sample_submission.csv
└── tests/
    └── smoke_test.py
```

## Пайплайн

Сервис выполняет этапы отдельными скриптами:

```text
src/load_data.py      загрузка input/test.csv
src/preprocess.py     препроцессинг данных
src/score.py          скоринг моделью
src/save_result.py    сохранение результата
src/run_pipeline.py   запуск всего пайплайна
```

## Входные данные

На вход подается файл:

```text
input/test.csv
```

Файл должен иметь формат `test.csv` из соревнования.

## Выходные файлы

После запуска в папке `output` создаются:

```text
sample_submission.csv
feature_importances.json
score_density.png
```

`sample_submission.csv` содержит предсказания в формате:

```text
index,prediction
```

`feature_importances.json` содержит топ-5 важных признаков модели.

`score_density.png` содержит график распределения предсказанных скоров.

## Сборка Docker image

В корне проекта выполнить:

```bash
docker build -t fraud_detector .
```

## Запуск контейнера

Перед запуском положить файл `test.csv` в папку `input`.

```bash
docker run --rm \
  -v $(pwd)/input:/app/input \
  -v $(pwd)/output:/app/output \
  fraud_detector
```

После выполнения результаты будут лежать в папке `output`.

## Быстрая проверка

Можно использовать пример файла:

```bash
cp examples/test.csv input/test.csv
```

Затем собрать и запустить контейнер:

```bash
docker build -t fraud_detector .
```

```bash
docker run --rm \
  -v $(pwd)/input:/app/input \
  -v $(pwd)/output:/app/output \
  fraud_detector
```

## Локальный запуск без Docker

```bash
pip install -r requirements.txt
cp examples/test.csv input/test.csv
python src/run_pipeline.py
```

## Модель

В проекте используется заранее обученная модель:

```text
models/model.joblib
```

Инференс выполняется на CPU.
