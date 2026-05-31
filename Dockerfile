FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY models ./models
COPY data ./data
COPY examples ./examples
COPY run.sh .
COPY README.md .

RUN mkdir -p input output && chmod +x run.sh

CMD ["./run.sh"]
