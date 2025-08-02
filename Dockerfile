FROM python:3.10.18

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY tests/ tests/
COPY models/ models/
ENV PYTHONPATH=/app
CMD ["python", "src/predict.py"]