FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD gunicorn backend:app --timeout 120 --workers 1 --bind 0.0.0.0:$PORT