FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend.py .
COPY templates/ ./templates/
COPY static/ ./static/

EXPOSE 8000

CMD gunicorn backend:app --timeout 120 --workers 1 --bind 0.0.0.0:$PORT