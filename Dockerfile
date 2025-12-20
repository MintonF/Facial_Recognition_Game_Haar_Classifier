FROM tiangolo/opencv-python:3.4.17-python3.8

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["/bin/sh", "-c", "python -m gunicorn --bind 0.0.0.0:8080 AppV27AccuracyAndRemovalFalsePositivesV10Rev6:app"]


