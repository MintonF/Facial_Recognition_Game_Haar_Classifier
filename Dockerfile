FROM python:3.11-slim-buster

WORKDIR /app

COPY . /app

RUN apt-get update && apt-get install -y libopencv-dev
RUN apt-get clean && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5000

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "AppV27AccuracyAndRemovalFalsePositivesV6:app"]




