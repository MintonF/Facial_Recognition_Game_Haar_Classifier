FROM python:3.11-slim-buster

WORKDIR /app

COPY . /app

RUN apt-get update && apt-get install -y libopencv-dev
RUN apt-get clean && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir -r requirements.txt

# Install development dependencies for linting, static analysis, and testing
RUN pip install --no-cache-dir flake8 pylint pytest

# Run linters
RUN flake8 .

# Run static analysis
RUN pylint AppV27AccuracyAndRemovalFalsePositivesV10Rev6.py

# Run tests
RUN pytest

EXPOSE 5000

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "AppV27AccuracyAndRemovalFalsePositivesV10Rev6:app"]

