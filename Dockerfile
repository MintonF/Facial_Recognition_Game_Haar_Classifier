FROM python:3.11-slim-buster

# Set the working directory
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Set environment variables (optional, but good practice)
ENV FLASK_APP=AppV27AccuracyAndRemovalFalsePositivesV10Rev6.py
ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_RUN_PORT=8000

# Expose port 8000 (or whatever port your Flask app uses)
EXPOSE 8000

# Use gunicorn for production
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "AppV27AccuracyAndRemovalFalsePositivesV10Rev6:app", "--workers", "3", "--timeout", "120"]


