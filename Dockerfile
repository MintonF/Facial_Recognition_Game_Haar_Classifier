FROM tiangolo/opencv-python:3.4.17-python3.8

WORKDIR /app

# Install OpenGL
RUN apt-get update && apt-get install -y libgl1-mesa-glx

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["gunicorn", "AppV27AccuracyAndRemovalFalsePositivesV10Rev6:app", "--bind", "0.0.0.0:8080", "--workers", "3"]