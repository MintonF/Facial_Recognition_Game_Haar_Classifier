FROM tiangolo/opencv-python:3.4.17-python3.8

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Set LD_LIBRARY_PATH to include potential locations of libGL.so.1
ENV LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:/usr/local/lib:/opt/mesa/lib:$LD_LIBRARY_PATH"

CMD ["/bin/sh", "-c", "python -m gunicorn --bind 0.0.0.0:8080 AppV27AccuracyAndRemovalFalsePositivesV10Rev6:app"]