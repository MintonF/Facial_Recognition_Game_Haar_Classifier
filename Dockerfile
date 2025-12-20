FROM public.ecr.aws/docker/python:3.8

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

RUN apt-get update && apt-get install -y --no-install-recommends libgl1-mesa-glx libgl1 libglapi-mesa libgtk2.0-0 ffmpeg

# Copy libGL.so.1 to /usr/lib
RUN cp /usr/lib/x86_64-linux-gnu/libGL.so.1 /usr/lib/

# Create a symbolic link if necessary
RUN if [ ! -f /usr/lib/libGL.so ]; then ln -s /usr/lib/libGL.so.1 /usr/lib/libGL.so; fi

COPY . .

# Find the location of gunicorn executable inside venv
RUN GUNICORN_PATH=$(python -c "import gunicorn; print(gunicorn.__file__)") && \
    echo "Gunicorn path: $GUNICORN_PATH"

CMD ["/root/.local/bin/gunicorn", "--bind", "0.0.0.0:8000", "AppV27AccuracyAndRemovalFalsePositivesV10Rev6:app"]


