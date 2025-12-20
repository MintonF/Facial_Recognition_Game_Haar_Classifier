
FROM public.ecr.aws/docker/python:3.8

WORKDIR /app

# Install pkg-config
RUN apt-get update && apt-get install -y pkg-config

# Copy libGL.so.1 to /usr/lib and /opt/lib
RUN apt-get update && apt-get install -y --no-install-recommends libgl1-mesa-glx libgl1 libglapi-mesa libgtk2.0-0 ffmpeg
RUN mkdir -p /opt/lib
RUN cp /usr/lib/x86_64-linux-gnu/libGL.so.1 /opt/lib/
RUN cp /usr/lib/x86_64-linux-gnu/libGL.so.1 /usr/lib/

# Create a symbolic link if necessary
RUN if [ ! -f /usr/lib/libGL.so ]; then ln -s /usr/lib/libGL.so.1 /usr/lib/libGL.so; fi

# Set LD_LIBRARY_PATH *before* installing dependencies
ENV LD_LIBRARY_PATH="/opt/lib:/usr/lib:${LD_LIBRARY_PATH}"

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["/bin/sh", "-c", "python -m gunicorn --bind 0.0.0.0:8080 AppV27AccuracyAndRemovalFalsePositivesV10Rev6:app"]

