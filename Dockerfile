FROM public.ecr.aws/docker/python:3.8

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

RUN apt-get update && apt-get install -y --no-install-recommends libgl1-mesa-glx libgl1 libglapi-mesa libgtk2.0-0 ffmpeg

# Update the dynamic linker cache
RUN ldconfig

# Set LD_LIBRARY_PATH to include /usr/lib/x86_64-linux-gnu if it exists
RUN if [ -d /usr/lib/x86_64-linux-gnu ]; then echo "/usr/lib/x86_64-linux-gnu" >> /etc/ld.so.conf.d/opencv.conf; ldconfig; fi

COPY . .

CMD ["gunicorn", "--bind", "0.0.0.0:8000", "AppV27AccuracyAndRemovalFalsePositivesV10Rev6:app"]


