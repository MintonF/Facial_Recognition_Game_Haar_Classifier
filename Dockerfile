FROM public.ecr.aws/aws-apprunner/python3.11:latest

WORKDIR /app

COPY . /app

RUN apt-get update && apt-get install -y libgl1-mesa-glx
RUN apt-get clean && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8080

CMD ["gunicorn", "--bind", "0.0.0.0:8080", "AppV27AccuracyAndRemovalFalsePositivesV10Rev6:app"]
