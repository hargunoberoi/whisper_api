FROM python:3.10-slim

WORKDIR /app
COPY . .

RUN apt update && apt install -y git

RUN apt-get update && apt-get install -y ffmpeg
RUN pip install --upgrade pip && pip install -r requirements.txt

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

ENV PYTHONUNBUFFERED=1