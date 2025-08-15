FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /

RUN apt-get update && apt-get install -y python3-pip git ffmpeg wget

RUN pip install runpod==1.7.13

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY handler.py .

CMD ["python3", "-u", "handler.py"]