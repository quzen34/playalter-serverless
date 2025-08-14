FROM nvidia/cuda:11.8.0-base-ubuntu22.04

WORKDIR /workspace

RUN apt-get update && apt-get install -y python3-pip git ffmpeg

COPY requirements.txt .
RUN pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118
RUN pip3 install -r requirements.txt

COPY handler.py .

CMD ["python3", "-u", "handler.py"]