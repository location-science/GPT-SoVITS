FROM python:3.10-slim

RUN apt-get update && \
    apt-get install -y build-essential wget ffmpeg unzip && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /src

COPY my_requirements.txt .
COPY download_models.py .
COPY my_app.py .
COPY main.py .
COPY ref_audio ./ref_audio
COPY GPT_SoVITS ./GPT_SoVITS
COPY tools ./tools

# Create a virtual environment at /opt/venv.
RUN python3 -m venv /opt/venv
RUN /opt/venv/bin/pip install --no-cache-dir --upgrade pip 
RUN /opt/venv/bin/pip install --no-cache-dir -r my_requirements.txt
ENV PATH="/opt/venv/bin:${PATH}"

# Download models
RUN python /src/download_models.py
RUN unzip /src/GPT_SoVITS/text/G2PWModel/G2PWModel_1.1.zip -d /src/GPT_SoVITS/text/G2PWModel/ && \
    rm GPT_SoVITS/text/G2PWModel/G2PWModel_1.1.zip
RUN mkdir -p /src/GPT_SoVITS/pretrained_models/fast_langdetect && \
    wget -O /src/GPT_SoVITS/pretrained_models/fast_langdetect/lid.176.bin \
    https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin


EXPOSE 9503
CMD ["uvicorn", "my_app:app", "--host=0.0.0.0", "--port=9503"]


