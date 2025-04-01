FROM python:3.10-slim

RUN apt-get update && \
    apt-get install -y build-essential git && \
    rm -rf /var/lib/apt/lists/*

# ARG API_KEY
# ENV TTS_DE_API_KEY=${API_KEY}

WORKDIR /src

COPY my_requirements.txt .
COPY download_models.py .
COPY app.py .
COPY main.py .

# Create a virtual environment at /opt/venv.
RUN python3 -m venv /opt/venv
RUN /opt/venv/bin/pip install --no-cache-dir --upgrade pip 
RUN /opt/venv/bin/pip install --no-cache-dir -r my_requirements.txt
ENV PATH="/opt/venv/bin:${PATH}"

RUN python /src/download_models.py

# RUN git clone https://github.com/modelscope/FunASR.git /src/FunASR
# Sparse clone only a subfolder from a large repo
RUN git clone --filter=blob:none --no-checkout https://github.com/modelscope/FunASR.git /src/FunASR && \
    cd /src/FunASR && \
    git sparse-checkout init --cone && \
    git sparse-checkout set fun_text_processing/text_normalization && \
    git checkout main
# Move the subfolder somewhere else in the image
RUN mv /src/FunASR/fun_text_processing/text_normalization /src/text_normalization && rm -rf /src/FunASR/

# Replace moses_tokenizer which is dependent of nemo package with local file
# RUN sed -i 's/from nemo.collections.common.tokenizers.moses_tokenizers import MosesProcessor/from .moses_tokenizer import MosesProcessor/' /src/FunASR/fun_text_processing/text_normalization/normalize.py
# COPY moses_tokenizer.py /src/FunASR/fun_text_processing/text_normalization/
RUN sed -i 's/from nemo.collections.common.tokenizers.moses_tokenizers import MosesProcessor/from .moses_tokenizer import MosesProcessor/' /src/text_normalization/normalize.py
COPY moses_tokenizer.py /src/text_normalization/

# Replace imports across all .py files
RUN find /src/text_normalization/ -name "*.py" -exec sed -i \
    's/from fun_text_processing\.text_normalization/from text_normalization/g' {} +


# Download GPT_SoVITS model
# Download G2PWModel_1.1
# nltk.download('averaged_perceptron_tagger_eng')

EXPOSE 9503
CMD ["uvicorn", "app:app", "--host=0.0.0.0", "--port=9503"]


