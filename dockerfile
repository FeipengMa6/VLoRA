FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel
RUN pip3 install transformers==4.37.2 \
    && pip3 install timm==0.6.13 \
    && pip3 install accelerate==0.21.0 \
    && pip3 install sentencepiece==0.1.99 \
    && pip3 install peft==0.4.0 \
    && pip3 install bitsandbytes==0.41.0 \
    && pip3 install einops-exts==0.0.4 \
    && pip3 install einops==0.6.1 \
    && pip3 install shortuuid \
    && pip3 install openpyxl \
    && pip3 install deepspeed \
    && pip3 install scipy \
    && apt update \
    && apt install -y git \
    && pip3 install packaging \
    && pip3 install ninja \
    && pip3 install flash-attn --no-build-isolation \
    && pip3 install datasets \
    && pip3 install protobuf \
    && pip3 install wandb \
    && pip3 install opencv-python-headless