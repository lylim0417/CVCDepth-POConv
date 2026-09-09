FROM nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.8 \
        python3-pip \
        python3-dev \
        git \
        libglib2.0-0 \
        libgl1 && \
    rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade \
        "pip<24" \
        setuptools \
        wheel

RUN python3 -m pip install \
        torch==1.12.0+cu113 \
        torchvision==0.13.0+cu113 \
        --extra-index-url https://download.pytorch.org/whl/cu113

COPY docker/requirements-minimal.txt /tmp/requirements-minimal.txt

RUN python3 -m pip install \
        -r /tmp/requirements-minimal.txt

WORKDIR /workspace

ENV PYTHONPATH=/workspace:/workspace/external/packnet_sfm

CMD ["python3"]
