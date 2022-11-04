FROM tensorflow/tensorflow:2.9.0-gpu AS tf-gpu
WORKDIR /ich
COPY requirement.txt .
RUN --mount=type=cache,target=/root/.cache \
    sed -i 's/archive.ubuntu.com/tw.archive.ubuntu.com/g' /etc/apt/sources.list && \
    apt-get update && \
    apt-get install git libopencv-dev -y && \
    /usr/bin/python3 -m pip install --upgrade pip && \
    pip install -r requirement.txt && \
    pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

