FROM tensorflow/tensorflow:2.9.0-gpu AS tf-gpu
WORKDIR /ich
COPY requirement.txt .
RUN --mount=type=cache,target=/root/.cache \
    sed -i 's/archive.ubuntu.com/tw.archive.ubuntu.com/g' /etc/apt/sources.list && \
    apt-get update && \
    apt-get install git libopencv-dev -y --no-install-recommends && \
    /usr/bin/python3 -m pip install --upgrade pip && \
    pip install -r requirement.txt && \
    pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

FROM tf-gpu AS tf-gpu-jupyter
ENV JUPYTER_ENABLE_LAB=yes
RUN --mount=type=cache,target=/root/.cache \
    apt-get update && \
    curl -fsSL https://deb.nodesource.com/setup_18.x | bash - &&\
    apt-get install nodejs -y --no-install-recommends && \
    pip install jupyterlab jupyterlab-lsp nbresuse && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    jupyter-lab --generate-config && \
    jupyter labextension enable && \
    sed -i "s/# c.ServerApp.password = ''/c.ServerApp.password = 'argon2:\$argon2id\$v=19\$m=10240,t=10,p=8\$z2AHb3647NEb9JqeQpGswg\$jwnLW1OAy53k\/Ci5QH+uaN0TutcnnGczJT9VGQVLnyY'/g" /root/.jupyter/jupyter_lab_config.py

CMD ["jupyter-lab", "--notebook-dir=/ich", "--allow-root", "--no-browser", "--ip=0.0.0.0", "--ServerApp.password_required=true"]
EXPOSE 8888


