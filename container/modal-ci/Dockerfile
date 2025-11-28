FROM ghcr.io/rust-gpu/rust-cuda-ubuntu24-cuda12:latest

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y python3 python3-pip python3-venv && \
    ln -sf /usr/bin/python3 /usr/local/bin/python && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

CMD ["python", "--version"]
