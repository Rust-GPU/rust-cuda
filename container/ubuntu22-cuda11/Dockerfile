ARG LLVM_BASE_IMAGE=ghcr.io/rust-gpu/rust-cuda-ubuntu22-llvm7:latest
FROM ${LLVM_BASE_IMAGE} AS toolchain

FROM nvcr.io/nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04 AS runtime

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get -y install \
        build-essential \
        clang \
        cmake \
        curl \
        libffi-dev \
        libfontconfig-dev \
        libssl-dev \
        libtinfo-dev \
        libedit-dev \
        libncurses5-dev \
        libxml2-dev \
        libx11-xcb-dev \
        libxcursor-dev \
        libxi-dev \
        libxinerama-dev \
        libxrandr-dev \
        pkg-config \
        python3 \
        xz-utils \
        zlib1g-dev && \
    rm -rf /var/lib/apt/lists/*

COPY --from=toolchain /opt/llvm-7 /opt/llvm-7
RUN ln -sf /opt/llvm-7/bin/llvm-config /usr/bin/llvm-config && \
    ln -sf /opt/llvm-7/bin/llvm-config /usr/bin/llvm-config-7

COPY --from=toolchain /root/.cargo /root/.cargo
COPY --from=toolchain /root/.rustup /root/.rustup
ENV PATH="/root/.cargo/bin:${PATH}"
RUN /root/.cargo/bin/rustup show && \
    rm -rf /root/.cargo/registry /root/.cargo/git /root/.rustup/tmp

WORKDIR /data/rust-cuda
ENV LD_LIBRARY_PATH="/usr/local/cuda/nvvm/lib64:${LD_LIBRARY_PATH}"
ENV LLVM_LINK_STATIC=1
ENV RUST_LOG=info

CMD ["bash"]
