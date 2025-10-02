FROM nvcr.io/nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04 AS builder

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
        ninja-build \
        pkg-config \
        python3 \
        xz-utils \
        zlib1g-dev && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /opt/build
RUN curl -sSf -L -O https://github.com/llvm/llvm-project/releases/download/llvmorg-7.1.0/llvm-7.1.0.src.tar.xz && \
    tar -xf llvm-7.1.0.src.tar.xz && \
    cd llvm-7.1.0.src && \
    mkdir build && cd build && \
    ARCH=$(dpkg --print-architecture) && \
    if [ "$ARCH" = "amd64" ]; then \
        TARGETS="X86;NVPTX"; \
    else \
        TARGETS="AArch64;NVPTX"; \
    fi && \
    cmake -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DLLVM_TARGETS_TO_BUILD="$TARGETS" \
        -DLLVM_BUILD_LLVM_DYLIB=ON \
        -DLLVM_LINK_LLVM_DYLIB=ON \
        -DLLVM_ENABLE_ASSERTIONS=OFF \
        -DLLVM_ENABLE_BINDINGS=OFF \
        -DLLVM_INCLUDE_EXAMPLES=OFF \
        -DLLVM_INCLUDE_TESTS=OFF \
        -DLLVM_INCLUDE_BENCHMARKS=OFF \
        -DLLVM_ENABLE_ZLIB=ON \
        -DLLVM_ENABLE_TERMINFO=ON \
        -DCMAKE_INSTALL_PREFIX=/opt/llvm-7 \
        .. && \
    ninja -j"$(nproc)" && \
    ninja install && \
    cd /opt && rm -rf /opt/build

RUN curl -sSf -L https://sh.rustup.rs | bash -s -- -y --profile minimal && \
    /root/.cargo/bin/rustup component add rustfmt clippy && \
    rm -rf /root/.cargo/registry /root/.cargo/git /root/.rustup/tmp
ENV PATH="/root/.cargo/bin:${PATH}"

FROM nvcr.io/nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04 AS runtime

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

COPY --from=builder /opt/llvm-7 /opt/llvm-7
RUN ln -sf /opt/llvm-7/bin/llvm-config /usr/bin/llvm-config && \
    ln -sf /opt/llvm-7/bin/llvm-config /usr/bin/llvm-config-7

COPY --from=builder /root/.cargo /root/.cargo
COPY --from=builder /root/.rustup /root/.rustup
ENV PATH="/root/.cargo/bin:${PATH}"
RUN /root/.cargo/bin/rustup show && \
    rm -rf /root/.cargo/registry /root/.cargo/git /root/.rustup/tmp

WORKDIR /data/rust-cuda
ENV LD_LIBRARY_PATH="/usr/local/cuda/nvvm/lib64:${LD_LIBRARY_PATH}"
ENV LLVM_LINK_STATIC=1
ENV RUST_LOG=info

CMD ["bash"]
