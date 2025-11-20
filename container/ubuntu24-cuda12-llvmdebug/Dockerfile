FROM nvcr.io/nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04 AS llvm-builder

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get -qq -y install \
    build-essential \
    clang \
    curl \
    libffi-dev \
    libedit-dev \
    libncurses5-dev \
    libssl-dev \
    libtinfo-dev \
    libxml2-dev \
    cmake \
    ninja-build \
    pkg-config \
    python3 \
    xz-utils \
    zlib1g-dev && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /data/llvm7

# Download and build LLVM 7.1.0 for all architectures.
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
        -DCMAKE_BUILD_TYPE=Debug \
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
    ninja -j$(nproc) && \
    ninja install && \
    cd ../.. && \
    rm -rf llvm-7.1.0.src*

FROM nvcr.io/nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get -qq -y install \
    build-essential \
    clang \
    curl \
    libssl-dev \
    libtinfo-dev \
    pkg-config \
    xz-utils \
    zlib1g-dev \
    cmake \
    libfontconfig-dev \
    libx11-xcb-dev \
    libxcursor-dev \
    libxi-dev \
    libxinerama-dev \
    libxrandr-dev && \
    rm -rf /var/lib/apt/lists/*

COPY --from=llvm-builder /opt/llvm-7 /opt/llvm-7
RUN ln -s /opt/llvm-7/bin/llvm-config /usr/bin/llvm-config && \
    ln -s /opt/llvm-7/bin/llvm-config /usr/bin/llvm-config-7

# Get Rust (install rustup; toolchain installed from rust-toolchain.toml below)
RUN curl -sSf -L https://sh.rustup.rs | bash -s -- -y --profile minimal --default-toolchain none
ENV PATH="/root/.cargo/bin:${PATH}"

# Setup the workspace
WORKDIR /data/Rust-CUDA
RUN --mount=type=bind,source=rust-toolchain.toml,target=/data/Rust-CUDA/rust-toolchain.toml \
    rustup show

# Add nvvm to LD_LIBRARY_PATH.
ENV LD_LIBRARY_PATH="/usr/local/cuda/nvvm/lib64:${LD_LIBRARY_PATH}"
ENV LLVM_LINK_STATIC=1
ENV RUST_LOG=info
