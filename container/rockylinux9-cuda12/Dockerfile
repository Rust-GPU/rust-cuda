FROM nvcr.io/nvidia/cuda:12.8.1-cudnn-devel-rockylinux9 AS llvm-builder

RUN dnf -y install \
    --nobest \
    --allowerasing \
    --setopt=install_weak_deps=False \
    openssl-devel \
    pkgconfig \
    which \
    xz \
    zlib-devel \
    libffi-devel \
    ncurses-devel \
    libxml2-devel \
    libedit-devel \
    python3 \
    make \
    cmake && \
    dnf clean all

WORKDIR /data/llvm7

# Download and build LLVM 7.1.0 for all architectures.
RUN curl -sSf -L -O https://github.com/llvm/llvm-project/releases/download/llvmorg-7.1.0/llvm-7.1.0.src.tar.xz && \
    tar -xf llvm-7.1.0.src.tar.xz && \
    cd llvm-7.1.0.src && \
    mkdir build && cd build && \
    ARCH=$(uname -m) && \
    if [ "$ARCH" = "x86_64" ]; then \
        TARGETS="X86;NVPTX"; \
    else \
        TARGETS="AArch64;NVPTX"; \
    fi && \
    cmake \
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
    make -j$(nproc) && \
    make install && \
    cd ../.. && \
    rm -rf llvm-7.1.0.src* && \
    dnf clean all

FROM nvcr.io/nvidia/cuda:12.8.1-cudnn-devel-rockylinux9

RUN dnf -y install \
    --nobest \
    --allowerasing \
    --setopt=install_weak_deps=False \
    clang \
    openssl-devel \
    fontconfig-devel \
    libX11-devel \
    libXcursor-devel \
    libXi-devel \
    libXrandr-devel \
    libxml2-devel \
    ncurses-devel \
    pkgconfig \
    which \
    xz \
    zlib-devel \
    cmake && \
    dnf clean all

COPY --from=llvm-builder /opt/llvm-7 /opt/llvm-7
RUN ln -s /opt/llvm-7/bin/llvm-config /usr/bin/llvm-config && \
    ln -s /opt/llvm-7/bin/llvm-config /usr/bin/llvm-config-7

# Get Rust (install rustup; toolchain installed from rust-toolchain.toml below)
RUN curl -sSf -L https://sh.rustup.rs | bash -s -- -y --profile minimal --default-toolchain none
ENV PATH="/root/.cargo/bin:${PATH}"

# Setup the workspace
WORKDIR /data/rust-cuda
RUN --mount=type=bind,source=rust-toolchain.toml,target=/data/rust-cuda/rust-toolchain.toml \
    rustup show

# Add nvvm to LD_LIBRARY_PATH.
ENV LD_LIBRARY_PATH="/usr/local/cuda/nvvm/lib64:${LD_LIBRARY_PATH}"
ENV LLVM_LINK_STATIC=1
ENV RUST_LOG=info
