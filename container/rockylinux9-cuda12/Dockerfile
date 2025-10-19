# Multi-stage build to keep final image slim while still providing CUDA, Python, LLVM 7, and Rust.
# Stage 1: build LLVM and install Rust.
FROM nvcr.io/nvidia/cuda:12.8.1-cudnn-devel-rockylinux9 AS builder

RUN dnf -y update && \
    dnf -y install \
        clang \
        cmake \
        fontconfig-devel \
        libX11-devel \
        libXcursor-devel \
        libXi-devel \
        libXrandr-devel \
        libffi-devel \
        libxml2-devel \
        libedit-devel \
        ncurses-devel \
        openssl-devel \
        pkgconfig \
        python3 \
        redhat-rpm-config \
        which \
        xz \
        zlib-devel \
        make && \
    dnf clean all

# Build LLVM 7.1.0 once and install into /opt/llvm-7
WORKDIR /opt/build
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
    make -j"$(nproc)" && \
    make install && \
    cd /opt && \
    rm -rf /opt/build

# Install Rust toolchain (minimal profile) and clean caches
RUN curl -sSf -L https://sh.rustup.rs | bash -s -- -y --profile minimal && \
    /root/.cargo/bin/rustup component add rustfmt clippy && \
    rm -rf /root/.cargo/registry /root/.cargo/git /root/.rustup/tmp
ENV PATH="/root/.cargo/bin:${PATH}"

# Stage 2: final runtime image
FROM nvcr.io/nvidia/cuda:12.8.1-cudnn-devel-rockylinux9 AS runtime

# Install runtime build dependencies (no compilers for LLVM build needed here)
RUN dnf -y update && \
    dnf -y install \
        clang \
        cmake \
        fontconfig-devel \
        libX11-devel \
        libXcursor-devel \
        libXi-devel \
        libXrandr-devel \
        openssl-devel \
        pkgconfig \
        python3 \
        redhat-rpm-config \
        which \
        xz \
        zlib-devel && \
    dnf clean all

# Copy LLVM and Rust from builder stage
COPY --from=builder /opt/llvm-7 /opt/llvm-7
RUN ln -sf /opt/llvm-7/bin/llvm-config /usr/bin/llvm-config && \
    ln -sf /opt/llvm-7/bin/llvm-config /usr/bin/llvm-config-7

COPY --from=builder /root/.cargo /root/.cargo
COPY --from=builder /root/.rustup /root/.rustup
ENV PATH="/root/.cargo/bin:${PATH}"
RUN /root/.cargo/bin/rustup show && \
    rm -rf /root/.cargo/registry /root/.cargo/git /root/.rustup/tmp

# Set up workspace environment defaults
WORKDIR /data/rust-cuda
ENV LD_LIBRARY_PATH="/usr/local/cuda/nvvm/lib64:${LD_LIBRARY_PATH}"
ENV LLVM_LINK_STATIC=1
ENV RUST_LOG=info

CMD ["bash"]
