ARG LLVM_BASE_IMAGE=ghcr.io/rust-gpu/rust-cuda-rockylinux9-llvm7:latest
FROM ${LLVM_BASE_IMAGE} AS toolchain

# Runtime image with CUDA, Python, LLVM 7, and Rust preinstalled
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

# Copy LLVM and Rust from the shared toolchain stage
COPY --from=toolchain /opt/llvm-7 /opt/llvm-7
RUN ln -sf /opt/llvm-7/bin/llvm-config /usr/bin/llvm-config && \
    ln -sf /opt/llvm-7/bin/llvm-config /usr/bin/llvm-config-7

COPY --from=toolchain /root/.cargo /root/.cargo
COPY --from=toolchain /root/.rustup /root/.rustup
ENV PATH="/root/.cargo/bin:${PATH}"
RUN /root/.cargo/bin/rustup show && \
    rm -rf /root/.cargo/registry /root/.cargo/git /root/.rustup/tmp

# Set up workspace environment defaults
WORKDIR /data/rust-cuda
ENV LD_LIBRARY_PATH="/usr/local/cuda/nvvm/lib64:${LD_LIBRARY_PATH}"
ENV LLVM_LINK_STATIC=1
ENV RUST_LOG=info

CMD ["bash"]
