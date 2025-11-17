FROM nvcr.io/nvidia/cuda:12.8.1-cudnn-devel-rockylinux9

RUN dnf -y update && \
    dnf -y install \
    clang \
    openssl-devel \
    pkgconfig \
    redhat-rpm-config \
    which \
    xz \
    zlib-devel && \
    dnf clean all

# Needed to build `path_tracer`, `optix/ex03_window` example
RUN dnf -y install \
    cmake \
    fontconfig-devel \
    libX11-devel  \
    libXcursor-devel \
    libXi-devel \
    libXrandr-devel && \
    dnf clean all

# Get LLVM 7
WORKDIR /data/llvm7

# Install dependencies for building LLVM
RUN dnf -y install epel-release && \
    dnf -y install \
    libffi-devel \
    ncurses-devel \
    libxml2-devel \
    libedit-devel \
    python3 \
    make && \
    dnf clean all

# Download and build LLVM 7.1.0 for all architectures
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
        -DCMAKE_INSTALL_PREFIX=/usr \
        .. && \
    make -j$(nproc) && \
    make install && \
    cd ../.. && \
    rm -rf llvm-7.1.0.src* && \
    ln -s /usr/bin/llvm-config /usr/bin/llvm-config-7 && \
    dnf clean all

# Get Rust
RUN curl -sSf -L https://sh.rustup.rs | bash -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Setup the workspace
WORKDIR /data/rust-cuda
RUN --mount=type=bind,source=rust-toolchain.toml,target=/data/rust-cuda/rust-toolchain.toml \
    rustup show

# Add nvvm to LD_LIBRARY_PATH.
ENV LD_LIBRARY_PATH="/usr/local/cuda/nvvm/lib64:${LD_LIBRARY_PATH}"
ENV LLVM_LINK_STATIC=1
ENV RUST_LOG=info

RUN dnf install -y https://repo.radeon.com/amdgpu-install/6.4.3/rhel/9.6/amdgpu-install-6.4.60403-1.el9.noarch.rpm && \
    dnf install -y https://dl.fedoraproject.org/pub/epel/epel-release-latest-9.noarch.rpm && \
    dnf install -y dnf-plugin-config-manager && \
    crb enable && \
    dnf install -y python3-setuptools python3-wheel ninja-build perl-core perl-devel openssl-devel && \
    dnf clean all

RUN \
    for ver in 6.4.3 6.4; do \
    printf '[ROCm-%s]\nname=ROCm%s\nbaseurl=https://repo.radeon.com/rocm/el9/%s/main\nenabled=1\npriority=50\ngpgcheck=1\ngpgkey=https://repo.radeon.com/rocm/rocm.gpg.key\n' "$ver" "$ver" "$ver" >> /etc/yum.repos.d/rocm.repo; \
    done && \
    dnf install rocm6.4.3 -y && \
    dnf clean all

# Add rocm libraries to LD_LIBRARY_PATH.
ENV LD_LIBRARY_PATH="/opt/rocm/lib:${LD_LIBRARY_PATH}"
# Add rocm binaries to PATH.
ENV PATH="/opt/rocm/bin:${PATH}"