{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    # LLVM 7 is no longer carried by nixpkgs-unstable. Pin a second nixpkgs just
    # for `llvmPackages_7` so someone else's compat patches do the hard work.
    nixpkgs-llvm7.url = "github:NixOS/nixpkgs/nixos-23.05";
    rust-overlay.url = "github:oxalica/rust-overlay";
    rust-overlay.inputs.nixpkgs.follows = "nixpkgs";
  };

  outputs = { nixpkgs, nixpkgs-llvm7, rust-overlay, ... }:
    let
      system = "x86_64-linux";
      # allowUnfree is required because CUDA is unfree.
      pkgs = import nixpkgs {
        inherit system;
        config.allowUnfree = true;
        overlays = [ rust-overlay.overlays.default ];
      };
      pkgsLlvm7 = import nixpkgs-llvm7 { inherit system; };
      lib = pkgs.lib;

      # ---- CUDA toolkit (Nix-managed) ----
      # The NVIDIA **driver** (libcuda.so.1, libnvidia-*) still comes from the
      # host — apt on Debian, hardware.nvidia on NixOS. Nix only provides the
      # **toolkit** (nvcc, libnvvm, cudart, headers).
      #
      # Toolkit pin chooses what PTX version NVVM emits, which then dictates
      # the minimum host driver version at runtime:
      #   CUDA 13.2 → NVVM 22.0 → PTX 9.2 → needs driver 580.x+ (CUDA 13)
      #   CUDA 12.9 → NVVM 21.x → PTX 8.x → runs on CUDA 12.x drivers
      # `cudatoolkit` is the kitchen-sink symlinkJoin maintained by nixpkgs —
      # every header path and lib layout is already wired correctly.
      cuda19Root = pkgs.cudaPackages_13_2.cudatoolkit;
      cuda7Root = pkgs.cudaPackages_12_9.cudatoolkit;

      driverLibDir = "/usr/lib/x86_64-linux-gnu";

      toolchain = pkgs.rust-bin.fromRustupToolchainFile ./rust-toolchain.toml;

      # ---- LLVM 19 (from current nixpkgs) ----
      llvm19 = pkgs.llvmPackages_19;
      llvm19Bin = lib.getBin llvm19.llvm;
      llvm19Dev = lib.getDev llvm19.llvm;
      llvm19CompatTools = pkgs.symlinkJoin {
        name = "llvm19-compat-tools";
        paths = [
          (pkgs.writeShellScriptBin "opt-19" ''exec ${llvm19Bin}/bin/opt "$@"'')
          (pkgs.writeShellScriptBin "llvm-as-19" ''exec ${llvm19Bin}/bin/llvm-as "$@"'')
          (pkgs.writeShellScriptBin "llvm-dis-19" ''exec ${llvm19Bin}/bin/llvm-dis "$@"'')
          (pkgs.writeShellScriptBin "llc-19" ''exec ${llvm19Bin}/bin/llc "$@"'')
        ];
      };

      # ---- LLVM 7.1.0 (from pinned nixos-23.05 nixpkgs) ----
      llvm7Pkg = pkgsLlvm7.llvmPackages_7.llvm;
      llvm7Bin = pkgsLlvm7.lib.getBin llvm7Pkg;
      llvm7Dev = pkgsLlvm7.lib.getDev llvm7Pkg;
      llvm7CompatTools = pkgs.symlinkJoin {
        name = "llvm7-compat-tools";
        paths = [
          (pkgs.writeShellScriptBin "llvm-config-7" ''exec ${llvm7Dev}/bin/llvm-config "$@"'')
          (pkgs.writeShellScriptBin "llvm-as-7" ''exec ${llvm7Bin}/bin/llvm-as "$@"'')
          (pkgs.writeShellScriptBin "llvm-dis-7" ''exec ${llvm7Bin}/bin/llvm-dis "$@"'')
          (pkgs.writeShellScriptBin "llc-7" ''exec ${llvm7Bin}/bin/llc "$@"'')
          (pkgs.writeShellScriptBin "opt-7" ''exec ${llvm7Bin}/bin/opt "$@"'')
        ];
      };

      # ---- Shared bits across both shells ----
      commonNativeInputs = [
        toolchain
        pkgs.pkg-config
        pkgs.cmake
        pkgs.ninja
      ];
      # The v19 shell uses unstable's runtime libs (modern glibc). The v7 shell has
      # to match LLVM 7's glibc generation (23.05), otherwise ncurses/libstdc++ from
      # unstable demand GLIBC_2.38+ symbols LLVM 7's linked glibc 2.37 doesn't have.
      v19BuildInputs = [
        pkgs.openssl
        pkgs.libxml2
        pkgs.zlib
        pkgs.ncurses
        pkgs.stdenv.cc.cc.lib
      ];
      v7BuildInputs = [
        pkgsLlvm7.openssl
        pkgsLlvm7.libxml2
        pkgsLlvm7.zlib
        pkgsLlvm7.ncurses
        pkgsLlvm7.stdenv.cc.cc.lib
      ];
      mkCudaEnv = root: {
        CUDA_HOME = "${root}";
        CUDA_ROOT = "${root}";
        CUDA_PATH = "${root}";
        CUDA_TOOLKIT_ROOT_DIR = "${root}";
        # Cover both lib/ (nix-style) and lib64/ (FHS-style) so downstream
        # build.rs scripts that probe either layout resolve libcudart + stubs.
        CUDA_LIBRARY_PATH =
          "${root}/lib:${root}/lib64:${root}/lib/stubs:${root}/lib64/stubs";
      };
      # Symlink every NVIDIA-shipped driver library (libcuda, libnvidia-*) into a
      # single shim dir that we then stick on LD_LIBRARY_PATH. libcuda alone is not
      # enough: the driver will dlopen companions like libnvidia-ptxjitcompiler.so.1
      # when JITing PTX, and failing to find them surfaces as
      # CUDA_ERROR_JIT_COMPILER_NOT_FOUND from cuModuleLoadDataEx.
      driverShimHook = ''
        driver_shim_dir="$PWD/.nix-driver-libs"
        mkdir -p "$driver_shim_dir"
        for src in "${driverLibDir}"/libcuda.so* "${driverLibDir}"/libnvidia-*.so*; do
          [ -e "$src" ] || continue
          ln -sf "$src" "$driver_shim_dir/$(basename "$src")"
        done
      '';

      # ---- LLVM 7-only shell (CUDA 12.9 toolkit) ----
      v7Shell = pkgs.mkShell ((mkCudaEnv cuda7Root) // {
        nativeBuildInputs = commonNativeInputs ++ [
          cuda7Root
          llvm7Bin
          llvm7Dev
          llvm7CompatTools
          pkgsLlvm7.llvmPackages_7.clang
          pkgsLlvm7.llvmPackages_7.libclang
        ];
        buildInputs = v7BuildInputs;
        LLVM_CONFIG = "${llvm7Dev}/bin/llvm-config";
        # Give bindgen an explicit libclang (matched to 23.05's glibc) so it doesn't
        # fall back to scanning system paths and pick up an apt-installed LLVM 19
        # with deps the v7 shell's LD_LIBRARY_PATH doesn't satisfy.
        LIBCLANG_PATH = "${pkgsLlvm7.lib.getLib pkgsLlvm7.llvmPackages_7.libclang}/lib";
        shellHook = driverShimHook + ''
          export PATH="${llvm7CompatTools}/bin:${llvm7Bin}/bin:${llvm7Dev}/bin:${cuda7Root}/bin:${cuda7Root}/nvvm/bin:$PATH"
          export LD_LIBRARY_PATH="$driver_shim_dir:${cuda7Root}/nvvm/lib:${cuda7Root}/nvvm/lib64:${cuda7Root}/lib64:${cuda7Root}/lib:${pkgsLlvm7.ncurses.out}/lib:${pkgsLlvm7.libxml2.out}/lib:${pkgsLlvm7.zlib.out}/lib:${pkgsLlvm7.stdenv.cc.cc.lib}/lib''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

          echo "rust-cuda llvm7 shell"
          echo "  CUDA_HOME=$CUDA_HOME"
          echo "  LLVM_CONFIG=$LLVM_CONFIG"
          echo "  NVIDIA_DRIVER_LIB=$driver_shim_dir/libcuda.so.1"
        '';
      });

      # ---- LLVM 19-only shell (CUDA 13.2 toolkit, the active-work shell) ----
      v19Shell = pkgs.mkShell ((mkCudaEnv cuda19Root) // {
        nativeBuildInputs = commonNativeInputs ++ [
          cuda19Root
          llvm19.clang
          llvm19.libclang
          llvm19Bin
          llvm19Dev
          llvm19CompatTools
        ];
        buildInputs = v19BuildInputs;
        LLVM_CONFIG_19 = "${llvm19Dev}/bin/llvm-config";
        LIBCLANG_PATH = "${lib.getLib llvm19.libclang}/lib";
        shellHook = driverShimHook + ''
          export PATH="${llvm19CompatTools}/bin:${llvm19Bin}/bin:${llvm19Dev}/bin:${cuda19Root}/bin:${cuda19Root}/nvvm/bin:$PATH"
          export LD_LIBRARY_PATH="$driver_shim_dir:${cuda19Root}/nvvm/lib:${cuda19Root}/nvvm/lib64:${cuda19Root}/lib64:${cuda19Root}/lib:${pkgs.ncurses.out}/lib:${pkgs.libxml2.out}/lib:${pkgs.zlib.out}/lib:${pkgs.stdenv.cc.cc.lib}/lib''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

          echo "rust-cuda llvm19 shell"
          echo "  CUDA_HOME=$CUDA_HOME"
          echo "  LLVM_CONFIG_19=$LLVM_CONFIG_19"
          echo "  NVIDIA_DRIVER_LIB=$driver_shim_dir/libcuda.so.1"
        '';
      });
    in
    {
      devShells.${system} = {
        default = v19Shell;
        v7 = v7Shell;
        v19 = v19Shell;
      };
    };
}
