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
      systems = [ "aarch64-linux" "x86_64-linux" ];
      forAllSystems = nixpkgs.lib.genAttrs systems;

      mkShells = system:
        let
          # allowUnfree is required because CUDA is unfree.
          pkgs = import nixpkgs {
            inherit system;
            config.allowUnfree = true;
            overlays = [ rust-overlay.overlays.default ];
          };
          pkgsLlvm7 = import nixpkgs-llvm7 { inherit system; };
          lib = pkgs.lib;

          # ---- CUDA toolkit (Nix-managed) ----
          # Toolkit pin chooses what PTX version NVVM emits, which then dictates
          # the minimum host driver version at runtime:
          #   CUDA 13.3.1 → modern LLVM 21 dialect → PTX 9.3 → needs driver 580.x+ (CUDA 13)
          #   CUDA 12.9 → NVVM 21.x → PTX 8.x → runs on CUDA 12.x drivers
          # `cudatoolkit` is the kitchen-sink symlinkJoin maintained by nixpkgs —
          # every header path and lib layout is already wired correctly. The host
          # NVIDIA driver (libcuda.so.1) is needed at runtime; it is *not* shimmed
          # in here — supply it via the system or extend LD_LIBRARY_PATH yourself
          # before running CUDA programs.
          cuda21Root = pkgs.cudaPackages_13_3.cudatoolkit;
          cuda7Root = pkgs.cudaPackages_12_9.cudatoolkit;

          toolchain = pkgs.rust-bin.fromRustupToolchainFile ./rust-toolchain.toml;

          # ---- LLVM 21 (from current nixpkgs) ----
          llvm21 = pkgs.llvmPackages_21;
          llvm21Bin = lib.getBin llvm21.llvm;
          llvm21Dev = lib.getDev llvm21.llvm;
          llvm21CompatTools = pkgs.symlinkJoin {
            name = "llvm21-compat-tools";
            paths = [
              (pkgs.writeShellScriptBin "opt-21" ''exec ${llvm21Bin}/bin/opt "$@"'')
              (pkgs.writeShellScriptBin "llvm-as-21" ''exec ${llvm21Bin}/bin/llvm-as "$@"'')
              (pkgs.writeShellScriptBin "llvm-dis-21" ''exec ${llvm21Bin}/bin/llvm-dis "$@"'')
              (pkgs.writeShellScriptBin "llc-21" ''exec ${llvm21Bin}/bin/llc "$@"'')
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
          # The v21 shell uses unstable's runtime libs (modern glibc). The v7 shell has
          # to match LLVM 7's glibc generation (23.05), otherwise ncurses/libstdc++ from
          # unstable demand GLIBC_2.38+ symbols LLVM 7's linked glibc 2.37 doesn't have.
          v21BuildInputs = [
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
            # fall back to scanning system paths and pick up an apt-installed LLVM 21
            # with deps the v7 shell's LD_LIBRARY_PATH doesn't satisfy.
            LIBCLANG_PATH = "${pkgsLlvm7.lib.getLib pkgsLlvm7.llvmPackages_7.libclang}/lib";
            shellHook = ''
              export PATH="${llvm7CompatTools}/bin:${llvm7Bin}/bin:${llvm7Dev}/bin:${cuda7Root}/bin:${cuda7Root}/nvvm/bin:$PATH"
              export LD_LIBRARY_PATH="${cuda7Root}/nvvm/lib:${cuda7Root}/nvvm/lib64:${cuda7Root}/lib64:${cuda7Root}/lib:${pkgsLlvm7.ncurses.out}/lib:${pkgsLlvm7.libxml2.out}/lib:${pkgsLlvm7.zlib.out}/lib:${pkgsLlvm7.stdenv.cc.cc.lib}/lib''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

              echo "rust-cuda llvm7 shell (${system})"
              echo "  CUDA_HOME=$CUDA_HOME"
              echo "  LLVM_CONFIG=$LLVM_CONFIG"
            '';
          });

          # ---- LLVM 21-only shell (CUDA 13.3.1 toolkit, the active-work shell) ----
          v21Shell = pkgs.mkShell ((mkCudaEnv cuda21Root) // {
            nativeBuildInputs = commonNativeInputs ++ [
              cuda21Root
              llvm21.clang
              llvm21.libclang
              llvm21Bin
              llvm21Dev
              llvm21CompatTools
            ];
            buildInputs = v21BuildInputs;
            LLVM_CONFIG_21 = "${llvm21Dev}/bin/llvm-config";
            LIBCLANG_PATH = "${lib.getLib llvm21.libclang}/lib";
            shellHook = ''
              export PATH="${llvm21CompatTools}/bin:${llvm21Bin}/bin:${llvm21Dev}/bin:${cuda21Root}/bin:${cuda21Root}/nvvm/bin:$PATH"
              export LD_LIBRARY_PATH="${cuda21Root}/nvvm/lib:${cuda21Root}/nvvm/lib64:${cuda21Root}/lib64:${cuda21Root}/lib:${pkgs.ncurses.out}/lib:${pkgs.libxml2.out}/lib:${pkgs.zlib.out}/lib:${pkgs.stdenv.cc.cc.lib}/lib''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

              echo "rust-cuda llvm21 shell (${system})"
              echo "  CUDA_HOME=$CUDA_HOME"
              echo "  LLVM_CONFIG_21=$LLVM_CONFIG_21"
            '';
          });
        in
        {
          default = v21Shell;
          v7 = v7Shell;
          v21 = v21Shell;
        };
    in
    {
      devShells = forAllSystems mkShells;
    };
}
