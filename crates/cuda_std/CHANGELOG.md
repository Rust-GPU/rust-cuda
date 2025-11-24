# Changelog

Notable changes to this project will be documented in this file.

## Unreleased

- Added warp shuffles, matches, reductions, and votes in the `warp` module.
- Added `activemask` in the `warp` module to query a mask of the active threads.
- Fixed `lane_id` generating invalid ptx.
- Removed `shared_array!` due to unsoundness.

## 0.2.2 - 2/7/22

- Thread/Block/Grid index/dim intrinsics now hint to llvm that their range is in some bound declared by CUDA. Hopefully allowing for more optimizations.

## 0.2.1 - 12/8/21

- Fixed `shared_array!` not using fully qualified MaybeUninit.
- Fixed `shared_array!` working on the CPU.
- Added experimental dynamic shared memory support through `shared::dynamic_shared_memory`.

## 0.2.0 - 12/5/21

- Added `#[externally_visible]` in conjunction with cg_nvvm dead code elimination changes to mark that
a function is externally visible.
- Added `#[address_space(...)]` in conjunction with cg_nvvm address space changes. Only meant for internal use
and advanced users.
- Added `cuda_std::ptr`.
- Added `is_in_address_space`
- Added `convert_generic_to_specific_address_space`
- Added `convert_specific_address_space_to_generic`
- Added basic static shared memory support with `cuda_std::shared_array`.
