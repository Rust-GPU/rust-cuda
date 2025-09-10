use crate::context::CodegenCx;
use crate::llvm::Type;
use crate::llvm::Value;
use rustc_codegen_ssa::traits::BaseTypeCodegenMethods;
use rustc_session::config::DebugInfo;

impl<'ll> CodegenCx<'ll, '_> {
    pub(crate) fn declare_intrinsic(&self, key: &str) -> Option<(&'ll Type, &'ll Value)> {
        let map = self.intrinsics_map.borrow();
        let (args, ret) = map.get(key)?;
        Some(self.insert_intrinsic(key, Some(args), ret))
    }

    #[rustfmt::skip] // stop rustfmt from making this 2k lines
    pub(crate) fn build_intrinsics_map(&mut self) {
        let mut map = self.intrinsics_map.borrow_mut();
        let mut remapped = self.remapped_integer_args.borrow_mut();

        macro_rules! ifn {
            ($map:expr, $($name:literal)|*, fn($($arg:expr),*) -> $ret:expr) => {
                for name in [$($name),*] {
                    map.insert(name, (vec![$($arg),*], $ret));
                }
            };
        }

        let real_t_i128 = self.type_i128();
        let real_t_i128_i1 = self.type_struct(&[real_t_i128, self.type_i1()], false);

        let i8p = self.type_i8p();
        let void = self.type_void();
        let i1 = self.type_i1();
        let t_i8 = self.type_i8();
        let t_i16 = self.type_i16();
        let t_i32 = self.type_i32();
        let t_i64 = self.type_i64();
        let t_i128 = self.type_vector(t_i64, 2);
        let t_f32 = self.type_f32();
        let t_f64 = self.type_f64();
        let t_isize = self.type_isize();

        let t_i8_i1 = self.type_struct(&[t_i8, i1], false);
        let t_i16_i1 = self.type_struct(&[t_i16, i1], false);
        let t_i32_i1 = self.type_struct(&[t_i32, i1], false);
        let t_i64_i1 = self.type_struct(&[t_i64, i1], false);
        let t_i128_i1 = self.type_struct(&[t_i128, i1], false);

        let voidp = self.voidp();

        ifn!(map, "llvm.trap" | "llvm.sideeffect", fn() -> void);
        ifn!(map, "llvm.assume", fn(i1) -> void);
        ifn!(map, "llvm.prefetch", fn(i8p, t_i32, t_i32, t_i32) -> void);

        ifn!(map, "llvm.sadd.with.overflow.i16", fn(t_i16, t_i16) -> t_i16_i1);
        ifn!(map, "llvm.sadd.with.overflow.i32", fn(t_i32, t_i32) -> t_i32_i1);
        ifn!(map, "llvm.sadd.with.overflow.i64", fn(t_i64, t_i64) -> t_i64_i1);

        ifn!(map, "llvm.uadd.with.overflow.i16", fn(t_i16, t_i16) -> t_i16_i1);
        ifn!(map, "llvm.uadd.with.overflow.i32", fn(t_i32, t_i32) -> t_i32_i1);
        ifn!(map, "llvm.uadd.with.overflow.i64", fn(t_i64, t_i64) -> t_i64_i1);

        ifn!(map, "llvm.ssub.with.overflow.i16", fn(t_i16, t_i16) -> t_i16_i1);
        ifn!(map, "llvm.ssub.with.overflow.i32", fn(t_i32, t_i32) -> t_i32_i1);
        ifn!(map, "llvm.ssub.with.overflow.i64", fn(t_i64, t_i64) -> t_i64_i1);

        ifn!(map, "llvm.usub.with.overflow.i16", fn(t_i16, t_i16) -> t_i16_i1);
        ifn!(map, "llvm.usub.with.overflow.i32", fn(t_i32, t_i32) -> t_i32_i1);
        ifn!(map, "llvm.usub.with.overflow.i64", fn(t_i64, t_i64) -> t_i64_i1);

        ifn!(map, "llvm.smul.with.overflow.i16", fn(t_i16, t_i16) -> t_i16_i1);
        ifn!(map, "llvm.smul.with.overflow.i32", fn(t_i32, t_i32) -> t_i32_i1);
        ifn!(map, "llvm.smul.with.overflow.i64", fn(t_i64, t_i64) -> t_i64_i1);

        ifn!(map, "llvm.umul.with.overflow.i16", fn(t_i16, t_i16) -> t_i16_i1);
        ifn!(map, "llvm.umul.with.overflow.i32", fn(t_i32, t_i32) -> t_i32_i1);
        ifn!(map, "llvm.umul.with.overflow.i64", fn(t_i64, t_i64) -> t_i64_i1);

        let i128_checked_binops = [
            "__nvvm_i128_addo",
            "__nvvm_u128_addo",
            "__nvvm_i128_subo",
            "__nvvm_u128_subo",
            "__nvvm_i128_mulo",
            "__nvvm_u128_mulo"
        ];

        for binop in i128_checked_binops {
            map.insert(binop, (vec![t_i128, t_i128], t_i128_i1));
            let llfn_ty = self.type_func(&[t_i128, t_i128], t_i128_i1);
            remapped.insert(llfn_ty, (Some(real_t_i128_i1), vec![(0, real_t_i128), (1, real_t_i128)]));
        }

        let i128_saturating_ops = [
            "llvm.sadd.sat.i128",
            "llvm.uadd.sat.i128",
            "llvm.ssub.sat.i128",
            "llvm.usub.sat.i128",
        ];

        for binop in i128_saturating_ops {
            map.insert(binop, (vec![t_i128, t_i128], t_i128));
            let llfn_ty = self.type_func(&[t_i128, t_i128], t_i128);
            remapped.insert(llfn_ty, (Some(real_t_i128), vec![(0, real_t_i128), (1, real_t_i128)]));
        }

        // for some very strange reason, they arent supported for i8 either, but that case
        // is easy to handle and we declare our own functions for that which just
        // zext to i16, use the i16 intrinsic, then trunc back to i8

        // these are declared in libintrinsics, see libintrinsics.ll
        ifn!(map, "__nvvm_i8_addo", fn(t_i8, t_i8) -> t_i8_i1);
        ifn!(map, "__nvvm_u8_addo", fn(t_i8, t_i8) -> t_i8_i1);
        ifn!(map, "__nvvm_i8_subo", fn(t_i8, t_i8) -> t_i8_i1);
        ifn!(map, "__nvvm_u8_subo", fn(t_i8, t_i8) -> t_i8_i1);
        ifn!(map, "__nvvm_i8_mulo", fn(t_i8, t_i8) -> t_i8_i1);
        ifn!(map, "__nvvm_u8_mulo", fn(t_i8, t_i8) -> t_i8_i1);

        // i128 arithmetic operations from compiler-builtins
        // Division and remainder
        ifn!(map, "__nvvm_divti3", fn(t_i128, t_i128) -> t_i128);
        ifn!(map, "__nvvm_udivti3", fn(t_i128, t_i128) -> t_i128);
        ifn!(map, "__nvvm_modti3", fn(t_i128, t_i128) -> t_i128);
        ifn!(map, "__nvvm_umodti3", fn(t_i128, t_i128) -> t_i128);

        // Multiplication
        ifn!(map, "__nvvm_multi3", fn(t_i128, t_i128) -> t_i128);

        // Shift operations
        ifn!(map, "__nvvm_ashlti3", fn(t_i128, t_i32) -> t_i128);
        ifn!(map, "__nvvm_ashrti3", fn(t_i128, t_i32) -> t_i128);
        ifn!(map, "__nvvm_lshrti3", fn(t_i128, t_i32) -> t_i128);

        // Add remapping for i128 binary operations (division, remainder, multiplication)
        // All have the same signature: (i128, i128) -> i128
        let i128_binary_llfn_ty = self.type_func(&[t_i128, t_i128], t_i128);
        remapped.insert(i128_binary_llfn_ty, (Some(real_t_i128), vec![(0, real_t_i128), (1, real_t_i128)]));

        // Add remapping for i128 shift operations
        // All have the same signature: (i128, i32) -> i128
        let i128_shift_llfn_ty = self.type_func(&[t_i128, t_i32], t_i128);
        remapped.insert(i128_shift_llfn_ty, (Some(real_t_i128), vec![(0, real_t_i128), (1, t_i32)]));

        // see comment in libintrinsics.ll
        // ifn!(map, "__nvvm_i128_trap", fn(t_i128, t_i128) -> t_i128);

        ifn!(map, "llvm.sadd.sat.i8", fn(t_i8, t_i8) -> t_i8);
        ifn!(map, "llvm.sadd.sat.i16", fn(t_i16, t_i16) -> t_i16);
        ifn!(map, "llvm.sadd.sat.i32", fn(t_i32, t_i32) -> t_i32);
        ifn!(map, "llvm.sadd.sat.i64", fn(t_i64, t_i64) -> t_i64);

        ifn!(map, "llvm.uadd.sat.i8", fn(t_i8, t_i8) -> t_i8);
        ifn!(map, "llvm.uadd.sat.i16", fn(t_i16, t_i16) -> t_i16);
        ifn!(map, "llvm.uadd.sat.i32", fn(t_i32, t_i32) -> t_i32);
        ifn!(map, "llvm.uadd.sat.i64", fn(t_i64, t_i64) -> t_i64);

        ifn!(map, "llvm.ssub.sat.i8", fn(t_i8, t_i8) -> t_i8);
        ifn!(map, "llvm.ssub.sat.i16", fn(t_i16, t_i16) -> t_i16);
        ifn!(map, "llvm.ssub.sat.i32", fn(t_i32, t_i32) -> t_i32);
        ifn!(map, "llvm.ssub.sat.i64", fn(t_i64, t_i64) -> t_i64);

        ifn!(map, "llvm.usub.sat.i8", fn(t_i8, t_i8) -> t_i8);
        ifn!(map, "llvm.usub.sat.i16", fn(t_i16, t_i16) -> t_i16);
        ifn!(map, "llvm.usub.sat.i32", fn(t_i32, t_i32) -> t_i32);
        ifn!(map, "llvm.usub.sat.i64", fn(t_i64, t_i64) -> t_i64);

        ifn!(map, "llvm.fshl.i8", fn(t_i8, t_i8, t_i8) -> t_i8);
        ifn!(map, "llvm.fshl.i16", fn(t_i16, t_i16, t_i16) -> t_i16);
        ifn!(map, "llvm.fshl.i32", fn(t_i32, t_i32, t_i32) -> t_i32);
        ifn!(map, "llvm.fshl.i64", fn(t_i64, t_i64, t_i64) -> t_i64);

        ifn!(map, "llvm.fshr.i8", fn(t_i8, t_i8, t_i8) -> t_i8);
        ifn!(map, "llvm.fshr.i16", fn(t_i16, t_i16, t_i16) -> t_i16);
        ifn!(map, "llvm.fshr.i32", fn(t_i32, t_i32, t_i32) -> t_i32);
        ifn!(map, "llvm.fshr.i64", fn(t_i64, t_i64, t_i64) -> t_i64);

        ifn!(map, "llvm.ctpop.i8", fn(t_i8) -> t_i8);
        ifn!(map, "llvm.ctpop.i16", fn(t_i16) -> t_i16);
        ifn!(map, "llvm.ctpop.i32", fn(t_i32) -> t_i32);
        ifn!(map, "llvm.ctpop.i64", fn(t_i64) -> t_i64);

        ifn!(map, "llvm.bitreverse.i8", fn(t_i8) -> t_i8);
        ifn!(map, "llvm.bitreverse.i16", fn(t_i16) -> t_i16);
        ifn!(map, "llvm.bitreverse.i32", fn(t_i32) -> t_i32);
        ifn!(map, "llvm.bitreverse.i64", fn(t_i64) -> t_i64);

        ifn!(map, "llvm.bswap.i16", fn(t_i16) -> t_i16);
        ifn!(map, "llvm.bswap.i32", fn(t_i32) -> t_i32);
        ifn!(map, "llvm.bswap.i64", fn(t_i64) -> t_i64);

        ifn!(map, "llvm.ctlz.i8", fn(t_i8, i1) -> t_i8);
        ifn!(map, "llvm.ctlz.i16", fn(t_i16, i1) -> t_i16);
        ifn!(map, "llvm.ctlz.i32", fn(t_i32, i1) -> t_i32);
        ifn!(map, "llvm.ctlz.i64", fn(t_i64, i1) -> t_i64);

        ifn!(map, "llvm.cttz.i8", fn(t_i8, i1) -> t_i8);
        ifn!(map, "llvm.cttz.i16", fn(t_i16, i1) -> t_i16);
        ifn!(map, "llvm.cttz.i32", fn(t_i32, i1) -> t_i32);
        ifn!(map, "llvm.cttz.i64", fn(t_i64, i1) -> t_i64);

        ifn!(map, "llvm.lifetime.start.p0i8", fn(t_i64, i8p) -> void);
        ifn!(map, "llvm.lifetime.end.p0i8", fn(t_i64, i8p) -> void);

        ifn!(map, "llvm.expect.i1", fn(i1, i1) -> i1);
        ifn!(map, "llvm.prefetch", fn(i8p, t_i32, t_i32, t_i32) -> void);

        // This isn't an "LLVM intrinsic", but LLVM's optimization passes
        // recognize it like one and we assume it exists in `core::slice::cmp`
        ifn!(map, "memcmp", fn(i8p, i8p, t_isize) -> t_i32);

        ifn!(map, "llvm.va_start", fn(i8p) -> void);
        ifn!(map, "llvm.va_end", fn(i8p) -> void);
        ifn!(map, "llvm.va_copy", fn(i8p, i8p) -> void);

        if self.tcx.sess.opts.debuginfo != DebugInfo::None {
            ifn!(map, "llvm.dbg.declare", fn(self.type_metadata(), self.type_metadata()) -> void);
            ifn!(map, "llvm.dbg.value", fn(self.type_metadata(), t_i64, self.type_metadata()) -> void);
        }

        // misc syscalls, only the ones we use

        ifn!(map, "vprintf", fn(i8p, voidp) -> t_i32);

        // so, nvvm, instead of allowing llvm math intrinsics and resolving them
        // to libdevice intrinsics, it forces us to explicitly use the libdevice
        // intrinsics and add libdevice as a module. so we need to completely
        // substitute the llvm intrinsics we would use, for the libdevice ones
        //
        // see https://docs.nvidia.com/cuda/pdf/libdevice-users-guide.pdf
        // and https://docs.nvidia.com/cuda/libdevice-users-guide/index.html
        // for docs on what these do

        // libdevice includes a lot of "exotic" intrinsics for common-ish formulas.
        // we might want to do a pass in the future to substitute common ops for
        // special libdevice intrinsics. We should also expose them as util traits
        // for f32 and f64 in cuda_std.

        ifn!(map, "__nv_abs", fn(t_i32) -> t_i32);

        // f64 -> f64 intrinsics
        ifn!(
            map,
            "__nv_acos" |
            "__nv_acosh" |
            "__nv_asin" |
            "__nv_asinh" |
            "__nv_atan" |
            "__nv_atanh" |
            "__nv_cbrt" |
            "__nv_ceil" |
            "__nv_cos" |
            "__nv_cosh" |
            "__nv_cospi" |
            "__nv_drcp_rd" |
            "__nv_drcp_rn" |
            "__nv_drcp_ru" |
            "__nv_drcp_rz" |
            "__nv_dsqrt_rd" |
            "__nv_dsqrt_rn" |
            "__nv_dsqrt_ru" |
            "__nv_dsqrt_rz" |
            "__nv_erf" |
            "__nv_erfc" |
            "__nv_erfcinv" |
            "__nv_erfcx" |
            "__nv_erfinv" |
            "__nv_exp" |
            "__nv_exp10" |
            "__nv_exp2" |
            "__nv_expm1" |
            "__nv_fabs" |
            "__nv_floor" |
            "__nv_j0" |
            "__nv_j1" |
            "__nv_lgamma" |
            "__nv_log" |
            "__nv_log10" |
            "__nv_log1p" |
            "__nv_log2" |
            "__nv_logb" |
            "__nv_nearbyint" |
            "__nv_normcdf" |
            "__nv_normcdfinv" |
            "__nv_rcbrt" |
            "__nv_rint" |
            "__nv_round" |
            "__nv_rsqrt" |
            "__nv_sin" |
            "__nv_sinh" |
            "__nv_sinpi" |
            "__nv_sqrt" |
            "__nv_tan" |
            "__nv_tanh" |
            "__nv_tgamma" |
            "__nv_trunc" |
            "__nv_y0" |
            "__nv_y1",
            fn(t_f64) -> t_f64
        );

        // f32 -> f32 intrinsics
        ifn!(
            map,
            "__nv_acosf" |
            "__nv_acoshf" |
            "__nv_asinf" |
            "__nv_asinhf" |
            "__nv_atanf" |
            "__nv_atanhf" |
            "__nv_cbrtf" |
            "__nv_ceilf" |
            "__nv_cosf" |
            "__nv_coshf" |
            "__nv_cospif" |
            "__nv_erff" |
            "__nv_erfcf" |
            "__nv_erfcinvf" |
            "__nv_erfcxf" |
            "__nv_erfinvf" |
            "__nv_expf" |
            "__nv_exp10f" |
            "__nv_exp2f" |
            "__nv_expm1f" |
            "__nv_fabsf" |
            "__nv_floorf" |
            "__nv_j0f" |
            "__nv_j1f" |
            "__nv_lgammaf" |
            "__nv_logf" |
            "__nv_log10f" |
            "__nv_log1pf" |
            "__nv_log2f" |
            "__nv_logbf" |
            "__nv_nearbyintf" |
            "__nv_normcdff" |
            "__nv_normcdfinvf" |
            "__nv_rcbrtf" |
            "__nv_rintf" |
            "__nv_roundf" |
            "__nv_rsqrtf" |
            "__nv_sinf" |
            "__nv_sinhf" |
            "__nv_sinpif" |
            "__nv_sqrtf" |
            "__nv_tanf" |
            "__nv_tanhf" |
            "__nv_tgammaf" |
            "__nv_truncf" |
            "__nv_y0f" |
            "__nv_y1f",
            fn(t_f32) -> t_f32
        );

        // f64, f64 -> f64 intrinsics
        ifn!(
            map,
            "__nv_atan2" |
            "__nv_copysign" |
            "__nv_dadd_rd" |
            "__nv_dadd_rn" |
            "__nv_dadd_ru" |
            "__nv_dadd_rz" |
            "__nv_ddiv_rd" |
            "__nv_ddiv_rn" |
            "__nv_ddiv_ru" |
            "__nv_ddiv_rz" |
            "__nv_dmul_rd" |
            "__nv_dmul_rn" |
            "__nv_dmul_ru" |
            "__nv_dmul_rz" |
            "__nv_fdim" |
            "__nv_fmax" |
            "__nv_fmin" |
            "__nv_fmod" |
            "__nv_hypot" |
            "__nv_nextafter" |
            "__nv_pow" |
            "__nv_remainder",
            fn(t_f64, t_f64) -> t_f64
        );

        // f32, f32 -> f32 intrinsics
        ifn!(
            map,
            "__nv_atan2f" |
            "__nv_copysignf" |
            "__nv_fadd_rd" |
            "__nv_fadd_rn" |
            "__nv_fadd_ru" |
            "__nv_fadd_rz" |
            "__nv_fast_fdividef" |
            "__nv_fast_powf" |
            "__nv_fdimf" |
            "__nv_fdiv_rd" |
            "__nv_fdiv_rn" |
            "__nv_fdiv_ru" |
            "__nv_fdiv_rz" |
            "__nv_fmaxf" |
            "__nv_fminf" |
            "__nv_fmodf" |
            "__nv_fmul_rd" |
            "__nv_fmul_rn" |
            "__nv_fmul_ru" |
            "__nv_fmul_rz" |
            "__nv_fsub_rd" |
            "__nv_fsub_rn" |
            "__nv_fsub_ru" |
            "__nv_fsub_rz" |
            "__nv_hypotf" |
            "__nv_nextafterf" |
            "__nv_powf" |
            "__nv_remainderf",
            fn(t_f32, t_f32) -> t_f32
        );

        // other intrinsics

        ifn!(
            map,
            "__nv_powi",
            fn(t_f64, t_i32) -> t_f64
        );

        ifn!(
            map,
            "__nv_powif",
            fn(t_f32, t_i32) -> t_f32
        );

        ifn!(
            map,
            "__nv_fma",
            fn(t_f64, t_f64, t_f64) -> t_f64
        );

        ifn!(
            map,
            "__nv_fmaf",
            fn(t_f32, t_f32, t_f32) -> t_f32
        );

        ifn!(
            map,
            "__nv_yn",
            fn(t_i32, t_f64) -> t_f64
        );

        ifn!(
            map,
            "__nv_ynf",
            fn(t_i32, t_f32) -> t_f32
        );
        // Address space checks
        ifn!(map, "llvm.nvvm.isspacep.const", fn(i8p) -> i1);
        ifn!(map, "llvm.nvvm.isspacep.global", fn(i8p) -> i1);
        ifn!(map, "llvm.nvvm.isspacep.local", fn(i8p) -> i1);
        ifn!(map, "llvm.nvvm.isspacep.shared", fn(i8p) -> i1);
  
    }
}
