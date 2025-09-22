use rustc_abi as abi;
use rustc_abi::{self, BackendRepr, Float, HasDataLayout, Primitive};
use rustc_codegen_ssa::errors::InvalidMonomorphization;
use rustc_codegen_ssa::mir::operand::OperandValue;
use rustc_codegen_ssa::mir::place::PlaceValue;
use rustc_codegen_ssa::mir::{operand::OperandRef, place::PlaceRef};
use rustc_codegen_ssa::traits::{
    BaseTypeCodegenMethods, BuilderMethods, ConstCodegenMethods, IntrinsicCallBuilderMethods,
    OverflowOp,
};
use rustc_middle::ty::layout::{FnAbiOf, HasTypingEnv, LayoutOf};
use rustc_middle::ty::{self, Ty};
use rustc_middle::{bug, span_bug};
use rustc_span::symbol::kw;
use rustc_span::{Span, Symbol, sym};
use rustc_target::callconv::PassMode;
use tracing::trace;

use crate::abi::LlvmType;
use crate::builder::{Builder, CountZerosKind};
use crate::context::CodegenCx;
use crate::llvm::{self, Type, Value};
use crate::ty::LayoutLlvmExt;

fn handle_128_bit_intrinsic<'ll>(
    b: &mut Builder<'_, 'll, '_>,
    name: Symbol,
    args: &[OperandRef<'_, &'ll Value>],
) -> &'ll Value {
    match name {
        sym::ctlz | sym::cttz => {
            // TODO(@LegNeato): LLVM 7.1 doesn't have llvm.ctlz.i128/llvm.cttz.i128
            // When we upgrade NVVM, we can call the real intrinsic directly
            let kind = if name == sym::ctlz {
                CountZerosKind::Leading
            } else {
                CountZerosKind::Trailing
            };
            b.emulate_i128_count_zeros(args[0].immediate(), kind, false)
        }
        sym::ctlz_nonzero | sym::cttz_nonzero => {
            // TODO(@LegNeato): LLVM 7.1 doesn't have llvm.ctlz.i128/llvm.cttz.i128
            // When we upgrade NVVM, we can call the real intrinsic directly
            let kind = if name == sym::ctlz_nonzero {
                CountZerosKind::Leading
            } else {
                CountZerosKind::Trailing
            };
            b.emulate_i128_count_zeros(args[0].immediate(), kind, true)
        }
        sym::ctpop => {
            // TODO(@LegNeato): LLVM 7.1 doesn't have llvm.ctpop.i128
            // When we upgrade NVVM, we can call the real intrinsic directly
            b.emulate_i128_ctpop(args[0].immediate())
        }
        sym::bswap => {
            // TODO(@LegNeato): LLVM 7.1 doesn't have llvm.bswap.i128 (added in LLVM 9.0)
            // When we upgrade NVVM, we can call the real intrinsic directly
            // For now, emulate it by swapping the two i64 halves and byte-swapping each
            b.emulate_i128_bswap(args[0].immediate())
        }
        sym::bitreverse => {
            // TODO(@LegNeato): LLVM 7.1 doesn't have llvm.bitreverse.i128
            // When we upgrade NVVM, we can call the real intrinsic directly
            b.emulate_i128_bitreverse(args[0].immediate())
        }
        sym::rotate_left | sym::rotate_right => {
            // TODO(@LegNeato): LLVM 7.1 doesn't have llvm.fshl.i128/llvm.fshr.i128
            // When we upgrade NVVM, we can call the real intrinsic directly
            let is_left = name == sym::rotate_left;
            let val = args[0].immediate();
            let shift = args[1].immediate();
            b.emulate_i128_rotate(val, shift, is_left)
        }
        _ => {
            // For any unsupported 128-bit intrinsics, return a fatal error
            // This shouldn't happen with the current set of intrinsics
            b.fatal(format!("unsupported 128-bit intrinsic: {name}"))
        }
    }
}

// llvm 7 does not have saturating intrinsics, so we reimplement them right here.
// This is derived from what rustc used to do before the intrinsics. It should map to the same assembly.
fn saturating_intrinsic_impl<'ll, 'tcx>(
    b: &mut Builder<'_, 'll, 'tcx>,
    width: u32,
    signed: bool,
    is_add: bool,
    args: &[OperandRef<'tcx, &'ll Value>],
) -> &'ll Value {
    use crate::intrinsic::OverflowOp;
    use rustc_codegen_ssa::common::IntPredicate;
    use rustc_middle::ty::IntTy::*;
    use rustc_middle::ty::UintTy::*;

    let tcx = b.tcx;
    let ty = match (signed, width) {
        (true, 8) => Ty::new_int(tcx, I8),
        (true, 16) => Ty::new_int(tcx, I16),
        (true, 32) => Ty::new_int(tcx, I32),
        (true, 64) => Ty::new_int(tcx, I64),
        (true, 128) => Ty::new_int(tcx, I128),
        (false, 8) => Ty::new_uint(tcx, U8),
        (false, 16) => Ty::new_uint(tcx, U16),
        (false, 32) => Ty::new_uint(tcx, U32),
        (false, 64) => Ty::new_uint(tcx, U64),
        (false, 128) => Ty::new_uint(tcx, U128),
        _ => unreachable!(),
    };

    let llty = b.type_ix(width as u64);
    let a = args[0].immediate();
    let c = args[1].immediate();

    // Perform the add or sub, returning the result and an overflow flag
    let (val, ov) = b.checked_binop(
        if is_add {
            OverflowOp::Add
        } else {
            OverflowOp::Sub
        },
        ty,
        a,
        c,
    );

    let zero = b.const_int(llty, 0);

    // Unsigned case: overflow means clamp to either max or min value
    if !signed {
        let all1 = b.not(zero);
        let clamp = if is_add { all1 } else { zero };
        return b.select(ov, clamp, val);
    }

    // Signed case: compute INT_MIN and INT_MAX
    let one = b.const_int(llty, 1);
    let sh = b.const_int(llty, (width - 1) as i64);
    let int_min = b.shl(one, sh);
    let int_max = b.sub(int_min, one);

    // Check if a is negative
    let a_lt0 = b.icmp(IntPredicate::IntSLT, a, zero);

    // Pick the saturation value depending on operation and operand signs
    let sat = if is_add {
        // Add overflow: if a is negative → INT_MIN, else → INT_MAX
        b.select(a_lt0, int_min, int_max)
    } else {
        // Sub overflow: if a is non-negative and c is negative → INT_MAX, else → INT_MIN
        let a_ge0 = b.not(a_lt0);
        let c_lt0 = b.icmp(IntPredicate::IntSLT, c, zero);
        let to_max = b.and(a_ge0, c_lt0);
        b.select(to_max, int_max, int_min)
    };

    // Return the saturation value if overflow, else the computed result
    b.select(ov, sat, val)
}

fn get_simple_intrinsic<'ll>(
    cx: &CodegenCx<'ll, '_>,
    name: Symbol,
) -> Option<(&'ll Type, &'ll Value)> {
    #[rustfmt::skip]
    let llvm_name = match name {
        sym::sqrtf32      => "__nv_sqrtf",
        sym::sqrtf64      => "__nv_sqrt",
        sym::powif32      => "__nv_powif",
        sym::powif64      => "__nv_powi",
        sym::sinf32       => "__nv_sinf",
        sym::sinf64       => "__nv_sin",
        sym::cosf32       => "__nv_cosf",
        sym::cosf64       => "__nv_cos",
        sym::powf32       => "__nv_powf",
        sym::powf64       => "__nv_pow",
        sym::expf32       => "__nv_expf",
        sym::expf64       => "__nv_exp",
        sym::exp2f32      => "__nv_exp2f",
        sym::exp2f64      => "__nv_exp2",
        sym::logf32       => "__nv_logf",
        sym::logf64       => "__nv_log",
        sym::log10f32     => "__nv_log10f",
        sym::log10f64     => "__nv_log10",
        sym::log2f32      => "__nv_log2f",
        sym::log2f64      => "__nv_log2",
        sym::fmaf32       => "__nv_fmaf",
        sym::fmaf64       => "__nv_fma",
        sym::fabsf32      => "__nv_fabsf",
        sym::fabsf64      => "__nv_fabs",
        sym::minnumf32    => "__nv_fminf",
        sym::minnumf64    => "__nv_fmin",
        sym::maxnumf32    => "__nv_fmaxf",
        sym::maxnumf64    => "__nv_fmax",
        sym::copysignf32  => "__nv_copysignf",
        sym::copysignf64  => "__nv_copysign",
        sym::floorf32     => "__nv_floorf",
        sym::floorf64     => "__nv_floor",
        sym::ceilf32      => "__nv_ceilf",
        sym::ceilf64      => "__nv_ceil",
        sym::truncf32     => "__nv_truncf",
        sym::truncf64     => "__nv_trunc",
        sym::roundf32     => "__nv_roundf",
        sym::roundf64     => "__nv_round",
        sym::round_ties_even_f32 => "__nv_rintf",
        sym::round_ties_even_f64 => "__nv_rint",
        _ => return None,
    };
    trace!("Retrieving nv intrinsic `{:?}`", llvm_name);
    Some(cx.get_intrinsic(llvm_name))
}

impl<'ll, 'tcx> IntrinsicCallBuilderMethods<'tcx> for Builder<'_, 'll, 'tcx> {
    fn codegen_intrinsic_call(
        &mut self,
        instance: ty::Instance<'tcx>,
        args: &[OperandRef<'tcx, &'ll Value>],
        result: PlaceRef<'tcx, &'ll Value>,
        span: Span,
    ) -> Result<(), ty::Instance<'tcx>> {
        let tcx = self.tcx;
        let callee_ty = instance.ty(tcx, self.typing_env());

        let ty::FnDef(def_id, fn_args) = *callee_ty.kind() else {
            bug!("expected fn item type, found {}", callee_ty);
        };

        let sig = callee_ty.fn_sig(tcx);
        let sig = tcx.normalize_erasing_late_bound_regions(self.typing_env(), sig);
        let arg_tys = sig.inputs();
        let ret_ty = sig.output();
        let name = tcx.item_name(def_id);
        let name_str = name.as_str();

        trace!(
            "Beginning intrinsic call: `{:?}`, args: `{:?}`, ret: `{:?}`",
            name, arg_tys, ret_ty
        );

        let llret_ty = self.layout_of(ret_ty).llvm_type(self);

        // Compute fn_abi for intrinsics that need it
        let fn_abi = self.cx.fn_abi_of_instance(instance, ty::List::empty());

        let simple = get_simple_intrinsic(self, name);
        let llval = match name {
            _ if simple.is_some() => {
                let (simple_ty, simple_fn) = simple.unwrap();
                self.call(
                    simple_ty,
                    None,
                    None,
                    simple_fn,
                    &args.iter().map(|arg| arg.immediate()).collect::<Vec<_>>(),
                    None,
                    Some(instance),
                )
            }
            sym::is_val_statically_known => {
                // LLVM 7 does not support this intrinsic, so always assume false.
                self.const_bool(false)
            }
            sym::select_unpredictable => {
                // This should set MD_unpredictable on the select instruction, but
                // nvvm ignores it, so just use a normal select.
                let cond = args[0].immediate();
                assert_eq!(args[1].layout, args[2].layout);
                match (args[1].val, args[2].val) {
                    (OperandValue::Ref(true_val), OperandValue::Ref(false_val)) => {
                        assert!(true_val.llextra.is_none());
                        assert!(false_val.llextra.is_none());
                        assert_eq!(true_val.align, false_val.align);
                        let ptr = self.select(cond, true_val.llval, false_val.llval);
                        let selected =
                            OperandValue::Ref(PlaceValue::new_sized(ptr, true_val.align));
                        selected.store(self, result);
                        return Ok(());
                    }
                    (OperandValue::Immediate(_), OperandValue::Immediate(_))
                    | (OperandValue::Pair(_, _), OperandValue::Pair(_, _)) => {
                        let true_val = args[1].immediate_or_packed_pair(self);
                        let false_val = args[2].immediate_or_packed_pair(self);
                        self.select(cond, true_val, false_val)
                    }
                    (OperandValue::ZeroSized, OperandValue::ZeroSized) => return Ok(()),
                    _ => span_bug!(span, "Incompatible OperandValue for select_unpredictable"),
                }
            }
            sym::likely => self.call_intrinsic(
                "llvm.expect.i1",
                &[args[0].immediate(), self.const_bool(true)],
            ),
            sym::unlikely => self.call_intrinsic(
                "llvm.expect.i1",
                &[args[0].immediate(), self.const_bool(false)],
            ),
            kw::Try => {
                let try_func = args[0].immediate();
                let data = args[1].immediate();

                self.call(self.type_i1(), None, None, try_func, &[data], None, None);
                let ret_align = self.data_layout().i32_align.abi;
                self.store(self.const_i32(0), result.val.llval, ret_align)
            }
            sym::breakpoint => {
                // debugtrap is not supported
                return Ok(());
            }
            sym::va_copy => {
                self.call_intrinsic("llvm.va_copy", &[args[0].immediate(), args[1].immediate()])
            }
            sym::va_arg => {
                match result.layout.backend_repr {
                    abi::BackendRepr::Scalar(scalar) => {
                        match scalar.primitive() {
                            Primitive::Int(..) => {
                                if self.cx().size_of(ret_ty).bytes() < 4 {
                                    // `va_arg` should not be called on a integer type
                                    // less than 4 bytes in length. If it is, promote
                                    // the integer to a `i32` and truncate the result
                                    // back to the smaller type.
                                    let promoted_result = self.va_arg(
                                        args[0].immediate(),
                                        self.cx.layout_of(tcx.types.i32).llvm_type(self.cx),
                                    );
                                    self.trunc(promoted_result, llret_ty)
                                } else {
                                    self.va_arg(
                                        args[0].immediate(),
                                        self.cx.layout_of(ret_ty).llvm_type(self.cx),
                                    )
                                }
                            }
                            Primitive::Float(Float::F16) => {
                                bug!("the va_arg intrinsic does not work with `f16`")
                            }
                            Primitive::Float(Float::F64) | Primitive::Pointer(_) => self.va_arg(
                                args[0].immediate(),
                                self.cx.layout_of(ret_ty).llvm_type(self.cx),
                            ),
                            // `va_arg` should never be used with the return type f32.
                            Primitive::Float(Float::F32) => {
                                bug!("the va_arg intrinsic does not work with `f32`")
                            }
                            Primitive::Float(Float::F128) => {
                                bug!("the va_arg intrinsic does not work with `f128`")
                            }
                        }
                    }
                    _ => bug!("the va_arg intrinsic does not work with non-scalar types"),
                }
            }
            sym::volatile_load | sym::unaligned_volatile_load => {
                let ptr = args[0].immediate();
                let load = self.volatile_load(result.layout.llvm_type(self), ptr);
                let align = if name == sym::unaligned_volatile_load {
                    1
                } else {
                    result.layout.align.abi.bytes() as u32
                };
                unsafe {
                    llvm::LLVMSetAlignment(load, align);
                }
                if !result.layout.is_zst() {
                    self.store_to_place(load, result.val);
                }
                return Ok(());
            }
            sym::volatile_store => {
                let dst = args[0].deref(self.cx());
                args[1].val.volatile_store(self, dst);
                return Ok(());
            }
            sym::unaligned_volatile_store => {
                let dst = args[0].deref(self.cx());
                args[1].val.unaligned_volatile_store(self, dst);
                return Ok(());
            }
            sym::prefetch_read_data
            | sym::prefetch_write_data
            | sym::prefetch_read_instruction
            | sym::prefetch_write_instruction => {
                let (rw, cache_type) = match name {
                    sym::prefetch_read_data => (0, 1),
                    sym::prefetch_write_data => (1, 1),
                    sym::prefetch_read_instruction => (0, 0),
                    sym::prefetch_write_instruction => (1, 0),
                    _ => bug!(),
                };
                self.call_intrinsic(
                    "llvm.prefetch",
                    &[
                        args[0].immediate(),
                        self.const_i32(rw),
                        args[1].immediate(),
                        self.const_i32(cache_type),
                    ],
                )
            }
            sym::carrying_mul_add => {
                let (size, signed) = fn_args.type_at(0).int_size_and_signed(self.tcx);

                let wide_llty = self.type_ix(size.bits() * 2);
                let args = args.as_array().unwrap();
                let [a, b, c, d] = args.map(|a| self.intcast(a.immediate(), wide_llty, signed));

                let wide = if signed {
                    let prod = self.unchecked_smul(a, b);
                    let acc = self.unchecked_sadd(prod, c);
                    self.unchecked_sadd(acc, d)
                } else {
                    let prod = self.unchecked_umul(a, b);
                    let acc = self.unchecked_uadd(prod, c);
                    self.unchecked_uadd(acc, d)
                };

                let narrow_llty = self.type_ix(size.bits());
                let low = self.trunc(wide, narrow_llty);
                let bits_const = self.const_uint(wide_llty, size.bits());
                // No need for ashr when signed; LLVM changes it to lshr anyway.
                let high = self.lshr(wide, bits_const);
                // FIXME: could be `trunc nuw`, even for signed.
                let high = self.trunc(high, narrow_llty);

                let pair_llty = self.type_struct(&[narrow_llty, narrow_llty], false);
                let pair = self.const_poison(pair_llty);
                let pair = self.insert_value(pair, low, 0);
                self.insert_value(pair, high, 1)
            }
            sym::unchecked_shl => self.shl(args[0].immediate(), args[1].immediate()),
            sym::unchecked_shr => {
                let lhs = args[0].immediate();
                let rhs = args[1].immediate();
                match arg_tys[0].kind() {
                    ty::Int(_) => self.ashr(lhs, rhs),
                    _ => self.lshr(lhs, rhs),
                }
            }
            sym::ctlz
            | sym::ctlz_nonzero
            | sym::cttz
            | sym::cttz_nonzero
            | sym::ctpop
            | sym::bswap
            | sym::bitreverse
            | sym::rotate_left
            | sym::rotate_right
            | sym::saturating_add
            | sym::saturating_sub => {
                let ty = arg_tys[0];
                if !ty.is_integral() {
                    tcx.dcx()
                        .emit_err(InvalidMonomorphization::BasicIntegerType { span, name, ty });
                    return Ok(());
                }
                let (size, signed) = ty.int_size_and_signed(self.tcx);
                let width = size.bits();
                if name == sym::saturating_add || name == sym::saturating_sub {
                    saturating_intrinsic_impl(
                        self,
                        width as u32,
                        signed,
                        name == sym::saturating_add,
                        args,
                    )
                } else if width == 128 {
                    handle_128_bit_intrinsic(self, name, args)
                } else {
                    match name {
                        sym::ctlz | sym::cttz => {
                            let y = self.const_bool(false);
                            let llvm_name = format!("llvm.{name}.i{width}");
                            self.call_intrinsic(&llvm_name, &[args[0].immediate(), y])
                        }
                        sym::ctlz_nonzero | sym::cttz_nonzero => {
                            let y = self.const_bool(true);
                            let llvm_name = format!("llvm.{}.i{width}", &name_str[..4]);
                            self.call_intrinsic(&llvm_name, &[args[0].immediate(), y])
                        }
                        sym::ctpop => self.call_intrinsic(
                            &format!("llvm.ctpop.i{width}"),
                            &[args[0].immediate()],
                        ),
                        sym::bswap => {
                            if width == 8 {
                                args[0].immediate() // byte swap a u8/i8 is just a no-op
                            } else {
                                self.call_intrinsic(
                                    &format!("llvm.bswap.i{width}"),
                                    &[args[0].immediate()],
                                )
                            }
                        }
                        sym::bitreverse => self.call_intrinsic(
                            &format!("llvm.bitreverse.i{width}"),
                            &[args[0].immediate()],
                        ),
                        sym::rotate_left | sym::rotate_right => {
                            let is_left = name == sym::rotate_left;
                            let val = args[0].immediate();
                            let raw_shift = args[1].immediate();
                            // rotate = funnel shift with first two args the same
                            let llvm_name =
                                &format!("llvm.fsh{}.i{}", if is_left { 'l' } else { 'r' }, width);

                            // llvm expects shift to be the same type as the values, but rust
                            // always uses `u32`.
                            let raw_shift = self.intcast(raw_shift, self.val_ty(val), false);

                            self.call_intrinsic(llvm_name, &[val, val, raw_shift])
                        }
                        sym::saturating_add | sym::saturating_sub => {
                            let is_add = name == sym::saturating_add;
                            let lhs = args[0].immediate();
                            let rhs = args[1].immediate();
                            let llvm_name = &format!(
                                "llvm.{}{}.sat.i{}",
                                if signed { 's' } else { 'u' },
                                if is_add { "add" } else { "sub" },
                                width
                            );
                            self.call_intrinsic(llvm_name, &[lhs, rhs])
                        }
                        _ => unreachable!(),
                    }
                }
            }
            sym::raw_eq => {
                use rustc_codegen_ssa::common::IntPredicate;
                let tp_ty = fn_args.type_at(0);
                let layout = self.layout_of(tp_ty).layout;
                let use_integer_compare = match layout.backend_repr() {
                    BackendRepr::Scalar(_) | BackendRepr::ScalarPair(_, _) => true,
                    BackendRepr::SimdVector { .. } => false,
                    BackendRepr::Memory { .. } => {
                        // For rusty ABIs, small aggregates are actually passed
                        // as `RegKind::Integer` (see `FnAbi::adjust_for_abi`),
                        // so we re-use that same threshold here.
                        layout.size <= self.data_layout().pointer_size() * 2
                    }
                };

                let a = args[0].immediate();
                let b = args[1].immediate();
                if layout.size.bytes() == 0 {
                    self.const_bool(true)
                } else if use_integer_compare {
                    let integer_ty = self.type_ix(layout.size.bits());
                    let ptr_ty = self.type_ptr_to(integer_ty);
                    let a_ptr = self.bitcast(a, ptr_ty);
                    let a_val = self.load(integer_ty, a_ptr, layout.align.abi);
                    let b_ptr = self.bitcast(b, ptr_ty);
                    let b_val = self.load(integer_ty, b_ptr, layout.align.abi);
                    self.icmp(IntPredicate::IntEQ, a_val, b_val)
                } else {
                    let i8p_ty = self.type_i8p();
                    let a_ptr = self.bitcast(a, i8p_ty);
                    let b_ptr = self.bitcast(b, i8p_ty);
                    let n = self.const_usize(layout.size.bytes());
                    let cmp = self.call_intrinsic("memcmp", &[a_ptr, b_ptr, n]);
                    self.icmp(IntPredicate::IntEQ, cmp, self.const_i32(0))
                }
            }
            sym::compare_bytes => self.call_intrinsic(
                "memcmp",
                &[
                    args[0].immediate(),
                    args[1].immediate(),
                    args[2].immediate(),
                ],
            ),

            sym::black_box => {
                args[0].val.store(self, result);
                let result_val_span = [result.val.llval];
                // We need to "use" the argument in some way LLVM can't introspect, and on
                // targets that support it we can typically leverage inline assembly to do
                // this. LLVM's interpretation of inline assembly is that it's, well, a black
                // box. This isn't the greatest implementation since it probably deoptimizes
                // more than we want, but it's so far good enough.
                //
                // For zero-sized types, the location pointed to by the result may be
                // uninitialized. Do not "use" the result in this case; instead just clobber
                // the memory.
                let (constraint, inputs): (&str, &[_]) = if result.layout.is_zst() {
                    ("~{memory}", &[])
                } else {
                    ("r,~{memory}", &result_val_span)
                };
                crate::asm::inline_asm_call(
                    self,
                    "",
                    constraint,
                    inputs,
                    self.type_void(),
                    true,
                    false,
                    llvm::AsmDialect::Att,
                    &[span],
                )
                .unwrap_or_else(|| bug!("failed to generate inline asm call for `black_box`"));

                // We have copied the value to `result` already.
                return Ok(());
            }

            // is this even supported by nvvm? i did not find a definitive answer
            _ if name_str.starts_with("simd_") => todo!("simd intrinsics"),
            // Fall back to a fallback intrinsic implementation, if possible
            _ => {
                // This piece of code was adapted from `rustc_codegen_cranelift`.
                let intrinsic = self.tcx.intrinsic(instance.def_id()).unwrap();
                if intrinsic.must_be_overridden {
                    bug!(
                        "intrinsic {} must be overridden by codegen_nvvm, but isn't",
                        intrinsic.name,
                    );
                }
                return Err(rustc_middle::ty::Instance::new_raw(
                    instance.def_id(),
                    instance.args,
                ));
            }
        };
        trace!("Finish intrinsic call: `{:?}`", llval);
        if !fn_abi.ret.is_ignore() {
            if let PassMode::Cast { cast, .. } = &fn_abi.ret.mode {
                let ptr_llty = self.type_ptr_to(cast.llvm_type(self));
                let ptr = self.pointercast(result.val.llval, ptr_llty);
                self.store(llval, ptr, result.val.align);
            } else {
                OperandRef::from_immediate_or_packed_pair(self, llval, result.layout)
                    .val
                    .store(self, result);
            }
        }
        Ok(())
    }

    fn abort(&mut self) {
        trace!("Generate abort call");
        self.call_intrinsic("llvm.trap", &[]);
    }

    fn assume(&mut self, val: &'ll Value) {
        trace!("Generate assume call with `{:?}`", val);
        self.call_intrinsic("llvm.assume", &[val]);
    }

    fn expect(&mut self, cond: &'ll Value, expected: bool) -> &'ll Value {
        trace!("Generate expect call with `{:?}`, {}", cond, expected);
        self.call_intrinsic("llvm.expect.i1", &[cond, self.const_bool(expected)])
    }

    fn type_checked_load(
        &mut self,
        _llvtable: Self::Value,
        _vtable_byte_offset: u64,
        _typeid: Self::Metadata,
    ) -> Self::Value {
        // LLVM CFI doesnt make sense on the GPU
        self.const_i32(0)
    }

    fn va_start(&mut self, va_list: &'ll Value) -> &'ll Value {
        trace!("Generate va_start `{:?}`", va_list);
        self.call_intrinsic("llvm.va.start", &[va_list])
    }

    fn va_end(&mut self, va_list: &'ll Value) -> &'ll Value {
        trace!("Generate va_end call `{:?}`", va_list);
        self.call_intrinsic("llvm.va_end", &[va_list])
    }
}
