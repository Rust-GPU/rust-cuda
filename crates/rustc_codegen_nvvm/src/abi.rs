use std::cmp;

use libc::c_uint;
use rustc_abi::BackendRepr::Scalar;
use rustc_abi::CanonAbi;
use rustc_abi::Size;
use rustc_abi::{HasDataLayout, Primitive, Reg, RegKind};
use rustc_codegen_ssa::mir::operand::OperandRef;
use rustc_codegen_ssa::mir::operand::OperandValue;
use rustc_codegen_ssa::mir::place::{PlaceRef, PlaceValue};
use rustc_codegen_ssa::{MemFlags, traits::*};
use rustc_middle::bug;
use rustc_middle::ty::layout::LayoutOf;
pub use rustc_middle::ty::layout::{WIDE_PTR_ADDR, WIDE_PTR_EXTRA};
use rustc_middle::ty::{Ty, TyCtxt, TyKind};
use rustc_target::callconv::{
    ArgAbi, ArgAttribute, ArgAttributes, ArgExtension, CastTarget, FnAbi, PassMode,
};
use tracing::trace;

use crate::builder::Builder;
use crate::context::CodegenCx;
use crate::int_replace::{get_transformed_type, transmute_llval};
use crate::llvm::{self, *};
use crate::ty::LayoutLlvmExt;

pub(crate) fn readjust_fn_abi<'tcx>(
    tcx: TyCtxt<'tcx>,
    fn_abi: &'tcx FnAbi<'tcx, Ty<'tcx>>,
) -> &'tcx FnAbi<'tcx, Ty<'tcx>> {
    // dont override anything in the rust abi for now
    if fn_abi.conv == CanonAbi::Rust {
        return fn_abi;
    }
    let readjust_arg_abi = |arg: &ArgAbi<'tcx, Ty<'tcx>>| {
        let mut arg = ArgAbi {
            layout: arg.layout,
            mode: arg.mode.clone(),
        };

        // ignore zsts
        if arg.layout.is_zst() {
            arg.mode = PassMode::Ignore;
        }

        if let TyKind::Ref(_, ty, _) = arg.layout.ty.kind()
            && matches!(ty.kind(), TyKind::Slice(_))
        {
            let mut ptr_attrs = ArgAttributes::new();
            if let PassMode::Indirect { attrs, .. } = arg.mode {
                ptr_attrs.regular = attrs.regular;
            }
            arg.mode = PassMode::Pair(ptr_attrs, ArgAttributes::new());
        }

        if arg.layout.ty.is_array() && !matches!(arg.mode, PassMode::Direct { .. }) {
            arg.mode = PassMode::Direct(ArgAttributes::new());
        }

        // pass all adts directly as values, ptx wants them to be passed all by value, but rustc's
        // ptx-kernel abi seems to be wrong, and it's unstable.
        if arg.layout.ty.is_adt() && !matches!(arg.mode, PassMode::Direct { .. }) {
            arg.mode = PassMode::Direct(ArgAttributes::new());
        }
        arg
    };
    tcx.arena.alloc(FnAbi {
        args: fn_abi.args.iter().map(readjust_arg_abi).collect(),
        ret: readjust_arg_abi(&fn_abi.ret),
        c_variadic: fn_abi.c_variadic,
        fixed_count: fn_abi.fixed_count,
        conv: fn_abi.conv,
        can_unwind: fn_abi.can_unwind,
    })
}

macro_rules! for_each_kind {
    ($flags: ident, $f: ident, $($kind: ident),+) => ({
        $(if $flags.contains(ArgAttribute::$kind) { $f(llvm::Attribute::$kind) })+
    })
}

trait ArgAttributeExt {
    fn for_each_kind<F>(&self, f: F)
    where
        F: FnMut(llvm::Attribute);
}

impl ArgAttributeExt for ArgAttribute {
    fn for_each_kind<F>(&self, mut f: F)
    where
        F: FnMut(llvm::Attribute),
    {
        for_each_kind!(self, f, NoAlias, NoCapture, NonNull, ReadOnly, InReg)
    }
}

pub(crate) trait ArgAttributesExt {
    fn apply_attrs_to_llfn(&self, idx: AttributePlace, cx: &CodegenCx<'_, '_>, llfn: &Value);
    fn apply_attrs_to_callsite(
        &self,
        idx: AttributePlace,
        cx: &CodegenCx<'_, '_>,
        callsite: &Value,
    );
}

impl ArgAttributesExt for ArgAttributes {
    fn apply_attrs_to_llfn(&self, idx: AttributePlace, _cx: &CodegenCx<'_, '_>, llfn: &Value) {
        let mut regular = self.regular;
        unsafe {
            let deref = self.pointee_size.bytes();
            if deref != 0 {
                if regular.contains(ArgAttribute::NonNull) {
                    llvm::LLVMRustAddDereferenceableAttr(llfn, idx.as_uint(), deref);
                } else {
                    llvm::LLVMRustAddDereferenceableOrNullAttr(llfn, idx.as_uint(), deref);
                }
                regular -= ArgAttribute::NonNull;
            }
            if let Some(align) = self.pointee_align {
                llvm::LLVMRustAddAlignmentAttr(llfn, idx.as_uint(), align.bytes() as u32);
            }
            regular.for_each_kind(|attr| attr.apply_llfn(idx, llfn));
            // TODO(RDambrosio016): Apply mutable noalias once we upgrade to LLVM 13
            match self.arg_ext {
                ArgExtension::None => {}
                ArgExtension::Zext => {
                    llvm::Attribute::ZExt.apply_llfn(idx, llfn);
                }
                ArgExtension::Sext => {
                    llvm::Attribute::SExt.apply_llfn(idx, llfn);
                }
            }
        }
    }

    fn apply_attrs_to_callsite(
        &self,
        idx: AttributePlace,
        _cx: &CodegenCx<'_, '_>,
        callsite: &Value,
    ) {
        let mut regular = self.regular;
        unsafe {
            let deref = self.pointee_size.bytes();
            if deref != 0 {
                if regular.contains(ArgAttribute::NonNull) {
                    llvm::LLVMRustAddDereferenceableCallSiteAttr(callsite, idx.as_uint(), deref);
                } else {
                    llvm::LLVMRustAddDereferenceableOrNullCallSiteAttr(
                        callsite,
                        idx.as_uint(),
                        deref,
                    );
                }
                regular -= ArgAttribute::NonNull;
            }
            if let Some(align) = self.pointee_align {
                llvm::LLVMRustAddAlignmentCallSiteAttr(
                    callsite,
                    idx.as_uint(),
                    align.bytes() as u32,
                );
            }
            regular.for_each_kind(|attr| attr.apply_callsite(idx, callsite));
            match self.arg_ext {
                ArgExtension::None => {}
                ArgExtension::Zext => {
                    llvm::Attribute::ZExt.apply_callsite(idx, callsite);
                }
                ArgExtension::Sext => {
                    llvm::Attribute::SExt.apply_callsite(idx, callsite);
                }
            }
        }
    }
}

pub(crate) trait LlvmType {
    fn llvm_type<'ll>(&self, cx: &CodegenCx<'ll, '_>) -> &'ll Type;
}

impl LlvmType for Reg {
    fn llvm_type<'ll>(&self, cx: &CodegenCx<'ll, '_>) -> &'ll Type {
        match self.kind {
            RegKind::Integer => cx.type_ix(self.size.bits()),
            RegKind::Float => match self.size.bits() {
                32 => cx.type_f32(),
                64 => cx.type_f64(),
                _ => bug!("unsupported float: {:?}", self),
            },
            RegKind::Vector => cx.type_vector(cx.type_i8(), self.size.bytes()),
        }
    }
}

impl LlvmType for CastTarget {
    fn llvm_type<'ll>(&self, cx: &CodegenCx<'ll, '_>) -> &'ll Type {
        let rest_ll_unit = self.rest.unit.llvm_type(cx);
        let rest_count = if self.rest.total == Size::ZERO {
            0
        } else {
            assert_ne!(
                self.rest.unit.size,
                Size::ZERO,
                "total size {:?} cannot be divided into units of zero size",
                self.rest.total
            );
            if !self
                .rest
                .total
                .bytes()
                .is_multiple_of(self.rest.unit.size.bytes())
            {
                assert_eq!(
                    self.rest.unit.kind,
                    RegKind::Integer,
                    "only int regs can be split"
                );
            }
            self.rest
                .total
                .bytes()
                .div_ceil(self.rest.unit.size.bytes())
        };

        // Simplify to a single unit or an array if there's no prefix.
        // This produces the same layout, but using a simpler type.
        if self.prefix.iter().all(|x| x.is_none()) {
            // We can't do this if is_consecutive is set and the unit would get
            // split on the target. Currently, this is only relevant for i128
            // registers.
            if rest_count == 1 && (!self.rest.is_consecutive || self.rest.unit != Reg::i128()) {
                return rest_ll_unit;
            }

            return cx.type_array(rest_ll_unit, rest_count);
        }

        // Generate a struct type with the prefix and the "rest" arguments.
        let prefix_args = self
            .prefix
            .iter()
            .flat_map(|option_reg| option_reg.map(|reg| reg.llvm_type(cx)));
        let rest_args = (0..rest_count).map(|_| rest_ll_unit);
        let args: Vec<_> = prefix_args.chain(rest_args).collect();
        cx.type_struct(&args, false)
    }
}

impl<'ll, 'tcx> ArgAbiBuilderMethods<'tcx> for Builder<'_, 'll, 'tcx> {
    fn store_fn_arg(
        &mut self,
        arg_abi: &ArgAbi<'tcx, Ty<'tcx>>,
        idx: &mut usize,
        dst: PlaceRef<'tcx, Self::Value>,
    ) {
        arg_abi.store_fn_arg(self, idx, dst)
    }
    fn store_arg(
        &mut self,
        arg_abi: &ArgAbi<'tcx, Ty<'tcx>>,
        val: &'ll Value,
        dst: PlaceRef<'tcx, &'ll Value>,
    ) {
        arg_abi.store(self, val, dst)
    }
}

pub(crate) trait FnAbiLlvmExt<'ll, 'tcx> {
    fn llvm_type(&self, cx: &CodegenCx<'ll, 'tcx>) -> &'ll Type;
    fn ptr_to_llvm_type(&self, cx: &CodegenCx<'ll, 'tcx>) -> &'ll Type;
    fn apply_attrs_llfn(&self, cx: &CodegenCx<'ll, 'tcx>, llfn: &'ll Value);
    fn apply_attrs_callsite<'a>(&self, bx: &mut Builder<'a, 'll, 'tcx>, callsite: &'ll Value);
}

impl<'ll, 'tcx> FnAbiLlvmExt<'ll, 'tcx> for FnAbi<'tcx, Ty<'tcx>> {
    fn llvm_type(&self, cx: &CodegenCx<'ll, 'tcx>) -> &'ll Type {
        // Ignore "extra" args when calling C variadic functions.
        // Only the "fixed" args are part of the LLVM function signature.
        let args = if self.c_variadic {
            &self.args[..self.fixed_count as usize]
        } else {
            &self.args
        };

        // This capacity calculation is approximate.
        let mut llargument_tys = Vec::with_capacity(
            self.args.len()
                + if let PassMode::Indirect { .. } = self.ret.mode {
                    1
                } else {
                    0
                },
        );

        // the current index of each parameter in the function. Cant use enumerate on args because
        // some pass modes pass args as multiple params, such as scalar pairs.
        let mut idx = 0;

        let mut llreturn_ty = match &self.ret.mode {
            PassMode::Ignore => cx.type_void(),
            PassMode::Direct(_) | PassMode::Pair(..) => self.ret.layout.immediate_llvm_type(cx),
            PassMode::Cast { cast, .. } => cast.llvm_type(cx),
            PassMode::Indirect { .. } => {
                idx += 1;
                llargument_tys.push(cx.type_ptr_to(self.ret.memory_ty(cx)));
                cx.type_void()
            }
        };

        let mut transformed_types = Vec::new();
        let mut old_ret_ty = Some(llreturn_ty);

        let (new_ret, changed) = get_transformed_type(cx, llreturn_ty);
        llreturn_ty = new_ret;
        if !changed {
            old_ret_ty = None;
        }

        for arg in args {
            let llarg_ty = match &arg.mode {
                PassMode::Ignore => continue,
                PassMode::Direct(_) => arg.layout.immediate_llvm_type(cx),
                PassMode::Pair(..) => {
                    llargument_tys.push(arg.layout.scalar_pair_element_llvm_type(cx, 0, true));
                    llargument_tys.push(arg.layout.scalar_pair_element_llvm_type(cx, 1, true));
                    idx += 2;
                    continue;
                }
                PassMode::Indirect {
                    attrs: _,
                    meta_attrs: Some(_),
                    on_stack: _,
                } => {
                    let ptr_ty = Ty::new_mut_ptr(cx.tcx, arg.layout.ty);
                    let ptr_layout = cx.layout_of(ptr_ty);
                    llargument_tys.push(ptr_layout.scalar_pair_element_llvm_type(cx, 0, true));
                    llargument_tys.push(ptr_layout.scalar_pair_element_llvm_type(cx, 1, true));
                    idx += 2;
                    continue;
                }
                PassMode::Cast { cast, pad_i32 } => {
                    // add padding
                    if *pad_i32 {
                        idx += 1;
                        llargument_tys.push(Reg::i32().llvm_type(cx));
                    }
                    cast.llvm_type(cx)
                }
                PassMode::Indirect {
                    attrs: _,
                    meta_attrs: None,
                    on_stack: _,
                } => cx.type_ptr_to(arg.memory_ty(cx)),
            };
            let (new, changed) = get_transformed_type(cx, llarg_ty);
            if changed {
                transformed_types.push((idx, llarg_ty));
            }
            llargument_tys.push(new);
            idx += 1;
        }

        let ty = if self.c_variadic {
            cx.type_variadic_func(&llargument_tys, llreturn_ty)
        } else {
            cx.type_func(&llargument_tys, llreturn_ty)
        };
        if !transformed_types.is_empty() || old_ret_ty.is_some() {
            trace!("remapping args in {:?} to {:?}", ty, transformed_types);
            cx.remapped_integer_args
                .borrow_mut()
                .insert(ty, (old_ret_ty, transformed_types));
        }
        ty
    }

    fn ptr_to_llvm_type(&self, cx: &CodegenCx<'ll, 'tcx>) -> &'ll Type {
        unsafe {
            llvm::LLVMPointerType(
                self.llvm_type(cx),
                cx.data_layout().instruction_address_space.0 as c_uint,
            )
        }
    }

    fn apply_attrs_llfn(&self, cx: &CodegenCx<'ll, 'tcx>, llfn: &'ll Value) {
        if self.ret.layout.is_uninhabited() {
            llvm::Attribute::NoReturn.apply_llfn(llvm::AttributePlace::Function, llfn);
        }

        // TODO(RDambrosio016): should this always/never be applied? unwinding
        // on the gpu doesnt exist.
        if !self.can_unwind {
            llvm::Attribute::NoUnwind.apply_llfn(llvm::AttributePlace::Function, llfn);
        }

        let mut i = 0;
        let mut apply = |attrs: &ArgAttributes| {
            attrs.apply_attrs_to_llfn(llvm::AttributePlace::Argument(i), cx, llfn);
            i += 1;
            i - 1
        };
        match &self.ret.mode {
            PassMode::Direct(attrs) => {
                attrs.apply_attrs_to_llfn(llvm::AttributePlace::ReturnValue, cx, llfn);
            }
            PassMode::Indirect {
                attrs,
                meta_attrs: _,
                on_stack,
            } => {
                assert!(!on_stack);
                let i = apply(attrs);
                llvm::Attribute::StructRet.apply_llfn(llvm::AttributePlace::Argument(i), llfn);
            }
            _ => {}
        }
        for arg in &self.args {
            match &arg.mode {
                PassMode::Ignore => {}
                PassMode::Indirect {
                    attrs,
                    meta_attrs: None,
                    on_stack: true,
                } => {
                    apply(attrs);
                    // TODO(RDambrosio016): we should technically apply byval here,
                    // llvm 7 seems to have it, but i could not find a way to apply it through the
                    // C++ API, so somebody more experienced in the C++ API should look at this.
                    // it shouldnt do anything bad since it seems to be only for optimization.
                }
                PassMode::Direct(attrs)
                | PassMode::Indirect {
                    attrs,
                    meta_attrs: None,
                    on_stack: false,
                } => {
                    apply(attrs);
                }
                PassMode::Indirect {
                    attrs,
                    meta_attrs: Some(extra_attrs),
                    on_stack,
                } => {
                    assert!(!on_stack);
                    apply(attrs);
                    apply(extra_attrs);
                }
                PassMode::Pair(a, b) => {
                    apply(a);
                    apply(b);
                }
                PassMode::Cast { cast, pad_i32 } => {
                    if *pad_i32 {
                        apply(&ArgAttributes::new());
                    }
                    apply(&cast.attrs);
                }
            }
        }
    }

    fn apply_attrs_callsite<'a>(&self, bx: &mut Builder<'a, 'll, 'tcx>, mut callsite: &'ll Value) {
        // HACK(RDambrosio016): We sometimes lie to rustc with return values and give it a bitcast
        // instead of a call. This is because we sometimes have to bitcast return types like <2 x i64> to i128.
        // So we just check if the last call was remapped.
        if let Some(old) = bx.cx.last_call_llfn.get() {
            callsite = old;
            bx.cx.last_call_llfn.set(None);
        }

        let mut i = 0;
        let mut apply = |cx: &CodegenCx<'_, '_>, attrs: &ArgAttributes| {
            attrs.apply_attrs_to_callsite(llvm::AttributePlace::Argument(i), cx, callsite);
            i += 1;
            i - 1
        };
        match self.ret.mode {
            PassMode::Direct(attrs) => {
                attrs.apply_attrs_to_callsite(llvm::AttributePlace::ReturnValue, bx.cx, callsite);
            }
            PassMode::Indirect {
                attrs,
                meta_attrs: _,
                on_stack,
            } => {
                assert!(!on_stack);
                apply(bx.cx, &attrs);
            }
            _ => {}
        }
        if let Scalar(scalar) = self.ret.layout.backend_repr {
            // If the value is a boolean, the range is 0..2 and that ultimately
            // become 0..0 when the type becomes i1, which would be rejected
            // by the LLVM verifier.
            if let Primitive::Int(..) = scalar.primitive()
                && !scalar.is_bool()
                && !scalar.is_always_valid(bx)
            {
                trace!("apply_attrs_callsite -> range_metadata");
                bx.range_metadata(callsite, scalar.valid_range(bx));
            }
        }
        for arg in self.args.iter() {
            match &arg.mode {
                PassMode::Ignore => {}
                PassMode::Indirect {
                    attrs,
                    meta_attrs: None,
                    on_stack: true,
                } => {
                    apply(bx.cx, attrs);
                }
                PassMode::Direct(attrs)
                | PassMode::Indirect {
                    attrs,
                    meta_attrs: None,
                    on_stack: false,
                } => {
                    apply(bx.cx, attrs);
                }
                PassMode::Indirect {
                    attrs,
                    meta_attrs: Some(meta_attrs),
                    on_stack: _,
                } => {
                    apply(bx.cx, attrs);
                    apply(bx.cx, meta_attrs);
                }
                PassMode::Pair(a, b) => {
                    apply(bx.cx, a);
                    apply(bx.cx, b);
                }
                PassMode::Cast { pad_i32, .. } => {
                    if *pad_i32 {
                        apply(bx.cx, &ArgAttributes::new());
                    }
                    apply(bx.cx, &ArgAttributes::new());
                }
            }
        }
    }
}

impl<'tcx> AbiBuilderMethods for Builder<'_, '_, 'tcx> {
    fn get_param(&mut self, index: usize) -> Self::Value {
        let val = llvm::get_param(self.llfn(), index as c_uint);
        // trace!("Get param `{:?}`", val);
        let val = unsafe {
            let llfnty = LLVMRustGetFunctionType(self.llfn());
            trace!("llfnty: {:?}", llfnty);
            // destructure so rustc doesnt complain in the call to transmute_llval
            let Self { cx, llbuilder } = self;
            let map = cx.remapped_integer_args.borrow();
            if let Some((_, key)) = map.get(&llfnty)
                && let Some((_, new_ty)) = key.iter().find(|t| t.0 == index)
            {
                trace!("Casting irregular param {:?} to {:?}", val, new_ty);
                return transmute_llval(llbuilder, cx, val, new_ty);
            }
            val
        };
        trace!("Get param `{:?}`", val);
        val
    }
}

pub(crate) trait ArgAbiExt<'ll, 'tcx> {
    fn memory_ty(&self, cx: &CodegenCx<'ll, 'tcx>) -> &'ll Type;
    fn store(
        &self,
        bx: &mut Builder<'_, 'll, 'tcx>,
        val: &'ll Value,
        dst: PlaceRef<'tcx, &'ll Value>,
    );
    fn store_fn_arg(
        &self,
        bx: &mut Builder<'_, 'll, 'tcx>,
        idx: &mut usize,
        dst: PlaceRef<'tcx, &'ll Value>,
    );
}

impl<'ll, 'tcx> ArgAbiExt<'ll, 'tcx> for ArgAbi<'tcx, Ty<'tcx>> {
    /// Gets the LLVM type for a place of the original Rust type of
    /// this argument/return, i.e., the result of `type_of::type_of`.
    fn memory_ty(&self, cx: &CodegenCx<'ll, 'tcx>) -> &'ll Type {
        self.layout.llvm_type(cx)
    }

    /// Stores a direct/indirect value described by this ArgAbi into a
    /// place for the original Rust type of this argument/return.
    /// Can be used for both storing formal arguments into Rust variables
    /// or results of call/invoke instructions into their destinations.
    fn store(
        &self,
        bx: &mut Builder<'_, 'll, 'tcx>,
        val: &'ll Value,
        dst: PlaceRef<'tcx, &'ll Value>,
    ) {
        match &self.mode {
            PassMode::Ignore => {}
            // Sized indirect arguments
            PassMode::Indirect {
                attrs,
                meta_attrs: None,
                on_stack: _,
            } => {
                let align = attrs.pointee_align.unwrap_or(self.layout.align.abi);
                OperandValue::Ref(PlaceValue::new_sized(val, align)).store(bx, dst);
            }
            // Unsized indirect arguments
            PassMode::Indirect {
                attrs: _,
                meta_attrs: Some(_),
                on_stack: _,
            } => {
                bug!("unsized `ArgAbi` must be handled through `store_fn_arg`");
            }
            PassMode::Cast { pad_i32: _, cast } => {
                trace!("store cast");
                // The ABI mandates that the value is passed as a different struct representation.
                // Spill and reload it from the stack to convert from the ABI representation to
                // the Rust representation.
                let scratch_size = cast.size(bx);
                let scratch_align = cast.align(bx);
                // Note that the ABI type may be either larger or smaller than the Rust type,
                // due to the presence or absence of trailing padding. For example:
                // - On some ABIs, the Rust layout { f64, f32, <f32 padding> } may omit padding
                //   when passed by value, making it smaller.
                // - On some ABIs, the Rust layout { u16, u16, u16 } may be padded up to 8 bytes
                //   when passed by value, making it larger.
                let copy_bytes =
                    cmp::min(cast.unaligned_size(bx).bytes(), self.layout.size.bytes());
                // Allocate some scratch space...
                let llscratch = bx.alloca(scratch_size, scratch_align);
                bx.lifetime_start(llscratch, scratch_size);
                // ...store the value...
                bx.store(val, llscratch, scratch_align);
                // ... and then memcpy it to the intended destination.
                bx.memcpy(
                    dst.val.llval,
                    self.layout.align.abi,
                    llscratch,
                    scratch_align,
                    bx.const_usize(copy_bytes),
                    MemFlags::empty(),
                );

                bx.lifetime_end(llscratch, scratch_size);
                trace!("store cast end");
            }
            _ => {
                OperandRef::from_immediate_or_packed_pair(bx, val, self.layout)
                    .val
                    .store(bx, dst);
            }
        }
    }

    fn store_fn_arg<'a>(
        &self,
        bx: &mut Builder<'a, 'll, 'tcx>,
        idx: &mut usize,
        dst: PlaceRef<'tcx, &'ll Value>,
    ) {
        let mut next = || {
            let val = llvm::get_param(bx.llfn(), *idx as c_uint);
            *idx += 1;
            val
        };
        match self.mode {
            PassMode::Ignore => {}
            PassMode::Pair(..) => {
                OperandValue::Pair(next(), next()).store(bx, dst);
            }
            PassMode::Indirect {
                attrs: _,
                meta_attrs: Some(_),
                on_stack: _,
            } => {
                let place_val = PlaceValue {
                    llval: next(),
                    llextra: Some(next()),
                    align: self.layout.align.abi,
                };
                OperandValue::Ref(place_val).store(bx, dst);
            }
            PassMode::Direct(_)
            | PassMode::Indirect {
                attrs: _,
                meta_attrs: None,
                on_stack: _,
            }
            | PassMode::Cast { .. } => {
                let next_arg = next();
                self.store(bx, next_arg, dst);
            }
        }
    }
}
