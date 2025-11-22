use std::borrow::Cow;
use std::ops::Deref;
use std::ptr;

use libc::{c_char, c_uint};
use rustc_abi as abi;
use rustc_abi::{AddressSpace, Align, HasDataLayout, Size, TargetDataLayout, WrappingRange};
use rustc_codegen_ssa::MemFlags;
use rustc_codegen_ssa::common::{AtomicRmwBinOp, IntPredicate, RealPredicate, TypeKind};
use rustc_codegen_ssa::mir::operand::{OperandRef, OperandValue};
use rustc_codegen_ssa::mir::place::PlaceRef;
use rustc_codegen_ssa::traits::*;
use rustc_data_structures::small_c_str::SmallCStr;
use rustc_hir::def_id::DefId;
use rustc_middle::bug;
use rustc_middle::middle::codegen_fn_attrs::CodegenFnAttrs;
use rustc_middle::ty::AtomicOrdering;
use rustc_middle::ty::layout::{
    FnAbiError, FnAbiOfHelpers, FnAbiRequest, HasTypingEnv, LayoutError, LayoutOfHelpers,
    TyAndLayout,
};
use rustc_middle::ty::{self, Instance, Ty, TyCtxt};
use rustc_session::config::OptLevel;
use rustc_span::Span;
use rustc_target::callconv::FnAbi;
use rustc_target::spec::{HasTargetSpec, Target};
use tracing::{debug, trace};

use crate::abi::FnAbiLlvmExt;
use crate::context::CodegenCx;
use crate::int_replace::{get_transformed_type, transmute_llval};
use crate::llvm::{self, BasicBlock, Type, Value};
use crate::ty::LayoutLlvmExt;

pub(crate) enum CountZerosKind {
    Leading,
    Trailing,
}

// All Builders must have an llfn associated with them
#[must_use]
pub(crate) struct Builder<'a, 'll, 'tcx> {
    pub llbuilder: &'ll mut llvm::Builder<'ll>,
    pub cx: &'a CodegenCx<'ll, 'tcx>,
}

impl Drop for Builder<'_, '_, '_> {
    fn drop(&mut self) {
        unsafe {
            llvm::LLVMDisposeBuilder(&mut *(self.llbuilder as *mut _));
        }
    }
}

const UNNAMED: *const c_char = c"".as_ptr();

/// Empty string, to be used where LLVM expects an instruction name, indicating
/// that the instruction is to be left unnamed (i.e. numbered, in textual IR).
pub(crate) fn unnamed() -> *const c_char {
    UNNAMED
}

impl<'ll, 'tcx> BackendTypes for Builder<'_, 'll, 'tcx> {
    type Value = <CodegenCx<'ll, 'tcx> as BackendTypes>::Value;
    type Function = <CodegenCx<'ll, 'tcx> as BackendTypes>::Function;
    type BasicBlock = <CodegenCx<'ll, 'tcx> as BackendTypes>::BasicBlock;
    type Type = <CodegenCx<'ll, 'tcx> as BackendTypes>::Type;
    type Funclet = <CodegenCx<'ll, 'tcx> as BackendTypes>::Funclet;

    type DIScope = <CodegenCx<'ll, 'tcx> as BackendTypes>::DIScope;
    type DILocation = <CodegenCx<'ll, 'tcx> as BackendTypes>::DILocation;
    type DIVariable = <CodegenCx<'ll, 'tcx> as BackendTypes>::DIVariable;

    type Metadata = <CodegenCx<'ll, 'tcx> as BackendTypes>::Metadata;
}

impl HasDataLayout for Builder<'_, '_, '_> {
    fn data_layout(&self) -> &TargetDataLayout {
        self.cx.data_layout()
    }
}

impl<'tcx> ty::layout::HasTyCtxt<'tcx> for Builder<'_, '_, 'tcx> {
    fn tcx(&self) -> TyCtxt<'tcx> {
        self.cx.tcx
    }
}

impl<'tcx> HasTypingEnv<'tcx> for Builder<'_, '_, 'tcx> {
    fn typing_env(&self) -> ty::TypingEnv<'tcx> {
        self.cx.typing_env()
    }
}

impl HasTargetSpec for Builder<'_, '_, '_> {
    fn target_spec(&self) -> &Target {
        self.cx.target_spec()
    }
}

impl<'tcx> LayoutOfHelpers<'tcx> for Builder<'_, '_, 'tcx> {
    type LayoutOfResult = TyAndLayout<'tcx>;

    #[inline]
    fn handle_layout_err(&self, err: LayoutError<'tcx>, span: Span, ty: Ty<'tcx>) -> ! {
        self.cx.handle_layout_err(err, span, ty)
    }
}

impl<'tcx> FnAbiOfHelpers<'tcx> for Builder<'_, '_, 'tcx> {
    type FnAbiOfResult = &'tcx FnAbi<'tcx, Ty<'tcx>>;

    #[inline]
    fn handle_fn_abi_err(
        &self,
        err: FnAbiError<'tcx>,
        span: Span,
        fn_abi_request: FnAbiRequest<'tcx>,
    ) -> ! {
        self.cx.handle_fn_abi_err(err, span, fn_abi_request)
    }
}

impl<'ll, 'tcx> Deref for Builder<'_, 'll, 'tcx> {
    type Target = CodegenCx<'ll, 'tcx>;

    #[inline]
    fn deref(&self) -> &Self::Target {
        self.cx
    }
}

macro_rules! imath_builder_methods {
    ($($self_:ident.$name:ident($($arg:ident),*) => $llvm_capi:ident => $op:block)+) => {
        $(fn $name(&mut $self_, $($arg: &'ll Value),*) -> &'ll Value {
            // Dispatch to i128 emulation or `compiler_builtins`-based intrinsic
            if $($self_.is_i128($arg))||*
                $op
            else {
                unsafe {
                    trace!("binary expr: {:?} with args {:?}", stringify!($name), [$($arg),*]);
                    llvm::$llvm_capi($self_.llbuilder, $($arg,)* UNNAMED)
                }
            }
        })+
    }
}

macro_rules! fmath_builder_methods {
    ($($self_:ident.$name:ident($($arg:ident),*) => $llvm_capi:ident)+) => {
        $(fn $name(&mut $self_, $($arg: &'ll Value),*) -> &'ll Value {
            unsafe {
                trace!("binary expr: {:?} with args {:?}", stringify!($name), [$($arg),*]);
                llvm::$llvm_capi($self_.llbuilder, $($arg,)* UNNAMED)
            }
        })+
    }
}

macro_rules! set_fmath_builder_methods {
    ($($name:ident($($arg:ident),*) => ($llvm_capi:ident, $llvm_set_math:ident)),+ $(,)?) => {
        $(fn $name(&mut self, $($arg: &'ll Value),*) -> &'ll Value {
            unsafe {
                let instr = llvm::$llvm_capi(self.llbuilder, $($arg,)* UNNAMED);
                llvm::$llvm_set_math(instr);
                instr
            }
        })+
    }
}

impl<'tcx> CoverageInfoBuilderMethods<'tcx> for Builder<'_, '_, 'tcx> {
    fn add_coverage(
        &mut self,
        _instance: Instance<'tcx>,
        _kind: &rustc_middle::mir::coverage::CoverageKind,
    ) {
    }
}

impl<'ll, 'tcx, 'a> BuilderMethods<'a, 'tcx> for Builder<'a, 'll, 'tcx> {
    type CodegenCx = CodegenCx<'ll, 'tcx>;

    fn build(cx: &'a CodegenCx<'ll, 'tcx>, llbb: &'ll BasicBlock) -> Self {
        let bx = Builder::with_cx(cx);
        unsafe {
            llvm::LLVMPositionBuilderAtEnd(bx.llbuilder, llbb);
        }
        bx
    }

    fn cx(&self) -> &CodegenCx<'ll, 'tcx> {
        self.cx
    }

    fn llbb(&self) -> &'ll BasicBlock {
        unsafe { llvm::LLVMGetInsertBlock(self.llbuilder) }
    }

    fn set_span(&mut self, _span: Span) {}

    fn append_block(cx: &'a CodegenCx<'ll, 'tcx>, llfn: &'ll Value, name: &str) -> &'ll BasicBlock {
        unsafe {
            let name = SmallCStr::new(name);
            llvm::LLVMAppendBasicBlockInContext(cx.llcx, llfn, name.as_ptr())
        }
    }

    fn append_sibling_block(&mut self, name: &str) -> &'ll BasicBlock {
        Self::append_block(self.cx, self.llfn(), name)
    }

    fn switch_to_block(&mut self, llbb: Self::BasicBlock) {
        *self = Self::build(self.cx, llbb)
    }

    fn ret_void(&mut self) {
        trace!("Ret void");
        unsafe {
            llvm::LLVMBuildRetVoid(self.llbuilder);
        }
    }

    fn ret(&mut self, mut v: &'ll Value) {
        trace!("Ret `{:?}`", v);
        unsafe {
            let ty = self.val_ty(v);
            let (new_ty, changed) = get_transformed_type(self.cx, ty);
            if changed {
                v = transmute_llval(self.llbuilder, self.cx, v, new_ty);
            }
            // Get the return type.
            let sig = llvm::LLVMGetElementType(self.val_ty(self.llfn()));
            let return_ty = llvm::LLVMGetReturnType(sig);
            // Check if new_ty & return_ty are different pointers.
            // FIXME: get rid of this nonsense once we are past LLVM 7 and don't have
            // to suffer from typed pointers.
            if return_ty != new_ty
                && llvm::LLVMRustGetTypeKind(return_ty) == llvm::TypeKind::Pointer
                && llvm::LLVMRustGetTypeKind(new_ty) == llvm::TypeKind::Pointer
            {
                v = llvm::LLVMBuildBitCast(
                    self.llbuilder,
                    v,
                    return_ty,
                    c"return pointer adjust".as_ptr(),
                );
            }
            llvm::LLVMBuildRet(self.llbuilder, v);
        }
    }

    fn br(&mut self, dest: &'ll BasicBlock) {
        trace!("Br");
        unsafe {
            llvm::LLVMBuildBr(self.llbuilder, dest);
        }
    }

    fn cond_br(
        &mut self,
        cond: &'ll Value,
        then_llbb: &'ll BasicBlock,
        else_llbb: &'ll BasicBlock,
    ) {
        trace!("Cond br `{:?}`", cond);
        unsafe {
            llvm::LLVMBuildCondBr(self.llbuilder, cond, then_llbb, else_llbb);
        }
    }

    fn switch(
        &mut self,
        v: &'ll Value,
        else_llbb: &'ll BasicBlock,
        cases: impl ExactSizeIterator<Item = (u128, &'ll BasicBlock)>,
    ) {
        trace!("Switch `{:?}`", v);
        let switch =
            unsafe { llvm::LLVMBuildSwitch(self.llbuilder, v, else_llbb, cases.len() as c_uint) };
        for (on_val, dest) in cases {
            let on_val = self.const_uint_big(self.val_ty(v), on_val);
            unsafe { llvm::LLVMAddCase(switch, on_val, dest) }
        }
    }

    fn invoke(
        &mut self,
        llty: &'ll Type,
        fn_attrs: Option<&CodegenFnAttrs>,
        fn_abi: Option<&FnAbi<'tcx, Ty<'tcx>>>,
        llfn: &'ll Value,
        args: &[&'ll Value],
        then: &'ll BasicBlock,
        _catch: &'ll BasicBlock,
        funclet: Option<&()>,
        instance: Option<Instance<'tcx>>,
    ) -> Self::Value {
        trace!("invoke");
        let call = self.call(llty, fn_attrs, fn_abi, llfn, args, funclet, instance);
        // exceptions arent a thing, go directly to the `then` block
        unsafe { llvm::LLVMBuildBr(self.llbuilder, then) };
        call
    }

    fn unreachable(&mut self) {
        trace!("Unreachable");
        unsafe {
            llvm::LLVMBuildUnreachable(self.llbuilder);
        }
    }

    imath_builder_methods! {
        self.add(a, b) => LLVMBuildAdd => { self.emulate_i128_add(a, b) }
        self.unchecked_uadd(a, b) => LLVMBuildNUWAdd => { self.emulate_i128_add(a, b) }
        self.unchecked_sadd(a, b) => LLVMBuildNSWAdd => { self.emulate_i128_add(a, b) }

        self.sub(a, b) => LLVMBuildSub => { self.emulate_i128_sub(a, b) }
        self.unchecked_usub(a, b) => LLVMBuildNUWSub => { self.emulate_i128_sub(a, b) }
        self.unchecked_ssub(a, b) => LLVMBuildNSWSub => { self.emulate_i128_sub(a, b) }

        self.mul(a, b) => LLVMBuildMul => { self.call_intrinsic("__nvvm_multi3", &[a, b]) }
        self.unchecked_umul(a, b) => LLVMBuildNUWMul => {
            self.call_intrinsic("__nvvm_multi3", &[a, b])
        }
        self.unchecked_smul(a, b) => LLVMBuildNSWMul => {
            self.call_intrinsic("__nvvm_multi3", &[a, b])
        }

        self.udiv(a, b) => LLVMBuildUDiv => { self.call_intrinsic("__nvvm_udivti3", &[a, b]) }
        self.exactudiv(a, b) => LLVMBuildExactUDiv => {
            self.call_intrinsic("__nvvm_udivti3", &[a, b])
        }
        self.sdiv(a, b) => LLVMBuildSDiv => { self.call_intrinsic("__nvvm_divti3", &[a, b]) }
        self.exactsdiv(a, b) => LLVMBuildExactSDiv => {
            self.call_intrinsic("__nvvm_divti3", &[a, b])
        }
        self.urem(a, b) => LLVMBuildURem => { self.call_intrinsic("__nvvm_umodti3", &[a, b]) }
        self.srem(a, b) => LLVMBuildSRem => { self.call_intrinsic("__nvvm_modti3", &[a, b]) }

        self.shl(a, b) => LLVMBuildShl => {
            // Convert shift amount to i32 for compiler-builtins.
            let b = self.trunc(b, self.type_i32());
            self.call_intrinsic("__nvvm_ashlti3", &[a, b])
        }
        self.lshr(a, b) => LLVMBuildLShr => {
            // Convert shift amount to i32 for compiler-builtins.
            let b = self.trunc(b, self.type_i32());
            self.call_intrinsic("__nvvm_lshrti3", &[a, b])
        }
        self.ashr(a, b) => LLVMBuildAShr => {
            // Convert shift amount to i32 for compiler-builtins.
            let b = self.trunc(b, self.type_i32());
            self.call_intrinsic("__nvvm_ashrti3", &[a, b])
        }
        self.and(a, b) => LLVMBuildAnd => { self.emulate_i128_and(a, b) }
        self.or(a, b) => LLVMBuildOr => { self.emulate_i128_or(a, b) }
        self.xor(a, b) => LLVMBuildXor => { self.emulate_i128_xor(a, b) }
        self.neg(a) => LLVMBuildNeg => { self.emulate_i128_neg(a) }
        self.not(a) => LLVMBuildNot => { self.emulate_i128_not(a) }
    }

    fmath_builder_methods! {
        self.fadd(a, b) => LLVMBuildFAdd
        self.fsub(a, b) => LLVMBuildFSub
        self.fmul(a, b) => LLVMBuildFMul
        self.fdiv(a, b) => LLVMBuildFDiv
        self.frem(a, b) => LLVMBuildFRem
        self.fneg(a) => LLVMBuildFNeg
    }

    set_fmath_builder_methods! {
        fadd_fast(x, y) => (LLVMBuildFAdd, LLVMRustSetFastMath),
        fsub_fast(x, y) => (LLVMBuildFSub, LLVMRustSetFastMath),
        fmul_fast(x, y) => (LLVMBuildFMul, LLVMRustSetFastMath),
        fdiv_fast(x, y) => (LLVMBuildFDiv, LLVMRustSetFastMath),
        frem_fast(x, y) => (LLVMBuildFRem, LLVMRustSetFastMath),
        fadd_algebraic(x, y) => (LLVMBuildFAdd, LLVMRustSetAlgebraicMath),
        fsub_algebraic(x, y) => (LLVMBuildFSub, LLVMRustSetAlgebraicMath),
        fmul_algebraic(x, y) => (LLVMBuildFMul, LLVMRustSetAlgebraicMath),
        fdiv_algebraic(x, y) => (LLVMBuildFDiv, LLVMRustSetAlgebraicMath),
        frem_algebraic(x, y) => (LLVMBuildFRem, LLVMRustSetAlgebraicMath),
    }

    fn checked_binop(
        &mut self,
        oop: OverflowOp,
        ty: Ty<'_>,
        lhs: Self::Value,
        rhs: Self::Value,
    ) -> (Self::Value, Self::Value) {
        trace!(
            "Checked binop `{:?}`, lhs: `{:?}`, rhs: `{:?}`",
            ty, lhs, rhs
        );
        use rustc_middle::ty::IntTy::*;
        use rustc_middle::ty::UintTy::*;
        use rustc_middle::ty::{Int, Uint};

        let new_kind = match ty.kind() {
            Int(t @ Isize) => Int(t.normalize(self.tcx.sess.target.pointer_width)),
            Uint(t @ Usize) => Uint(t.normalize(self.tcx.sess.target.pointer_width)),
            t @ (Uint(_) | Int(_)) => *t,
            _ => panic!("tried to get overflow intrinsic for op applied to non-int type"),
        };

        let name = match oop {
            OverflowOp::Add => match new_kind {
                Int(I8) => "__nvvm_i8_addo",
                Int(I16) => "llvm.sadd.with.overflow.i16",
                Int(I32) => "llvm.sadd.with.overflow.i32",
                Int(I64) => "llvm.sadd.with.overflow.i64",
                Int(I128) => "__nvvm_i128_addo",

                Uint(U8) => "__nvvm_u8_addo",
                Uint(U16) => "llvm.uadd.with.overflow.i16",
                Uint(U32) => "llvm.uadd.with.overflow.i32",
                Uint(U64) => "llvm.uadd.with.overflow.i64",
                Uint(U128) => "__nvvm_u128_addo",
                _ => unreachable!(),
            },
            OverflowOp::Sub => match new_kind {
                Int(I8) => "__nvvm_i8_subo",
                Int(I16) => "llvm.ssub.with.overflow.i16",
                Int(I32) => "llvm.ssub.with.overflow.i32",
                Int(I64) => "llvm.ssub.with.overflow.i64",
                Int(I128) => "__nvvm_i128_subo",

                Uint(U8) => "__nvvm_u8_subo",
                Uint(U16) => "llvm.usub.with.overflow.i16",
                Uint(U32) => "llvm.usub.with.overflow.i32",
                Uint(U64) => "llvm.usub.with.overflow.i64",
                Uint(U128) => "__nvvm_u128_subo",

                _ => unreachable!(),
            },
            OverflowOp::Mul => match new_kind {
                Int(I8) => "__nvvm_i8_mulo",
                Int(I16) => "llvm.smul.with.overflow.i16",
                Int(I32) => "llvm.smul.with.overflow.i32",
                Int(I64) => "llvm.smul.with.overflow.i64",
                Int(I128) => "__nvvm_i128_mulo",

                Uint(U8) => "__nvvm_u8_mulo",
                Uint(U16) => "llvm.umul.with.overflow.i16",
                Uint(U32) => "llvm.umul.with.overflow.i32",
                Uint(U64) => "llvm.umul.with.overflow.i64",
                Uint(U128) => "__nvvm_u128_mulo",

                _ => unreachable!(),
            },
        };

        let (ty, f) = self.get_intrinsic(name);
        let res = self.call(ty, None, None, f, &[lhs, rhs], None, None);
        (self.extract_value(res, 0), self.extract_value(res, 1))
    }

    fn from_immediate(&mut self, val: Self::Value) -> Self::Value {
        if self.cx().val_ty(val) == self.cx().type_i1() {
            self.zext(val, self.cx().type_i8())
        } else {
            val
        }
    }
    fn to_immediate_scalar(&mut self, val: Self::Value, scalar: abi::Scalar) -> Self::Value {
        if scalar.is_bool() {
            return self.trunc(val, self.cx().type_i1());
        }
        val
    }

    fn alloca(&mut self, size: Size, align: Align) -> &'ll Value {
        trace!("Alloca `{:?}`", size);
        let mut bx = Builder::with_cx(self.cx);
        bx.position_at_start(unsafe { llvm::LLVMGetFirstBasicBlock(self.llfn()) });
        let ty = self.cx().type_array(self.cx().type_i8(), size.bytes());
        unsafe {
            let alloca = llvm::LLVMBuildAlloca(bx.llbuilder, ty, UNNAMED);
            llvm::LLVMSetAlignment(alloca, align.bytes() as c_uint);
            // Cast to default addrspace if necessary
            let alloca_ty = llvm::LLVMTypeOf(alloca);
            let alloca_addrspace = llvm::LLVMGetPointerAddressSpace(alloca_ty);
            let dest_ty = self.cx().type_ptr();
            let dest_addrspace = llvm::LLVMGetPointerAddressSpace(dest_ty);
            if alloca_addrspace != dest_addrspace {
                llvm::LLVMBuildAddrSpaceCast(bx.llbuilder, alloca, dest_ty, UNNAMED)
            } else {
                alloca
            }
        }
    }

    fn load(&mut self, ty: &'ll Type, ptr: &'ll Value, align: Align) -> &'ll Value {
        trace!("Load {ty:?} {:?}", ptr);
        let ptr = self.pointercast(ptr, self.cx.type_ptr_to(ty));
        unsafe {
            let load = llvm::LLVMBuildLoad(self.llbuilder, ptr, UNNAMED);
            llvm::LLVMSetAlignment(load, align.bytes() as c_uint);
            load
        }
    }

    fn volatile_load(&mut self, ty: &'ll Type, ptr: &'ll Value) -> &'ll Value {
        trace!("Volatile load `{:?}`", ptr);
        let ptr = self.pointercast(ptr, self.cx.type_ptr_to(ty));
        unsafe {
            let load = llvm::LLVMBuildLoad(self.llbuilder, ptr, UNNAMED);
            llvm::LLVMSetVolatile(load, llvm::True);
            load
        }
    }

    fn atomic_load(
        &mut self,
        ty: &'ll Type,
        ptr: &'ll Value,
        order: AtomicOrdering,
        _size: Size,
    ) -> &'ll Value {
        // Since for any A, A | 0 = A, and performing atomics on constant memory is UB in Rust, we
        // can abuse bitwise-or to perform atomic reads.
        //
        // njn: is `ty` the type of the loaded value, or the type of the
        // pointer to the loaded-from address? i.e. `T` or `*const T`? I'm
        // assuming `T`
        let ret_ptr = unsafe { llvm::LLVMRustGetTypeKind(ty) == llvm::TypeKind::Pointer };
        self.atomic_rmw(
            AtomicRmwBinOp::AtomicOr,
            ptr,
            self.const_int(ty, 0),
            order,
            ret_ptr,
        )
    }

    fn load_operand(&mut self, place: PlaceRef<'tcx, &'ll Value>) -> OperandRef<'tcx, &'ll Value> {
        if place.layout.is_unsized() {
            let tail = self
                .tcx
                .struct_tail_for_codegen(place.layout.ty, self.typing_env());
            if matches!(tail.kind(), ty::Foreign(..)) {
                // Unsized locals and, at least conceptually, even unsized arguments must be copied
                // around, which requires dynamically determining their size. Therefore, we cannot
                // allow `extern` types here. Consult t-opsem before removing this check.
                panic!("unsized locals must not be `extern` types");
            }
        }
        assert_eq!(place.val.llextra.is_some(), place.layout.is_unsized());

        if place.layout.is_zst() {
            return OperandRef::zero_sized(place.layout);
        }

        fn scalar_load_metadata<'ll, 'tcx>(
            bx: &mut Builder<'_, 'll, 'tcx>,
            load: &'ll Value,
            scalar: abi::Scalar,
            layout: TyAndLayout<'tcx>,
            offset: Size,
        ) {
            if bx.sess().opts.optimize == OptLevel::No {
                // Don't emit metadata we're not going to use
                return;
            }

            if !scalar.is_uninit_valid() {
                bx.noundef_metadata(load);
            }

            match scalar.primitive() {
                abi::Primitive::Int(..) => {
                    if !scalar.is_always_valid(bx) {
                        bx.range_metadata(load, scalar.valid_range(bx));
                    }
                }
                abi::Primitive::Pointer(_) => {
                    if !scalar.valid_range(bx).contains(0) {
                        bx.nonnull_metadata(load);
                    }

                    if let Some(pointee) = layout.pointee_info_at(bx, offset)
                        && pointee.safe.is_some()
                    {
                        bx.align_metadata(load, pointee.align);
                    }
                }
                abi::Primitive::Float(_) => {}
            }
        }

        let val = if place.val.llextra.is_some() {
            // FIXME: Merge with the `else` below?
            OperandValue::Ref(place.val)
        } else if place.layout.is_llvm_immediate() {
            let mut const_llval = None;
            let llty = place.layout.llvm_type(self);
            unsafe {
                if let Some(global) = llvm::LLVMIsAGlobalVariable(place.val.llval)
                    && llvm::LLVMIsGlobalConstant(global) == llvm::True
                    && let Some(init) = llvm::LLVMGetInitializer(global)
                    && self.val_ty(init) == llty
                {
                    const_llval = Some(init);
                }
            }

            let llval = const_llval.unwrap_or_else(|| {
                let load = self.load(llty, place.val.llval, place.val.align);
                if let abi::BackendRepr::Scalar(scalar) = place.layout.backend_repr {
                    scalar_load_metadata(self, load, scalar, place.layout, Size::ZERO);
                    self.to_immediate_scalar(load, scalar)
                } else {
                    load
                }
            });

            OperandValue::Immediate(llval)
        } else if let abi::BackendRepr::ScalarPair(a, b) = place.layout.backend_repr {
            let b_offset = a.size(self).align_to(b.align(self).abi);

            let mut load = |i, scalar: abi::Scalar, layout, align, offset| {
                let llptr = if i == 0 {
                    place.val.llval
                } else {
                    self.inbounds_ptradd(place.val.llval, self.const_usize(b_offset.bytes()))
                };
                let llty = place.layout.scalar_pair_element_llvm_type(self, i, false);
                let load = self.load(llty, llptr, align);
                scalar_load_metadata(self, load, scalar, layout, offset);
                self.to_immediate_scalar(load, scalar)
            };

            OperandValue::Pair(
                load(0, a, place.layout, place.val.align, Size::ZERO),
                load(
                    1,
                    b,
                    place.layout,
                    place.val.align.restrict_for_offset(b_offset),
                    b_offset,
                ),
            )
        } else {
            OperandValue::Ref(place.val)
        };

        OperandRef {
            val,
            layout: place.layout,
        }
    }

    fn write_operand_repeatedly(
        &mut self,
        cg_elem: OperandRef<'tcx, &'ll Value>,
        count: u64,
        dest: PlaceRef<'tcx, &'ll Value>,
    ) {
        trace!("write operand repeatedly");
        let zero = self.const_usize(0);
        let count = self.const_usize(count);

        let header_bb = self.append_sibling_block("repeat_loop_header");
        let body_bb = self.append_sibling_block("repeat_loop_body");
        let next_bb = self.append_sibling_block("repeat_loop_next");

        self.br(header_bb);

        let mut header_bx = Self::build(self.cx, header_bb);
        let i = header_bx.phi(self.val_ty(zero), &[zero], &[self.llbb()]);

        let keep_going = header_bx.icmp(IntPredicate::IntULT, i, count);
        header_bx.cond_br(keep_going, body_bb, next_bb);

        let mut body_bx = Self::build(self.cx, body_bb);
        let dest_elem = dest.project_index(&mut body_bx, i);
        cg_elem.val.store(&mut body_bx, dest_elem);

        let next = body_bx.unchecked_uadd(i, self.const_usize(1));
        body_bx.br(header_bb);
        header_bx.add_incoming_to_phi(i, next, body_bb);

        *self = Self::build(self.cx, next_bb);
    }

    fn range_metadata(&mut self, load: &'ll Value, range: WrappingRange) {
        trace!("range metadata on {load:?}: {range:?}");
        unsafe {
            let llty = self.cx.val_ty(load);
            let v = [
                self.cx.const_uint_big(llty, range.start),
                self.cx.const_uint_big(llty, range.end.wrapping_add(1)),
            ];

            llvm::LLVMSetMetadata(
                load,
                llvm::MetadataType::MD_range as c_uint,
                llvm::LLVMMDNodeInContext(self.cx.llcx, v.as_ptr(), v.len() as c_uint),
            );
        }
    }

    fn nonnull_metadata(&mut self, load: &'ll Value) {
        unsafe {
            llvm::LLVMSetMetadata(
                load,
                llvm::MetadataType::MD_nonnull as c_uint,
                llvm::LLVMMDNodeInContext(self.cx.llcx, ptr::null(), 0),
            );
        }
    }

    fn store(&mut self, val: &'ll Value, ptr: &'ll Value, align: Align) -> &'ll Value {
        trace!("Store val `{:?}` into ptr `{:?}`", val, ptr);
        self.store_with_flags(val, ptr, align, MemFlags::empty())
    }

    fn store_with_flags(
        &mut self,
        val: &'ll Value,
        ptr: &'ll Value,
        align: Align,
        flags: MemFlags,
    ) -> &'ll Value {
        trace!(
            "store_with_flags: {:?} into {:?} with align {:?}",
            val,
            ptr,
            align.bytes()
        );
        assert_eq!(self.cx.type_kind(self.cx.val_ty(ptr)), TypeKind::Pointer);
        let ptr = self.check_store(val, ptr);
        unsafe {
            let store = llvm::LLVMBuildStore(self.llbuilder, val, ptr);
            let align = if flags.contains(MemFlags::UNALIGNED) {
                1
            } else {
                align.bytes() as c_uint
            };
            llvm::LLVMSetAlignment(store, align);
            if flags.contains(MemFlags::VOLATILE) {
                llvm::LLVMSetVolatile(store, llvm::True);
            }
            if flags.contains(MemFlags::NONTEMPORAL) {
                // According to LLVM [1] building a nontemporal store must
                // *always* point to a metadata value of the integer 1.
                //
                // [1]: http://llvm.org/docs/LangRef.html#store-instruction
                let one = self.cx.const_i32(1);
                let node = llvm::LLVMMDNodeInContext(self.cx.llcx, &one, 1);
                llvm::LLVMSetMetadata(store, llvm::MetadataType::MD_nontemporal as c_uint, node);
            }

            store
        }
    }

    fn atomic_store(
        &mut self,
        val: &'ll Value,
        ptr: &'ll Value,
        order: AtomicOrdering,
        _size: Size,
    ) {
        // We can exchange *ptr with val, and then discard the result.
        let ret_ptr =
            unsafe { llvm::LLVMRustGetTypeKind(llvm::LLVMTypeOf(val)) == llvm::TypeKind::Pointer };
        self.atomic_rmw(AtomicRmwBinOp::AtomicXchg, ptr, val, order, ret_ptr);
    }

    fn gep(&mut self, ty: &'ll Type, ptr: &'ll Value, indices: &[&'ll Value]) -> &'ll Value {
        trace!("gep: {ty:?} {:?} with indices {:?}", ptr, indices);
        let ptr = self.pointercast(ptr, self.cx().type_ptr_to(ty));
        unsafe {
            llvm::LLVMBuildGEP2(
                self.llbuilder,
                ty,
                ptr,
                indices.as_ptr(),
                indices.len() as c_uint,
                UNNAMED,
            )
        }
    }

    fn inbounds_gep(
        &mut self,
        ty: &'ll Type,
        ptr: &'ll Value,
        indices: &[&'ll Value],
    ) -> &'ll Value {
        trace!("gep inbounds: {ty:?} {:?} with indices {:?}", ptr, indices);
        let ptr = self.pointercast(ptr, self.cx().type_ptr_to(ty));
        unsafe {
            llvm::LLVMBuildInBoundsGEP2(
                self.llbuilder,
                ty,
                ptr,
                indices.as_ptr(),
                indices.len() as c_uint,
                UNNAMED,
            )
        }
    }

    /* Casts */
    fn trunc(&mut self, val: &'ll Value, dest_ty: &'ll Type) -> &'ll Value {
        trace!("trunc {:?} to {:?}", val, dest_ty);
        unsafe { llvm::LLVMBuildTrunc(self.llbuilder, val, dest_ty, UNNAMED) }
    }

    fn sext(&mut self, val: &'ll Value, dest_ty: &'ll Type) -> &'ll Value {
        trace!("sext {:?} to {:?}", val, dest_ty);
        unsafe { llvm::LLVMBuildSExt(self.llbuilder, val, dest_ty, UNNAMED) }
    }

    fn fptoui_sat(&mut self, val: &'ll Value, dest_ty: &'ll Type) -> &'ll Value {
        // NVVM does not support saturating casts, however, they are relatively simple to implement.
        // (at least for unsigned ints). So, we emulate them here.

        // In order to clamp the value, we need to know it's type.
        let val_ty = self.val_ty(val);
        // Find the min / max intrinsics
        let (min, max) = match self.cx().float_width(val_ty) {
            64 => ("__nv_fmin", "__nv_fmax"),
            32 => ("__nv_fminf", "__nv_fmaxf"),
            _ => {
                self.cx().sess().dcx()
                    .warn("Saturated float to int conversion is not supported in NVVM for type {val_ty:?}. Defaulting to UB prone conversion.");
                return self.fptoui(val, dest_ty);
            }
        };
        let (max_ty, max) = self.cx().get_intrinsic(max);
        let (min_ty, min) = self.cx().get_intrinsic(min);
        // Find the zero value, and the max value of a given int.
        let zero = self.const_real(val_ty, 0.0);
        let max_value = match self.int_width(dest_ty) {
            8 => u8::MAX as f64,
            16 => u16::MAX as f64,
            32 => u32::MAX as f64,
            64 => u64::MAX as f64,
            128 => u128::MAX as f64,
            _ => todo!("Unsupported int type {dest_ty:?}"),
        };
        let max_value = self.const_real(val_ty, max_value);
        // Compute max(val, 0). This will clamp negative values to zero **AND**
        // replace NaNs with 0s(just like how Rust is specified to behave)
        let res = self.call(max_ty, None, None, max, &[val, zero], None, None);
        // Clamp all values higher than max to max
        let res = self.call(min_ty, None, None, min, &[res, max_value], None, None);
        // Now, we know that `res` is non-nan, and in range (min, max). So, it is well-defined
        // for all inputs :D!
        self.fptoui(res, dest_ty)
    }

    fn fptosi_sat(&mut self, val: &'ll Value, dest_ty: &'ll Type) -> &'ll Value {
        // NVVM does not have support for saturated conversion. Setting rustc flag
        // `-Z saturating_float_casts=false` falls back to non-saturated, UB-prone
        // conversion, and should prevent this codegen. Otherwise, fall back to UB
        // prone conversion.
        self.cx().sess().dcx()
            .warn("Saturated float to int conversion is not supported on NVVM. Defaulting to UB prone conversion.");
        self.fptosi(val, dest_ty)
    }

    fn fptoui(&mut self, val: &'ll Value, dest_ty: &'ll Type) -> &'ll Value {
        trace!("fptoui {:?} to {:?}", val, dest_ty);
        unsafe { llvm::LLVMBuildFPToUI(self.llbuilder, val, dest_ty, UNNAMED) }
    }

    fn fptosi(&mut self, val: &'ll Value, dest_ty: &'ll Type) -> &'ll Value {
        trace!("fptosi {:?} to {:?}", val, dest_ty);
        unsafe { llvm::LLVMBuildFPToSI(self.llbuilder, val, dest_ty, UNNAMED) }
    }

    fn uitofp(&mut self, val: &'ll Value, dest_ty: &'ll Type) -> &'ll Value {
        trace!("uitofp {:?} to {:?}", val, dest_ty);
        unsafe { llvm::LLVMBuildUIToFP(self.llbuilder, val, dest_ty, UNNAMED) }
    }

    fn sitofp(&mut self, val: &'ll Value, dest_ty: &'ll Type) -> &'ll Value {
        trace!("sitofp {:?} to {:?}", val, dest_ty);
        unsafe { llvm::LLVMBuildSIToFP(self.llbuilder, val, dest_ty, UNNAMED) }
    }

    fn fptrunc(&mut self, val: &'ll Value, dest_ty: &'ll Type) -> &'ll Value {
        trace!("fptrunc {:?} to {:?}", val, dest_ty);
        unsafe { llvm::LLVMBuildFPTrunc(self.llbuilder, val, dest_ty, UNNAMED) }
    }

    fn fpext(&mut self, val: &'ll Value, dest_ty: &'ll Type) -> &'ll Value {
        trace!("fpext {:?} to {:?}", val, dest_ty);
        unsafe { llvm::LLVMBuildFPExt(self.llbuilder, val, dest_ty, UNNAMED) }
    }

    fn ptrtoint(&mut self, val: &'ll Value, dest_ty: &'ll Type) -> &'ll Value {
        trace!("ptrtoint {:?} to {:?}", val, dest_ty);
        unsafe { llvm::LLVMBuildPtrToInt(self.llbuilder, val, dest_ty, UNNAMED) }
    }

    fn inttoptr(&mut self, val: &'ll Value, dest_ty: &'ll Type) -> &'ll Value {
        trace!("inttoptr {:?} to {:?}", val, dest_ty);
        unsafe { llvm::LLVMBuildIntToPtr(self.llbuilder, val, dest_ty, UNNAMED) }
    }

    fn bitcast(&mut self, mut val: &'ll Value, dest_ty: &'ll Type) -> &'ll Value {
        trace!("Bitcast `{:?}` to ty `{:?}`", val, dest_ty);
        unsafe {
            let ty = self.val_ty(val);
            let kind = llvm::LLVMRustGetTypeKind(ty);
            if kind == llvm::TypeKind::Pointer {
                let element = self.element_type(ty);
                let addrspace = llvm::LLVMGetPointerAddressSpace(ty);
                let new_ty = self.type_ptr_to_ext(element, AddressSpace::ZERO);
                if addrspace != 0 {
                    trace!("injecting addrspace cast for `{:?}` to `{:?}`", ty, new_ty);
                    val = llvm::LLVMBuildAddrSpaceCast(self.llbuilder, val, new_ty, UNNAMED);
                }
            }
            llvm::LLVMBuildBitCast(self.llbuilder, val, dest_ty, UNNAMED)
        }
    }

    fn intcast(&mut self, val: &'ll Value, dest_ty: &'ll Type, is_signed: bool) -> &'ll Value {
        trace!("Intcast `{:?}` to ty `{:?}`", val, dest_ty);
        unsafe { llvm::LLVMRustBuildIntCast(self.llbuilder, val, dest_ty, is_signed) }
    }

    fn pointercast(&mut self, val: &'ll Value, dest_ty: &'ll Type) -> &'ll Value {
        trace!("Pointercast `{:?}` to ty `{:?}`", val, dest_ty);
        unsafe {
            let val_ty = self.val_ty(val);
            let val_addrspace = llvm::LLVMGetPointerAddressSpace(val_ty);
            let dest_addrspace = llvm::LLVMGetPointerAddressSpace(dest_ty);

            if val_addrspace != dest_addrspace {
                // Address spaces differ, use addrspacecast
                llvm::LLVMBuildAddrSpaceCast(self.llbuilder, val, dest_ty, unnamed())
            } else {
                // Same address space, use regular pointer cast
                llvm::LLVMBuildPointerCast(self.llbuilder, val, dest_ty, unnamed())
            }
        }
    }

    /* Comparisons */
    fn icmp(&mut self, op: IntPredicate, mut lhs: &'ll Value, rhs: &'ll Value) -> &'ll Value {
        trace!("Icmp lhs: `{:?}`, rhs: `{:?}`", lhs, rhs);
        // FIXME(FractalFir): Once again, a bunch of nosense to make the LLVM typed pointers happy.
        // Get rid of this as soon as we move past typed pointers.
        let lhs_ty = self.val_ty(lhs);
        let rhs_ty = self.val_ty(rhs);
        if lhs_ty != rhs_ty {
            lhs = unsafe {
                llvm::LLVMBuildBitCast(self.llbuilder, lhs, rhs_ty, c"icmp_cast".as_ptr())
            };
        }
        unsafe {
            let op = llvm::IntPredicate::from_generic(op);
            llvm::LLVMBuildICmp(self.llbuilder, op as c_uint, lhs, rhs, unnamed())
        }
    }

    fn fcmp(&mut self, op: RealPredicate, lhs: &'ll Value, rhs: &'ll Value) -> &'ll Value {
        trace!("Fcmp lhs: `{:?}`, rhs: `{:?}`", lhs, rhs);
        unsafe { llvm::LLVMBuildFCmp(self.llbuilder, op as c_uint, lhs, rhs, unnamed()) }
    }

    /* Miscellaneous instructions */
    fn memcpy(
        &mut self,
        dst: &'ll Value,
        dst_align: Align,
        src: &'ll Value,
        src_align: Align,
        size: &'ll Value,
        flags: MemFlags,
    ) {
        assert!(
            !flags.contains(MemFlags::NONTEMPORAL),
            "non-temporal memcpy not supported"
        );
        let size = self.intcast(size, self.type_isize(), false);
        let is_volatile = flags.contains(MemFlags::VOLATILE);
        unsafe {
            llvm::LLVMRustBuildMemCpy(
                self.llbuilder,
                dst,
                dst_align.bytes() as c_uint,
                src,
                src_align.bytes() as c_uint,
                size,
                is_volatile,
            );
        }
    }

    fn memmove(
        &mut self,
        dst: &'ll Value,
        dst_align: Align,
        src: &'ll Value,
        src_align: Align,
        size: &'ll Value,
        flags: MemFlags,
    ) {
        assert!(
            !flags.contains(MemFlags::NONTEMPORAL),
            "non-temporal memmove not supported"
        );
        let size = self.intcast(size, self.type_isize(), false);
        let is_volatile = flags.contains(MemFlags::VOLATILE);
        unsafe {
            llvm::LLVMRustBuildMemMove(
                self.llbuilder,
                dst,
                dst_align.bytes() as c_uint,
                src,
                src_align.bytes() as c_uint,
                size,
                is_volatile,
            );
        }
    }

    fn memset(
        &mut self,
        ptr: &'ll Value,
        fill_byte: &'ll Value,
        size: &'ll Value,
        align: Align,
        flags: MemFlags,
    ) {
        let is_volatile = flags.contains(MemFlags::VOLATILE);
        unsafe {
            llvm::LLVMRustBuildMemSet(
                self.llbuilder,
                ptr,
                align.bytes() as c_uint,
                fill_byte,
                size,
                is_volatile,
            );
        }
    }

    fn select(
        &mut self,
        mut cond: &'ll Value,
        then_val: &'ll Value,
        else_val: &'ll Value,
    ) -> &'ll Value {
        unsafe {
            if self.val_ty(cond) == llvm::LLVMVectorType(self.type_i1(), 2) {
                cond = self.const_bool(false);
            }
            llvm::LLVMBuildSelect(self.llbuilder, cond, then_val, else_val, unnamed())
        }
    }

    fn va_arg(&mut self, list: &'ll Value, ty: &'ll Type) -> &'ll Value {
        unsafe { llvm::LLVMBuildVAArg(self.llbuilder, list, ty, unnamed()) }
    }

    fn extract_element(&mut self, vec: &'ll Value, idx: &'ll Value) -> &'ll Value {
        trace!("extract element {:?}, {:?}", vec, idx);
        unsafe { llvm::LLVMBuildExtractElement(self.llbuilder, vec, idx, unnamed()) }
    }

    fn vector_splat(&mut self, _num_elts: usize, _elt: &'ll Value) -> &'ll Value {
        self.unsupported("vector splats");
    }

    fn extract_value(&mut self, agg_val: &'ll Value, idx: u64) -> &'ll Value {
        trace!("extract value {:?}, {:?}", agg_val, idx);
        assert_eq!(idx as c_uint as u64, idx);
        unsafe { llvm::LLVMBuildExtractValue(self.llbuilder, agg_val, idx as c_uint, unnamed()) }
    }

    fn insert_value(&mut self, agg_val: &'ll Value, mut elt: &'ll Value, idx: u64) -> &'ll Value {
        trace!("insert value {:?}, {:?}, {:?}", agg_val, elt, idx);
        assert_eq!(idx as c_uint as u64, idx);

        let elt_ty = self.cx.val_ty(elt);
        if self.cx.type_kind(elt_ty) == TypeKind::Pointer {
            let agg_ty = self.cx.val_ty(agg_val);
            let idx_ty = match self.cx.type_kind(agg_ty) {
                TypeKind::Struct => unsafe {
                    llvm::LLVMStructGetTypeAtIndex(agg_ty, idx as c_uint)
                },
                TypeKind::Array => unsafe { llvm::LLVMGetElementType(agg_ty) },
                _ => bug!(
                    "insert_value: expected struct or array type, found {:?}",
                    self.cx.type_kind(agg_ty)
                ),
            };
            assert_eq!(self.cx.type_kind(idx_ty), TypeKind::Pointer);
            if idx_ty != elt_ty {
                elt = self.pointercast(elt, idx_ty);
            }
        }

        unsafe { llvm::LLVMBuildInsertValue(self.llbuilder, agg_val, elt, idx as c_uint, UNNAMED) }
    }

    fn cleanup_landing_pad(&mut self, _pers_fn: &'ll Value) -> (&'ll Value, &'ll Value) {
        todo!()
    }

    fn filter_landing_pad(&mut self, _pers_fn: &'ll Value) {
        todo!()
    }

    fn resume(&mut self, _exn0: &'ll Value, _exn1: &'ll Value) {
        self.unsupported("resumes");
    }

    fn cleanup_pad(&mut self, _parent: Option<&'ll Value>, _args: &[&'ll Value]) {}

    fn cleanup_ret(&mut self, _funclet: &(), _unwind: Option<&'ll BasicBlock>) {}

    fn catch_pad(&mut self, _parent: &'ll Value, _args: &[&'ll Value]) {}

    fn catch_switch(
        &mut self,
        _parent: Option<&'ll Value>,
        _unwind: Option<&'ll BasicBlock>,
        _handlers: &[&'ll BasicBlock],
    ) -> &'ll Value {
        self.unsupported("catch switches");
    }

    fn set_personality_fn(&mut self, _personality: &'ll Value) {}

    // Atomic Operations
    fn atomic_cmpxchg(
        &mut self,
        dst: &'ll Value,
        cmp: &'ll Value,
        src: &'ll Value,
        order: AtomicOrdering,
        failure_order: AtomicOrdering,
        weak: bool,
    ) -> (&'ll Value, &'ll Value) {
        // LLVM verifier rejects cases where the `failure_order` is stronger than `order`
        match (order, failure_order) {
            // Failure order `Release` & `AcqRel` is simply invalid.
            (_, AtomicOrdering::Release | AtomicOrdering::AcqRel) => {
                self.abort();
                return (
                    self.const_undef(self.val_ty(cmp)),
                    self.const_undef(self.type_i1()),
                );
            }
            // Success & failure ordering are the same - OK.
            (AtomicOrdering::SeqCst, AtomicOrdering::SeqCst)
            | (AtomicOrdering::Relaxed, AtomicOrdering::Relaxed)
            | (AtomicOrdering::Acquire, AtomicOrdering::Acquire) => (),
            // Failure is `SeqCst`(strongest) & success is anything else(weaker) - reject.
            (_, AtomicOrdering::SeqCst) => {
                self.abort();
                return (
                    self.const_undef(self.val_ty(cmp)),
                    self.const_undef(self.type_i1()),
                );
            }
            // Failure is Relaxed(weakest), and success is anything - OK.
            (_, AtomicOrdering::Relaxed) => (),
            // Failure is anything, and success is SeqCest(strongest) - OK.
            (AtomicOrdering::SeqCst, _) => (),
            // Failure is Acquire, and success is Release - OK.
            (AtomicOrdering::Release, AtomicOrdering::Acquire) => (),
            // Success is AcqRel & failure is Acquire - OK
            (AtomicOrdering::AcqRel, AtomicOrdering::Acquire) => (),
            // Success is weaker than failure - reject.
            (AtomicOrdering::Relaxed, AtomicOrdering::Acquire) => {
                self.abort();
                return (
                    self.const_undef(self.val_ty(cmp)),
                    self.const_undef(self.type_i1()),
                );
            }
        };
        let res = self.atomic_op(
            dst,
            |builder, dst| {
                // We are in a supported address space - just use ordinary atomics
                unsafe {
                    llvm::LLVMRustBuildAtomicCmpXchg(
                        builder.llbuilder,
                        dst,
                        cmp,
                        src,
                        crate::llvm::AtomicOrdering::from_generic(order),
                        crate::llvm::AtomicOrdering::from_generic(failure_order),
                        weak as u32,
                    )
                }
            },
            |builder, dst| {
                // Local space is only accessible to the current thread.
                // So, there are no synchronization issues, and we can emulate it using a simple load / compare / store.
                let load: &'ll Value =
                    unsafe { llvm::LLVMBuildLoad(builder.llbuilder, dst, UNNAMED) };
                let compare = builder.icmp(IntPredicate::IntEQ, load, cmp);
                // We can do something smart & branchless here:
                // We select either the current value(if the comparison fails), or a new value.
                // We then *undconditionally* write that back to local memory(which is very, very cheap).
                // TODO: measure if this has a positive impact, or if we should just use more blocks, and conditional writes.
                let value = builder.select(compare, src, load);
                unsafe { llvm::LLVMBuildStore(builder.llbuilder, value, dst) };
                let res_type =
                    builder.type_struct(&[builder.val_ty(cmp), builder.type_ix(1)], false);
                // We pack the result, to match the behaviour of proper atomics / emulated thread-local atomics.
                let res = builder.const_undef(res_type);
                let res = builder.insert_value(res, load, 0);
                builder.insert_value(res, compare, 1)
            },
        );
        // Unpack the result
        let val = self.extract_value(res, 0);
        let success = self.extract_value(res, 1);
        (val, success)
    }

    fn atomic_rmw(
        &mut self,
        op: AtomicRmwBinOp,
        dst: &'ll Value,
        src: &'ll Value,
        order: AtomicOrdering,
        ret_ptr: bool,
    ) -> &'ll Value {
        if matches!(op, AtomicRmwBinOp::AtomicNand) {
            self.fatal("Atomic NAND not supported yet!")
        }
        let mut res = self.atomic_op(
            dst,
            |builder, dst| {
                // We are in a supported address space - just use ordinary atomics
                unsafe {
                    llvm::LLVMBuildAtomicRMW(
                        builder.llbuilder,
                        op.into(),
                        dst,
                        src,
                        crate::llvm::AtomicOrdering::from_generic(order),
                        0,
                    )
                }
            },
            |builder, dst| {
                // Local space is only accessible to the current thread. So, there are no
                // synchronization issues, and we can emulate it using a simple load/compare/store.
                let load: &'ll Value =
                    unsafe { llvm::LLVMBuildLoad(builder.llbuilder, dst, UNNAMED) };
                let next_val = match op {
                    AtomicRmwBinOp::AtomicXchg => src,
                    AtomicRmwBinOp::AtomicAdd => builder.add(load, src),
                    AtomicRmwBinOp::AtomicSub => builder.sub(load, src),
                    AtomicRmwBinOp::AtomicAnd => builder.and(load, src),
                    AtomicRmwBinOp::AtomicNand => {
                        let and = builder.and(load, src);
                        builder.not(and)
                    }
                    AtomicRmwBinOp::AtomicOr => builder.or(load, src),
                    AtomicRmwBinOp::AtomicXor => builder.xor(load, src),
                    AtomicRmwBinOp::AtomicMax => {
                        let is_src_bigger = builder.icmp(IntPredicate::IntSGT, src, load);
                        builder.select(is_src_bigger, src, load)
                    }
                    AtomicRmwBinOp::AtomicMin => {
                        let is_src_smaller = builder.icmp(IntPredicate::IntSLT, src, load);
                        builder.select(is_src_smaller, src, load)
                    }
                    AtomicRmwBinOp::AtomicUMax => {
                        let is_src_bigger = builder.icmp(IntPredicate::IntUGT, src, load);
                        builder.select(is_src_bigger, src, load)
                    }
                    AtomicRmwBinOp::AtomicUMin => {
                        let is_src_smaller = builder.icmp(IntPredicate::IntULT, src, load);
                        builder.select(is_src_smaller, src, load)
                    }
                };
                unsafe { llvm::LLVMBuildStore(builder.llbuilder, next_val, dst) };
                load
            },
        );

        // njn:
        // - copied from rustc_codegen_llvm
        // - but Fractal said: Here, if ret_ptr is true, we should cast dst to *usize, src to
        //   usize, and then cast the return value back to a *T(by checking the original type of
        //   src).
        if ret_ptr && self.val_ty(res) != self.type_ptr() {
            res = self.inttoptr(res, self.type_ptr());
        }
        res
    }

    fn atomic_fence(
        &mut self,
        _order: AtomicOrdering,
        _scope: rustc_codegen_ssa::common::SynchronizationScope,
    ) {
        self.fatal("atomic fence is not supported, use cuda_std intrinsics instead")
    }

    fn set_invariant_load(&mut self, load: &'ll Value) {
        unsafe {
            llvm::LLVMSetMetadata(
                load,
                llvm::MetadataType::MD_invariant_load as c_uint,
                llvm::LLVMMDNodeInContext(self.cx.llcx, ptr::null(), 0),
            );
        }
    }

    fn lifetime_start(&mut self, ptr: &'ll Value, size: Size) {
        self.call_lifetime_intrinsic("llvm.lifetime.start.p0i8", ptr, size);
    }

    fn lifetime_end(&mut self, ptr: &'ll Value, size: Size) {
        self.call_lifetime_intrinsic("llvm.lifetime.end.p0i8", ptr, size);
    }

    fn call(
        &mut self,
        llty: &'ll Type,
        _fn_attrs: Option<&CodegenFnAttrs>,
        fn_abi: Option<&FnAbi<'tcx, Ty<'tcx>>>,
        llfn: &'ll Value,
        args: &[&'ll Value],
        _funclet: Option<&Self::Funclet>,
        _instance: Option<Instance<'tcx>>,
    ) -> &'ll Value {
        trace!("Calling fn {:?} with args {:?}", llfn, args);
        self.cx.last_call_llfn.set(None);
        let args = self.check_call("call", llty, llfn, args);

        let mut call = unsafe {
            llvm::LLVMRustBuildCall(
                self.llbuilder,
                llfn,
                args.as_ptr(),
                args.len() as c_uint,
                None,
            )
        };
        if let Some(fn_abi) = fn_abi {
            fn_abi.apply_attrs_callsite(self, call);
        }

        // bitcast return type if the type was remapped
        let map = self.cx.remapped_integer_args.borrow();
        let mut fn_ty = self.val_ty(llfn);
        while self.cx.type_kind(fn_ty) == TypeKind::Pointer {
            fn_ty = self.cx.element_type(fn_ty);
        }
        if let Some((Some(ret_ty), _)) = map.get(&fn_ty) {
            self.cx.last_call_llfn.set(Some(call));
            call = transmute_llval(self.llbuilder, self.cx, call, ret_ty);
        }

        call
    }

    fn tail_call(
        &mut self,
        _llty: Self::Type,
        _fn_attrs: Option<&CodegenFnAttrs>,
        _fn_abi: &FnAbi<'tcx, Ty<'tcx>>,
        _llfn: Self::Value,
        _args: &[Self::Value],
        _funclet: Option<&Self::Funclet>,
        _instance: Option<ty::Instance<'tcx>>,
    ) {
        todo!()
    }

    fn zext(&mut self, val: &'ll Value, dest_ty: &'ll Type) -> &'ll Value {
        trace!("Zext {:?} to {:?}", val, dest_ty);
        unsafe { llvm::LLVMBuildZExt(self.llbuilder, val, dest_ty, unnamed()) }
    }

    fn apply_attrs_to_cleanup_callsite(&mut self, llret: &'ll Value) {
        // Cleanup is always the cold path.
        llvm::Attribute::Cold.apply_callsite(llvm::AttributePlace::Function, llret);

        // In LLVM versions with deferred inlining (currently, system LLVM < 14),
        // inlining drop glue can lead to exponential size blowup.
        // See rust_lang/rust #41696 and #92110.
        llvm::Attribute::NoInline.apply_callsite(llvm::AttributePlace::Function, llret);
    }

    fn assume_nonnull(&mut self, val: Self::Value) {
        assert_eq!(self.cx.type_kind(self.cx.val_ty(val)), TypeKind::Pointer);
        let val_ty = self.cx.val_ty(val);
        let null = self.cx.const_null(val_ty);
        let is_null = self.icmp(IntPredicate::IntNE, val, null);
        self.assume(is_null);
    }
}

impl<'ll> StaticBuilderMethods for Builder<'_, 'll, '_> {
    fn get_static(&mut self, def_id: DefId) -> &'ll Value {
        self.cx.get_static(def_id)
    }
}

impl<'a, 'll, 'tcx> Builder<'a, 'll, 'tcx> {
    // Helper function to check if a value is 128-bit integer
    fn is_i128(&self, val: &'ll Value) -> bool {
        let ty = self.val_ty(val);
        if unsafe { llvm::LLVMRustGetTypeKind(ty) == llvm::TypeKind::Integer } {
            unsafe { llvm::LLVMGetIntTypeWidth(ty) == 128 }
        } else {
            false
        }
    }

    // Helper to split i128 into low and high u64 parts
    fn split_i128(&mut self, val: &'ll Value) -> (&'ll Value, &'ll Value) {
        let i64_ty = self.type_i64();
        let const_64 = self.const_u128(64);

        let lo = self.trunc(val, i64_ty);
        let shifted = unsafe { llvm::LLVMBuildLShr(self.llbuilder, val, const_64, UNNAMED) };
        let hi = self.trunc(shifted, i64_ty);

        (lo, hi)
    }

    // Helper to combine two u64 values into i128
    fn combine_i128(&mut self, lo: &'ll Value, hi: &'ll Value) -> &'ll Value {
        let i128_ty = self.type_i128();
        let const_64 = self.const_u128(64);

        let lo_ext = self.zext(lo, i128_ty);
        let hi_ext = self.zext(hi, i128_ty);
        let hi_shifted = unsafe { llvm::LLVMBuildShl(self.llbuilder, hi_ext, const_64, UNNAMED) };
        unsafe { llvm::LLVMBuildOr(self.llbuilder, lo_ext, hi_shifted, UNNAMED) }
    }

    // Emulate 128-bit addition using two 64-bit additions with carry
    fn emulate_i128_add(&mut self, lhs: &'ll Value, rhs: &'ll Value) -> &'ll Value {
        let i64_ty = self.type_i64();
        let (lhs_lo, lhs_hi) = self.split_i128(lhs);
        let (rhs_lo, rhs_hi) = self.split_i128(rhs);

        // Add low parts
        let sum_lo = unsafe { llvm::LLVMBuildAdd(self.llbuilder, lhs_lo, rhs_lo, UNNAMED) };

        // Check for carry from low addition
        let carry = self.icmp(IntPredicate::IntULT, sum_lo, lhs_lo);
        let carry_ext = self.zext(carry, i64_ty);

        // Add high parts with carry
        let sum_hi_temp = unsafe { llvm::LLVMBuildAdd(self.llbuilder, lhs_hi, rhs_hi, UNNAMED) };
        let sum_hi = unsafe { llvm::LLVMBuildAdd(self.llbuilder, sum_hi_temp, carry_ext, UNNAMED) };

        self.combine_i128(sum_lo, sum_hi)
    }

    // Emulate 128-bit subtraction
    fn emulate_i128_sub(&mut self, lhs: &'ll Value, rhs: &'ll Value) -> &'ll Value {
        let i64_ty = self.type_i64();
        let (lhs_lo, lhs_hi) = self.split_i128(lhs);
        let (rhs_lo, rhs_hi) = self.split_i128(rhs);

        // Subtract low parts
        let diff_lo = unsafe { llvm::LLVMBuildSub(self.llbuilder, lhs_lo, rhs_lo, UNNAMED) };

        // Check for borrow
        let borrow = self.icmp(IntPredicate::IntUGT, rhs_lo, lhs_lo);
        let borrow_ext = self.zext(borrow, i64_ty);

        // Subtract high parts with borrow
        let diff_hi_temp = unsafe { llvm::LLVMBuildSub(self.llbuilder, lhs_hi, rhs_hi, UNNAMED) };
        let diff_hi =
            unsafe { llvm::LLVMBuildSub(self.llbuilder, diff_hi_temp, borrow_ext, UNNAMED) };

        self.combine_i128(diff_lo, diff_hi)
    }

    // Emulate 128-bit bitwise AND
    fn emulate_i128_and(&mut self, lhs: &'ll Value, rhs: &'ll Value) -> &'ll Value {
        let (lhs_lo, lhs_hi) = self.split_i128(lhs);
        let (rhs_lo, rhs_hi) = self.split_i128(rhs);

        let and_lo = unsafe { llvm::LLVMBuildAnd(self.llbuilder, lhs_lo, rhs_lo, UNNAMED) };
        let and_hi = unsafe { llvm::LLVMBuildAnd(self.llbuilder, lhs_hi, rhs_hi, UNNAMED) };

        self.combine_i128(and_lo, and_hi)
    }

    // Emulate 128-bit bitwise OR
    fn emulate_i128_or(&mut self, lhs: &'ll Value, rhs: &'ll Value) -> &'ll Value {
        let (lhs_lo, lhs_hi) = self.split_i128(lhs);
        let (rhs_lo, rhs_hi) = self.split_i128(rhs);

        let or_lo = unsafe { llvm::LLVMBuildOr(self.llbuilder, lhs_lo, rhs_lo, UNNAMED) };
        let or_hi = unsafe { llvm::LLVMBuildOr(self.llbuilder, lhs_hi, rhs_hi, UNNAMED) };

        self.combine_i128(or_lo, or_hi)
    }

    // Emulate 128-bit bitwise XOR
    fn emulate_i128_xor(&mut self, lhs: &'ll Value, rhs: &'ll Value) -> &'ll Value {
        let (lhs_lo, lhs_hi) = self.split_i128(lhs);
        let (rhs_lo, rhs_hi) = self.split_i128(rhs);

        let xor_lo = unsafe { llvm::LLVMBuildXor(self.llbuilder, lhs_lo, rhs_lo, UNNAMED) };
        let xor_hi = unsafe { llvm::LLVMBuildXor(self.llbuilder, lhs_hi, rhs_hi, UNNAMED) };

        self.combine_i128(xor_lo, xor_hi)
    }

    // Emulate 128-bit bitwise NOT
    fn emulate_i128_not(&mut self, val: &'ll Value) -> &'ll Value {
        let (lo, hi) = self.split_i128(val);

        let not_lo = unsafe { llvm::LLVMBuildNot(self.llbuilder, lo, UNNAMED) };
        let not_hi = unsafe { llvm::LLVMBuildNot(self.llbuilder, hi, UNNAMED) };

        self.combine_i128(not_lo, not_hi)
    }

    // Emulate 128-bit negation (two's complement)
    fn emulate_i128_neg(&mut self, val: &'ll Value) -> &'ll Value {
        // Two's complement: ~val + 1
        let not_val = self.emulate_i128_not(val);
        let one = self.const_u128(1);
        self.emulate_i128_add(not_val, one)
    }

    pub(crate) fn emulate_i128_bswap(&mut self, val: &'ll Value) -> &'ll Value {
        // Split the 128-bit value into two 64-bit halves
        let (lo, hi) = self.split_i128(val);

        // Byte-swap each 64-bit half using the LLVM intrinsic (which exists in LLVM 7.1)
        let swapped_lo = self.call_intrinsic("llvm.bswap.i64", &[lo]);
        let swapped_hi = self.call_intrinsic("llvm.bswap.i64", &[hi]);

        // Swap the halves: the high part becomes low and vice versa
        self.combine_i128(swapped_hi, swapped_lo)
    }

    pub(crate) fn emulate_i128_count_zeros(
        &mut self,
        val: &'ll Value,
        kind: CountZerosKind,
        is_nonzero: bool,
    ) -> &'ll Value {
        // Split the 128-bit value into two 64-bit halves
        let (lo, hi) = self.split_i128(val);

        match kind {
            CountZerosKind::Leading => {
                // Count leading zeros: check high part first
                let hi_is_zero = self.icmp(IntPredicate::IntEQ, hi, self.const_u64(0));
                let hi_ctlz =
                    self.call_intrinsic("llvm.ctlz.i64", &[hi, self.const_bool(is_nonzero)]);
                let lo_ctlz =
                    self.call_intrinsic("llvm.ctlz.i64", &[lo, self.const_bool(is_nonzero)]);

                // If high part is zero, result is 64 + ctlz(lo), otherwise ctlz(hi)
                let lo_ctlz_plus_64 = self.add(lo_ctlz, self.const_u64(64));
                let result_64 = self.select(hi_is_zero, lo_ctlz_plus_64, hi_ctlz);

                // Zero-extend to i128
                self.zext(result_64, self.type_i128())
            }
            CountZerosKind::Trailing => {
                // Count trailing zeros: check low part first
                let lo_is_zero = self.icmp(IntPredicate::IntEQ, lo, self.const_u64(0));
                let lo_cttz =
                    self.call_intrinsic("llvm.cttz.i64", &[lo, self.const_bool(is_nonzero)]);
                let hi_cttz =
                    self.call_intrinsic("llvm.cttz.i64", &[hi, self.const_bool(is_nonzero)]);

                // If low part is zero, result is 64 + cttz(hi), otherwise cttz(lo)
                let hi_cttz_plus_64 = self.add(hi_cttz, self.const_u64(64));
                let result_64 = self.select(lo_is_zero, hi_cttz_plus_64, lo_cttz);

                // Zero-extend to i128
                self.zext(result_64, self.type_i128())
            }
        }
    }

    pub(crate) fn emulate_i128_ctpop(&mut self, val: &'ll Value) -> &'ll Value {
        // Split the 128-bit value into two 64-bit halves
        let (lo, hi) = self.split_i128(val);

        // Count population (number of 1 bits) in each half
        let lo_popcount = self.call_intrinsic("llvm.ctpop.i64", &[lo]);
        let hi_popcount = self.call_intrinsic("llvm.ctpop.i64", &[hi]);

        // Add the two counts
        let total_64 = self.add(lo_popcount, hi_popcount);

        // Zero-extend to i128
        self.zext(total_64, self.type_i128())
    }

    pub(crate) fn emulate_i128_rotate(
        &mut self,
        val: &'ll Value,
        shift: &'ll Value,
        is_left: bool,
    ) -> &'ll Value {
        // Rotate is implemented as: (val << shift) | (val >> (128 - shift))
        // For rotate right: (val >> shift) | (val << (128 - shift))

        // Ensure shift is i128
        let shift_128 = if self.val_ty(shift) == self.type_i128() {
            shift
        } else {
            self.zext(shift, self.type_i128())
        };

        // Calculate 128 - shift for the complementary shift
        let bits_128 = self.const_u128(128);
        let shift_complement = self.sub(bits_128, shift_128);

        // Perform the two shifts
        let (first_shift, second_shift) = if is_left {
            (self.shl(val, shift_128), self.lshr(val, shift_complement))
        } else {
            (self.lshr(val, shift_128), self.shl(val, shift_complement))
        };

        // Combine with OR
        self.or(first_shift, second_shift)
    }

    pub(crate) fn emulate_i128_bitreverse(&mut self, val: &'ll Value) -> &'ll Value {
        // Split the 128-bit value into two 64-bit halves
        let (lo, hi) = self.split_i128(val);

        // Reverse bits in each half using the 64-bit intrinsic
        let reversed_lo = self.call_intrinsic("llvm.bitreverse.i64", &[lo]);
        let reversed_hi = self.call_intrinsic("llvm.bitreverse.i64", &[hi]);

        // Swap the halves: reversed high becomes low and vice versa
        self.combine_i128(reversed_hi, reversed_lo)
    }

    fn with_cx(cx: &'a CodegenCx<'ll, 'tcx>) -> Self {
        // Create a fresh builder from the crate context.
        let llbuilder = unsafe { llvm::LLVMCreateBuilderInContext(cx.llcx) };
        Builder { llbuilder, cx }
    }

    pub fn llfn(&self) -> &'ll Value {
        unsafe { llvm::LLVMGetBasicBlockParent(self.llbb()) }
    }

    fn position_at_start(&mut self, llbb: &'ll BasicBlock) {
        unsafe {
            llvm::LLVMRustPositionBuilderAtStart(self.llbuilder, llbb);
        }
    }

    fn align_metadata(&mut self, _load: &'ll Value, _align: Align) {}

    fn noundef_metadata(&mut self, _load: &'ll Value) {}

    fn check_store(&mut self, val: &'ll Value, ptr: &'ll Value) -> &'ll Value {
        let dest_ptr_ty = self.cx.val_ty(ptr);
        let stored_ty = self.cx.val_ty(val);
        let stored_ptr_ty = self.cx.type_ptr_to(stored_ty);

        assert_eq!(self.cx.type_kind(dest_ptr_ty), TypeKind::Pointer);

        if dest_ptr_ty == stored_ptr_ty {
            ptr
        } else {
            self.bitcast(ptr, stored_ptr_ty)
        }
    }

    fn check_call<'b>(
        &mut self,
        typ: &str,
        fn_ty: &'ll Type,
        llfn: &'ll Value,
        args: &'b [&'ll Value],
    ) -> Cow<'b, [&'ll Value]> {
        assert!(
            self.cx.type_kind(fn_ty) == TypeKind::Function,
            "builder::{typ} not passed a function, but {fn_ty:?}"
        );

        let param_tys = self.cx.func_params_types(fn_ty);

        let all_args_match = param_tys
            .iter()
            .zip(args.iter().map(|&v| self.val_ty(v)))
            .all(|(expected_ty, actual_ty)| *expected_ty == actual_ty);

        if all_args_match {
            return Cow::Borrowed(args);
        }

        let casted_args: Vec<_> = param_tys
            .into_iter()
            .zip(args.iter())
            .enumerate()
            .map(|(i, (expected_ty, &actual_val))| {
                let actual_ty = self.val_ty(actual_val);
                if expected_ty != actual_ty {
                    debug!(
                        "type mismatch in function call of {:?}. \
                            Expected {:?} for param {}, got {:?}; injecting bitcast",
                        llfn, expected_ty, i, actual_ty
                    );
                    self.bitcast(actual_val, expected_ty)
                } else {
                    actual_val
                }
            })
            .collect();

        Cow::Owned(casted_args)
    }

    pub fn va_arg(&mut self, list: &'ll Value, ty: &'ll Type) -> &'ll Value {
        unsafe { llvm::LLVMBuildVAArg(self.llbuilder, list, ty, unnamed()) }
    }

    pub(crate) fn call_intrinsic(&mut self, intrinsic: &str, args: &[&'ll Value]) -> &'ll Value {
        let (ty, f) = self.cx.get_intrinsic(intrinsic);
        self.call(ty, None, None, f, args, None, None)
    }

    fn call_lifetime_intrinsic(&mut self, intrinsic: &'static str, ptr: &'ll Value, size: Size) {
        let size = size.bytes();
        if size == 0 {
            return;
        }

        if !self.cx().sess().emit_lifetime_markers() {
            return;
        }

        self.call_intrinsic(intrinsic, &[self.cx.const_u64(size), ptr]);
    }

    pub(crate) fn phi(
        &mut self,
        ty: &'ll Type,
        vals: &[&'ll Value],
        bbs: &[&'ll BasicBlock],
    ) -> &'ll Value {
        assert_eq!(vals.len(), bbs.len());
        let phi = unsafe { llvm::LLVMBuildPhi(self.llbuilder, ty, unnamed()) };
        unsafe {
            llvm::LLVMAddIncoming(phi, vals.as_ptr(), bbs.as_ptr(), vals.len() as c_uint);
            phi
        }
    }

    fn add_incoming_to_phi(&mut self, phi: &'ll Value, val: &'ll Value, bb: &'ll BasicBlock) {
        unsafe {
            llvm::LLVMAddIncoming(phi, &val, &bb, 1);
        }
    }
}
impl<'ll, 'tcx, 'a> Builder<'a, 'll, 'tcx> {
    /// Implements a standard atomic, using LLVM intrinsics(in `atomic_supported`, if `dst` is in a supported address space)
    /// or emulation(with `emulate_local`, if `dst` points to a thread-local address space).
    /// FIXME(FractalFir): this code assumess all pointers are generic. Adjust it once we support address spaces.
    fn atomic_op(
        &mut self,
        dst: &'ll Value,
        atomic_supported: impl FnOnce(&mut Builder<'a, 'll, 'tcx>, &'ll Value) -> &'ll Value,
        emulate_local: impl FnOnce(&mut Builder<'a, 'll, 'tcx>, &'ll Value) -> &'ll Value,
    ) -> &'ll Value {
        // (FractalFir) Atomics in CUDA have some limitations, and we have to work around them.
        // For example, they are restricted in what address space they operate on.
        // CUDA has 4 address spaces(and a generic one, which is an union of all of those).
        // An atomic instruction can soundly operate on:
        // 1. The global address space
        // 2. The shared(cluster) address space.
        // It can't operate on:
        // 1. The const address space(atomics on consts are UB anyway)
        // 2. The thread address space(which should be only accessible to 1 thread, anyway?)
        // So, we do the following:
        // 1. Check if the pointer is in one of the address spaces atomics support.
        //  a) if so, we perform an atomic operation
        // 2. Check if the pointer is in the thread-local address space. If it is, we use non-atomic ops here,
        // **ASSUMING** only the current thread can access thread-local memory. (FIXME: is this sound?)
        // 3. If the pointer is not in a supported address space, and is not thread-local, then we bail, and trap.

        // We check if the `dst` pointer is in the `global` address space.
        let (isspacep_global_ty, isspacep_global_fn) =
            self.get_intrinsic("llvm.nvvm.isspacep.global");
        let isspacep_global = self.call(
            isspacep_global_ty,
            None,
            None,
            isspacep_global_fn,
            &[dst],
            None,
            None,
        );
        // We check if the `dst` pointer is in the `shared` address space.
        let (isspacep_shared_ty, isspacep_shared_fn) =
            self.get_intrinsic("llvm.nvvm.isspacep.shared");
        let isspacep_shared = self.call(
            isspacep_shared_ty,
            None,
            None,
            isspacep_shared_fn,
            &[dst],
            None,
            None,
        );
        // Combine those to check if we are in a supported address space.
        let atomic_supported_addrspace = self.or(isspacep_shared, isspacep_global);
        // We create 2 blocks here: one we branch to if atomic is in the right address space, and one we branch to otherwise.
        let supported_bb = self.append_sibling_block("atomic_space_supported");
        let unsupported_bb = self.append_sibling_block("atomic_space_unsupported");
        self.cond_br(atomic_supported_addrspace, supported_bb, unsupported_bb);
        //  We also create a "merge" block we will jump to, after the the atomic ops finish.
        let merge_bb = self.append_sibling_block("atomic_op_done");
        // Execute atomic op if supported, then jump to merge
        self.switch_to_block(supported_bb);
        let supported_res = atomic_supported(self, dst);
        self.br(merge_bb);
        // Check if the pointer is in the thread space. If so, we can emulate it.
        self.switch_to_block(unsupported_bb);
        let (isspacep_local_ty, isspacep_local_fn) = self.get_intrinsic("llvm.nvvm.isspacep.local");
        let isspacep_local = self.call(
            isspacep_local_ty,
            None,
            None,
            isspacep_local_fn,
            &[dst],
            None,
            None,
        );
        let local_bb = self.append_sibling_block("atomic_local_space");
        let atomic_ub_bb = self.append_sibling_block("atomic_space_ub");
        self.cond_br(isspacep_local, local_bb, atomic_ub_bb);
        // The pointer is in the thread(local) space.
        self.switch_to_block(local_bb);
        let local_res = emulate_local(self, dst);
        self.br(merge_bb);
        // The pointer is neither in the supported address space, nor the local space.
        // This is very likely UB. So, we trap here.
        // TODO: should we print some kind of a message here? NVVM supports printf.
        self.switch_to_block(atomic_ub_bb);
        self.abort();
        self.unreachable();
        // Atomic is impl has finished, and we can now switch to the merge_bb
        self.switch_to_block(merge_bb);
        self.phi(
            self.val_ty(local_res),
            &[supported_res, local_res],
            &[supported_bb, local_bb],
        )
    }
}
