use crate::abi::{FnAbiLlvmExt, LlvmType};
use crate::context::CodegenCx;
use crate::llvm::{self, Bool, False, True, Type, Value};
use libc::c_uint;
use rustc_abi::Primitive::{Float, Int, Pointer};
use rustc_abi::{
    AddressSpace, Align, BackendRepr, FieldsShape, Integer, PointeeInfo, Reg, Scalar, Size,
    TyAbiInterface, Variants,
};
use rustc_codegen_ssa::common::TypeKind;
use rustc_codegen_ssa::traits::*;
use rustc_data_structures::small_c_str::SmallCStr;
use rustc_middle::bug;
use rustc_middle::ty::layout::{FnAbiOf, LayoutOf, TyAndLayout};
use rustc_middle::ty::print::with_no_trimmed_paths;
use rustc_middle::ty::{self, CoroutineArgsExt, Ty, TypeVisitableExt};
use rustc_target::callconv::{CastTarget, FnAbi};
use std::ffi::CString;
use std::fmt::{Debug, Write};
use std::hash::Hash;
use std::ptr;
use tracing::trace;

impl PartialEq for Type {
    fn eq(&self, other: &Self) -> bool {
        ptr::eq(self, other)
    }
}

impl Eq for Type {}

impl Debug for Type {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        unsafe {
            let ptr = llvm::LLVMPrintTypeToString(self);
            let cstring = CString::from_raw(ptr);
            let string = cstring.to_string_lossy();
            f.write_str(&string)
        }
    }
}

impl Hash for Type {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        (self as *const Type).hash(state);
    }
}

impl Type {
    // Creates an integer type with the given number of bits, e.g., i24
    pub fn ix_llcx(llcx: &llvm::Context, num_bits: u64) -> &Type {
        unsafe { llvm::LLVMIntTypeInContext(llcx, num_bits as c_uint) }
    }
}

impl<'ll> CodegenCx<'ll, '_> {
    pub(crate) fn voidp(&self) -> &'ll Type {
        // llvm uses i8* for void ptrs, void* is invalid
        let i8_ty = self.type_i8();
        self.type_ptr_to_ext(i8_ty, AddressSpace::ZERO)
    }

    pub(crate) fn type_named_struct(&self, name: &str) -> &'ll Type {
        let name = SmallCStr::new(name);
        unsafe { llvm::LLVMStructCreateNamed(self.llcx, name.as_ptr()) }
    }

    pub(crate) fn type_struct(&self, els: &[&'ll Type], packed: bool) -> &'ll Type {
        unsafe {
            llvm::LLVMStructTypeInContext(
                self.llcx,
                els.as_ptr(),
                els.len() as c_uint,
                packed as Bool,
            )
        }
    }

    pub(crate) fn set_struct_body(&self, ty: &'ll Type, els: &[&'ll Type], packed: bool) {
        unsafe { llvm::LLVMStructSetBody(ty, els.as_ptr(), els.len() as c_uint, packed as Bool) }
    }

    pub(crate) fn type_void(&self) -> &'ll Type {
        unsafe { llvm::LLVMVoidTypeInContext(self.llcx) }
    }

    pub(crate) fn type_metadata(&self) -> &'ll Type {
        unsafe { llvm::LLVMRustMetadataTypeInContext(self.llcx) }
    }

    pub(crate) fn type_i1(&self) -> &'ll Type {
        unsafe { llvm::LLVMInt1TypeInContext(self.llcx) }
    }

    pub(crate) fn type_i8p(&self) -> &'ll Type {
        self.type_i8p_ext(AddressSpace::ZERO)
    }

    pub(crate) fn type_i8p_ext(&self, address_space: AddressSpace) -> &'ll Type {
        self.type_ptr_to_ext(self.type_i8(), address_space)
    }

    ///x Creates an integer type with the given number of bits, e.g., i24
    pub(crate) fn type_ix(&self, num_bits: u64) -> &'ll Type {
        unsafe { llvm::LLVMIntTypeInContext(self.llcx, num_bits as c_uint) }
    }

    pub(crate) fn type_vector(&self, ty: &'ll Type, len: u64) -> &'ll Type {
        unsafe { llvm::LLVMVectorType(ty, len as c_uint) }
    }

    pub(crate) fn type_ptr_to(&self, ty: &'ll Type) -> &'ll Type {
        assert_ne!(
            self.type_kind(ty),
            TypeKind::Function,
            "don't call ptr_to on function types, use ptr_to_llvm_type on FnAbi instead or explicitly specify an address space if it makes sense"
        );

        unsafe { llvm::LLVMPointerType(ty, AddressSpace::ZERO.0) }
    }

    pub(crate) fn type_ptr_to_ext(&self, ty: &'ll Type, address_space: AddressSpace) -> &'ll Type {
        unsafe { llvm::LLVMPointerType(ty, address_space.0) }
    }

    pub(crate) fn func_params_types(&self, ty: &'ll Type) -> Vec<&'ll Type> {
        unsafe {
            let n_args = llvm::LLVMCountParamTypes(ty) as usize;
            let mut args = Vec::with_capacity(n_args);
            llvm::LLVMGetParamTypes(ty, args.as_mut_ptr());
            args.set_len(n_args);
            args
        }
    }

    pub(crate) fn type_pointee_for_align(&self, align: Align) -> &'ll Type {
        let ity = Integer::approximate_align(self, align);
        self.type_from_integer(ity)
    }

    /// Return a LLVM type that has at most the required alignment,
    /// and exactly the required size, as a best-effort padding array.
    pub(crate) fn type_padding_filler(&self, size: Size, align: Align) -> &'ll Type {
        let unit = Integer::approximate_align(self, align);
        let size = size.bytes();
        let unit_size = unit.size().bytes();
        assert_eq!(size % unit_size, 0);
        self.type_array(self.type_from_integer(unit), size / unit_size)
    }

    pub(crate) fn type_variadic_func(&self, args: &[&'ll Type], ret: &'ll Type) -> &'ll Type {
        unsafe { llvm::LLVMFunctionType(ret, args.as_ptr(), args.len() as c_uint, True) }
    }

    pub(crate) fn type_array(&self, ty: &'ll Type, len: u64) -> &'ll Type {
        unsafe { llvm::LLVMRustArrayType(ty, len) }
    }
}

impl<'ll, 'tcx> BaseTypeCodegenMethods for CodegenCx<'ll, 'tcx> {
    fn type_i8(&self) -> &'ll Type {
        unsafe { llvm::LLVMInt8TypeInContext(self.llcx) }
    }

    fn type_i16(&self) -> &'ll Type {
        unsafe { llvm::LLVMInt16TypeInContext(self.llcx) }
    }

    fn type_i32(&self) -> &'ll Type {
        unsafe { llvm::LLVMInt32TypeInContext(self.llcx) }
    }

    fn type_i64(&self) -> &'ll Type {
        unsafe { llvm::LLVMInt64TypeInContext(self.llcx) }
    }

    fn type_i128(&self) -> &'ll Type {
        unsafe { llvm::LLVMIntTypeInContext(self.llcx, 128) }
    }

    fn type_isize(&self) -> &'ll Type {
        self.isize_ty
    }

    fn type_f16(&self) -> &'ll Type {
        unsafe { llvm::LLVMHalfTypeInContext(self.llcx) }
    }

    fn type_f32(&self) -> &'ll Type {
        unsafe { llvm::LLVMFloatTypeInContext(self.llcx) }
    }

    fn type_f64(&self) -> &'ll Type {
        unsafe { llvm::LLVMDoubleTypeInContext(self.llcx) }
    }

    fn type_f128(&self) -> &'ll Type {
        unsafe { llvm::LLVMFP128TypeInContext(self.llcx) }
    }

    fn type_array(&self, ty: Self::Type, len: u64) -> Self::Type {
        unsafe { llvm::LLVMRustArrayType(ty, len) }
    }

    fn type_func(&self, args: &[&'ll Type], ret: &'ll Type) -> &'ll Type {
        unsafe { llvm::LLVMFunctionType(ret, args.as_ptr(), args.len() as c_uint, False) }
    }

    fn type_kind(&self, ty: &'ll Type) -> TypeKind {
        unsafe { llvm::LLVMRustGetTypeKind(ty).to_generic() }
    }

    fn type_ptr(&self) -> Self::Type {
        self.type_ptr_ext(AddressSpace::ZERO)
    }

    fn type_ptr_ext(&self, address_space: AddressSpace) -> Self::Type {
        self.type_ptr_to_ext(self.type_i8(), address_space)
    }

    fn element_type(&self, ty: &'ll Type) -> &'ll Type {
        unsafe { llvm::LLVMGetElementType(ty) }
    }

    fn vector_length(&self, ty: &'ll Type) -> usize {
        unsafe { llvm::LLVMGetVectorSize(ty) as usize }
    }

    fn float_width(&self, ty: &'ll Type) -> usize {
        match self.type_kind(ty) {
            TypeKind::Float => 32,
            TypeKind::Double => 64,
            TypeKind::X86_FP80 => 80,
            TypeKind::FP128 | TypeKind::PPC_FP128 => 128,
            _ => bug!("llvm_float_width called on a non-float type"),
        }
    }

    fn int_width(&self, ty: &'ll Type) -> u64 {
        match self.type_kind(ty) {
            TypeKind::Integer => unsafe { llvm::LLVMGetIntTypeWidth(ty) as u64 },
            _ => bug!("llvm_int_width called on a non-integer type"),
        }
    }

    fn val_ty(&self, v: &'ll Value) -> &'ll Type {
        unsafe { llvm::LLVMTypeOf(v) }
    }
}

impl<'ll, 'tcx> LayoutTypeCodegenMethods<'tcx> for CodegenCx<'ll, 'tcx> {
    fn backend_type(&self, layout: TyAndLayout<'tcx>) -> &'ll Type {
        layout.llvm_type(self)
    }
    fn immediate_backend_type(&self, layout: TyAndLayout<'tcx>) -> &'ll Type {
        layout.immediate_llvm_type(self)
    }
    fn is_backend_immediate(&self, layout: TyAndLayout<'tcx>) -> bool {
        layout.is_llvm_immediate()
    }
    fn is_backend_scalar_pair(&self, layout: TyAndLayout<'tcx>) -> bool {
        layout.is_llvm_scalar_pair()
    }
    fn scalar_pair_element_backend_type(
        &self,
        layout: TyAndLayout<'tcx>,
        index: usize,
        immediate: bool,
    ) -> &'ll Type {
        layout.scalar_pair_element_llvm_type(self, index, immediate)
    }
    fn cast_backend_type(&self, ty: &CastTarget) -> &'ll Type {
        ty.llvm_type(self)
    }
    fn fn_ptr_backend_type(&self, fn_abi: &FnAbi<'tcx, Ty<'tcx>>) -> &'ll Type {
        fn_abi.ptr_to_llvm_type(self)
    }
    fn reg_backend_type(&self, ty: &Reg) -> &'ll Type {
        ty.llvm_type(self)
    }
    fn fn_decl_backend_type(&self, fn_abi: &FnAbi<'tcx, Ty<'tcx>>) -> &'ll Type {
        fn_abi.llvm_type(self)
    }
}

// exists in rustc_codegen_llvm but cant use it because its private :'(
pub(crate) trait LayoutLlvmExt<'tcx> {
    fn is_llvm_immediate(&self) -> bool;
    fn is_llvm_scalar_pair(&self) -> bool;
    fn llvm_type<'a>(&self, cx: &CodegenCx<'a, 'tcx>) -> &'a Type;
    fn immediate_llvm_type<'a>(&self, cx: &CodegenCx<'a, 'tcx>) -> &'a Type;
    fn scalar_llvm_type_at<'a>(&self, cx: &CodegenCx<'a, 'tcx>, scalar: Scalar) -> &'a Type;
    fn scalar_pair_element_llvm_type<'a>(
        &self,
        cx: &CodegenCx<'a, 'tcx>,
        index: usize,
        immediate: bool,
    ) -> &'a Type;
    fn pointee_info_at<'a>(&self, cx: &CodegenCx<'a, 'tcx>, offset: Size) -> Option<PointeeInfo>;
}

impl<'tcx> LayoutLlvmExt<'tcx> for TyAndLayout<'tcx> {
    fn is_llvm_immediate(&self) -> bool {
        match self.backend_repr {
            BackendRepr::Scalar(_) | BackendRepr::SimdVector { .. } => true,
            BackendRepr::ScalarPair(..) => false,
            BackendRepr::Memory { .. } => self.is_zst(),
        }
    }

    fn is_llvm_scalar_pair(&self) -> bool {
        match self.backend_repr {
            BackendRepr::ScalarPair(..) => true,
            BackendRepr::Scalar(_)
            | BackendRepr::SimdVector { .. }
            | BackendRepr::Memory { .. } => false,
        }
    }

    /// Gets the LLVM type corresponding to a Rust type, i.e., `rustc_middle::ty::Ty`.
    /// The pointee type of the pointer in `PlaceRef` is always this type.
    /// For sized types, it is also the right LLVM type for an `alloca`
    /// containing a value of that type, and most immediates (except `bool`).
    /// Unsized types, however, are represented by a "minimal unit", e.g.
    /// `[T]` becomes `T`, while `str` and `Trait` turn into `i8` - this
    /// is useful for indexing slices, as `&[T]`'s data pointer is `T*`.
    /// If the type is an unsized struct, the regular layout is generated,
    /// with the inner-most trailing unsized field using the "minimal unit"
    /// of that field's type - this is useful for taking the address of
    /// that field and ensuring the struct has the right alignment.
    fn llvm_type<'a>(&self, cx: &CodegenCx<'a, 'tcx>) -> &'a Type {
        // This must produce the same result for `repr(transparent)` wrappers as for the inner type!
        // In other words, this should generally not look at the type at all, but only at the
        // layout.
        if let BackendRepr::Scalar(scalar) = self.backend_repr {
            // Use a different cache for scalars because pointers to DSTs
            // can be either fat or thin (data pointers of fat pointers).
            if let Some(&llty) = cx.scalar_lltypes.borrow().get(&self.ty) {
                return llty;
            }

            let llty = match *self.ty.kind() {
                ty::Ref(_, ty, _) | ty::RawPtr(ty, _) => {
                    cx.type_ptr_to(cx.layout_of(ty).llvm_type(cx))
                }
                ty::Adt(def, _) if def.is_box() => {
                    cx.type_ptr_to(cx.layout_of(self.ty.expect_boxed_ty()).llvm_type(cx))
                }
                ty::FnPtr(sig, hdr) => {
                    cx.fn_ptr_backend_type(cx.fn_abi_of_fn_ptr(sig.with(hdr), ty::List::empty()))
                }
                _ => self.scalar_llvm_type_at(cx, scalar),
            };
            cx.scalar_lltypes.borrow_mut().insert(self.ty, llty);
            return llty;
        }

        // Check the cache.
        let variant_index = match self.variants {
            Variants::Single { index } => Some(index),
            _ => None,
        };
        if let Some(&llty) = cx.lltypes.borrow().get(&(self.ty, variant_index)) {
            return llty;
        }

        assert!(
            !self.ty.has_escaping_bound_vars(),
            "{:?} has escaping bound vars",
            self.ty
        );

        // Make sure lifetimes are erased, to avoid generating distinct LLVM
        // types for Rust types that only differ in the choice of lifetimes.
        let normal_ty = cx.tcx.erase_regions(self.ty);

        let mut defer = None;
        let llty = if self.ty != normal_ty {
            let mut layout = cx.layout_of(normal_ty);
            if let Some(v) = variant_index {
                layout = layout.for_variant(cx, v);
            }
            layout.llvm_type(cx)
        } else {
            uncached_llvm_type(cx, *self, &mut defer)
        };

        cx.lltypes
            .borrow_mut()
            .insert((self.ty, variant_index), llty);

        if let Some((llty, layout)) = defer {
            let (llfields, packed) = struct_llfields(cx, layout);
            cx.set_struct_body(llty, &llfields, packed)
        }

        llty
    }

    fn immediate_llvm_type<'a>(&self, cx: &CodegenCx<'a, 'tcx>) -> &'a Type {
        match self.backend_repr {
            BackendRepr::Scalar(ref scalar) => {
                if scalar.is_bool() {
                    return cx.type_i1();
                }
            }
            BackendRepr::ScalarPair(..) => {
                // An immediate pair always contains just the two elements, without any padding
                // filler, as it should never be stored to memory.
                return cx.type_struct(
                    &[
                        self.scalar_pair_element_llvm_type(cx, 0, true),
                        self.scalar_pair_element_llvm_type(cx, 1, true),
                    ],
                    false,
                );
            }
            _ => {}
        };
        self.llvm_type(cx)
    }

    fn scalar_llvm_type_at<'a>(&self, cx: &CodegenCx<'a, 'tcx>, scalar: Scalar) -> &'a Type {
        match scalar.primitive() {
            Int(i, _) => cx.type_from_integer(i),
            Float(f) => cx.type_from_float(f),
            Pointer(address_space) => {
                // If we know the alignment, pick something better than i8.
                let (pointee, address_space) = if let Some(PointeeInfo {
                    safe: Some(_),
                    align,
                    ..
                }) = self.pointee_info_at(cx, Size::ZERO)
                {
                    (cx.type_pointee_for_align(align), address_space)
                } else {
                    (cx.type_i8(), AddressSpace::ZERO)
                };
                cx.type_ptr_to_ext(pointee, address_space)
            }
        }
    }

    fn scalar_pair_element_llvm_type<'a>(
        &self,
        cx: &CodegenCx<'a, 'tcx>,
        index: usize,
        immediate: bool,
    ) -> &'a Type {
        // This must produce the same result for `repr(transparent)` wrappers as for the inner type!
        // In other words, this should generally not look at the type at all, but only at the
        // layout.

        // HACK(eddyb) special-case fat pointers until LLVM removes
        // pointee types, to avoid bitcasting every `OperandRef::deref`.
        match self.ty.kind() {
            ty::Ref(..) | ty::RawPtr(..) => {
                return self.field(cx, index).llvm_type(cx);
            }
            ty::Adt(def, _) if def.is_box() => {
                let ptr_ty = Ty::new_mut_ptr(cx.tcx, self.ty.boxed_ty().unwrap());
                return cx
                    .layout_of(ptr_ty)
                    .scalar_pair_element_llvm_type(cx, index, immediate);
            }
            _ => {}
        }

        let BackendRepr::ScalarPair(a, b) = self.backend_repr else {
            bug!(
                "TyAndLayout::scalar_pair_element_llty({:?}): not applicable",
                self
            );
        };
        let scalar = [a, b][index];

        // Make sure to return the same type `immediate_llvm_type` would when
        // dealing with an immediate pair.  This means that `(bool, bool)` is
        // effectively represented as `{i8, i8}` in memory and two `i1`s as an
        // immediate, just like `bool` is typically `i8` in memory and only `i1`
        // when immediate.  We need to load/store `bool` as `i8` to avoid
        // crippling LLVM optimizations or triggering other LLVM bugs with `i1`.
        if immediate && scalar.is_bool() {
            return cx.type_i1();
        }

        self.scalar_llvm_type_at(cx, scalar)
    }

    fn pointee_info_at<'a>(&self, cx: &CodegenCx<'a, 'tcx>, offset: Size) -> Option<PointeeInfo> {
        if let Some(&pointee) = cx.pointee_infos.borrow().get(&(self.ty, offset)) {
            return pointee;
        }

        let result = Ty::ty_and_layout_pointee_info_at(*self, cx, offset);

        cx.pointee_infos
            .borrow_mut()
            .insert((self.ty, offset), result);
        result
    }
}

fn uncached_llvm_type<'a, 'tcx>(
    cx: &CodegenCx<'a, 'tcx>,
    layout: TyAndLayout<'tcx>,
    defer: &mut Option<(&'a Type, TyAndLayout<'tcx>)>,
) -> &'a Type {
    trace!("Uncached LLVM type of {:?}", layout);
    match layout.backend_repr {
        BackendRepr::Scalar(_) => bug!("handled elsewhere"),
        BackendRepr::SimdVector { element, count } => {
            let element = layout.scalar_llvm_type_at(cx, element);
            return cx.type_vector(element, count);
        }
        BackendRepr::ScalarPair(..) => {
            return cx.type_struct(
                &[
                    layout.scalar_pair_element_llvm_type(cx, 0, false),
                    layout.scalar_pair_element_llvm_type(cx, 1, false),
                ],
                false,
            );
        }
        BackendRepr::Memory { .. } => {}
    }

    let name = match layout.ty.kind() {
        // FIXME(eddyb) producing readable type names for trait objects can result
        // in problematically distinct types due to HRTB and subtyping (see #47638).
        // ty::Dynamic(..) |
        ty::Adt(..) | ty::Closure(..) | ty::Foreign(..) | ty::Coroutine(..) | ty::Str
            // For performance reasons we use names only when emitting LLVM IR.
            if !cx.sess().fewer_names() =>
        {
            let mut name = with_no_trimmed_paths!(layout.ty.to_string());
            if let (&ty::Adt(def, _), &Variants::Single { index }) =
                (layout.ty.kind(), &layout.variants)
                && def.is_enum()
                && !def.variants().is_empty()
            {
                write!(&mut name, "::{}", def.variant(index).name).unwrap();
            }
            if let (&ty::Coroutine(_, _), &Variants::Single { index }) =
                (layout.ty.kind(), &layout.variants)
            {
                write!(&mut name, "::{}", ty::CoroutineArgs::variant_name(index)).unwrap();
            }
            Some(name)
        }
        _ => None,
    };

    match layout.fields {
        FieldsShape::Primitive | FieldsShape::Union(_) => {
            let fill = cx.type_padding_filler(layout.size, layout.align.abi);
            let packed = false;
            match name {
                None => cx.type_struct(&[fill], packed),
                Some(ref name) => {
                    let llty = cx.type_named_struct(name);
                    cx.set_struct_body(llty, &[fill], packed);
                    llty
                }
            }
        }
        FieldsShape::Array { count, .. } => cx.type_array(layout.field(cx, 0).llvm_type(cx), count),
        FieldsShape::Arbitrary { .. } => match name {
            None => {
                let (llfields, packed) = struct_llfields(cx, layout);
                cx.type_struct(&llfields, packed)
            }
            Some(ref name) => {
                let llty = cx.type_named_struct(name);
                *defer = Some((llty, layout));
                llty
            }
        },
    }
}

fn struct_llfields<'a, 'tcx>(
    cx: &CodegenCx<'a, 'tcx>,
    layout: TyAndLayout<'tcx>,
) -> (Vec<&'a Type>, bool) {
    let field_count = layout.fields.count();

    let mut packed = false;
    let mut offset = Size::ZERO;
    let mut prev_effective_align = layout.align.abi;
    let mut result: Vec<_> = Vec::with_capacity(1 + field_count * 2);
    for i in layout.fields.index_by_increasing_offset() {
        let target_offset = layout.fields.offset(i);
        let field = layout.field(cx, i);
        let effective_field_align = layout
            .align
            .abi
            .min(field.align.abi)
            .restrict_for_offset(target_offset);
        packed |= effective_field_align < field.align.abi;

        assert!(target_offset >= offset);
        let padding = target_offset - offset;
        if padding != Size::ZERO {
            let padding_align = prev_effective_align.min(effective_field_align);
            assert_eq!(offset.align_to(padding_align) + padding, target_offset);
            result.push(cx.type_padding_filler(padding, padding_align));
        }
        match field.ty.kind() {
            // This is a workaround for recursive types.
            rustc_middle::ty::TyKind::FnPtr(_, _) => result.push(cx.type_i8p()),
            _ => result.push(field.llvm_type(cx)),
        }

        offset = target_offset + field.size;
        prev_effective_align = effective_field_align;
    }
    if layout.is_sized() && field_count > 0 {
        if offset > layout.size {
            bug!(
                "layout: {:#?} stride: {:?} offset: {:?}",
                layout,
                layout.size,
                offset
            );
        }
        let padding = layout.size - offset;
        if padding != Size::ZERO {
            let padding_align = prev_effective_align;
            assert_eq!(offset.align_to(padding_align) + padding, layout.size);
            result.push(cx.type_padding_filler(padding, padding_align));
        }
    }

    (result, packed)
}

impl<'tcx> CodegenCx<'_, 'tcx> {
    pub fn align_of(&self, ty: Ty<'tcx>) -> Align {
        self.layout_of(ty).align.abi
    }

    pub fn size_of(&self, ty: Ty<'tcx>) -> Size {
        self.layout_of(ty).size
    }

    pub fn size_and_align_of(&self, ty: Ty<'tcx>) -> (Size, Align) {
        let layout = self.layout_of(ty);
        (layout.size, layout.align.abi)
    }
}

impl<'tcx> TypeMembershipCodegenMethods<'tcx> for CodegenCx<'_, 'tcx> {
    fn add_type_metadata(&self, _function: Self::Function, _typeid: String) {}

    fn set_type_metadata(&self, _function: Self::Function, _typeid: String) {}

    fn typeid_metadata(&self, _typeid: String) -> Option<Self::Metadata> {
        None
    }

    fn add_kcfi_type_metadata(&self, _function: Self::Function, _typeid: u32) {}

    fn set_kcfi_type_metadata(&self, _function: Self::Function, _typeid: u32) {}
}
