use crate::llvm::{self, Bool, False, True, Type, Value};
use crate::{consts::const_alloc_to_llvm, context::CodegenCx, ty::LayoutLlvmExt};
use libc::c_uint;
use rustc_abi as abi;
use rustc_abi::Primitive::Pointer;
use rustc_abi::{AddressSpace, HasDataLayout};
use rustc_ast::Mutability;
use rustc_codegen_ssa::common::TypeKind;
use rustc_codegen_ssa::traits::*;
use rustc_data_structures::stable_hasher::{HashStable, StableHasher};
use rustc_hashes::Hash128;
use rustc_middle::bug;
use rustc_middle::mir::interpret::{ConstAllocation, GlobalAlloc, Scalar};
use rustc_middle::ty::layout::LayoutOf;
use tracing::trace;

impl<'ll, 'tcx> ConstCodegenMethods for CodegenCx<'ll, 'tcx> {
    fn const_data_from_alloc(&self, alloc: ConstAllocation) -> &'ll Value {
        const_alloc_to_llvm(self, alloc, /*static*/ false)
    }

    fn const_null(&self, t: &'ll Type) -> &'ll Value {
        unsafe { llvm::LLVMConstNull(t) }
    }

    fn const_undef(&self, t: &'ll Type) -> &'ll Value {
        unsafe { llvm::LLVMGetUndef(t) }
    }

    fn const_int(&self, t: &'ll Type, i: i64) -> &'ll Value {
        unsafe { llvm::LLVMConstInt(t, i as u64, True) }
    }

    fn const_uint(&self, t: &'ll Type, i: u64) -> &'ll Value {
        debug_assert!(
            self.type_kind(t) == TypeKind::Integer,
            "only allows integer types in const_int"
        );
        unsafe { llvm::LLVMConstInt(t, i, False) }
    }

    fn const_uint_big(&self, t: &'ll Type, u: u128) -> &'ll Value {
        debug_assert!(
            self.type_kind(t) == TypeKind::Integer,
            "only allows integer types in const_uint_big"
        );
        unsafe {
            let words = [u as u64, (u >> 64) as u64];
            llvm::LLVMConstIntOfArbitraryPrecision(t, 2, words.as_ptr())
        }
    }

    fn const_bool(&self, val: bool) -> &'ll Value {
        self.const_uint(self.type_i1(), val as u64)
    }

    fn const_i32(&self, i: i32) -> &'ll Value {
        self.const_int(self.type_i32(), i as i64)
    }

    fn const_u32(&self, i: u32) -> &'ll Value {
        self.const_uint(self.type_i32(), i as u64)
    }

    fn const_u64(&self, i: u64) -> &'ll Value {
        self.const_uint(self.type_i64(), i)
    }

    fn const_usize(&self, i: u64) -> &'ll Value {
        self.const_uint(self.isize_ty, i)
    }

    fn const_u8(&self, i: u8) -> &'ll Value {
        self.const_uint(self.type_i8(), i as u64)
    }

    fn const_real(&self, t: &'ll Type, val: f64) -> &'ll Value {
        unsafe { llvm::LLVMConstReal(t, val) }
    }

    fn const_str(&self, s: &str) -> (&'ll Value, &'ll Value) {
        let val = *self
            .const_cstr_cache
            .borrow_mut()
            .entry(s.to_string())
            .or_insert_with(|| {
                let sc = self.const_bytes(s.as_bytes());
                let sym = self.generate_local_symbol_name("str");
                let g = self
                    .define_global(&sym[..], self.val_ty(sc), AddressSpace::DATA)
                    .unwrap_or_else(|| {
                        bug!("symbol `{}` is already defined", sym);
                    });
                unsafe {
                    llvm::LLVMSetInitializer(g, sc);
                    llvm::LLVMSetGlobalConstant(g, True);
                    llvm::LLVMRustSetLinkage(g, llvm::Linkage::InternalLinkage);
                }
                g
            });
        let len = s.len();
        let ty = self.type_ptr_to(self.layout_of(self.tcx.types.str_).llvm_type(self));
        let cs = unsafe { llvm::LLVMConstPointerCast(val, ty) };
        (cs, self.const_usize(len as u64))
    }

    fn const_struct(&self, elts: &[&'ll Value], packed: bool) -> &'ll Value {
        unsafe {
            llvm::LLVMConstStructInContext(
                self.llcx,
                elts.as_ptr(),
                elts.len() as c_uint,
                packed as Bool,
            )
        }
    }

    fn const_to_opt_uint(&self, v: &'ll Value) -> Option<u64> {
        unsafe { llvm::LLVMIsAConstantInt(v).map(|v| llvm::LLVMConstIntGetZExtValue(v)) }
    }

    fn const_to_opt_u128(&self, v: &'ll Value, sign_ext: bool) -> Option<u128> {
        unsafe {
            llvm::LLVMIsAConstantInt(v).and_then(|v| {
                let (mut lo, mut hi) = (0u64, 0u64);
                let success = llvm::LLVMRustConstInt128Get(v, sign_ext, &mut hi, &mut lo);
                success.then(|| hi_lo_to_u128(lo, hi))
            })
        }
    }

    fn scalar_to_backend(
        &self,
        cv: Scalar,
        layout: abi::Scalar,
        mut llty: &'ll Type,
    ) -> &'ll Value {
        trace!("Scalar to backend `{:?}`, `{:?}`, `{:?}`", cv, layout, llty);
        let bitsize = if layout.is_bool() {
            1
        } else {
            layout.size(self).bits()
        };
        let val = match cv {
            Scalar::Int(int) => {
                let data = int.to_bits(layout.size(self));
                let llval = self.const_uint_big(self.type_ix(bitsize), data);
                if matches!(layout.primitive(), abi::Primitive::Pointer(_)) {
                    unsafe { llvm::LLVMConstIntToPtr(llval, llty) }
                } else {
                    self.const_bitcast(llval, llty)
                }
            }
            Scalar::Ptr(ptr, _) => {
                let (prov, offset) = ptr.prov_and_relative_offset();
                let (base_addr, base_addr_space) = match self.tcx.global_alloc(prov.alloc_id()) {
                    GlobalAlloc::Memory(alloc) => {
                        // For ZSTs directly codegen an aligned pointer.
                        // This avoids generating a zero-sized constant value and actually needing a
                        // real address at runtime.
                        if alloc.inner().len() == 0 {
                            assert_eq!(offset.bytes(), 0);
                            let llval = self.const_usize(alloc.inner().align.bytes());
                            return if matches!(layout.primitive(), abi::Primitive::Pointer(_)) {
                                unsafe { llvm::LLVMConstIntToPtr(llval, llty) }
                            } else {
                                self.const_bitcast(llval, llty)
                            };
                        } else {
                            let init = const_alloc_to_llvm(self, alloc, /*static*/ false);
                            let alloc = alloc.inner();
                            let value = match alloc.mutability {
                                Mutability::Mut => self.static_addr_of_mut(init, alloc.align, None),
                                _ => self.static_addr_of(init, alloc.align, None),
                            };
                            if !self.sess().fewer_names() && llvm::get_value_name(value).is_empty()
                            {
                                let hash = self.tcx.with_stable_hashing_context(|mut hcx| {
                                    let mut hasher = StableHasher::new();
                                    alloc.hash_stable(&mut hcx, &mut hasher);
                                    hasher.finish::<Hash128>()
                                });
                                llvm::set_value_name(
                                    value,
                                    format!("alloc_{hash:032x}").as_bytes(),
                                );
                            }
                            (value, AddressSpace::DATA)
                        }
                    }
                    GlobalAlloc::Function { instance, .. } => (
                        self.get_fn_addr(instance),
                        self.data_layout().instruction_address_space,
                    ),
                    GlobalAlloc::VTable(ty, dyn_ty) => {
                        let alloc = self
                            .tcx
                            .global_alloc(self.tcx.vtable_allocation((
                                ty,
                                dyn_ty.principal().map(|principal| {
                                    self.tcx.instantiate_bound_regions_with_erased(principal)
                                }),
                            )))
                            .unwrap_memory();
                        let init = const_alloc_to_llvm(self, alloc, /*static*/ false);
                        let value = self.static_addr_of(init, alloc.inner().align, None);
                        (value, AddressSpace::DATA)
                    }
                    GlobalAlloc::Static(def_id) => {
                        assert!(self.tcx.is_static(def_id));
                        assert!(!self.tcx.is_thread_local_static(def_id));
                        let val = self.get_static(def_id);
                        let addrspace =
                            unsafe { llvm::LLVMGetPointerAddressSpace(self.val_ty(val)) };
                        (val, AddressSpace(addrspace))
                    }
                };
                let llval = unsafe {
                    llvm::LLVMConstInBoundsGEP2(
                        self.type_i8(),
                        // Cast to the required address space if necessary
                        self.const_bitcast(base_addr, self.type_ptr_ext(base_addr_space)),
                        &self.const_usize(offset.bytes()),
                        1,
                    )
                };

                if !matches!(layout.primitive(), Pointer(_)) {
                    unsafe { llvm::LLVMConstPtrToInt(llval, llty) }
                } else {
                    if base_addr_space != AddressSpace::DATA {
                        unsafe {
                            let element = llvm::LLVMGetElementType(llty);
                            llty = self.type_ptr_to_ext(element, base_addr_space);
                        }
                    }
                    self.const_bitcast(llval, llty)
                }
            }
        };

        trace!("...Scalar to backend: `{:?}`", val);
        trace!("{:?}", std::backtrace::Backtrace::force_capture());

        val
    }

    fn const_poison(&self, t: Self::Type) -> Self::Value {
        // FIXME: Use LLVMGetPoision when possible.
        self.const_undef(t)
    }

    fn const_i8(&self, i: i8) -> Self::Value {
        self.const_int(self.type_i8(), i as i64)
    }

    fn const_i16(&self, i: i16) -> Self::Value {
        self.const_int(self.type_i16(), i as i64)
    }

    fn const_u128(&self, i: u128) -> Self::Value {
        trace!("const_u128 i = {i:?}");
        trace!("{}", std::backtrace::Backtrace::force_capture());
        self.const_uint_big(self.type_i128(), i)
    }

    fn const_vector(&self, elts: &[&'ll Value]) -> &'ll Value {
        let len = c_uint::try_from(elts.len()).expect("LLVMConstVector elements len overflow");
        unsafe { llvm::LLVMConstVector(elts.as_ptr(), len) }
    }

    fn const_ptr_byte_offset(&self, mut base_addr: Self::Value, offset: abi::Size) -> Self::Value {
        base_addr = self.const_ptrcast(base_addr, self.type_i8p());
        unsafe {
            llvm::LLVMConstInBoundsGEP2(
                self.type_i8(),
                base_addr,
                &self.const_usize(offset.bytes()),
                1,
            )
        }
    }
}

#[inline]
fn hi_lo_to_u128(lo: u64, hi: u64) -> u128 {
    ((hi as u128) << 64) | (lo as u128)
}
