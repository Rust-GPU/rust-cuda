use crate::LlvmMod;
use crate::llvm::{self, False, True};
use crate::target;
use libc::c_uint;
use rustc_ast::expand::allocator::{
    AllocatorMethod, AllocatorTy, NO_ALLOC_SHIM_IS_UNSTABLE, default_fn_name, global_fn_name,
};
use rustc_middle::ty::TyCtxt;
use rustc_symbol_mangling::mangle_internal_symbol;

// adapted from rustc_codegen_llvm
pub(crate) fn codegen(
    tcx: TyCtxt<'_>,
    mods: &mut LlvmMod,
    _module_name: &str,
    methods: &[AllocatorMethod],
) {
    unsafe {
        let llcx = &*mods.llcx;
        let llmod = mods.llmod.as_ref().unwrap();
        let usize = target::usize_ty(llcx);
        let i8 = llvm::LLVMInt8TypeInContext(llcx);
        let i8p = llvm::LLVMPointerType(i8, 0);
        let void = llvm::LLVMVoidTypeInContext(llcx);

        let mut used = Vec::new();

        for method in methods {
            let mut args = Vec::with_capacity(method.inputs.len());
            for input in method.inputs {
                match input.ty {
                    AllocatorTy::Layout => {
                        args.push(usize);
                        args.push(usize);
                    }
                    AllocatorTy::Ptr => args.push(i8p),
                    AllocatorTy::Usize => args.push(usize),
                    AllocatorTy::Never | AllocatorTy::ResultPtr | AllocatorTy::Unit => {
                        panic!("invalid allocator arg")
                    }
                }
            }

            let no_return = matches!(method.output, AllocatorTy::Never);
            let output = match method.output {
                AllocatorTy::ResultPtr => Some(i8p),
                AllocatorTy::Never | AllocatorTy::Unit => None,
                AllocatorTy::Layout | AllocatorTy::Usize | AllocatorTy::Ptr => {
                    panic!("invalid allocator output")
                }
            };

            let ty = llvm::LLVMFunctionType(
                output.unwrap_or(void),
                args.as_ptr(),
                args.len() as c_uint,
                False,
            );
            let from_name = mangle_internal_symbol(tcx, &global_fn_name(method.name));
            let llfn = llvm::LLVMRustGetOrInsertFunction(
                llmod,
                from_name.as_ptr().cast(),
                from_name.len(),
                ty,
            );
            used.push(llfn);
            if no_return {
                llvm::Attribute::NoReturn.apply_llfn(llvm::AttributePlace::Function, llfn);
            }

            let to_name = mangle_internal_symbol(tcx, &default_fn_name(method.name));
            let callee = llvm::LLVMRustGetOrInsertFunction(
                llmod,
                to_name.as_ptr().cast(),
                to_name.len(),
                ty,
            );
            used.push(callee);
            if no_return {
                llvm::Attribute::NoReturn.apply_llfn(llvm::AttributePlace::Function, callee);
            }
            llvm::LLVMRustSetVisibility(callee, llvm::Visibility::Hidden);

            let llbb = llvm::LLVMAppendBasicBlockInContext(llcx, llfn, c"entry".as_ptr().cast());
            let llbuilder = llvm::LLVMCreateBuilderInContext(llcx);
            llvm::LLVMPositionBuilderAtEnd(llbuilder, llbb);
            let args = args
                .iter()
                .enumerate()
                .map(|(i, _)| llvm::LLVMGetParam(llfn, i as c_uint))
                .collect::<Vec<_>>();
            let ret = llvm::LLVMRustBuildCall(
                llbuilder,
                callee,
                args.as_ptr(),
                args.len() as c_uint,
                None,
            );
            llvm::LLVMSetTailCall(ret, True);
            if output.is_some() {
                llvm::LLVMBuildRet(llbuilder, ret);
            } else {
                llvm::LLVMBuildRetVoid(llbuilder);
            }
            llvm::LLVMDisposeBuilder(llbuilder);
        }

        let shim_ty = llvm::LLVMFunctionType(void, std::ptr::null(), 0, False);
        let shim_name = mangle_internal_symbol(tcx, NO_ALLOC_SHIM_IS_UNSTABLE);
        let shim = llvm::LLVMRustGetOrInsertFunction(
            llmod,
            shim_name.as_ptr().cast(),
            shim_name.len(),
            shim_ty,
        );
        used.push(shim);

        let ptr_ty = llvm::LLVMPointerType(llvm::LLVMInt8TypeInContext(llcx), 0);
        for used in &mut used {
            *used = llvm::LLVMConstBitCast(used, ptr_ty);
        }

        let section = c"llvm.metadata";
        let array = llvm::LLVMConstArray(ptr_ty, used.as_ptr(), used.len() as u32);
        let g = llvm::LLVMAddGlobal(llmod, llvm::LLVMTypeOf(array), c"llvm.used".as_ptr().cast());
        llvm::LLVMSetInitializer(g, array);
        llvm::LLVMRustSetLinkage(g, llvm::Linkage::AppendingLinkage);
        llvm::LLVMSetSection(g, section.as_ptr());
    }
}
