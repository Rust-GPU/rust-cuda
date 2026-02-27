use crate::LlvmMod;
use crate::llvm::{self, False, True};
use crate::target;
use libc::c_uint;
use rustc_ast::expand::allocator::{
    ALLOCATOR_METHODS, AllocatorKind, AllocatorTy, alloc_error_handler_name, default_fn_name,
};
use rustc_middle::ty::TyCtxt;

// adapted from rustc_codegen_llvm
pub(crate) unsafe fn codegen(
    _tcx: TyCtxt<'_>,
    mods: &mut LlvmMod,
    _module_name: &str,
    kind: AllocatorKind,
    alloc_error_handler_kind: AllocatorKind,
) {
    let llcx = &*mods.llcx;
    let llmod = unsafe { mods.llmod.as_ref().unwrap() };
    let usize = unsafe { target::usize_ty(llcx) };
    let i8 = unsafe { llvm::LLVMInt8TypeInContext(llcx) };
    let i8p = unsafe { llvm::LLVMPointerType(i8, 0) };
    let void = unsafe { llvm::LLVMVoidTypeInContext(llcx) };

    let mut used = Vec::new();

    if kind == AllocatorKind::Default {
        for method in ALLOCATOR_METHODS {
            let mut args = Vec::with_capacity(method.inputs.len());
            for ty in method.inputs.iter() {
                match ty.ty {
                    AllocatorTy::Layout => {
                        args.push(usize); // size
                        args.push(usize); // align
                    }
                    AllocatorTy::Ptr => args.push(i8p),
                    AllocatorTy::Usize => args.push(usize),

                    AllocatorTy::ResultPtr | AllocatorTy::Unit => panic!("invalid allocator arg"),
                }
            }
            let output = match method.output {
                AllocatorTy::ResultPtr => Some(i8p),
                AllocatorTy::Unit => None,

                AllocatorTy::Layout | AllocatorTy::Usize | AllocatorTy::Ptr => {
                    panic!("invalid allocator output")
                }
            };

            let ty = unsafe {
                llvm::LLVMFunctionType(
                    output.unwrap_or(void),
                    args.as_ptr(),
                    args.len() as c_uint,
                    False,
                )
            };
            let name = format!("__rust_{}", method.name);
            let llfn = unsafe {
                llvm::LLVMRustGetOrInsertFunction(llmod, name.as_ptr().cast(), name.len(), ty)
            };

            used.push(llfn);
            // nvvm doesnt support uwtable so dont try to generate it

            let callee = default_fn_name(method.name);
            let callee = unsafe {
                llvm::LLVMRustGetOrInsertFunction(llmod, callee.as_ptr().cast(), callee.len(), ty)
            };
            unsafe { llvm::LLVMRustSetVisibility(callee, llvm::Visibility::Hidden) };

            let llbb = unsafe {
                llvm::LLVMAppendBasicBlockInContext(llcx, llfn, c"entry".as_ptr().cast())
            };

            let llbuilder = unsafe { llvm::LLVMCreateBuilderInContext(llcx) };
            unsafe { llvm::LLVMPositionBuilderAtEnd(llbuilder, llbb) };
            let args = args
                .iter()
                .enumerate()
                .map(|(i, _)| unsafe { llvm::LLVMGetParam(llfn, i as c_uint) })
                .collect::<Vec<_>>();
            let ret = unsafe {
                llvm::LLVMRustBuildCall(
                    llbuilder,
                    callee,
                    args.as_ptr(),
                    args.len() as c_uint,
                    None,
                )
            };
            unsafe { llvm::LLVMSetTailCall(ret, True) };
            if output.is_some() {
                unsafe { llvm::LLVMBuildRet(llbuilder, ret) };
            } else {
                unsafe { llvm::LLVMBuildRetVoid(llbuilder) };
            }
            unsafe { llvm::LLVMDisposeBuilder(llbuilder) };
        }
    }

    // rust alloc error handler
    let args = [usize, usize]; // size, align

    let ty = unsafe { llvm::LLVMFunctionType(void, args.as_ptr(), args.len() as c_uint, False) };
    let name = "__rust_alloc_error_handler".to_string();
    let llfn =
        unsafe { llvm::LLVMRustGetOrInsertFunction(llmod, name.as_ptr().cast(), name.len(), ty) };

    used.push(llfn);

    // -> ! DIFlagNoReturn
    llvm::Attribute::NoReturn.apply_llfn(llvm::AttributePlace::Function, llfn);

    let callee = alloc_error_handler_name(alloc_error_handler_kind);
    let callee = unsafe {
        llvm::LLVMRustGetOrInsertFunction(llmod, callee.as_ptr().cast(), callee.len(), ty)
    };

    used.push(callee);

    // -> ! DIFlagNoReturn
    llvm::Attribute::NoReturn.apply_llfn(llvm::AttributePlace::Function, callee);
    unsafe { llvm::LLVMRustSetVisibility(callee, llvm::Visibility::Hidden) };

    let llbb = unsafe { llvm::LLVMAppendBasicBlockInContext(llcx, llfn, c"entry".as_ptr().cast()) };

    let llbuilder = unsafe { llvm::LLVMCreateBuilderInContext(llcx) };
    unsafe { llvm::LLVMPositionBuilderAtEnd(llbuilder, llbb) };
    let args = args
        .iter()
        .enumerate()
        .map(|(i, _)| unsafe { llvm::LLVMGetParam(llfn, i as c_uint) })
        .collect::<Vec<_>>();
    let ret = unsafe {
        llvm::LLVMRustBuildCall(llbuilder, callee, args.as_ptr(), args.len() as c_uint, None)
    };
    unsafe { llvm::LLVMSetTailCall(ret, True) };
    unsafe { llvm::LLVMBuildRetVoid(llbuilder) };
    unsafe { llvm::LLVMDisposeBuilder(llbuilder) };

    for used in &mut used {
        *used = unsafe { llvm::LLVMConstBitCast(used, i8p) };
    }

    let section = c"llvm.metadata";
    let array = unsafe { llvm::LLVMConstArray(i8p, used.as_ptr(), used.len() as u32) };
    let g = unsafe {
        llvm::LLVMAddGlobal(llmod, llvm::LLVMTypeOf(array), c"llvm.used".as_ptr().cast())
    };
    unsafe { llvm::LLVMSetInitializer(g, array) };
    unsafe { llvm::LLVMRustSetLinkage(g, llvm::Linkage::AppendingLinkage) };
    unsafe { llvm::LLVMSetSection(g, section.as_ptr()) };
}
