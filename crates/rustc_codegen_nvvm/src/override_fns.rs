//! Functions for overriding certain functions in certain crates with special
//! codegen-builtin methods. Currently the only use for this is overriding libm functions
//! with libdevice intrinsics (which are much faster and smaller).

use crate::abi::FnAbiLlvmExt;
use crate::builder::Builder;
use crate::context::CodegenCx;
use crate::llvm;
use rustc_codegen_ssa::mono_item::MonoItemExt;
use rustc_codegen_ssa::traits::{BaseTypeCodegenMethods, BuilderMethods};
use rustc_hir::def::DefKind;
use rustc_hir::def_id::LOCAL_CRATE;
use rustc_middle::mir::mono::{Linkage, MonoItem, MonoItemData, Visibility};
use rustc_middle::ty::layout::FnAbiOf;
use rustc_middle::ty::{self, Instance};

/// Either override or define a function.
pub(crate) fn define_or_override_fn<'tcx>(func: Instance<'tcx>, cx: &mut CodegenCx<'_, 'tcx>) {
    if should_override(func, cx) {
        override_libm_function(func, cx);
    } else {
        MonoItem::define::<Builder<'_, '_, '_>>(
            &MonoItem::Fn(func),
            cx,
            "mono_item",
            MonoItemData {
                inlined: false,
                linkage: Linkage::External,
                visibility: Visibility::Default,
                size_estimate: 0,
            },
        );
    }
}

fn should_override<'tcx>(func: Instance<'tcx>, cx: &CodegenCx<'_, 'tcx>) -> bool {
    if !cx.codegen_args.override_libm {
        return false;
    }

    // there is no better way to do this without putting some sort of diagnostic/lang item in libm
    let is_libm = cx.tcx.crate_name(LOCAL_CRATE).as_str() == "libm";
    if !is_libm {
        return false;
    }

    // Only try to override top-level/assoc functions; closures/anon fns cause ICE via item_name.
    match cx.tcx.def_kind(func.def_id()) {
        DefKind::Fn | DefKind::AssocFn => {}
        _ => return false,
    }

    let sym = cx.tcx.item_name(func.def_id());
    let name = sym.as_str();

    if is_unsupported_libdevice_fn(name) {
        return false;
    }

    let libdevice_name = format!("__nv_{name}");
    let ld_fn = if let Some((args, ret)) = cx.intrinsics_map.borrow().get(libdevice_name.as_str()) {
        cx.type_func(args, ret)
    } else {
        return false;
    };

    // Check the function signatures match.
    let lm_fn = cx.fn_abi_of_instance(func, ty::List::empty()).llvm_type(cx);
    lm_fn == ld_fn
}

fn is_unsupported_libdevice_fn(name: &str) -> bool {
    // libm functions for which libdevice has no intrinsics for.
    const UNSUPPORTED: &[&str] = &[
        // doesnt exist
        "lgamma_r",
        "lgammaf_r",
        // different signatures
        "sincos",
        "sincosf",
    ];
    UNSUPPORTED.contains(&name)
}

fn override_libm_function<'tcx>(func: Instance<'tcx>, cx: &CodegenCx<'_, 'tcx>) {
    let name = cx.tcx.item_name(func.def_id());
    let nv_name = format!("__nv_{}", name.as_str());
    let (intrinsic_llfn_ty, intrinsic_llfn) = cx.get_intrinsic(nv_name.as_str());

    let llfn = cx.get_fn(func);
    let start = Builder::append_block(cx, llfn, "start");
    let mut bx = Builder::build(cx, start);

    let params = llvm::get_params(llfn);
    let llcall = bx.call(
        intrinsic_llfn_ty,
        None,
        None,
        intrinsic_llfn,
        &params,
        None,
        None,
    );
    bx.ret(llcall);
}
