use crate::llvm::{self, AttributePlace::*, Value};
use rustc_ast::{LitKind, MetaItemInner, MetaItemLit};
use rustc_hir::Attribute;
use rustc_hir::attrs::{InlineAttr, OptimizeAttr};
use rustc_middle::{middle::codegen_fn_attrs::CodegenFnAttrFlags, ty};
use rustc_session::{Session, config::OptLevel};
use rustc_span::{Symbol, sym};

use crate::context::CodegenCx;

#[inline] // so meta
fn inline(val: &'_ Value, inline: InlineAttr) {
    use InlineAttr::*;
    match inline {
        Hint => llvm::Attribute::InlineHint.apply_llfn(Function, val),
        Always => llvm::Attribute::AlwaysInline.apply_llfn(Function, val),
        Never => llvm::Attribute::NoInline.apply_llfn(Function, val),
        None => {}
        Force { .. } => llvm::Attribute::AlwaysInline.apply_llfn(Function, val),
    }
}

pub(crate) fn default_optimisation_attrs(sess: &Session, llfn: &'_ Value) {
    match sess.opts.optimize {
        OptLevel::Size => {
            llvm::Attribute::MinSize.unapply_llfn(Function, llfn);
            llvm::Attribute::OptimizeForSize.apply_llfn(Function, llfn);
            llvm::Attribute::OptimizeNone.unapply_llfn(Function, llfn);
        }
        OptLevel::SizeMin => {
            llvm::Attribute::MinSize.apply_llfn(Function, llfn);
            llvm::Attribute::OptimizeForSize.apply_llfn(Function, llfn);
            llvm::Attribute::OptimizeNone.unapply_llfn(Function, llfn);
        }
        OptLevel::No => {
            llvm::Attribute::MinSize.unapply_llfn(Function, llfn);
            llvm::Attribute::OptimizeForSize.unapply_llfn(Function, llfn);
            llvm::Attribute::OptimizeNone.unapply_llfn(Function, llfn);
        }
        _ => {}
    }
}

/// Composite function which sets LLVM attributes for function depending on its AST (`#[attribute]`)
/// attributes.
pub(crate) fn from_fn_attrs<'ll, 'tcx>(
    cx: &CodegenCx<'ll, 'tcx>,
    llfn: &'ll Value,
    instance: ty::Instance<'tcx>,
) {
    let codegen_fn_attrs = cx.tcx.codegen_fn_attrs(instance.def_id());

    match codegen_fn_attrs.optimize {
        OptimizeAttr::Default => {
            default_optimisation_attrs(cx.tcx.sess, llfn);
        }
        OptimizeAttr::Speed => {
            llvm::Attribute::MinSize.unapply_llfn(Function, llfn);
            llvm::Attribute::OptimizeForSize.unapply_llfn(Function, llfn);
            llvm::Attribute::OptimizeNone.unapply_llfn(Function, llfn);
        }
        OptimizeAttr::Size => {
            llvm::Attribute::MinSize.apply_llfn(Function, llfn);
            llvm::Attribute::OptimizeForSize.apply_llfn(Function, llfn);
            llvm::Attribute::OptimizeNone.unapply_llfn(Function, llfn);
        }
        OptimizeAttr::DoNotOptimize => {
            llvm::Attribute::MinSize.unapply_llfn(Function, llfn);
            llvm::Attribute::OptimizeForSize.unapply_llfn(Function, llfn);
            llvm::Attribute::OptimizeNone.apply_llfn(Function, llfn);
        }
    }

    let inline_attr = if codegen_fn_attrs.flags.contains(CodegenFnAttrFlags::NAKED) {
        InlineAttr::Never
    } else if codegen_fn_attrs.inline == InlineAttr::None && instance.def.requires_inline(cx.tcx) {
        InlineAttr::Hint
    } else {
        codegen_fn_attrs.inline
    };
    inline(llfn, inline_attr);

    if codegen_fn_attrs.flags.contains(CodegenFnAttrFlags::COLD) {
        llvm::Attribute::Cold.apply_llfn(Function, llfn);
    }
    if codegen_fn_attrs
        .flags
        .contains(CodegenFnAttrFlags::FFI_PURE)
    {
        llvm::Attribute::ReadOnly.apply_llfn(Function, llfn);
    }
    if codegen_fn_attrs
        .flags
        .contains(CodegenFnAttrFlags::FFI_CONST)
    {
        llvm::Attribute::ReadNone.apply_llfn(Function, llfn);
    }
}

pub struct Symbols {
    pub nvvm_internal: Symbol,
    pub kernel: Symbol,
    pub addrspace: Symbol,
}

// inspired by rust-gpu's attribute handling
#[derive(Default, Clone, PartialEq)]
pub(crate) struct NvvmAttributes {
    pub kernel: bool,
    pub used: bool,
    pub addrspace: Option<u8>,
}

impl NvvmAttributes {
    pub fn parse<'tcx>(cx: &CodegenCx<'_, 'tcx>, attrs: &'tcx [Attribute]) -> Self {
        let mut nvvm_attrs = Self::default();

        for attr in attrs {
            if attr.path_matches(&[cx.symbols.nvvm_internal, cx.symbols.kernel]) {
                nvvm_attrs.kernel = true;
            } else if attr.path_matches(&[cx.symbols.nvvm_internal, sym::used]) {
                nvvm_attrs.used = true;
            } else if attr.path_matches(&[cx.symbols.nvvm_internal, cx.symbols.addrspace]) {
                let args = attr.meta_item_list().unwrap_or_default();
                if let Some(MetaItemInner::Lit(MetaItemLit {
                    kind: LitKind::Int(val, _),
                    ..
                })) = args.first()
                {
                    nvvm_attrs.addrspace = Some(val.get() as u8);
                } else {
                    panic!();
                }
            }
        }
        nvvm_attrs
    }
}
