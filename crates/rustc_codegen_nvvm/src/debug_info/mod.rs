use std::cell::OnceCell;
use std::cell::RefCell;
use std::ffi::CString;
use std::iter;
use std::ops::Range;
use std::sync::Arc;

use libc::c_uint;
use rustc_abi::Size;
use rustc_codegen_ssa::debuginfo::type_names;
use rustc_codegen_ssa::mir::debuginfo::VariableKind::*;
use rustc_codegen_ssa::mir::debuginfo::{DebugScope, FunctionDebugContext, VariableKind};
use rustc_codegen_ssa::traits::*;
use rustc_data_structures::unord::UnordMap;
use rustc_hir::def_id::{DefId, DefIdMap};
use rustc_index::IndexVec;
use rustc_middle::mir;
use rustc_middle::ty::layout::{HasTyCtxt, HasTypingEnv};
use rustc_middle::ty::{self, GenericArgKind, GenericArgsRef, Instance, Ty, TypeVisitableExt};
use rustc_session::config::{self, DebugInfo};
use rustc_span::symbol::Symbol;
use rustc_span::{
    self, BytePos, Pos, SourceFile, SourceFileAndLine, SourceFileHash, Span, StableSourceFileId,
};
use rustc_target::callconv::FnAbi;
use smallvec::SmallVec;

use crate::builder::Builder;
use crate::common::AsCCharPtr;
use crate::context::CodegenCx;
use crate::debug_info::util::{create_DIArray, is_node_local_to_unit};
use crate::llvm::{self, Value, debuginfo::*};

use self::namespace::*;
use self::util::DIB;
use create_scope_map::compute_mir_scopes;

mod create_scope_map;
mod dwarf_const;
pub(crate) mod metadata;
mod namespace;
mod util;

pub(crate) use metadata::*;

#[allow(non_upper_case_globals)]
const DW_TAG_auto_variable: c_uint = 0x100;
#[allow(non_upper_case_globals)]
const DW_TAG_arg_variable: c_uint = 0x101;

pub struct CodegenUnitDebugContext<'ll, 'tcx> {
    #[allow(dead_code)]
    llcontext: &'ll llvm::Context,
    llmod: &'ll llvm::Module,
    builder: &'ll mut DIBuilder<'ll>,
    created_files: RefCell<UnordMap<Option<(StableSourceFileId, SourceFileHash)>, &'ll DIFile>>,

    type_map: metadata::TypeMap<'ll, 'tcx>,
    namespace_map: RefCell<DefIdMap<&'ll DIScope>>,
    recursion_marker_type: OnceCell<&'ll DIType>,
}

impl Drop for CodegenUnitDebugContext<'_, '_> {
    fn drop(&mut self) {
        unsafe {
            llvm::LLVMRustDIBuilderDispose(&mut *(self.builder as *mut _));
        }
    }
}

impl<'a> CodegenUnitDebugContext<'a, '_> {
    pub(crate) fn new(llmod: &'a llvm::Module) -> Self {
        let builder = unsafe { llvm::LLVMRustDIBuilderCreate(llmod) };
        // DIBuilder inherits context from the module, so we'd better use the same one
        let llcontext = unsafe { llvm::LLVMGetModuleContext(llmod) };
        CodegenUnitDebugContext {
            llcontext,
            llmod,
            builder,
            created_files: Default::default(),
            type_map: Default::default(),
            namespace_map: RefCell::new(Default::default()),
            recursion_marker_type: OnceCell::new(),
        }
    }

    pub(crate) fn finalize(&self) {
        unsafe {
            llvm::LLVMRustDIBuilderFinalize(self.builder);

            // Prevent bitcode readers from deleting the debug info.
            llvm::LLVMRustAddModuleFlag(
                self.llmod,
                c"Debug Info Version".as_ptr(),
                llvm::LLVMRustDebugMetadataVersion(),
            );
        }
    }
}

/// Creates any deferred debug metadata nodes
pub(crate) fn finalize(cx: &CodegenCx<'_, '_>) {
    if let Some(dbg_cx) = &cx.dbg_cx {
        dbg_cx.finalize();
    }
}

impl<'ll> DebugInfoBuilderMethods for Builder<'_, 'll, '_> {
    fn dbg_var_addr(
        &mut self,
        dbg_var: &'ll DIVariable,
        dbg_loc: &'ll DILocation,
        variable_alloca: &'ll Value,
        direct_offset: Size,
        indirect_offsets: &[Size],
        fragment: Option<Range<Size>>,
    ) {
        use dwarf_const::{DW_OP_LLVM_fragment, DW_OP_deref, DW_OP_plus_uconst};

        let mut addr_ops = SmallVec::<[i64; 8]>::new();

        if direct_offset.bytes() > 0 {
            addr_ops.push(DW_OP_plus_uconst);
            addr_ops.push(direct_offset.bytes() as i64);
        }
        for &offset in indirect_offsets {
            addr_ops.push(DW_OP_deref);
            if offset.bytes() > 0 {
                addr_ops.push(DW_OP_plus_uconst);
                addr_ops.push(offset.bytes() as i64);
            }
        }

        if let Some(fragment) = fragment {
            // `DW_OP_LLVM_fragment` takes as arguments the fragment's
            // offset and size, both of them in bits.
            addr_ops.push(DW_OP_LLVM_fragment);
            addr_ops.push(fragment.start.bits() as i64);
            addr_ops.push((fragment.end - fragment.start).bits() as i64);
        }

        unsafe {
            llvm::LLVMRustDIBuilderInsertDeclareAtEnd(
                DIB(self.cx()),
                variable_alloca,
                dbg_var,
                addr_ops.as_ptr(),
                addr_ops.len() as c_uint,
                dbg_loc,
                self.llbb(),
            );
        }
    }

    fn set_dbg_loc(&mut self, dbg_loc: &'ll DILocation) {
        unsafe {
            let dbg_loc_as_llval = llvm::LLVMRustMetadataAsValue(self.cx().llcx, dbg_loc);
            llvm::LLVMSetCurrentDebugLocation(self.llbuilder, Some(dbg_loc_as_llval));
        }
    }

    fn insert_reference_to_gdb_debug_scripts_section_global(&mut self) {
        // do nothing
    }

    fn set_var_name(&mut self, value: &'ll Value, name: &str) {
        if self.sess().fewer_names() {
            return;
        }

        // Only function parameters and instructions are local to a function,
        // don't change the name of anything else (e.g. globals).
        let param_or_inst = unsafe {
            llvm::LLVMIsAArgument(value).is_some() || llvm::LLVMIsAInstruction(value).is_some()
        };
        if !param_or_inst {
            return;
        }

        // Avoid replacing the name if it already exists.
        // While we could combine the names somehow, it'd
        // get noisy quick, and the usefulness is dubious.
        if llvm::get_value_name(value).is_empty() {
            llvm::set_value_name(value, name.as_bytes());
        }
    }

    fn clear_dbg_loc(&mut self) {
        unsafe {
            llvm::LLVMSetCurrentDebugLocation(self.llbuilder, None);
        }
    }
}

/// A source code location used to generate debug information.
pub struct DebugLoc {
    /// Information about the original source file.
    pub file: Arc<SourceFile>,
    /// The (1-based) line number.
    pub line: u32,
    /// The (1-based) column number.
    pub col: u32,
}

impl CodegenCx<'_, '_> {
    /// Looks up debug source information about a `BytePos`.
    pub fn lookup_debug_loc(&self, pos: BytePos) -> DebugLoc {
        let (file, line, col) = match self.sess().source_map().lookup_line(pos) {
            Ok(SourceFileAndLine { sf: file, line }) => {
                let line_pos = file.line_begin_pos(pos);

                // Use 1-based indexing.
                let line = (line + 1) as u32;
                let col = (pos - line_pos).to_u32() + 1;

                (file, line, col)
            }
            Err(file) => (file, UNKNOWN_LINE_NUMBER, UNKNOWN_COLUMN_NUMBER),
        };

        DebugLoc { file, line, col }
    }
}

impl<'ll, 'tcx> DebugInfoCodegenMethods<'tcx> for CodegenCx<'ll, 'tcx> {
    fn create_function_debug_context(
        &self,
        instance: Instance<'tcx>,
        fn_abi: &FnAbi<'tcx, Ty<'tcx>>,
        llfn: Self::Function,
        mir: &mir::Body<'tcx>,
    ) -> Option<FunctionDebugContext<'tcx, Self::DIScope, Self::DILocation>> {
        if self.sess().opts.debuginfo == DebugInfo::None {
            return None;
        }

        // Initialize fn debug context (including scopes).
        let empty_scope = DebugScope {
            dbg_scope: self.dbg_scope_fn(instance, fn_abi, Some(llfn)),
            inlined_at: None,
            file_start_pos: BytePos(0),
            file_end_pos: BytePos(0),
        };
        let mut fn_debug_context = FunctionDebugContext {
            scopes: IndexVec::from_elem(empty_scope, &mir.source_scopes),
            inlined_function_scopes: Default::default(),
        };

        // Fill in all the scopes, with the information from the MIR body.
        compute_mir_scopes(self, instance, mir, &mut fn_debug_context);

        Some(fn_debug_context)
    }

    fn dbg_scope_fn(
        &self,
        instance: Instance<'tcx>,
        fn_abi: &FnAbi<'tcx, Ty<'tcx>>,
        maybe_definition_llfn: Option<&'ll Value>,
    ) -> &'ll DIScope {
        let tcx = self.tcx();

        let def_id = instance.def_id();
        let containing_scope = get_containing_scope(self, instance);
        let span = tcx.def_span(def_id);
        let loc = self.lookup_debug_loc(span.lo());
        let file_metadata = file_metadata(self, &loc.file);

        let function_type_metadata = unsafe {
            let fn_signature = get_function_signature(self, fn_abi);
            llvm::LLVMRustDIBuilderCreateSubroutineType(DIB(self), fn_signature)
        };

        let mut name = String::new();
        type_names::push_item_name(tcx, def_id, false, &mut name);

        // Find the enclosing function, in case this is a closure.
        let enclosing_fn_def_id = tcx.typeck_root_def_id(def_id);

        // Get_template_parameters() will append a `<...>` clause to the function
        // name if necessary.
        let generics = tcx.generics_of(enclosing_fn_def_id);
        let args = instance.args.truncate_to(tcx, generics);
        let template_parameters = get_template_parameters(self, generics, args);

        let linkage_name = &mangled_name_of_instance(self, instance).name;
        // Omit the linkage_name if it is the same as subprogram name.
        let linkage_name = if &name == linkage_name {
            ""
        } else {
            linkage_name
        };
        let name = CString::new(name).unwrap();
        let linkage_name = CString::new(linkage_name).unwrap();

        let scope_line = loc.line;

        let mut flags = DIFlags::FlagPrototyped;

        if fn_abi.ret.layout.is_uninhabited() {
            flags |= DIFlags::FlagNoReturn;
        }

        unsafe {
            return llvm::LLVMRustDIBuilderCreateFunction(
                DIB(self),
                containing_scope.0,
                name.as_ptr(),
                linkage_name.as_ptr(),
                file_metadata,
                loc.line,
                function_type_metadata,
                is_node_local_to_unit(self, def_id),
                true,
                scope_line,
                flags,
                self.sess().opts.optimize != config::OptLevel::No,
                maybe_definition_llfn,
                template_parameters,
                None,
            );
        }

        fn get_function_signature<'ll, 'tcx>(
            cx: &CodegenCx<'ll, 'tcx>,
            fn_abi: &FnAbi<'tcx, Ty<'tcx>>,
        ) -> &'ll DIArray {
            if cx.sess().opts.debuginfo == DebugInfo::Limited {
                return create_DIArray(DIB(cx), &[]);
            }

            let mut signature = Vec::with_capacity(fn_abi.args.len() + 1);

            // Return type -- llvm::DIBuilder wants this at index 0
            signature.push(if fn_abi.ret.is_ignore() {
                None
            } else {
                Some(type_di_node(cx, fn_abi.ret.layout.ty))
            });

            signature.extend(
                fn_abi
                    .args
                    .iter()
                    .map(|arg| Some(type_di_node(cx, arg.layout.ty))),
            );

            create_DIArray(DIB(cx), &signature[..])
        }

        fn get_template_parameters<'ll, 'tcx>(
            cx: &CodegenCx<'ll, 'tcx>,
            generics: &ty::Generics,
            args: GenericArgsRef<'tcx>,
        ) -> &'ll DIArray {
            if args.types().next().is_none() {
                return create_DIArray(DIB(cx), &[]);
            }

            // Again, only create type information if full debuginfo is enabled
            let template_params: Vec<_> = if cx.sess().opts.debuginfo == DebugInfo::Full {
                let names = get_parameter_names(cx, generics);
                iter::zip(args, names)
                    .filter_map(|(kind, name)| {
                        if let GenericArgKind::Type(ty) = kind.kind() {
                            let actual_type = cx.tcx.normalize_erasing_regions(cx.typing_env(), ty);
                            let actual_type_metadata = type_di_node(cx, actual_type);
                            let name = name.as_str();
                            Some(unsafe {
                                Some(llvm::LLVMRustDIBuilderCreateTemplateTypeParameter(
                                    DIB(cx),
                                    None,
                                    name.as_c_char_ptr(),
                                    name.len(),
                                    actual_type_metadata,
                                ))
                            })
                        } else {
                            None
                        }
                    })
                    .collect()
            } else {
                vec![]
            };

            create_DIArray(DIB(cx), &template_params)
        }

        fn get_parameter_names(cx: &CodegenCx<'_, '_>, generics: &ty::Generics) -> Vec<Symbol> {
            let mut names = generics.parent.map_or_else(Vec::new, |def_id| {
                get_parameter_names(cx, cx.tcx.generics_of(def_id))
            });
            names.extend(generics.own_params.iter().map(|param| param.name));
            names
        }

        fn get_containing_scope<'ll, 'tcx>(
            cx: &CodegenCx<'ll, 'tcx>,
            instance: Instance<'tcx>,
        ) -> (&'ll DIScope, bool) {
            // First, let's see if this is a method within an inherent impl. Because
            // if yes, we want to make the result subroutine DIE a child of the
            // subroutine's self-type.
            if let Some(impl_def_id) = cx.tcx.impl_of_assoc(instance.def_id()) {
                // If the method does *not* belong to a trait, proceed
                if cx.tcx.trait_id_of_impl(impl_def_id).is_none() {
                    let impl_self_ty = cx.tcx.instantiate_and_normalize_erasing_regions(
                        instance.args,
                        cx.typing_env(),
                        cx.tcx.type_of(impl_def_id),
                    );

                    // Only "class" methods are generally understood by LLVM,
                    // so avoid methods on other types (e.g., `<*mut T>::null`).
                    if let ty::Adt(def, ..) = impl_self_ty.kind()
                        && !def.is_box()
                    {
                        // Again, only create type information if full debuginfo is enabled
                        if cx.sess().opts.debuginfo == DebugInfo::Full && !impl_self_ty.has_param()
                        {
                            return (type_di_node(cx, impl_self_ty), true);
                        } else {
                            return (namespace::item_namespace(cx, def.did()), false);
                        }
                    }
                } else {
                    // For trait method impls we still use the "parallel namespace"
                    // strategy
                }
            }
            let scope = namespace::item_namespace(
                cx,
                DefId {
                    krate: instance.def_id().krate,
                    index: cx
                        .tcx
                        .def_key(instance.def_id())
                        .parent
                        .expect("get_containing_scope: missing parent?"),
                },
            );

            (scope, false)
        }
    }

    fn dbg_loc(
        &self,
        scope: Self::DIScope,
        inlined_at: Option<Self::DILocation>,
        span: Span,
    ) -> Self::DILocation {
        let DebugLoc { line, col, .. } = self.lookup_debug_loc(span.lo());

        unsafe { llvm::LLVMRustDIBuilderCreateDebugLocation(line, col, scope, inlined_at) }
    }

    fn create_vtable_debuginfo(
        &self,
        ty: Ty<'tcx>,
        trait_ref: Option<ty::ExistentialTraitRef<'tcx>>,
        vtable: Self::Value,
    ) {
        metadata::create_vtable_di_node(self, ty, trait_ref, vtable)
    }

    fn extend_scope_to_file(
        &self,
        scope_metadata: Self::DIScope,
        file: &SourceFile,
    ) -> Self::DIScope {
        metadata::extend_scope_to_file(self, scope_metadata, file)
    }

    fn debuginfo_finalize(&self) {
        finalize(self)
    }

    fn create_dbg_var(
        &self,
        variable_name: Symbol,
        variable_type: Ty<'tcx>,
        scope_metadata: Self::DIScope,
        variable_kind: VariableKind,
        span: Span,
    ) -> Self::DIVariable {
        let loc = self.lookup_debug_loc(span.lo());
        let file_metadata = file_metadata(self, &loc.file);

        let type_metadata = type_di_node(self, variable_type);

        let (argument_index, dwarf_tag) = match variable_kind {
            ArgumentVariable(index) => (index as c_uint, DW_TAG_arg_variable),
            LocalVariable => (0, DW_TAG_auto_variable),
        };
        let align = self.align_of(variable_type);

        let name = CString::new(variable_name.as_str()).unwrap();
        unsafe {
            llvm::LLVMRustDIBuilderCreateVariable(
                DIB(self),
                dwarf_tag,
                scope_metadata,
                name.as_ptr().cast(),
                file_metadata,
                loc.line,
                type_metadata,
                true,
                DIFlags::FlagZero,
                argument_index,
                align.bytes() as u32,
            )
        }
    }
}
