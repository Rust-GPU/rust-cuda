use std::os::raw::{c_char, c_uint};

use crate::common::AsCCharPtr;
use crate::{
    llvm::{self, Value},
    ty::LayoutLlvmExt,
};
use rustc_ast::{InlineAsmOptions, InlineAsmTemplatePiece};
use rustc_codegen_ssa::{
    mir::operand::OperandValue,
    traits::{
        AsmBuilderMethods, AsmCodegenMethods, BaseTypeCodegenMethods, BuilderMethods,
        ConstCodegenMethods, GlobalAsmOperandRef, InlineAsmOperandRef,
    },
};
use rustc_data_structures::fx::FxHashMap;
use rustc_middle::{span_bug, ty::Instance};
use rustc_span::{Pos, Span};
use rustc_target::asm::{InlineAsmRegClass, InlineAsmRegOrRegClass, NvptxInlineAsmRegClass};

use crate::{builder::Builder, context::CodegenCx};

impl<'tcx> AsmBuilderMethods<'tcx> for Builder<'_, '_, 'tcx> {
    fn codegen_inline_asm(
        &mut self,
        template: &[InlineAsmTemplatePiece],
        operands: &[InlineAsmOperandRef<'tcx, Self>],
        options: rustc_ast::InlineAsmOptions,
        line_spans: &[Span],
        _inst: Instance,
        _dest: Option<Self::BasicBlock>,
        _catch_funclet: Option<(Self::BasicBlock, Option<&Self::Funclet>)>,
    ) {
        // Collect the types of output operands
        let mut constraints = vec![];
        let mut output_types = vec![];
        let mut op_idx = FxHashMap::default();
        for (idx, op) in operands.iter().enumerate() {
            match *op {
                InlineAsmOperandRef::Out { reg, late, place } => {
                    let ty = if let Some(ref place) = place {
                        place.layout.llvm_type(self)
                    } else {
                        match reg.reg_class() {
                            InlineAsmRegClass::Nvptx(NvptxInlineAsmRegClass::reg16) => {
                                self.type_i16()
                            }
                            InlineAsmRegClass::Nvptx(NvptxInlineAsmRegClass::reg32) => {
                                self.type_i32()
                            }
                            InlineAsmRegClass::Nvptx(NvptxInlineAsmRegClass::reg64) => {
                                self.type_i64()
                            }
                            _ => unreachable!(),
                        }
                    };
                    output_types.push(ty);
                    op_idx.insert(idx, constraints.len());
                    let prefix = if late { "=" } else { "=&" };
                    constraints.push(format!("{}{}", prefix, reg_to_llvm(reg)));
                }
                InlineAsmOperandRef::InOut {
                    reg,
                    late,
                    in_value,
                    out_place,
                } => {
                    let layout = if let Some(ref out_place) = out_place {
                        &out_place.layout
                    } else {
                        // LLVM required tied operands to have the same type,
                        // so we just use the type of the input.
                        &in_value.layout
                    };
                    let ty = layout.llvm_type(self);
                    output_types.push(ty);
                    op_idx.insert(idx, constraints.len());
                    let prefix = if late { "=" } else { "=&" };
                    constraints.push(format!("{}{}", prefix, reg_to_llvm(reg)));
                }
                _ => {}
            }
        }

        // Collect input operands
        let mut inputs = vec![];
        for (idx, op) in operands.iter().enumerate() {
            match *op {
                InlineAsmOperandRef::In { reg, value } => {
                    let llval = value.immediate();
                    inputs.push(llval);
                    op_idx.insert(idx, constraints.len());
                    constraints.push(reg_to_llvm(reg));
                }
                InlineAsmOperandRef::InOut {
                    late: _,
                    in_value,
                    out_place: _,
                    ..
                } => {
                    let value = in_value.immediate();
                    inputs.push(value);
                    constraints.push(format!("{}", op_idx[&idx]));
                }
                InlineAsmOperandRef::SymFn { instance } => {
                    inputs.push(self.cx.get_fn(instance));
                    op_idx.insert(idx, constraints.len());
                    constraints.push("s".to_string());
                }
                InlineAsmOperandRef::SymStatic { def_id } => {
                    inputs.push(self.cx.get_static(def_id));
                    op_idx.insert(idx, constraints.len());
                    constraints.push("s".to_string());
                }
                _ => {}
            }
        }

        // Build the template string
        let mut template_str = String::new();
        for piece in template {
            match *piece {
                InlineAsmTemplatePiece::String(ref s) => {
                    if s.contains('$') {
                        for c in s.chars() {
                            if c == '$' {
                                template_str.push_str("$$");
                            } else {
                                template_str.push(c);
                            }
                        }
                    } else {
                        template_str.push_str(s)
                    }
                }
                InlineAsmTemplatePiece::Placeholder {
                    operand_idx, span, ..
                } => {
                    match operands[operand_idx] {
                        InlineAsmOperandRef::In { .. }
                        | InlineAsmOperandRef::Out { .. }
                        | InlineAsmOperandRef::InOut { .. } => {
                            template_str.push_str(&format!("${{{}}}", op_idx[&operand_idx]));
                        }
                        InlineAsmOperandRef::Const { ref string } => {
                            // Const operands get injected directly into the template
                            template_str.push_str(string);
                        }
                        InlineAsmOperandRef::SymFn { .. }
                        | InlineAsmOperandRef::SymStatic { .. } => {
                            // Only emit the raw symbol name
                            template_str.push_str(&format!("${{{}:c}}", op_idx[&operand_idx]));
                        }
                        InlineAsmOperandRef::Label { .. } => {
                            // template_str.push_str(&format!("${{{}:l}}", constraints.len()));
                            // constraints.push("!i".to_owned());
                            // labels.push(label);

                            self.tcx
                                .sess
                                .dcx()
                                .span_fatal(span, "Operands with label refs are unsupported");
                        }
                    }
                }
            }
        }

        if !options.contains(InlineAsmOptions::NOMEM) {
            // This is actually ignored by LLVM, but it's probably best to keep
            // it just in case. LLVM instead uses the ReadOnly/ReadNone
            // attributes on the call instruction to optimize.
            constraints.push("~{memory}".to_string());
        }
        let volatile = !options.contains(InlineAsmOptions::PURE);
        let alignstack = !options.contains(InlineAsmOptions::NOSTACK);
        let output_type = match &output_types[..] {
            [] => self.type_void(),
            [ty] => ty,
            tys => self.type_struct(tys, false),
        };
        let dialect = llvm::AsmDialect::Att;
        let result = inline_asm_call(
            self,
            &template_str,
            &constraints.join(","),
            &inputs,
            output_type,
            volatile,
            alignstack,
            dialect,
            line_spans,
        )
        .unwrap_or_else(|| span_bug!(line_spans[0], "LLVM asm constraint validation failed"));

        if options.contains(InlineAsmOptions::PURE) {
            if options.contains(InlineAsmOptions::NOMEM) {
                llvm::Attribute::ReadNone.apply_callsite(llvm::AttributePlace::Function, result);
            } else if options.contains(InlineAsmOptions::READONLY) {
                llvm::Attribute::ReadOnly.apply_callsite(llvm::AttributePlace::Function, result);
            }
        }

        // Write results to outputs
        for (idx, op) in operands.iter().enumerate() {
            if let InlineAsmOperandRef::Out {
                place: Some(place), ..
            }
            | InlineAsmOperandRef::InOut {
                out_place: Some(place),
                ..
            } = *op
            {
                let value = if output_types.len() == 1 {
                    result
                } else {
                    self.extract_value(result, op_idx[&idx] as u64)
                };
                OperandValue::Immediate(value).store(self, place);
            }
        }
    }
}

impl<'tcx> AsmCodegenMethods<'tcx> for CodegenCx<'_, 'tcx> {
    fn codegen_global_asm(
        &mut self,
        template: &[InlineAsmTemplatePiece],
        operands: &[GlobalAsmOperandRef],
        _options: InlineAsmOptions,
        _line_spans: &[Span],
    ) {
        // Build the template string
        let mut template_str = String::new();
        for piece in template {
            match *piece {
                InlineAsmTemplatePiece::String(ref s) => template_str.push_str(s),
                InlineAsmTemplatePiece::Placeholder {
                    operand_idx,
                    modifier: _,
                    span: _,
                } => {
                    match operands[operand_idx] {
                        GlobalAsmOperandRef::Const { ref string } => {
                            // Const operands get injected directly into the
                            // template. Note that we don't need to escape $
                            // here unlike normal inline assembly.
                            template_str.push_str(string);
                        }
                        GlobalAsmOperandRef::SymFn { .. } => todo!(),
                        GlobalAsmOperandRef::SymStatic { .. } => todo!(),
                    }
                }
            }
        }

        unsafe {
            llvm::LLVMRustAppendModuleInlineAsm(
                self.llmod,
                template_str.as_c_char_ptr(),
                template_str.len(),
            );
        }
    }

    fn mangled_name(&self, _instance: Instance<'tcx>) -> String {
        todo!()
    }
}

fn reg_to_llvm(reg: InlineAsmRegOrRegClass) -> String {
    match reg {
        InlineAsmRegOrRegClass::Reg(reg) => {
            format!("{{{}}}", reg.name())
        }
        InlineAsmRegOrRegClass::RegClass(reg) => match reg {
            InlineAsmRegClass::Nvptx(NvptxInlineAsmRegClass::reg16) => "h",
            InlineAsmRegClass::Nvptx(NvptxInlineAsmRegClass::reg32) => "r",
            InlineAsmRegClass::Nvptx(NvptxInlineAsmRegClass::reg64) => "l",
            _ => unreachable!(),
        }
        .to_string(),
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn inline_asm_call<'ll>(
    bx: &mut Builder<'_, 'll, '_>,
    asm: &str,
    cons: &str,
    inputs: &[&'ll Value],
    output: &'ll llvm::Type,
    volatile: bool,
    alignstack: bool,
    dia: llvm::AsmDialect,
    line_spans: &[Span],
) -> Option<&'ll Value> {
    let volatile = if volatile { llvm::True } else { llvm::False };
    let alignstack = if alignstack { llvm::True } else { llvm::False };

    let argtys = inputs.iter().map(|v| bx.cx.val_ty(*v)).collect::<Vec<_>>();

    let fty = bx.cx.type_func(&argtys[..], output);
    unsafe {
        // Ask LLVM to verify that the constraints are well-formed.
        let constraints_ok = llvm::LLVMRustInlineAsmVerify(fty, cons.as_ptr().cast(), cons.len());
        if constraints_ok {
            let v = llvm::LLVMRustInlineAsm(
                fty,
                asm.as_ptr().cast(),
                asm.len(),
                cons.as_ptr().cast(),
                cons.len(),
                volatile,
                alignstack,
                dia,
            );
            let call = bx.call(fty, None, None, v, inputs, None, None);

            // Store mark in a metadata node so we can map LLVM errors
            // back to source locations.  See #17552.
            let key = "srcloc";
            let kind = llvm::LLVMGetMDKindIDInContext(
                bx.llcx,
                key.as_ptr() as *const c_char,
                key.len() as c_uint,
            );

            let mut srcloc = vec![];
            srcloc.extend(
                line_spans
                    .iter()
                    .map(|span| bx.const_i32(span.lo().to_u32() as i32)),
            );
            let md = llvm::LLVMMDNodeInContext(bx.llcx, srcloc.as_ptr(), srcloc.len() as u32);
            llvm::LLVMSetMetadata(call, kind, md);

            Some(call)
        } else {
            // LLVM has detected an issue with our constraints, bail out
            None
        }
    }
}
