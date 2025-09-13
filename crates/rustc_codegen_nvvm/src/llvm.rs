//! LLVM FFI functions.
//!
//! The reason we don't use rustc's llvm FFI is because rustc uses llvm 13 (at the time of writing).
//! While NVVM expects llvm 7 bitcode/suppported things. And we don't use llvm-sys because this allows us
//! to only include what we need, as well as use safe references instead of pointers.
//!
//! Most of this code was taken from rustc_codegen_llvm with many things removed.

#![allow(
    non_camel_case_types,
    non_snake_case,
    non_upper_case_globals,
    clippy::enum_variant_names
)]
// we have a lot of functions we linked to from cg_llvm that we don't use
// but likely will use in the future, so we ignore any unused functions
// in case we need them in the future for things like debug info or LTO.
#![allow(dead_code)]
use libc::{c_char, c_uint, c_void, size_t};
use libc::{c_int, c_ulonglong};
use rustc_codegen_ssa::common::AtomicRmwBinOp;
use std::ffi::{CStr, CString};
use std::fmt;
use std::hash::{Hash, Hasher};
use std::marker::PhantomData;
use std::ptr::{self};

use crate::builder::unnamed;
pub use debuginfo::*;

impl PartialEq for Value {
    fn eq(&self, other: &Self) -> bool {
        ptr::eq(self, other)
    }
}

impl Eq for Value {}

impl Hash for Value {
    fn hash<H: Hasher>(&self, hasher: &mut H) {
        (self as *const Self).hash(hasher);
    }
}

impl fmt::Debug for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        unsafe {
            // ideally we'd print the type but the llvm 7 C api doesnt have a way to do this :(
            f.write_str("(")?;
            let ptr = LLVMPrintValueToString(self);
            let cstr = CString::from_raw(ptr);
            let string = cstr.to_string_lossy();
            f.write_str(&string)?;
            f.write_str(")")
        }
    }
}

impl LLVMRustResult {
    pub fn into_result(self) -> Result<(), ()> {
        match self {
            LLVMRustResult::Success => Ok(()),
            LLVMRustResult::Failure => Err(()),
        }
    }
}

#[derive(Copy, Clone, PartialEq)]
#[repr(C)]
pub enum CodeGenOptSize {
    CodeGenOptSizeNone = 0,
    CodeGenOptSizeDefault = 1,
    CodeGenOptSizeAggressive = 2,
}

pub use self::CodeGenOptSize::*;

#[derive(Copy, Clone)]
pub enum AttributePlace {
    ReturnValue,
    Argument(u32),
    Function,
}

impl AttributePlace {
    pub fn as_uint(self) -> c_uint {
        match self {
            AttributePlace::ReturnValue => 0,
            AttributePlace::Argument(i) => 1 + i,
            AttributePlace::Function => !0,
        }
    }
}

impl Attribute {
    pub fn apply_llfn(&self, idx: AttributePlace, llfn: &Value) {
        unsafe { LLVMRustAddFunctionAttribute(llfn, idx.as_uint(), *self) }
    }

    pub fn apply_callsite(&self, idx: AttributePlace, callsite: &Value) {
        unsafe { LLVMRustAddCallSiteAttribute(callsite, idx.as_uint(), *self) }
    }

    pub fn unapply_llfn(&self, idx: AttributePlace, llfn: &Value) {
        unsafe { LLVMRustRemoveFunctionAttributes(llfn, idx.as_uint(), *self) }
    }
}

/// Safe wrapper around `LLVMGetParam`, because segfaults are no fun.
pub(crate) fn get_param(llfn: &Value, index: c_uint) -> &Value {
    unsafe {
        assert!(
            index < LLVMCountParams(llfn),
            "out of bounds argument access: {} out of {} arguments",
            index,
            LLVMCountParams(llfn)
        );
        LLVMGetParam(llfn, index)
    }
}

/// Safe wrapper around `LLVMGetParams`.
pub(crate) fn get_params(llfn: &Value) -> Vec<&Value> {
    unsafe {
        let count = LLVMCountParams(llfn) as usize;
        let mut params = Vec::with_capacity(count);
        LLVMGetParams(llfn, params.as_mut_ptr());
        params.set_len(count);
        params
    }
}

/// Safe wrapper for `LLVMGetValueName2` into a byte slice
pub(crate) fn get_value_name(value: &Value) -> &[u8] {
    unsafe {
        let mut len = 0;
        let data = LLVMGetValueName2(value, &mut len);
        std::slice::from_raw_parts(data.cast(), len)
    }
}

/// Safe wrapper for `LLVMSetValueName2` from a byte slice
pub(crate) fn set_value_name(value: &Value, name: &[u8]) {
    unsafe {
        let data = name.as_ptr().cast();
        LLVMSetValueName2(value, data, name.len());
    }
}

pub fn last_error() -> Option<String> {
    unsafe {
        let cstr = LLVMRustGetLastError();
        if cstr.is_null() {
            None
        } else {
            let err = CStr::from_ptr(cstr).to_bytes();
            let err = String::from_utf8_lossy(err).to_string();
            libc::free(cstr as *mut _);
            Some(err)
        }
    }
}

pub(crate) fn SetUnnamedAddress(global: &'_ Value, unnamed: UnnamedAddr) {
    unsafe {
        LLVMSetUnnamedAddress(global, unnamed);
    }
}

pub(crate) type Bool = c_uint;

pub const True: Bool = 1;
pub const False: Bool = 0;

#[derive(Copy, Clone, PartialEq)]
#[repr(C)]
#[allow(dead_code)] // Variants constructed by C++.
pub(crate) enum LLVMRustResult {
    Success,
    Failure,
}

/// LLVMRustLinkage
#[derive(Copy, Clone, PartialEq)]
#[repr(C)]
pub(crate) enum Linkage {
    ExternalLinkage = 0,
    AvailableExternallyLinkage = 1,
    LinkOnceAnyLinkage = 2,
    LinkOnceODRLinkage = 3,
    WeakAnyLinkage = 4,
    WeakODRLinkage = 5,
    AppendingLinkage = 6,
    InternalLinkage = 7,
    PrivateLinkage = 8,
    ExternalWeakLinkage = 9,
    CommonLinkage = 10,
}

// LLVMRustVisibility
#[repr(C)]
#[derive(Copy, Clone, PartialEq)]
pub(crate) enum Visibility {
    Default = 0,
    Hidden = 1,
    Protected = 2,
}

/// LLVMUnnamedAddr
#[repr(C)]
pub(crate) enum UnnamedAddr {
    No,
    Local,
    Global,
}

/// LLVMDLLStorageClass
#[derive(Copy, Clone)]
#[repr(C)]
pub(crate) enum DLLStorageClass {
    #[allow(dead_code)]
    Default = 0,
    #[allow(dead_code)]
    DllExport = 2, // Function to be accessible from DLL.
}

/// Matches LLVMRustAttribute in LLVMWrapper.h
/// Semantically a subset of the C++ enum llvm::Attribute::AttrKind,
/// though it is not ABI compatible (since it's a C++ enum)
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub(crate) enum Attribute {
    AlwaysInline = 0,
    Cold = 2,
    InlineHint = 3,
    MinSize = 4,
    NoAlias = 6,
    NoCapture = 7,
    NoInline = 8,
    NonNull = 9,
    NoReturn = 11,
    NoUnwind = 12,
    OptimizeForSize = 13,
    OptimizeNone = 14,
    ReadOnly = 15,
    SExt = 16,
    StructRet = 17,
    ZExt = 19,
    InReg = 20,
    ReadNone = 24,
}

/// LLVMIntPredicate
#[derive(Copy, Clone)]
#[repr(C)]
pub(crate) enum IntPredicate {
    IntEQ = 32,
    IntNE = 33,
    IntUGT = 34,
    IntUGE = 35,
    IntULT = 36,
    IntULE = 37,
    IntSGT = 38,
    IntSGE = 39,
    IntSLT = 40,
    IntSLE = 41,
}

impl IntPredicate {
    pub fn from_generic(intpre: rustc_codegen_ssa::common::IntPredicate) -> Self {
        match intpre {
            rustc_codegen_ssa::common::IntPredicate::IntEQ => IntPredicate::IntEQ,
            rustc_codegen_ssa::common::IntPredicate::IntNE => IntPredicate::IntNE,
            rustc_codegen_ssa::common::IntPredicate::IntUGT => IntPredicate::IntUGT,
            rustc_codegen_ssa::common::IntPredicate::IntUGE => IntPredicate::IntUGE,
            rustc_codegen_ssa::common::IntPredicate::IntULT => IntPredicate::IntULT,
            rustc_codegen_ssa::common::IntPredicate::IntULE => IntPredicate::IntULE,
            rustc_codegen_ssa::common::IntPredicate::IntSGT => IntPredicate::IntSGT,
            rustc_codegen_ssa::common::IntPredicate::IntSGE => IntPredicate::IntSGE,
            rustc_codegen_ssa::common::IntPredicate::IntSLT => IntPredicate::IntSLT,
            rustc_codegen_ssa::common::IntPredicate::IntSLE => IntPredicate::IntSLE,
        }
    }
}

/// LLVMTypeKind
#[allow(dead_code)]
#[derive(Copy, Clone, PartialEq, Debug)]
#[repr(C)]
pub(crate) enum TypeKind {
    Void = 0,
    Half = 1,
    Float = 2,
    Double = 3,
    Label = 7,
    Integer = 8,
    Function = 9,
    Struct = 10,
    Array = 11,
    Pointer = 12,
    Vector = 13,
    Metadata = 14,
    Token = 16,
    ScalableVector = 17,
    BFloat = 18,
}

impl TypeKind {
    pub fn to_generic(self) -> rustc_codegen_ssa::common::TypeKind {
        match self {
            TypeKind::Void => rustc_codegen_ssa::common::TypeKind::Void,
            TypeKind::Half => rustc_codegen_ssa::common::TypeKind::Half,
            TypeKind::Float => rustc_codegen_ssa::common::TypeKind::Float,
            TypeKind::Double => rustc_codegen_ssa::common::TypeKind::Double,
            TypeKind::Label => rustc_codegen_ssa::common::TypeKind::Label,
            TypeKind::Integer => rustc_codegen_ssa::common::TypeKind::Integer,
            TypeKind::Function => rustc_codegen_ssa::common::TypeKind::Function,
            TypeKind::Struct => rustc_codegen_ssa::common::TypeKind::Struct,
            TypeKind::Array => rustc_codegen_ssa::common::TypeKind::Array,
            TypeKind::Pointer => rustc_codegen_ssa::common::TypeKind::Pointer,
            TypeKind::Vector => rustc_codegen_ssa::common::TypeKind::Vector,
            TypeKind::Metadata => rustc_codegen_ssa::common::TypeKind::Metadata,
            TypeKind::Token => rustc_codegen_ssa::common::TypeKind::Token,
            TypeKind::ScalableVector => rustc_codegen_ssa::common::TypeKind::ScalableVector,
            TypeKind::BFloat => rustc_codegen_ssa::common::TypeKind::BFloat,
        }
    }
}

/// LLVMMetadataType
#[derive(Copy, Clone)]
#[repr(C)]
pub(crate) enum MetadataType {
    MD_range = 4,
    MD_invariant_load = 6,
    MD_nontemporal = 9,
    MD_nonnull = 11,
}

/// LLVMRustAsmDialect
#[derive(Copy, Clone)]
#[repr(C)]
pub enum AsmDialect {
    Other,
    Att,
    Intel,
}

// impl AsmDialect {
//     pub fn from_generic(asm: rustc_ast::LlvmAsmDialect) -> Self {
//         match asm {
//             rustc_ast::LlvmAsmDialect::Att => AsmDialect::Att,
//             rustc_ast::LlvmAsmDialect::Intel => AsmDialect::Intel,
//         }
//     }
// }

/// LLVMRustDiagnosticKind
#[derive(Copy, Clone)]
#[repr(C)]
#[allow(dead_code)] // Variants constructed by C++.
pub(crate) enum DiagnosticKind {
    Other,
    InlineAsm,
    StackSize,
    DebugMetadataVersion,
    SampleProfile,
    OptimizationRemark,
    OptimizationRemarkMissed,
    OptimizationRemarkAnalysis,
    OptimizationRemarkAnalysisFPCommute,
    OptimizationRemarkAnalysisAliasing,
    OptimizationRemarkOther,
    OptimizationFailure,
    PGOProfile,
    Linker,
    Unsupported,
}

/// LLVMRustDiagnosticLevel
#[derive(Copy, Clone)]
#[repr(C)]
#[allow(dead_code)] // Variants constructed by C++.
pub(crate) enum DiagnosticLevel {
    Error,
    Warning,
    Note,
    Remark,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum LLVMVerifierFailureAction {
    /// Print to stderr and abort the process.
    LLVMAbortProcessAction = 0,
    /// Print to stderr and return 1.
    LLVMPrintMessageAction = 1,
    /// Return 1 and print nothing.
    LLVMReturnStatusAction = 2,
}

/// LLVMRustPassKind
#[derive(Copy, Clone, PartialEq, Debug)]
#[repr(C)]
#[allow(dead_code)] // Variants constructed by C++.
pub(crate) enum PassKind {
    Other,
    Function,
    Module,
}

// LLVMRustThinLTOData
unsafe extern "C" {
    pub(crate) type ThinLTOData;
}

unsafe impl Send for ThinLTOData {}

// LLVMRustThinLTOBuffer
unsafe extern "C" {
    pub(crate) type ThinLTOBuffer;
}

unsafe impl Send for ThinLTOBuffer {}

/// LLVMRustThinLTOModule
#[repr(C)]
pub(crate) struct ThinLTOModule {
    pub identifier: *const c_char,
    pub data: *const u8,
    pub len: usize,
}

unsafe extern "C" {
    type Opaque;
}
#[repr(C)]
struct InvariantOpaque<'a> {
    _marker: PhantomData<&'a mut &'a ()>,
    _opaque: Opaque,
}

// Opaque pointer types
unsafe extern "C" {
    pub(crate) type Module;
}
unsafe extern "C" {
    pub type Context;
}

unsafe impl Send for Context {}

unsafe extern "C" {
    pub(crate) type Type;
}
unsafe extern "C" {
    pub(crate) type Value;
}
unsafe extern "C" {
    pub(crate) type ConstantInt;
}
unsafe extern "C" {
    pub type Metadata;
}
unsafe extern "C" {
    pub(crate) type BasicBlock;
}
#[repr(C)]
pub(crate) struct Builder<'a> {
    _inv: InvariantOpaque<'a>,
}
#[repr(C)]
pub(crate) struct OperandBundleDef<'a>(InvariantOpaque<'a>);

unsafe extern "C" {
    pub(crate) type ModuleBuffer;
}

unsafe impl Send for ModuleBuffer {}

#[repr(C)]
pub struct PassManager<'a>(InvariantOpaque<'a>);
unsafe extern "C" {
    pub type PassManagerBuilder;
}
unsafe extern "C" {
    pub type Pass;
}
unsafe extern "C" {
    pub type TargetMachine;
}
unsafe extern "C" {
    pub(crate) type MemoryBuffer;
}

/// LLVMRustChecksumKind
#[derive(Copy, Clone)]
#[repr(C)]
pub enum ChecksumKind {
    None,
    MD5,
    SHA1,
}

pub mod debuginfo {
    use super::{InvariantOpaque, Metadata};
    use bitflags::bitflags;

    #[repr(C)]
    pub(crate) struct DIBuilder<'a>(InvariantOpaque<'a>);

    pub type DIDescriptor = Metadata;
    pub type DIScope = DIDescriptor;
    pub type DILocation = DIDescriptor;
    pub type DIFile = DIScope;
    pub type DILexicalBlock = DIScope;
    pub type DISubprogram = DIScope;
    pub type DINameSpace = DIScope;
    pub type DIType = DIDescriptor;
    pub type DIBasicType = DIType;
    pub type DIDerivedType = DIType;
    pub type DICompositeType = DIDerivedType;
    pub type DIVariable = DIDescriptor;
    pub type DIGlobalVariable = DIDescriptor;
    pub type DIArray = DIDescriptor;
    pub type DISubrange = DIDescriptor;
    pub type DIEnumerator = DIDescriptor;
    pub type DITemplateTypeParameter = DIDescriptor;

    bitflags! {
        /// Must match the layout of `LLVMDIFlags` in the LLVM-C API.
        ///
        /// Each value declared here must also be covered by the static
        /// assertions in `RustWrapper.cpp` used by `fromRust(LLVMDIFlags)`.
        #[repr(C)]
        #[derive(Clone, Copy, Default)]
        pub struct DIFlags: u32 {
            const FlagZero                = 0;
            const FlagPrivate             = 1;
            const FlagProtected           = 2;
            const FlagPublic              = 3;
            const FlagFwdDecl             = (1 << 2);
            const FlagAppleBlock          = (1 << 3);
            const FlagBlockByrefStruct    = (1 << 4);
            const FlagVirtual             = (1 << 5);
            const FlagArtificial          = (1 << 6);
            const FlagExplicit            = (1 << 7);
            const FlagPrototyped          = (1 << 8);
            const FlagObjcClassComplete   = (1 << 9);
            const FlagObjectPointer       = (1 << 10);
            const FlagVector              = (1 << 11);
            const FlagStaticMember        = (1 << 12);
            const FlagLValueReference     = (1 << 13);
            const FlagRValueReference     = (1 << 14);
            const FlagExternalTypeRef     = (1 << 15);
            const FlagIntroducedVirtual   = (1 << 18);
            const FlagBitField            = (1 << 19);
            const FlagNoReturn            = (1 << 20);
            const FlagMainSubprogram      = (1 << 21);
        }
    }

    /// LLVMRustDebugEmissionKind
    #[derive(Copy, Clone)]
    #[repr(C)]
    pub enum DebugEmissionKind {
        NoDebug,
        FullDebug,
        LineTablesOnly,
    }

    impl DebugEmissionKind {
        pub(crate) fn from_generic(kind: rustc_session::config::DebugInfo) -> Self {
            // We should be setting LLVM's emission kind to `LineTablesOnly` if
            // we are compiling with "limited" debuginfo. However, some of the
            // existing tools relied on slightly more debuginfo being generated than
            // would be the case with `LineTablesOnly`, and we did not want to break
            // these tools in a "drive-by fix", without a good idea or plan about
            // what limited debuginfo should exactly look like. So for now we are
            // instead adding a new debuginfo option "line-tables-only" so as to
            // not break anything and to allow users to have 'limited' debug info.
            //
            // See https://github.com/rust-lang/rust/issues/60020 for details.
            use rustc_session::config::DebugInfo;
            match kind {
                // NVVM: Llvm 7 is missing LineDirectivesOnly, so don't emit anything.
                DebugInfo::None | DebugInfo::LineDirectivesOnly => DebugEmissionKind::NoDebug,
                DebugInfo::LineTablesOnly => DebugEmissionKind::LineTablesOnly,
                DebugInfo::Limited | DebugInfo::Full => DebugEmissionKind::FullDebug,
            }
        }
    }
}

// These functions are kind of a hack for the future. They wrap LLVM 7 rust shim functions
// and turn them into the API that the llvm 12 shim has. This way, if nvidia ever updates their
// dinosaur llvm version, switching for us should be extremely easy. `Name` is assumed to be
// a utf8 string
pub(crate) unsafe fn LLVMRustGetOrInsertFunction<'a>(
    M: &'a Module,
    Name: *const c_char,
    NameLen: usize,
    FunctionTy: &'a Type,
) -> &'a Value {
    unsafe {
        let str = std::str::from_utf8_unchecked(std::slice::from_raw_parts(Name.cast(), NameLen));
        let cstring = CString::new(str).expect("str with nul");
        __LLVMRustGetOrInsertFunction(M, cstring.as_ptr(), FunctionTy)
    }
}

pub(crate) unsafe fn LLVMRustBuildCall<'a>(
    B: &Builder<'a>,
    Fn: &'a Value,
    Args: *const &'a Value,
    NumArgs: c_uint,
    Bundle: Option<&OperandBundleDef<'a>>,
) -> &'a Value {
    unsafe { __LLVMRustBuildCall(B, Fn, Args, NumArgs, Bundle, unnamed()) }
}

/// LLVMRustCodeGenOptLevel
#[derive(Copy, Clone, PartialEq)]
#[repr(C)]
pub enum CodeGenOptLevel {
    Other,
    None,
    Less,
    Default,
    Aggressive,
}

/// LLVMRelocMode
#[derive(Copy, Clone, PartialEq)]
#[allow(clippy::upper_case_acronyms)]
#[repr(C)]
pub enum RelocMode {
    Default,
    Static,
    PIC,
    DynamicNoPic,
    ROPI,
    RWPI,
    ROPI_RWPI,
}

/// LLVMRustCodeModel
#[derive(Copy, Clone)]
#[repr(C)]
pub enum CodeModel {
    Other,
    Small,
    Kernel,
    Medium,
    Large,
    None,
}

unsafe extern "C" {
    #[link_name = "LLVMRustBuildCall"]
    pub(crate) fn __LLVMRustBuildCall<'a>(
        B: &Builder<'a>,
        Fn: &'a Value,
        Args: *const &'a Value,
        NumArgs: c_uint,
        Bundle: Option<&OperandBundleDef<'a>>,
        Name: *const c_char,
    ) -> &'a Value;

    // see comment on function before this extern block
    #[link_name = "LLVMRustGetOrInsertFunction"]
    fn __LLVMRustGetOrInsertFunction<'a>(
        M: &'a Module,
        Name: *const c_char,
        FunctionTy: &'a Type,
    ) -> &'a Value;

    // dont trace these functions or cargo will error, see init.rs
    pub(crate) fn LLVMStartMultithreaded() -> Bool;
    pub(crate) fn LLVMInitializeNVPTXTargetInfo();
    pub(crate) fn LLVMInitializeNVPTXTarget();
    pub(crate) fn LLVMInitializeNVPTXTargetMC();
    pub(crate) fn LLVMInitializeNVPTXAsmPrinter();
    pub(crate) fn LLVMInitializePasses();
    pub(crate) fn LLVMRustSetLLVMOptions(Argc: c_int, Argv: *const *const c_char);
}

// use rustc_codegen_nvvm_macros::trace_ffi_calls;
// #[trace_ffi_calls]
unsafe extern "C" {
    pub(crate) fn LLVMGetPointerAddressSpace(PointerTy: &Type) -> c_uint;
    pub(crate) fn LLVMBuildAddrSpaceCast<'a>(
        arg1: &Builder<'a>,
        Val: &'a Value,
        DestTy: &'a Type,
        Name: *const c_char,
    ) -> &'a Value;
    pub(crate) fn LLVMRustGetOrInsertGlobal<'a>(
        M: &'a Module,
        Name: *const c_char,
        NameLen: usize,
        T: &'a Type,
        AddressSpace: c_uint,
    ) -> &'a Value;
    pub(crate) fn LLVMAddGlobalDCEPass(PM: &mut PassManager);
    pub(crate) fn LLVMGetNamedMetadataOperands(M: &Module, name: *const c_char, Dest: *mut &Value);
    pub(crate) fn LLVMGetNamedMetadataNumOperands(M: &Module, name: *const c_char) -> c_uint;
    pub(crate) fn LLVMGetMDNodeOperands(V: &Value, Dest: *mut &Value);
    pub(crate) fn LLVMGetMDNodeNumOperands(V: &Value) -> c_uint;
    pub(crate) fn LLVMGetFirstFunction(M: &Module) -> Option<&Value>;
    pub(crate) fn LLVMGetNextFunction(Fn: &Value) -> Option<&Value>;
    pub(crate) fn LLVMAddGlobalInAddressSpace<'a>(
        M: &'a Module,
        Ty: &'a Type,
        Name: *const c_char,
        AddressSpace: c_uint,
    ) -> &'a Value;
    pub(crate) fn LLVMGetOperand(Val: &Value, Index: c_uint) -> &Value;
    pub(crate) fn LLVMIsABitCastInst(Val: &Value) -> Option<&Value>;
    pub(crate) fn LLVMIsASelectInst(Val: &Value) -> Option<&Value>;
    pub(crate) fn LLVMRustGetFunctionType(V: &Value) -> &Type;
    pub(crate) fn LLVMLinkModules2(Dest: &Module, Src: &Module) -> Bool;
    pub(crate) fn LLVMParseIRInContext<'ll, 'a, 'b>(
        ContextRef: &'ll Context,
        MemBuf: &'a MemoryBuffer,
        OutM: *mut &'b Module,
        OutMessage: *mut *mut c_char,
    ) -> Bool;
    pub(crate) fn LLVMCreateMemoryBufferWithMemoryRange<'a>(
        InputData: *const c_char,
        InputDataLength: usize,
        BufferName: *const c_char,
        RequiresNullTerminator: Bool,
    ) -> &'a mut MemoryBuffer;
    pub(crate) fn LLVMDisposeMemoryBuffer<'a>(MemBuf: &'a mut MemoryBuffer);

    pub(crate) fn LLVMGetCurrentDebugLocation<'a>(Builder: &Builder<'a>) -> Option<&'a Value>;
    pub(crate) fn LLVMSetCurrentDebugLocation<'a>(Builder: &Builder<'a>, L: Option<&'a Value>);

    pub(crate) fn LLVMGetModuleContext(M: &Module) -> &Context;
    pub(crate) fn LLVMGetMDKindIDInContext(
        C: &Context,
        Name: *const c_char,
        SLen: c_uint,
    ) -> c_uint;

    pub(crate) fn LLVMRustDebugMetadataVersion() -> u32;
    pub(crate) fn LLVMRustVersionMajor() -> u32;
    pub(crate) fn LLVMRustVersionMinor() -> u32;

    pub(crate) fn LLVMRustAddModuleFlag(M: &Module, name: *const c_char, value: u32);

    pub(crate) fn LLVMRustMetadataAsValue<'a>(C: &'a Context, MD: &'a Metadata) -> &'a Value;

    pub(crate) fn LLVMRustDIBuilderCreate<'a>(M: &'a Module) -> &'a mut DIBuilder<'a>;

    pub(crate) fn LLVMRustDIBuilderDispose<'a>(Builder: &'a mut DIBuilder<'a>);

    pub(crate) fn LLVMRustDIBuilderFinalize<'a>(Builder: &DIBuilder<'a>);

    pub fn LLVMRustDIBuilderCreateCompileUnit<'a>(
        Builder: &DIBuilder<'a>,
        Lang: c_uint,
        File: &'a DIFile,
        Producer: *const c_char,
        ProducerLen: size_t,
        isOptimized: bool,
        Flags: *const c_char,
        RuntimeVer: c_uint,
        SplitName: *const c_char,
        SplitNameLen: size_t,
        kind: DebugEmissionKind,
        DWOId: u64,
        SplitDebugInlining: bool,
    ) -> &'a DIDescriptor;

    pub(crate) fn LLVMRustDIBuilderCreateFile<'a>(
        Builder: &DIBuilder<'a>,
        Filename: *const c_char,
        FileNameLen: size_t,
        Directory: *const c_char,
        DirectoryLen: size_t,
        CSKind: ChecksumKind,
        Checksum: *const c_char,
        ChecksomLen: size_t,
        Source: *const c_char,
        SourceLen: size_t,
    ) -> &'a DIFile;

    pub(crate) fn LLVMRustDIBuilderCreateSubroutineType<'a>(
        Builder: &DIBuilder<'a>,
        ParameterTypes: &'a DIArray,
    ) -> &'a DICompositeType;

    pub(crate) fn LLVMRustDIBuilderCreateFunction<'a>(
        Builder: &DIBuilder<'a>,
        Scope: &'a DIDescriptor,
        Name: *const c_char,
        LinkageName: *const c_char,
        File: &'a DIFile,
        LineNo: c_uint,
        Ty: &'a DIType,
        isLocalToUnit: bool,
        isDefinition: bool,
        ScopeLine: c_uint,
        Flags: DIFlags,
        isOptimized: bool,
        MaybeFn: Option<&'a Value>,
        TParam: &'a DIArray,
        Decl: Option<&'a DIDescriptor>,
    ) -> &'a DISubprogram;

    pub(crate) fn LLVMRustDIBuilderCreateBasicType<'a>(
        Builder: &DIBuilder<'a>,
        Name: *const c_char,
        NameLen: size_t,
        SizeInBits: u64,
        Encoding: c_uint,
    ) -> &'a DIBasicType;

    pub fn LLVMRustDIBuilderCreateTypedef<'a>(
        Builder: &DIBuilder<'a>,
        Type: &'a DIBasicType,
        Name: *const c_char,
        NameLen: size_t,
        File: &'a DIFile,
        LineNo: c_uint,
        Scope: Option<&'a DIScope>,
    ) -> &'a DIDerivedType;

    pub(crate) fn LLVMRustDIBuilderCreatePointerType<'a>(
        Builder: &DIBuilder<'a>,
        PointeeTy: &'a DIType,
        SizeInBits: u64,
        AlignInBits: u32,
        Name: *const c_char,
    ) -> &'a DIDerivedType;

    pub(crate) fn LLVMRustDIBuilderCreateStructType<'a>(
        Builder: &DIBuilder<'a>,
        Scope: Option<&'a DIDescriptor>,
        Name: *const c_char,
        NameLen: size_t,
        File: &'a DIFile,
        LineNumber: c_uint,
        SizeInBits: u64,
        AlignInBits: u32,
        Flags: DIFlags,
        DerivedFrom: Option<&'a DIType>,
        Elements: &'a DIArray,
        RunTimeLang: c_uint,
        VTableHolder: Option<&'a DIType>,
        UniqueId: *const c_char,
        UniqueIdLen: size_t,
    ) -> &'a DICompositeType;

    pub(crate) fn LLVMRustDIBuilderCreateMemberType<'a>(
        Builder: &DIBuilder<'a>,
        Scope: &'a DIDescriptor,
        Name: *const c_char,
        NameLen: size_t,
        File: &'a DIFile,
        LineNo: c_uint,
        SizeInBits: u64,
        AlignInBits: u32,
        OffsetInBits: u64,
        Flags: DIFlags,
        Ty: &'a DIType,
    ) -> &'a DIDerivedType;

    pub(crate) fn LLVMRustDIBuilderCreateVariantMemberType<'a>(
        Builder: &DIBuilder<'a>,
        Scope: &'a DIScope,
        Name: *const c_char,
        NameLen: size_t,
        File: &'a DIFile,
        LineNumber: c_uint,
        SizeInBits: u64,
        AlignInBits: u32,
        OffsetInBits: u64,
        Discriminant: Option<&'a Value>,
        Flags: DIFlags,
        Ty: &'a DIType,
    ) -> &'a DIType;

    pub(crate) fn LLVMRustDIBuilderCreateLexicalBlock<'a>(
        Builder: &DIBuilder<'a>,
        Scope: &'a DIScope,
        File: &'a DIFile,
        Line: c_uint,
        Col: c_uint,
    ) -> &'a DILexicalBlock;

    pub(crate) fn LLVMRustDIBuilderCreateLexicalBlockFile<'a>(
        Builder: &DIBuilder<'a>,
        Scope: &'a DIScope,
        File: &'a DIFile,
    ) -> &'a DILexicalBlock;

    pub(crate) fn LLVMRustDIBuilderCreateStaticVariable<'a>(
        Builder: &DIBuilder<'a>,
        Context: Option<&'a DIScope>,
        Name: *const c_char,
        NameLen: size_t,
        LinkageName: *const c_char,
        LinkageNameLen: size_t,
        File: &'a DIFile,
        LineNo: c_uint,
        Ty: &'a DIType,
        isLocalToUnit: bool,
        Val: &'a Value,
        Decl: Option<&'a DIDescriptor>,
        AlignInBits: u32,
    ) -> &'a DIGlobalVariable;

    pub(crate) fn LLVMRustDIBuilderCreateVariable<'a>(
        Builder: &DIBuilder<'a>,
        Tag: c_uint,
        Scope: &'a DIDescriptor,
        Name: *const c_char,
        File: &'a DIFile,
        LineNo: c_uint,
        Ty: &'a DIType,
        AlwaysPreserve: bool,
        Flags: DIFlags,
        ArgNo: c_uint,
        AlignInBits: u32,
    ) -> &'a DIVariable;

    pub(crate) fn LLVMRustDIBuilderCreateArrayType<'a>(
        Builder: &DIBuilder<'a>,
        Size: u64,
        AlignInBits: u32,
        Ty: &'a DIType,
        Subscripts: &'a DIArray,
    ) -> &'a DIType;

    pub(crate) fn LLVMRustDIBuilderCreateVectorType<'a>(
        Builder: &DIBuilder<'a>,
        Size: u64,
        AlignInBits: u32,
        Ty: &'a DIType,
        Subscripts: &'a DIArray,
    ) -> &'a DIType;

    pub(crate) fn LLVMRustDIBuilderGetOrCreateSubrange<'a>(
        Builder: &DIBuilder<'a>,
        Lo: i64,
        Count: i64,
    ) -> &'a DISubrange;

    pub(crate) fn LLVMRustDIBuilderGetOrCreateArray<'a>(
        Builder: &DIBuilder<'a>,
        Ptr: *const Option<&'a DIDescriptor>,
        Count: c_uint,
    ) -> &'a DIArray;

    pub(crate) fn LLVMRustDIBuilderInsertDeclareAtEnd<'a>(
        Builder: &DIBuilder<'a>,
        Val: &'a Value,
        VarInfo: &'a DIVariable,
        AddrOps: *const i64,
        AddrOpsCount: c_uint,
        DL: &'a DILocation,
        InsertAtEnd: &'a BasicBlock,
    ) -> &'a Value;

    pub(crate) fn LLVMRustDIBuilderCreateEnumerator<'a>(
        Builder: &DIBuilder<'a>,
        Name: *const c_char,
        NameLen: size_t,
        Val: i64,
        IsUnsigned: bool,
    ) -> &'a DIEnumerator;

    pub(crate) fn LLVMRustDIBuilderCreateEnumerationType<'a>(
        Builder: &DIBuilder<'a>,
        Scope: &'a DIScope,
        Name: *const c_char,
        NameLen: size_t,
        File: &'a DIFile,
        LineNumber: c_uint,
        SizeInBits: u64,
        AlignInBits: u32,
        Elements: &'a DIArray,
        ClassType: &'a DIType,
    ) -> &'a DIType;

    pub(crate) fn LLVMRustDIBuilderCreateUnionType<'a>(
        Builder: &DIBuilder<'a>,
        Scope: Option<&'a DIScope>,
        Name: *const c_char,
        NameLen: size_t,
        File: &'a DIFile,
        LineNumber: c_uint,
        SizeInBits: u64,
        AlignInBits: u32,
        Flags: DIFlags,
        Elements: Option<&'a DIArray>,
        RunTimeLang: c_uint,
        UniqueId: *const c_char,
        UniqueIdLen: size_t,
    ) -> &'a DIType;

    pub(crate) fn LLVMRustDIBuilderCreateVariantPart<'a>(
        Builder: &DIBuilder<'a>,
        Scope: &'a DIScope,
        Name: *const c_char,
        NameLen: size_t,
        File: &'a DIFile,
        LineNo: c_uint,
        SizeInBits: u64,
        AlignInBits: u32,
        Flags: DIFlags,
        Discriminator: Option<&'a DIDerivedType>,
        Elements: &'a DIArray,
        UniqueId: *const c_char,
        UniqueIdLen: size_t,
    ) -> &'a DIDerivedType;

    pub(crate) fn LLVMSetUnnamedAddr<'a>(GlobalVar: &'a Value, UnnamedAddr: Bool);

    pub(crate) fn LLVMRustDIBuilderCreateTemplateTypeParameter<'a>(
        Builder: &DIBuilder<'a>,
        Scope: Option<&'a DIScope>,
        Name: *const c_char,
        NameLen: size_t,
        Ty: &'a DIType,
    ) -> &'a DITemplateTypeParameter;

    pub(crate) fn LLVMRustDIBuilderCreateNameSpace<'a>(
        Builder: &DIBuilder<'a>,
        Scope: Option<&'a DIScope>,
        Name: *const c_char,
        NameLen: size_t,
    ) -> &'a DINameSpace;

    pub(crate) fn LLVMRustDICompositeTypeReplaceArrays<'a>(
        Builder: &DIBuilder<'a>,
        CompositeType: &'a DIType,
        Elements: Option<&'a DIArray>,
        Params: Option<&'a DIArray>,
    );

    pub(crate) fn LLVMRustDICompositeTypeSetTypeArray<'a>(
        Builder: &DIBuilder<'a>,
        CompositeType: &'a DIType,
        TypeArray: &'a DIArray,
    );

    pub(crate) fn LLVMRustDIBuilderCreateDebugLocation<'a>(
        Line: c_uint,
        Column: c_uint,
        Scope: &'a DIScope,
        InlinedAt: Option<&'a Metadata>,
    ) -> &'a DILocation;
    pub(crate) fn LLVMRustDILocationCloneWithBaseDiscriminator<'a>(
        Location: &'a DILocation,
        BD: c_uint,
    ) -> Option<&'a DILocation>;

    pub(crate) fn LLVMRustRunFunctionPassManager(PM: &PassManager, M: &Module);
    pub(crate) fn LLVMRustAddAlwaysInlinePass(P: &PassManagerBuilder, AddLifetimes: bool);

    pub(crate) fn LLVMRustAddBuilderLibraryInfo(
        PMB: &PassManagerBuilder,
        M: &Module,
        DisableSimplifyLibCalls: bool,
    );

    pub(crate) fn LLVMRustConfigurePassManagerBuilder(
        PMB: &PassManagerBuilder,
        OptLevel: CodeGenOptLevel,
        MergeFunctions: bool,
        SLPVectorize: bool,
        LoopVectorize: bool,
        PrepareForThinLTO: bool,
        PGOGenPath: *const c_char,
        PGOUsePath: *const c_char,
    );

    pub(crate) fn LLVMRustCreateTargetMachine<'a>(
        Triple: *const c_char,
        TripleLen: size_t,
        CPU: *const c_char,
        CPULen: size_t,
        Features: *const c_char,
        FeaturesLen: size_t,
        Model: CodeModel,
        Reloc: RelocMode,
        Level: CodeGenOptLevel,
        UseSoftFP: bool,
        PositionIndependentExecutable: bool,
        FunctionSections: bool,
        DataSections: bool,
        TrapUnreachable: bool,
        Singlethread: bool,
    ) -> Option<&'static mut TargetMachine>;

    pub(crate) fn LLVMRustAddAnalysisPasses<'a>(
        T: &'a TargetMachine,
        PM: &'a PassManager,
        M: &'a Module,
    );
    pub(crate) fn LLVMRustPassKind(Pass: &Pass) -> PassKind;
    pub(crate) fn LLVMRustFindAndCreatePass(
        Pass: *const c_char,
        PassNameLen: size_t,
    ) -> Option<&'static mut Pass>;
    pub(crate) fn LLVMRustAddPass<'a>(PM: &'a PassManager, Pass: &'static mut Pass);

    /// Writes a module to the specified path. Returns 0 on success.
    pub(crate) fn LLVMWriteBitcodeToFile(M: &Module, Path: *const c_char) -> c_int;

    /// Creates a pass manager.
    pub(crate) fn LLVMCreatePassManager<'a>() -> &'a mut PassManager<'a>;

    /// Creates a function-by-function pass manager
    pub(crate) fn LLVMCreateFunctionPassManagerForModule<'a>(
        M: &'a Module,
    ) -> &'a mut PassManager<'a>;

    /// Disposes a pass manager.
    pub(crate) fn LLVMDisposePassManager<'a>(PM: &'a mut PassManager<'a>);

    /// Runs a pass manager on a module.
    pub(crate) fn LLVMRunPassManager<'a>(PM: &PassManager<'a>, M: &'a Module) -> Bool;

    pub(crate) fn LLVMTimeTraceProfilerFinish(FileName: *const c_char);

    pub(crate) fn LLVMAddAnalysisPasses<'a>(T: &'a TargetMachine, PM: &PassManager<'a>);

    pub(crate) fn LLVMPassManagerBuilderCreate() -> &'static mut PassManagerBuilder;
    pub(crate) fn LLVMPassManagerBuilderDispose(PMB: &'static mut PassManagerBuilder);
    pub(crate) fn LLVMPassManagerBuilderSetSizeLevel(PMB: &PassManagerBuilder, Value: Bool);
    pub(crate) fn LLVMPassManagerBuilderSetDisableUnrollLoops(
        PMB: &PassManagerBuilder,
        Value: Bool,
    );
    pub(crate) fn LLVMPassManagerBuilderUseInlinerWithThreshold(
        PMB: &PassManagerBuilder,
        threshold: c_uint,
    );
    pub(crate) fn LLVMPassManagerBuilderPopulateModulePassManager(
        PMB: &PassManagerBuilder,
        PM: &PassManager<'_>,
    );

    pub(crate) fn LLVMPassManagerBuilderPopulateFunctionPassManager(
        PMB: &PassManagerBuilder,
        PM: &PassManager<'_>,
    );
    pub(crate) fn LLVMPassManagerBuilderPopulateLTOPassManager(
        PMB: &PassManagerBuilder,
        PM: &PassManager<'_>,
        Internalize: Bool,
        RunInliner: Bool,
    );
    pub(crate) fn LLVMRustPassManagerBuilderPopulateThinLTOPassManager(
        PMB: &PassManagerBuilder,
        PM: &PassManager<'_>,
    );

    // functions that cg_llvm doesnt use but we do. mostly for int_replace.
    pub(crate) fn LLVMGetReturnType(FunctionTy: &Type) -> &Type;
    pub(crate) fn LLVMGetParams(Fn: &Value, Params: *mut &Value);
    pub(crate) fn LLVMGetEntryBasicBlock(Fn: &Value) -> &BasicBlock;
    pub(crate) fn LLVMGetNamedFunction(M: &Module, Name: *const c_char) -> &Value;
    pub(crate) fn LLVMRustGetFunctionReturnType(V: &Value) -> &Type;

    pub(crate) fn LLVMSetTarget(M: &Module, Triple: *const c_char);

    // Create and destroy contexts.
    pub(crate) fn LLVMRustContextCreate(shouldDiscardNames: bool) -> &'static mut Context;
    pub(crate) fn LLVMContextDispose(C: &'static mut Context);

    // Create modules.
    pub(crate) fn LLVMModuleCreateWithNameInContext(
        ModuleID: *const c_char,
        C: &Context,
    ) -> &Module;

    pub(crate) fn LLVMSetDataLayout(M: &Module, Triple: *const c_char);

    pub(crate) fn LLVMRustAppendModuleInlineAsm(M: &Module, Asm: *const c_char, AsmLen: size_t);

    /// See llvm::LLVMTypeKind::getTypeID.
    pub(crate) fn LLVMRustGetTypeKind(Ty: &Type) -> TypeKind;

    pub(crate) fn LLVMPrintTypeToString(Val: &Type) -> *mut c_char;

    // Operations on integer types
    pub(crate) fn LLVMInt1TypeInContext(C: &Context) -> &Type;
    pub(crate) fn LLVMInt8TypeInContext(C: &Context) -> &Type;
    pub(crate) fn LLVMInt16TypeInContext(C: &Context) -> &Type;
    pub(crate) fn LLVMInt32TypeInContext(C: &Context) -> &Type;
    pub(crate) fn LLVMInt64TypeInContext(C: &Context) -> &Type;
    pub(crate) fn LLVMIntTypeInContext(C: &Context, NumBits: c_uint) -> &Type;

    pub(crate) fn LLVMGetIntTypeWidth(IntegerTy: &Type) -> c_uint;

    // Operations on real types
    pub(crate) fn LLVMHalfTypeInContext(C: &Context) -> &Type;
    pub(crate) fn LLVMFloatTypeInContext(C: &Context) -> &Type;
    pub(crate) fn LLVMDoubleTypeInContext(C: &Context) -> &Type;
    pub(crate) fn LLVMFP128TypeInContext(C: &Context) -> &Type;

    // Operations on function types
    pub(crate) fn LLVMFunctionType<'a>(
        ReturnType: &'a Type,
        ParamTypes: *const &'a Type,
        ParamCount: c_uint,
        IsVarArg: Bool,
    ) -> &'a Type;
    pub(crate) fn LLVMCountParamTypes(FunctionTy: &Type) -> c_uint;
    pub(crate) fn LLVMGetParamTypes<'a>(FunctionTy: &'a Type, Dest: *mut &'a Type);

    // Operations on struct types
    pub(crate) fn LLVMStructTypeInContext<'a>(
        C: &'a Context,
        ElementTypes: *const &'a Type,
        ElementCount: c_uint,
        Packed: Bool,
    ) -> &'a Type;
    pub(crate) fn LLVMGetStructElementTypes<'a>(StructTy: &'a Type, Dest: *mut &'a Type);
    pub(crate) fn LLVMCountStructElementTypes(StructTy: &Type) -> c_uint;
    pub(crate) fn LLVMIsPackedStruct(StructTy: &Type) -> Bool;
    pub(crate) fn LLVMStructGetTypeAtIndex(StructTy: &Type, i: c_uint) -> &Type;

    // Operations on array, pointer, and vector types (sequence types)
    pub(crate) fn LLVMRustArrayType(ElementType: &Type, ElementCount: u64) -> &Type;
    pub(crate) fn LLVMPointerType(ElementType: &Type, AddressSpace: c_uint) -> &Type;
    pub(crate) fn LLVMVectorType(ElementType: &Type, ElementCount: c_uint) -> &Type;

    pub(crate) fn LLVMGetElementType(Ty: &Type) -> &Type;
    pub(crate) fn LLVMGetVectorSize(VectorTy: &Type) -> c_uint;

    // Operations on other types
    pub(crate) fn LLVMVoidTypeInContext(C: &Context) -> &Type;
    pub(crate) fn LLVMRustMetadataTypeInContext(C: &Context) -> &Type;

    // Operations on all values
    pub(crate) fn LLVMIsUndef(Val: &Value) -> Bool;
    pub(crate) fn LLVMTypeOf(Val: &Value) -> &Type;
    pub(crate) fn LLVMGetValueName2(Val: &Value, Length: *mut size_t) -> *const c_char;
    pub(crate) fn LLVMSetValueName2(Val: &Value, Name: *const c_char, NameLen: size_t);
    pub(crate) fn LLVMReplaceAllUsesWith<'a>(OldVal: &'a Value, NewVal: &'a Value);
    pub(crate) fn LLVMSetMetadata<'a>(Val: &'a Value, KindID: c_uint, Node: &'a Value);
    pub(crate) fn LLVMPrintValueToString<'a>(Val: &'a Value) -> *mut c_char;

    // Operations on constants of any type
    pub(crate) fn LLVMConstNull(Ty: &Type) -> &Value;
    pub(crate) fn LLVMGetUndef(Ty: &Type) -> &Value;

    // Operations on metadata
    pub(crate) fn LLVMMDStringInContext(C: &Context, Str: *const c_char, SLen: c_uint) -> &Value;
    pub(crate) fn LLVMMDNodeInContext<'a>(
        C: &'a Context,
        Vals: *const &'a Value,
        Count: c_uint,
    ) -> &'a Value;
    pub(crate) fn LLVMAddNamedMetadataOperand<'a>(
        M: &'a Module,
        Name: *const c_char,
        Val: &'a Value,
    );

    // Operations on scalar constants
    pub(crate) fn LLVMConstInt(IntTy: &Type, N: c_ulonglong, SignExtend: Bool) -> &Value;
    pub(crate) fn LLVMConstIntOfArbitraryPrecision(
        IntTy: &Type,
        Wn: c_uint,
        Ws: *const u64,
    ) -> &Value;
    pub(crate) fn LLVMConstReal(RealTy: &Type, N: f64) -> &Value;
    pub(crate) fn LLVMConstIntGetZExtValue(ConstantVal: &ConstantInt) -> c_ulonglong;
    pub(crate) fn LLVMRustConstInt128Get(
        ConstantVal: &ConstantInt,
        SExt: bool,
        high: &mut u64,
        low: &mut u64,
    ) -> bool;

    // Operations on composite constants
    pub(crate) fn LLVMConstStringInContext(
        C: &Context,
        Str: *const c_char,
        Length: c_uint,
        DontNullTerminate: Bool,
    ) -> &Value;
    pub(crate) fn LLVMConstStructInContext<'a>(
        C: &'a Context,
        ConstantVals: *const &'a Value,
        Count: c_uint,
        Packed: Bool,
    ) -> &'a Value;

    pub(crate) fn LLVMConstArray<'a>(
        ElementTy: &'a Type,
        ConstantVals: *const &'a Value,
        Length: c_uint,
    ) -> &'a Value;
    pub(crate) fn LLVMConstVector(ScalarConstantVals: *const &Value, Size: c_uint) -> &Value;

    // Constant expressions
    pub(crate) fn LLVMConstInBoundsGEP2<'a>(
        Ty: &'a Type,
        ConstantVal: &'a Value,
        ConstantIndices: *const &'a Value,
        NumIndices: c_uint,
    ) -> &'a Value;
    pub(crate) fn LLVMConstZExt<'a>(ConstantVal: &'a Value, ToType: &'a Type) -> &'a Value;
    pub(crate) fn LLVMConstPtrToInt<'a>(ConstantVal: &'a Value, ToType: &'a Type) -> &'a Value;
    pub(crate) fn LLVMConstIntToPtr<'a>(ConstantVal: &'a Value, ToType: &'a Type) -> &'a Value;
    pub(crate) fn LLVMConstBitCast<'a>(ConstantVal: &'a Value, ToType: &'a Type) -> &'a Value;
    pub(crate) fn LLVMConstPointerCast<'a>(ConstantVal: &'a Value, ToType: &'a Type) -> &'a Value;
    pub(crate) fn LLVMConstExtractValue(
        AggConstant: &Value,
        IdxList: *const c_uint,
        NumIdx: c_uint,
    ) -> &Value;

    // Operations on global variables, functions, and aliases (globals)
    pub(crate) fn LLVMIsDeclaration(Global: &Value) -> Bool;
    pub(crate) fn LLVMRustGetLinkage(Global: &Value) -> Linkage;
    pub(crate) fn LLVMRustSetLinkage(Global: &Value, RustLinkage: Linkage);
    pub(crate) fn LLVMSetSection(Global: &Value, Section: *const c_char);
    pub(crate) fn LLVMRustGetVisibility(Global: &Value) -> Visibility;
    pub(crate) fn LLVMRustSetVisibility(Global: &Value, Viz: Visibility);
    pub(crate) fn LLVMRustSetDSOLocal(Global: &Value, is_dso_local: bool);
    pub(crate) fn LLVMGetAlignment(Global: &Value) -> c_uint;
    pub(crate) fn LLVMSetAlignment(Global: &Value, Bytes: c_uint);
    pub(crate) fn LLVMSetDLLStorageClass(V: &Value, C: DLLStorageClass);

    // Operations on global variables
    pub(crate) fn LLVMIsAGlobalVariable(GlobalVar: &Value) -> Option<&Value>;
    pub(crate) fn LLVMAddGlobal<'a>(M: &'a Module, Ty: &'a Type, Name: *const c_char) -> &'a Value;
    pub(crate) fn LLVMGetNamedGlobal(M: &Module, Name: *const c_char) -> Option<&Value>;
    pub(crate) fn LLVMRustInsertPrivateGlobal<'a>(M: &'a Module, T: &'a Type) -> &'a Value;
    pub(crate) fn LLVMGetFirstGlobal(M: &Module) -> Option<&Value>;
    pub(crate) fn LLVMGetNextGlobal(GlobalVar: &Value) -> Option<&Value>;
    pub(crate) fn LLVMDeleteGlobal(GlobalVar: &Value);
    pub(crate) fn LLVMGetInitializer(GlobalVar: &Value) -> Option<&Value>;
    pub(crate) fn LLVMSetInitializer<'a>(GlobalVar: &'a Value, ConstantVal: &'a Value);
    pub(crate) fn LLVMIsGlobalConstant(GlobalVar: &Value) -> Bool;
    pub(crate) fn LLVMSetGlobalConstant(GlobalVar: &Value, IsConstant: Bool);
    pub(crate) fn LLVMRustGetNamedValue(
        M: &Module,
        Name: *const c_char,
        NameLen: size_t,
    ) -> Option<&Value>;
    pub(crate) fn LLVMSetTailCall(CallInst: &Value, IsTailCall: Bool);
    pub(crate) fn LLVMSetUnnamedAddress(Global: &Value, UnnamedAddr: UnnamedAddr);

    // Operations on functions
    pub(crate) fn LLVMSetFunctionCallConv(Fn: &Value, CC: c_uint);
    pub(crate) fn LLVMRustAddAlignmentAttr(Fn: &Value, index: c_uint, bytes: u32);
    pub(crate) fn LLVMRustAddFunctionAttribute(Fn: &Value, index: c_uint, attr: Attribute);
    pub(crate) fn LLVMRustAddFunctionAttrStringValue(
        Fn: &Value,
        index: c_uint,
        Name: *const c_char,
        NameLen: size_t,
        Value: *const c_char,
        ValueLen: size_t,
    );
    pub(crate) fn LLVMRustRemoveFunctionAttributes(Fn: &Value, index: c_uint, attr: Attribute);

    // Operations on parameters
    pub(crate) fn LLVMIsAArgument(Val: &Value) -> Option<&Value>;
    pub(crate) fn LLVMCountParams(Fn: &Value) -> c_uint;
    pub(crate) fn LLVMGetParam(Fn: &Value, Index: c_uint) -> &Value;

    // Operations on basic blocks
    pub(crate) fn LLVMGetBasicBlockParent(BB: &BasicBlock) -> &Value;
    pub(crate) fn LLVMAppendBasicBlockInContext<'a>(
        C: &'a Context,
        Fn: &'a Value,
        Name: *const c_char,
    ) -> &'a BasicBlock;

    // Operations on instructions
    pub(crate) fn LLVMIsAInstruction(Val: &Value) -> Option<&Value>;
    pub(crate) fn LLVMGetFirstBasicBlock(Fn: &Value) -> &BasicBlock;

    // Operations on call sites
    pub(crate) fn LLVMRustAddCallSiteAttribute(Instr: &Value, index: c_uint, attr: Attribute);
    pub(crate) fn LLVMRustAddCallSiteAttrString(Instr: &Value, index: c_uint, Name: *const c_char);
    pub(crate) fn LLVMRustAddAlignmentCallSiteAttr(Instr: &Value, index: c_uint, bytes: u32);
    pub(crate) fn LLVMRustAddDereferenceableCallSiteAttr(Instr: &Value, index: c_uint, bytes: u64);
    pub(crate) fn LLVMRustAddDereferenceableOrNullCallSiteAttr(
        Instr: &Value,
        index: c_uint,
        bytes: u64,
    );

    // Operations on load/store instructions (only)
    pub(crate) fn LLVMSetVolatile(MemoryAccessInst: &Value, volatile: Bool);

    // Operations on phi nodes
    pub(crate) fn LLVMAddIncoming<'a>(
        PhiNode: &'a Value,
        IncomingValues: *const &'a Value,
        IncomingBlocks: *const &'a BasicBlock,
        Count: c_uint,
    );

    // Instruction builders
    pub(crate) fn LLVMCreateBuilderInContext<'a>(C: &'a Context) -> &'a mut Builder<'a>;
    pub(crate) fn LLVMPositionBuilderAtEnd<'a>(Builder: &Builder<'a>, Block: &'a BasicBlock);
    pub(crate) fn LLVMGetInsertBlock<'a>(Builder: &Builder<'a>) -> &'a BasicBlock;
    pub(crate) fn LLVMDisposeBuilder<'a>(Builder: &'a mut Builder<'a>);

    pub(crate) fn LLVMBuildUnreachable<'a>(B: &Builder<'a>) -> &'a Value;

    // Terminators
    pub(crate) fn LLVMBuildRetVoid<'a>(B: &Builder<'a>) -> &'a Value;
    pub(crate) fn LLVMBuildRet<'a>(B: &Builder<'a>, V: &'a Value) -> &'a Value;
    pub(crate) fn LLVMBuildBr<'a>(B: &Builder<'a>, Dest: &'a BasicBlock) -> &'a Value;
    pub(crate) fn LLVMBuildCondBr<'a>(
        B: &Builder<'a>,
        If: &'a Value,
        Then: &'a BasicBlock,
        Else: &'a BasicBlock,
    ) -> &'a Value;
    pub(crate) fn LLVMBuildSwitch<'a>(
        B: &Builder<'a>,
        V: &'a Value,
        Else: &'a BasicBlock,
        NumCases: c_uint,
    ) -> &'a Value;

    // Add a case to the switch instruction
    pub(crate) fn LLVMAddCase<'a>(Switch: &'a Value, OnVal: &'a Value, Dest: &'a BasicBlock);

    // Arithmetic
    pub(crate) fn LLVMBuildAdd<'a>(
        B: &Builder<'a>,
        LHS: &'a Value,
        RHS: &'a Value,
        Name: *const c_char,
    ) -> &'a Value;
    pub(crate) fn LLVMBuildFAdd<'a>(
        B: &Builder<'a>,
        LHS: &'a Value,
        RHS: &'a Value,
        Name: *const c_char,
    ) -> &'a Value;
    pub(crate) fn LLVMBuildSub<'a>(
        B: &Builder<'a>,
        LHS: &'a Value,
        RHS: &'a Value,
        Name: *const c_char,
    ) -> &'a Value;
    pub(crate) fn LLVMBuildFSub<'a>(
        B: &Builder<'a>,
        LHS: &'a Value,
        RHS: &'a Value,
        Name: *const c_char,
    ) -> &'a Value;
    pub(crate) fn LLVMBuildMul<'a>(
        B: &Builder<'a>,
        LHS: &'a Value,
        RHS: &'a Value,
        Name: *const c_char,
    ) -> &'a Value;
    pub(crate) fn LLVMBuildFMul<'a>(
        B: &Builder<'a>,
        LHS: &'a Value,
        RHS: &'a Value,
        Name: *const c_char,
    ) -> &'a Value;
    pub(crate) fn LLVMBuildUDiv<'a>(
        B: &Builder<'a>,
        LHS: &'a Value,
        RHS: &'a Value,
        Name: *const c_char,
    ) -> &'a Value;
    pub(crate) fn LLVMBuildExactUDiv<'a>(
        B: &Builder<'a>,
        LHS: &'a Value,
        RHS: &'a Value,
        Name: *const c_char,
    ) -> &'a Value;
    pub(crate) fn LLVMBuildSDiv<'a>(
        B: &Builder<'a>,
        LHS: &'a Value,
        RHS: &'a Value,
        Name: *const c_char,
    ) -> &'a Value;
    pub(crate) fn LLVMBuildExactSDiv<'a>(
        B: &Builder<'a>,
        LHS: &'a Value,
        RHS: &'a Value,
        Name: *const c_char,
    ) -> &'a Value;
    pub(crate) fn LLVMBuildFDiv<'a>(
        B: &Builder<'a>,
        LHS: &'a Value,
        RHS: &'a Value,
        Name: *const c_char,
    ) -> &'a Value;
    pub(crate) fn LLVMBuildURem<'a>(
        B: &Builder<'a>,
        LHS: &'a Value,
        RHS: &'a Value,
        Name: *const c_char,
    ) -> &'a Value;
    pub(crate) fn LLVMBuildSRem<'a>(
        B: &Builder<'a>,
        LHS: &'a Value,
        RHS: &'a Value,
        Name: *const c_char,
    ) -> &'a Value;
    pub(crate) fn LLVMBuildFRem<'a>(
        B: &Builder<'a>,
        LHS: &'a Value,
        RHS: &'a Value,
        Name: *const c_char,
    ) -> &'a Value;
    pub(crate) fn LLVMBuildShl<'a>(
        B: &Builder<'a>,
        LHS: &'a Value,
        RHS: &'a Value,
        Name: *const c_char,
    ) -> &'a Value;
    pub(crate) fn LLVMBuildLShr<'a>(
        B: &Builder<'a>,
        LHS: &'a Value,
        RHS: &'a Value,
        Name: *const c_char,
    ) -> &'a Value;
    pub(crate) fn LLVMBuildAShr<'a>(
        B: &Builder<'a>,
        LHS: &'a Value,
        RHS: &'a Value,
        Name: *const c_char,
    ) -> &'a Value;
    pub(crate) fn LLVMBuildNSWAdd<'a>(
        B: &Builder<'a>,
        LHS: &'a Value,
        RHS: &'a Value,
        Name: *const c_char,
    ) -> &'a Value;
    pub(crate) fn LLVMBuildNUWAdd<'a>(
        B: &Builder<'a>,
        LHS: &'a Value,
        RHS: &'a Value,
        Name: *const c_char,
    ) -> &'a Value;
    pub(crate) fn LLVMBuildNSWSub<'a>(
        B: &Builder<'a>,
        LHS: &'a Value,
        RHS: &'a Value,
        Name: *const c_char,
    ) -> &'a Value;
    pub(crate) fn LLVMBuildNUWSub<'a>(
        B: &Builder<'a>,
        LHS: &'a Value,
        RHS: &'a Value,
        Name: *const c_char,
    ) -> &'a Value;
    pub(crate) fn LLVMBuildNSWMul<'a>(
        B: &Builder<'a>,
        LHS: &'a Value,
        RHS: &'a Value,
        Name: *const c_char,
    ) -> &'a Value;
    pub(crate) fn LLVMBuildNUWMul<'a>(
        B: &Builder<'a>,
        LHS: &'a Value,
        RHS: &'a Value,
        Name: *const c_char,
    ) -> &'a Value;
    pub(crate) fn LLVMBuildAnd<'a>(
        B: &Builder<'a>,
        LHS: &'a Value,
        RHS: &'a Value,
        Name: *const c_char,
    ) -> &'a Value;
    pub(crate) fn LLVMBuildOr<'a>(
        B: &Builder<'a>,
        LHS: &'a Value,
        RHS: &'a Value,
        Name: *const c_char,
    ) -> &'a Value;
    pub(crate) fn LLVMBuildXor<'a>(
        B: &Builder<'a>,
        LHS: &'a Value,
        RHS: &'a Value,
        Name: *const c_char,
    ) -> &'a Value;
    pub(crate) fn LLVMBuildNeg<'a>(B: &Builder<'a>, V: &'a Value, Name: *const c_char)
    -> &'a Value;
    pub(crate) fn LLVMBuildFNeg<'a>(
        B: &Builder<'a>,
        V: &'a Value,
        Name: *const c_char,
    ) -> &'a Value;
    pub(crate) fn LLVMBuildNot<'a>(B: &Builder<'a>, V: &'a Value, Name: *const c_char)
    -> &'a Value;
    pub(crate) fn LLVMRustSetFastMath(Instr: &Value);
    pub(crate) fn LLVMRustSetAlgebraicMath(Instr: &Value);

    // Memory
    pub(crate) fn LLVMBuildAlloca<'a>(
        B: &Builder<'a>,
        Ty: &'a Type,
        Name: *const c_char,
    ) -> &'a Value;
    pub(crate) fn LLVMBuildArrayAlloca<'a>(
        B: &Builder<'a>,
        Ty: &'a Type,
        Val: &'a Value,
        Name: *const c_char,
    ) -> &'a Value;
    pub(crate) fn LLVMBuildLoad<'a>(
        B: &Builder<'a>,
        PointerVal: &'a Value,
        Name: *const c_char,
    ) -> &'a Value;

    pub(crate) fn LLVMBuildStore<'a>(B: &Builder<'a>, Val: &'a Value, Ptr: &'a Value) -> &'a Value;

    pub(crate) fn LLVMBuildGEP2<'a>(
        B: &Builder<'a>,
        Ty: &'a Type,
        Pointer: &'a Value,
        Indices: *const &'a Value,
        NumIndices: c_uint,
        Name: *const c_char,
    ) -> &'a Value;
    pub(crate) fn LLVMBuildInBoundsGEP2<'a>(
        B: &Builder<'a>,
        Ty: &'a Type,
        Pointer: &'a Value,
        Indices: *const &'a Value,
        NumIndices: c_uint,
        Name: *const c_char,
    ) -> &'a Value;

    // Casts
    pub(crate) fn LLVMBuildTrunc<'a>(
        B: &Builder<'a>,
        Val: &'a Value,
        DestTy: &'a Type,
        Name: *const c_char,
    ) -> &'a Value;
    pub(crate) fn LLVMBuildZExt<'a>(
        B: &Builder<'a>,
        Val: &'a Value,
        DestTy: &'a Type,
        Name: *const c_char,
    ) -> &'a Value;
    pub(crate) fn LLVMBuildSExt<'a>(
        B: &Builder<'a>,
        Val: &'a Value,
        DestTy: &'a Type,
        Name: *const c_char,
    ) -> &'a Value;
    pub(crate) fn LLVMBuildFPToUI<'a>(
        B: &Builder<'a>,
        Val: &'a Value,
        DestTy: &'a Type,
        Name: *const c_char,
    ) -> &'a Value;
    pub(crate) fn LLVMBuildFPToSI<'a>(
        B: &Builder<'a>,
        Val: &'a Value,
        DestTy: &'a Type,
        Name: *const c_char,
    ) -> &'a Value;
    pub(crate) fn LLVMBuildUIToFP<'a>(
        B: &Builder<'a>,
        Val: &'a Value,
        DestTy: &'a Type,
        Name: *const c_char,
    ) -> &'a Value;
    pub(crate) fn LLVMBuildSIToFP<'a>(
        B: &Builder<'a>,
        Val: &'a Value,
        DestTy: &'a Type,
        Name: *const c_char,
    ) -> &'a Value;
    pub(crate) fn LLVMBuildFPTrunc<'a>(
        B: &Builder<'a>,
        Val: &'a Value,
        DestTy: &'a Type,
        Name: *const c_char,
    ) -> &'a Value;
    pub(crate) fn LLVMBuildFPExt<'a>(
        B: &Builder<'a>,
        Val: &'a Value,
        DestTy: &'a Type,
        Name: *const c_char,
    ) -> &'a Value;
    pub(crate) fn LLVMBuildPtrToInt<'a>(
        B: &Builder<'a>,
        Val: &'a Value,
        DestTy: &'a Type,
        Name: *const c_char,
    ) -> &'a Value;
    pub(crate) fn LLVMBuildIntToPtr<'a>(
        B: &Builder<'a>,
        Val: &'a Value,
        DestTy: &'a Type,
        Name: *const c_char,
    ) -> &'a Value;
    pub(crate) fn LLVMBuildBitCast<'a>(
        B: &Builder<'a>,
        Val: &'a Value,
        DestTy: &'a Type,
        Name: *const c_char,
    ) -> &'a Value;
    pub(crate) fn LLVMBuildPointerCast<'a>(
        B: &Builder<'a>,
        Val: &'a Value,
        DestTy: &'a Type,
        Name: *const c_char,
    ) -> &'a Value;
    pub(crate) fn LLVMRustBuildIntCast<'a>(
        B: &Builder<'a>,
        Val: &'a Value,
        DestTy: &'a Type,
        IsSized: bool,
    ) -> &'a Value;

    // Comparisons
    pub(crate) fn LLVMBuildICmp<'a>(
        B: &Builder<'a>,
        Op: c_uint,
        LHS: &'a Value,
        RHS: &'a Value,
        Name: *const c_char,
    ) -> &'a Value;
    pub(crate) fn LLVMBuildFCmp<'a>(
        B: &Builder<'a>,
        Op: c_uint,
        LHS: &'a Value,
        RHS: &'a Value,
        Name: *const c_char,
    ) -> &'a Value;

    // Miscellaneous instructions
    pub(crate) fn LLVMBuildPhi<'a>(B: &Builder<'a>, Ty: &'a Type, Name: *const c_char)
    -> &'a Value;
    pub(crate) fn LLVMRustGetInstrProfIncrementIntrinsic<'a>(M: &Module) -> &'a Value;
    pub(crate) fn LLVMRustBuildMemCpy<'a>(
        B: &Builder<'a>,
        Dst: &'a Value,
        DstAlign: c_uint,
        Src: &'a Value,
        SrcAlign: c_uint,
        Size: &'a Value,
        IsVolatile: bool,
    ) -> &'a Value;
    pub(crate) fn LLVMRustBuildMemMove<'a>(
        B: &Builder<'a>,
        Dst: &'a Value,
        DstAlign: c_uint,
        Src: &'a Value,
        SrcAlign: c_uint,
        Size: &'a Value,
        IsVolatile: bool,
    ) -> &'a Value;
    pub(crate) fn LLVMRustBuildMemSet<'a>(
        B: &Builder<'a>,
        Dst: &'a Value,
        DstAlign: c_uint,
        Val: &'a Value,
        Size: &'a Value,
        IsVolatile: bool,
    ) -> &'a Value;
    pub(crate) fn LLVMBuildSelect<'a>(
        B: &Builder<'a>,
        If: &'a Value,
        Then: &'a Value,
        Else: &'a Value,
        Name: *const c_char,
    ) -> &'a Value;
    pub(crate) fn LLVMBuildVAArg<'a>(
        B: &Builder<'a>,
        list: &'a Value,
        Ty: &'a Type,
        Name: *const c_char,
    ) -> &'a Value;
    pub(crate) fn LLVMBuildExtractElement<'a>(
        B: &Builder<'a>,
        VecVal: &'a Value,
        Index: &'a Value,
        Name: *const c_char,
    ) -> &'a Value;
    pub(crate) fn LLVMBuildInsertElement<'a>(
        B: &Builder<'a>,
        VecVal: &'a Value,
        EltVal: &'a Value,
        Index: &'a Value,
        Name: *const c_char,
    ) -> &'a Value;
    pub(crate) fn LLVMBuildShuffleVector<'a>(
        B: &Builder<'a>,
        V1: &'a Value,
        V2: &'a Value,
        Mask: &'a Value,
        Name: *const c_char,
    ) -> &'a Value;
    pub(crate) fn LLVMBuildExtractValue<'a>(
        B: &Builder<'a>,
        AggVal: &'a Value,
        Index: c_uint,
        Name: *const c_char,
    ) -> &'a Value;
    pub(crate) fn LLVMBuildInsertValue<'a>(
        B: &Builder<'a>,
        AggVal: &'a Value,
        EltVal: &'a Value,
        Index: c_uint,
        Name: *const c_char,
    ) -> &'a Value;

    pub(crate) fn LLVMRustBuildVectorReduceFAdd<'a>(
        B: &Builder<'a>,
        Acc: &'a Value,
        Src: &'a Value,
    ) -> &'a Value;
    pub(crate) fn LLVMRustBuildVectorReduceFMul<'a>(
        B: &Builder<'a>,
        Acc: &'a Value,
        Src: &'a Value,
    ) -> &'a Value;
    pub(crate) fn LLVMRustBuildVectorReduceAdd<'a>(B: &Builder<'a>, Src: &'a Value) -> &'a Value;
    pub(crate) fn LLVMRustBuildVectorReduceMul<'a>(B: &Builder<'a>, Src: &'a Value) -> &'a Value;
    pub(crate) fn LLVMRustBuildVectorReduceAnd<'a>(B: &Builder<'a>, Src: &'a Value) -> &'a Value;
    pub(crate) fn LLVMRustBuildVectorReduceOr<'a>(B: &Builder<'a>, Src: &'a Value) -> &'a Value;
    pub(crate) fn LLVMRustBuildVectorReduceXor<'a>(B: &Builder<'a>, Src: &'a Value) -> &'a Value;
    pub(crate) fn LLVMRustBuildVectorReduceMin<'a>(
        B: &Builder<'a>,
        Src: &'a Value,
        IsSigned: bool,
    ) -> &'a Value;
    pub(crate) fn LLVMRustBuildVectorReduceMax<'a>(
        B: &Builder<'a>,
        Src: &'a Value,
        IsSigned: bool,
    ) -> &'a Value;
    pub(crate) fn LLVMRustBuildVectorReduceFMin<'a>(
        B: &Builder<'a>,
        Src: &'a Value,
        IsNaN: bool,
    ) -> &'a Value;
    pub(crate) fn LLVMRustBuildVectorReduceFMax<'a>(
        B: &Builder<'a>,
        Src: &'a Value,
        IsNaN: bool,
    ) -> &'a Value;

    pub(crate) fn LLVMRustBuildMinNum<'a>(
        B: &Builder<'a>,
        LHS: &'a Value,
        RHS: &'a Value,
    ) -> &'a Value;
    pub(crate) fn LLVMRustBuildMaxNum<'a>(
        B: &Builder<'a>,
        LHS: &'a Value,
        RHS: &'a Value,
    ) -> &'a Value;

    pub(crate) fn LLVMDisposeMessage(message: *mut c_char);

    /// Returns a string describing the last error caused by an LLVMRust* call.
    pub(crate) fn LLVMRustGetLastError() -> *const c_char;

    pub(crate) fn LLVMStructCreateNamed(C: &Context, Name: *const c_char) -> &Type;

    pub(crate) fn LLVMStructSetBody<'a>(
        StructTy: &'a Type,
        ElementTypes: *const &'a Type,
        ElementCount: c_uint,
        Packed: Bool,
    );

    /// Prepares inline assembly.
    pub(crate) fn LLVMRustInlineAsm(
        Ty: &Type,
        AsmString: *const c_char,
        AsmStringLen: size_t,
        Constraints: *const c_char,
        ConstraintsLen: size_t,
        SideEffects: Bool,
        AlignStack: Bool,
        Dialect: AsmDialect,
    ) -> &Value;
    pub(crate) fn LLVMRustInlineAsmVerify(
        Ty: &Type,
        Constraints: *const c_char,
        ConstraintsLen: size_t,
    ) -> bool;

    pub(crate) fn LLVMIsAConstantInt(value_ref: &Value) -> Option<&ConstantInt>;

    pub(crate) fn LLVMRustPrintModule<'a>(
        M: &'a Module,
        Output: *const c_char,
        OutputLen: size_t,
        Demangle: extern "C" fn(*const c_char, size_t, *mut c_char, size_t) -> size_t,
    ) -> LLVMRustResult;

    pub(crate) fn LLVMRustModuleBufferCreate(M: &Module) -> &'static mut ModuleBuffer;
    pub(crate) fn LLVMRustModuleBufferPtr(p: &ModuleBuffer) -> *const u8;
    pub(crate) fn LLVMRustModuleBufferLen(p: &ModuleBuffer) -> usize;
    pub(crate) fn LLVMRustModuleBufferFree(p: &'static mut ModuleBuffer);
    pub(crate) fn LLVMRustModuleCost(M: &Module) -> u64;

    pub(crate) fn LLVMRustThinLTOBufferCreate(M: &Module) -> &'static mut ThinLTOBuffer;
    pub(crate) fn LLVMRustThinLTOBufferFree(M: &'static mut ThinLTOBuffer);
    pub(crate) fn LLVMRustThinLTOBufferPtr(M: &ThinLTOBuffer) -> *const c_char;
    pub(crate) fn LLVMRustThinLTOBufferLen(M: &ThinLTOBuffer) -> size_t;
    pub(crate) fn LLVMRustCreateThinLTOData(
        Modules: *const ThinLTOModule,
        NumModules: c_uint,
        PreservedSymbols: *const *const c_char,
        PreservedSymbolsLen: c_uint,
    ) -> Option<&'static mut ThinLTOData>;
    pub(crate) fn LLVMRustPrepareThinLTOResolveWeak(Data: &ThinLTOData, Module: &Module) -> bool;
    pub(crate) fn LLVMRustPrepareThinLTOInternalize(Data: &ThinLTOData, Module: &Module) -> bool;
    pub(crate) fn LLVMRustFreeThinLTOData(Data: &'static mut ThinLTOData);
    pub(crate) fn LLVMRustParseBitcodeForLTO(
        Context: &Context,
        Data: *const u8,
        len: usize,
        Identifier: *const c_char,
        IdentiferLen: size_t,
    ) -> Option<&Module>;
    pub(crate) fn LLVMRustGetBitcodeSliceFromObjectData(
        Data: *const u8,
        len: usize,
        out_len: &mut usize,
    ) -> *const u8;
    pub(crate) fn LLVMRustThinLTOGetDICompileUnit(
        M: &Module,
        CU1: &mut *mut c_void,
        CU2: &mut *mut c_void,
    );
    pub(crate) fn LLVMRustThinLTOPatchDICompileUnit(M: &Module, CU: *mut c_void);

    pub(crate) fn LLVMRustAddDereferenceableAttr(Fn: &Value, index: c_uint, bytes: u64);
    pub(crate) fn LLVMRustAddDereferenceableOrNullAttr(Fn: &Value, index: c_uint, bytes: u64);

    pub(crate) fn LLVMRustPositionBuilderAtStart<'a>(B: &Builder<'a>, BB: &'a BasicBlock);
    // Atomics
    pub fn LLVMRustBuildAtomicCmpXchg<'a>(
        B: &Builder<'a>,
        LHS: &Value,
        CMP: &Value,
        RHS: &Value,
        Order: AtomicOrdering,
        FailureOrder: AtomicOrdering,
        Weak: Bool,
    ) -> &'a Value;

    pub fn LLVMBuildAtomicRMW<'a>(
        B: &Builder<'a>,
        Op: LLVMAtomicRmwBinOp,
        LHS: &Value,
        RHS: &Value,
        Order: AtomicOrdering,
        SingleThreaded: Bool,
    ) -> &'a Value;
}
/// LLVMAtomicOrdering
#[derive(Copy, Clone)]
#[repr(C)]
pub(crate) enum AtomicOrdering {
    #[allow(dead_code)]
    NotAtomic = 0,
    #[allow(dead_code)]
    Unordered = 1,
    Monotonic = 2,
    // Consume = 3,  // Not specified yet.
    Acquire = 4,
    Release = 5,
    AcquireRelease = 6,
    SequentiallyConsistent = 7,
}
impl AtomicOrdering {
    pub(crate) fn from_generic(ao: rustc_middle::ty::AtomicOrdering) -> Self {
        use rustc_middle::ty::AtomicOrdering as Common;
        match ao {
            Common::Relaxed => Self::Monotonic,
            Common::Acquire => Self::Acquire,
            Common::Release => Self::Release,
            Common::AcqRel => Self::AcquireRelease,
            Common::SeqCst => Self::SequentiallyConsistent,
        }
    }
}

/// FFI-safe mirror of LLVMAtomicRMWBinOp from the LLVM C API.
#[derive(Copy, Clone)]
#[repr(C)]
pub enum LLVMAtomicRmwBinOp {
    AtomicXchg = 0,
    AtomicAdd = 1,
    AtomicSub = 2,
    AtomicAnd = 3,
    AtomicNand = 4,
    AtomicOr = 5,
    AtomicXor = 6,
    AtomicMax = 7,
    AtomicMin = 8,
    AtomicUMax = 9,
    AtomicUMin = 10,
}

impl From<AtomicRmwBinOp> for LLVMAtomicRmwBinOp {
    fn from(op: AtomicRmwBinOp) -> Self {
        match op {
            AtomicRmwBinOp::AtomicXchg => Self::AtomicXchg,
            AtomicRmwBinOp::AtomicAdd => Self::AtomicAdd,
            AtomicRmwBinOp::AtomicSub => Self::AtomicSub,
            AtomicRmwBinOp::AtomicAnd => Self::AtomicAnd,
            AtomicRmwBinOp::AtomicNand => Self::AtomicNand,
            AtomicRmwBinOp::AtomicOr => Self::AtomicOr,
            AtomicRmwBinOp::AtomicXor => Self::AtomicXor,
            AtomicRmwBinOp::AtomicMax => Self::AtomicMax,
            AtomicRmwBinOp::AtomicMin => Self::AtomicMin,
            AtomicRmwBinOp::AtomicUMax => Self::AtomicUMax,
            AtomicRmwBinOp::AtomicUMin => Self::AtomicUMin,
        }
    }
}
