use std::ffi::CString;
use std::sync::Arc;

use rustc_codegen_ssa::{
    ModuleCodegen,
    back::{
        lto::{SerializedModule, ThinModule, ThinShared},
        write::CodegenContext,
    },
    traits::{ModuleBufferMethods, ThinBufferMethods},
};
use rustc_errors::{DiagCtxtHandle, FatalError};
use rustc_middle::dep_graph::WorkProduct;
use tracing::{debug, trace};

use crate::NvvmCodegenBackend;
use crate::common::AsCCharPtr;
use crate::{LlvmMod, llvm};

pub struct ModuleBuffer(&'static mut llvm::ModuleBuffer);

unsafe impl Send for ModuleBuffer {}
unsafe impl Sync for ModuleBuffer {}

impl ModuleBuffer {
    pub(crate) fn new(m: &llvm::Module) -> ModuleBuffer {
        ModuleBuffer(unsafe { llvm::LLVMRustModuleBufferCreate(m) })
    }
}

impl ModuleBufferMethods for ModuleBuffer {
    fn data(&self) -> &[u8] {
        unsafe {
            trace!("Retrieving data in module buffer");
            let ptr = llvm::LLVMRustModuleBufferPtr(self.0);
            let len = llvm::LLVMRustModuleBufferLen(self.0);
            std::slice::from_raw_parts(ptr, len)
        }
    }
}

impl Drop for ModuleBuffer {
    fn drop(&mut self) {
        unsafe {
            llvm::LLVMRustModuleBufferFree(&mut *(self.0 as *mut _));
        }
    }
}

pub struct ThinBuffer(&'static mut llvm::ThinLTOBuffer);

unsafe impl Send for ThinBuffer {}
unsafe impl Sync for ThinBuffer {}

impl ThinBuffer {
    pub(crate) fn new(m: &llvm::Module) -> ThinBuffer {
        unsafe {
            let buffer = llvm::LLVMRustThinLTOBufferCreate(m);

            ThinBuffer(buffer)
        }
    }
}

impl ThinBufferMethods for ThinBuffer {
    fn data(&self) -> &[u8] {
        unsafe {
            trace!("Retrieving data in thin buffer");
            let ptr = llvm::LLVMRustThinLTOBufferPtr(self.0) as *const _;

            let len = llvm::LLVMRustThinLTOBufferLen(self.0);

            std::slice::from_raw_parts(ptr, len)
        }
    }

    fn thin_link_data(&self) -> &[u8] {
        todo!()
    }
}

impl Drop for ThinBuffer {
    fn drop(&mut self) {
        unsafe {
            llvm::LLVMRustThinLTOBufferFree(&mut *(self.0 as *mut _));
        }
    }
}

#[allow(dead_code)]
pub struct ThinData(&'static mut llvm::ThinLTOData);

unsafe impl Send for ThinData {}
unsafe impl Sync for ThinData {}

impl Drop for ThinData {
    fn drop(&mut self) {
        unsafe {
            llvm::LLVMRustFreeThinLTOData(&mut *(self.0 as *mut _));
        }
    }
}

// essentially does nothing for now.
pub(crate) fn run_thin(
    _cgcx: &CodegenContext<NvvmCodegenBackend>,
    modules: Vec<(String, ThinBuffer)>,
    cached_modules: Vec<(SerializedModule<ModuleBuffer>, WorkProduct)>,
) -> Result<(Vec<ThinModule<NvvmCodegenBackend>>, Vec<WorkProduct>), FatalError> {
    debug!("Running thin LTO");
    let mut thin_buffers = Vec::with_capacity(modules.len());
    let mut module_names = Vec::with_capacity(modules.len() + cached_modules.len());
    // let thin_modules = Vec::with_capacity(modules.len() + cached_modules.len());

    for (name, buf) in modules {
        let cname = CString::new(name.clone()).unwrap();
        // thin_modules.push(
        //     llvm::ThinLTOModule {
        //         identifier: cname.as_ptr(),
        //         data: buf.data().as_ptr(),
        //         len: buf.data().len()
        //     }
        // );
        thin_buffers.push(buf);
        module_names.push(cname);
    }

    let mut serialized_modules = Vec::with_capacity(cached_modules.len());

    for (sm, wp) in cached_modules {
        let _slice_u8 = sm.data();
        serialized_modules.push(sm);
        module_names.push(CString::new(wp.cgu_name).unwrap());
    }

    let shared = Arc::new(ThinShared {
        data: (),
        thin_buffers,
        serialized_modules,
        module_names,
    });

    let mut opt_jobs = vec![];
    for (module_index, _) in shared.module_names.iter().enumerate() {
        opt_jobs.push(ThinModule {
            shared: shared.clone(),
            idx: module_index,
        });
    }

    Ok((opt_jobs, vec![]))
}

pub(crate) unsafe fn optimize_thin(
    cgcx: &CodegenContext<NvvmCodegenBackend>,
    thin_module: ThinModule<NvvmCodegenBackend>,
) -> Result<ModuleCodegen<LlvmMod>, FatalError> {
    // essentially does nothing
    let dcx = cgcx.create_dcx();
    let dcx = dcx.handle();

    let module_name = &thin_module.shared.module_names[thin_module.idx];

    let llcx = unsafe { llvm::LLVMRustContextCreate(cgcx.fewer_names) };
    let llmod =
        parse_module(llcx, module_name.to_str().unwrap(), thin_module.data(), dcx)? as *const _;

    let module =
        ModuleCodegen::new_regular(thin_module.name().to_string(), LlvmMod { llcx, llmod });
    Ok(module)
}

pub(crate) fn parse_module<'a>(
    cx: &'a llvm::Context,
    name: &str,
    data: &[u8],
    dcx: DiagCtxtHandle<'_>,
) -> Result<&'a llvm::Module, FatalError> {
    unsafe {
        llvm::LLVMRustParseBitcodeForLTO(
            cx,
            data.as_ptr(),
            data.len(),
            name.as_c_char_ptr(),
            name.len(),
        )
        .ok_or_else(|| {
            let msg = "failed to parse bitcode for LTO module";
            crate::back::llvm_err(dcx, msg)
        })
    }
}
