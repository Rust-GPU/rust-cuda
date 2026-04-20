use std::ffi::CString;
use std::mem;
use std::path::Path;
use std::str;
use std::sync::Once;
use std::sync::atomic::{AtomicBool, Ordering};

use libc::c_int;
use rustc_middle::bug;
use rustc_session::Session;
use rustc_target::spec::MergeFunctions;

use crate::llvm;

static POISONED: AtomicBool = AtomicBool::new(false);
static INIT: Once = Once::new();

pub(crate) fn init(sess: &Session) {
    unsafe {
        // Before we touch LLVM, make sure that multithreading is enabled.
        INIT.call_once(|| {
            if llvm::LLVMStartMultithreaded() != 1 {
                // use an extra bool to make sure that all future usage of LLVM
                // cannot proceed despite the Once not running more than once.
                POISONED.store(true, Ordering::SeqCst);
            }

            configure_llvm(sess);
        });

        if POISONED.load(Ordering::SeqCst) {
            bug!("couldn't enable multi-threaded LLVM");
        }
    }
}

unsafe fn configure_llvm(sess: &Session) {
    // TODO(RDambrosio016): We override the meaning of llvm-args to pass our own nvvm args,
    // but we should probably retain a way to pass args to LLVM.
    let n_args = sess.opts.cg.llvm_args.len() + sess.target.llvm_args.len();
    let mut llvm_c_strs = Vec::with_capacity(n_args + 1);
    let mut llvm_args = Vec::with_capacity(n_args + 1);

    // fn llvm_arg_to_arg_name(full_arg: &str) -> &str {
    //     full_arg
    //         .trim()
    //         .split(|c: char| c == '=' || c.is_whitespace())
    //         .next()
    //         .unwrap_or("")
    // }

    // let cg_opts = sess.opts.cg.llvm_args.iter();
    // let tg_opts = sess.target.llvm_args.iter();
    // let sess_args = cg_opts.chain(tg_opts);

    // dont print anything in here or it will interfere with cargo trying to print stuff which will
    // cause the compilation to fail in mysterious ways.

    // let user_specified_args: FxHashSet<_> = sess_args
    //     .clone()
    //     .map(|s| llvm_arg_to_arg_name(s))
    //     .filter(|s| !s.is_empty())
    //     .collect();

    {
        // This adds the given argument to LLVM. Unless `force` is true
        // user specified arguments are *not* overridden.
        let mut add = |arg: &str, _force: bool| {
            // if force || !user_specified_args.contains(llvm_arg_to_arg_name(arg)) {
            let s = CString::new(arg).unwrap();
            llvm_args.push(s.as_ptr());
            llvm_c_strs.push(s);
            // }
        };
        // Set the llvm "program name" to make usage and invalid argument messages more clear.
        // add("rustc -Cllvm-args=\"...\" with", true);

        if sess.opts.unstable_opts.time_llvm_passes {
            add("-time-passes", false);
        }
        if sess.opts.unstable_opts.print_llvm_passes {
            add("-debug-pass=Structure", false);
        }
        if !sess.opts.unstable_opts.no_generate_arange_section {
            add("-generate-arange-section", false);
        }

        match sess
            .opts
            .unstable_opts
            .merge_functions
            .unwrap_or(sess.target.merge_functions)
        {
            MergeFunctions::Disabled | MergeFunctions::Trampolines => {}
            MergeFunctions::Aliases => {
                add("-mergefunc-use-aliases", false);
            }
        }

        // HACK(eddyb) LLVM inserts `llvm.assume` calls to preserve align attributes
        // during inlining. Unfortunately these may block other optimizations.
        add(
            "-preserve-alignment-assumptions-during-inlining=false",
            false,
        );

        // Use non-zero `import-instr-limit` multiplier for cold callsites.
        add("-import-cold-multiplier=0.1", false);

        // Forward unroll-threshold if specified in llvm_args
        for arg in &sess.opts.cg.llvm_args {
            if arg.starts_with("-unroll-threshold=") {
                add(arg, true);
            }
        }
    }

    unsafe { llvm::LLVMInitializePasses() };

    for plugin in &sess.opts.unstable_opts.llvm_plugins {
        let path = Path::new(plugin);
        let res = unsafe { libloading::Library::new(path) };
        match res {
            Ok(_) => {}
            Err(e) => bug!("couldn't load plugin: {}", e),
        }
        mem::forget(res);
    }

    unsafe {
        llvm::LLVMInitializeNVPTXTarget();
        llvm::LLVMInitializeNVPTXTargetInfo();
        llvm::LLVMInitializeNVPTXTargetMC();
        llvm::LLVMInitializeNVPTXAsmPrinter();

        llvm::LLVMRustSetLLVMOptions(llvm_args.len() as c_int, llvm_args.as_ptr());
    }
}
