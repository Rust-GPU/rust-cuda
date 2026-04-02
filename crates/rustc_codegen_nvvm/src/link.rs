use object::{Object, ObjectSection};
use rustc_codegen_ssa::{CompiledModule, CompiledModules, CrateInfo, NativeLib};
use rustc_data_structures::fx::FxHashSet;
use rustc_data_structures::memmap::Mmap;
use rustc_data_structures::owned_slice::{OwnedSlice, slice_owned, try_slice_owned};
use rustc_hir::attrs::NativeLibKind;
use rustc_metadata::creader::MetadataLoader;
use rustc_middle::bug;
use rustc_middle::middle::dependency_format::Linkage;
use rustc_session::output::out_filename;
use rustc_session::{
    Session,
    config::{CrateType, OutputFilenames, OutputType},
    output::check_file_is_writeable,
};
use rustc_span::Symbol;
use rustc_target::spec::Target;
use std::ops::Deref;
use std::{
    ffi::OsStr,
    fs::File,
    io::{self, Read},
    path::{Path, PathBuf},
};
use tar::{Archive, Builder, Header};
use tracing::{debug, trace};

use crate::LlvmMod;
use crate::context::CodegenArgs;
use crate::ptx_filter::{PtxFilter, PtxFilterConfig};

pub(crate) struct NvvmMetadataLoader;

fn load_metadata_with(
    path: &Path,
    f: impl for<'a> FnOnce(&'a [u8]) -> Result<&'a [u8], String>,
) -> Result<OwnedSlice, String> {
    let file =
        File::open(path).map_err(|e| format!("failed to open file '{}': {}", path.display(), e))?;

    unsafe { Mmap::map(file) }
        .map_err(|e| format!("failed to mmap file '{}': {}", path.display(), e))
        .and_then(|mmap| try_slice_owned(mmap, |mmap| f(mmap)))
}

impl MetadataLoader for NvvmMetadataLoader {
    fn get_rlib_metadata(&self, _target: &Target, path: &Path) -> Result<OwnedSlice, String> {
        trace!("Retrieving rlib metadata for `{:?}`", path);
        read_metadata(path)
    }

    fn get_dylib_metadata(&self, target: &Target, path: &Path) -> Result<OwnedSlice, String> {
        debug!("getting rlib metadata for {}", path.display());
        // This is required for loading metadata from proc macro crates compiled as dylibs for the host target.
        if target.is_like_aix {
            bug!("aix dynlibs unsupported");
        } else {
            load_metadata_with(path, |data| search_for_section(path, data, ".rustc"))
        }
    }
}

fn read_metadata(rlib: &Path) -> Result<OwnedSlice, String> {
    let read_meta = || -> Result<Option<OwnedSlice>, io::Error> {
        for entry in Archive::new(File::open(rlib)?).entries()? {
            let mut entry = entry?;
            if entry.path()? == Path::new(".metadata") {
                let mut bytes = Vec::new();
                entry.read_to_end(&mut bytes)?;
                return Ok(Some(slice_owned(bytes, Deref::deref)));
            }
        }
        Ok(None)
    };

    match read_meta() {
        Ok(Some(m)) => Ok(m),
        Ok(None) => Err(format!("No .metadata file in rlib: {rlib:?}")),
        Err(io) => Err(format!("Failed to read rlib at {rlib:?}: {io}")),
    }
}

fn search_for_section<'a>(path: &Path, bytes: &'a [u8], section: &str) -> Result<&'a [u8], String> {
    let Ok(file) = object::File::parse(bytes) else {
        // The parse above could fail for odd reasons like corruption, but for
        // now we just interpret it as this target doesn't support metadata
        // emission in object files so the entire byte slice itself is probably
        // a metadata file. Ideally though if necessary we could at least check
        // the prefix of bytes to see if it's an actual metadata object and if
        // not forward the error along here.
        return Ok(bytes);
    };
    file.section_by_name(section)
        .ok_or_else(|| format!("no `{}` section in '{}'", section, path.display()))?
        .data()
        .map_err(|e| {
            format!(
                "failed to read {} section in '{}': {}",
                section,
                path.display(),
                e
            )
        })
}

pub fn link(
    sess: &Session,
    compiled_modules: CompiledModules,
    crate_info: CrateInfo,
    metadata: rustc_metadata::EncodedMetadata,
    outputs: &OutputFilenames,
) {
    let crate_name = crate_info.local_crate_name.as_str();
    debug!("Linking crate `{}`", crate_name);
    // largely inspired by rust-gpu
    let output_metadata = sess.opts.output_types.contains_key(&OutputType::Metadata);
    for &crate_type in sess.opts.crate_types.iter() {
        if (sess.opts.unstable_opts.no_codegen || !sess.opts.output_types.should_codegen())
            && !output_metadata
            && crate_type == CrateType::Executable
        {
            continue;
        }

        for obj in compiled_modules
            .modules
            .iter()
            .filter_map(|m| m.object.as_ref())
        {
            check_file_is_writeable(obj, sess);
        }

        if outputs.outputs.should_codegen() {
            let out_filename = out_filename(sess, crate_type, outputs, Symbol::intern(crate_name));
            let out_filename_file_for_writing =
                out_filename.file_for_writing(outputs, OutputType::Exe, "", None);
            match crate_type {
                CrateType::Rlib => {
                    link_rlib(
                        sess,
                        &compiled_modules,
                        &crate_info,
                        &out_filename_file_for_writing,
                        &metadata,
                    );
                }
                CrateType::Executable | CrateType::Cdylib | CrateType::Dylib => {
                    let _ = link_exe(
                        &compiled_modules.allocator_module,
                        sess,
                        crate_type,
                        &out_filename_file_for_writing,
                        &compiled_modules,
                        &crate_info,
                    );
                }
                other => sess.dcx().fatal(format!("Invalid crate type: {other:?}")),
            }
        }
    }
}

fn link_rlib(
    sess: &Session,
    compiled_modules: &CompiledModules,
    crate_info: &CrateInfo,
    out_filename: &Path,
    metadata: &rustc_metadata::EncodedMetadata,
) {
    debug!("Linking rlib `{:?}`", out_filename);
    let mut file_list = Vec::<&Path>::new();

    for obj in compiled_modules
        .modules
        .iter()
        .filter_map(|m| m.object.as_ref())
    {
        file_list.push(obj);
    }

    for lib in crate_info.used_libraries.iter() {
        // native libraries in cuda doesnt make much sense, extern functions
        // do exist in nvvm for stuff like cuda syscalls and cuda provided functions
        // but including libraries doesnt make sense because nvvm would have to translate
        // the binary directly to ptx. We might want to add some way of linking in
        // ptx files or custom bitcode modules as "libraries" perhaps in the future.
        if let NativeLibKind::Static {
            bundle: None | Some(true),
            ..
        } = lib.kind
        {
            sess.dcx().err(format!(
                "Adding native libraries to rlib is not supported in CUDA: {}",
                lib.name
            ));
        }
    }
    trace!("Files linked in rlib:\n{:#?}", file_list);

    create_archive(sess, &file_list, metadata.stub_or_full(), out_filename);
}

fn link_exe(
    allocator: &Option<CompiledModule>,
    sess: &Session,
    crate_type: CrateType,
    out_filename: &Path,
    compiled_modules: &CompiledModules,
    crate_info: &CrateInfo,
) -> io::Result<()> {
    let mut objects = Vec::new();
    let mut rlibs = Vec::new();
    for obj in compiled_modules
        .modules
        .iter()
        .filter_map(|m| m.object.as_ref())
    {
        objects.push(obj.clone());
    }

    link_local_crate_native_libs_and_dependent_crate_libs(&mut rlibs, sess, crate_type, crate_info);

    let mut root_file_name = out_filename.file_name().unwrap().to_owned();
    root_file_name.push(".dir");
    let out_dir = out_filename.with_file_name(root_file_name);
    if !out_dir.is_dir() {
        std::fs::create_dir_all(&out_dir)?;
    }

    codegen_into_ptx_file(allocator, sess, &objects, &rlibs, out_filename)
}

/// This is the meat of the codegen, taking all of the llvm bitcode modules we have, and giving them to
/// nvvm to make into a final
fn codegen_into_ptx_file(
    allocator: &Option<CompiledModule>,
    sess: &Session,
    objects: &[PathBuf],
    rlibs: &[PathBuf],
    out_filename: &Path,
) -> io::Result<()> {
    debug!(
        "Codegenning crate into PTX, allocator: {}, objects:\n{:#?}, rlibs:\n{:#?}, out_filename:\n{:#?}",
        allocator.is_some(),
        objects,
        rlibs,
        out_filename
    );

    // we need to make a new llvm context because we need it for linking together modules,
    // but we dont have our original one because rustc drops tyctxt and codegencx before linking.
    let cx = LlvmMod::new("link_tmp");

    let mut modules = Vec::with_capacity(objects.len() + rlibs.len());

    // object files (theyre not object files, they are impostors ඞ) are the bitcode modules produced by this codegen session
    // they *should* be the final crate.
    for obj in objects {
        let bitcode = std::fs::read(obj)?;
        modules.push(bitcode);
    }

    // rlibs are archives that we made previously, they are usually made for crates that are referenced
    // in this crate. We must unpack them and devour their bitcode to link in.
    for rlib in rlibs {
        let mut cgus = Vec::with_capacity(16);
        for entry in Archive::new(File::open(rlib)?).entries()? {
            let mut entry = entry?;
            // metadata is where rustc puts rlib metadata, so its not a cgu we are interested in.
            if entry.path().unwrap() != Path::new(".metadata") {
                // std::fs::read adds 1 to the size, so do the same here - see comment:
                // https://github.com/rust-lang/rust/blob/72868e017bdade60603a25889e253f556305f996/library/std/src/fs.rs#L200-L202
                let mut bitcode = Vec::with_capacity(entry.size() as usize + 1);
                entry.read_to_end(&mut bitcode).unwrap();
                cgus.push(bitcode);
            }
        }

        modules.extend(cgus);
    }

    if let Some(alloc) = allocator {
        let bc = std::fs::read(
            alloc
                .object
                .clone()
                .expect("expected obj path for allocator module"),
        )?;
        modules.push(bc);
    }

    // now that we have our nice bitcode modules, we just need to find libdevice and give our
    // modules to nvvm to make a final ptx file

    // we need to actually parse the codegen args again, because codegencx is not available at link time.
    let args = CodegenArgs::from_session(sess);

    let ptx_bytes = match crate::nvvm::codegen_bitcode_modules(&args, sess, modules, cx.llcx) {
        Ok(bytes) => bytes,
        Err(err) => {
            // TODO(RDambrosio016): maybe include the nvvm log with this fatal error
            sess.dcx().fatal(err.to_string())
        }
    };

    // If disassembly is requested, print PTX to stderr
    if args.disassemble.is_some()
        && let Ok(ptx_str) = std::str::from_utf8(&ptx_bytes)
    {
        let config = PtxFilterConfig::from_codegen_args(&args);
        let filter = PtxFilter::new(config);
        let output = filter.filter(ptx_str);
        if !output.is_empty() {
            // Check if we're in JSON mode by checking the error format
            use rustc_session::config::ErrorOutputType;
            match sess.opts.error_format {
                ErrorOutputType::Json { .. } => {
                    sess.dcx()
                        .err("PTX disassembly output in JSON mode is not supported");
                }
                _ => {
                    // In normal mode, just print to stderr
                    // Replace tabs with spaces for cleaner output
                    let output = output.replace('\t', "    ");
                    eprintln!("{output}");
                }
            }
        }
    }

    std::fs::write(out_filename, ptx_bytes)
}

fn create_archive(sess: &Session, files: &[&Path], metadata: &[u8], out_filename: &Path) {
    if let Err(err) = try_create_archive(files, metadata, out_filename) {
        sess.dcx().fatal(format!("Failed to create archive: {err}"));
    }
}

fn try_create_archive(files: &[&Path], metadata: &[u8], out_filename: &Path) -> io::Result<()> {
    let file = File::create(out_filename)?;
    let mut builder = Builder::new(file);
    {
        let mut header = Header::new_gnu();
        header.set_path(".metadata")?;
        header.set_size(metadata.len() as u64);
        header.set_cksum();
        builder.append(&header, metadata)?;
    }
    let mut filenames = FxHashSet::default();
    filenames.insert(OsStr::new(".metadata"));
    for file in files {
        assert!(
            filenames.insert(file.file_name().unwrap()),
            "Duplicate filename in archive: {:?}",
            file.file_name().unwrap()
        );
        builder.append_path_with_name(file, file.file_name().unwrap())?;
    }
    builder.into_inner()?;
    Ok(())
}

// most of the code from here is derived from rust-gpu

fn link_local_crate_native_libs_and_dependent_crate_libs(
    rlibs: &mut Vec<PathBuf>,
    sess: &Session,
    crate_type: CrateType,
    crate_info: &CrateInfo,
) {
    if sess.opts.unstable_opts.link_native_libraries {
        add_local_native_libraries(sess, crate_info);
    }
    add_upstream_rust_crates(sess, rlibs, crate_info, crate_type);
    if sess.opts.unstable_opts.link_native_libraries {
        add_upstream_native_libraries(sess, crate_info, crate_type);
    }
}

fn add_local_native_libraries(sess: &Session, crate_info: &CrateInfo) {
    let relevant_libs = crate_info
        .used_libraries
        .iter()
        .filter(|l| relevant_lib(sess, l));
    assert_eq!(relevant_libs.count(), 0);
}

fn add_upstream_rust_crates(
    sess: &Session,
    rlibs: &mut Vec<PathBuf>,
    crate_info: &CrateInfo,
    crate_type: CrateType,
) {
    let (_, data) = crate_info
        .dependency_formats
        .iter()
        .find(|(ty, _)| **ty == crate_type)
        .expect("failed to find crate type in dependency format list");
    let deps = &crate_info.used_crates;
    for cnum in deps.iter() {
        let src = &crate_info.used_crate_source[cnum];
        match data[*cnum] {
            Linkage::NotLinked => {}
            Linkage::Static => rlibs.push(src.rlib.as_ref().unwrap().clone()),
            // should we just ignore includedFromDylib?
            Linkage::Dynamic | Linkage::IncludedFromDylib => {
                sess.dcx().fatal("Dynamic Linking is not supported in CUDA")
            }
        }
    }
}

fn add_upstream_native_libraries(sess: &Session, crate_info: &CrateInfo, _crate_type: CrateType) {
    let crates = &crate_info.used_crates;
    for cnum in crates {
        for lib in crate_info.native_libraries[cnum].iter() {
            if !relevant_lib(sess, lib) {
                continue;
            }
            sess.dcx()
                .fatal("Native libraries are not supported in CUDA");
        }
    }
}

fn relevant_lib(sess: &Session, lib: &NativeLib) -> bool {
    match &lib.cfg {
        Some(cfg) => rustc_attr_parsing::eval_config_entry(sess, cfg).as_bool(),
        None => true,
    }
}
