use std::cell;
use std::fs;
use std::path;
use std::sync;

use bindgen::callbacks::{DeriveInfo, ItemInfo, ItemKind, MacroParsingBehavior, ParseCallbacks};

/// Enum to handle different callback combinations
#[derive(Debug)]
pub(crate) enum BindgenCallbacks {
    /// For bindings that need function renaming (driver, runtime, cublas)
    WithFunctionRenames {
        function_renames: Box<FunctionRenames>,
        cargo_callbacks: bindgen::CargoCallbacks,
    },
    /// For bindings that only need comment processing (nvptx, nvvm)
    Simple {
        cargo_callbacks: bindgen::CargoCallbacks,
    },
}

impl BindgenCallbacks {
    pub fn with_function_renames(function_renames: FunctionRenames) -> Self {
        Self::WithFunctionRenames {
            function_renames: Box::new(function_renames),
            cargo_callbacks: bindgen::CargoCallbacks::new(),
        }
    }

    pub fn simple() -> Self {
        Self::Simple {
            cargo_callbacks: bindgen::CargoCallbacks::new(),
        }
    }
}

impl ParseCallbacks for BindgenCallbacks {
    fn will_parse_macro(&self, name: &str) -> MacroParsingBehavior {
        match self {
            Self::WithFunctionRenames {
                function_renames, ..
            } => function_renames.will_parse_macro(name),
            Self::Simple { .. } => MacroParsingBehavior::Default,
        }
    }

    fn item_name(&self, original_item_name: &str) -> Option<String> {
        match self {
            Self::WithFunctionRenames {
                function_renames, ..
            } => function_renames.item_name(original_item_name),
            Self::Simple { .. } => None,
        }
    }

    fn add_derives(&self, info: &DeriveInfo) -> Vec<String> {
        match self {
            Self::WithFunctionRenames {
                function_renames, ..
            } => ParseCallbacks::add_derives(function_renames.as_ref(), info),
            Self::Simple { .. } => vec![],
        }
    }

    fn generated_name_override(&self, item_info: ItemInfo<'_>) -> Option<String> {
        match self {
            Self::WithFunctionRenames {
                function_renames, ..
            } => ParseCallbacks::generated_name_override(function_renames.as_ref(), item_info),
            Self::Simple { .. } => None,
        }
    }

    fn generated_link_name_override(&self, item_info: ItemInfo<'_>) -> Option<String> {
        match self {
            Self::WithFunctionRenames {
                function_renames, ..
            } => ParseCallbacks::generated_link_name_override(function_renames.as_ref(), item_info),
            Self::Simple { .. } => None,
        }
    }

    // Delegate cargo callbacks
    fn include_file(&self, filename: &str) {
        match self {
            Self::WithFunctionRenames {
                cargo_callbacks, ..
            }
            | Self::Simple { cargo_callbacks } => cargo_callbacks.include_file(filename),
        }
    }

    fn read_env_var(&self, var: &str) {
        match self {
            Self::WithFunctionRenames {
                cargo_callbacks, ..
            }
            | Self::Simple { cargo_callbacks } => cargo_callbacks.read_env_var(var),
        }
    }
}

/// Struct to handle renaming of functions through macro expansion.
#[derive(Debug)]
pub(crate) struct FunctionRenames {
    func_prefix: &'static str,
    out_dir: path::PathBuf,
    includes: path::PathBuf,
    include_dirs: Vec<path::PathBuf>,
    macro_names: cell::RefCell<Vec<String>>,
    func_remaps: sync::OnceLock<bimap::BiHashMap<String, String>>,
}

impl FunctionRenames {
    pub fn new<P: AsRef<path::Path>, I: Into<path::PathBuf>>(
        func_prefix: &'static str,
        out_dir: P,
        includes: I,
        include_dirs: Vec<path::PathBuf>,
    ) -> Self {
        Self {
            func_prefix,
            out_dir: out_dir.as_ref().to_path_buf(),
            includes: includes.into(),
            include_dirs,
            macro_names: cell::RefCell::new(Vec::new()),
            func_remaps: sync::OnceLock::new(),
        }
    }

    fn record_macro(&self, name: &str) {
        self.macro_names.borrow_mut().push(name.to_string());
    }

    fn expand(&self) -> &bimap::BiHashMap<String, String> {
        self.func_remaps.get_or_init(|| {
            if self.macro_names.borrow().is_empty() {
                return bimap::BiHashMap::new();
            }

            let expand_me = self.out_dir.join("expand_macros.c");
            let includes = fs::read_to_string(&self.includes)
                .expect("Failed to read includes for function renames");

            let mut template = format!(
                r#"{includes}
#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)
#define RENAMED(from, to) "RUST_RENAMED" TOSTRING(from) TOSTRING(to)
"#
            );

            for name in self.macro_names.borrow().iter() {
                // Add an underscore to the left so that it won't get expanded.
                template.push_str(&format!("RENAMED(_{name}, {name})\n"));
            }

            {
                let mut temp = fs::File::create(&expand_me).unwrap();
                std::io::Write::write_all(&mut temp, template.as_bytes()).unwrap();
            }

            let mut build = cc::Build::new();
            build
                .file(&expand_me)
                .includes(&self.include_dirs)
                .cargo_warnings(false);

            let expanded = match build.try_expand() {
                Ok(expanded) => expanded,
                Err(e) => panic!("Failed to expand macros: {e}"),
            };
            let expanded = str::from_utf8(&expanded).unwrap();

            let mut remaps = bimap::BiHashMap::new();
            for line in expanded.lines().rev() {
                let rename_prefix = "\"RUST_RENAMED\" ";

                if let Some((original, expanded)) = line
                    .strip_prefix(rename_prefix)
                    .map(|s| s.replace("\"", ""))
                    .and_then(|s| {
                        s.split_once(' ')
                            .map(|(l, r)| (l[1..].to_string(), r.to_string()))
                    })
                    .filter(|(l, r)| l != r && !r.is_empty())
                {
                    remaps.insert(original.to_string(), expanded.to_string());
                }
            }

            fs::remove_file(&expand_me).expect("Failed to remove temporary file");
            remaps
        })
    }
}

impl ParseCallbacks for FunctionRenames {
    fn will_parse_macro(&self, name: &str) -> MacroParsingBehavior {
        if name.starts_with(self.func_prefix) {
            self.record_macro(name);
        }
        MacroParsingBehavior::Default
    }

    fn generated_name_override(&self, item_info: ItemInfo<'_>) -> Option<String> {
        let remaps = self.expand();
        match item_info.kind {
            ItemKind::Function => remaps.get_by_right(item_info.name).cloned(),
            _ => None,
        }
    }

    fn generated_link_name_override(&self, item_info: ItemInfo<'_>) -> Option<String> {
        let remaps = self.expand();
        match item_info.kind {
            ItemKind::Function => remaps.get_by_left(item_info.name).cloned(),
            _ => None,
        }
    }

    fn add_derives(&self, _info: &DeriveInfo) -> Vec<String> {
        vec![]
    }
}
