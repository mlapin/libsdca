//! Compiles the C ABI (src/c_api.cpp) into a static library.
//!
//! SDCA_BLAS_BACKEND selects the BLAS as in the Python build: BUILTIN (the
//! default unless SDCA_BLAS is set) or EXTERNAL, linked with the SDCA_BLAS
//! libraries (for example "-lopenblas" or "-L/opt/blas/lib -lblas"; the
//! forms of `directives`). CXX and AR choose the tools (CXX may carry flags).
//! The C++ sources come from SDCA_SOURCE_DIR, else native/ (vendored by
//! package.sh), else the repository around the crate.
use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

fn run(command: &mut Command) {
    let status = command
        .status()
        .unwrap_or_else(|e| panic!("cannot run {command:?}: {e}"));
    assert!(status.success(), "{command:?} failed with {status}");
}

/// The BLAS macro of the headers and the linker flags of an external BLAS.
fn blas() -> (&'static str, Vec<String>) {
    let flags = env::var("SDCA_BLAS").unwrap_or_default();
    let backend = env::var("SDCA_BLAS_BACKEND").unwrap_or_else(|_| "AUTO".into());
    match (backend.to_uppercase().as_str(), flags.is_empty()) {
        ("BUILTIN", _) | ("AUTO", true) => ("BLAS_BUILTIN", Vec::new()),
        ("EXTERNAL", true) => panic!("SDCA_BLAS_BACKEND=EXTERNAL needs SDCA_BLAS linker flags"),
        ("EXTERNAL", false) | ("AUTO", false) => (
            "BLAS_DEFAULT_LOCAL_HEADER",
            flags.split_whitespace().map(String::from).collect(),
        ),
        _ => panic!("SDCA_BLAS_BACKEND must be AUTO, EXTERNAL, or BUILTIN"),
    }
}

/// The Cargo directives of one SDCA_BLAS entry: -L<dir>, -l<name>,
/// -l:<file> or an absolute path to a .so or .a library.
fn directives(flag: &str) -> Vec<String> {
    let nonempty = |s: &&str| !s.is_empty();
    if let Some(dir) = flag.strip_prefix("-L").filter(nonempty) {
        return vec![format!("rustc-link-search=native={dir}")];
    }
    if let Some(file) = flag.strip_prefix("-l:").filter(nonempty) {
        return vec![verbatim(file)];
    }
    if let Some(name) = flag.strip_prefix("-l").filter(nonempty) {
        return vec![format!("rustc-link-lib={name}")];
    }
    let path = Path::new(flag);
    let file = path.file_name().and_then(|f| f.to_str()).unwrap_or("");
    let library = file.ends_with(".a") || file.ends_with(".so") || file.contains(".so.");
    match path.parent().and_then(Path::to_str) {
        Some(dir) if path.is_absolute() && library => {
            vec![format!("rustc-link-search=native={dir}"), verbatim(file)]
        }
        _ => panic!(
            "SDCA_BLAS entry {flag:?} is not -L<dir>, -l<name>, -l:<file> or an absolute \
             path to a .so or .a library; pass other linker flags in RUSTFLAGS"
        ),
    }
}

/// Link the library file `file` by its exact name, as the linker's -l:file.
/// Archives stay unbundled, so the final link resolves them like the linker.
fn verbatim(file: &str) -> String {
    let kind = if file.ends_with(".a") {
        "static:-bundle,"
    } else {
        "dylib:"
    };
    format!("rustc-link-lib={kind}+verbatim={file}")
}

fn main() {
    // The default tools build for the host.
    let cross = env::var("TARGET").ok() != env::var("HOST").ok();
    if cross && (env::var_os("CXX").is_none() || env::var_os("AR").is_none()) {
        panic!("Cross compilation needs CXX and AR set to the target's tools");
    }
    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    // The repository's sources in a checkout, else those vendored into a packaged crate: the
    // copy that rust/package.sh leaves in rust/native/ goes stale as the sources change.
    let repository = manifest.join("..");
    let default = if repository.join("include/sdca/c_api.h").exists() {
        repository
    } else {
        manifest.join("native")
    };
    let root = env::var_os("SDCA_SOURCE_DIR").map_or(default, PathBuf::from);
    let out = PathBuf::from(env::var("OUT_DIR").unwrap());
    let (macro_name, flags) = blas();
    let object = out.join("c_api.o");
    let version = env::var("CARGO_PKG_VERSION").unwrap();
    let cxx = env::var("CXX").unwrap_or_else(|_| "c++".into());
    let mut cxx = cxx.split_whitespace();
    run(Command::new(cxx.next().expect("CXX is empty"))
        .args(cxx)
        .args([
            "-std=c++11",
            "-O3",
            "-DNDEBUG",
            "-fPIC",
            "-pthread",
            "-fno-fast-math",
            "-ffp-contract=off",
        ])
        .arg(format!("-D{macro_name}"))
        .arg("-DSDCA_USE_CHRONO")
        .arg(format!("-DSDCA_VERSION=\"{version}\""))
        .arg("-I")
        .arg(root.join("include"))
        .arg("-c")
        .arg(root.join("src/c_api.cpp"))
        .arg("-o")
        .arg(&object));
    run(Command::new(env::var("AR").unwrap_or_else(|_| "ar".into()))
        .arg("crs")
        .arg(out.join("libsdca_c.a"))
        .arg(&object));
    println!("cargo:rustc-link-search=native={}", out.display());
    println!("cargo:rustc-link-lib=static=sdca_c");
    for directive in flags.iter().flat_map(|flag| directives(flag)) {
        println!("cargo:{directive}");
    }
    let std = if env::var("CARGO_CFG_TARGET_VENDOR").as_deref() == Ok("apple") {
        "c++"
    } else {
        "stdc++"
    };
    println!("cargo:rustc-link-lib=dylib={std}");
    for path in ["include", "src/c_api.cpp", "src/tensor_losses.h"] {
        println!("cargo:rerun-if-changed={}", root.join(path).display());
    }
    for var in [
        "CXX",
        "AR",
        "SDCA_BLAS",
        "SDCA_BLAS_BACKEND",
        "SDCA_SOURCE_DIR",
    ] {
        println!("cargo:rerun-if-env-changed={var}");
    }
}
