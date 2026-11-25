//! Compiles the C ABI (src/c_api.cpp) into a static library.
//!
//! The compile flags and the BLAS come from src/native_build.txt, as in the Python and CMake
//! builds: SDCA_BLAS_BACKEND is AUTO (the default), EXTERNAL or BUILTIN; AUTO and EXTERNAL
//! link the first candidate (or SDCA_BLAS, for example "-lopenblas" or
//! "-L/opt/blas/lib -lblas"; the forms of `directives`) with which src/blas_probe.cpp links,
//! and AUTO without one takes the built-in products with a warning. CXX and AR choose the
//! tools (CXX may carry flags). The C++ sources come from SDCA_SOURCE_DIR, else native/
//! (vendored by package.sh), else the repository around the crate.
use std::env;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

fn run(command: &mut Command) {
    let status = command
        .status()
        .unwrap_or_else(|e| panic!("cannot run {command:?}: {e}"));
    assert!(status.success(), "{command:?} failed with {status}");
}

/// The entries of `key` in src/native_build.txt.
fn settings(root: &Path, key: &str) -> Vec<String> {
    let path = root.join("src/native_build.txt");
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
    let prefix = format!("{key} ");
    text.lines()
        .filter_map(|line| line.strip_prefix(&prefix))
        .map(|entry| entry.trim().to_string())
        .collect()
}

/// Whether src/blas_probe.cpp links with the linker flags.
fn links(cxx: &[String], root: &Path, out: &Path, flags: &[String]) -> bool {
    Command::new(&cxx[0])
        .args(&cxx[1..])
        .args(["-std=c++11", "-I"])
        .arg(root.join("include"))
        .arg(root.join("src/blas_probe.cpp"))
        .arg("-o")
        .arg(out.join("blas_probe"))
        .args(flags)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok_and(|status| status.success())
}

/// The macros of the selected BLAS and its linker flags.
fn blas(cxx: &[String], root: &Path, out: &Path) -> (Vec<&'static str>, Vec<String>) {
    let backend = env::var("SDCA_BLAS_BACKEND").unwrap_or_else(|_| "AUTO".into());
    let backend = backend.to_uppercase();
    match backend.as_str() {
        "BUILTIN" => return (vec!["BLAS_BUILTIN"], Vec::new()),
        "AUTO" | "EXTERNAL" => {}
        _ => panic!("SDCA_BLAS_BACKEND must be AUTO, EXTERNAL, or BUILTIN"),
    }
    let given = env::var("SDCA_BLAS").ok().filter(|flags| !flags.is_empty());
    let candidates = given
        .clone()
        .map_or_else(|| settings(root, "blas"), |flags| vec![flags]);
    for candidate in &candidates {
        let flags: Vec<String> = candidate.split_whitespace().map(String::from).collect();
        if links(cxx, root, out, &flags) {
            return (vec!["BLAS_DEFAULT_LOCAL_HEADER"], flags);
        }
    }
    let tried = candidates.join(", ");
    if backend == "EXTERNAL" || given.is_some() {
        panic!("No CBLAS links with {tried}; set SDCA_BLAS to the linker flags of one, or choose SDCA_BLAS_BACKEND=BUILTIN");
    }
    println!(
        "cargo:warning=libsdca found no BLAS (tried {tried}) and uses its built-in products: \
         install OpenBLAS (or another CBLAS) or set SDCA_BLAS to its linker flags, then rebuild"
    );
    (vec!["BLAS_BUILTIN", "SDCA_BLAS_FALLBACK"], Vec::new())
}

/// The Cargo directives of one SDCA_BLAS entry: -L<dir>, -l<name>,
/// -l:<file> or an absolute path to a .so or .a library.
fn directives(flag: &str) -> Vec<String> {
    let nonempty = |s: &&str| !s.is_empty();
    if let Some(name) = flag.strip_prefix("framework=").filter(nonempty) {
        return vec![format!("rustc-link-lib=framework={name}")];
    }
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

/// The linker flags with "-framework <name>" joined into one "framework=<name>" entry.
fn link_entries(flags: &[String]) -> Vec<String> {
    let mut entries = Vec::new();
    let mut iter = flags.iter();
    while let Some(flag) = iter.next() {
        match (flag.as_str(), iter.clone().next()) {
            ("-framework", Some(name)) => {
                entries.push(format!("framework={name}"));
                iter.next();
            }
            _ => entries.push(flag.clone()),
        }
    }
    entries
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
    // Cargo copies the external package readme to RUST.md in the crate archive.
    let guide = manifest.join("../docs/RUST.md");
    let guide = if guide.exists() {
        guide
    } else {
        manifest.join("RUST.md")
    };
    std::fs::copy(&guide, out.join("guide.md")).expect("copy the Rust guide for doctests");
    println!("cargo:rerun-if-changed={}", guide.display());
    let cxx: Vec<String> = env::var("CXX")
        .unwrap_or_else(|_| "c++".into())
        .split_whitespace()
        .map(String::from)
        .collect();
    assert!(!cxx.is_empty(), "CXX is empty");
    let (macros, flags) = blas(&cxx, &root, &out);
    let object = out.join("c_api.o");
    let version = env::var("CARGO_PKG_VERSION").unwrap();
    run(Command::new(&cxx[0])
        .args(&cxx[1..])
        .args(settings(&root, "flags"))
        .arg("-fPIC")
        .args(macros.iter().map(|name| format!("-D{name}")))
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
    for directive in link_entries(&flags)
        .iter()
        .flat_map(|flag| directives(flag))
    {
        println!("cargo:{directive}");
    }
    let std = if env::var("CARGO_CFG_TARGET_VENDOR").as_deref() == Ok("apple") {
        "c++"
    } else {
        "stdc++"
    };
    println!("cargo:rustc-link-lib=dylib={std}");
    for path in [
        "include",
        "src/c_api.cpp",
        "src/blas_probe.cpp",
        "src/native_build.txt",
    ] {
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
