use anyhow::{anyhow, Result};
use bindgen::Builder;
use cmake::Config;
use git2::Repository;
use std::{
    env,
    path::{Path, PathBuf},
};

const REPO_URL: &str = "https://github.com/rw0x0/HElib";
const HELIB_FOLDER: &str = "HElib";
const INSTALL_FOLDER: &str = "helib_install";
const LIB_FOLDER: &str = "helib_pack/lib";
const INCLUDE_FOLDER: &str = "helib_pack/include";
const HELIB_LIB: &str = "helib";
const GMP_LIB: &str = "gmp";
const NTL_LIB: &str = "ntl";

#[allow(dead_code)]
fn checkout_commit(repo: &Repository, commit: &str) -> Result<()> {
    let (object, reference) = repo.revparse_ext(commit)?;

    repo.checkout_tree(&object, None)?;

    match reference {
        // gref is an actual reference like branches or tags
        Some(gref) => repo.set_head(gref.name().ok_or(anyhow!("Gref has name"))?),
        // this is a commit, not a reference
        None => repo.set_head_detached(object.id()),
    }?;
    Ok(())
}

fn download(out_dir: &Path) -> Result<()> {
    let out_folder = out_dir.join(HELIB_FOLDER);
    if out_folder.exists() {
        std::fs::remove_dir_all(&out_folder)?;
    }

    let _repo = Repository::clone(REPO_URL, out_folder)?;

    // Checkout correct hash
    // checkout_commit(&_repo, "9f531cd23ca51ffdc780257f30fdc14c9de98c3b")?;

    Ok(())
}

fn build(out_dir: &Path) -> Result<()> {
    #[cfg(feature = "clang")]
    {
        env::set_var("CC", "clang");
        env::set_var("CXX", "clang++");
    }

    let mut cfg = Config::new(out_dir.join(Path::new(HELIB_FOLDER)));

    let install_dir = out_dir.join(INSTALL_FOLDER);
    cfg.define("PACKAGE_BUILD", "ON");
    cfg.define("CMAKE_INSTALL_PREFIX", install_dir);

    // Build SEAL in Release except specified otherwise.
    #[cfg(feature = "debug")]
    cfg.profile("Debug");
    #[cfg(not(feature = "debug"))]
    cfg.profile("RelWithDebInfo");

    let _dst = cfg.build();
    Ok(())
}

fn link(out_dir: &Path) -> Result<()> {
    let lib_dir = out_dir.join(INSTALL_FOLDER).join(LIB_FOLDER);

    println!("cargo:rustc-link-search={}", lib_dir.display());
    println!("cargo:rustc-link-lib=dylib={}", GMP_LIB);
    println!("cargo:rustc-link-lib=dylib={}", NTL_LIB);
    println!("cargo:rustc-link-lib=static={}", HELIB_LIB);
    println!("cargo:rustc-link-lib=stdc++");
    Ok(())
}

fn exist(out_dir: &Path) -> bool {
    let lib_dir = out_dir.join(INSTALL_FOLDER).join(LIB_FOLDER);
    if lib_dir.exists() {
        return true;
    }
    false
}

fn bindgen(out_dir: &Path) -> Result<()> {
    let include_dir = out_dir.join(INSTALL_FOLDER).join(INCLUDE_FOLDER);

    let incl_str = match include_dir.to_str() {
        Some(x) => x,
        None => return Err(anyhow!("Error Path does not exist?")),
    };

    let bindings = Builder::default()
        .derive_copy(true)
        .derive_debug(true)
        .header("src/wrapper.h")
        .clang_arg(format!("-I{}", incl_str))
        .clang_arg("-std=c++17")
        .clang_arg("-x")
        .clang_arg("c++")
        .generate();

    let unpacked = match bindings {
        Ok(v) => v,
        Err(_) => {
            return Err(anyhow!("Error generating bindings"));
        }
    };

    // Write the bindings to the $OUT_DIR/bindings.rs file.
    unpacked.write_to_file(out_dir.join("bindings.rs"))?;
    Ok(())
}

fn main() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("getting environment var failed"));
    if !exist(&out_dir) {
        download(&out_dir).expect("Download failed");
        build(&out_dir).expect("Build failed");
    }
    link(&out_dir).expect("Link failed");
    bindgen(&out_dir).expect("Bindgen failed");
}
