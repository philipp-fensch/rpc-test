fn main() {
    // Link C-Part
    println!("cargo:rustc-link-search=./");
    println!("cargo:rustc-link-lib=static=rpc-connection");

    // Link RPC-Lib
    println!("cargo:rustc-link-search=rpc-lib/submodules/libtirpc/install/lib/");
    println!("cargo:rustc-link-lib=static=tirpc");
}
