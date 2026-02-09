//! Build script for ZVD
//!
//! This handles compilation of protobuf files when the server feature is enabled.

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Only compile proto files when server feature is enabled
    #[cfg(feature = "server")]
    {
        // Set the protoc path to use the vendored version
        std::env::set_var("PROTOC", protobuf_src::protoc());

        let proto_file = "proto/worker.proto";

        // Tell Cargo to rerun this build script if the proto file changes
        println!("cargo:rerun-if-changed={}", proto_file);
        println!("cargo:rerun-if-changed=proto");

        // Compile the proto file
        tonic_build::configure()
            .build_server(true)
            .build_client(true)
            .compile_protos(&[proto_file], &["proto"])?;
    }

    Ok(())
}
