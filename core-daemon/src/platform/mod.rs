//! Platform layer. Only Apple Silicon macOS is supported by this build.

#[cfg(target_os = "macos")]
mod macos;

#[cfg(target_os = "macos")]
pub use macos::*;

#[cfg(not(target_os = "macos"))]
compile_error!("MEXT-M1 targets macOS on Apple Silicon (madvise/MADV_DONTNEED + vm_stat).");
