//! MEXT-M1 core library: predictive file/memory tiering for Apple Silicon.
//!
//! Layering (bottom → top):
//! * [`platform`]  — macOS sysctl / vm_stat / madvise wrappers (safe API).
//! * [`memmap`]    — page-aligned anonymous mappings we fully own, so
//!                   `madvise(MADV_DONTNEED)` is legal and returns pages to XNU.
//! * [`arc`]       — byte-capacity ARC (Adaptive Replacement Cache).
//! * [`arena`]     — refcounted page-packing arena for small compressed frames.
//! * [`pool`]      — LZ4 compressed RAM pool backed by ARC + madvise eviction.
//! * [`predictor`] — Markov transition graph + EMA frequency + hour-of-day model.
//! * [`monitor`]   — FSEvents watcher (`notify`) and optional `fs_usage` tracer.
//! * [`core`]      — actor that wires monitor → predictor → pool → AI bridge.
//! * [`ipc`]       — newline-delimited JSON-RPC 2.0 over a Unix domain socket.
//! * [`dashboard`] — axum HTTP + embedded single-page dashboard.

pub mod arc;
pub mod arena;
pub mod bench;
pub mod bridge;
pub mod config;
pub mod core;
pub mod dashboard;
pub mod demo;
pub mod ipc;
pub mod memmap;
pub mod metrics;
pub mod monitor;
pub mod platform;
pub mod pool;
pub mod predictor;
pub mod run;
