//! Synthetic activity generator for `--demo`.
//!
//! Without it an idle machine produces an empty dashboard. The generator does a
//! biased random walk over the workspace: it prefers files in the same directory
//! as the last one, which mimics how people actually edit code (switch between a
//! few neighbouring files, occasionally jump).

use crate::core::Core;
use crate::monitor::Source;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

/// xorshift64* — deterministic, dependency-free RNG.
fn next_rand(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *state = x;
    x.wrapping_mul(0x2545_F491_4F6C_DD1D)
}

pub fn spawn(core: Arc<Core>, files: Vec<PathBuf>, seed: u64) {
    if files.is_empty() {
        eprintln!("[demo] no files in workspace — demo mode idle");
        return;
    }
    std::thread::Builder::new()
        .name("mext-demo".into())
        .spawn(move || {
            let mut state = seed | 1;
            // People do not browse uniformly: they work through a small set of
            // neighbouring files for a while. That is exactly the pattern the
            // Markov predictor is meant to learn, so the demo produces it.
            let mut session: Vec<usize> =
                pick_session(&files, next_rand(&mut state) as usize % files.len());
            let mut pos = 0usize;
            loop {
                if core.shutdown_requested() {
                    return;
                }
                let index = session[pos.min(session.len() - 1)];
                core.observe(&files[index], Source::Demo);

                let roll = next_rand(&mut state) % 100;
                if roll < 62 {
                    pos = (pos + 1) % session.len();
                } else if roll < 84 {
                    pos = (pos + session.len() - 1) % session.len();
                } else if roll < 93 {
                    pos = next_rand(&mut state) as usize % session.len();
                } else {
                    // Switch to another directory and keep going.
                    let anchor = next_rand(&mut state) as usize % files.len();
                    session = pick_session(&files, anchor);
                    pos = 0;
                }
                let pause = 220 + next_rand(&mut state) % 620;
                std::thread::sleep(Duration::from_millis(pause));
            }
        })
        .ok();
}

/// Up to 10 files from the directory of `anchor` (plus the anchor itself).
fn pick_session(files: &[PathBuf], anchor: usize) -> Vec<usize> {
    let dir = files[anchor].parent().map(|p| p.to_path_buf());
    let mut same: Vec<usize> = files
        .iter()
        .enumerate()
        .filter(|(_, p)| p.parent().map(|d| d.to_path_buf()) == dir)
        .map(|(i, _)| i)
        .collect();
    if same.is_empty() {
        same.push(anchor);
    }
    same.sort_unstable();
    same.truncate(10);
    if !same.contains(&anchor) {
        same.insert(0, anchor);
        same.truncate(10);
    }
    // Start the rotation at the anchor so the first observation is natural.
    if let Some(offset) = same.iter().position(|i| *i == anchor) {
        same.rotate_left(offset);
    }
    same
}
