# Contributing to MEXT-M1

Thanks for taking a look. MEXT is a local-first systems project: Rust for the
memory tiering core, Python/MLX for model and retrieval layers.

## Requirements

- macOS 13+ on Apple Silicon (the code uses `madvise`, `vm_stat`, FSEvents).
- Rust stable (see `rust-toolchain.toml`), Python 3.10–3.13.
- 8 GB of RAM is enough; use `--ram-limit 2G` for local testing.

## Setup

```bash
./setup.sh              # venv + mlx/lancedb + release build
make doctor             # environment checks
```

## Before opening a PR

```bash
make review             # release build + cargo test + pytest + node check + selftest
make e2e                # boots a real daemon: CLI + HTTP, no MLX required
make lint               # fmt-check + clippy -D warnings + ruff
```

All four must pass. CI runs the same set on `macos-14`.

## Code style

- Rust: `cargo fmt`, `clippy --all-targets -- -D warnings`, no new heavy
  dependencies. New behaviour needs unit tests in the same module or
  `core-daemon/tests/`.
- Python: `ruff check` (config in `ai-engine/pyproject.toml`), type hints on
  public functions, tests under `ai-engine/tests/`.
- Dashboard assets are embedded at compile time via `include_str!`; after
  editing `dashboard/*`, rebuild the Rust binary before testing.
- Every new metric must be documented in `docs/grafana/README.md` and covered by
  `core-daemon/tests/grafana.rs` if it is exposed in Prometheus.

## Measuring changes

Performance and memory claims need evidence:

```bash
mext bench <dir> --ram-limit 2G --files 200
make bench
```

Attach the table to the PR. For predictor changes, record the online hit rate
before/after on a real workload.

## Commit / PR conventions

- Keep commits focused; a PR should explain the *why*, not just the diff.
- Do not commit tokens, `~/.mext` state, model weights, or `.venv`.
- Security-sensitive changes (sudo tracers, sockets, payments) need a note in
  the PR and a test for the failure path.

## Reporting bugs

Use the issue templates. `mext doctor` output plus `~/.mext/mext.log` covers most
cases. Security issues: see [SECURITY.md](SECURITY.md).
