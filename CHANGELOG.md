# Changelog

All notable changes to MEXT-M1 are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.6.1] — 2026-11-13

### Fixed
- Release checksums are generated with a relative filename, so
  `scripts/install-release.sh` verifies downloads correctly.
- The installer busts the release CDN cache after an asset re-upload.
- `scripts/e2e.sh` falls back to the system Python when `.venv` is absent.
- CI moved to `macos-15` and now gates on `clippy -D warnings`, `ruff`, and both
  e2e modes.

## [0.6.0] — 2026-11-13

### Added
- `mext gguf inspect|warm|prewarm` — GGUF v1–v3 parser and layer page warmer
  (metadata, tensor table, GGML sizes, expert detection) without loading weights.
- HTTP surface for external runners: `GET /api/gguf/inspect`, `POST /api/gguf/warm`.
- `scripts/llama-with-mext.sh` — starts llama.cpp with a background, paced
  layer warmer; `--dry-run` supported.
- p95 warm/RAG latency in `/api/snapshot` (`latency` block) and in the live
  dashboard SSD card.
- `mext profiles merge --auto` — per-source weight derived from Jaccard
  similarity of seen files.

### Changed
- GGUF tensor offsets now follow the spec (relative to the data section).
- `mext gguf` delegates to the Python toolkit; no daemon required.

## [0.5.0] — 2026-11-13

### Added
- Prometheus latency histograms (`mext_pool_warm_latency_ms`,
  `mext_rag_query_latency_ms`) and a p95 Grafana panel.
- `mext profiles list|show|merge` for per-workspace predictor profiles.
- GitHub Actions CI on `macos-14`, `make dev`/`review`/`ci`,
  `scripts/install-service.sh` (launchd user service).
- `mext profiles merge` and histogram tests.

### Fixed
- Manual RAM budget is sticky across the pressure rebalancer.
- Predictor state is persisted on short sessions (`persist_now`).

## [0.4.0] — 2026-11-13

### Added
- Real MoE checkpoint support: `SwitchGLU` → `PagedSwitchGLU`, per-expert
  tensors on SSD, fp16 and 4/8-bit experts, router-guided lookahead.
- Grafana dashboard (17 panels) and `core-daemon/tests/grafana.rs`.

## [0.3.0] — 2026-11-13

### Added
- Shared page arena for small compressed frames (refcounted `madvise`).
- SSE `/api/stream` with polling fallback, Prometheus `/metrics`.
- `eslogger`/`fs_usage` `open(2)` tracing backends, per-workspace predictor
  profiles, router-guided synthetic MoE demo (`--moe`).

### Fixed
- MLX arrays are only used on their owning thread (`no Stream(gpu, 0)`).
- `vm.swapusage` parsing on macOS releases without `sysctlbyname`.

## [0.2.0] — 2026-11-13

### Added
- 4-bit / 8-bit quantised weight streaming (`StreamedQuantizedLinear`,
  `mx.quantized_matmul`), numerically identical to resident models.
- End-to-end harness (`scripts/e2e.sh`, `make e2e` / `e2e-ai`).

## [0.1.0] — 2026-11-13

### Added
- Initial release: LZ4 compressed RAM pool with byte-capacity ARC and
  `madvise(MADV_DONTNEED)`, Markov/EMA/hour-of-day file predictor, MLX layer
  streamer, LanceDB zero-RAM RAG with streaming query prefetch, CLI, Unix-socket
  JSON-RPC, live dashboard.
